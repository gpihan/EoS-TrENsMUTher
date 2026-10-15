"""Low-level immutable-array helpers used by file-backed worker state.

Ordinary validated inputs are still copied onto an immutable ``bytes``
backing.  Arrays loaded with ``numpy.load(..., mmap_mode="r")`` are already
immutable at the operating-system mapping level, however, and copying those
arrays would defeat cross-process sharing.  This module centralizes the small
exception to the usual copy-on-ingest rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


def mmap_backing_path(value: Any) -> Optional[Path]:
    """Return the backing file for a read-only NumPy memmap view, if any."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.flags.writeable:
        return None
    current: Any = value
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, np.memmap):
            mode = getattr(current, "mode", None)
            filename = getattr(current, "filename", None)
            if mode == "r" and filename is not None:
                return Path(filename).expanduser().resolve()
            return None
        current = getattr(current, "base", None)
    return None


def is_read_only_mmap_backed(value: Any) -> bool:
    """Whether *value* ultimately references a read-only ``numpy.memmap``."""

    return mmap_backing_path(value) is not None


def immutable_array(value: Any, dtype: Optional[Any] = None) -> np.ndarray:
    """Return immutable C-order storage without copying read-only mmap views.

    Caller-owned arrays continue to receive the stronger bytes-backed copy
    used throughout the project.  Only a C-contiguous array whose base chain
    reaches a read-only NumPy memmap is retained in place.
    """

    array = np.asarray(value, dtype=dtype)
    if array.flags.c_contiguous and is_read_only_mmap_backed(array):
        return array
    original_shape = array.shape
    contiguous = np.ascontiguousarray(array)
    immutable = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=contiguous.dtype
    )
    # ``np.ascontiguousarray`` promotes a zero-dimensional array to shape
    # ``(1,)``.  Preserve the caller's logical query shape so scalar public
    # results remain zero-dimensional on both NumPy 1.x and 2.x.
    return immutable.reshape(original_shape)


__all__ = [
    "immutable_array",
    "is_read_only_mmap_backed",
    "mmap_backing_path",
]
