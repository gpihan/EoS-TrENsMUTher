"""Read-only file-backed state for persistent spawned inversion workers.

The store contains only deterministic immutable ordinary-table state.  Solver
scratch, target chunks, local continuation caches, and result arrays remain
worker-private.  First-order tables deliberately stay on their validated
phase-aware construction path.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ._array_storage import is_read_only_mmap_backed, mmap_backing_path
from .interpolation import MultilinearInterpolator
from .reachability import (
    CellTargetEnclosures,
    ReachabilityConfig,
    ReachabilityIndex,
)
from .reachability_hierarchy import (
    RoutingHierarchy,
    RoutingHierarchyLevel,
)
from .tables import ForwardTable


SHARED_WORKER_STATE_SCHEMA_VERSION = 1
_MANIFEST_NAME = "shared_state.json"


class SharedStateError(RuntimeError):
    """Raised when a file-backed worker snapshot is incomplete or mismatched."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _manifest_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _reachability_dict(config: ReachabilityConfig) -> Dict[str, Any]:
    return {
        item.name: getattr(config, item.name) for item in dataclass_fields(config)
    }


def _array_descriptor(path: Path, array: np.ndarray, root: Path) -> Dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "dtype": array.dtype.str,
        "shape": [int(size) for size in array.shape],
        "nbytes": int(array.nbytes),
    }


def _save_array(
    root: Path, relative_stem: str, value: Any
) -> Dict[str, Any]:
    path = root / (relative_stem + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    np.save(path, array, allow_pickle=False)
    return _array_descriptor(path, array, root)


def _resolve_array_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SharedStateError("shared-state array path is invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise SharedStateError("shared-state array path escapes its store") from exc
    return path


def _load_array(root: Path, descriptor: Mapping[str, Any]) -> np.ndarray:
    if not isinstance(descriptor, Mapping):
        raise SharedStateError("shared-state array descriptor is invalid")
    path = _resolve_array_path(root, descriptor.get("path"))
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SharedStateError(
            "could not attach shared-state array %s: %s" % (path, exc)
        ) from exc
    expected_shape = tuple(int(size) for size in descriptor.get("shape", ()))
    try:
        expected_dtype = np.dtype(descriptor.get("dtype"))
    except (TypeError, ValueError) as exc:
        raise SharedStateError("shared-state array dtype is invalid") from exc
    if (
        array.shape != expected_shape
        or array.dtype != expected_dtype
        or int(array.nbytes) != int(descriptor.get("nbytes", -1))
        or array.flags.writeable
        or not array.flags.c_contiguous
        or not is_read_only_mmap_backed(array)
    ):
        raise SharedStateError(
            "shared-state array %s does not match its descriptor" % path
        )
    return array


def read_shared_state_manifest(path: Any) -> Dict[str, Any]:
    """Load and integrity-check a shared-state manifest."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise SharedStateError(
            "could not read shared-state manifest %s: %s" % (manifest_path, exc)
        ) from exc
    if not isinstance(document, dict):
        raise SharedStateError("shared-state manifest must be an object")
    integrity = document.pop("integrity", None)
    if (
        not isinstance(integrity, dict)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("schema") != "shared_worker_state_manifest_v1"
        or integrity.get("digest") != _manifest_digest(document)
    ):
        raise SharedStateError("shared-state manifest integrity check failed")
    if document.get("schema_version") != SHARED_WORKER_STATE_SCHEMA_VERSION:
        raise SharedStateError("unsupported shared-state schema version")
    document["integrity"] = integrity
    document["_manifest_path"] = str(manifest_path)
    return document


def _manifest_root(document: Mapping[str, Any]) -> Path:
    try:
        return Path(str(document["_manifest_path"])).parent.resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise SharedStateError("shared-state manifest root is unavailable") from exc


def load_shared_table(path: Any) -> ForwardTable:
    """Attach a validated ordinary :class:`ForwardTable` to mmap arrays."""

    document = read_shared_state_manifest(path)
    root = _manifest_root(document)
    table_info = document.get("table")
    if not isinstance(table_info, Mapping):
        raise SharedStateError("shared-state table descriptor is invalid")
    table = ForwardTable(
        table_info.get("metadata_path"), data_path=table_info.get("data_path")
    )
    table.load_metadata()
    axes = tuple(
        _load_array(root, descriptor)
        for descriptor in table_info.get("axes", ())
    )
    field_descriptors = table_info.get("fields")
    if not isinstance(field_descriptors, Mapping):
        raise SharedStateError("shared-state field descriptors are invalid")
    fields = {
        str(name): _load_array(root, descriptor)
        for name, descriptor in field_descriptors.items()
    }
    row_permutation = _load_array(root, table_info.get("row_permutation", {}))
    table._attach_validated_mmap_snapshot(
        axes=axes,
        fields=fields,
        row_permutation=row_permutation,
        uniform_by_axis=table_info.get("uniform_by_axis", ()),
        was_reordered=bool(table_info.get("was_reordered", False)),
        duplicate_count=int(table_info.get("duplicate_count", 0)),
        missing_count=int(table_info.get("missing_count", 0)),
        snapshot_fingerprint=str(table_info.get("snapshot_fingerprint", "")),
    )
    if table.snapshot_fingerprint() != table_info.get("snapshot_fingerprint"):
        raise SharedStateError("attached table fingerprint is inconsistent")
    return table


def _load_reachability_components(
    document: Mapping[str, Any],
    interpolator: MultilinearInterpolator,
) -> Tuple[CellTargetEnclosures, Optional[RoutingHierarchy]]:
    root = _manifest_root(document)
    info = document.get("reachability")
    if not isinstance(info, Mapping):
        raise SharedStateError("shared-state reachability descriptor is invalid")
    arrays = info.get("enclosures")
    if not isinstance(arrays, Mapping):
        raise SharedStateError("shared-state enclosure descriptors are invalid")
    enclosures = CellTargetEnclosures(
        field_names=tuple(info.get("field_names", ())),
        cell_shape=tuple(int(size) for size in info.get("cell_shape", ())),
        cell_flat_indices=_load_array(root, arrays.get("cell_flat_indices", {})),
        exact_lower=_load_array(root, arrays.get("exact_lower", {})),
        exact_upper=_load_array(root, arrays.get("exact_upper", {})),
        routing_lower=_load_array(root, arrays.get("routing_lower", {})),
        routing_upper=_load_array(root, arrays.get("routing_upper", {})),
        residual_scales=np.asarray(info.get("residual_scales", ()), dtype=float),
        routing_padding=np.asarray(info.get("routing_padding", ()), dtype=float),
        root_residual_tolerance=float(info.get("root_residual_tolerance")),
        rigorous=bool(info.get("rigorous")),
        outward_rounded=bool(info.get("outward_rounded")),
    )
    hierarchy_info = info.get("hierarchy")
    if hierarchy_info is None:
        return enclosures, None
    if not isinstance(hierarchy_info, Mapping):
        raise SharedStateError("shared-state hierarchy descriptor is invalid")
    levels = []
    for level_info in hierarchy_info.get("levels", ()):
        if not isinstance(level_info, Mapping):
            raise SharedStateError("shared-state hierarchy level is invalid")
        aliases_leaf = bool(level_info.get("aliases_leaf", False))
        if aliases_leaf:
            lower = enclosures.routing_lower
            upper = enclosures.routing_upper
        else:
            lower = _load_array(root, level_info.get("routing_lower", {}))
            upper = _load_array(root, level_info.get("routing_upper", {}))
        levels.append(
            RoutingHierarchyLevel(
                logical_level=int(level_info.get("logical_level")),
                cell_shape=tuple(int(v) for v in level_info.get("cell_shape", ())),
                block_shape=tuple(int(v) for v in level_info.get("block_shape", ())),
                axis_boundaries=tuple(
                    np.asarray(boundary, dtype=np.int64)
                    for boundary in level_info.get("axis_boundaries", ())
                ),
                routing_lower=lower,
                routing_upper=upper,
                aliases_leaf=aliases_leaf,
            )
        )
    hierarchy = RoutingHierarchy(
        cell_shape=tuple(int(v) for v in hierarchy_info.get("cell_shape", ())),
        requested_levels=int(hierarchy_info.get("requested_levels")),
        levels=tuple(levels),
        max_hierarchy_bytes=int(hierarchy_info.get("max_hierarchy_bytes")),
    )
    return enclosures, hierarchy


def load_shared_reachability(
    path: Any,
    interpolator: MultilinearInterpolator,
    plan: Any,
    config: ReachabilityConfig,
) -> ReachabilityIndex:
    """Attach an index to mmap enclosures and hierarchy bounds."""

    document = read_shared_state_manifest(path)
    if document.get("table_fingerprint") != interpolator.source_snapshot_fingerprint:
        raise SharedStateError("shared state belongs to a different table snapshot")
    if document.get("reachability_configuration") != _reachability_dict(config):
        raise SharedStateError("shared-state reachability configuration mismatch")
    enclosures, hierarchy = _load_reachability_components(document, interpolator)
    return ReachabilityIndex.from_precomputed(
        interpolator,
        plan,
        enclosures,
        hierarchy=hierarchy,
        config=config,
    )


def shared_state_backing_report(
    table: ForwardTable, index: ReachabilityIndex
) -> Dict[str, Any]:
    """Describe and verify the important arrays retained by one worker."""

    rows = []

    def add(name: str, array: np.ndarray) -> None:
        path = mmap_backing_path(array)
        rows.append(
            {
                "name": name,
                "shape": [int(size) for size in array.shape],
                "dtype": array.dtype.str,
                "nbytes": int(array.nbytes),
                "mmap_backed": path is not None,
                "path": None if path is None else str(path),
            }
        )

    for name, values in table.fields.items():
        add("table.fields.%s" % name, values)
    add("table.grid.row_permutation", table.grid.row_permutation)
    for name in (
        "cell_flat_indices",
        "exact_lower",
        "exact_upper",
        "routing_lower",
        "routing_upper",
    ):
        add("reachability.enclosures.%s" % name, getattr(index.enclosures, name))
    if index.hierarchy is not None:
        for level in index.hierarchy.levels:
            add(
                "reachability.hierarchy.level%d.routing_lower"
                % level.logical_level,
                level.routing_lower,
            )
            add(
                "reachability.hierarchy.level%d.routing_upper"
                % level.logical_level,
                level.routing_upper,
            )
    important = [row for row in rows if row["nbytes"] >= 1024 * 1024]
    unique_large_files = {
        row["path"]: int(row["nbytes"])
        for row in important
        if row["path"] is not None
    }
    return {
        "backend": "numpy_load_mmap_mode_r",
        "all_large_arrays_mmap_backed": all(
            row["mmap_backed"] for row in important
        ),
        "large_array_count": len(important),
        "large_array_reference_bytes": int(
            sum(row["nbytes"] for row in important)
        ),
        "large_array_bytes": int(sum(unique_large_files.values())),
        "unique_large_backing_files": len(unique_large_files),
        "arrays": rows,
    }


def _validate_existing_store(
    manifest_path: Path,
    table: ForwardTable,
    config: ReachabilityConfig,
) -> None:
    document = read_shared_state_manifest(manifest_path)
    if document.get("table_fingerprint") != table.snapshot_fingerprint():
        raise SharedStateError("existing shared state belongs to another table")
    if document.get("reachability_configuration") != _reachability_dict(config):
        raise SharedStateError(
            "existing shared state has a different reachability configuration"
        )


def create_shared_worker_state(
    table: ForwardTable,
    plan: Any,
    config: ReachabilityConfig,
    directory: Any,
) -> Path:
    """Create or validate a versioned ordinary-table worker-state store."""

    if not isinstance(table, ForwardTable):
        raise TypeError("table must be a ForwardTable")
    table.require_validated()
    if table.first_order_transition is not None:
        raise SharedStateError(
            "first-order tables are intentionally excluded from mmap worker state"
        )
    if not isinstance(config, ReachabilityConfig):
        raise TypeError("config must be a ReachabilityConfig")
    target = Path(directory).expanduser().resolve()
    manifest_path = target / _MANIFEST_NAME
    if manifest_path.is_file():
        _validate_existing_store(manifest_path, table, config)
        return manifest_path
    if target.exists():
        raise SharedStateError(
            "shared-state directory exists without a complete manifest: %s" % target
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=target.name + ".tmp-", dir=str(target.parent))
    )
    try:
        interpolator = MultilinearInterpolator(table, plan.target_names)
        index = ReachabilityIndex.from_plan(interpolator, plan, config=config)
        table_info: Dict[str, Any] = {
            "metadata_path": str(table.metadata_path),
            "data_path": str(table.data_path),
            "snapshot_fingerprint": table.snapshot_fingerprint(),
            "uniform_by_axis": list(table.grid.uniform_by_axis),
            "was_reordered": bool(table.grid.was_reordered),
            "duplicate_count": int(table.grid.duplicate_count),
            "missing_count": int(table.grid.missing_count),
            "axes": [
                _save_array(temporary, "table/axis_%d" % axis, values)
                for axis, values in enumerate(table.axes)
            ],
            "fields": {
                name: _save_array(temporary, "table/field_%s" % name, values)
                for name, values in table.fields.items()
            },
            "row_permutation": _save_array(
                temporary, "table/row_permutation", table.grid.row_permutation
            ),
        }
        enclosure_arrays = {
            name: _save_array(
                temporary,
                "reachability/%s" % name,
                getattr(index.enclosures, name),
            )
            for name in (
                "cell_flat_indices",
                "exact_lower",
                "exact_upper",
                "routing_lower",
                "routing_upper",
            )
        }
        hierarchy_info = None
        if index.hierarchy is not None:
            level_rows = []
            for level in index.hierarchy.levels:
                row: Dict[str, Any] = {
                    "logical_level": level.logical_level,
                    "cell_shape": list(level.cell_shape),
                    "block_shape": list(level.block_shape),
                    "axis_boundaries": [
                        boundary.tolist() for boundary in level.axis_boundaries
                    ],
                    "aliases_leaf": level.aliases_leaf,
                }
                if not level.aliases_leaf:
                    row["routing_lower"] = _save_array(
                        temporary,
                        "hierarchy/level_%d_lower" % level.logical_level,
                        level.routing_lower,
                    )
                    row["routing_upper"] = _save_array(
                        temporary,
                        "hierarchy/level_%d_upper" % level.logical_level,
                        level.routing_upper,
                    )
                level_rows.append(row)
            hierarchy_info = {
                "cell_shape": list(index.hierarchy.cell_shape),
                "requested_levels": index.hierarchy.requested_levels,
                "max_hierarchy_bytes": index.hierarchy.max_hierarchy_bytes,
                "levels": level_rows,
            }
        document: Dict[str, Any] = {
            "schema_version": SHARED_WORKER_STATE_SCHEMA_VERSION,
            "backend": "numpy_npy_mmap_mode_r",
            "table_fingerprint": table.snapshot_fingerprint(),
            "reachability_configuration": _reachability_dict(config),
            "table": table_info,
            "reachability": {
                "field_names": list(index.enclosures.field_names),
                "cell_shape": list(index.enclosures.cell_shape),
                "residual_scales": index.enclosures.residual_scales.tolist(),
                "routing_padding": index.enclosures.routing_padding.tolist(),
                "root_residual_tolerance": (
                    index.enclosures.root_residual_tolerance
                ),
                "rigorous": index.enclosures.rigorous,
                "outward_rounded": index.enclosures.outward_rounded,
                "enclosures": enclosure_arrays,
                "hierarchy": hierarchy_info,
            },
        }
        document["integrity"] = {
            "algorithm": "sha256",
            "schema": "shared_worker_state_manifest_v1",
            "digest": _manifest_digest(document),
        }
        temporary_manifest = temporary / _MANIFEST_NAME
        temporary_manifest.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _validate_existing_store(manifest_path, table, config)
    return manifest_path


def attach_table_to_shared_state(table: ForwardTable, path: Any) -> None:
    """Drop private raw/canonical arrays and attach an existing mmap snapshot."""

    mapped = load_shared_table(path)
    if table.snapshot_fingerprint() != mapped.snapshot_fingerprint():
        raise SharedStateError("cannot attach a different table snapshot")
    table._attach_validated_mmap_snapshot(
        axes=mapped.axes,
        fields=mapped.fields,
        row_permutation=mapped.grid.row_permutation,
        uniform_by_axis=mapped.grid.uniform_by_axis,
        was_reordered=mapped.grid.was_reordered,
        duplicate_count=mapped.grid.duplicate_count,
        missing_count=mapped.grid.missing_count,
        snapshot_fingerprint=mapped.snapshot_fingerprint(),
    )


__all__ = [
    "SHARED_WORKER_STATE_SCHEMA_VERSION",
    "SharedStateError",
    "attach_table_to_shared_state",
    "create_shared_worker_state",
    "load_shared_reachability",
    "load_shared_table",
    "read_shared_state_manifest",
    "shared_state_backing_report",
]
