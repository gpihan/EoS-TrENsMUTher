#!/usr/bin/env python3
"""Source-tree entry point for the metadata-driven EoS-TrENsMUTher pipeline.

``python3 EoS-TrENsMUTher.py parameters.py`` performs the complete local workflow.
Explicit subcommands are also available; run with ``--help`` for details.
"""

from __future__ import annotations

import sys
from pathlib import Path


_SOURCE_DIRECTORY = Path(__file__).resolve().parent / "src"
if str(_SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIRECTORY))

from eos_trensmuther.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
