#!/usr/bin/env python3
"""Quickly print target or tilde envelopes for one metadata-described EoS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from eos_trensmuther import EoSError, ForwardTable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an EoS table and print component-wise node envelopes. "
            "These rectangles are sampling envelopes, not reachability proofs."
        )
    )
    parser.add_argument("metadata", type=Path, help="path to the table metadata JSON")
    parser.add_argument(
        "--space",
        choices=("target", "tilde", "both"),
        default="both",
        help="coordinate system to report (default: both)",
    )
    parser.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="report zero-span axes instead of rejecting them",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    eos = ForwardTable(args.metadata)
    spaces = ("target", "tilde") if args.space == "both" else (args.space,)
    try:
        envelopes = {
            space: eos.find_boundaries(
                space=space,
                require_non_degenerate=not args.allow_degenerate,
            ).to_dict()
            for space in spaces
        }
    except (EoSError, ValueError) as exc:
        print("Boundary check failed: %s" % exc)
        return 1
    print(json.dumps(envelopes, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
