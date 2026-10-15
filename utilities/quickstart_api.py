#!/usr/bin/env python3
"""Small public-API inversion using the bundled non-physical table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    PipelineConfig,
    ProjectionConfig,
    load_pipeline_result,
    save_pipeline_result,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "examples/regular/metadata.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--projection",
        action="store_true",
        help="project certified-unreachable targets onto reachable states",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    table = ForwardTable.load_validated(arguments.metadata)
    config = PipelineConfig(
        grid=InversionGridSpec(
            "target",
            {"e": GridAxisSpec(3), "nB": GridAxisSpec(3)},
        ),
        projection=ProjectionConfig(
            enabled=arguments.projection,
            nearest_cell_starts=2,
            max_evaluations_per_start=60,
        ),
        execution=ExecutionConfig(
            workers=1,
            chunk_size=3,
            worker_task_size=2,
        ),
        auxiliary_fields=("P", "s"),
    )

    result = EoSInversionPipeline(table, config).run()
    arguments.output.mkdir(parents=True, exist_ok=True)
    artifact = save_pipeline_result(
        result,
        arguments.output / "inverse.eosinv",
        overwrite=True,
    )
    restored = load_pipeline_result(artifact)
    np.testing.assert_array_equal(
        result.disposition_codes,
        restored.disposition_codes,
    )

    summary = {
        "artifact": str(artifact),
        "artifact_round_trip": True,
        "certified_unreachable": int(
            np.count_nonzero(result.certified_unreachable_mask)
        ),
        "errors": int(np.count_nonzero(result.error_mask)),
        "projected": int(np.count_nonzero(result.projected_mask)),
        "projection_enabled": arguments.projection,
        "solved": int(np.count_nonzero(result.solution_mask)),
        "targets": int(result.grid.size),
        "unresolved": int(np.count_nonzero(result.unknown_mask)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
