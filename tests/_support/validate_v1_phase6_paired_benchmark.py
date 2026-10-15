#!/usr/bin/env python3
"""Replay the frozen paired-400 benchmark through production OLD and NEW.

Common table attachment, complete reachability localization, and
candidate-center seed scoring are untimed.  OLD and NEW each call the current
production ``solve_with_candidate_cells(..., strategy="fallback")`` entry
point exactly once per target, with subcell fallback disabled and enabled
respectively.  Target order alternates and every completed pair is
checkpointed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from eos_trensmuther import (
    EoSRootSolver,
    MultilinearInterpolator,
    ProofStatus,
    ReachabilityConfig,
    SolveStatus,
    load_pipeline_result,
)
from eos_trensmuther.shared_state import load_shared_reachability, load_shared_table

try:
    from tests._support.validate_v1_phase5_root_preservation import _candidate_center_seed
except ModuleNotFoundError:  # direct helper execution
    from validate_v1_phase5_root_preservation import _candidate_center_seed


SCHEMA_VERSION = 1
TARGET_COUNT = 400
MAX_DEPTH = 7
MAX_ACTIVE_BOXES = 1_000_000
MATERIAL_SPEEDUP_MARGIN = 0.05
FROZEN_SELECTION_SHA256 = (
    "bc848164426de8be22d0e7c4900569a3e36002ac4a562b83c0a774a7c91f2e2e"
)
FROZEN_PROTOTYPE_REPORT_SHA256 = (
    "495946a9a4a38c69e9575a355388c8f6e1aca7b638cbc3ee1d4b4f6cfba58117"
)
OUTLIER_FLAT_INDICES = (412906, 465165)

DEFAULT_ARTIFACT = Path(
    "Output_TEST/HRG4D30Mmap8/HRG4D30Mmap8/inverse.eosinv"
)
DEFAULT_SHARED_STATE = Path(
    "Output_TEST/HRG4D30Mmap8/HRG4D30Mmap8/diagnostic/"
    "shared_worker_state_v1/shared_state.json"
)
DEFAULT_FROZEN_ROOT = Path(
    "Output_TEST/HRG4DSubcellReachability/paired_fallback_400"
)
DEFAULT_OUTPUT = Path("Output_TEST/V1Phase6PairedBenchmark")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "minimum": float(np.min(array)),
        "p25": float(np.percentile(array, 25.0)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75.0)),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
        "sum": float(np.sum(array)),
    }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the common BSD interfaces report KiB.
    return value if sys.platform == "darwin" else value * 1024


def load_frozen_selection(path: Path) -> Dict[str, np.ndarray]:
    digest = _sha256(path)
    if digest != FROZEN_SELECTION_SHA256:
        raise RuntimeError("frozen paired-400 selection SHA-256 is inconsistent")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "rng_seed",
            "flat_indices",
            "multiplicity_bin",
            "canonical_targets",
            "n0",
            "production_fast_status_codes",
            "production_final_status_codes",
            "candidate_offsets",
            "candidate_cell_flat_indices",
        }
        if set(archive.files) != required:
            raise RuntimeError("frozen paired-400 selection fields are inconsistent")
        selection = {name: np.asarray(archive[name]) for name in archive.files}
    if (
        int(selection["rng_seed"]) != 20261006
        or selection["flat_indices"].shape != (TARGET_COUNT,)
        or selection["canonical_targets"].shape != (TARGET_COUNT, 4)
        or selection["n0"].shape != (TARGET_COUNT,)
        or selection["candidate_offsets"].shape != (TARGET_COUNT + 1,)
        or int(selection["candidate_offsets"][-1])
        != selection["candidate_cell_flat_indices"].size
        or not np.array_equal(
            np.diff(selection["candidate_offsets"]), selection["n0"]
        )
    ):
        raise RuntimeError("frozen paired-400 selection shape/count contract failed")
    return selection


def _subcell_report(report: Any) -> Any:
    if report is None:
        return None
    return {
        "complete": bool(report.complete),
        "completeness": report.completeness.value,
        "stop_reason": report.stop_reason.value,
        "requested_depth": int(report.requested_depth),
        "completed_depth": int(report.completed_depth),
        "initial_cell_count": int(report.initial_cell_count),
        "total_boxes_evaluated": int(report.total_boxes_evaluated),
        "first_exclusion_depth": report.first_exclusion_depth,
        "levels": [
            {
                "depth": int(level.depth),
                "complete": bool(level.complete),
                "boxes_entering": int(level.boxes_entering),
                "boxes_generated": int(level.boxes_generated),
                "boxes_evaluated": int(level.boxes_evaluated),
                "boxes_surviving": int(level.boxes_surviving),
                "undefined_boxes_retained": int(
                    level.undefined_boxes_retained
                ),
                "step_volume_fraction": level.step_volume_fraction,
                "original_volume_fraction": level.original_volume_fraction,
                "elapsed_seconds": float(level.elapsed_seconds),
            }
            for level in report.levels
        ],
    }


def _run_route(
    solver: EoSRootSolver,
    target: np.ndarray,
    localization: Any,
    row: int,
    seed: np.ndarray,
    provenance: Any,
) -> Dict[str, Any]:
    peak_before = _peak_rss_bytes()
    started = time.perf_counter()
    result = solver.solve_with_candidate_cells(
        target,
        localization,
        row,
        seed,
        expected_provenance=provenance,
        strategy="fallback",
    )
    elapsed = float(time.perf_counter() - started)
    peak_after = _peak_rss_bytes()
    fallback = result.fallback
    if fallback is None:
        raise RuntimeError("paired production FALLBACK returned no report")
    return {
        "elapsed_seconds": elapsed,
        "peak_rss_before_bytes": peak_before,
        "peak_rss_after_bytes": peak_after,
        "peak_rss_increase_bytes": max(0, peak_after - peak_before),
        "production_solver_calls": 1,
        "status": result.status.value,
        "proof_status": result.proof_status.value,
        "path": list(result.path),
        "fallback_status": fallback.status.value,
        "scan_complete": bool(fallback.scan_complete),
        "localization_complete": bool(fallback.localization_complete),
        "candidate_source": fallback.candidate_source,
        "total_cells": int(fallback.total_cells),
        "cells_scanned": int(fallback.cells_scanned),
        "candidate_cells": int(fallback.candidate_cells),
        "retained_candidate_cells": int(fallback.retained_candidate_cells),
        "cells_excluded_by_range": int(fallback.cells_excluded_by_range),
        "boxes_processed": int(fallback.boxes_processed),
        "boxes_excluded": int(fallback.boxes_excluded),
        "boxes_contracted": int(fallback.boxes_contracted),
        "boxes_subdivided": int(fallback.boxes_subdivided),
        "unresolved_boxes": int(fallback.unresolved_boxes),
        "maximum_depth_reached": int(fallback.maximum_depth_reached),
        "unprocessed_candidate_cells": int(
            fallback.unprocessed_candidate_cells
        ),
        "subcell_residual_box_count": int(fallback.subcell_residual_box_count),
        "subcell": _subcell_report(fallback.subcell_report),
    }


def _proof_safe_certified_no_root(route: Mapping[str, Any]) -> bool:
    subcell = route.get("subcell")
    return bool(
        route.get("status") == SolveStatus.CERTIFIED_NO_ROOT.value
        and route.get("proof_status") == ProofStatus.NO_ROOT_IN_DOMAIN.value
        and route.get("scan_complete") is True
        and route.get("localization_complete") is True
        and int(route.get("unresolved_boxes", -1)) == 0
        and int(route.get("unprocessed_candidate_cells", -1)) == 0
        and (subcell is None or subcell.get("complete") is True)
    )


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = TARGET_COUNT,
    outlier_flat_indices: Sequence[int] = OUTLIER_FLAT_INDICES,
    historical_rows: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    old_times = np.asarray(
        [float(row["old"]["elapsed_seconds"]) for row in rows], dtype=float
    )
    new_times = np.asarray(
        [float(row["new"]["elapsed_seconds"]) for row in rows], dtype=float
    )
    paired_agreement = sum(
        row["old"]["status"] == row["new"]["status"]
        and row["old"]["proof_status"] == row["new"]["proof_status"]
        for row in rows
    )
    old_safe = sum(_proof_safe_certified_no_root(row["old"]) for row in rows)
    new_safe = sum(_proof_safe_certified_no_root(row["new"]) for row in rows)
    total_old = float(np.sum(old_times)) if rows else 0.0
    total_new = float(np.sum(new_times)) if rows else 0.0
    outliers = []
    for flat_index in outlier_flat_indices:
        matching = [row for row in rows if int(row["flat_index"]) == flat_index]
        if not matching:
            continue
        row = matching[0]
        new = row["new"]
        historical = None if historical_rows is None else historical_rows.get(flat_index)
        outliers.append(
            {
                "flat_index": flat_index,
                "residual_boxes": int(new["subcell_residual_box_count"]),
                "batched_krawczyk_boxes_processed": int(new["boxes_processed"]),
                "recursive_subdivisions": int(new["boxes_subdivided"]),
                "production_solver_calls": int(new["production_solver_calls"]),
                "new_seconds": float(new["elapsed_seconds"]),
                "historical_per_box_seconds": (
                    None
                    if historical is None
                    else float(historical["new"]["elapsed_seconds"])
                ),
                "overhead_removed": bool(
                    int(new["subcell_residual_box_count"]) == 4433
                    and int(new["boxes_processed"]) == 4433
                    and int(new["boxes_subdivided"]) == 0
                    and int(new["production_solver_calls"]) == 1
                ),
            }
        )
    complete = len(rows) == expected_count
    materially_faster = bool(
        complete and total_new < (1.0 - MATERIAL_SPEEDUP_MARGIN) * total_old
    )
    overhead_removed = bool(
        len(outliers) == len(tuple(outlier_flat_indices))
        and all(item["overhead_removed"] for item in outliers)
    )
    acceptance = {
        "all_expected_rows_complete": complete,
        "proof_status_agreement_400_of_400": (
            complete and paired_agreement == expected_count
        ),
        "old_proof_safe_certified_no_root_400_of_400": (
            complete and old_safe == expected_count
        ),
        "new_proof_safe_certified_no_root_400_of_400": (
            complete and new_safe == expected_count
        ),
        "four_thousand_four_hundred_thirty_three_box_overhead_removed": (
            overhead_removed
        ),
        "new_total_time_materially_below_old": materially_faster,
    }
    acceptance["gate_satisfied"] = bool(all(acceptance.values()))
    return {
        "target_count": len(rows),
        "paired_status_and_proof_agreement": int(paired_agreement),
        "old_proof_safe_certified_no_root": int(old_safe),
        "new_proof_safe_certified_no_root": int(new_safe),
        "old_status_counts": dict(Counter(row["old"]["status"] for row in rows)),
        "new_status_counts": dict(Counter(row["new"]["status"] for row in rows)),
        "old_seconds": _distribution(old_times) if rows else None,
        "new_seconds": _distribution(new_times) if rows else None,
        "overall_speedup": None if not rows or total_new == 0.0 else total_old / total_new,
        "new_materially_faster_targets": int(
            np.count_nonzero(new_times < (1.0 - MATERIAL_SPEEDUP_MARGIN) * old_times)
        ),
        "approximately_equal_targets": int(
            np.count_nonzero(
                (new_times >= (1.0 - MATERIAL_SPEEDUP_MARGIN) * old_times)
                & (new_times <= (1.0 + MATERIAL_SPEEDUP_MARGIN) * old_times)
            )
        ),
        "new_materially_slower_targets": int(
            np.count_nonzero(new_times > (1.0 + MATERIAL_SPEEDUP_MARGIN) * old_times)
        ),
        "new_subcell_certificate_depth_counts": dict(
            Counter(
                str(row["new"]["subcell"]["first_exclusion_depth"])
                for row in rows
                if row["new"].get("subcell") is not None
                and row["new"]["subcell"]["first_exclusion_depth"] is not None
            )
        ),
        "new_residual_krawczyk_target_count": int(
            sum(row["new"]["subcell_residual_box_count"] > 0 for row in rows)
        ),
        "old_krawczyk_evaluations": int(
            sum(row["old"]["boxes_processed"] for row in rows)
        ),
        "new_krawczyk_evaluations": int(
            sum(row["new"]["boxes_processed"] for row in rows)
        ),
        "old_recursive_subdivisions": int(
            sum(row["old"]["boxes_subdivided"] for row in rows)
        ),
        "new_recursive_subdivisions": int(
            sum(row["new"]["boxes_subdivided"] for row in rows)
        ),
        "peak_rss_bytes": int(
            max(
                (
                    max(
                        int(row["old"]["peak_rss_after_bytes"]),
                        int(row["new"]["peak_rss_after_bytes"]),
                    )
                    for row in rows
                ),
                default=0,
            )
        ),
        "outlier_4433": outliers,
        "acceptance": acceptance,
    }


def _report(
    *,
    rows: Sequence[Mapping[str, Any]],
    selection_sha256: str,
    prototype_report_sha256: str,
    source_fingerprint: str,
    old_config: Any,
    new_config: Any,
    localization_seconds: float,
    candidate_pair_count: int,
    historical_rows: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    summary = summarize_rows(rows, historical_rows=historical_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "production-integrated exact paired-400 OLD/NEW FALLBACK replay",
        "complete": len(rows) == TARGET_COUNT,
        "selection": {
            "sha256": selection_sha256,
            "target_count": TARGET_COUNT,
            "candidate_pair_count": candidate_pair_count,
            "csr_matches_frozen_selection_exactly": True,
        },
        "frozen_prototype_report_sha256": prototype_report_sha256,
        "source_fingerprint": source_fingerprint,
        "configuration": {
            "old_solver": asdict(old_config),
            "new_solver": asdict(new_config),
            "reachability": {"method": "hierarchical", "hierarchy_levels": 7},
            "timing_contract": (
                "one production strategy=fallback call per route; table attachment, "
                "complete localization, and candidate-center seed scoring excluded"
            ),
            "execution_order": "alternating OLD-first/NEW-first by sample position",
            "material_speedup_margin": MATERIAL_SPEEDUP_MARGIN,
        },
        "untimed_localization_seconds": localization_seconds,
        "summary": summary,
        "rows": list(rows),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    selection_path = args.frozen_root / "paired_fallback_selection.npz"
    prototype_path = args.frozen_root / "paired_fallback_benchmark.json"
    selection = load_frozen_selection(selection_path)
    if _sha256(prototype_path) != FROZEN_PROTOTYPE_REPORT_SHA256:
        raise RuntimeError("frozen prototype benchmark SHA-256 is inconsistent")
    prototype = json.loads(prototype_path.read_text(encoding="utf-8"))
    historical_rows = {
        int(row["flat_index"]): row for row in prototype["rows"]
    }

    artifact = load_pipeline_result(args.artifact)
    table = load_shared_table(args.shared_state)
    interpolator = MultilinearInterpolator(table, artifact.plan.target_names)
    reachability_config = ReachabilityConfig(
        method="hierarchical", hierarchy_levels=7
    )
    reachability = load_shared_reachability(
        args.shared_state, interpolator, artifact.plan, reachability_config
    )
    targets = np.asarray(selection["canonical_targets"], dtype=float)
    localization_started = time.perf_counter()
    localization = reachability.localize(targets)
    localization_seconds = float(time.perf_counter() - localization_started)
    localization.require_complete_for(reachability.provenance)
    if (
        not np.array_equal(localization.offsets, selection["candidate_offsets"])
        or not np.array_equal(
            localization.cell_flat_indices,
            selection["candidate_cell_flat_indices"],
        )
        or not np.array_equal(localization.candidate_counts, selection["n0"])
    ):
        raise RuntimeError("production localization does not match frozen C0 exactly")

    base_config = artifact.plan.config.solver
    old_config = replace(base_config, enable_subcell_fallback=False)
    new_config = replace(
        base_config,
        enable_subcell_fallback=True,
        max_subcell_depth=MAX_DEPTH,
        max_subcell_active_boxes=MAX_ACTIVE_BOXES,
    )
    scales = np.asarray(artifact.plan.residual_scales, dtype=float)
    old_solver = EoSRootSolver(interpolator, old_config, residual_scales=scales)
    new_solver = EoSRootSolver(interpolator, new_config, residual_scales=scales)

    checkpoint_path = args.output / "phase6_checkpoint.json"
    report_path = args.output / "phase6_report.json"
    rows = []
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("selection", {}).get("sha256")
            != FROZEN_SELECTION_SHA256
            or checkpoint.get("source_fingerprint")
            != interpolator.source_snapshot_fingerprint
            or checkpoint.get("configuration", {}).get("old_solver")
            != _plain(asdict(old_config))
            or checkpoint.get("configuration", {}).get("new_solver")
            != _plain(asdict(new_config))
        ):
            raise RuntimeError("existing Phase-6 checkpoint is incompatible")
        rows = list(checkpoint.get("rows", []))
        if checkpoint.get("complete"):
            _write_json(report_path, checkpoint)
            return checkpoint
    completed = {int(row["sample_position"]) for row in rows}

    frozen_rows = {int(row["sample_position"]): row for row in prototype["rows"]}
    for position in range(TARGET_COUNT):
        if position in completed:
            continue
        target = targets[position]
        candidates = localization.cells_for(position)
        seed, seed_cell, seed_residual = _candidate_center_seed(
            interpolator, target, candidates, scales
        )
        frozen = frozen_rows[position]
        if (
            int(seed_cell) != int(frozen["candidate_center_cell"])
            or not np.isclose(
                seed_residual,
                float(frozen["candidate_center_scaled_residual_norm_inf"]),
                rtol=2.0e-15,
                atol=0.0,
            )
        ):
            raise RuntimeError("production candidate-center seed differs from frozen pair")
        order = "OLD_THEN_NEW" if position % 2 == 0 else "NEW_THEN_OLD"
        try:
            if order == "OLD_THEN_NEW":
                old = _run_route(
                    old_solver,
                    target,
                    localization,
                    position,
                    seed,
                    reachability.provenance,
                )
                new = _run_route(
                    new_solver,
                    target,
                    localization,
                    position,
                    seed,
                    reachability.provenance,
                )
            else:
                new = _run_route(
                    new_solver,
                    target,
                    localization,
                    position,
                    seed,
                    reachability.provenance,
                )
                old = _run_route(
                    old_solver,
                    target,
                    localization,
                    position,
                    seed,
                    reachability.provenance,
                )
        except Exception as exc:
            failure = {
                "sample_position": position,
                "flat_index": int(selection["flat_indices"][position]),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            _write_json(args.output / "phase6_failure.json", failure)
            raise
        row = {
            "sample_position": position,
            "flat_index": int(selection["flat_indices"][position]),
            "multiplicity_bin": str(selection["multiplicity_bin"][position]),
            "n0": int(selection["n0"][position]),
            "target": target.tolist(),
            "candidate_center_cell": int(seed_cell),
            "candidate_center_scaled_residual_norm_inf": float(seed_residual),
            "execution_order": order,
            "old": old,
            "new": new,
        }
        rows.append(row)
        rows.sort(key=lambda item: int(item["sample_position"]))
        report = _report(
            rows=rows,
            selection_sha256=FROZEN_SELECTION_SHA256,
            prototype_report_sha256=FROZEN_PROTOTYPE_REPORT_SHA256,
            source_fingerprint=interpolator.source_snapshot_fingerprint,
            old_config=old_config,
            new_config=new_config,
            localization_seconds=localization_seconds,
            candidate_pair_count=localization.pair_count,
            historical_rows=historical_rows,
        )
        _write_json(checkpoint_path, report)
        print(
            "%d/%d flat=%d N0=%d OLD=%s %.6fs NEW=%s %.6fs"
            % (
                len(rows),
                TARGET_COUNT,
                row["flat_index"],
                row["n0"],
                old["status"],
                old["elapsed_seconds"],
                new["status"],
                new["elapsed_seconds"],
            ),
            flush=True,
        )
        if (
            old["status"] != new["status"]
            or old["proof_status"] != new["proof_status"]
            or not _proof_safe_certified_no_root(old)
            or not _proof_safe_certified_no_root(new)
        ):
            _write_json(args.output / "phase6_discrepancy.json", row)
            raise RuntimeError("Phase-6 proof/status discrepancy at row %d" % position)

    report = _report(
        rows=rows,
        selection_sha256=FROZEN_SELECTION_SHA256,
        prototype_report_sha256=FROZEN_PROTOTYPE_REPORT_SHA256,
        source_fingerprint=interpolator.source_snapshot_fingerprint,
        old_config=old_config,
        new_config=new_config,
        localization_seconds=localization_seconds,
        candidate_pair_count=localization.pair_count,
        historical_rows=historical_rows,
    )
    _write_json(checkpoint_path, report)
    _write_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--shared-state", type=Path, default=DEFAULT_SHARED_STATE)
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(_plain(report["summary"]), indent=2))
    if not report["summary"]["acceptance"]["gate_satisfied"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
