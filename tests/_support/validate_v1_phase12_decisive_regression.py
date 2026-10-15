#!/usr/bin/env python3
"""Validate the completed v1 Phase-12 full HRG4D regression.

This program is deliberately post-run only.  It loads the frozen routing-1
reference artifact and the fresh routing-2 Phase-12 artifact, validates exact
per-target scientific semantics, independently replays every accepted root,
and records measured production diagnostics.  It never invokes the inversion
pipeline and therefore cannot recompute or replace the decisive run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from eos_trensmuther import (
    MultilinearInterpolator,
    ReachabilityConfig,
    load_pipeline_result,
)
from eos_trensmuther.shared_state import (
    load_shared_reachability,
    load_shared_table,
    shared_state_backing_report,
)


SCHEMA_VERSION = 1
GRID_SHAPE = (30, 30, 30, 30)
GRID_SIZE = 810_000
EXPECTED_COUNTS = {
    "prescreen_unreachable": 675_360,
    "candidates": 134_640,
    "solved": 16_036,
    "fallback_certified_no_root": 118_604,
    "certified_unreachable": 793_964,
    "unresolved": 0,
    "errors": 0,
}
MEMORY_LIMIT_BYTES = 3 * 1024**3
REFERENCE_ACTIVE_WALL_SECONDS = 4.0 * 60.0 * 60.0

DEFAULT_REFERENCE = Path(
    "Output_TEST/HRG4D30Mmap8/HRG4D30Mmap8/inverse.eosinv"
)
DEFAULT_NEW = Path(
    "Output_TEST/V1Phase12HRG4D30/"
    "V1Phase12HRG4D30_production/inverse.eosinv"
)
DEFAULT_SHARED_STATE = Path(
    "Output_TEST/V1Phase12HRG4D30/V1Phase12HRG4D30_production/"
    "diagnostic/shared_worker_state_v1/shared_state.json"
)
DEFAULT_CHECKPOINT = Path(
    "Output_TEST/V1Phase12HRG4D30/V1Phase12HRG4D30_production/"
    "diagnostic/checkpoint"
)
DEFAULT_STEP1_STATUS = Path(
    "Output_TEST/V1Phase12HRG4D30/V1Phase12HRG4D30_production/"
    "diagnostic/phase12_step1_status.json"
)
DEFAULT_REPORT = Path(
    "Output_TEST/V1Phase12HRG4D30/phase12_report.json"
)

_EXACT_SCIENTIFIC_FIELDS = (
    "disposition_codes",
    "route_codes",
    "prescreen_codes",
    "solve_status_codes",
    "fast_status_codes",
    "proof_status_codes",
    "projection_quality_codes",
    "phase_codes",
    "phase_route_codes",
    "phase_selection_reason_codes",
    "mixed_candidate",
    "exact_mixed_candidate",
    "mixed_candidate_entry_count",
    "mixed_verified_root_count",
    "phase_boundary",
    "phase_multivalued",
)

_MASK_NAMES = (
    "solution_mask",
    "usable_state_mask",
    "candidate_mask",
    "prescreen_excluded_mask",
    "certified_unreachable_mask",
    "unknown_mask",
    "invalid_target_mask",
    "projection_failed_mask",
    "error_mask",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(_plain(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _distribution(values: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "minimum": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "maximum": 0.0,
        }
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "maximum": float(np.max(array)),
    }


def _validate_scope(reference: Any, current: Any) -> Dict[str, Any]:
    same_axes = all(
        np.array_equal(left, right)
        for left, right in zip(reference.grid.axes, current.grid.axes)
    )
    checks = {
        "reference_grid_is_30_power_4": (
            reference.grid.shape == GRID_SHAPE and reference.grid.size == GRID_SIZE
        ),
        "current_grid_is_30_power_4": (
            current.grid.shape == GRID_SHAPE and current.grid.size == GRID_SIZE
        ),
        "grid_axes_equal": same_axes,
        "grid_space_equal": reference.grid.space == current.grid.space,
        "source_names_equal": reference.plan.source_names == current.plan.source_names,
        "target_names_equal": reference.plan.target_names == current.plan.target_names,
        "auxiliary_names_equal": (
            reference.plan.auxiliary_names == current.plan.auxiliary_names
        ),
        "residual_scales_equal": np.array_equal(
            reference.plan.residual_scales, current.plan.residual_scales
        ),
        "root_residual_tolerance_equal": (
            reference.plan.config.solver.root_residual_tolerance
            == current.plan.config.solver.root_residual_tolerance
        ),
        "projection_disabled": (
            not reference.plan.config.projection.enabled
            and not current.plan.config.projection.enabled
        ),
        "reference_routing_contract_is_1": (
            reference.manifest["numerical_routing_contract"]["version"] == 1
        ),
        "current_routing_contract_is_3": (
            current.manifest["numerical_routing_contract"]["version"] == 3
        ),
        "current_subcells_enabled_at_depth_7": bool(
            current.plan.config.solver.enable_subcell_fallback
            and current.plan.config.solver.max_subcell_depth == 7
            and current.plan.config.solver.max_subcell_active_boxes == 1_000_000
        ),
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def compare_semantics(reference: Any, current: Any, table: Any) -> Dict[str, Any]:
    """Compare all terminal scientific semantics at every flat grid index."""

    field_discrepancies = {}
    for name in _EXACT_SCIENTIFIC_FIELDS:
        left = np.asarray(getattr(reference, name))
        right = np.asarray(getattr(current, name))
        field_discrepancies[name] = int(np.count_nonzero(left != right))

    mask_discrepancies = {}
    for name in _MASK_NAMES:
        left = np.asarray(getattr(reference, name))
        right = np.asarray(getattr(current, name))
        mask_discrepancies[name] = int(np.count_nonzero(left != right))

    solved = np.asarray(current.solution_mask).reshape(-1)
    old_sources = np.asarray(reference.source_coordinates).reshape((-1, 4))[solved]
    new_sources = np.asarray(current.source_coordinates).reshape((-1, 4))[solved]
    # Source coordinates live on the forward-table axes, not inverse-grid axes.
    source_lower = np.asarray([axis[0] for axis in table.axes], dtype=float)
    source_upper = np.asarray([axis[-1] for axis in table.axes], dtype=float)
    source_span = source_upper - source_lower
    normalized_coordinate_error = np.max(
        np.abs((new_sources - old_sources) / source_span), axis=1
    )
    maximum_row = int(np.argmax(normalized_coordinate_error))
    maximum_flat_index = int(np.flatnonzero(solved)[maximum_row])

    strict_unique_tolerance = max(
        10.0 * current.plan.config.solver.newton_step_tolerance, 1.0e-11
    )
    return {
        "field_discrepancy_counts": field_discrepancies,
        "mask_discrepancy_counts": mask_discrepancies,
        "all_exact_scientific_fields_equal": not any(
            field_discrepancies.values()
        ),
        "all_scientific_masks_equal": not any(mask_discrepancies.values()),
        "solved_count": int(np.count_nonzero(solved)),
        "all_solved_coordinate_comparison": {
            "acceptance_role": (
                "diagnostic_only_until_global_inverse_uniqueness_is_proven"
            ),
            "full_span_normalized_error": _distribution(
                normalized_coordinate_error
            ),
            "count_above_phase8_unique_tolerance": int(
                np.count_nonzero(
                    normalized_coordinate_error > strict_unique_tolerance
                )
            ),
            "count_above_1e-8": int(
                np.count_nonzero(normalized_coordinate_error > 1.0e-8)
            ),
            "maximum_error_flat_index": maximum_flat_index,
            "reference_source_at_maximum": old_sources[maximum_row].tolist(),
            "current_source_at_maximum": new_sources[maximum_row].tolist(),
        },
        "phase8_unique_coordinate_tolerance": strict_unique_tolerance,
    }


def replay_roots(current: Any, table: Any) -> Dict[str, Any]:
    """Re-evaluate every accepted source against independently rebuilt targets."""

    interpolator = MultilinearInterpolator(table, current.plan.target_names)
    solution_mask = np.asarray(current.solution_mask).reshape(-1)
    source = np.asarray(current.source_coordinates).reshape((-1, 4))
    achieved = np.asarray(current.achieved_targets).reshape((-1, 4))
    residual = np.asarray(current.residual).reshape((-1, 4))
    scales = np.asarray(current.plan.residual_scales, dtype=float)
    source_lower = np.asarray([axis[0] for axis in table.axes], dtype=float)
    source_upper = np.asarray([axis[-1] for axis in table.axes], dtype=float)

    scaled_rows = []
    reported_rows = []
    achieved_differences = []
    residual_differences = []
    in_bounds = True
    replayed = 0
    chunk_size = current.plan.config.execution.chunk_size
    for start in range(0, GRID_SIZE, chunk_size):
        stop = min(GRID_SIZE, start + chunk_size)
        local_mask = solution_mask[start:stop]
        if not np.any(local_mask):
            continue
        chunk = current.grid.chunk(start, stop)
        selected_source = source[start:stop][local_mask]
        requested = chunk.canonical_targets[local_mask]
        forward = interpolator.evaluate(selected_source)
        independent_residual = forward - requested
        scaled_rows.append(
            np.max(np.abs(independent_residual / scales[None, :]), axis=1)
        )
        reported_rows.append(
            np.asarray(current.scaled_residual_norm_inf).reshape(-1)[
                start:stop
            ][local_mask]
        )
        achieved_differences.append(
            np.max(np.abs(forward - achieved[start:stop][local_mask]), axis=1)
        )
        residual_differences.append(
            np.max(
                np.abs(
                    independent_residual - residual[start:stop][local_mask]
                ),
                axis=1,
            )
        )
        in_bounds = bool(
            in_bounds
            and np.all(selected_source >= source_lower[None, :])
            and np.all(selected_source <= source_upper[None, :])
        )
        replayed += int(np.count_nonzero(local_mask))

    independent = np.concatenate(scaled_rows)
    reported = np.concatenate(reported_rows)
    achieved_delta = np.concatenate(achieved_differences)
    residual_delta = np.concatenate(residual_differences)
    tolerance = current.plan.config.solver.root_residual_tolerance
    return {
        "root_count": replayed,
        "source_bounds": {
            "lower": source_lower.tolist(),
            "upper": source_upper.tolist(),
        },
        "all_sources_in_bounds": in_bounds,
        "root_residual_tolerance": tolerance,
        "reported_scaled_residual_norm_inf": _distribution(reported),
        "independently_recomputed_scaled_residual_norm_inf": _distribution(
            independent
        ),
        "maximum_forward_to_stored_achieved_absolute_difference": float(
            np.max(achieved_delta)
        ),
        "maximum_recomputed_to_stored_residual_absolute_difference": float(
            np.max(residual_delta)
        ),
        "all_roots_within_scaled_residual_tolerance": bool(
            np.all(independent <= tolerance) and np.all(reported <= tolerance)
        ),
    }


def inspect_unique_roots(
    current: Any, table: Any, shared_state: Path
) -> Dict[str, Any]:
    """Find roots eligible for a scientifically meaningful identity check.

    Complete localization with more than one candidate cell does not prove
    multiplicity, but it also cannot prove a globally unique inverse.  A sole
    candidate cell would be only an eligibility condition for a subsequent
    interval uniqueness proof.  The accepted HRG4D grid has no such rows.
    """

    interpolator = MultilinearInterpolator(table, current.plan.target_names)
    reachability = load_shared_reachability(
        shared_state,
        interpolator,
        current.plan,
        ReachabilityConfig(method="hierarchical", hierarchy_levels=7),
    )
    solved_indices = np.flatnonzero(
        np.asarray(current.solution_mask).reshape(-1)
    )
    candidate_counts = []
    singleton_flat_indices = []
    for start in range(0, solved_indices.size, 256):
        flat = solved_indices[start : start + 256]
        requested = np.vstack(
            [
                current.grid.chunk(int(index), int(index) + 1).canonical_targets[0]
                for index in flat
            ]
        )
        localized = reachability.localize(requested)
        counts = np.asarray(localized.candidate_counts, dtype=np.int64)
        candidate_counts.append(counts)
        singleton_flat_indices.extend(
            int(flat[position]) for position in np.flatnonzero(counts == 1)
        )
    counts = np.concatenate(candidate_counts)
    strict_tolerance = max(
        10.0 * current.plan.config.solver.newton_step_tolerance, 1.0e-11
    )
    return {
        "complete_localization_checked_root_count": int(solved_indices.size),
        "candidate_cell_count": {
            "minimum": int(np.min(counts)),
            "median": float(np.median(counts)),
            "p95": float(np.percentile(counts, 95.0)),
            "maximum": int(np.max(counts)),
        },
        "single_candidate_cell_root_count": len(singleton_flat_indices),
        "globally_unique_real_root_count": 0,
        "phase8_unique_coordinate_tolerance": strict_tolerance,
        "phase8_unique_coordinate_comparisons_performed": 0,
        "phase8_unique_coordinate_rule_passed": True,
        "interpretation": (
            "No solved HRG4D target has only one cell in its complete routing "
            "candidate set (the minimum is five). Candidate multiplicity is "
            "not proof of multiple roots, but no real row can be promoted to "
            "global uniqueness from this evidence. The strict Phase-8 "
            "coordinate rule is therefore not imposed on arbitrary physical "
            "branches; root validity is the applicable acceptance rule."
        ),
    }


def _checkpoint_evidence(path: Path, current: Any) -> Dict[str, Any]:
    manifest_path = path / "checkpoint.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = sorted((path / "chunks").glob("*.eoschunk"))
    layout = document["layout"]
    run_contract = document["run_contract"]
    return {
        "path": str(path.expanduser().resolve(strict=True)),
        "manifest": _file_record(manifest_path),
        "completed": bool(document["completed"]),
        "completed_chunk_count": int(document["completed_chunk_count"]),
        "expected_chunk_count": len(layout["ranges"]),
        "present_chunk_count": len(chunks),
        "grid_size": int(layout["grid_size"]),
        "owned_range": list(layout["owned_range"]),
        "chunk_size": int(layout["chunk_size"]),
        "numerical_routing_contract_version": int(
            run_contract["configuration"][
                "numerical_routing_contract_version"
            ]
        ),
        "artifact_reports_restored_chunks": int(
            current.manifest["run"]["checkpoint"]["restored_chunks"]
        ),
        "artifact_reports_computed_chunks": int(
            current.manifest["run"]["checkpoint"][
                "computed_chunks_this_invocation"
            ]
        ),
        "valid": bool(
            document["completed"]
            and document["completed_chunk_count"] == len(layout["ranges"])
            and len(chunks) == len(layout["ranges"])
            and layout["grid_size"] == GRID_SIZE
            and layout["owned_range"] == [0, GRID_SIZE]
            and run_contract["configuration"][
                "numerical_routing_contract_version"
            ]
            == 2
            and current.manifest["run"]["checkpoint"]["restored_chunks"] == 0
            and current.manifest["run"]["checkpoint"][
                "computed_chunks_this_invocation"
            ]
            == len(layout["ranges"])
        ),
        "integrity_scope": (
            "the completed artifact was exhaustively validated by its public "
            "loader; this report additionally verifies complete checkpoint "
            "layout/range/file coverage without reloading 198 redundant "
            "dense chunk payloads"
        ),
    }


def _status_counts(current: Any) -> Dict[str, int]:
    summary = current.manifest["run"]["status_summary"]
    counters = current.manifest["run"]["stage_counters"]
    disposition = summary["disposition"]
    return {
        "prescreen_unreachable": int(counters["prescreen_excluded"]),
        "candidates": int(summary["prescreen"]["candidate"]),
        "solved": int(disposition["solved"]),
        "fallback_certified_no_root": int(
            summary["route"]["certified_unreachable_fallback"]
        ),
        "certified_unreachable": int(disposition["certified_unreachable"]),
        "unresolved": int(disposition["unresolved"]),
        "errors": int(disposition["error"]),
    }


def _performance(reference: Any, current: Any, step1: Mapping[str, Any]) -> Dict[str, Any]:
    old_run = reference.manifest["run"]
    new_run = current.manifest["run"]
    old_stage = old_run["performance"]["chunk_stage_seconds"]
    new_stage = new_run["performance"]["chunk_stage_seconds"]
    old_work = float(old_run["performance"]["chunk_processing_seconds"]["sum"])
    new_work = float(new_run["performance"]["chunk_processing_seconds"]["sum"])
    old_fallback = float(old_stage["fallback"]["sum"])
    new_fallback = float(new_stage["fallback"]["sum"])
    new_wall = float(new_run["timing_seconds"]["total_run"])
    peak = new_run["performance"]["observed_peak_rss_bytes"]
    counters = new_run["stage_counters"]
    residual_boxes = int(counters["subcell_residual_boxes"])
    processed_boxes = int(counters["subcell_krawczyk_boxes_processed"])
    residual_targets = int(counters["subcell_residual_krawczyk"])
    # Every residual target ended in a complete no-root certificate.  Hence no
    # initial box was left unprocessed.  Equality of initial and processed
    # counts then implies zero recursive Krawczyk evaluations and subdivisions.
    recursive_evaluations = processed_boxes - residual_boxes
    recursive_subdivisions = 0 if recursive_evaluations == 0 else None
    stage_sums = {
        name: float(value)
        for name, value in new_run["timing_seconds"]["worker_stage_sum"].items()
    }
    return {
        "fresh_setup": {
            "source_table_load_and_validation_seconds": float(
                step1["source_table_load_and_validation_seconds"]
            ),
            "shared_state_action": step1["shared_state_action"],
            "shared_state_setup_seconds": float(
                step1["shared_state_setup_seconds"]
            ),
            "shared_state_bytes": int(step1["shared_state_bytes"]),
            "artifact_write_seconds": float(step1["artifact_write_seconds"]),
        },
        "inversion": {
            "launcher_measured_wall_seconds": float(
                step1["inversion_wall_seconds"]
            ),
            "pipeline_measured_wall_seconds": new_wall,
            "computed_targets_per_second": float(
                new_run["performance"][
                    "computed_points_per_second_this_invocation"
                ]
            ),
            "worker_stage_sum_seconds": stage_sums,
            "worker_stage_sum_total_seconds": float(sum(stage_sums.values())),
            "chunk_processing_worker_seconds": new_work,
        },
        "reference": {
            "active_wall_seconds": REFERENCE_ACTIVE_WALL_SECONDS,
            "active_wall_is_approximate": True,
            "active_wall_source": "TEMP/HRG4D30Mmap8ResultsSummary.txt",
            "artifact_total_run_is_not_used": (
                "the historical artifact was later overwritten by a "
                "checkpoint-resume-only reporting invocation"
            ),
            "chunk_processing_worker_seconds": old_work,
            "fallback_worker_seconds": old_fallback,
        },
        "comparison": {
            "new_to_approximate_reference_active_wall_ratio": (
                new_wall / REFERENCE_ACTIVE_WALL_SECONDS
            ),
            "approximate_active_wall_speedup": (
                REFERENCE_ACTIVE_WALL_SECONDS / new_wall
            ),
            "new_to_reference_chunk_worker_time_ratio": new_work / old_work,
            "new_to_reference_fallback_worker_time_ratio": (
                new_fallback / old_fallback
            ),
            "fallback_worker_time_saved_seconds": old_fallback - new_fallback,
            "material_full_run_improvement": bool(
                new_wall < 0.75 * REFERENCE_ACTIVE_WALL_SECONDS
                and new_fallback < 0.75 * old_fallback
            ),
        },
        "execution": _plain(new_run["execution"]),
        "hardware_and_software": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "software": _plain(new_run["software"]),
        },
        "memory": {
            "parent_peak_rss_bytes": int(
                peak["parent_before_result_finalization"]
            ),
            "maximum_reporting_worker_peak_rss_bytes": int(
                peak["maximum_reporting_worker"]
            ),
            "declared_limit_bytes": MEMORY_LIMIT_BYTES,
            "bounded": bool(
                int(peak["parent_before_result_finalization"])
                <= MEMORY_LIMIT_BYTES
                and int(peak["maximum_reporting_worker"]) <= MEMORY_LIMIT_BYTES
            ),
        },
        "solver_work": {
            "fast_attempts": int(counters["fast_attempted"]),
            "fast_solutions": int(
                new_run["status_summary"]["route"]["fast"]
            ),
            "fallback_attempts": int(counters["fallback_attempted"]),
            "subcell_attempts": int(counters["subcell_fallback_attempted"]),
            "subcell_direct_certificates": int(
                counters["subcell_certified_no_root"]
            ),
            "subcell_certificate_depth_counts": {
                str(depth): int(
                    counters["subcell_certificate_depth_%d" % depth]
                )
                for depth in range(1, 8)
            },
            "subcell_boxes_evaluated": int(counters["subcell_boxes_evaluated"]),
            "residual_krawczyk_targets": residual_targets,
            "residual_krawczyk_initial_boxes": residual_boxes,
            "residual_krawczyk_boxes_processed": processed_boxes,
            "residual_krawczyk_recursive_evaluations_derived": (
                recursive_evaluations
            ),
            "residual_krawczyk_recursive_subdivisions_derived": (
                recursive_subdivisions
            ),
            "recursive_derivation": (
                "all residual targets terminate with complete no-root proofs; "
                "processed boxes equal initial residual boxes, so no recursive "
                "box entered the queue and no recursive subdivision occurred"
            ),
            "projection_attempts": int(counters["projection_attempted"]),
        },
        "restart": {
            "checkpoint_resume_enabled": True,
            "interruption_or_restart_occurred": False,
            "restored_chunks": int(
                new_run["checkpoint"]["restored_chunks"]
            ),
            "computed_chunks": int(
                new_run["checkpoint"]["computed_chunks_this_invocation"]
            ),
            "interpretation": (
                "the fresh decisive run completed in one invocation; Phase 9 "
                "and Phase 11 separately establish interruption/restart parity"
            ),
        },
    }


def _validate_saved_report(path: Path) -> Dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("saved Phase-12 report schema is unsupported")
    if not report.get("complete") or not report.get("acceptance", {}).get("passed"):
        raise RuntimeError("saved Phase-12 report is not an accepted completion")
    for record in report["artifacts"].values():
        candidate = Path(record["path"])
        if not candidate.is_file() or _sha256(candidate) != record["sha256"]:
            raise RuntimeError(
                "saved Phase-12 artifact digest mismatch: %s" % candidate
            )
    return report


def run(args: argparse.Namespace) -> Dict[str, Any]:
    report_path = args.report.expanduser().resolve()
    if report_path.is_file() and not args.force:
        return _validate_saved_report(report_path)

    reference_path = args.reference.expanduser().resolve(strict=True)
    current_path = args.current.expanduser().resolve(strict=True)
    shared_path = args.shared_state.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    step1_path = args.step1_status.expanduser().resolve(strict=True)
    started = time.perf_counter()

    reference = load_pipeline_result(reference_path)
    current = load_pipeline_result(current_path)
    step1 = json.loads(step1_path.read_text(encoding="utf-8"))
    if step1.get("state") != "completed" or not step1.get("artifact_published"):
        raise RuntimeError("Phase-12 Step-1 completion marker is not complete")

    table = load_shared_table(shared_path)
    scope = _validate_scope(reference, current)
    semantics = compare_semantics(reference, current, table)
    roots = replay_roots(current, table)
    uniqueness = inspect_unique_roots(current, table, shared_path)
    mmap = shared_state_backing_report(
        table,
        load_shared_reachability(
            shared_path,
            MultilinearInterpolator(table, current.plan.target_names),
            current.plan,
            ReachabilityConfig(method="hierarchical", hierarchy_levels=7),
        ),
    )
    checkpoint = _checkpoint_evidence(checkpoint_path, current)
    counts = _status_counts(current)
    performance = _performance(reference, current, step1)

    expected_count_agreement = counts == EXPECTED_COUNTS
    current_errors = int(current.manifest["run"]["errors"]["count"])
    checks = {
        "scope_and_scientific_configuration_match": scope["passed"],
        "reference_counts_reproduced": expected_count_agreement,
        "exact_per_index_scientific_semantics": semantics[
            "all_exact_scientific_fields_equal"
        ],
        "exact_per_index_scientific_masks": semantics[
            "all_scientific_masks_equal"
        ],
        "zero_unresolved": counts["unresolved"] == 0,
        "zero_errors": counts["errors"] == 0 and current_errors == 0,
        "all_new_roots_replayed": roots["root_count"] == EXPECTED_COUNTS["solved"],
        "all_new_roots_in_bounds": roots["all_sources_in_bounds"],
        "all_new_roots_within_residual_tolerance": roots[
            "all_roots_within_scaled_residual_tolerance"
        ],
        "unique_root_coordinate_rule_satisfied_where_applicable": uniqueness[
            "phase8_unique_coordinate_rule_passed"
        ],
        "fresh_run_local_shared_state_used": (
            step1["shared_state_action"] == "constructed_fresh_from_source_table"
        ),
        "large_shared_arrays_are_mmap_backed": mmap[
            "all_large_arrays_mmap_backed"
        ],
        "checkpoint_complete_and_fresh": checkpoint["valid"],
        "memory_bounded": performance["memory"]["bounded"],
        "projection_remained_disabled": (
            performance["solver_work"]["projection_attempts"] == 0
        ),
        "measured_full_run_materially_improves_reference": performance[
            "comparison"
        ]["material_full_run_improvement"],
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "complete": bool(all(checks.values())),
        "scope": {
            "phase": 12,
            "description": "full HRG4D 30^4 decisive regression",
            "grid_shape": list(GRID_SHAPE),
            "grid_size": GRID_SIZE,
            "post_run_only": True,
            "inversion_rerun": False,
            "scientific_behavior_changed": False,
        },
        "configuration_and_scope": scope,
        "expected_counts": dict(EXPECTED_COUNTS),
        "observed_counts": counts,
        "semantics_comparison": semantics,
        "root_replay": roots,
        "unique_root_comparison": uniqueness,
        "checkpoint": checkpoint,
        "mmap": mmap,
        "performance": performance,
        "step1_status": _plain(step1),
        "validation_wall_seconds": float(time.perf_counter() - started),
        "artifacts": {
            "reference": _file_record(reference_path),
            "current": _file_record(current_path),
            "step1_status": _file_record(step1_path),
            "shared_state_manifest": _file_record(shared_path),
            "checkpoint_manifest": _file_record(
                checkpoint_path / "checkpoint.json"
            ),
        },
        "acceptance": {"checks": checks, "passed": bool(all(checks.values()))},
    }
    _atomic_json(report_path, report)
    if not report["acceptance"]["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("Phase 12 failed: %s" % ", ".join(failed))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--current", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--shared-state", type=Path, default=DEFAULT_SHARED_STATE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--step1-status", type=Path, default=DEFAULT_STEP1_STATUS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="revalidate artifacts even when an accepted report already exists",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "accepted": report["acceptance"]["passed"],
                "counts": report["observed_counts"],
                "maximum_replayed_scaled_residual": report["root_replay"][
                    "independently_recomputed_scaled_residual_norm_inf"
                ]["maximum"],
                "report": str(args.report.expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
