#!/usr/bin/env python3
"""Run the restartable v1 Phase-11 medium HRG4D release candidate.

The driver deliberately interrupts a two-process checkpointed run after two
atomic chunks have been published, then resumes the same scientific run with
four workers and a different worker-task size.  It compares that result with
a serial reference, round-trips the completed artifact, validates the 4D
vHLLE export, and records mmap/RSS evidence.  Existing completed evidence is
verified and reused unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from eos_trensmuther import (
    CheckpointConfig,
    CoarseSeedConfig,
    EoSInversionPipeline,
    ExecutionConfig,
    ExportDataError,
    GridAxisSpec,
    InversionGridSpec,
    MultilinearInterpolator,
    PipelineConfig,
    PipelineResult,
    PointDisposition,
    ProjectionConfig,
    ReachabilityConfig,
    SolverConfig,
    load_pipeline_result,
    pipeline_status_code,
    save_pipeline_result,
)
from eos_trensmuther.checkpoint import CheckpointStore
from eos_trensmuther.exporters import VHLLEExportConfig, export_music, export_vhlle
from eos_trensmuther.shared_state import (
    load_shared_reachability,
    load_shared_table,
    shared_state_backing_report,
)


SCHEMA_VERSION = 1
GRID_COUNT = 10
GRID_SIZE = GRID_COUNT**4
CHUNK_SIZE = 250
INTERRUPT_AFTER_PUBLISHED_CHUNKS = 2
MEMORY_LIMIT_BYTES = 3 * 1024**3
PROCESS_TO_SERIAL_WALL_LIMIT = 1.25

_MU_B_TILDE_MAX = 1.7201102328734932
_GRID_BOUNDS = {
    "T_tilde": (0.009777504201390019, 0.31403115649771435),
    "muB_tilde": (-_MU_B_TILDE_MAX, _MU_B_TILDE_MAX),
    "muQ_tilde": (-_MU_B_TILDE_MAX / 5.0, _MU_B_TILDE_MAX / 5.0),
    "muS_tilde": (
        -2.0 * _MU_B_TILDE_MAX / 5.0,
        2.0 * _MU_B_TILDE_MAX / 5.0,
    ),
}
_RESIDUAL_SCALES = {
    "e": 0.151971673718546,
    "nB": 0.16451246235940004,
    "nQ": 0.0860935593077,
    "nS": 0.24048482455340003,
}

DEFAULT_SHARED_STATE = Path(
    "Output_TEST/HRG4D30Mmap8/HRG4D30Mmap8/diagnostic/"
    "shared_worker_state_v1/shared_state.json"
)
DEFAULT_OUTPUT = Path("Output_TEST/V1Phase11ReleaseCandidate")

_RESULT_ARRAY_NAMES = (
    "source_coordinates",
    "achieved_targets",
    "residual",
    "scaled_residual",
    "scaled_residual_norm_inf",
    "scaled_distance_l2",
    "auxiliary_values",
    "auxiliary_defined",
    "achieved_tilde_values",
    "achieved_tilde_defined",
    "disposition_codes",
    "route_codes",
    "prescreen_codes",
    "solve_status_codes",
    "fast_status_codes",
    "proof_status_codes",
    "projection_quality_codes",
    "initial_guess_source_codes",
    "phase_codes",
    "phase_route_codes",
    "phase_selection_reason_codes",
    "mixed_candidate",
    "exact_mixed_candidate",
    "mixed_candidate_entry_count",
    "mixed_verified_root_count",
    "phase_boundary",
    "phase_multivalued",
    "line_mixture_fraction",
    "line_mixture_fraction_defined",
    "alternate_single_phase_source",
    "alternate_single_phase_source_defined",
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


class IntentionalPhase11Interruption(RuntimeError):
    """Internal marker for the controlled checkpoint interruption."""


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(_plain(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
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
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def build_config(
    *, workers: int, worker_task_size: int, max_pending_tasks: int
) -> PipelineConfig:
    """Return the frozen Phase-11 scientific configuration."""

    return PipelineConfig(
        grid=InversionGridSpec(
            "tilde",
            {
                name: GridAxisSpec(
                    GRID_COUNT, lower=bounds[0], upper=bounds[1]
                )
                for name, bounds in _GRID_BOUNDS.items()
            },
        ),
        residual_scales=_RESIDUAL_SCALES,
        solver=SolverConfig(),
        projection=ProjectionConfig(enabled=False),
        execution=ExecutionConfig(
            workers=workers,
            start_method="spawn",
            chunk_size=CHUNK_SIZE,
            worker_task_size=worker_task_size,
            max_pending_chunks=max_pending_tasks,
        ),
        coarse_seed=CoarseSeedConfig(enabled=True, points_per_axis=5),
        use_reachability_seed=True,
        auxiliary_fields=("P", "s"),
    )


def reachability_config() -> ReachabilityConfig:
    return ReachabilityConfig(method="hierarchical", hierarchy_levels=7)


def _ensure_shared_state_link(output: Path, source_manifest: Path) -> Path:
    source_manifest = source_manifest.expanduser().resolve(strict=True)
    source_directory = source_manifest.parent
    link = output / "diagnostic" / "shared_worker_state_v1"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve(strict=True) != source_directory:
            raise RuntimeError("Phase-11 shared-state link points elsewhere")
    elif link.exists():
        candidate = (link / "shared_state.json").resolve(strict=True)
        if candidate != source_manifest:
            raise RuntimeError(
                "Phase-11 shared-state directory conflicts with the requested store"
            )
    else:
        link.symlink_to(source_directory, target_is_directory=True)
    return link / "shared_state.json"


def _checkpoint_chunk_records(checkpoint: Path) -> Dict[str, Dict[str, Any]]:
    chunks = checkpoint / "chunks"
    if not chunks.exists():
        return {}
    return {
        path.name: _file_record(path)
        for path in sorted(chunks.glob("*.eoschunk"))
    }


def _load_or_run_serial(
    output: Path, shared_manifest: Path
) -> Tuple[PipelineResult, Dict[str, Any]]:
    artifact = output / "serial_reference.eosinv"
    if artifact.is_file():
        result = load_pipeline_result(artifact)
        if result.grid.shape != (GRID_COUNT,) * 4:
            raise RuntimeError("saved Phase-11 serial artifact has the wrong grid")
        return result, {
            "reused": True,
            "artifact": _file_record(artifact),
            "wall_seconds": 0.0,
            "shared_state_backing": None,
        }

    table = load_shared_table(shared_manifest)
    pipeline = EoSInversionPipeline(
        table,
        build_config(workers=1, worker_task_size=8, max_pending_tasks=1),
        reachability_config=reachability_config(),
    )
    setup_started = time.perf_counter()
    pipeline._prepare(shared_state_manifest=shared_manifest)
    setup_seconds = float(time.perf_counter() - setup_started)
    if pipeline._reachability is None:
        raise RuntimeError("serial Phase-11 reachability was not prepared")
    backing = shared_state_backing_report(table, pipeline._reachability)
    started = time.perf_counter()
    result = pipeline.run()
    wall_seconds = float(time.perf_counter() - started)
    save_pipeline_result(result, artifact, compression="none")
    return result, {
        "reused": False,
        "artifact": _file_record(artifact),
        "external_mmap_setup_seconds": setup_seconds,
        "wall_seconds": wall_seconds,
        "shared_state_backing": backing,
    }


def _run_controlled_interruption(
    output: Path, shared_manifest: Path
) -> Dict[str, Any]:
    checkpoint = output / "diagnostic" / "checkpoint"
    before = _checkpoint_chunk_records(checkpoint)
    if before:
        evidence_path = output / "diagnostic" / "interruption.json"
        if not evidence_path.is_file():
            raise RuntimeError(
                "partial Phase-11 checkpoint exists without interruption evidence"
            )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if int(evidence["published_chunk_count"]) < 1:
            raise RuntimeError("saved interruption evidence has no published chunk")
        return evidence

    table = load_shared_table(shared_manifest)
    pipeline = EoSInversionPipeline(
        table,
        build_config(workers=2, worker_task_size=8, max_pending_tasks=4),
        reachability_config=reachability_config(),
    )
    original = CheckpointStore.save_chunk
    published: list[Tuple[int, int]] = []

    def interrupting_save(store: CheckpointStore, chunk: Any) -> Path:
        path = original(store, chunk)
        published.append((int(chunk.start), int(chunk.stop)))
        if len(published) == INTERRUPT_AFTER_PUBLISHED_CHUNKS:
            raise IntentionalPhase11Interruption(
                "controlled stop after %d atomic chunks"
                % INTERRUPT_AFTER_PUBLISHED_CHUNKS
            )
        return path

    started = time.perf_counter()
    observed = None
    CheckpointStore.save_chunk = interrupting_save
    try:
        pipeline.run(checkpoint=CheckpointConfig(checkpoint, resume=True))
    except IntentionalPhase11Interruption as exc:
        observed = str(exc)
    finally:
        CheckpointStore.save_chunk = original
    elapsed = float(time.perf_counter() - started)
    if observed is None:
        raise RuntimeError("Phase-11 controlled interruption did not occur")
    chunks = _checkpoint_chunk_records(checkpoint)
    if len(chunks) != INTERRUPT_AFTER_PUBLISHED_CHUNKS:
        raise RuntimeError(
            "controlled interruption published %d checkpoint chunks, expected %d"
            % (len(chunks), INTERRUPT_AFTER_PUBLISHED_CHUNKS)
        )
    evidence = {
        "kind": "controlled_parent_checkpoint_publication_stop",
        "message": observed,
        "elapsed_seconds": elapsed,
        "published_ranges": [list(item) for item in published],
        "published_chunk_count": len(chunks),
        "chunks_before_resume": chunks,
        "initial_execution": {
            "workers": 2,
            "worker_task_size": 8,
            "maximum_pending_tasks": 4,
        },
    }
    _write_json(output / "diagnostic" / "interruption.json", evidence)
    return evidence


def _run_resume(
    output: Path, shared_manifest: Path, interruption: Mapping[str, Any]
) -> Tuple[PipelineResult, Dict[str, Any]]:
    checkpoint = output / "diagnostic" / "checkpoint"
    artifact = output / "process_resumed.eosinv"
    before = {
        name: dict(record)
        for name, record in interruption["chunks_before_resume"].items()
    }
    if artifact.is_file():
        result = load_pipeline_result(artifact)
        run_manifest = result.manifest["run"]
        after = _checkpoint_chunk_records(checkpoint)
        preserved = all(
            name in after and after[name]["sha256"] == record["sha256"]
            for name, record in before.items()
        )
        if not preserved:
            raise RuntimeError("published checkpoint chunks changed during resume")
        return result, {
            "reused": True,
            "wall_seconds": float(run_manifest["timing_seconds"]["total_run"]),
            "artifact": _file_record(artifact),
            "resume_execution": {
                "workers": 4,
                "worker_task_size": 16,
                "maximum_pending_tasks": 8,
            },
            "checkpoint_chunks_before_resume": len(before),
            "checkpoint_chunks_after_resume": len(after),
            "published_chunk_hashes_preserved": preserved,
            "manifest_checkpoint": _plain(run_manifest["checkpoint"]),
            "manifest_execution": _plain(run_manifest["execution"]),
        }
    table = load_shared_table(shared_manifest)
    pipeline = EoSInversionPipeline(
        table,
        build_config(workers=4, worker_task_size=16, max_pending_tasks=8),
        reachability_config=reachability_config(),
    )
    started = time.perf_counter()
    result = pipeline.run(checkpoint=CheckpointConfig(checkpoint, resume=True))
    wall_seconds = float(time.perf_counter() - started)
    after = _checkpoint_chunk_records(checkpoint)
    preserved = all(
        name in after and after[name]["sha256"] == record["sha256"]
        for name, record in before.items()
    )
    if not preserved:
        raise RuntimeError("published checkpoint chunks changed during resume")
    save_pipeline_result(result, artifact, overwrite=artifact.exists(), compression="none")
    run_manifest = result.manifest["run"]
    return result, {
        "reused": False,
        "wall_seconds": wall_seconds,
        "artifact": _file_record(artifact),
        "resume_execution": {
            "workers": 4,
            "worker_task_size": 16,
            "maximum_pending_tasks": 8,
        },
        "checkpoint_chunks_before_resume": len(before),
        "checkpoint_chunks_after_resume": len(after),
        "published_chunk_hashes_preserved": preserved,
        "manifest_checkpoint": _plain(run_manifest["checkpoint"]),
        "manifest_execution": _plain(run_manifest["execution"]),
    }


def _code_counts(array: np.ndarray, codebook: Mapping[str, str]) -> Dict[str, int]:
    values = np.asarray(array, dtype=np.uint8).reshape(-1)
    counts = Counter(int(value) for value in values)
    return {
        str(label): int(counts.get(int(code), 0))
        for code, label in sorted(codebook.items(), key=lambda item: int(item[0]))
    }


def _compare_results(
    serial: PipelineResult,
    process: PipelineResult,
    table: Any,
) -> Dict[str, Any]:
    exact = {}
    discrepancies = {}
    for name in _EXACT_SCIENTIFIC_FIELDS:
        left = np.asarray(getattr(serial, name))
        right = np.asarray(getattr(process, name))
        same = bool(np.array_equal(left, right))
        exact[name] = same
        discrepancies[name] = int(np.count_nonzero(left != right))

    solved_code = pipeline_status_code(PointDisposition.SOLVED)
    serial_solved = np.asarray(serial.disposition_codes) == solved_code
    process_solved = np.asarray(process.disposition_codes) == solved_code
    solved_masks_equal = bool(np.array_equal(serial_solved, process_solved))
    source_lower = np.asarray([axis[0] for axis in table.axes], dtype=float)
    source_upper = np.asarray([axis[-1] for axis in table.axes], dtype=float)
    source_span = source_upper - source_lower
    process_source = np.asarray(process.source_coordinates)[process_solved]
    in_bounds = bool(
        np.all(process_source >= source_lower)
        and np.all(process_source <= source_upper)
    )
    reported_residual = np.asarray(process.scaled_residual_norm_inf)[
        process_solved
    ]
    maximum_reported = (
        float(np.max(reported_residual)) if reported_residual.size else 0.0
    )

    interpolator = MultilinearInterpolator(table, process.plan.target_names)
    forward = interpolator.evaluate(process_source)
    achieved = np.asarray(process.achieved_targets)[process_solved]
    stored_residual = np.asarray(process.residual)[process_solved]
    requested = achieved - stored_residual
    independent_scaled = np.max(
        np.abs((forward - requested) / process.plan.residual_scales), axis=1
    )
    maximum_independent = (
        float(np.max(independent_scaled)) if independent_scaled.size else 0.0
    )
    forward_agreement = (
        float(np.max(np.abs(forward - achieved))) if forward.size else 0.0
    )

    coordinate_error = np.zeros(0, dtype=float)
    if solved_masks_equal and np.any(process_solved):
        coordinate_error = np.max(
            np.abs(
                (
                    np.asarray(process.source_coordinates)[process_solved]
                    - np.asarray(serial.source_coordinates)[serial_solved]
                )
                / source_span
            ),
            axis=1,
        )

    tolerance = process.plan.config.solver.root_residual_tolerance
    return {
        "exact_scientific_fields": exact,
        "field_discrepancy_counts": discrepancies,
        "all_exact_scientific_fields_equal": all(exact.values()),
        "solved_masks_equal": solved_masks_equal,
        "solved_count": int(np.count_nonzero(process_solved)),
        "process_solutions_in_source_bounds": in_bounds,
        "root_residual_tolerance": tolerance,
        "maximum_reported_scaled_residual": maximum_reported,
        "maximum_independent_scaled_residual": maximum_independent,
        "maximum_forward_replay_absolute_difference": forward_agreement,
        "maximum_full_span_normalized_serial_process_coordinate_difference": (
            float(np.max(coordinate_error)) if coordinate_error.size else 0.0
        ),
        "all_process_roots_within_residual_tolerance": bool(
            maximum_reported <= tolerance and maximum_independent <= tolerance
        ),
    }


def _validate_artifact_roundtrip(
    expected: PipelineResult, artifact: Path
) -> Dict[str, Any]:
    loaded = load_pipeline_result(artifact)
    matches = {}
    for name in _RESULT_ARRAY_NAMES:
        matches[name] = bool(
            np.array_equal(
                np.asarray(getattr(expected, name)),
                np.asarray(getattr(loaded, name)),
                equal_nan=True,
            )
        )
    return {
        "artifact": _file_record(artifact),
        "all_dense_arrays_equal": all(matches.values()),
        "array_equality": matches,
        "manifest_equal": _plain(expected.manifest) == _plain(loaded.manifest),
    }


def _count_text_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip() and not line.startswith("#"))


def _validate_exports(result: PipelineResult, output: Path) -> Dict[str, Any]:
    export_directory = output / "vhlle"
    export_directory.mkdir(parents=True, exist_ok=True)
    table_path = export_directory / "EoS_all.dat"
    exported = export_vhlle(
        result,
        table_path,
        config=VHLLEExportConfig(allow_unusable=True),
        overwrite=table_path.exists(),
    )
    if exported.manifest_path is None:
        raise RuntimeError("Phase-11 vHLLE export omitted its manifest")
    manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
    rows = _count_text_rows(table_path)
    music_error = None
    try:
        export_music(result, output / "music" / "EoS_all.dat")
    except ExportDataError as exc:
        music_error = str(exc)
    if music_error is None or "supports dimensions 2D" not in music_error:
        raise RuntimeError("4D MUSIC applicability check did not fail clearly")
    return {
        "vhlle": {
            "table": _file_record(table_path),
            "manifest": _file_record(exported.manifest_path),
            "row_count": rows,
            "reported_row_count": int(exported.row_count),
            "manifest_row_count": int(manifest["row_count"]),
            "dimension": int(manifest["dimension"]),
            "column_contract": manifest["column_contract"],
            "valid": bool(
                rows == GRID_SIZE
                and exported.row_count == GRID_SIZE
                and manifest["row_count"] == GRID_SIZE
                and manifest["dimension"] == 4
                and manifest["table_sha256"] == _sha256(table_path)
            ),
        },
        "music": {
            "applicable": False,
            "expected_rejection": music_error,
            "valid": True,
        },
    }


def route_coverage(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    counters = manifest["stage_counters"]
    status = manifest["status_summary"]
    requirements = {
        "prescreen_exclusion": int(counters["prescreen_excluded"]) > 0,
        "fast_attempt": int(counters["fast_attempted"]) > 0,
        "fast_solution": int(status["route"]["fast"]) > 0,
        "fallback_attempt": int(counters["fallback_attempted"]) > 0,
        "subcell_attempt": int(counters["subcell_fallback_attempted"]) > 0,
        "subcell_certificate": int(counters["subcell_certified_no_root"]) > 0,
    }
    depth_counts = {
        str(depth): int(counters["subcell_certificate_depth_%d" % depth])
        for depth in range(1, 8)
    }
    return {
        "required_routes": requirements,
        "all_required_routes_observed": all(requirements.values()),
        "subcell_certificate_depth_counts": depth_counts,
        "residual_krawczyk_targets": int(counters["subcell_residual_krawczyk"]),
        "subcell_boxes_evaluated": int(counters["subcell_boxes_evaluated"]),
        "subcell_residual_boxes": int(counters["subcell_residual_boxes"]),
        "subcell_krawczyk_boxes_processed": int(
            counters["subcell_krawczyk_boxes_processed"]
        ),
    }


def _performance_evidence(
    serial: PipelineResult,
    process: PipelineResult,
    interruption: Mapping[str, Any],
) -> Dict[str, Any]:
    serial_run = serial.manifest["run"]
    process_run = process.manifest["run"]
    serial_timing = serial_run["timing_seconds"]
    process_timing = process_run["timing_seconds"]
    serial_total = float(serial_timing["total_run"])
    process_total = float(process_timing["total_run"])
    interrupted = float(interruption["elapsed_seconds"])
    serial_fallback = float(serial_timing["worker_stage_sum"]["fallback"])
    process_fallback = float(process_timing["worker_stage_sum"]["fallback"])
    serial_counters = serial_run["stage_counters"]
    process_counters = process_run["stage_counters"]
    subcell_counter_names = (
        "subcell_fallback_attempted",
        "subcell_certified_no_root",
        "subcell_guarded_incomplete",
        "subcell_residual_krawczyk",
        "subcell_boxes_evaluated",
        "subcell_residual_boxes",
        "subcell_krawczyk_boxes_processed",
    ) + tuple(
        "subcell_certificate_depth_%d" % depth for depth in range(1, 8)
    )
    subcell_counter_parity = {
        name: int(serial_counters[name]) == int(process_counters[name])
        for name in subcell_counter_names
    }
    process_execution = process_run["execution"]
    process_perf = process_run["performance"]
    peak = process_perf["observed_peak_rss_bytes"]
    parent_peak = int(peak["parent_before_result_finalization"])
    worker_peak = int(peak["maximum_reporting_worker"])
    scheduler_bound_ok = bool(
        int(process_execution["maximum_concurrent_internal_batches"])
        <= int(process_execution["max_pending_chunks"])
    )
    return {
        "serial_total_seconds": serial_total,
        "resumed_process_invocation_seconds": process_total,
        "controlled_interrupted_invocation_seconds": interrupted,
        "total_process_validation_seconds_including_intentional_redo": (
            process_total + interrupted
        ),
        "resumed_process_to_serial_wall_ratio": process_total / serial_total,
        "serial_fallback_worker_seconds": serial_fallback,
        "process_fallback_worker_seconds": process_fallback,
        "process_to_serial_summed_fallback_worker_seconds_ratio": (
            process_fallback / serial_fallback if serial_fallback else 0.0
        ),
        "fallback_timing_interpretation": (
            "diagnostic sum of per-worker wall durations; concurrent sums are "
            "not an algorithmic-work measure"
        ),
        "subcell_work_counter_parity": subcell_counter_parity,
        "scheduler_pending_bound_respected": scheduler_bound_ok,
        "parent_peak_rss_bytes": parent_peak,
        "maximum_reporting_worker_peak_rss_bytes": worker_peak,
        "rss_limit_bytes": MEMORY_LIMIT_BYTES,
        "bounded_memory": bool(
            parent_peak <= MEMORY_LIMIT_BYTES and worker_peak <= MEMORY_LIMIT_BYTES
        ),
        "no_scheduler_wall_regression": bool(
            process_total <= PROCESS_TO_SERIAL_WALL_LIMIT * serial_total
        ),
        "no_subcell_work_regression": bool(
            all(subcell_counter_parity.values())
        ),
        "declared_limits": {
            "resumed_process_to_serial_wall": PROCESS_TO_SERIAL_WALL_LIMIT,
        },
    }


def _validate_completed_report(path: Path) -> Dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("completed Phase-11 report schema is unsupported")
    if not report.get("complete") or not report.get("acceptance", {}).get("passed"):
        raise RuntimeError("saved Phase-11 report is not an accepted completion")
    for record in report["artifacts"].values():
        candidate = Path(record["path"])
        if not candidate.is_file() or _sha256(candidate) != record["sha256"]:
            raise RuntimeError("saved Phase-11 artifact digest mismatch: %s" % candidate)
    return report


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase11_report.json"
    if report_path.is_file() and not args.force:
        return _validate_completed_report(report_path)

    shared_manifest = _ensure_shared_state_link(output, args.shared_state)
    serial, serial_record = _load_or_run_serial(output, shared_manifest)
    interruption = _run_controlled_interruption(output, shared_manifest)
    process, process_record = _run_resume(output, shared_manifest, interruption)
    table = load_shared_table(shared_manifest)

    comparison = _compare_results(serial, process, table)
    roundtrip = _validate_artifact_roundtrip(
        process, output / "process_resumed.eosinv"
    )
    exports = _validate_exports(process, output)
    process_run_manifest = process.manifest["run"]
    coverage = route_coverage(process_run_manifest)
    performance = _performance_evidence(serial, process, interruption)
    backing = serial_record.get("shared_state_backing")
    if backing is None:
        interpolator = MultilinearInterpolator(table, process.plan.target_names)
        index = load_shared_reachability(
            shared_manifest, interpolator, process.plan, reachability_config()
        )
        backing = shared_state_backing_report(table, index)

    status_counts = {
        name: _code_counts(
            getattr(process, name + "_codes"), process.codebooks[name]
        )
        for name in ("disposition", "route", "prescreen", "solve_status", "proof_status")
    }
    errors = int(process_run_manifest["errors"]["count"])
    acceptance_checks = {
        "grid_is_exactly_10_power_4": process.grid.size == GRID_SIZE,
        "serial_process_scientific_status_parity": comparison[
            "all_exact_scientific_fields_equal"
        ],
        "no_point_errors": errors == 0,
        "roots_are_valid": bool(
            comparison["process_solutions_in_source_bounds"]
            and comparison["all_process_roots_within_residual_tolerance"]
        ),
        "controlled_checkpoint_interruption_observed": int(
            interruption["published_chunk_count"]
        )
        >= 1,
        "published_chunks_preserved_on_resume": process_record[
            "published_chunk_hashes_preserved"
        ],
        "resume_changed_worker_and_task_settings": bool(
            interruption["initial_execution"]["workers"]
            != process_record["resume_execution"]["workers"]
            and interruption["initial_execution"]["worker_task_size"]
            != process_record["resume_execution"]["worker_task_size"]
        ),
        "artifact_roundtrip_valid": bool(
            roundtrip["all_dense_arrays_equal"] and roundtrip["manifest_equal"]
        ),
        "applicable_exports_valid": bool(
            exports["vhlle"]["valid"] and exports["music"]["valid"]
        ),
        "required_routes_observed": coverage["all_required_routes_observed"],
        "large_worker_state_arrays_are_mmap_backed": bool(
            backing["all_large_arrays_mmap_backed"]
        ),
        "memory_is_bounded": performance["bounded_memory"],
        "scheduler_pending_bound_respected": performance[
            "scheduler_pending_bound_respected"
        ],
        "no_scheduler_wall_regression": performance[
            "no_scheduler_wall_regression"
        ],
        "no_subcell_work_regression": performance[
            "no_subcell_work_regression"
        ],
    }
    artifacts = {
        "serial_reference": serial_record["artifact"],
        "process_resumed": process_record["artifact"],
        "vhlle_table": exports["vhlle"]["table"],
        "vhlle_manifest": exports["vhlle"]["manifest"],
        "interruption": _file_record(
            output / "diagnostic" / "interruption.json"
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "complete": all(acceptance_checks.values()),
        "scope": {
            "phase": 11,
            "description": "medium end-to-end HRG4D release candidate",
            "grid_shape": [GRID_COUNT] * 4,
            "grid_size": GRID_SIZE,
            "grid_space": "tilde",
            "scientific_behavior_changed": False,
        },
        "configuration": {
            "grid_bounds": {
                name: list(bounds) for name, bounds in _GRID_BOUNDS.items()
            },
            "residual_scales": dict(_RESIDUAL_SCALES),
            "root_residual_tolerance": (
                process.plan.config.solver.root_residual_tolerance
            ),
            "reachability": {
                "method": "hierarchical",
                "hierarchy_levels": 7,
            },
            "subcell_fallback": {
                "enabled": process.plan.config.solver.enable_subcell_fallback,
                "maximum_depth": process.plan.config.solver.max_subcell_depth,
                "maximum_active_boxes": (
                    process.plan.config.solver.max_subcell_active_boxes
                ),
            },
            "chunk_size": CHUNK_SIZE,
        },
        "serial_reference": serial_record,
        "interruption": interruption,
        "resumed_process": process_record,
        "comparison": comparison,
        "status_counts": status_counts,
        "route_coverage": coverage,
        "artifact_roundtrip": roundtrip,
        "exports": exports,
        "mmap": backing,
        "performance": performance,
        "errors": _plain(process_run_manifest["errors"]),
        "artifacts": artifacts,
        "acceptance": {
            "checks": acceptance_checks,
            "passed": all(acceptance_checks.values()),
        },
    }
    _write_json(report_path, report)
    if not report["acceptance"]["passed"]:
        failed = [name for name, passed in acceptance_checks.items() if not passed]
        raise RuntimeError("Phase 11 failed: %s" % ", ".join(failed))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-state", type=Path, default=DEFAULT_SHARED_STATE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="revalidate outputs even when an accepted report already exists",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "accepted": report["acceptance"]["passed"],
                "grid_size": report["scope"]["grid_size"],
                "status_counts": report["status_counts"]["disposition"],
                "report": str(
                    parse_args(argv).output.expanduser().resolve()
                    / "phase11_report.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
