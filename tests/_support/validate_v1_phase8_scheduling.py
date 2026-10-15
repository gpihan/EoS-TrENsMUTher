#!/usr/bin/env python3
"""Measure the Phase-7 scheduler and validate Phase-8 HRG4D parity.

The fixed 128-target workload is assembled from preserved production evidence:
prescreen exclusions and known roots from the completed HRG4D run/Phase 5,
plus shallow, intermediate, deep, and residual-Krawczyk no-root cases from the
accepted Phase-6 report.  Workers attach the existing read-only mmap state;
the 1 GiB source table and hierarchy are never rebuilt by this driver.

Every matrix cell is saved independently and can be reused after interruption.
The benchmark does not alter production checkpoints or result artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import resource
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields as dataclass_fields, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from eos_trensmuther import (
    EoSInversionPipeline,
    ExecutionConfig,
    PipelineConfig,
    PrescreenStatus,
    ReachabilityConfig,
    SolveStatus,
    load_pipeline_result,
    pipeline_status_code,
)
from eos_trensmuther._scheduler import execute_dynamic_tasks
from eos_trensmuther.pipeline import _worker_thread_environment
from eos_trensmuther.shared_state import load_shared_table


SCHEMA_VERSION = 1
SELECTION_SEED = 20261007
TARGET_COUNT = 128
OUTER_CHUNK_SIZE = 64
TASK_SIZES = (8, 16, 32)
WORKER_COUNTS = (1, 2, 4, 8)
NEAR_BEST_FRACTION = 0.10
MINIMUM_FOUR_WORKER_SPEEDUP = 1.25

CATEGORY_QUOTAS = {
    "prescreen_excluded": 48,
    "fast_root": 32,
    "subcell_shallow": 24,
    "subcell_intermediate": 12,
    "subcell_deep": 6,
    "residual_krawczyk": 6,
}

DEFAULT_ARTIFACT = Path(
    "Output_TEST/HRG4D30Mmap8/HRG4D30Mmap8/inverse.eosinv"
)
DEFAULT_SHARED_STATE = Path(
    "Output_TEST/HRG4D30Mmap8/HRG4D30Mmap8/diagnostic/"
    "shared_worker_state_v1/shared_state.json"
)
DEFAULT_PHASE5_ROOT = Path("Output_TEST/V1Phase5RootPreservation")
DEFAULT_PHASE6_REPORT = Path(
    "Output_TEST/V1Phase6PairedBenchmarkOptimized/phase6_report.json"
)
DEFAULT_OUTPUT = Path("Output_TEST/V1Phase8TaskSizeBenchmark")

_STATUS_ARRAY_NAMES = ("prescreen", "status", "proof")
_FLOAT_ARRAY_NAMES = ("source_coordinates", "scaled_residual_norm_inf")


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


def _write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
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


def _json_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distribution(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        return {
            "minimum": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "maximum": 0.0,
            "sum": 0.0,
        }
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
        "sum": float(np.sum(array)),
    }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 1 or source.size < count or count <= 0:
        raise RuntimeError("selection pool cannot satisfy its declared quota")
    positions = np.linspace(0, source.size - 1, count, dtype=np.int64)
    result = source[positions]
    if np.unique(result).size != count:
        raise RuntimeError("stratified selection produced duplicate rows")
    return result


def _grid_targets(grid: Any, flat_indices: np.ndarray) -> np.ndarray:
    rows = [
        grid.chunk(int(index), int(index) + 1).canonical_targets[0]
        for index in np.asarray(flat_indices, dtype=np.int64)
    ]
    return np.asarray(rows, dtype=float)


def _create_selection(
    artifact_path: Path,
    phase5_root: Path,
    phase6_report_path: Path,
) -> Dict[str, np.ndarray]:
    artifact = load_pipeline_result(artifact_path)
    excluded_pool = np.flatnonzero(
        np.asarray(artifact.prescreen_excluded_mask).reshape(-1)
    )
    excluded_indices = _evenly_spaced(
        excluded_pool, CATEGORY_QUOTAS["prescreen_excluded"]
    )
    excluded_targets = _grid_targets(artifact.plan.grid, excluded_indices)

    with np.load(phase5_root / "tier_b_selection.npz", allow_pickle=False) as saved:
        tier_b_targets = np.asarray(saved["targets"], dtype=float)
        tier_b_sources = np.asarray(saved["source_coordinates"], dtype=float)
        tier_b_flat = np.asarray(saved["tier_a_indices"], dtype=np.int64)
        tier_b_candidate_counts = np.asarray(
            saved["candidate_counts"], dtype=np.int64
        )
    phase5_report = json.loads(
        (phase5_root / "phase5_report.json").read_text(encoding="utf-8")
    )
    converged_positions = np.asarray(
        [
            int(row["index"])
            for row in phase5_report["tier_b_rows"]
            if row["status"] == SolveStatus.CONVERGED_FAST.value
        ],
        dtype=np.int64,
    )
    phase5_elapsed = {
        int(row["index"]): float(row["elapsed_seconds"])
        for row in phase5_report["tier_b_rows"]
    }
    root_positions = np.asarray(
        sorted(
            (int(value) for value in converged_positions),
            key=lambda value: (
                int(tier_b_candidate_counts[value]),
                phase5_elapsed[value],
                value,
            ),
        )[: CATEGORY_QUOTAS["fast_root"]],
        dtype=np.int64,
    )

    phase6 = json.loads(phase6_report_path.read_text(encoding="utf-8"))
    phase6_rows = list(phase6["rows"])

    def select_phase6(
        name: str, predicate: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        quota = CATEGORY_QUOTAS[name]
        available = [row for row in phase6_rows if predicate(row)]
        selected = sorted(
            available,
            key=lambda row: (
                float(row["new"]["elapsed_seconds"]),
                int(row["flat_index"]),
            ),
        )[:quota]
        if len(selected) != quota:
            raise RuntimeError("Phase-6 evidence cannot satisfy %s quota" % name)
        return (
            np.asarray([row["target"] for row in selected], dtype=float),
            np.asarray([row["flat_index"] for row in selected], dtype=np.int64),
            np.asarray(
                [row["new"]["subcell"]["completed_depth"] for row in selected],
                dtype=np.int64,
            ),
            np.asarray(
                [row["new"]["elapsed_seconds"] for row in selected], dtype=float
            ),
        )

    shallow = select_phase6(
        "subcell_shallow",
        lambda row: row["new"]["subcell_residual_box_count"] == 0
        and row["new"]["subcell"]["completed_depth"] <= 2,
    )
    intermediate = select_phase6(
        "subcell_intermediate",
        lambda row: row["new"]["subcell_residual_box_count"] == 0
        and 3 <= row["new"]["subcell"]["completed_depth"] <= 5,
    )
    deep = select_phase6(
        "subcell_deep",
        lambda row: row["new"]["subcell_residual_box_count"] == 0
        and row["new"]["subcell"]["completed_depth"] >= 6,
    )
    residual = select_phase6(
        "residual_krawczyk",
        lambda row: row["new"]["subcell_residual_box_count"] > 0,
    )

    targets = []
    categories = []
    source_flat_indices = []
    expected_depths = []
    known_sources = []
    historical_seconds = []

    def append(
        name: str,
        selected_targets: np.ndarray,
        flat_indices: np.ndarray,
        depths: np.ndarray,
        sources: Optional[np.ndarray] = None,
        durations: Optional[np.ndarray] = None,
    ) -> None:
        count = selected_targets.shape[0]
        targets.extend(selected_targets)
        categories.extend([name] * count)
        source_flat_indices.extend(np.asarray(flat_indices, dtype=np.int64))
        expected_depths.extend(np.asarray(depths, dtype=np.int64))
        if sources is None:
            known_sources.extend(np.full((count, 4), np.nan, dtype=float))
        else:
            known_sources.extend(np.asarray(sources, dtype=float))
        if durations is None:
            historical_seconds.extend(np.zeros(count, dtype=float))
        else:
            historical_seconds.extend(np.asarray(durations, dtype=float))

    append(
        "prescreen_excluded",
        excluded_targets,
        excluded_indices,
        np.full(excluded_indices.size, -1, dtype=np.int64),
    )
    append(
        "fast_root",
        tier_b_targets[root_positions],
        tier_b_flat[root_positions],
        np.full(root_positions.size, -1, dtype=np.int64),
        tier_b_sources[root_positions],
    )
    for name, selected in (
        ("subcell_shallow", shallow),
        ("subcell_intermediate", intermediate),
        ("subcell_deep", deep),
        ("residual_krawczyk", residual),
    ):
        append(name, selected[0], selected[1], selected[2], durations=selected[3])

    targets_array = np.asarray(targets, dtype=float)
    categories_array = np.asarray(categories, dtype="U32")
    flat_array = np.asarray(source_flat_indices, dtype=np.int64)
    depth_array = np.asarray(expected_depths, dtype=np.int64)
    source_array = np.asarray(known_sources, dtype=float)
    duration_array = np.asarray(historical_seconds, dtype=float)
    if targets_array.shape != (TARGET_COUNT, 4):
        raise RuntimeError("fixed Phase-8 workload has the wrong size")
    rng = np.random.default_rng(SELECTION_SEED)
    order = rng.permutation(TARGET_COUNT)
    return {
        "targets": targets_array[order],
        "categories": categories_array[order],
        "source_flat_indices": flat_array[order],
        "expected_subcell_depth": depth_array[order],
        "known_source_coordinates": source_array[order],
        "historical_phase6_seconds": duration_array[order],
        "selection_order": order.astype(np.int64),
    }


def prepare_selection(
    output: Path,
    artifact_path: Path,
    phase5_root: Path,
    phase6_report_path: Path,
) -> Tuple[Dict[str, np.ndarray], str]:
    selection_path = output / "selection.npz"
    manifest_path = output / "selection.json"
    inputs = {
        "artifact_sha256": _sha256(artifact_path),
        "phase5_selection_sha256": _sha256(phase5_root / "tier_b_selection.npz"),
        "phase5_report_sha256": _sha256(phase5_root / "phase5_report.json"),
        "phase6_report_sha256": _sha256(phase6_report_path),
    }
    if selection_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("inputs") != inputs:
            raise RuntimeError("saved Phase-8 selection belongs to different inputs")
        if manifest.get("selection_sha256") != _sha256(selection_path):
            raise RuntimeError("saved Phase-8 selection hash is inconsistent")
        with np.load(selection_path, allow_pickle=False) as archive:
            selection = {name: np.asarray(archive[name]) for name in archive.files}
    else:
        selection = _create_selection(
            artifact_path, phase5_root, phase6_report_path
        )
        _write_npz(selection_path, **selection)
        counts = Counter(str(item) for item in selection["categories"])
        if dict(counts) != CATEGORY_QUOTAS:
            raise RuntimeError("saved selection does not meet category quotas")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "selection_seed": SELECTION_SEED,
            "target_count": TARGET_COUNT,
            "category_counts": dict(counts),
            "inputs": inputs,
            "selection_sha256": _sha256(selection_path),
            "selection_rule": (
                "deterministic stratified real-HRG4D workload; saved Phase-6 "
                "subcell cases choose the lowest historical times and FAST "
                "roots choose the smallest complete candidate sets within "
                "each required mechanism to keep the full matrix bounded"
            ),
        }
        _write_json(manifest_path, manifest)
    required = {
        "targets",
        "categories",
        "source_flat_indices",
        "expected_subcell_depth",
        "known_source_coordinates",
        "historical_phase6_seconds",
        "selection_order",
    }
    if set(selection) != required or selection["targets"].shape != (TARGET_COUNT, 4):
        raise RuntimeError("saved Phase-8 selection schema is inconsistent")
    return selection, _sha256(selection_path)


@dataclass(frozen=True)
class _WorkerPayload:
    shared_state_manifest: str
    pipeline_configuration: Mapping[str, Any]
    reachability_configuration: Mapping[str, Any]


@dataclass(frozen=True)
class _TargetBatch:
    start: int
    stop: int
    targets: np.ndarray


@dataclass
class _TargetResult:
    start: int
    stop: int
    prescreen: np.ndarray
    status: np.ndarray
    proof: np.ndarray
    source_coordinates: np.ndarray
    scaled_residual_norm_inf: np.ndarray
    in_source_bounds: np.ndarray
    subcell_depth: np.ndarray
    residual_subcell_boxes: np.ndarray
    guarded_incomplete: np.ndarray
    task_seconds: float
    worker_started: float
    worker_finished: float
    worker_pid: int
    worker_peak_rss_bytes: int
    worker_setup_seconds: float


_WORKER_PIPELINE: Optional[EoSInversionPipeline] = None
_WORKER_SETUP_SECONDS = 0.0
_WORKER_SETUP_REPORTED = False


def _initialize_worker(payload: _WorkerPayload) -> None:
    global _WORKER_PIPELINE, _WORKER_SETUP_SECONDS, _WORKER_SETUP_REPORTED
    started = time.perf_counter()
    table = load_shared_table(payload.shared_state_manifest)
    config = PipelineConfig.from_dict(payload.pipeline_configuration)
    reachability = ReachabilityConfig(**dict(payload.reachability_configuration))
    pipeline = EoSInversionPipeline(
        table, config, reachability_config=reachability
    )
    pipeline._prepare(shared_state_manifest=payload.shared_state_manifest)
    _WORKER_PIPELINE = pipeline
    _WORKER_SETUP_SECONDS = float(time.perf_counter() - started)
    _WORKER_SETUP_REPORTED = False


def _run_target_batch(batch: _TargetBatch) -> _TargetResult:
    global _WORKER_SETUP_REPORTED
    pipeline = _WORKER_PIPELINE
    if pipeline is None or pipeline._reachability is None or pipeline._solver is None:
        raise RuntimeError("Phase-8 worker was not initialized")
    started = time.perf_counter()
    targets = np.asarray(batch.targets, dtype=float)
    count = batch.stop - batch.start
    if targets.shape != (count, pipeline.plan().grid.dimension):
        raise RuntimeError("Phase-8 target task has an invalid shape")
    localization = pipeline._reachability.localize(targets)
    localization.require_complete_for(pipeline._reachability.provenance)
    codes = np.asarray(localization.result.prescreen_codes).reshape(-1)
    excluded_code = pipeline_status_code(PrescreenStatus.DEFINITELY_UNREACHABLE)
    candidate_code = pipeline_status_code(PrescreenStatus.CANDIDATE)

    prescreen = np.empty(count, dtype="U32")
    status = np.empty(count, dtype="U32")
    proof = np.empty(count, dtype="U32")
    source = np.full((count, targets.shape[1]), np.nan, dtype=float)
    residual_norm = np.full(count, np.nan, dtype=float)
    in_bounds = np.zeros(count, dtype=bool)
    depth = np.full(count, -1, dtype=np.int64)
    residual_boxes = np.zeros(count, dtype=np.int64)
    guarded = np.zeros(count, dtype=bool)

    for local, target in enumerate(targets):
        code = int(codes[local])
        if code == excluded_code:
            prescreen[local] = PrescreenStatus.DEFINITELY_UNREACHABLE.value
            status[local] = "prescreen_excluded"
            proof[local] = "reachability_no_candidate"
            continue
        if code != candidate_code:
            raise RuntimeError("valid HRG4D target has no terminal prescreen code")
        prescreen[local] = PrescreenStatus.CANDIDATE.value
        # These targets are a stratified list, not neighbors on one regular
        # inverse grid.  Using their synthetic list positions for continuation
        # would let unrelated targets seed one another and would measure a
        # harness artifact.  The complete candidate-cell center is the stable
        # production seed shared by the preserved Phase-5/6 evidence.
        selection = pipeline._candidate_center_seed(
            target, localization, local
        )
        result = pipeline._solver.solve_with_candidate_cells(
            target,
            localization,
            local,
            selection.point,
            expected_provenance=pipeline._reachability.provenance,
            strategy="auto",
        )
        status[local] = result.status.value
        proof[local] = result.proof_status.value
        if result.solution is not None:
            source[local] = np.asarray(result.solution, dtype=float)
            residual_norm[local] = float(result.scaled_residual_norm)
            in_bounds[local] = bool(
                pipeline._target_interpolator.contains(result.solution)
            )
        fallback = result.fallback
        if fallback is not None:
            residual_boxes[local] = int(fallback.subcell_residual_box_count)
            report = fallback.subcell_report
            if report is not None:
                depth[local] = int(report.completed_depth)
                guarded[local] = not bool(report.complete)

    finished = time.perf_counter()
    setup_seconds = 0.0
    if not _WORKER_SETUP_REPORTED:
        setup_seconds = _WORKER_SETUP_SECONDS
        _WORKER_SETUP_REPORTED = True
    return _TargetResult(
        start=batch.start,
        stop=batch.stop,
        prescreen=prescreen,
        status=status,
        proof=proof,
        source_coordinates=source,
        scaled_residual_norm_inf=residual_norm,
        in_source_bounds=in_bounds,
        subcell_depth=depth,
        residual_subcell_boxes=residual_boxes,
        guarded_incomplete=guarded,
        task_seconds=float(finished - started),
        worker_started=float(started),
        worker_finished=float(finished),
        worker_pid=os.getpid(),
        worker_peak_rss_bytes=_peak_rss_bytes(),
        worker_setup_seconds=setup_seconds,
    )


def _combine_results(
    start: int, stop: int, results: Sequence[_TargetResult]
) -> _TargetResult:
    ordered = sorted(results, key=lambda item: (item.start, item.stop))
    if (
        not ordered
        or ordered[0].start != start
        or ordered[-1].stop != stop
        or any(left.stop != right.start for left, right in zip(ordered[:-1], ordered[1:]))
    ):
        raise RuntimeError("Phase-8 task results do not cover their range exactly")
    return _TargetResult(
        start=start,
        stop=stop,
        prescreen=np.concatenate([item.prescreen for item in ordered]),
        status=np.concatenate([item.status for item in ordered]),
        proof=np.concatenate([item.proof for item in ordered]),
        source_coordinates=np.concatenate(
            [item.source_coordinates for item in ordered], axis=0
        ),
        scaled_residual_norm_inf=np.concatenate(
            [item.scaled_residual_norm_inf for item in ordered]
        ),
        in_source_bounds=np.concatenate([item.in_source_bounds for item in ordered]),
        subcell_depth=np.concatenate([item.subcell_depth for item in ordered]),
        residual_subcell_boxes=np.concatenate(
            [item.residual_subcell_boxes for item in ordered]
        ),
        guarded_incomplete=np.concatenate(
            [item.guarded_incomplete for item in ordered]
        ),
        task_seconds=float(sum(item.task_seconds for item in ordered)),
        worker_started=float(min(item.worker_started for item in ordered)),
        worker_finished=float(max(item.worker_finished for item in ordered)),
        worker_pid=-1,
        worker_peak_rss_bytes=max(item.worker_peak_rss_bytes for item in ordered),
        worker_setup_seconds=float(
            sum(item.worker_setup_seconds for item in ordered)
        ),
    )


def _result_arrays(result: _TargetResult) -> Dict[str, np.ndarray]:
    return {
        "prescreen": np.asarray(result.prescreen),
        "status": np.asarray(result.status),
        "proof": np.asarray(result.proof),
        "source_coordinates": np.asarray(result.source_coordinates, dtype=float),
        "scaled_residual_norm_inf": np.asarray(
            result.scaled_residual_norm_inf, dtype=float
        ),
        "in_source_bounds": np.asarray(result.in_source_bounds, dtype=bool),
        "subcell_depth": np.asarray(result.subcell_depth, dtype=np.int64),
        "residual_subcell_boxes": np.asarray(
            result.residual_subcell_boxes, dtype=np.int64
        ),
        "guarded_incomplete": np.asarray(result.guarded_incomplete, dtype=bool),
    }


def _load_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def compare_result_arrays(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    residual_tolerance: float,
    coordinate_tolerance: float,
) -> Dict[str, Any]:
    exact = {
        name: bool(np.array_equal(reference[name], candidate[name]))
        for name in _STATUS_ARRAY_NAMES
    }
    solved = np.isin(
        np.asarray(reference["status"]),
        (SolveStatus.CONVERGED_FAST.value, SolveStatus.CONVERGED_FALLBACK.value),
    )
    candidate_solved = np.isin(
        np.asarray(candidate["status"]),
        (SolveStatus.CONVERGED_FAST.value, SolveStatus.CONVERGED_FALLBACK.value),
    )
    solved_mask_equal = bool(np.array_equal(solved, candidate_solved))
    span = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    reference_source = np.asarray(reference["source_coordinates"], dtype=float)
    candidate_source = np.asarray(candidate["source_coordinates"], dtype=float)
    if np.any(solved):
        reference_normalized = (
            reference_source[solved] - np.asarray(lower, dtype=float)
        ) / span[np.newaxis, :]
        candidate_normalized = (
            candidate_source[solved] - np.asarray(lower, dtype=float)
        ) / span[np.newaxis, :]
        coordinate_error = np.max(
            np.abs(candidate_normalized - reference_normalized), axis=1
        )
        maximum_coordinate_error = float(np.max(coordinate_error))
        coordinates_close = bool(
            maximum_coordinate_error <= coordinate_tolerance
            and
            np.allclose(
                candidate_normalized,
                reference_normalized,
                rtol=coordinate_tolerance,
                atol=coordinate_tolerance,
            )
        )
    else:
        maximum_coordinate_error = 0.0
        coordinates_close = True
    candidate_residual = np.asarray(
        candidate["scaled_residual_norm_inf"], dtype=float
    )
    roots_valid = bool(
        solved_mask_equal
        and np.all(np.asarray(candidate["in_source_bounds"])[candidate_solved])
        and np.all(candidate_residual[candidate_solved] <= residual_tolerance)
    )
    return {
        "exact_status_fields": exact,
        "all_status_proof_prescreen_exact": all(exact.values()),
        "solved_mask_equal": solved_mask_equal,
        "solved_count": int(np.count_nonzero(solved)),
        "accepted_roots_in_bounds_and_within_residual_tolerance": roots_valid,
        "coordinate_tolerance_full_span_normalized": coordinate_tolerance,
        "maximum_full_span_normalized_coordinate_difference": (
            maximum_coordinate_error
        ),
        "solved_coordinates_close": coordinates_close,
        "parity_passed": bool(
            all(exact.values()) and roots_valid and coordinates_close
        ),
    }


def select_measured_default(
    rows: Sequence[Mapping[str, Any]],
    task_sizes: Sequence[int] = TASK_SIZES,
    *,
    near_best_fraction: float = NEAR_BEST_FRACTION,
) -> Dict[str, Any]:
    aggregates = {}
    for task_size in task_sizes:
        selected = [
            row
            for row in rows
            if int(row["task_size"]) == int(task_size)
        ]
        if not selected:
            raise ValueError("every task size requires at least one benchmark row")
        aggregates[int(task_size)] = float(
            sum(float(row["scheduler_wall_seconds"]) for row in selected)
        )
    best_seconds = min(aggregates.values())
    eligible = [
        task_size
        for task_size, seconds in aggregates.items()
        if seconds <= best_seconds * (1.0 + near_best_fraction)
    ]
    chosen = min(eligible)
    return {
        "aggregate_scheduler_wall_seconds": {
            str(key): value for key, value in sorted(aggregates.items())
        },
        "best_aggregate_seconds": best_seconds,
        "near_best_fraction": near_best_fraction,
        "eligible_task_sizes": sorted(eligible),
        "selected_task_size": chosen,
        "rule": "smallest tested size within 10% of best aggregate wall time",
    }


def _payload(
    artifact: Any, shared_state: Path, workers: int, task_size: int
) -> Tuple[_WorkerPayload, np.ndarray, np.ndarray, float, float]:
    config = replace(
        artifact.plan.config,
        solver=replace(
            artifact.plan.config.solver,
            enable_subcell_fallback=True,
            max_subcell_depth=7,
            max_subcell_active_boxes=1_000_000,
        ),
        execution=ExecutionConfig(
            workers=max(1, workers),
            start_method="spawn",
            chunk_size=OUTER_CHUNK_SIZE,
            worker_task_size=task_size,
            max_pending_chunks=max(1, 2 * max(1, workers)),
            error_policy=artifact.plan.config.execution.error_policy,
        ),
    )
    reachability = ReachabilityConfig(method="hierarchical", hierarchy_levels=7)
    payload = _WorkerPayload(
        shared_state_manifest=str(shared_state.resolve()),
        pipeline_configuration=config.to_dict(),
        reachability_configuration={
            item.name: getattr(reachability, item.name)
            for item in dataclass_fields(reachability)
        },
    )
    table = load_shared_table(shared_state)
    lower = np.asarray([axis[0] for axis in table.axes], dtype=float)
    upper = np.asarray([axis[-1] for axis in table.axes], dtype=float)
    return (
        payload,
        lower,
        upper,
        float(config.solver.root_residual_tolerance),
        float(max(10.0 * config.solver.newton_step_tolerance, 1.0e-11)),
    )


def _serial_reference(
    payload: _WorkerPayload, targets: np.ndarray
) -> Tuple[_TargetResult, Dict[str, Any]]:
    _initialize_worker(payload)
    started = time.perf_counter()
    result = _run_target_batch(_TargetBatch(0, targets.shape[0], targets))
    wall = float(time.perf_counter() - started)
    return result, {
        "mode": "serial",
        "workers": 1,
        "task_size": targets.shape[0],
        "scheduler_wall_seconds": wall,
        "pool_lifetime_seconds": wall + result.worker_setup_seconds,
        "summed_worker_task_seconds": result.task_seconds,
        "worker_setup_seconds": result.worker_setup_seconds,
        "maximum_worker_peak_rss_bytes": result.worker_peak_rss_bytes,
        "parent_peak_rss_bytes": _peak_rss_bytes(),
    }


def _process_run(
    payload: _WorkerPayload,
    targets: np.ndarray,
    workers: int,
    task_size: int,
) -> Tuple[_TargetResult, Dict[str, Any]]:
    outer = tuple(
        _TargetBatch(start, min(targets.shape[0], start + OUTER_CHUNK_SIZE), targets[start : min(targets.shape[0], start + OUTER_CHUNK_SIZE)])
        for start in range(0, targets.shape[0], OUTER_CHUNK_SIZE)
    )
    published = []
    task_results = []

    def split(chunk: _TargetBatch) -> Tuple[_TargetBatch, ...]:
        return tuple(
            _TargetBatch(
                start,
                min(chunk.stop, start + task_size),
                chunk.targets[
                    start - chunk.start : min(chunk.stop, start + task_size)
                    - chunk.start
                ],
            )
            for start in range(chunk.start, chunk.stop, task_size)
        )

    def assemble(
        chunk: _TargetBatch, results: Sequence[_TargetResult]
    ) -> _TargetResult:
        task_results.extend(results)
        return _combine_results(chunk.start, chunk.stop, results)

    def publish(_chunk: _TargetBatch, result: _TargetResult) -> None:
        published.append(result)

    context = mp.get_context("spawn")
    pool_started = time.perf_counter()
    with _worker_thread_environment():
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(payload,),
        ) as executor:
            scheduler_started = time.perf_counter()
            schedule = execute_dynamic_tasks(
                executor,
                outer,
                split_chunk=split,
                run_task=_run_target_batch,
                assemble_chunk=assemble,
                publish_chunk=publish,
                maximum_pending_tasks=max(1, 2 * workers),
            )
            scheduler_wall = float(time.perf_counter() - scheduler_started)
    pool_lifetime = float(time.perf_counter() - pool_started)
    result = _combine_results(0, targets.shape[0], published)
    task_durations = [item.task_seconds for item in task_results]
    finishes = sorted(item.worker_finished for item in task_results)
    idle_tail = 0.0 if len(finishes) < 2 else float(finishes[-1] - finishes[-2])
    active_span = float(
        max(item.worker_finished for item in task_results)
        - min(item.worker_started for item in task_results)
    )
    pids = sorted({int(item.worker_pid) for item in task_results})
    return result, {
        "mode": "process",
        "workers": workers,
        "task_size": task_size,
        "outer_chunk_size": OUTER_CHUNK_SIZE,
        "outer_chunk_count": len(outer),
        "scheduler_wall_seconds": scheduler_wall,
        "pool_lifetime_seconds": pool_lifetime,
        "summed_worker_task_seconds": float(sum(task_durations)),
        "task_seconds": _distribution(task_durations),
        "dispatch_and_parent_overhead_seconds": max(0.0, scheduler_wall - active_span),
        "idle_tail_seconds": idle_tail,
        "worker_setup_seconds": float(
            sum(item.worker_setup_seconds for item in task_results)
        ),
        "worker_pids": pids,
        "workers_observed": len(pids),
        "maximum_worker_peak_rss_bytes": max(
            item.worker_peak_rss_bytes for item in task_results
        ),
        "parent_peak_rss_bytes": _peak_rss_bytes(),
        "submitted_tasks": schedule.submitted_task_count,
        "completed_tasks": schedule.completed_task_count,
        "published_outer_chunks": schedule.published_chunk_count,
        "maximum_pending_tasks": schedule.maximum_pending_tasks_observed,
        "maximum_active_outer_chunks": schedule.maximum_active_chunks_observed,
    }


def _validate_composition(
    selection: Mapping[str, np.ndarray], arrays: Mapping[str, np.ndarray]
) -> Dict[str, Any]:
    categories = np.asarray(selection["categories"])
    status = np.asarray(arrays["status"])
    prescreen = np.asarray(arrays["prescreen"])
    depth = np.asarray(arrays["subcell_depth"])
    residual = np.asarray(arrays["residual_subcell_boxes"])
    checks = {
        "prescreen_exclusions": bool(
            np.all(
                prescreen[categories == "prescreen_excluded"]
                == PrescreenStatus.DEFINITELY_UNREACHABLE.value
            )
        ),
        "fast_roots": bool(
            np.all(
                status[categories == "fast_root"]
                == SolveStatus.CONVERGED_FAST.value
            )
        ),
        "shallow_subcell_certificates": bool(
            np.all(np.isin(depth[categories == "subcell_shallow"], (1, 2)))
            and np.all(
                status[categories == "subcell_shallow"]
                == SolveStatus.CERTIFIED_NO_ROOT.value
            )
        ),
        "intermediate_subcell_certificates": bool(
            np.all(
                (depth[categories == "subcell_intermediate"] >= 3)
                & (depth[categories == "subcell_intermediate"] <= 5)
            )
        ),
        "deep_subcell_certificates": bool(
            np.all(depth[categories == "subcell_deep"] >= 6)
        ),
        "residual_krawczyk": bool(
            np.all(residual[categories == "residual_krawczyk"] > 0)
            and np.all(
                status[categories == "residual_krawczyk"]
                == SolveStatus.CERTIFIED_NO_ROOT.value
            )
        ),
    }
    return {
        "category_counts": dict(Counter(str(item) for item in categories)),
        "checks": checks,
        "all_required_mechanisms_observed": all(checks.values()),
        "observed_status_counts": dict(Counter(str(item) for item in status)),
        "observed_subcell_depth_counts": dict(
            Counter(str(int(item)) for item in depth if int(item) >= 0)
        ),
        "residual_krawczyk_target_count": int(np.count_nonzero(residual > 0)),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection, selection_sha256 = prepare_selection(
        output, args.artifact, args.phase5_root, args.phase6_report
    )
    artifact = load_pipeline_result(args.artifact)
    payload, lower, upper, residual_tolerance, coordinate_tolerance = _payload(
        artifact, args.shared_state, 1, TASK_SIZES[0]
    )
    pipeline_scientific_configuration = dict(payload.pipeline_configuration)
    pipeline_scientific_configuration.pop("execution", None)
    scientific_contract = {
        "pipeline": pipeline_scientific_configuration,
        "reachability": payload.reachability_configuration,
        "selection_sha256": selection_sha256,
    }
    configuration_fingerprint = _json_digest(scientific_contract)
    # Early completed rows included the fixed execution configuration and, in
    # the first version, the requested matrix axes in this fingerprint.  None
    # affects a worker's scientific result.  Accept only those exact legacy
    # contracts, verify each result hash, and migrate the row instead of
    # recomputing it.
    legacy_contract = {
        "pipeline": payload.pipeline_configuration,
        "reachability": payload.reachability_configuration,
        "selection_sha256": selection_sha256,
    }
    compatible_fingerprints = {
        configuration_fingerprint,
        _json_digest(legacy_contract),
        *(
            _json_digest(
                {
                    **legacy_contract,
                    "task_sizes": list(TASK_SIZES),
                    "worker_counts": list(worker_counts),
                }
            )
            for worker_counts in ((1, 2, 4), (1, 2, 4, 8))
        ),
    }
    serial_path = output / "serial_reference.npz"
    serial_row_path = output / "serial_reference.json"
    if serial_path.is_file() and serial_row_path.is_file():
        serial_row = json.loads(serial_row_path.read_text(encoding="utf-8"))
        if serial_row.get("configuration_fingerprint") not in compatible_fingerprints:
            raise RuntimeError("saved serial reference has an incompatible contract")
    else:
        serial_result, serial_row = _serial_reference(
            payload, np.asarray(selection["targets"], dtype=float)
        )
        _write_npz(serial_path, **_result_arrays(serial_result))
        serial_row.update(
            {
                "complete": True,
                "configuration_fingerprint": configuration_fingerprint,
                "result_sha256": _sha256(serial_path),
            }
        )
        _write_json(serial_row_path, serial_row)
    if serial_row.get("result_sha256") != _sha256(serial_path):
        raise RuntimeError("saved serial reference result hash is inconsistent")
    if serial_row.get("configuration_fingerprint") != configuration_fingerprint:
        serial_row["configuration_fingerprint"] = configuration_fingerprint
        _write_json(serial_row_path, serial_row)
    reference = _load_arrays(serial_path)
    composition = _validate_composition(selection, reference)

    rows = []
    parity = []
    for task_size in args.task_sizes:
        for workers in args.worker_counts:
            stem = "workers_%d_task_%d" % (workers, task_size)
            result_path = output / (stem + ".npz")
            row_path = output / (stem + ".json")
            if result_path.is_file() and row_path.is_file():
                row = json.loads(row_path.read_text(encoding="utf-8"))
                if row.get("configuration_fingerprint") not in compatible_fingerprints:
                    raise RuntimeError("saved matrix row has an incompatible contract")
                if row.get("result_sha256") != _sha256(result_path):
                    raise RuntimeError("saved matrix row hash is inconsistent")
                if row.get("configuration_fingerprint") != configuration_fingerprint:
                    row["configuration_fingerprint"] = configuration_fingerprint
                    _write_json(row_path, row)
            else:
                result, row = _process_run(
                    payload,
                    np.asarray(selection["targets"], dtype=float),
                    workers,
                    task_size,
                )
                _write_npz(result_path, **_result_arrays(result))
                row.update(
                    {
                        "complete": True,
                        "configuration_fingerprint": configuration_fingerprint,
                        "result_sha256": _sha256(result_path),
                    }
                )
                _write_json(row_path, row)
            arrays = _load_arrays(result_path)
            comparison = compare_result_arrays(
                reference,
                arrays,
                lower=lower,
                upper=upper,
                residual_tolerance=residual_tolerance,
                coordinate_tolerance=coordinate_tolerance,
            )
            row = dict(row)
            row["parity"] = comparison
            rows.append(row)
            parity.append(comparison)
            print(
                "Phase 8 workers=%d task=%d wall=%.6f s parity=%s"
                % (
                    workers,
                    task_size,
                    row["scheduler_wall_seconds"],
                    comparison["parity_passed"],
                ),
                flush=True,
            )

    del artifact
    default = select_measured_default(rows, args.task_sizes)
    selected_size = int(default["selected_task_size"])
    by_worker = {
        int(row["workers"]): row
        for row in rows
        if int(row["task_size"]) == selected_size
    }
    one_to_four = float(
        by_worker[1]["scheduler_wall_seconds"]
        / by_worker[4]["scheduler_wall_seconds"]
    )
    one_to_eight = (
        None
        if 8 not in by_worker
        else float(
            by_worker[1]["scheduler_wall_seconds"]
            / by_worker[8]["scheduler_wall_seconds"]
        )
    )
    acceptance = {
        "all_required_real_hrg4d_mechanisms_observed": composition[
            "all_required_mechanisms_observed"
        ],
        "all_matrix_rows_scientifically_equivalent_to_serial": all(
            item["parity_passed"] for item in parity
        ),
        "serial_two_and_four_process_parity": all(
            row["parity"]["parity_passed"]
            for row in rows
            if int(row["workers"]) in (2, 4)
        ),
        "useful_four_worker_scaling": one_to_four >= MINIMUM_FOUR_WORKER_SPEEDUP,
        "measured_default_selected": selected_size in args.task_sizes,
    }
    acceptance["gate_satisfied"] = all(acceptance.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "scope": "V1 Phase 8 fixed mixed HRG4D scheduling benchmark",
        "selection": {
            "path": str((output / "selection.npz").resolve()),
            "sha256": selection_sha256,
            "target_count": TARGET_COUNT,
            "composition": composition,
        },
        "configuration_fingerprint": configuration_fingerprint,
        "task_sizes": list(args.task_sizes),
        "worker_counts": list(args.worker_counts),
        "serial_reference": serial_row,
        "matrix": rows,
        "default_selection": default,
        "scaling": {
            "selected_task_size": selected_size,
            "one_to_four_worker_speedup": one_to_four,
            "one_to_eight_worker_speedup": one_to_eight,
            "minimum_accepted_four_worker_speedup": (
                MINIMUM_FOUR_WORKER_SPEEDUP
            ),
        },
        "scientific_tolerances": {
            "root_scaled_residual_linf": residual_tolerance,
            "coordinate_full_span_normalized": coordinate_tolerance,
        },
        "acceptance": acceptance,
    }
    _write_json(output / "phase8_report.json", report)
    return report


def _parse_positive_csv(value: str, name: str) -> Tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("%s must be comma-separated integers" % name) from exc
    if not parsed or any(item <= 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("%s must contain unique positive integers" % name)
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--shared-state", type=Path, default=DEFAULT_SHARED_STATE)
    parser.add_argument("--phase5-root", type=Path, default=DEFAULT_PHASE5_ROOT)
    parser.add_argument("--phase6-report", type=Path, default=DEFAULT_PHASE6_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-sizes", default="8,16,32")
    parser.add_argument("--worker-counts", default="1,2,4,8")
    args = parser.parse_args(argv)
    args.task_sizes = _parse_positive_csv(args.task_sizes, "task sizes")
    args.worker_counts = _parse_positive_csv(args.worker_counts, "worker counts")
    if set(args.task_sizes) != set(TASK_SIZES):
        parser.error("the accepted Phase-8 matrix requires task sizes 8,16,32")
    if not {1, 2, 4}.issubset(args.worker_counts):
        parser.error("the accepted Phase-8 matrix requires workers 1,2,4")
    report = run(args)
    print(args.output / "phase8_report.json")
    return 0 if report["acceptance"]["gate_satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
