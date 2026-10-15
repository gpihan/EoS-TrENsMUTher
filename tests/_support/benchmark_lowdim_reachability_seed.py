#!/usr/bin/env python3
"""Benchmark complete localization and candidate-center seeds in 1D--3D.

The repository currently contains no ordinary (non-first-order) real tables in
all three lower dimensions.  This driver therefore creates deterministic,
validated, smoothly coupled synthetic tables and exercises the production
pipeline unchanged.  It compares ``UseReachabilitySeed`` on/off, measures the
cost of collecting complete routing-cell CSR rows, and separately profiles
FAST from candidate-cell-center and coarse-catalog seeds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from eos_trensmuther import (
    CoarseSeedConfig,
    EoSInversionPipeline,
    ExecutionConfig,
    ForwardTable,
    GridAxisSpec,
    InversionGridSpec,
    PipelineConfig,
    ProjectionConfig,
    ReachabilityConfig,
    SolveStrategy,
)


SOURCE_NAMES = ("T", "muB", "muQ")
TARGET_NAMES = ("e", "nB", "nQ")
DEFAULT_SOURCE_POINTS = {1: 33, 2: 17, 3: 9}
DEFAULT_TARGET_POINTS = {1: 301, 2: 49, 3: 15}
QUICK_SOURCE_POINTS = {1: 9, 2: 6, 3: 4}
QUICK_TARGET_POINTS = {1: 31, 2: 9, 3: 5}


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


def _field(role: str, column: int, unit: str) -> Dict[str, Any]:
    return {
        "role": role,
        "column": column,
        "source_unit": unit,
        "canonical_unit": unit,
        "transform": {
            "scale_factor": 1.0,
            "temperature_power": 0,
            "hbarc_power": 0,
        },
        "nonfinite_policy": "forbid",
        "output": True,
        "interpolation": "linear",
    }


def _metadata(dimension: int, column_count: int) -> Dict[str, Any]:
    source_names = SOURCE_NAMES[:dimension]
    target_names = TARGET_NAMES[:dimension]
    fields = {
        name: _field("coordinate", index, "GeV")
        for index, name in enumerate(source_names)
    }
    fields.update(
        {
            name: _field(
                "target",
                dimension + index,
                "GeV^4" if name == "e" else "GeV^3",
            )
            for index, name in enumerate(target_names)
        }
    )
    return {
        "schema_version": 1,
        "name": "Phase 11 smooth ordinary synthetic %dD" % dimension,
        "table_revision": "phase11-v1",
        "description": (
            "Deterministic benchmark fixture for lower-dimensional complete "
            "localization and candidate-center seeding."
        ),
        "dimension": dimension,
        "data_file": "table.dat",
        "format": {
            "delimiter": "whitespace",
            "comment_prefix": "#",
            "skip_rows": 0,
            "encoding": "utf-8",
            "expected_columns": column_count,
            "missing_values": [],
        },
        "fields": fields,
        "grid": {
            "kind": "rectilinear",
            "require_complete": True,
            "require_uniform": True,
            "reorder": True,
            "duplicate_policy": "error",
            "absolute_tolerance": 1.0e-12,
            "relative_tolerance": 1.0e-10,
            "min_axis_points": 2,
            "expected_bounds": {},
            "expected_spacing": {},
        },
        "scientific": {
            "origin": {"generator": "Phase 11 benchmark script"},
            "publications": [],
            "phase_structure": {
                "status": "none",
                "critical_points": [],
                "first_order_transition": None,
            },
        },
    }


def _forward_values(point: Sequence[float]) -> np.ndarray:
    """Smooth, weakly coupled and one-to-one benchmark map."""

    source = np.asarray(point, dtype=float)
    temperature = float(source[0])
    chemical = source[1:]
    energy = temperature + 0.35 * temperature**2
    if chemical.size:
        energy += 0.012 * float(np.dot(chemical, chemical))
    targets = [energy]
    for index, value in enumerate(chemical):
        others = float(np.sum(chemical) - value)
        density = (
            value * (0.65 + temperature)
            + 0.025 * value**3
            + 0.01 * others
        )
        targets.append(density)
    return np.asarray(targets, dtype=float)


class _SyntheticTable:
    def __init__(self, dimension: int, points_per_axis: int) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix="eosinverter-phase11-%dd-" % dimension
        )
        directory = Path(self._temporary.name)
        source_axes = [np.linspace(0.12, 0.36, points_per_axis)]
        source_axes.extend(
            np.linspace(-0.5, 0.5, points_per_axis)
            for _ in range(dimension - 1)
        )
        rows = []
        for point in itertools.product(*source_axes):
            rows.append(list(point) + _forward_values(point).tolist())
        metadata = _metadata(dimension, 2 * dimension)
        self.metadata_path = directory / "metadata.json"
        self.data_path = directory / "table.dat"
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.savetxt(self.data_path, np.asarray(rows, dtype=float), fmt="%.17g")

    def close(self) -> None:
        self._temporary.cleanup()


@dataclass
class _FastMetrics:
    calls: int = 0
    converged: int = 0
    localization_function_evaluations: int = 0
    localization_jacobian_evaluations: int = 0
    newton_iterations: int = 0
    newton_function_evaluations: int = 0
    newton_jacobian_evaluations: int = 0
    wall_seconds: float = 0.0
    scaled_residual_norms: list[float] = field(default_factory=list)

    def record(self, result: Any, elapsed: float) -> None:
        self.calls += 1
        self.wall_seconds += float(elapsed)
        self.converged += int(bool(result.converged))
        if result.localization is not None:
            self.localization_function_evaluations += int(
                result.localization.function_evaluations
            )
            self.localization_jacobian_evaluations += int(
                result.localization.jacobian_evaluations or 0
            )
        if result.newton is not None:
            self.newton_iterations += int(result.newton.iterations)
            self.newton_function_evaluations += int(
                result.newton.function_evaluations
            )
            self.newton_jacobian_evaluations += int(
                result.newton.jacobian_evaluations
            )
        norm = result.scaled_residual_norm
        if norm is not None and math.isfinite(norm):
            self.scaled_residual_norms.append(float(norm))

    def to_dict(self) -> Dict[str, Any]:
        calls = max(self.calls, 1)
        return {
            "calls": self.calls,
            "converged": self.converged,
            "success_fraction": self.converged / calls,
            "wall_seconds": self.wall_seconds,
            "localization_function_evaluations": (
                self.localization_function_evaluations
            ),
            "localization_jacobian_evaluations": (
                self.localization_jacobian_evaluations
            ),
            "newton_iterations": self.newton_iterations,
            "newton_function_evaluations": self.newton_function_evaluations,
            "newton_jacobian_evaluations": self.newton_jacobian_evaluations,
            "mean_localization_function_evaluations": (
                self.localization_function_evaluations / calls
            ),
            "mean_newton_iterations": self.newton_iterations / calls,
            "mean_total_function_evaluations": (
                self.localization_function_evaluations
                + self.newton_function_evaluations
            )
            / calls,
            "maximum_scaled_residual_norm": (
                max(self.scaled_residual_norms)
                if self.scaled_residual_norms
                else None
            ),
        }


def _pipeline_config(
    table: ForwardTable,
    target_points: int,
    use_reachability_seed: bool,
) -> PipelineConfig:
    envelope = table.find_boundaries(space="target", require_non_degenerate=True)
    axes = {
        name: GridAxisSpec(target_points)
        for name in envelope.names
    }
    scales = {
        name: float(upper - lower)
        for name, lower, upper in zip(
            envelope.names, envelope.lower, envelope.upper
        )
    }
    return PipelineConfig(
        grid=InversionGridSpec("target", axes),
        residual_scales=scales,
        projection=ProjectionConfig(enabled=False),
        execution=ExecutionConfig(
            workers=1,
            chunk_size=math.prod(spec.count for spec in axes.values()),
        ),
        coarse_seed=CoarseSeedConfig(
            enabled=True,
            points_per_axis=3,
            maximum_points=27,
        ),
        use_reachability_seed=use_reachability_seed,
        auxiliary_fields=(),
    )


def _all_targets(pipeline: EoSInversionPipeline) -> np.ndarray:
    chunks = tuple(pipeline.plan().iter_chunks())
    if not chunks:
        return np.empty((0, pipeline.plan().grid.dimension), dtype=float)
    return np.concatenate(
        [np.asarray(chunk.canonical_targets, dtype=float) for chunk in chunks],
        axis=0,
    )


def _duration_summary(samples: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in samples]
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def _multiplicity_summary(counts: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(counts, dtype=np.int64)
    candidate = values[values > 0]
    if not candidate.size:
        return {
            "candidate_targets": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "maximum": None,
            "histogram": {},
        }
    histogram = {
        "1": int(np.count_nonzero(candidate == 1)),
        "2-4": int(np.count_nonzero((candidate >= 2) & (candidate <= 4))),
        "5-10": int(np.count_nonzero((candidate >= 5) & (candidate <= 10))),
        "11-100": int(np.count_nonzero((candidate >= 11) & (candidate <= 100))),
        ">100": int(np.count_nonzero(candidate > 100)),
    }
    return {
        "candidate_targets": int(candidate.size),
        "minimum": int(np.min(candidate)),
        "median": float(np.median(candidate)),
        "mean": float(np.mean(candidate)),
        "p90": float(np.percentile(candidate, 90.0)),
        "p95": float(np.percentile(candidate, 95.0)),
        "p99": float(np.percentile(candidate, 99.0)),
        "maximum": int(np.max(candidate)),
        "histogram": histogram,
    }


def _instrument_pipeline_fast(pipeline: EoSInversionPipeline) -> _FastMetrics:
    metrics = _FastMetrics()
    solver = pipeline._solver
    if solver is None:
        raise RuntimeError("pipeline must be prepared before instrumentation")
    original = solver.solve

    def measured(target: Any, initial_guess: Any, *, strategy: Any = SolveStrategy.AUTO):
        started = time.perf_counter()
        result = original(target, initial_guess, strategy=strategy)
        elapsed = time.perf_counter() - started
        if strategy is SolveStrategy.FAST:
            metrics.record(result, elapsed)
        return result

    solver.solve = measured
    return metrics


def _pipeline_case(
    table: ForwardTable,
    target_points: int,
    enabled: bool,
) -> tuple[Dict[str, Any], Any]:
    pipeline = EoSInversionPipeline(
        table,
        _pipeline_config(table, target_points, enabled),
        reachability_config=ReachabilityConfig(method="exhaustive"),
    )
    setup_started = time.perf_counter()
    pipeline._prepare()
    explicit_setup_seconds = float(time.perf_counter() - setup_started)
    fast_metrics = _instrument_pipeline_fast(pipeline)
    started = time.perf_counter()
    result = pipeline.run()
    run_seconds = float(time.perf_counter() - started)
    run = result.manifest["run"]
    report = {
        "use_reachability_seed": enabled,
        "explicit_setup_seconds": explicit_setup_seconds,
        "run_seconds_after_setup": run_seconds,
        "total_seconds": explicit_setup_seconds + run_seconds,
        "worker_stage_seconds": _plain(
            run["timing_seconds"]["worker_stage_sum"]
        ),
        "setup_stage_seconds": _plain(
            run["timing_seconds"]["serial_setup_stages"]
        ),
        "stage_counters": _plain(run["stage_counters"]),
        "status_counts": _plain(run["status_counts"]),
        "initial_guess_source_counts": _plain(
            run["status_summary"]["initial_guess_source"]
        ),
        "fast_diagnostics": fast_metrics.to_dict(),
        "observed_peak_rss_bytes": _plain(
            run["performance"]["observed_peak_rss_bytes"]
        ),
    }
    return report, result


def _compare_results(left: Any, right: Any) -> Dict[str, Any]:
    code_fields = (
        "disposition_codes",
        "route_codes",
        "prescreen_codes",
        "solve_status_codes",
        "fast_status_codes",
        "proof_status_codes",
    )
    codes_equal = all(
        np.array_equal(getattr(left, name), getattr(right, name))
        for name in code_fields
    )
    valid = np.asarray(left.solution_mask) & np.asarray(right.solution_mask)
    source_close = bool(
        np.allclose(
            np.asarray(left.source_coordinates)[valid],
            np.asarray(right.source_coordinates)[valid],
            rtol=2.0e-9,
            atol=2.0e-11,
        )
    )
    return {
        "terminal_codes_equal": codes_equal,
        "solutions_compared": int(np.count_nonzero(valid)),
        "source_coordinates_close": source_close,
    }


def _localization_profile(
    pipeline: EoSInversionPipeline,
    repeats: int,
    maximum_seed_samples: int,
) -> Dict[str, Any]:
    targets = _all_targets(pipeline)
    classify_seconds = []
    localize_seconds = []
    localization = None
    for repeat in range(repeats):
        operations = ("classify", "localize")
        if repeat % 2:
            operations = tuple(reversed(operations))
        for operation in operations:
            started = time.perf_counter()
            if operation == "classify":
                pipeline._reachability.classify(targets)
                classify_seconds.append(float(time.perf_counter() - started))
            else:
                localization = pipeline._reachability.localize(targets)
                localize_seconds.append(float(time.perf_counter() - started))
    assert localization is not None
    candidate_indices = np.flatnonzero(
        localization.result.candidate_mask.reshape(-1)
    )
    if candidate_indices.size > maximum_seed_samples:
        selection = np.unique(
            np.rint(
                np.linspace(
                    0, candidate_indices.size - 1, maximum_seed_samples
                )
            ).astype(np.int64)
        )
        candidate_indices = candidate_indices[selection]

    center_metrics = _FastMetrics()
    coarse_metrics = _FastMetrics()
    scoring_seconds = 0.0
    scoring_wall_started = time.perf_counter()
    for local in candidate_indices:
        target = targets[int(local)]
        center = pipeline._candidate_center_seed(
            target, localization, int(local)
        )
        scoring_seconds += center.scoring_seconds
        started = time.perf_counter()
        center_result = pipeline._solver.solve(
            target, center.point, strategy=SolveStrategy.FAST
        )
        center_metrics.record(center_result, time.perf_counter() - started)

        coarse = pipeline._coarse_seeds.nearest(int(local))
        if coarse is None:
            coarse = pipeline._generic_initial_guess()
        started = time.perf_counter()
        coarse_result = pipeline._solver.solve(
            target, coarse, strategy=SolveStrategy.FAST
        )
        coarse_metrics.record(coarse_result, time.perf_counter() - started)
    scoring_outer_seconds = float(time.perf_counter() - scoring_wall_started)

    return {
        "target_count": int(targets.shape[0]),
        "boolean_classification_seconds": _duration_summary(classify_seconds),
        "complete_localization_seconds": _duration_summary(localize_seconds),
        "complete_over_boolean_median_ratio": (
            statistics.median(localize_seconds)
            / statistics.median(classify_seconds)
        ),
        "csr_pairs": localization.pair_count,
        "csr_bytes": localization.csr_nbytes,
        "candidate_multiplicity": _multiplicity_summary(
            localization.candidate_counts
        ),
        "exact_candidate_targets": int(
            np.count_nonzero(localization.result.exact_candidate_mask)
        ),
        "routing_shell_only_targets": int(
            np.count_nonzero(localization.result.tolerance_shell_mask)
        ),
        "isolated_seed_sample_count": int(candidate_indices.size),
        "candidate_center_scoring_internal_seconds": scoring_seconds,
        "isolated_comparison_outer_seconds": scoring_outer_seconds,
        "candidate_center_fast": center_metrics.to_dict(),
        "coarse_or_generic_fast": coarse_metrics.to_dict(),
    }


def _aggregate_pipeline_cases(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "repeats": len(cases),
        "explicit_setup_seconds": _duration_summary(
            item["explicit_setup_seconds"] for item in cases
        ),
        "run_seconds_after_setup": _duration_summary(
            item["run_seconds_after_setup"] for item in cases
        ),
        "total_seconds": _duration_summary(item["total_seconds"] for item in cases),
        "fallback_attempts": [
            int(item["stage_counters"]["fallback_attempted"])
            for item in cases
        ],
        "fast_attempts": [
            int(item["stage_counters"]["fast_attempted"])
            for item in cases
        ],
        "cases": list(cases),
    }


def _benchmark_dimension(
    dimension: int,
    source_points: int,
    target_points: int,
    repeats: int,
    localization_repeats: int,
    maximum_seed_samples: int,
) -> Dict[str, Any]:
    fixture = _SyntheticTable(dimension, source_points)
    try:
        table = ForwardTable.load_validated(fixture.metadata_path)
        profile_pipeline = EoSInversionPipeline(
            table,
            _pipeline_config(table, target_points, True),
            reachability_config=ReachabilityConfig(method="exhaustive"),
        )
        profile_pipeline._prepare()
        localization = _localization_profile(
            profile_pipeline, localization_repeats, maximum_seed_samples
        )

        cases: Dict[bool, list[Dict[str, Any]]] = {True: [], False: []}
        parity = []
        for repeat in range(repeats):
            order = (True, False) if repeat % 2 == 0 else (False, True)
            results = {}
            for enabled in order:
                case, result = _pipeline_case(table, target_points, enabled)
                case["repeat"] = repeat
                cases[enabled].append(case)
                results[enabled] = result
            parity.append(_compare_results(results[True], results[False]))

        enabled_aggregate = _aggregate_pipeline_cases(cases[True])
        disabled_aggregate = _aggregate_pipeline_cases(cases[False])
        enabled_median = enabled_aggregate["total_seconds"]["median"]
        disabled_median = disabled_aggregate["total_seconds"]["median"]
        parity_passed = all(
            item["terminal_codes_equal"]
            and item["source_coordinates_close"]
            for item in parity
        )
        # Timing runs this small are noisy.  Candidate centers remain the
        # default only when parity holds and their median total time is within
        # five percent of the alternative (or faster).  This makes total run
        # time, not localization pair count, the deciding metric.
        recommend_enabled = bool(
            parity_passed and enabled_median <= 1.05 * disabled_median
        )
        return {
            "dimension": dimension,
            "source_shape": [source_points] * dimension,
            "source_nodes": source_points**dimension,
            "source_cells": (source_points - 1) ** dimension,
            "target_shape": [target_points] * dimension,
            "target_count": target_points**dimension,
            "table_snapshot_fingerprint": table.snapshot_fingerprint(),
            "localization": localization,
            "pipeline": {
                "enabled": enabled_aggregate,
                "disabled": disabled_aggregate,
                "parity": parity,
                "enabled_over_disabled_median_total_ratio": (
                    enabled_median / disabled_median
                ),
            },
            "decision": {
                "use_reachability_seed_default": recommend_enabled,
                "primary_metric": "median_total_pipeline_seconds_including_setup",
                "maximum_accepted_slowdown_fraction": 0.05,
                "parity_required": True,
                "parity_passed": parity_passed,
            },
        }
    finally:
        fixture.close()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 11 low-dimensional reachability-seed benchmark",
        "",
        "The benchmark uses validated, ordinary smooth synthetic tables because "
        "the repository does not contain non-first-order real tables in every "
        "dimension 1--3. The production exhaustive localization, solver, seed "
        "selection, and restricted FALLBACK paths are exercised unchanged.",
        "",
        "Total pipeline time (setup + inversion) is the primary default-selection "
        "metric. Numerical/status parity is mandatory.",
        "",
        "| Dim | Source cells | Targets | CSR pairs | median multiplicity | "
        "complete/boolean | enabled total (s) | disabled total (s) | ratio | "
        "FALLBACK on/off | default |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in report["dimensions"]:
        localization = item["localization"]
        pipeline = item["pipeline"]
        enabled = pipeline["enabled"]
        disabled = pipeline["disabled"]
        multiplicity = localization["candidate_multiplicity"]["median"]
        lines.append(
            "| {dimension} | {cells} | {targets} | {pairs} | {multiplicity} | "
            "{collection:.3f} | {enabled:.6f} | {disabled:.6f} | {ratio:.3f} | "
            "{fallback_on}/{fallback_off} | {default} |".format(
                dimension=item["dimension"],
                cells=item["source_cells"],
                targets=item["target_count"],
                pairs=localization["csr_pairs"],
                multiplicity=multiplicity,
                collection=localization["complete_over_boolean_median_ratio"],
                enabled=enabled["total_seconds"]["median"],
                disabled=disabled["total_seconds"]["median"],
                ratio=pipeline["enabled_over_disabled_median_total_ratio"],
                fallback_on=enabled["fallback_attempts"][0],
                fallback_off=disabled["fallback_attempts"][0],
                default=(
                    "enabled"
                    if item["decision"]["use_reachability_seed_default"]
                    else "disabled"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Isolated FAST seed comparison",
            "",
            "| Dim | Samples | center success | coarse success | center mean "
            "function evals | coarse mean function evals | center scoring (s) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["dimensions"]:
        localization = item["localization"]
        center = localization["candidate_center_fast"]
        coarse = localization["coarse_or_generic_fast"]
        lines.append(
            "| {dimension} | {samples} | {center_success:.3f} | "
            "{coarse_success:.3f} | {center_evals:.3f} | {coarse_evals:.3f} | "
            "{scoring:.6f} |".format(
                dimension=item["dimension"],
                samples=localization["isolated_seed_sample_count"],
                center_success=center["success_fraction"],
                coarse_success=coarse["success_fraction"],
                center_evals=center["mean_total_function_evaluations"],
                coarse_evals=coarse["mean_total_function_evaluations"],
                scoring=localization[
                    "candidate_center_scoring_internal_seconds"
                ],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["conclusion"],
            "",
            "The hierarchy remains 4D-only. Dimensions 1--3 use exhaustive "
            "complete localization. First-order LINE/BINODALS tables were not "
            "included and remain on their separately validated path.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--localization-repeats", type=int, default=5)
    parser.add_argument("--maximum-seed-samples", type=int, default=512)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use small fixtures suitable for a smoke test",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("repeats", "localization_repeats", "maximum_seed_samples"):
        if getattr(args, name) < 1:
            raise SystemExit("--%s must be positive" % name.replace("_", "-"))
    source_points = QUICK_SOURCE_POINTS if args.quick else DEFAULT_SOURCE_POINTS
    target_points = QUICK_TARGET_POINTS if args.quick else DEFAULT_TARGET_POINTS
    started = time.perf_counter()
    dimensions = [
        _benchmark_dimension(
            dimension,
            source_points[dimension],
            target_points[dimension],
            args.repeats,
            args.localization_repeats,
            args.maximum_seed_samples,
        )
        for dimension in (1, 2, 3)
    ]
    enabled_dimensions = [
        item["dimension"]
        for item in dimensions
        if item["decision"]["use_reachability_seed_default"]
    ]
    if enabled_dimensions == [1, 2, 3]:
        conclusion = (
            "All dimensions passed parity and remained within the declared 5% "
            "median total-time guard. Keep candidate-cell-center seeding enabled "
            "by default for ordinary dimensions 1--3."
        )
    else:
        conclusion = (
            "Do not use one unconditional lower-dimensional default: the "
            "measured dimensions recommended for enablement were %s. Review "
            "dimension-aware policy before changing production defaults."
            % enabled_dimensions
        )
    report = {
        "benchmark_schema_version": 1,
        "benchmark": "phase11_lowdim_reachability_seed",
        "quick": bool(args.quick),
        "repeats": args.repeats,
        "localization_repeats": args.localization_repeats,
        "maximum_seed_samples": args.maximum_seed_samples,
        "total_benchmark_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
        },
        "source_map": (
            "e=T+0.35*T^2+0.012*sum(mu_i^2); "
            "n_i=mu_i*(0.65+T)+0.025*mu_i^3+0.01*sum(other_mu)"
        ),
        "first_order_tables_excluded": True,
        "hierarchy_scope_unchanged": "4D only",
        "decision_rule": {
            "primary_metric": "median total pipeline time including setup",
            "parity_required": True,
            "maximum_enabled_slowdown_fraction": 0.05,
        },
        "dimensions": dimensions,
        "conclusion": conclusion,
    }
    output = args.output_directory.resolve()
    _write_atomic(
        output / "benchmark.json",
        (json.dumps(_plain(report), indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    _write_atomic(
        output / "BENCHMARK.md", _markdown(report).encode("utf-8")
    )
    print(output / "benchmark.json")
    print(output / "BENCHMARK.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
