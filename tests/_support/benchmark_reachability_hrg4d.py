#!/usr/bin/env python3
"""Guarded, solver-free benchmark for hierarchical 4D reachability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

from eos_trensmuther import (
    ForwardTable,
    GridAxisSpec,
    InversionGrid,
    InversionGridSpec,
    MultilinearInterpolator,
    ReachabilityConfig,
    ReachabilityIndex,
)
from eos_trensmuther.transforms import ConstrainedTildeTransform
from eos_trensmuther.reachability_hierarchy import (
    build_routing_hierarchy,
    classify_hierarchy_leaves,
    profile_hierarchy_pruning,
)
from eos_trensmuther.reachability import _scan_target_boxes


def _positive_integer(value: str) -> int:
    converted = int(value)
    if converted <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return converted


def _nonnegative_float(value: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return converted


def _nonnegative_integer(value: str) -> int:
    converted = int(value)
    if converted < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark only 4D reachability hierarchy construction, routing, "
            "and optionally exact leaves. No root solver is invoked."
        )
    )
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hierarchy-levels", type=int, choices=(3, 5, 7), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--target-mode", choices=("representative", "grid"), required=True
    )
    parser.add_argument("--grid-space", choices=("target", "tilde"), default="tilde")
    parser.add_argument("--grid-count", type=_positive_integer)
    parser.add_argument("--target-chunk-size", type=_positive_integer, default=8192)
    parser.add_argument("--known-reachable", type=_positive_integer, default=64)
    parser.add_argument("--obvious-unreachable", type=_positive_integer, default=64)
    parser.add_argument("--boundary-targets", type=_positive_integer, default=64)
    parser.add_argument("--random-envelope", type=_positive_integer, default=64)
    parser.add_argument(
        "--stratified-grid-samples", type=_nonnegative_integer, default=0
    )
    parser.add_argument(
        "--stratified-grid-count", type=_positive_integer, default=30
    )
    parser.add_argument(
        "--zero-charge-diagonal", type=_nonnegative_integer, default=0
    )
    parser.add_argument(
        "--root-residual-tolerance",
        type=_nonnegative_float,
        required=True,
    )
    parser.add_argument("--query-batch-size", type=_positive_integer, default=256)
    parser.add_argument(
        "--maximum-pairwise-entries", type=_positive_integer, default=1_048_576
    )
    parser.add_argument(
        "--maximum-hierarchy-bytes", type=_positive_integer, required=True
    )
    parser.add_argument(
        "--maximum-benchmark-memory-bytes", type=_positive_integer, required=True
    )
    parser.add_argument(
        "--maximum-estimated-leaf-pairs", type=_positive_integer, required=True
    )
    parser.add_argument(
        "--maximum-estimated-leaf-seconds",
        type=_nonnegative_float,
        required=True,
    )
    parser.add_argument(
        "--maximum-benchmark-wall-seconds",
        type=_nonnegative_float,
        required=True,
    )
    parser.add_argument(
        "--throughput-targets", type=_positive_integer, default=8
    )
    parser.add_argument(
        "--throughput-leaves", type=_positive_integer, default=65536
    )
    parser.add_argument(
        "--complete-leaves",
        action="store_true",
        help="opt in to exact leaf completion when every configured gate passes",
    )
    parser.add_argument(
        "--oracle-equivalence-sample",
        type=_nonnegative_integer,
        default=0,
        help=(
            "when leaves complete, compare this many stratified targets with "
            "the unchanged exhaustive oracle"
        ),
    )
    return parser


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _representative_targets(table, interpolator, boxes, arguments):
    random = np.random.RandomState(arguments.seed)
    source = np.column_stack(
        [
            random.uniform(axis[0], axis[-1], size=arguments.known_reachable)
            for axis in interpolator.axes
        ]
    )
    known = np.asarray(interpolator(source), dtype=float)

    target_lower = np.min(boxes.exact_lower, axis=0)
    target_upper = np.max(boxes.exact_upper, axis=0)
    span = np.maximum(target_upper - target_lower, np.finfo(float).tiny)
    outside = np.tile(0.5 * (target_lower + target_upper), (arguments.obvious_unreachable, 1))
    for index in range(arguments.obvious_unreachable):
        axis = index % 4
        direction = -1.0 if (index // 4) % 2 == 0 else 1.0
        outside[index, axis] = (
            target_lower[axis] - span[axis]
            if direction < 0.0
            else target_upper[axis] + span[axis]
        )

    selected = random.randint(0, boxes.active_cell_count, size=arguments.boundary_targets)
    boundary = np.empty((arguments.boundary_targets, 4), dtype=float)
    for index, cell in enumerate(selected):
        axis = index % 4
        boundary[index] = 0.5 * (
            boxes.exact_lower[cell] + boxes.exact_upper[cell]
        )
        if index % 2 == 0:
            boundary[index, axis] = boxes.exact_lower[cell, axis]
        else:
            # A routing-shell probe lies strictly beyond this exact face but
            # remains within its tolerance-expanded routing box.
            boundary[index, axis] = 0.5 * (
                boxes.exact_upper[cell, axis]
                + boxes.routing_upper[cell, axis]
            )

    random_envelope = random.uniform(
        target_lower,
        target_upper,
        size=(arguments.random_envelope, 4),
    )
    extra_targets = []
    extra_categories = []
    if arguments.stratified_grid_samples:
        envelope = table.find_boundaries(space="tilde")
        grid = InversionGrid.from_table(
            table,
            InversionGridSpec(
                "tilde",
                {
                    name: GridAxisSpec(arguments.stratified_grid_count)
                    for name in envelope.names
                },
                max_points=arguments.stratified_grid_count**4,
            ),
        )
        sample_count = min(arguments.stratified_grid_samples, grid.size)
        sample_random = np.random.RandomState(arguments.seed + 1)
        edges = np.linspace(0, grid.size, sample_count + 1, dtype=np.int64)
        flat = np.asarray(
            [
                sample_random.randint(int(left), int(right))
                for left, right in zip(edges[:-1], edges[1:])
            ],
            dtype=np.int64,
        )
        multi = np.unravel_index(flat, grid.shape, order="C")
        requested = np.column_stack(
            [axis[index] for axis, index in zip(grid.axes, multi)]
        )
        transformed = ConstrainedTildeTransform(
            table.charge_constraints
        ).independent_to_targets(requested)
        extra_targets.append(np.asarray(transformed))
        extra_categories.extend(
            ["stratified_%d^4_tilde_grid" % arguments.stratified_grid_count]
            * transformed.shape[0]
        )
    if arguments.zero_charge_diagonal:
        envelope = table.find_boundaries(space="tilde")
        bounds = {
            name: (float(lower), float(upper))
            for name, lower, upper in zip(
                envelope.names, envelope.lower, envelope.upper
            )
        }
        for name in ("muQ_tilde", "muS_tilde"):
            if name not in bounds or not (bounds[name][0] <= 0.0 <= bounds[name][1]):
                raise ValueError(
                    "zero-charge diagonal is outside the tilde envelope for %s"
                    % name
                )
        diagonal_count = arguments.zero_charge_diagonal
        requested = np.column_stack(
            (
                np.linspace(*bounds["T_tilde"], diagonal_count),
                np.linspace(*bounds["muB_tilde"], diagonal_count),
                np.zeros(diagonal_count),
                np.zeros(diagonal_count),
            )
        )
        transformed = ConstrainedTildeTransform(
            table.charge_constraints
        ).independent_to_targets(requested)
        extra_targets.append(np.asarray(transformed))
        extra_categories.extend(["zero_charge_diagonal"] * diagonal_count)

    targets = np.vstack((outside, known, boundary, random_envelope, *extra_targets))
    categories = np.asarray(
        ["obviously_unreachable"] * outside.shape[0]
        + ["known_reachable"] * known.shape[0]
        + ["boundary_near_boundary"] * boundary.shape[0]
        + ["random_inverse_envelope"] * random_envelope.shape[0]
        + extra_categories
    )
    valid = np.ones(targets.shape[0], dtype=bool)
    provenance = {
        "mode": "representative",
        "counts": {
            "obviously_unreachable": int(outside.shape[0]),
            "known_reachable": int(known.shape[0]),
            "boundary_near_boundary": int(boundary.shape[0]),
            "random_inverse_envelope": int(random_envelope.shape[0]),
            "stratified_%d^4_tilde_grid" % arguments.stratified_grid_count: int(
                arguments.stratified_grid_samples
            ),
            "zero_charge_diagonal": int(arguments.zero_charge_diagonal),
        },
        "source_sampling": "independent_uniform_on_each_validated_source_axis",
        "envelope_sampling": "fixed_seed_uniform_canonical_target_envelope",
    }
    return targets, valid, categories, provenance


def _grid_targets(table, arguments):
    if arguments.grid_count is None:
        raise ValueError("--grid-count is required when --target-mode=grid")
    envelope = table.find_boundaries(space=arguments.grid_space)
    specification = InversionGridSpec(
        arguments.grid_space,
        {
            name: GridAxisSpec(arguments.grid_count)
            for name in envelope.names
        },
        max_points=arguments.grid_count**4,
        max_result_bytes=max(1, arguments.maximum_benchmark_memory_bytes),
    )
    grid = InversionGrid.from_table(table, specification)
    targets = []
    valid = []
    for chunk in grid.iter_chunks(arguments.target_chunk_size):
        targets.append(np.asarray(chunk.canonical_targets))
        valid.append(np.asarray(chunk.transform_valid))
    canonical = np.vstack(targets)
    transform_valid = np.concatenate(valid)
    categories = np.full(canonical.shape[0], "%s_grid" % arguments.grid_space)
    provenance = {
        "mode": "grid",
        "space": arguments.grid_space,
        "count_per_axis": arguments.grid_count,
        "shape": list(grid.shape),
        "ordering": grid.ordering,
        "axis_names": list(grid.axis_names),
        "axis_bounds": [
            [float(axis[0]), float(axis[-1])] for axis in grid.axes
        ],
        "transform_id": grid.transform_id,
    }
    return canonical, transform_valid, categories, provenance


def _measure_leaf_throughput(targets, valid, boxes, arguments):
    positions = np.flatnonzero(valid)[: arguments.throughput_targets]
    if positions.size == 0:
        raise ValueError("no finite transform-valid targets for throughput sampling")
    leaf_count = min(arguments.throughput_leaves, boxes.active_cell_count)
    sample_targets = np.asarray(targets[positions], dtype=float)
    started = time.perf_counter()
    exact = np.all(
        (sample_targets[:, None, :] >= boxes.exact_lower[None, :leaf_count])
        & (sample_targets[:, None, :] <= boxes.exact_upper[None, :leaf_count]),
        axis=2,
    )
    routing = np.all(
        (sample_targets[:, None, :] >= boxes.routing_lower[None, :leaf_count])
        & (sample_targets[:, None, :] <= boxes.routing_upper[None, :leaf_count]),
        axis=2,
    )
    # Consume both arrays so the benchmark cannot be optimized away.
    match_count = int(np.count_nonzero(exact) + np.count_nonzero(routing))
    seconds = time.perf_counter() - started
    pairs = int(sample_targets.shape[0] * leaf_count)
    throughput = pairs / max(seconds, np.finfo(float).tiny)
    return throughput, {
        "target_count": int(sample_targets.shape[0]),
        "leaf_count": int(leaf_count),
        "pair_count": pairs,
        "seconds": float(seconds),
        "pairs_per_second": float(throughput),
        "match_count_consumed": match_count,
    }


def _write_outputs(directory: Path, document, markdown: str, overwrite: bool):
    directory = Path(os.path.abspath(os.fspath(directory.expanduser())))
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "reachability_benchmark.json"
    markdown_path = directory / "reachability_benchmark.md"
    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "benchmark output exists; use --overwrite: %s"
            % ", ".join(str(path) for path in existing)
        )
    json_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path


def _markdown(document) -> str:
    timing = document["timing_seconds"]
    completion = document["leaf_completion"]
    profile = document["profile"]
    return """# HRG4D reachability benchmark

- Depth: {depth}
- Targets: {targets}
- Completed routing traversal: {traversal}
- Exact leaf classification completed: {completed}
- Stop reason: {reason}
- Hierarchy build: {build:.6f} s
- Coarse traversal: {coarse:.6f} s
- Exact leaf phase: {leaf}
- Total measured/estimated reachability time: {total:.6f} s
- Residual descendant leaf pairs: {pairs}
- Classification-only leaf pairs: {classification}
- Additional minimum-witness leaf pairs: {witness}
- Additional vector-batch leaf pairs: {batch}
- Maximum estimated leaf seconds (run configuration): {limit}
- Peak RSS: {rss} bytes
""".format(
        depth=document["configuration"]["hierarchy_levels"],
        targets=document["targets"]["count"],
        traversal=profile["traversal_completed"],
        completed=completion["completed"],
        reason=completion["stopped_reason"],
        build=timing["hierarchy_build"],
        coarse=timing["coarse_traversal"],
        leaf=(
            "%.6f s" % timing["exact_leaf"]
            if timing["exact_leaf"] is not None
            else "not run"
        ),
        total=timing["total_reachability_measured_or_estimated"],
        pairs=profile["descendant_leaf_pairs"],
        classification=completion["classification_leaf_pairs"],
        witness=completion["additional_minimum_witness_leaf_pairs"],
        batch=completion["additional_batch_leaf_pairs"],
        limit=document["configuration"]["maximum_estimated_leaf_seconds"],
        rss=document["memory_bytes"]["peak_rss"],
    )


def main(argv=None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.target_mode == "representative" and arguments.grid_count is not None:
        parser.error("--grid-count is valid only with --target-mode=grid")
    run_started = time.perf_counter()
    table = ForwardTable.load_validated(arguments.metadata, data_path=arguments.data)
    if len(table.coordinate_names) != 4:
        parser.error("the hierarchy benchmark requires a 4D table")
    interpolator = MultilinearInterpolator(table, table.target_names)
    exhaustive_index = ReachabilityIndex(
        interpolator,
        np.maximum(
            interpolator.node_value_bounds.upper
            - interpolator.node_value_bounds.lower,
            np.finfo(float).tiny,
        ),
        arguments.root_residual_tolerance,
        config=ReachabilityConfig(method="exhaustive"),
    )
    boxes = exhaustive_index.enclosures

    if arguments.target_mode == "representative":
        targets, valid, categories, provenance = _representative_targets(
            table, interpolator, boxes, arguments
        )
    else:
        try:
            targets, valid, categories, provenance = _grid_targets(table, arguments)
        except ValueError as exc:
            parser.error(str(exc))

    target_bytes = int(targets.nbytes + valid.nbytes + categories.nbytes)
    predicted_minimum_memory = int(
        boxes.nbytes + target_bytes + arguments.maximum_hierarchy_bytes
    )
    if predicted_minimum_memory > arguments.maximum_benchmark_memory_bytes:
        raise MemoryError(
            "declared benchmark memory is below the conservative input/index/"
            "hierarchy allowance (%d > %d)"
            % (predicted_minimum_memory, arguments.maximum_benchmark_memory_bytes)
        )

    throughput, throughput_report = _measure_leaf_throughput(
        targets, valid, boxes, arguments
    )
    hierarchy_started = time.perf_counter()
    hierarchy = build_routing_hierarchy(
        interpolator.cell_shape,
        boxes.routing_lower,
        boxes.routing_upper,
        hierarchy_levels=arguments.hierarchy_levels,
        max_hierarchy_bytes=arguments.maximum_hierarchy_bytes,
    )
    hierarchy_seconds = time.perf_counter() - hierarchy_started
    profile = profile_hierarchy_pruning(
        hierarchy,
        targets,
        valid_mask=valid,
        categories=categories,
        maximum_estimated_leaf_seconds=arguments.maximum_estimated_leaf_seconds,
        leaf_pairs_per_second=throughput,
        query_batch_size=arguments.query_batch_size,
        maximum_pairwise_entries=arguments.maximum_pairwise_entries,
        hierarchy_construction_seconds=hierarchy_seconds,
    )

    elapsed_before_leaf = time.perf_counter() - run_started
    gates = {
        "explicit_leaf_completion_requested": bool(arguments.complete_leaves),
        "routing_traversal_completed": bool(profile.traversal_completed),
        "estimated_leaf_seconds_within_limit": bool(profile.leaf_completion_allowed),
        "estimated_leaf_pairs_within_limit": bool(
            profile.descendant_leaf_pairs
            <= arguments.maximum_estimated_leaf_pairs
        ),
        "estimated_total_wall_time_within_limit": bool(
            elapsed_before_leaf + profile.estimated_leaf_seconds
            <= arguments.maximum_benchmark_wall_seconds
        ),
    }
    complete_leaves = all(gates.values())
    leaf_result = None
    leaf_wall_seconds = None
    if complete_leaves:
        leaf_started = time.perf_counter()
        leaf_result = classify_hierarchy_leaves(
            hierarchy,
            targets,
            boxes.exact_lower,
            boxes.exact_upper,
            boxes.routing_lower,
            boxes.routing_upper,
            valid_mask=valid,
            query_batch_size=arguments.query_batch_size,
            leaf_batch_size=min(
                4096, arguments.maximum_pairwise_entries
            ),
            maximum_pairwise_entries=arguments.maximum_pairwise_entries,
        )
        leaf_wall_seconds = time.perf_counter() - leaf_started

    if leaf_result is not None:
        stopped_reason = None
        classification_pairs = leaf_result.diagnostics.classification_leaf_pairs
        witness_pairs = (
            leaf_result.diagnostics.additional_minimum_witness_leaf_pairs
        )
        batch_pairs = leaf_result.diagnostics.additional_batch_leaf_pairs
        actual_pairs = leaf_result.diagnostics.actual_leaf_pairs
    else:
        failed_gates = [name for name, passed in gates.items() if not passed]
        stopped_reason = profile.stopped_reason or ",".join(failed_gates)
        classification_pairs = None
        witness_pairs = None
        batch_pairs = None
        actual_pairs = None

    oracle_report = {
        "requested_sample_count": arguments.oracle_equivalence_sample,
        "completed": False,
        "sample_count": 0,
        "seconds": None,
        "classification_mismatches": None,
        "exact_mismatches": None,
        "routing_witness_mismatches": None,
        "exact_witness_mismatches": None,
    }
    if leaf_result is not None and arguments.oracle_equivalence_sample:
        eligible = np.flatnonzero(valid)
        sample_count = min(arguments.oracle_equivalence_sample, eligible.size)
        positions = eligible[
            np.unique(
                np.linspace(0, eligible.size - 1, sample_count, dtype=np.int64)
            )
        ]
        sample_targets = targets[positions]
        oracle_query_batch = min(
            arguments.query_batch_size, max(1, sample_targets.shape[0])
        )
        oracle_cell_batch = min(
            4096,
            max(1, arguments.maximum_pairwise_entries // oracle_query_batch),
        )
        oracle_started = time.perf_counter()
        oracle = _scan_target_boxes(
            sample_targets,
            np.ones(sample_targets.shape[0], dtype=bool),
            boxes.exact_lower,
            boxes.exact_upper,
            boxes.routing_lower,
            boxes.routing_upper,
            boxes.cell_flat_indices,
            ReachabilityConfig(
                method="exhaustive",
                query_batch_size=oracle_query_batch,
                cell_batch_size=oracle_cell_batch,
                max_pairwise_entries=arguments.maximum_pairwise_entries,
            ),
        )
        oracle_seconds = time.perf_counter() - oracle_started
        mismatches = {
            "classification_mismatches": int(
                np.count_nonzero(
                    oracle.routing_candidate
                    != leaf_result.routing_candidate[positions]
                )
            ),
            "exact_mismatches": int(
                np.count_nonzero(
                    oracle.exact_candidate
                    != leaf_result.exact_candidate[positions]
                )
            ),
            "routing_witness_mismatches": int(
                np.count_nonzero(
                    oracle.routing_witness
                    != leaf_result.routing_witness[positions]
                )
            ),
            "exact_witness_mismatches": int(
                np.count_nonzero(
                    oracle.exact_witness
                    != leaf_result.exact_witness[positions]
                )
            ),
        }
        if any(mismatches.values()):
            raise RuntimeError(
                "hierarchical classification disagrees with the exhaustive oracle: %r"
                % mismatches
            )
        oracle_report.update(
            completed=True,
            sample_count=int(positions.size),
            seconds=float(oracle_seconds),
            **mismatches,
        )

    target_digest = hashlib.sha256()
    target_digest.update(np.ascontiguousarray(targets).tobytes(order="C"))
    target_digest.update(np.ascontiguousarray(valid).tobytes(order="C"))
    total_reachability = (
        hierarchy_seconds + leaf_wall_seconds
        if leaf_wall_seconds is not None
        else hierarchy_seconds
        + profile.coarse_traversal_seconds
        + profile.estimated_leaf_seconds
    )
    document = {
        "benchmark_schema_version": 2,
        "benchmark_kind": "4d_hierarchical_reachability_no_solver",
        "configuration": {
            "hierarchy_levels": arguments.hierarchy_levels,
            "seed": arguments.seed,
            "query_batch_size": arguments.query_batch_size,
            "maximum_pairwise_entries": arguments.maximum_pairwise_entries,
            "maximum_hierarchy_bytes": arguments.maximum_hierarchy_bytes,
            "maximum_benchmark_memory_bytes": arguments.maximum_benchmark_memory_bytes,
            "maximum_estimated_leaf_pairs": arguments.maximum_estimated_leaf_pairs,
            "maximum_estimated_leaf_seconds": arguments.maximum_estimated_leaf_seconds,
            "maximum_benchmark_wall_seconds": arguments.maximum_benchmark_wall_seconds,
            "root_residual_tolerance": arguments.root_residual_tolerance,
            "residual_scales": boxes.residual_scales.tolist(),
        },
        "table": {
            "metadata_path": str(table.metadata_path),
            "snapshot_fingerprint": table.snapshot_fingerprint(),
            "cell_shape": list(interpolator.cell_shape),
            "cell_count": boxes.active_cell_count,
        },
        "targets": {
            "count": int(targets.shape[0]),
            "valid_count": int(np.count_nonzero(valid)),
            "sha256": target_digest.hexdigest(),
            "provenance": provenance,
        },
        "throughput_calibration": throughput_report,
        "profile": profile.to_dict(),
        "leaf_completion": {
            "gates": gates,
            "completed": leaf_result is not None,
            "stopped_reason": stopped_reason,
            "actual_leaf_pairs": actual_pairs,
            "classification_leaf_pairs": classification_pairs,
            "additional_minimum_witness_leaf_pairs": witness_pairs,
            "additional_batch_leaf_pairs": batch_pairs,
        },
        "oracle_equivalence": oracle_report,
        "timing_seconds": {
            "hierarchy_build": float(hierarchy_seconds),
            "coarse_traversal": float(profile.coarse_traversal_seconds),
            "exact_leaf": leaf_wall_seconds,
            "exact_classifier_coarse": (
                None
                if leaf_result is None
                else leaf_result.diagnostics.coarse_seconds
            ),
            "exact_leaf_comparisons": (
                None
                if leaf_result is None
                else leaf_result.diagnostics.leaf_seconds
            ),
            "total_reachability_measured_or_estimated": float(
                total_reachability
            ),
            "benchmark_wall": float(time.perf_counter() - run_started),
        },
        "memory_bytes": {
            "enclosures": boxes.nbytes,
            "hierarchy_owned": hierarchy.owned_nbytes,
            "hierarchy_referenced_leaf": hierarchy.referenced_leaf_nbytes,
            "targets": target_bytes,
            "peak_rss": _rss_bytes(),
        },
        "scientific_classification_complete": leaf_result is not None,
        "solver_invoked": False,
    }
    markdown = _markdown(document)
    json_path, markdown_path = _write_outputs(
        arguments.output_directory, document, markdown, arguments.overwrite
    )
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
