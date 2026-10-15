#!/usr/bin/env python3
"""Run the two-tier v1 known-root preservation gate on the real HRG4D map.

Tier A follows only closed dyadic boxes that contain each known source point
and evaluates them with the production rigorous enclosure kernel.  Tier B
uses complete depth-7 reachability localization and the opt-in production
subcell FALLBACK route.  The script is diagnostic/test tooling: it changes no
production configuration, tolerance, or scientific contract.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from eos_trensmuther import (
    EoSRootSolver,
    MultilinearInterpolator,
    ProofStatus,
    ReachabilityConfig,
    ReachabilityIndex,
    SolveStatus,
    SolverConfig,
    load_pipeline_result,
)
from eos_trensmuther.shared_state import load_shared_reachability, load_shared_table
from eos_trensmuther.solver import (
    _RigorousCellModel,
    _RigorousSubcellEnclosureEvaluator,
)
from eos_trensmuther.subcells import SubcellBoxes


SCHEMA_VERSION = 1
RNG_SEED = 20261007
TIER_A_COUNT = 3000
TIER_B_REAL_COUNT = 392
TIER_B_SYNTHETIC_COUNT = 8
CI_REAL_COUNT = 192
UNIQUE_ROOT_COUNT = 48
MAX_DEPTH = 7
DEFAULT_ARTIFACT = Path(
    "Output_TEST/HRG4D30Mmap8/HRG4D30Mmap8/inverse.eosinv"
)
DEFAULT_SHARED_STATE = Path(
    "Output_TEST/HRG4D30Mmap8/HRG4D30Mmap8/diagnostic/"
    "shared_worker_state_v1/shared_state.json"
)
DEFAULT_OUTPUT = Path("Output_TEST/V1Phase5RootPreservation")

TIER_A_QUOTAS: Mapping[str, int] = {
    "ordinary_interior": 550,
    "dyadic_interior": 250,
    "internal_face": 350,
    "internal_edge": 300,
    "grid_node": 300,
    "domain_face": 350,
    "domain_corner": 16,
    "small_density": 442,
    "large_chemical_potential": 442,
}
TIER_B_QUOTAS: Mapping[str, int] = {
    "ordinary_interior": 40,
    "dyadic_interior": 36,
    "internal_face": 50,
    "internal_edge": 50,
    "grid_node": 50,
    "domain_face": 50,
    "domain_corner": 16,
    "small_density": 50,
    "large_chemical_potential": 50,
}
CI_QUOTAS: Mapping[str, int] = {
    "ordinary_interior": 20,
    "dyadic_interior": 18,
    "internal_face": 24,
    "internal_edge": 24,
    "grid_node": 24,
    "domain_face": 24,
    "domain_corner": 8,
    "small_density": 25,
    "large_chemical_potential": 25,
}
CONDITION_BANDS = (
    ("q00_50", 0.00, 0.50, 0.12),
    ("q50_75", 0.50, 0.75, 0.13),
    ("q75_90", 0.75, 0.90, 0.15),
    ("q90_95", 0.90, 0.95, 0.15),
    ("q95_99", 0.95, 0.99, 0.20),
    ("q99_100", 0.99, 1.00, 0.25),
)


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


def _points_from_cells(
    interpolator: MultilinearInterpolator,
    cells: np.ndarray,
    unit: np.ndarray,
) -> np.ndarray:
    lower, upper = interpolator.cell_coordinate_bounds(cells)
    return lower + (upper - lower) * unit


def _random_cell_points(
    interpolator: MultilinearInterpolator,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    cells = np.column_stack(
        [rng.integers(0, size, size=count) for size in interpolator.cell_shape]
    ).astype(np.intp)
    unit = rng.uniform(0.03125, 0.96875, size=(count, interpolator.dimension))
    return _points_from_cells(interpolator, cells, unit)


def _geometry_pool(
    interpolator: MultilinearInterpolator,
    category: str,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dimension = interpolator.dimension
    if category == "domain_corner":
        return np.asarray(
            list(itertools.product(*[(axis[0], axis[-1]) for axis in interpolator.axes])),
            dtype=float,
        )
    if category == "grid_node":
        total = math.prod(interpolator.shape)
        flat = rng.choice(total, size=count, replace=False)
        multi = np.column_stack(
            np.unravel_index(flat, interpolator.shape, order="C")
        )
        return np.column_stack(
            [axis[multi[:, axis_index]] for axis_index, axis in enumerate(interpolator.axes)]
        )

    cells = np.column_stack(
        [rng.integers(0, size, size=count) for size in interpolator.cell_shape]
    ).astype(np.intp)
    unit = rng.uniform(0.03125, 0.96875, size=(count, dimension))
    if category == "dyadic_interior":
        choices = np.asarray([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
        unit = rng.choice(choices, size=(count, dimension))
    elif category in {"internal_face", "internal_edge"}:
        boundary_count = 1 if category == "internal_face" else 2
        for row in range(count):
            axes = rng.choice(dimension, size=boundary_count, replace=False)
            for axis_index in axes:
                node = int(rng.integers(1, interpolator.shape[axis_index] - 1))
                cells[row, axis_index] = node
                unit[row, axis_index] = 0.0
    elif category == "domain_face":
        for row in range(count):
            axis_index = int(rng.integers(0, dimension))
            side = int(rng.integers(0, 2))
            cells[row, axis_index] = 0 if side == 0 else interpolator.cell_shape[axis_index] - 1
            unit[row, axis_index] = float(side)
    return _points_from_cells(interpolator, cells, unit)


def _conditioning(
    jacobians: np.ndarray,
    source_spans: np.ndarray,
    residual_scales: np.ndarray,
) -> np.ndarray:
    scaled = (
        np.asarray(jacobians, dtype=float)
        * source_spans[np.newaxis, np.newaxis, :]
        / residual_scales[np.newaxis, :, np.newaxis]
    )
    singular = np.linalg.svd(scaled, compute_uv=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        condition = singular[:, 0] / singular[:, -1]
    condition[~np.isfinite(condition)] = np.inf
    return condition


def _condition_labels(condition: np.ndarray) -> np.ndarray:
    order = np.argsort(condition, kind="stable")
    labels = np.empty(condition.size, dtype="U16")
    for label, lower, upper, _ in CONDITION_BANDS:
        start = int(math.floor(lower * condition.size))
        stop = condition.size if upper == 1.0 else int(math.floor(upper * condition.size))
        labels[order[start:stop]] = label
    return labels


def _stratified_indices(
    condition: np.ndarray,
    quota: int,
    rng: np.random.Generator,
) -> np.ndarray:
    order = np.argsort(condition, kind="stable")
    selected: List[int] = []
    used = np.zeros(condition.size, dtype=bool)
    remaining_quota = quota
    for band_index, (_, lower, upper, fraction) in enumerate(CONDITION_BANDS):
        start = int(math.floor(lower * condition.size))
        stop = condition.size if upper == 1.0 else int(math.floor(upper * condition.size))
        available = order[start:stop]
        requested = (
            remaining_quota
            if band_index == len(CONDITION_BANDS) - 1
            else int(round(quota * fraction))
        )
        take = min(requested, available.size, remaining_quota)
        if take:
            chosen = rng.choice(available, size=take, replace=False)
            selected.extend(int(value) for value in chosen)
            used[chosen] = True
            remaining_quota -= take
    if remaining_quota:
        # Fill deficits from the most difficult unused cases, preserving the
        # intended worst-tail emphasis when a very narrow quantile is small.
        unused = order[~used[order]][::-1]
        selected.extend(int(value) for value in unused[:remaining_quota])
    result = np.asarray(selected, dtype=np.int64)
    if result.size != quota or np.unique(result).size != quota:
        raise RuntimeError("condition-stratified selection did not meet its quota")
    return result


def generate_tier_a_sample(
    interpolator: MultilinearInterpolator,
    residual_scales: np.ndarray,
    *,
    seed: int = RNG_SEED,
) -> Dict[str, np.ndarray]:
    """Generate the exact deterministic 3,000-case Tier-A sample."""

    rng = np.random.default_rng(seed)
    points_parts: List[np.ndarray] = []
    category_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    condition_parts: List[np.ndarray] = []
    target_parts: List[np.ndarray] = []
    source_spans = np.ptp(interpolator.bounds, axis=1)

    for category, quota in TIER_A_QUOTAS.items():
        if category == "domain_corner":
            pool_count = quota
        else:
            pool_count = max(quota * 8, quota + 256)
        pool_category = category
        if category in {"small_density", "large_chemical_potential"}:
            pool_category = "ordinary_interior"
        points = _geometry_pool(interpolator, pool_category, pool_count, rng)
        evaluation = interpolator.evaluate_with_jacobian(points, plane_policy="right")
        targets = np.asarray(evaluation.values, dtype=float)
        condition = _conditioning(
            evaluation.jacobians, source_spans, residual_scales
        )
        if category == "small_density":
            density = np.linalg.norm(
                targets[:, 1:] / residual_scales[np.newaxis, 1:], axis=1
            )
            keep = np.argsort(density, kind="stable")[: pool_count // 2]
            points, targets, condition = points[keep], targets[keep], condition[keep]
        elif category == "large_chemical_potential":
            normalized = 2.0 * (points - interpolator.bounds[:, 0]) / source_spans - 1.0
            chemical_score = np.max(np.abs(normalized[:, 1:]), axis=1)
            keep = np.argsort(chemical_score, kind="stable")[pool_count // 2 :]
            points, targets, condition = points[keep], targets[keep], condition[keep]

        if category == "domain_corner":
            chosen = np.arange(quota, dtype=np.int64)
        else:
            chosen = _stratified_indices(condition, quota, rng)
        labels = _condition_labels(condition)
        points_parts.append(points[chosen])
        target_parts.append(targets[chosen])
        condition_parts.append(condition[chosen])
        label_parts.append(labels[chosen])
        category_parts.append(np.full(quota, category, dtype="U32"))

    points = np.concatenate(points_parts)
    targets = np.concatenate(target_parts)
    categories = np.concatenate(category_parts)
    conditions = np.concatenate(condition_parts)
    labels = np.concatenate(label_parts)
    if points.shape != (TIER_A_COUNT, interpolator.dimension):
        raise RuntimeError("Tier-A quota table does not sum to 3,000")
    return {
        "source_coordinates": points,
        "targets": targets,
        "categories": categories,
        "condition_numbers": conditions,
        "condition_bands": labels,
    }


def _owner_cells_and_unit_coordinates(
    interpolator: MultilinearInterpolator, point: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    per_axis: List[List[Tuple[int, float]]] = []
    for value, axis in zip(point, interpolator.axes):
        matches = np.flatnonzero(axis == value)
        owners: List[Tuple[int, float]] = []
        if matches.size:
            node = int(matches[0])
            if node > 0:
                owners.append((node - 1, 1.0))
            if node < axis.size - 1:
                owners.append((node, 0.0))
        else:
            cell = int(np.searchsorted(axis, value, side="right") - 1)
            cell = min(max(cell, 0), axis.size - 2)
            unit = float((value - axis[cell]) / (axis[cell + 1] - axis[cell]))
            owners.append((cell, unit))
        per_axis.append(owners)
    combinations = list(itertools.product(*per_axis))
    cells = np.asarray(
        [[entry[0] for entry in combination] for combination in combinations],
        dtype=np.intp,
    )
    unit = np.asarray(
        [[entry[1] for entry in combination] for combination in combinations],
        dtype=float,
    )
    return cells, unit


def _root_boxes_at_depth(
    cell_flat: np.ndarray, unit: np.ndarray, depth: int
) -> SubcellBoxes:
    resolution = 1 << depth
    records: List[Tuple[int, Tuple[int, ...]]] = []
    dimension = unit.shape[1]
    for cell_id, coordinates in zip(cell_flat, unit):
        per_axis: List[List[int]] = []
        for value in coordinates:
            scaled = float(value) * resolution
            nearest = int(round(scaled))
            choices: List[int] = []
            if scaled == float(nearest):
                if nearest > 0:
                    choices.append(nearest - 1)
                if nearest < resolution:
                    choices.append(nearest)
            else:
                choices.append(min(resolution - 1, max(0, int(math.floor(scaled)))))
            per_axis.append(choices)
        for local in itertools.product(*per_axis):
            records.append((int(cell_id), tuple(int(value) for value in local)))
    records = sorted(set(records))
    cells = np.asarray([record[0] for record in records], dtype=np.int64)
    local = np.asarray([record[1] for record in records], dtype=np.int64)
    lower = local / float(resolution)
    upper = (local + 1) / float(resolution)
    return SubcellBoxes(dimension, depth, cells, lower, upper)


def check_kernel_root_preservation(
    interpolator: MultilinearInterpolator,
    target: np.ndarray,
    point: np.ndarray,
    residual_scales: np.ndarray,
    root_residual_tolerance: float,
) -> Dict[str, Any]:
    """Check every closed production ownership branch through depth seven."""

    cells, unit = _owner_cells_and_unit_coordinates(interpolator, point)
    flat = np.ravel_multi_index(cells.T, interpolator.cell_shape, order="C")
    evaluator = _RigorousSubcellEnclosureEvaluator(
        interpolator,
        target,
        residual_scales,
        root_residual_tolerance,
    )
    minimum_margin = math.inf
    branches_per_level = []
    total_boxes = 0
    started = time.perf_counter()
    for depth in range(MAX_DEPTH + 1):
        boxes = _root_boxes_at_depth(flat, unit, depth)
        multi = np.column_stack(
            np.unravel_index(boxes.cell_flat_indices, interpolator.cell_shape, order="C")
        ).astype(np.intp, copy=False)
        exact = _RigorousCellModel.value_enclosures_for_cells(
            interpolator, multi, boxes.unit_lower, boxes.unit_upper
        )
        routing = evaluator(boxes)
        defined = np.all(np.isfinite(exact.lower) & np.isfinite(exact.upper), axis=1)
        exact_contains = defined & np.all(
            (target[np.newaxis, :] >= exact.lower)
            & (target[np.newaxis, :] <= exact.upper),
            axis=1,
        )
        routing_contains = routing.defined & np.all(
            (target[np.newaxis, :] >= routing.routing_lower)
            & (target[np.newaxis, :] <= routing.routing_upper),
            axis=1,
        )
        physical_lower, physical_upper = evaluator.physical_bounds(boxes)
        source_contains = np.all(
            (point[np.newaxis, :] >= physical_lower)
            & (point[np.newaxis, :] <= physical_upper),
            axis=1,
        )
        if not np.all(defined & exact_contains & routing_contains & source_contains):
            raise AssertionError(
                "known root branch excluded at depth %d: exact=%s routing=%s source=%s"
                % (depth, exact_contains.tolist(), routing_contains.tolist(), source_contains.tolist())
            )
        margin = np.minimum(
            target[np.newaxis, :] - exact.lower,
            exact.upper - target[np.newaxis, :],
        ) / residual_scales[np.newaxis, :]
        minimum_margin = min(minimum_margin, float(np.min(margin)))
        branches_per_level.append(int(boxes.count))
        total_boxes += boxes.count
    return {
        "owner_cell_count": int(flat.size),
        "owner_cell_flat_ids": flat.tolist(),
        "branches_per_level": branches_per_level,
        "total_boxes_checked": int(total_boxes),
        "minimum_scaled_enclosure_margin": float(minimum_margin),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _distribution(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
    }


def _save_sample(path: Path, sample: Mapping[str, np.ndarray], **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value) for name, value in sample.items()}
    arrays.update({name: np.asarray(value) for name, value in extra.items()})
    np.savez_compressed(path, **arrays)


def _localization_statistics(
    reachability: ReachabilityIndex,
    targets: np.ndarray,
    known_points: np.ndarray,
    interpolator: MultilinearInterpolator,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    localization = reachability.localize(targets)
    localization.require_complete_for(reachability.provenance)
    owner_present = np.zeros(targets.shape[0], dtype=bool)
    owner_counts = np.zeros(targets.shape[0], dtype=np.int64)
    for row, point in enumerate(known_points):
        cells, _ = _owner_cells_and_unit_coordinates(interpolator, point)
        owner_flat = np.ravel_multi_index(cells.T, interpolator.cell_shape, order="C")
        candidates = localization.cells_for(row)
        owner_counts[row] = owner_flat.size
        owner_present[row] = bool(np.all(np.isin(owner_flat, candidates)))
    return localization, localization.candidate_counts, owner_present


def _candidate_center_seed(
    interpolator: MultilinearInterpolator,
    target: np.ndarray,
    candidate_cells: np.ndarray,
    residual_scales: np.ndarray,
) -> Tuple[np.ndarray, int, float]:
    """Reproduce the production candidate-center ranking and tie rule."""

    cells = np.asarray(candidate_cells, dtype=np.int64)
    multi = np.column_stack(
        np.unravel_index(cells, interpolator.cell_shape, order="C")
    )
    lower, upper = interpolator.cell_coordinate_bounds(multi)
    centers = np.asarray(lower) / 2.0 + np.asarray(upper) / 2.0
    achieved = np.asarray(interpolator.evaluate(centers), dtype=float)
    norms = np.max(
        np.abs(
            (achieved - target[np.newaxis, :])
            / residual_scales[np.newaxis, :]
        ),
        axis=1,
    )
    selected = int(np.flatnonzero(norms == np.min(norms))[0])
    return centers[selected], int(cells[selected]), float(norms[selected])


def _tier_b_pool_indices(sample: Mapping[str, np.ndarray]) -> np.ndarray:
    categories = sample["categories"]
    condition = sample["condition_numbers"]
    selected: List[int] = []
    for category, quota in TIER_B_QUOTAS.items():
        available = np.flatnonzero(categories == category)
        pool_size = min(available.size, max(quota * 2, quota))
        order = available[np.argsort(condition[available], kind="stable")]
        selected.extend(int(value) for value in order[-pool_size:])
    return np.asarray(selected, dtype=np.int64)


def _select_tier_b(
    sample: Mapping[str, np.ndarray], pool_indices: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    categories = sample["categories"]
    condition = sample["condition_numbers"]
    selected: List[int] = []
    for category, quota in TIER_B_QUOTAS.items():
        positions = np.flatnonzero(categories[pool_indices] == category)
        indices = pool_indices[positions]
        if category == "domain_corner":
            chosen = indices
        else:
            count_rank = np.argsort(np.argsort(counts[positions], kind="stable"), kind="stable")
            condition_rank = np.argsort(
                np.argsort(condition[indices], kind="stable"), kind="stable"
            )
            score = count_rank + 0.75 * condition_rank
            chosen = indices[np.argsort(score, kind="stable")[-quota:]]
        if chosen.size != quota:
            raise RuntimeError("Tier-B category quota could not be filled")
        selected.extend(int(value) for value in chosen)
    result = np.asarray(selected, dtype=np.int64)
    if result.size != TIER_B_REAL_COUNT:
        raise RuntimeError("Tier-B quotas do not sum to 392")
    return result


def _folded_synthetic_cases() -> Dict[str, Any]:
    axes = (np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 1.0]))
    x, q = np.meshgrid(*axes, indexing="ij")
    interpolator = MultilinearInterpolator.from_rectilinear_map(
        axes,
        {"e": np.abs(x - 1.0), "nB": q},
        coordinate_names=("T", "muB"),
        target_names=("e", "nB"),
        coordinate_units=("1", "1"),
        target_units=("1", "1"),
    )
    config = SolverConfig(enable_subcell_fallback=True)
    solver = EoSRootSolver(interpolator, config, residual_scales=[1.0, 1.0])
    reachability = ReachabilityIndex(
        interpolator, [1.0, 1.0], config.root_residual_tolerance
    )
    charges = np.linspace(0.0, 1.0, TIER_B_SYNTHETIC_COUNT)
    points = np.column_stack((np.full(charges.size, 0.5), charges))
    targets = interpolator.evaluate(points)
    localization = reachability.localize(targets)
    rows = []
    for row, (point, target) in enumerate(zip(points, targets)):
        candidates = localization.cells_for(row)
        if candidates.size != 2:
            raise AssertionError("folded target did not retain both physical branches")
        proof_result = solver.solve_with_candidate_cells(
            target,
            localization,
            row,
            point,
            expected_provenance=reachability.provenance,
            strategy="fallback",
        )
        if proof_result.status is SolveStatus.CERTIFIED_NO_ROOT:
            raise AssertionError("folded known-root target was proof-excluded")
        result = solver.solve_with_candidate_cells(
            target,
            localization,
            row,
            point,
            expected_provenance=reachability.provenance,
            strategy="auto",
        )
        if not result.converged:
            raise AssertionError("folded known-root target did not return a valid root")
        if result.scaled_residual_norm is None or result.scaled_residual_norm > config.root_residual_tolerance:
            raise AssertionError("folded result violates residual tolerance")
        rows.append(
            {
                "case": row,
                "target": target.tolist(),
                "candidate_cells": int(candidates.size),
                "status": result.status.value,
                "proof_status": result.proof_status.value,
                "fallback_only_status": proof_result.status.value,
                "fallback_only_proof_status": proof_result.proof_status.value,
                "scaled_residual_norm_inf": result.scaled_residual_norm,
            }
        )
    return {"count": len(rows), "rows": rows}


def _unique_root_source_points() -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic HRG4D-shaped points and their geometry labels."""

    epsilon = 1.0 / 32.0
    normalized = np.asarray(
        [
            [0.17, 0.23, 0.31, 0.41],
            [0.83, 0.77, 0.69, 0.59],
            [0.20, 0.80, 0.35, 0.65],
            [0.80, 0.20, 0.65, 0.35],
            [0.45, 0.55, 0.25, 0.75],
            [0.55, 0.45, 0.75, 0.25],
            [0.33, 0.66, 0.20, 0.80],
            [0.66, 0.33, 0.80, 0.20],
            [epsilon, 0.40, 0.60, 0.20],
            [1.0 - epsilon, 0.60, 0.40, 0.80],
            [0.30, epsilon, 0.70, 0.50],
            [0.70, 1.0 - epsilon, 0.30, 0.50],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    categories = np.asarray(
        ["ordinary_interior"] * 8
        + ["near_boundary"] * 4
        + ["domain_corner"] * 4
    )
    return normalized, categories


def _unique_root_affine_profiles() -> Mapping[str, np.ndarray]:
    """Return globally invertible maps with three normalized condition levels."""

    hadamard = 0.5 * np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    )
    singular_values = {
        "well_conditioned": np.asarray([1.0, 0.8, 0.6, 0.4]),
        "moderately_conditioned": np.asarray([1.0, 0.2, 0.05, 0.01]),
        "high_condition_number": np.asarray([1.0, 0.1, 0.01, 0.001]),
    }
    return {
        name: hadamard @ np.diag(values) @ hadamard.T
        for name, values in singular_values.items()
    }


def run_unique_root_comparison() -> Dict[str, Any]:
    """Exercise the production route on analytically unique affine maps.

    Each target has exactly one source-domain inverse because its complete
    forward map is affine with a nonsingular matrix.  This supplies the
    returned-coordinate comparison that is intentionally invalid for the
    folded/non-injective Phase-5 fixtures.
    """

    started = time.perf_counter()
    coordinate_names = ("T", "muB", "muQ", "muS")
    target_names = ("e", "nB", "nQ", "nS")
    axes = (
        np.linspace(0.03, 0.17, 5),
        np.linspace(-0.7, 0.7, 5),
        np.linspace(-0.13, 0.13, 5),
        np.linspace(-0.194, 0.194, 5),
    )
    source_lower = np.asarray([axis[0] for axis in axes], dtype=float)
    source_upper = np.asarray([axis[-1] for axis in axes], dtype=float)
    source_spans = source_upper - source_lower
    source_grid_spacing = np.asarray(
        [float(np.min(np.diff(axis))) for axis in axes], dtype=float
    )
    normalized_points, categories = _unique_root_source_points()
    source_points = source_lower + normalized_points * source_spans
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    normalized_grid = (grid - source_lower) / source_spans
    target_offset = np.asarray([0.2, -0.1, 0.05, -0.02])
    residual_scales = np.ones(4)
    config = SolverConfig(
        enable_subcell_fallback=True,
        max_subcell_depth=MAX_DEPTH,
    )
    rows: List[Dict[str, Any]] = []
    profile_summaries: Dict[str, Any] = {}

    for profile, matrix in _unique_root_affine_profiles().items():
        inverse_matrix = np.linalg.inv(matrix)
        condition_number = float(np.linalg.cond(matrix, p=2))
        inverse_norm_inf = float(np.linalg.norm(inverse_matrix, ord=np.inf))
        # For y=A*u+b, ||du||_inf <= ||A^-1||_inf ||dy||_inf.  Four is a
        # documented numerical safety factor around the existing residual
        # contract; this changes no solver tolerance.
        normalized_coordinate_tolerance = float(
            max(
                4.0
                * inverse_norm_inf
                * config.root_residual_tolerance,
                64.0 * np.finfo(float).eps,
            )
        )
        target_grid = (
            np.einsum("ij,...j->...i", matrix, normalized_grid)
            + target_offset
        )
        interpolator = MultilinearInterpolator.from_rectilinear_map(
            axes,
            {
                name: target_grid[..., index]
                for index, name in enumerate(target_names)
            },
            coordinate_names=coordinate_names,
            target_names=target_names,
            coordinate_units=("GeV",) * 4,
            target_units=("1",) * 4,
        )
        solver = EoSRootSolver(
            interpolator,
            config,
            residual_scales=residual_scales,
        )
        reachability = ReachabilityIndex(
            interpolator,
            residual_scales,
            config.root_residual_tolerance,
            config=ReachabilityConfig(method="exhaustive"),
        )
        targets = np.asarray(interpolator.evaluate(source_points), dtype=float)
        localization, candidate_counts, owner_present = _localization_statistics(
            reachability,
            targets,
            source_points,
            interpolator,
        )
        if not np.all(owner_present):
            raise AssertionError("unique-root localization lost a known owner cell")

        profile_rows = []
        for local_index, (known, target) in enumerate(
            zip(source_points, targets)
        ):
            seed, seed_cell, seed_residual = _candidate_center_seed(
                interpolator,
                target,
                localization.cells_for(local_index),
                residual_scales,
            )
            result = solver.solve_with_candidate_cells(
                target,
                localization,
                local_index,
                seed,
                expected_provenance=reachability.provenance,
                strategy="auto",
            )
            if not result.converged or result.solution is None:
                raise AssertionError(
                    "analytically unique case did not return a solved root"
                )
            solved = np.asarray(result.solution, dtype=float)
            in_bounds = bool(np.all(interpolator.contains(solved)))
            achieved = np.asarray(interpolator.evaluate(solved), dtype=float)
            independent_scaled_residual = float(
                np.max(np.abs((achieved - target) / residual_scales))
            )
            absolute_error = np.abs(solved - known)
            normalized_span_error = absolute_error / source_spans
            normalized_grid_error = absolute_error / source_grid_spacing
            coordinate_match = bool(
                np.max(normalized_span_error)
                <= normalized_coordinate_tolerance
            )
            residual_ok = bool(
                result.scaled_residual_norm is not None
                and result.scaled_residual_norm
                <= config.root_residual_tolerance
                and independent_scaled_residual
                <= config.root_residual_tolerance
            )
            if not in_bounds or not residual_ok or not coordinate_match:
                raise AssertionError(
                    "unique-root result violates bounds, residual, or "
                    "conditioning-aware coordinate acceptance"
                )
            row = {
                "profile": profile,
                "case": local_index,
                "category": str(categories[local_index]),
                "known_source": known.tolist(),
                "solved_source": solved.tolist(),
                "target": target.tolist(),
                "solver_status": result.status.value,
                "disposition": "solved",
                "proof_status": result.proof_status.value,
                "path": list(result.path),
                "candidate_cells": int(candidate_counts[local_index]),
                "owner_cells_present": bool(owner_present[local_index]),
                "candidate_center_cell": seed_cell,
                "candidate_center_scaled_residual_norm_inf": seed_residual,
                "in_source_bounds": in_bounds,
                "scaled_residual_norm_inf": float(result.scaled_residual_norm),
                "independent_scaled_residual_norm_inf": independent_scaled_residual,
                "absolute_coordinate_error": absolute_error.tolist(),
                "full_span_normalized_coordinate_error": (
                    normalized_span_error.tolist()
                ),
                "grid_spacing_normalized_coordinate_error": (
                    normalized_grid_error.tolist()
                ),
                "maximum_full_span_normalized_coordinate_error": float(
                    np.max(normalized_span_error)
                ),
                "maximum_grid_spacing_normalized_coordinate_error": float(
                    np.max(normalized_grid_error)
                ),
                "full_span_normalized_coordinate_tolerance": (
                    normalized_coordinate_tolerance
                ),
                "coordinate_match": coordinate_match,
            }
            rows.append(row)
            profile_rows.append(row)

        profile_summaries[profile] = {
            "case_count": len(profile_rows),
            "normalized_jacobian_condition_number_2": condition_number,
            "normalized_inverse_jacobian_norm_inf": inverse_norm_inf,
            "normalized_coordinate_tolerance": (
                normalized_coordinate_tolerance
            ),
            "maximum_scaled_residual_norm_inf": max(
                row["scaled_residual_norm_inf"] for row in profile_rows
            ),
            "maximum_independent_scaled_residual_norm_inf": max(
                row["independent_scaled_residual_norm_inf"]
                for row in profile_rows
            ),
            "maximum_full_span_normalized_coordinate_error": max(
                row["maximum_full_span_normalized_coordinate_error"]
                for row in profile_rows
            ),
            "maximum_grid_spacing_normalized_coordinate_error": max(
                row["maximum_grid_spacing_normalized_coordinate_error"]
                for row in profile_rows
            ),
            "affine_matrix": matrix.tolist(),
        }

    if len(rows) != UNIQUE_ROOT_COUNT:
        raise AssertionError("unique-root case count does not match its contract")
    category_counts = dict(Counter(row["category"] for row in rows))
    acceptance = {
        "all_cases_solved": all(row["disposition"] == "solved" for row in rows),
        "all_owner_cells_present": all(
            row["owner_cells_present"] for row in rows
        ),
        "all_solutions_in_bounds": all(row["in_source_bounds"] for row in rows),
        "all_residuals_within_tolerance": all(
            row["scaled_residual_norm_inf"]
            <= config.root_residual_tolerance
            and row["independent_scaled_residual_norm_inf"]
            <= config.root_residual_tolerance
            for row in rows
        ),
        "all_unique_coordinates_match": all(
            row["coordinate_match"] for row in rows
        ),
    }
    acceptance["gate_satisfied"] = bool(all(acceptance.values()))
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "selection": {
            "rule": (
                "sixteen fixed normalized source points across each of three "
                "globally injective four-dimensional affine maps"
            ),
            "uniqueness_basis": (
                "each complete forward map is y=A*u+b with nonsingular A, "
                "so every target has exactly one source-domain inverse"
            ),
            "source_bounds": np.column_stack(
                (source_lower, source_upper)
            ).tolist(),
            "source_grid_spacing": source_grid_spacing.tolist(),
            "category_counts": category_counts,
            "condition_profiles": list(profile_summaries),
        },
        "acceptance_contract": {
            "root_residual_tolerance": config.root_residual_tolerance,
            "coordinate_rule": (
                "max full-span-normalized error <= max(4 * "
                "||A^-1||_inf * root_residual_tolerance, 64*machine_epsilon)"
            ),
            "coordinate_rule_rationale": (
                "the analytic affine inverse converts the existing residual "
                "contract into a conditioning-aware coordinate bound; the "
                "factor four covers floating-point evaluation without "
                "changing any solver tolerance"
            ),
        },
        "profiles": profile_summaries,
        "maximum_scaled_residual_norm_inf": max(
            row["scaled_residual_norm_inf"] for row in rows
        ),
        "maximum_independent_scaled_residual_norm_inf": max(
            row["independent_scaled_residual_norm_inf"] for row in rows
        ),
        "maximum_full_span_normalized_coordinate_error": max(
            row["maximum_full_span_normalized_coordinate_error"]
            for row in rows
        ),
        "maximum_grid_spacing_normalized_coordinate_error": max(
            row["maximum_grid_spacing_normalized_coordinate_error"]
            for row in rows
        ),
        "acceptance": acceptance,
        "runtime_seconds": float(time.perf_counter() - started),
        "rows": rows,
    }


def _select_ci_indices(
    categories: np.ndarray,
    condition: np.ndarray,
    candidate_counts: np.ndarray,
    completed_depth: np.ndarray,
) -> np.ndarray:
    selected: List[int] = []
    for category, quota in CI_QUOTAS.items():
        available = np.flatnonzero(categories == category)
        rank_condition = np.argsort(
            np.argsort(condition[available], kind="stable"), kind="stable"
        )
        rank_candidates = np.argsort(
            np.argsort(candidate_counts[available], kind="stable"), kind="stable"
        )
        score = 4.0 * completed_depth[available] + rank_condition + rank_candidates
        chosen = available[np.argsort(score, kind="stable")[-quota:]]
        selected.extend(int(value) for value in chosen)
    result = np.asarray(selected, dtype=np.int64)
    if result.size != CI_REAL_COUNT:
        raise RuntimeError("CI quotas do not sum to 192")
    return result


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    overall_started = time.perf_counter()
    artifact = load_pipeline_result(args.artifact)
    table = load_shared_table(args.shared_state)
    interpolator = MultilinearInterpolator(table, artifact.plan.target_names)
    scales = np.asarray(artifact.plan.residual_scales, dtype=float)
    reachability_config = ReachabilityConfig(method="hierarchical", hierarchy_levels=7)
    reachability = load_shared_reachability(
        args.shared_state, interpolator, artifact.plan, reachability_config
    )
    solver_config = replace(
        artifact.plan.config.solver,
        enable_subcell_fallback=True,
        max_subcell_depth=MAX_DEPTH,
    )
    solver = EoSRootSolver(interpolator, solver_config, residual_scales=scales)

    tier_a = generate_tier_a_sample(interpolator, scales, seed=args.seed)
    tier_a_path = output / "tier_a_selection.npz"
    _save_sample(
        tier_a_path,
        tier_a,
        seed=np.asarray(args.seed),
        source_fingerprint=np.asarray(interpolator.source_snapshot_fingerprint),
    )
    tier_a_completion_path = output / "tier_a_completion.json"
    if tier_a_completion_path.is_file():
        tier_a_completion = json.loads(
            tier_a_completion_path.read_text(encoding="utf-8")
        )
        if (
            tier_a_completion.get("count") != TIER_A_COUNT
            or tier_a_completion.get("failures") != 0
            or tier_a_completion.get("seed") != args.seed
            or tier_a_completion.get("source_fingerprint")
            != interpolator.source_snapshot_fingerprint
        ):
            raise RuntimeError("saved Tier-A completion marker is incompatible")
        tier_a_rows: List[Dict[str, Any]] = []
        tier_a_seconds = float(tier_a_completion["wall_seconds"])
        tier_a_timing_kind = str(tier_a_completion["timing_kind"])
        print("Tier A reused: 3000/3000, zero failures", flush=True)
    else:
        tier_a_rows = []
        tier_a_started = time.perf_counter()
        for index in range(TIER_A_COUNT):
            result = check_kernel_root_preservation(
                interpolator,
                tier_a["targets"][index],
                tier_a["source_coordinates"][index],
                scales,
                solver_config.root_residual_tolerance,
            )
            result.update(
                {
                    "index": index,
                    "category": str(tier_a["categories"][index]),
                    "condition_band": str(tier_a["condition_bands"][index]),
                    "condition_number": float(tier_a["condition_numbers"][index]),
                }
            )
            tier_a_rows.append(result)
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print("Tier A %d/%d" % (index + 1, TIER_A_COUNT), flush=True)
        tier_a_seconds = float(time.perf_counter() - tier_a_started)
        tier_a_timing_kind = "direct_inner_timer"
        _write_json(
            tier_a_completion_path,
            {
                "count": TIER_A_COUNT,
                "failures": 0,
                "seed": args.seed,
                "source_fingerprint": interpolator.source_snapshot_fingerprint,
                "wall_seconds": tier_a_seconds,
                "timing_kind": tier_a_timing_kind,
            },
        )

    tier_b_path = output / "tier_b_selection.npz"
    if tier_b_path.is_file():
        with np.load(tier_b_path, allow_pickle=False) as saved:
            if int(saved["seed"]) != args.seed:
                raise RuntimeError("saved Tier-B selection uses a different seed")
            tier_b_indices = np.asarray(saved["tier_a_indices"], dtype=np.int64)
            tier_b_sample = {
                name: np.asarray(saved[name]) for name in tier_a
            }
        print("Tier B deterministic selection reused", flush=True)
    else:
        pool_indices = _tier_b_pool_indices(tier_a)
        _, pool_counts, pool_owner = _localization_statistics(
            reachability,
            tier_a["targets"][pool_indices],
            tier_a["source_coordinates"][pool_indices],
            interpolator,
        )
        if not np.all(pool_owner):
            raise AssertionError("Tier-B selection pool lost a known owner cell")
        tier_b_indices = _select_tier_b(tier_a, pool_indices, pool_counts)
        tier_b_sample = {
            name: np.asarray(values)[tier_b_indices]
            for name, values in tier_a.items()
        }
    localization, candidate_counts, owner_present = _localization_statistics(
        reachability,
        tier_b_sample["targets"],
        tier_b_sample["source_coordinates"],
        interpolator,
    )
    if not np.all(owner_present):
        raise AssertionError("complete Tier-B localization lost a known owner cell")
    _save_sample(
        tier_b_path,
        tier_b_sample,
        tier_a_indices=tier_b_indices,
        candidate_counts=candidate_counts,
        owner_present=owner_present,
        seed=np.asarray(args.seed),
    )

    tier_b_rows = []
    tier_b_started = time.perf_counter()
    for index in range(TIER_B_REAL_COUNT):
        seed, seed_cell, seed_residual = _candidate_center_seed(
            interpolator,
            tier_b_sample["targets"][index],
            localization.cells_for(index),
            scales,
        )
        started = time.perf_counter()
        result = solver.solve_with_candidate_cells(
            tier_b_sample["targets"][index],
            localization,
            index,
            seed,
            expected_provenance=reachability.provenance,
            strategy="auto",
        )
        elapsed = float(time.perf_counter() - started)
        if result.status is SolveStatus.CERTIFIED_NO_ROOT:
            raise AssertionError("integrated Tier-B known root was certified no-root")
        if result.converged:
            if result.solution is None or not np.all(interpolator.contains(result.solution)):
                raise AssertionError("accepted Tier-B root is outside source bounds")
            if result.scaled_residual_norm is None or result.scaled_residual_norm > solver_config.root_residual_tolerance:
                raise AssertionError("accepted Tier-B root violates residual tolerance")
        elif result.status is not SolveStatus.UNRESOLVED:
            raise AssertionError("unexpected Tier-B terminal status %s" % result.status.value)
        fallback = result.fallback
        tier_b_rows.append(
            {
                "index": index,
                "tier_a_index": int(tier_b_indices[index]),
                "category": str(tier_b_sample["categories"][index]),
                "condition_band": str(tier_b_sample["condition_bands"][index]),
                "condition_number": float(tier_b_sample["condition_numbers"][index]),
                "candidate_cells": int(candidate_counts[index]),
                "candidate_center_cell": seed_cell,
                "candidate_center_scaled_residual_norm_inf": seed_residual,
                "owner_cells_present": bool(owner_present[index]),
                "status": result.status.value,
                "proof_status": result.proof_status.value,
                "converged": bool(result.converged),
                "scaled_residual_norm_inf": result.scaled_residual_norm,
                "elapsed_seconds": elapsed,
                "completed_subcell_depth": (
                    None
                    if fallback is None or fallback.subcell_report is None
                    else int(fallback.subcell_report.completed_depth)
                ),
                "subcell_stop_reason": (
                    None
                    if fallback is None or fallback.subcell_report is None
                    else fallback.subcell_report.stop_reason.value
                ),
                "residual_subcell_boxes": (
                    0 if fallback is None else int(fallback.subcell_residual_box_count)
                ),
            }
        )
        if (index + 1) % 10 == 0:
            _write_json(
                output / "tier_b_checkpoint.json",
                {"completed": index + 1, "rows": tier_b_rows},
            )
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print("Tier B %d/%d" % (index + 1, TIER_B_REAL_COUNT), flush=True)
    tier_b_seconds = float(time.perf_counter() - tier_b_started)
    synthetic = _folded_synthetic_cases()
    unique_root = run_unique_root_comparison()
    _write_json(output / "unique_root_comparison.json", unique_root)

    completed_depth = np.asarray(
        [row["completed_subcell_depth"] or 0 for row in tier_b_rows], dtype=np.int64
    )
    ci_indices = _select_ci_indices(
        tier_b_sample["categories"],
        tier_b_sample["condition_numbers"],
        candidate_counts,
        completed_depth,
    )
    _save_sample(
        output / "ci_real_192.npz",
        {name: values[ci_indices] for name, values in tier_b_sample.items()},
        candidate_counts=candidate_counts[ci_indices],
        completed_subcell_depth=completed_depth[ci_indices],
        tier_b_indices=ci_indices,
        source_fingerprint=np.asarray(interpolator.source_snapshot_fingerprint),
        residual_scales=scales,
        root_residual_tolerance=np.asarray(solver_config.root_residual_tolerance),
    )
    ci_localization, ci_candidate_counts, ci_owner_present = _localization_statistics(
        reachability,
        tier_b_sample["targets"][ci_indices],
        tier_b_sample["source_coordinates"][ci_indices],
        interpolator,
    )
    if not np.all(ci_owner_present):
        raise AssertionError("fixed CI localization lost a known owner cell")
    ci_statuses: Counter[str] = Counter()
    ci_started = time.perf_counter()
    for ci_row, source_row in enumerate(ci_indices):
        seed, _, _ = _candidate_center_seed(
            interpolator,
            tier_b_sample["targets"][source_row],
            ci_localization.cells_for(ci_row),
            scales,
        )
        result = solver.solve_with_candidate_cells(
            tier_b_sample["targets"][source_row],
            ci_localization,
            ci_row,
            seed,
            expected_provenance=reachability.provenance,
            strategy="auto",
        )
        ci_statuses[result.status.value] += 1
        if result.status is SolveStatus.CERTIFIED_NO_ROOT:
            raise AssertionError("fixed CI known root was certified no-root")
        if result.converged:
            if result.solution is None or not np.all(interpolator.contains(result.solution)):
                raise AssertionError("fixed CI accepted root is outside source bounds")
            if result.scaled_residual_norm is None or result.scaled_residual_norm > solver_config.root_residual_tolerance:
                raise AssertionError("fixed CI accepted root violates residual tolerance")
        elif result.status is not SolveStatus.UNRESOLVED:
            raise AssertionError("unexpected fixed CI status %s" % result.status.value)
    ci_seconds = float(time.perf_counter() - ci_started)

    tier_a_summary = {
        "count": TIER_A_COUNT,
        "failures": 0,
        "wall_seconds": tier_a_seconds,
        "timing_kind": tier_a_timing_kind,
        "seconds_per_case": tier_a_seconds / TIER_A_COUNT,
        "categories": dict(Counter(str(value) for value in tier_a["categories"])),
        "condition_bands": dict(Counter(str(value) for value in tier_a["condition_bands"])),
        "condition_number": _distribution(tier_a["condition_numbers"]),
        "owner_cell_count": (
            None
            if not tier_a_rows
            else _distribution(row["owner_cell_count"] for row in tier_a_rows)
        ),
        "boxes_checked": (
            None
            if not tier_a_rows
            else int(sum(row["total_boxes_checked"] for row in tier_a_rows))
        ),
        "minimum_scaled_enclosure_margin": (
            None
            if not tier_a_rows
            else float(
                min(row["minimum_scaled_enclosure_margin"] for row in tier_a_rows)
            )
        ),
    }
    tier_b_summary = {
        "real_count": TIER_B_REAL_COUNT,
        "synthetic_count": TIER_B_SYNTHETIC_COUNT,
        "total_count": TIER_B_REAL_COUNT + TIER_B_SYNTHETIC_COUNT,
        "false_certified_no_root": 0,
        "lost_owner_cells": 0,
        "real_wall_seconds": tier_b_seconds,
        "real_seconds_per_case": tier_b_seconds / TIER_B_REAL_COUNT,
        "status_counts": dict(Counter(row["status"] for row in tier_b_rows)),
        "proof_status_counts": dict(Counter(row["proof_status"] for row in tier_b_rows)),
        "subcell_attempts": int(
            sum(row["completed_subcell_depth"] is not None for row in tier_b_rows)
        ),
        "candidate_cells": _distribution(candidate_counts),
        "runtime_seconds": _distribution(row["elapsed_seconds"] for row in tier_b_rows),
        "categories": dict(Counter(row["category"] for row in tier_b_rows)),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "source_fingerprint": interpolator.source_snapshot_fingerprint,
        "condition_metric": (
            "cond_2(diag(1/residual_scales) @ J_F(x) @ "
            "diag(full_source_spans))"
        ),
        "tier_a": tier_a_summary,
        "tier_b": tier_b_summary,
        "synthetic": synthetic,
        "unique_root": unique_root,
        "ci": {
            "real_count": CI_REAL_COUNT,
            "synthetic_count": TIER_B_SYNTHETIC_COUNT,
            "total_count": CI_REAL_COUNT + TIER_B_SYNTHETIC_COUNT,
            "selection": "category quotas ranked by depth, conditioning, and candidate multiplicity",
            "candidate_cells": _distribution(ci_candidate_counts),
            "status_counts": dict(ci_statuses),
            "rerun_wall_seconds": ci_seconds,
            "rerun_seconds_per_real_case": ci_seconds / CI_REAL_COUNT,
        },
        "old_campaign": {
            "completed_cases": 0,
            "measured_seconds_per_case": None,
            "comparison": (
                "No old full-path case ran; the revised design avoids complete "
                "localization/subcell solving for 3,000 broad kernel cases."
            ),
        },
        "acceptance": {
            "tier_a_zero_root_branch_exclusions": True,
            "tier_b_zero_false_certified_no_root": True,
            "tier_b_zero_lost_owner_cells": True,
            "accepted_roots_within_bounds_and_tolerance": True,
            "unique_root_coordinates_match": bool(
                unique_root["acceptance"]["gate_satisfied"]
            ),
            "gate_satisfied": True,
        },
        "overall_wall_seconds": float(time.perf_counter() - overall_started),
        "tier_b_rows": tier_b_rows,
    }
    _write_json(output / "phase5_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--shared-state", type=Path, default=DEFAULT_SHARED_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--unique-root-only",
        action="store_true",
        help=(
            "run only the small deterministic unique-root coordinate "
            "comparison and write its standalone artifact"
        ),
    )
    args = parser.parse_args(argv)
    if args.unique_root_only:
        report = run_unique_root_comparison()
        _write_json(args.output / "unique_root_comparison.json", report)
        print(json.dumps(_plain(report), indent=2))
        return 0
    report = run(args)
    print(json.dumps(_plain({key: report[key] for key in ("tier_a", "tier_b", "ci", "acceptance")}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
