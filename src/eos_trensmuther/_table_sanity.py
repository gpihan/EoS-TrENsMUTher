"""Bounded local-geometry diagnostics for validated forward tables.

This module is deliberately diagnostic.  It does not certify global
injectivity and it is not used by interpolation, reachability, or inversion.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_MAX_SAMPLED_CELLS = 50_000
DEFAULT_MAX_REFINEMENT_CELLS = 10_000
DEFAULT_CONDITION_WARNING = 1.0e12
_DETERMINANT_EPSILON_FACTOR = 64.0

POSITIVE = "regular_positive"
NEGATIVE = "regular_negative"
SINGULAR = "sign_changing_or_singular"
INVALID = "invalid"


def _determinant_classification(
    determinants: np.ndarray, column_norm_products: np.ndarray
) -> np.ndarray:
    determinants = np.asarray(determinants, dtype=float)
    scales = np.asarray(column_norm_products, dtype=float)
    invalid = ~np.isfinite(determinants) | ~np.isfinite(scales)
    thresholds = _DETERMINANT_EPSILON_FACTOR * np.finfo(float).eps * scales
    result = np.full(determinants.shape, 0, dtype=np.int8)
    result[determinants > thresholds] = 1
    result[determinants < -thresholds] = -1
    result[invalid] = 2
    return result


def _component_summary(mask: np.ndarray) -> Tuple[int, int]:
    """Return connected-component and singleton counts using a one-cell halo."""

    pending = {tuple(int(value) for value in row) for row in np.argwhere(mask)}
    components = 0
    singletons = 0
    offsets = tuple(
        offset
        for offset in itertools.product((-1, 0, 1), repeat=mask.ndim)
        if any(offset)
    )
    while pending:
        start = pending.pop()
        stack = [start]
        size = 1
        while stack:
            current = stack.pop()
            for offset in offsets:
                neighbor = tuple(
                    current[axis] + offset[axis] for axis in range(mask.ndim)
                )
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
                    size += 1
        components += 1
        if size == 1:
            singletons += 1
    return components, singletons


def _location(
    cell: Sequence[int], axes: Sequence[np.ndarray], **extra: Any
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "cell_index": [int(value) for value in cell],
        "source_bounds": [
            [float(axis[index]), float(axis[index + 1])]
            for axis, index in zip(axes, cell)
        ],
    }
    result.update(extra)
    return result


def _exact_one_dimensional(
    axes: Sequence[np.ndarray], targets: Sequence[np.ndarray]
) -> Dict[str, Any]:
    axis = np.asarray(axes[0], dtype=float)
    values = np.asarray(targets[0], dtype=float)
    derivative = np.diff(values) / np.diff(axis)
    norms = np.abs(derivative)
    codes = _determinant_classification(derivative, norms)
    invalid = codes == 2
    positive = codes == 1
    negative = codes == -1
    singular = codes == 0
    locations = [
        _location((int(index),), axes, derivative=float(derivative[index]))
        for index in np.flatnonzero(singular | invalid)[:32]
    ]
    return {
        "strategy": "exhaustive_exact_1d",
        "deterministic": True,
        "injectivity_proof": False,
        "total_cells": int(derivative.size),
        "sampled_cells": int(derivative.size),
        "maximum_sampled_cells": int(derivative.size),
        "refined_cells": 0,
        "counts": {
            POSITIVE: int(np.count_nonzero(positive)),
            NEGATIVE: int(np.count_nonzero(negative)),
            SINGULAR: int(np.count_nonzero(singular)),
            INVALID: int(np.count_nonzero(invalid)),
            "ill_conditioned": 0,
        },
        "suspicious_components": _component_summary(singular | invalid)[0],
        "isolated_defects": _component_summary(singular | invalid)[1],
        "orientation_transition_pairs": int(
            np.count_nonzero(codes[:-1] * codes[1:] == -1)
        ),
        "locations": locations,
    }


def _exact_two_dimensional(
    axes: Sequence[np.ndarray], targets: Sequence[np.ndarray]
) -> Dict[str, Any]:
    x = np.asarray(axes[0], dtype=float)
    y = np.asarray(axes[1], dtype=float)
    dx = np.diff(x)[:, None]
    dy = np.diff(y)[None, :]
    corner_determinants = []
    corner_scales = []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for u, v in ((0, 0), (1, 0), (0, 1), (1, 1)):
            columns = []
            for values in targets:
                field = np.asarray(values, dtype=float)
                d_x_low = (field[1:, :-1] - field[:-1, :-1]) / dx
                d_x_high = (field[1:, 1:] - field[:-1, 1:]) / dx
                d_y_low = (field[:-1, 1:] - field[:-1, :-1]) / dy
                d_y_high = (field[1:, 1:] - field[1:, :-1]) / dy
                derivative_x = d_x_low if v == 0 else d_x_high
                derivative_y = d_y_low if u == 0 else d_y_high
                columns.append((derivative_x, derivative_y))
            first_x, first_y = columns[0]
            second_x, second_y = columns[1]
            determinant = first_x * second_y - first_y * second_x
            first_norm = np.hypot(first_x, second_x)
            second_norm = np.hypot(first_y, second_y)
            corner_determinants.append(determinant)
            corner_scales.append(first_norm * second_norm)

    determinants = np.stack(corner_determinants, axis=-1)
    scales = np.stack(corner_scales, axis=-1)
    corner_codes = _determinant_classification(determinants, scales)
    invalid = np.any(corner_codes == 2, axis=-1)
    positive = (~invalid) & np.all(corner_codes == 1, axis=-1)
    negative = (~invalid) & np.all(corner_codes == -1, axis=-1)
    singular = ~(invalid | positive | negative)
    suspicious = singular | invalid
    components, singletons = _component_summary(suspicious)
    locations = []
    for row in np.argwhere(suspicious)[:32]:
        cell = tuple(int(value) for value in row)
        values = determinants[cell]
        locations.append(
            _location(
                cell,
                axes,
                determinant_minimum=float(np.min(values)),
                determinant_maximum=float(np.max(values)),
                classification=(INVALID if invalid[cell] else SINGULAR),
            )
        )
    return {
        "strategy": "exhaustive_exact_bilinear_2d",
        "deterministic": True,
        "injectivity_proof": False,
        "total_cells": int(positive.size),
        "sampled_cells": int(positive.size),
        "maximum_sampled_cells": int(positive.size),
        "refined_cells": 0,
        "counts": {
            POSITIVE: int(np.count_nonzero(positive)),
            NEGATIVE: int(np.count_nonzero(negative)),
            SINGULAR: int(np.count_nonzero(singular)),
            INVALID: int(np.count_nonzero(invalid)),
            "ill_conditioned": 0,
        },
        "suspicious_components": components,
        "isolated_defects": singletons,
        "orientation_transition_pairs": 0,
        "locations": locations,
    }


def _sample_axis_counts(cell_shape: Tuple[int, ...], limit: int) -> Tuple[int, ...]:
    counts = [1 if size == 1 else 2 for size in cell_shape]
    while True:
        candidates = []
        current_product = math.prod(counts)
        for axis, size in enumerate(cell_shape):
            if counts[axis] >= size:
                continue
            candidate_product = current_product // counts[axis] * (counts[axis] + 1)
            if candidate_product <= limit:
                candidates.append((float(size) / counts[axis], -axis, axis))
        if not candidates:
            break
        counts[max(candidates)[2]] += 1
    return tuple(counts)


def _regular_sample(
    cell_shape: Tuple[int, ...], limit: int
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], Tuple[int, ...]]:
    counts = _sample_axis_counts(cell_shape, limit)
    sampled_axes = tuple(
        np.unique(np.linspace(0, size - 1, count, dtype=np.int64))
        for size, count in zip(cell_shape, counts)
    )
    mesh = np.meshgrid(*sampled_axes, indexing="ij")
    cells = np.stack([values.reshape(-1) for values in mesh], axis=1)
    return cells, sampled_axes, tuple(len(values) for values in sampled_axes)


def _center_jacobians(
    cells: np.ndarray,
    axes: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> np.ndarray:
    dimension = len(axes)
    result = np.empty((cells.shape[0], dimension, dimension), dtype=float)
    for derivative_axis in range(dimension):
        other_axes = [axis for axis in range(dimension) if axis != derivative_axis]
        offsets = tuple(itertools.product((0, 1), repeat=len(other_axes)))
        for target_index, values in enumerate(targets):
            field = np.asarray(values, dtype=float)
            difference = np.zeros(cells.shape[0], dtype=float)
            for offset in offsets:
                low = [cells[:, axis].copy() for axis in range(dimension)]
                for axis, increment in zip(other_axes, offset):
                    low[axis] += increment
                high = list(low)
                high[derivative_axis] = high[derivative_axis] + 1
                difference += field[tuple(high)] - field[tuple(low)]
            difference /= float(len(offsets))
            coordinate = np.asarray(axes[derivative_axis], dtype=float)
            spacing = (
                coordinate[cells[:, derivative_axis] + 1]
                - coordinate[cells[:, derivative_axis]]
            )
            result[:, target_index, derivative_axis] = difference / spacing
    return result


def _sample_classification(jacobians: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        determinants = np.linalg.det(jacobians)
        column_norms = np.linalg.norm(jacobians, axis=1)
        scale = np.prod(column_norms, axis=1)
        conditions = np.linalg.cond(jacobians)
    return _determinant_classification(determinants, scale), conditions


def _bounded_high_dimensional(
    axes: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    max_sampled_cells: int,
    max_refinement_cells: int,
    condition_warning: float,
) -> Dict[str, Any]:
    cell_shape = tuple(int(axis.size - 1) for axis in axes)
    total_cells = int(math.prod(cell_shape))
    if total_cells <= max_sampled_cells:
        flat = np.arange(total_cells, dtype=np.int64)
        initial = np.column_stack(np.unravel_index(flat, cell_shape))
        sampled_axes = tuple(np.arange(size, dtype=np.int64) for size in cell_shape)
        sample_shape = cell_shape
        strategy = "exhaustive_cell_center"
    else:
        initial, sampled_axes, sample_shape = _regular_sample(
            cell_shape, max_sampled_cells
        )
        strategy = "bounded_deterministic_center_sample_with_refinement"

    initial_jacobians = _center_jacobians(initial, axes, targets)
    initial_codes, initial_conditions = _sample_classification(initial_jacobians)
    classification_grid = initial_codes.reshape(sample_shape)
    # Refinement itself must remain bounded even for a deliberately
    # checkerboard/pathological orientation pattern.  One seed can generate
    # at most a complete one-cell halo (3**dimension cells), so cap seeds
    # before constructing any Python sets.
    maximum_refinement_seeds = max(
        1, max_refinement_cells // (3 ** len(cell_shape))
    )
    seed_cells = []
    candidate_refinement_seeds = 0
    suspicious = (initial_codes == 0) | (initial_codes == 2) | (
        initial_conditions > condition_warning
    )
    suspicious_indices = np.flatnonzero(suspicious)
    candidate_refinement_seeds += int(suspicious_indices.size)
    seed_cells.extend(
        initial[suspicious_indices[:maximum_refinement_seeds]]
    )
    orientation_transition_pairs = 0
    for axis in range(len(cell_shape)):
        left_selector = [slice(None)] * len(cell_shape)
        right_selector = [slice(None)] * len(cell_shape)
        left_selector[axis] = slice(0, -1)
        right_selector[axis] = slice(1, None)
        left = classification_grid[tuple(left_selector)]
        right = classification_grid[tuple(right_selector)]
        transitions = (left * right) == -1
        transition_positions = np.argwhere(transitions)
        orientation_transition_pairs += int(transition_positions.shape[0])
        candidate_refinement_seeds += 3 * int(transition_positions.shape[0])
        for sampled_position in transition_positions:
            if len(seed_cells) >= maximum_refinement_seeds:
                break
            lower_position = sampled_position.copy()
            upper_position = sampled_position.copy()
            upper_position[axis] += 1
            lower_cell = np.asarray(
                [sampled_axes[i][lower_position[i]] for i in range(len(cell_shape))]
            )
            upper_cell = np.asarray(
                [sampled_axes[i][upper_position[i]] for i in range(len(cell_shape))]
            )
            for seed in (lower_cell, upper_cell, (lower_cell + upper_cell) // 2):
                if len(seed_cells) >= maximum_refinement_seeds:
                    break
                seed_cells.append(seed)

    refinement = set()
    offsets = tuple(itertools.product((-1, 0, 1), repeat=len(cell_shape)))
    for seed in seed_cells:
        for offset in offsets:
            cell = tuple(int(seed[axis]) + offset[axis] for axis in range(len(cell_shape)))
            if all(0 <= cell[axis] < cell_shape[axis] for axis in range(len(cell_shape))):
                refinement.add(cell)
    initial_set = {tuple(int(value) for value in row) for row in initial}
    refinement.difference_update(initial_set)
    refinement_cells = sorted(refinement)[:max_refinement_cells]
    if refinement_cells:
        refinement_array = np.asarray(refinement_cells, dtype=np.int64)
        refinement_jacobians = _center_jacobians(refinement_array, axes, targets)
        refinement_codes, refinement_conditions = _sample_classification(
            refinement_jacobians
        )
        cells = np.vstack((initial, refinement_array))
        codes = np.concatenate((initial_codes, refinement_codes))
        conditions = np.concatenate((initial_conditions, refinement_conditions))
    else:
        cells = initial
        codes = initial_codes
        conditions = initial_conditions

    # A singular matrix legitimately has an infinite condition number.  It is
    # suspicious geometry, not an invalid/non-finite derivative evaluation.
    # Reserve INVALID for a non-finite determinant or derivative scale.
    invalid = codes == 2
    singular = (codes == 0) & ~invalid
    ill_conditioned = (~invalid) & (
        ~np.isfinite(conditions) | (conditions > condition_warning)
    )
    locations = []
    flagged = np.flatnonzero(invalid | singular | ill_conditioned)
    for index in flagged[:32]:
        locations.append(
            _location(
                cells[index],
                axes,
                classification=(INVALID if invalid[index] else SINGULAR),
                condition_number=(
                    float(conditions[index])
                    if np.isfinite(conditions[index])
                    else None
                ),
            )
        )
    return {
        "strategy": strategy,
        "deterministic": True,
        "injectivity_proof": False,
        "total_cells": total_cells,
        "sampled_cells": int(cells.shape[0]),
        "initial_sampled_cells": int(initial.shape[0]),
        "maximum_sampled_cells": int(max_sampled_cells + max_refinement_cells),
        "refined_cells": int(len(refinement_cells)),
        "refinement_truncated": (
            candidate_refinement_seeds > len(seed_cells)
            or len(refinement) > max_refinement_cells
        ),
        "candidate_refinement_seeds": candidate_refinement_seeds,
        "used_refinement_seeds": len(seed_cells),
        "counts": {
            POSITIVE: int(np.count_nonzero(codes == 1)),
            NEGATIVE: int(np.count_nonzero(codes == -1)),
            SINGULAR: int(np.count_nonzero(singular)),
            INVALID: int(np.count_nonzero(invalid)),
            "ill_conditioned": int(np.count_nonzero(ill_conditioned)),
        },
        "suspicious_components": None,
        "isolated_defects": None,
        "orientation_transition_pairs": orientation_transition_pairs,
        "condition_warning": float(condition_warning),
        "locations": locations,
    }


def scan_table_geometry(
    axes: Sequence[np.ndarray],
    target_fields: Mapping[str, np.ndarray],
    target_names: Sequence[str],
    *,
    max_sampled_cells: int = DEFAULT_MAX_SAMPLED_CELLS,
    max_refinement_cells: int = DEFAULT_MAX_REFINEMENT_CELLS,
    condition_warning: float = DEFAULT_CONDITION_WARNING,
) -> Dict[str, Any]:
    """Return deterministic local-Jacobian diagnostics for one forward table."""

    axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
    names = tuple(target_names)
    dimension = len(axes)
    if dimension not in (1, 2, 3, 4) or len(names) != dimension:
        raise ValueError("geometry scan requires equal source/target dimension 1--4")
    shape = tuple(axis.size for axis in axes)
    targets = tuple(np.asarray(target_fields[name], dtype=float) for name in names)
    if any(values.shape != shape for values in targets):
        raise ValueError("geometry scan target fields do not match the source grid")
    if dimension == 1:
        return _exact_one_dimensional(axes, targets)
    if dimension == 2:
        return _exact_two_dimensional(axes, targets)
    return _bounded_high_dimensional(
        axes,
        targets,
        max_sampled_cells,
        max_refinement_cells,
        condition_warning,
    )


__all__ = [
    "DEFAULT_CONDITION_WARNING",
    "DEFAULT_MAX_REFINEMENT_CELLS",
    "DEFAULT_MAX_SAMPLED_CELLS",
    "INVALID",
    "NEGATIVE",
    "POSITIVE",
    "SINGULAR",
    "scan_table_geometry",
]
