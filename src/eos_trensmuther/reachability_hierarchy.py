"""Proof-safe source-topology hierarchy for ordinary 4-D reachability.

The module owns immutable hierarchy construction, routing-only profiling, and
complete leaf resolution.  ``ReachabilityIndex`` selects it explicitly for a
full active 4-D cell lattice; exhaustive fine-cell semantics in
``reachability.py`` remain the correctness oracle and default method.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._array_storage import immutable_array as _storage_immutable_array

SUPPORTED_HIERARCHY_LEVELS = (3, 5, 7)
DEFAULT_MAX_HIERARCHY_BYTES = 2 * 1024**3


class ReachabilityHierarchyError(ValueError):
    """Raised when a hierarchy cannot be constructed proof-safely."""


class HierarchyPairLimitError(ReachabilityHierarchyError):
    """Raised when a vectorized traversal exceeds its bounded pair storage."""


def _immutable_array(value: Any, dtype: Any = None) -> np.ndarray:
    return _storage_immutable_array(value, dtype=dtype)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ReachabilityHierarchyError("%s must be a positive integer" % name)
    converted = int(value)
    if converted <= 0:
        raise ReachabilityHierarchyError("%s must be a positive integer" % name)
    return converted


def _cell_shape(value: Sequence[int]) -> Tuple[int, int, int, int]:
    try:
        shape = tuple(value)
    except TypeError as exc:
        raise ReachabilityHierarchyError(
            "cell_shape must contain four positive integers"
        ) from exc
    if len(shape) != 4:
        raise ReachabilityHierarchyError(
            "the initial hierarchy requires a four-dimensional cell_shape"
        )
    return tuple(
        _positive_integer(item, "cell_shape entry") for item in shape
    )  # type: ignore[return-value]


def _axis_boundaries(cell_count: int, block_count: int) -> np.ndarray:
    """Return balanced integer block boundaries including both endpoints."""

    if block_count > cell_count:
        raise ReachabilityHierarchyError(
            "an axis cannot contain more blocks than fine cells"
        )
    numerators = np.arange(block_count + 1, dtype=np.int64) * np.int64(cell_count)
    boundaries = np.floor_divide(numerators, np.int64(block_count))
    if (
        boundaries[0] != 0
        or boundaries[-1] != cell_count
        or np.any(np.diff(boundaries) <= 0)
    ):
        raise ReachabilityHierarchyError(
            "balanced axis boundaries are not a complete positive partition"
        )
    return boundaries


def _logical_block_shape(
    cell_shape: Tuple[int, int, int, int], logical_level: int
) -> Tuple[int, int, int, int]:
    nominal = 1 << (logical_level - 1)
    return tuple(min(nominal, count) for count in cell_shape)  # type: ignore[return-value]


def _reduce_axis_blocks(
    values: np.ndarray, starts: np.ndarray, axis: int, operation: str
) -> np.ndarray:
    if operation == "minimum":
        return np.minimum.reduceat(values, starts, axis=axis)
    if operation == "maximum":
        return np.maximum.reduceat(values, starts, axis=axis)
    raise AssertionError("unknown block reduction")


def _aggregate_from_leaf(
    leaf_values: np.ndarray,
    cell_shape: Tuple[int, int, int, int],
    boundaries: Tuple[np.ndarray, ...],
    operation: str,
) -> np.ndarray:
    values = np.asarray(leaf_values, dtype=float).reshape(cell_shape + (4,), order="C")
    for axis, axis_boundaries in enumerate(boundaries):
        values = _reduce_axis_blocks(
            values, np.asarray(axis_boundaries[:-1], dtype=np.int64), axis, operation
        )
    return np.ascontiguousarray(values.reshape(-1, 4, order="C"))


def _aggregate_from_child(
    child_values: np.ndarray,
    child_shape: Tuple[int, int, int, int],
    child_boundaries: Tuple[np.ndarray, ...],
    parent_boundaries: Tuple[np.ndarray, ...],
    operation: str,
) -> np.ndarray:
    values = np.asarray(child_values, dtype=float).reshape(
        child_shape + (4,), order="C"
    )
    for axis, (child_axis, parent_axis) in enumerate(
        zip(child_boundaries, parent_boundaries)
    ):
        starts = np.searchsorted(child_axis, parent_axis[:-1], side="left")
        if not np.array_equal(child_axis[starts], parent_axis[:-1]):
            raise ReachabilityHierarchyError(
                "hierarchy axis partitions are not nested"
            )
        values = _reduce_axis_blocks(values, starts, axis, operation)
    return np.ascontiguousarray(values.reshape(-1, 4, order="C"))


def _outward_bounds(
    lower: np.ndarray, upper: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore"):
        outward_lower = np.nextafter(lower, -np.inf)
        outward_upper = np.nextafter(upper, np.inf)
    if np.any(np.isnan(outward_lower)) or np.any(np.isnan(outward_upper)):
        raise ReachabilityHierarchyError(
            "outward-rounded hierarchy bounds produced NaN"
        )
    return outward_lower, outward_upper


@dataclass(frozen=True)
class RoutingHierarchyLevel:
    """One immutable logical source-block level and its routing bounds."""

    logical_level: int
    cell_shape: Tuple[int, int, int, int]
    block_shape: Tuple[int, int, int, int]
    axis_boundaries: Tuple[np.ndarray, ...]
    routing_lower: np.ndarray
    routing_upper: np.ndarray
    aliases_leaf: bool = False

    def __post_init__(self) -> None:
        logical_level = _positive_integer(self.logical_level, "logical_level")
        cell_shape = _cell_shape(self.cell_shape)
        block_shape = _cell_shape(self.block_shape)
        if any(block > cells for block, cells in zip(block_shape, cell_shape)):
            raise ReachabilityHierarchyError(
                "block_shape cannot exceed cell_shape"
            )
        if len(self.axis_boundaries) != 4:
            raise ReachabilityHierarchyError(
                "axis_boundaries must contain four arrays"
            )
        boundaries = []
        for cells, blocks, supplied in zip(
            cell_shape, block_shape, self.axis_boundaries
        ):
            array = np.asarray(supplied)
            if (
                array.shape != (blocks + 1,)
                or not np.issubdtype(array.dtype, np.integer)
                or array[0] != 0
                or array[-1] != cells
                or np.any(np.diff(array) <= 0)
            ):
                raise ReachabilityHierarchyError(
                    "an axis boundary array is not a complete positive partition"
                )
            boundaries.append(_immutable_array(array, dtype=np.int64))
        block_count = math.prod(block_shape)
        lower = np.asarray(self.routing_lower, dtype=float)
        upper = np.asarray(self.routing_upper, dtype=float)
        if (
            lower.shape != (block_count, 4)
            or upper.shape != (block_count, 4)
            or np.any(np.isnan(lower))
            or np.any(np.isnan(upper))
            or np.any(lower > upper)
        ):
            raise ReachabilityHierarchyError(
                "routing bounds must be ordered (block_count, 4) arrays"
            )
        if not isinstance(self.aliases_leaf, (bool, np.bool_)):
            raise ReachabilityHierarchyError("aliases_leaf must be boolean")
        aliases_leaf = bool(self.aliases_leaf)
        if aliases_leaf and (
            lower.dtype != np.dtype(np.float64)
            or upper.dtype != np.dtype(np.float64)
            or not lower.flags.c_contiguous
            or not upper.flags.c_contiguous
            or lower.flags.writeable
            or upper.flags.writeable
        ):
            raise ReachabilityHierarchyError(
                "aliased leaf bounds must be immutable contiguous float64 arrays"
            )
        object.__setattr__(self, "logical_level", logical_level)
        object.__setattr__(self, "cell_shape", cell_shape)
        object.__setattr__(self, "block_shape", block_shape)
        object.__setattr__(self, "axis_boundaries", tuple(boundaries))
        object.__setattr__(
            self,
            "routing_lower",
            lower if aliases_leaf else _immutable_array(lower),
        )
        object.__setattr__(
            self,
            "routing_upper",
            upper if aliases_leaf else _immutable_array(upper),
        )
        object.__setattr__(self, "aliases_leaf", aliases_leaf)

    @property
    def block_count(self) -> int:
        return math.prod(self.block_shape)

    @property
    def bounds_nbytes(self) -> int:
        return int(self.routing_lower.nbytes + self.routing_upper.nbytes)

    @property
    def axis_nbytes(self) -> int:
        return int(sum(boundary.nbytes for boundary in self.axis_boundaries))

    @property
    def nbytes(self) -> int:
        return int(self.bounds_nbytes + self.axis_nbytes)

    def source_cell_bounds(self, block_indices: Any) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(block_indices)
        scalar = indices.shape == (4,)
        if scalar:
            indices = indices.reshape(1, 4)
        if (
            indices.ndim != 2
            or indices.shape[1] != 4
            or not np.issubdtype(indices.dtype, np.integer)
        ):
            raise ReachabilityHierarchyError(
                "block_indices must have shape (..., 4) and integer dtype"
            )
        checked = np.asarray(indices, dtype=np.int64)
        if any(
            np.any((checked[:, axis] < 0) | (checked[:, axis] >= size))
            for axis, size in enumerate(self.block_shape)
        ):
            raise ReachabilityHierarchyError("block index is outside the level")
        lower = np.column_stack(
            [
                boundary[checked[:, axis]]
                for axis, boundary in enumerate(self.axis_boundaries)
            ]
        )
        upper = np.column_stack(
            [
                boundary[checked[:, axis] + 1]
                for axis, boundary in enumerate(self.axis_boundaries)
            ]
        )
        if scalar:
            return _immutable_array(lower[0]), _immutable_array(upper[0])
        original_shape = np.asarray(block_indices).shape[:-1] + (4,)
        return (
            _immutable_array(lower.reshape(original_shape)),
            _immutable_array(upper.reshape(original_shape)),
        )


@dataclass(frozen=True)
class RoutingHierarchy:
    """Immutable requested prefix of the balanced four-dimensional tree."""

    cell_shape: Tuple[int, int, int, int]
    requested_levels: int
    levels: Tuple[RoutingHierarchyLevel, ...]
    max_hierarchy_bytes: int

    def __post_init__(self) -> None:
        cell_shape = _cell_shape(self.cell_shape)
        requested = _positive_integer(self.requested_levels, "requested_levels")
        if requested not in SUPPORTED_HIERARCHY_LEVELS:
            raise ReachabilityHierarchyError(
                "requested_levels must be one of %s"
                % (SUPPORTED_HIERARCHY_LEVELS,)
            )
        maximum = _positive_integer(
            self.max_hierarchy_bytes, "max_hierarchy_bytes"
        )
        levels = tuple(self.levels)
        if not levels or any(
            not isinstance(level, RoutingHierarchyLevel) for level in levels
        ):
            raise ReachabilityHierarchyError(
                "levels must contain RoutingHierarchyLevel values"
            )
        expected_numbers = tuple(range(1, len(levels) + 1))
        if tuple(level.logical_level for level in levels) != expected_numbers:
            raise ReachabilityHierarchyError(
                "stored hierarchy levels must be a consecutive root prefix"
            )
        if levels[-1].logical_level > requested:
            raise ReachabilityHierarchyError(
                "stored hierarchy exceeds the requested logical depth"
            )
        for level in levels:
            if level.cell_shape != cell_shape:
                raise ReachabilityHierarchyError(
                    "every hierarchy level must use the same fine cell shape"
                )
        if self.owned_nbytes > maximum:
            raise ReachabilityHierarchyError(
                "hierarchy owns %d bytes, exceeding max_hierarchy_bytes=%d"
                % (self.owned_nbytes, maximum)
            )
        object.__setattr__(self, "cell_shape", cell_shape)
        object.__setattr__(self, "requested_levels", requested)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "max_hierarchy_bytes", maximum)

    @property
    def leaf_count(self) -> int:
        return math.prod(self.cell_shape)

    @property
    def reaches_leaf(self) -> bool:
        return self.levels[-1].block_shape == self.cell_shape

    @property
    def owned_nbytes(self) -> int:
        return int(
            sum(
                level.axis_nbytes
                + (0 if level.aliases_leaf else level.bounds_nbytes)
                for level in self.levels
            )
        )

    @property
    def referenced_leaf_nbytes(self) -> int:
        return int(
            sum(level.bounds_nbytes for level in self.levels if level.aliases_leaf)
        )


@dataclass(frozen=True)
class HierarchyLevelProfile:
    """Aggregate routing work observed at one logical hierarchy level."""

    logical_level: int
    block_shape: Tuple[int, int, int, int]
    stored_blocks: int
    tested_pairs: int
    surviving_pairs: int
    distinct_targets_surviving: int
    traversal_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_level": self.logical_level,
            "block_shape": list(self.block_shape),
            "stored_blocks": self.stored_blocks,
            "tested_pairs": self.tested_pairs,
            "surviving_pairs": self.surviving_pairs,
            "distinct_targets_surviving": self.distinct_targets_surviving,
            "traversal_seconds": self.traversal_seconds,
        }


@dataclass(frozen=True)
class HierarchyCategoryProfile:
    """Residual leaf-work summary for one caller-provided target category."""

    target_count: int
    rejected_before_leaf: int
    requiring_leaf: int
    leaf_pairs_sum: int
    leaf_pairs_mean: float
    leaf_pairs_median: float
    leaf_pairs_p95: float
    leaf_pairs_maximum: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_count": self.target_count,
            "rejected_before_leaf": self.rejected_before_leaf,
            "requiring_leaf": self.requiring_leaf,
            "leaf_pairs_sum": self.leaf_pairs_sum,
            "leaf_pairs_mean": self.leaf_pairs_mean,
            "leaf_pairs_median": self.leaf_pairs_median,
            "leaf_pairs_p95": self.leaf_pairs_p95,
            "leaf_pairs_maximum": self.leaf_pairs_maximum,
        }


@dataclass(frozen=True)
class HierarchyPruningReport:
    """Non-authoritative coarse traversal report; never a classification."""

    target_count: int
    valid_target_count: int
    profiled_valid_target_count: int
    hierarchy_levels: int
    level_profiles: Tuple[HierarchyLevelProfile, ...]
    traversal_completed: bool
    complete_classification: bool
    stopped_reason: Optional[str]
    targets_rejected_before_leaf: int
    targets_requiring_leaf: int
    descendant_leaf_pairs: int
    leaf_pairs_mean: float
    leaf_pairs_median: float
    leaf_pairs_p95: float
    leaf_pairs_maximum: int
    leaf_pairs_per_second: float
    estimated_leaf_seconds: float
    maximum_estimated_leaf_seconds: float
    leaf_completion_allowed: bool
    hierarchy_construction_seconds: float
    coarse_traversal_seconds: float
    hierarchy_owned_bytes: int
    hierarchy_referenced_leaf_bytes: int
    query_batch_size: int
    maximum_pairwise_entries: int
    retry_seconds: float
    category_profiles: Mapping[str, HierarchyCategoryProfile]
    profiled_target_indices: np.ndarray
    descendant_leaf_pairs_by_profiled_target: np.ndarray
    classification_leaf_pairs: Optional[int] = None
    additional_minimum_witness_leaf_pairs: Optional[int] = None
    additional_minimum_witness_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.complete_classification:
            raise ReachabilityHierarchyError(
                "routing-only hierarchy reports cannot claim classification"
            )
        object.__setattr__(self, "level_profiles", tuple(self.level_profiles))
        indices = _immutable_array(self.profiled_target_indices, dtype=np.int64)
        leaf_pairs = _immutable_array(
            self.descendant_leaf_pairs_by_profiled_target, dtype=np.int64
        )
        if indices.ndim != 1 or leaf_pairs.shape != indices.shape:
            raise ReachabilityHierarchyError(
                "profiled target indices and leaf-pair counts must be matching vectors"
            )
        object.__setattr__(self, "profiled_target_indices", indices)
        object.__setattr__(
            self, "descendant_leaf_pairs_by_profiled_target", leaf_pairs
        )
        object.__setattr__(
            self,
            "category_profiles",
            MappingProxyType(dict(self.category_profiles)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_type": "EoS-TrENsMUTher.hierarchy_pruning",
            "target_count": self.target_count,
            "valid_target_count": self.valid_target_count,
            "profiled_valid_target_count": self.profiled_valid_target_count,
            "hierarchy_levels": self.hierarchy_levels,
            "level_profiles": [item.to_dict() for item in self.level_profiles],
            "traversal_completed": self.traversal_completed,
            "complete_classification": self.complete_classification,
            "stopped_reason": self.stopped_reason,
            "targets_rejected_before_leaf": self.targets_rejected_before_leaf,
            "targets_requiring_leaf": self.targets_requiring_leaf,
            "descendant_leaf_pairs": self.descendant_leaf_pairs,
            "profiled_target_indices": self.profiled_target_indices.tolist(),
            "descendant_leaf_pairs_by_profiled_target": (
                self.descendant_leaf_pairs_by_profiled_target.tolist()
            ),
            "leaf_pairs": {
                "mean": self.leaf_pairs_mean,
                "median": self.leaf_pairs_median,
                "p95": self.leaf_pairs_p95,
                "maximum": self.leaf_pairs_maximum,
                "classification_only": self.classification_leaf_pairs,
                "additional_minimum_witness": (
                    self.additional_minimum_witness_leaf_pairs
                ),
            },
            "leaf_pairs_per_second": self.leaf_pairs_per_second,
            "estimated_leaf_seconds": self.estimated_leaf_seconds,
            "maximum_estimated_leaf_seconds": (
                self.maximum_estimated_leaf_seconds
            ),
            "leaf_completion_allowed": self.leaf_completion_allowed,
            "timing_seconds": {
                "hierarchy_construction": self.hierarchy_construction_seconds,
                "coarse_traversal": self.coarse_traversal_seconds,
                "retry_work": self.retry_seconds,
                "additional_minimum_witness": (
                    self.additional_minimum_witness_seconds
                ),
            },
            "memory_bytes": {
                "hierarchy_owned": self.hierarchy_owned_bytes,
                "referenced_leaf": self.hierarchy_referenced_leaf_bytes,
            },
            "query_batch_size": self.query_batch_size,
            "maximum_pairwise_entries": self.maximum_pairwise_entries,
            "categories": {
                name: value.to_dict()
                for name, value in self.category_profiles.items()
            },
        }


@dataclass(frozen=True)
class _BatchTraversal:
    leaf_pairs_by_target: np.ndarray
    surviving_query_ids: np.ndarray
    surviving_block_ids: np.ndarray
    level_tested: Tuple[int, ...]
    level_surviving: Tuple[int, ...]
    level_distinct: Tuple[int, ...]
    level_seconds: Tuple[float, ...]


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ReachabilityHierarchyError("%s must be finite and non-negative" % name)
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ReachabilityHierarchyError(
            "%s must be finite and non-negative" % name
        ) from exc
    if not math.isfinite(converted) or converted < 0.0:
        raise ReachabilityHierarchyError(
            "%s must be finite and non-negative" % name
        )
    return converted


def _finite_positive(value: Any, name: str) -> float:
    converted = _finite_nonnegative(value, name)
    if converted <= 0.0:
        raise ReachabilityHierarchyError("%s must be positive" % name)
    return converted


def _validate_profile_targets(
    targets: Any, valid_mask: Optional[Any]
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(targets, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ReachabilityHierarchyError("targets must have shape (count, 4)")
    finite = np.all(np.isfinite(values), axis=1)
    if valid_mask is None:
        valid = finite
    else:
        supplied = np.asarray(valid_mask)
        if supplied.shape != (values.shape[0],) or supplied.dtype.kind != "b":
            raise ReachabilityHierarchyError(
                "valid_mask must be a boolean vector matching targets"
            )
        valid = finite & supplied
    return values, valid


def _child_ranges(
    parent: RoutingHierarchyLevel, child: RoutingHierarchyLevel
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    starts = []
    stops = []
    for parent_axis, child_axis in zip(
        parent.axis_boundaries, child.axis_boundaries
    ):
        axis_starts = np.searchsorted(
            child_axis, parent_axis[:-1], side="left"
        ).astype(np.int64)
        axis_stops = np.searchsorted(
            child_axis, parent_axis[1:], side="left"
        ).astype(np.int64)
        if (
            not np.array_equal(child_axis[axis_starts], parent_axis[:-1])
            or not np.array_equal(child_axis[axis_stops], parent_axis[1:])
            or np.any(axis_stops <= axis_starts)
            or np.any(axis_stops - axis_starts > 2)
        ):
            raise ReachabilityHierarchyError(
                "adjacent hierarchy partitions are not nested binary refinements"
            )
        starts.append(axis_starts)
        stops.append(axis_stops)
    return tuple(starts), tuple(stops)


def _contained_pairs(
    targets: np.ndarray,
    query_ids: np.ndarray,
    block_ids: np.ndarray,
    level: RoutingHierarchyLevel,
) -> np.ndarray:
    return np.all(
        (targets[query_ids] >= level.routing_lower[block_ids])
        & (targets[query_ids] <= level.routing_upper[block_ids]),
        axis=1,
    )


def _traverse_batch(
    hierarchy: RoutingHierarchy,
    targets: np.ndarray,
    maximum_pairwise_entries: int,
) -> _BatchTraversal:
    """Traverse one finite target batch or fail before unbounded allocation."""

    level_tested: List[int] = []
    level_surviving: List[int] = []
    level_distinct: List[int] = []
    level_seconds: List[float] = []

    root = hierarchy.levels[0]
    started = time.perf_counter()
    query_ids = np.arange(targets.shape[0], dtype=np.int64)
    block_ids = np.zeros(targets.shape[0], dtype=np.int64)
    keep = _contained_pairs(targets, query_ids, block_ids, root)
    query_ids = query_ids[keep]
    block_ids = block_ids[keep]
    level_tested.append(int(targets.shape[0]))
    level_surviving.append(int(query_ids.size))
    level_distinct.append(int(np.unique(query_ids).size))
    level_seconds.append(time.perf_counter() - started)

    for parent, child in zip(hierarchy.levels[:-1], hierarchy.levels[1:]):
        started = time.perf_counter()
        if query_ids.size == 0:
            level_tested.append(0)
            level_surviving.append(0)
            level_distinct.append(0)
            level_seconds.append(time.perf_counter() - started)
            continue
        parent_multi = np.column_stack(
            np.unravel_index(block_ids, parent.block_shape, order="C")
        ).astype(np.int64, copy=False)
        starts, stops = _child_ranges(parent, child)
        child_stops = np.column_stack(
            [stops[axis][parent_multi[:, axis]] for axis in range(4)]
        )
        surviving_queries: List[np.ndarray] = []
        surviving_blocks: List[np.ndarray] = []
        tested_count = 0
        surviving_count = 0
        for child_bits in range(16):
            offsets = np.asarray(
                [(child_bits >> (3 - axis)) & 1 for axis in range(4)],
                dtype=np.int64,
            )
            child_multi = np.column_stack(
                [
                    starts[axis][parent_multi[:, axis]] + offsets[axis]
                    for axis in range(4)
                ]
            )
            valid_child = np.all(child_multi < child_stops, axis=1)
            if not np.any(valid_child):
                continue
            candidate_queries = query_ids[valid_child]
            candidate_multi = child_multi[valid_child]
            candidate_blocks = np.ravel_multi_index(
                tuple(candidate_multi[:, axis] for axis in range(4)),
                child.block_shape,
                order="C",
            ).astype(np.int64, copy=False)
            tested_count += int(candidate_queries.size)
            contained = _contained_pairs(
                targets, candidate_queries, candidate_blocks, child
            )
            if np.any(contained):
                kept_queries = candidate_queries[contained]
                kept_blocks = candidate_blocks[contained]
                surviving_count += int(kept_queries.size)
                if surviving_count > maximum_pairwise_entries:
                    raise HierarchyPairLimitError(
                        "surviving query/block pairs exceed "
                        "maximum_pairwise_entries=%d"
                        % maximum_pairwise_entries
                    )
                surviving_queries.append(kept_queries)
                surviving_blocks.append(kept_blocks)
        if surviving_queries:
            query_ids = np.concatenate(surviving_queries)
            block_ids = np.concatenate(surviving_blocks)
            order = np.lexsort((block_ids, query_ids))
            query_ids = query_ids[order]
            block_ids = block_ids[order]
        else:
            query_ids = np.empty(0, dtype=np.int64)
            block_ids = np.empty(0, dtype=np.int64)
        level_tested.append(tested_count)
        level_surviving.append(int(query_ids.size))
        level_distinct.append(int(np.unique(query_ids).size))
        level_seconds.append(time.perf_counter() - started)

    leaf_pairs = np.zeros(targets.shape[0], dtype=np.int64)
    if query_ids.size:
        deepest = hierarchy.levels[-1]
        deepest_multi = np.column_stack(
            np.unravel_index(block_ids, deepest.block_shape, order="C")
        ).astype(np.int64, copy=False)
        source_lower, source_upper = deepest.source_cell_bounds(deepest_multi)
        descendants = np.prod(source_upper - source_lower, axis=1, dtype=np.int64)
        np.add.at(leaf_pairs, query_ids, descendants)
    return _BatchTraversal(
        leaf_pairs_by_target=leaf_pairs,
        surviving_query_ids=query_ids,
        surviving_block_ids=block_ids,
        level_tested=tuple(level_tested),
        level_surviving=tuple(level_surviving),
        level_distinct=tuple(level_distinct),
        level_seconds=tuple(level_seconds),
    )


@dataclass(frozen=True)
class HierarchyLeafDiagnostics:
    """Leaf-comparison cost and minimum-witness overhead diagnostics."""

    actual_leaf_pairs: int
    classification_leaf_pairs: int
    additional_minimum_witness_leaf_pairs: int
    additional_batch_leaf_pairs: int
    coarse_pairs_tested: int
    coarse_seconds: float
    leaf_seconds: float
    actual_leaf_pairs_by_target: np.ndarray
    classification_leaf_pairs_by_target: np.ndarray
    additional_minimum_witness_leaf_pairs_by_target: np.ndarray
    additional_batch_leaf_pairs_by_target: np.ndarray

    def __post_init__(self) -> None:
        names = (
            "actual_leaf_pairs_by_target",
            "classification_leaf_pairs_by_target",
            "additional_minimum_witness_leaf_pairs_by_target",
            "additional_batch_leaf_pairs_by_target",
        )
        arrays = []
        for name in names:
            array = _immutable_array(getattr(self, name), dtype=np.int64)
            if array.ndim != 1 or np.any(array < 0):
                raise ReachabilityHierarchyError(
                    "%s must be a non-negative vector" % name
                )
            arrays.append(array)
            object.__setattr__(self, name, array)
        if not all(array.shape == arrays[0].shape for array in arrays[1:]):
            raise ReachabilityHierarchyError(
                "per-target leaf diagnostic vectors must have matching shapes"
            )
        if np.any(arrays[0] != arrays[1] + arrays[2] + arrays[3]):
            raise ReachabilityHierarchyError(
                "actual leaf work must equal classification, witness, and batch work"
            )
        if (
            self.actual_leaf_pairs != int(np.sum(arrays[0], dtype=np.int64))
            or self.classification_leaf_pairs
            != int(np.sum(arrays[1], dtype=np.int64))
            or self.additional_minimum_witness_leaf_pairs
            != int(np.sum(arrays[2], dtype=np.int64))
            or self.additional_batch_leaf_pairs
            != int(np.sum(arrays[3], dtype=np.int64))
        ):
            raise ReachabilityHierarchyError(
                "aggregate and per-target leaf diagnostics disagree"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actual_leaf_pairs": self.actual_leaf_pairs,
            "classification_leaf_pairs": self.classification_leaf_pairs,
            "additional_minimum_witness_leaf_pairs": (
                self.additional_minimum_witness_leaf_pairs
            ),
            "additional_batch_leaf_pairs": self.additional_batch_leaf_pairs,
            "coarse_pairs_tested": self.coarse_pairs_tested,
            "coarse_seconds": self.coarse_seconds,
            "leaf_seconds": self.leaf_seconds,
            "actual_leaf_pairs_by_target": (
                self.actual_leaf_pairs_by_target.tolist()
            ),
            "classification_leaf_pairs_by_target": (
                self.classification_leaf_pairs_by_target.tolist()
            ),
            "additional_minimum_witness_leaf_pairs_by_target": (
                self.additional_minimum_witness_leaf_pairs_by_target.tolist()
            ),
            "additional_batch_leaf_pairs_by_target": (
                self.additional_batch_leaf_pairs_by_target.tolist()
            ),
        }


@dataclass(frozen=True)
class HierarchyLeafResolution:
    """Complete private leaf result, independent of pipeline status types."""

    valid: np.ndarray
    routing_candidate: np.ndarray
    exact_candidate: np.ndarray
    routing_witness: np.ndarray
    exact_witness: np.ndarray
    diagnostics: HierarchyLeafDiagnostics

    def __post_init__(self) -> None:
        valid = _immutable_array(self.valid, dtype=bool)
        routing = _immutable_array(self.routing_candidate, dtype=bool)
        exact = _immutable_array(self.exact_candidate, dtype=bool)
        routing_witness = _immutable_array(self.routing_witness, dtype=np.int64)
        exact_witness = _immutable_array(self.exact_witness, dtype=np.int64)
        if valid.ndim != 1 or not all(
            array.shape == valid.shape
            for array in (routing, exact, routing_witness, exact_witness)
        ):
            raise ReachabilityHierarchyError(
                "hierarchy leaf results must be matching vectors"
            )
        if (
            np.any(routing != (routing_witness >= 0))
            or np.any(exact != (exact_witness >= 0))
            or np.any(exact & ~routing)
            or np.any(~valid & (routing | exact))
        ):
            raise ReachabilityHierarchyError(
                "hierarchy leaf flags and witnesses are inconsistent"
            )
        if not isinstance(self.diagnostics, HierarchyLeafDiagnostics):
            raise ReachabilityHierarchyError(
                "diagnostics must be HierarchyLeafDiagnostics"
            )
        if self.diagnostics.actual_leaf_pairs_by_target.shape != valid.shape:
            raise ReachabilityHierarchyError(
                "leaf diagnostics must contain one row per target"
            )
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "routing_candidate", routing)
        object.__setattr__(self, "exact_candidate", exact)
        object.__setattr__(self, "routing_witness", routing_witness)
        object.__setattr__(self, "exact_witness", exact_witness)


@dataclass(frozen=True)
class HierarchyLeafLocalization:
    """Complete target-major CSR routing matches from one leaf resolution.

    This is the hierarchy backend's private, provenance-free representation.
    :class:`eos_trensmuther.reachability.ReachabilityIndex` is responsible for
    attaching the public source-domain provenance when backend dispatch is
    integrated.  Construction here is possible only after complete leaf
    resolution; benchmark-only stopped traversals never produce this type.
    """

    result: HierarchyLeafResolution
    leaf_count: int
    offsets: np.ndarray
    cell_flat_indices: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.result, HierarchyLeafResolution):
            raise ReachabilityHierarchyError(
                "result must be a HierarchyLeafResolution"
            )
        if isinstance(self.leaf_count, (bool, np.bool_)) or not isinstance(
            self.leaf_count, (int, np.integer)
        ):
            raise ReachabilityHierarchyError("leaf_count must be a positive integer")
        leaf_count = int(self.leaf_count)
        if leaf_count <= 0:
            raise ReachabilityHierarchyError("leaf_count must be a positive integer")
        raw_offsets = np.asarray(self.offsets)
        raw_cells = np.asarray(self.cell_flat_indices)
        if raw_offsets.dtype.kind not in "iu" or raw_offsets.dtype.kind == "b":
            raise ReachabilityHierarchyError("CSR offsets must be integers")
        if raw_cells.dtype.kind not in "iu" or raw_cells.dtype.kind == "b":
            raise ReachabilityHierarchyError(
                "routing leaf cell IDs must be integers"
            )
        offsets = np.asarray(raw_offsets, dtype=np.int64)
        cells = np.asarray(raw_cells, dtype=np.int64)
        target_count = int(self.result.valid.size)
        if offsets.shape != (target_count + 1,):
            raise ReachabilityHierarchyError(
                "CSR offsets must have one entry beyond every target"
            )
        if cells.ndim != 1:
            raise ReachabilityHierarchyError(
                "routing leaf cell IDs must be one-dimensional"
            )
        if int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
            raise ReachabilityHierarchyError(
                "CSR offsets must start at zero and be nondecreasing"
            )
        if int(offsets[-1]) != int(cells.size):
            raise ReachabilityHierarchyError(
                "the final CSR offset must equal the routing-pair count"
            )

        counts = np.diff(offsets)
        if np.any(self.result.routing_candidate != (counts > 0)):
            raise ReachabilityHierarchyError(
                "routing candidate flags and complete CSR rows disagree"
            )
        for target_index in range(target_count):
            start = int(offsets[target_index])
            stop = int(offsets[target_index + 1])
            row = cells[start:stop]
            if row.size == 0:
                continue
            if np.any(row < 0) or np.any(row >= leaf_count):
                raise ReachabilityHierarchyError(
                    "routing leaf cell ID is outside the fine-cell lattice"
                )
            if row.size > 1 and np.any(np.diff(row) <= 0):
                raise ReachabilityHierarchyError(
                    "routing leaf cell IDs must be unique and strictly "
                    "increasing within each target row"
                )
            if int(row[0]) != int(self.result.routing_witness[target_index]):
                raise ReachabilityHierarchyError(
                    "the first routing leaf must equal the routing witness"
                )
            exact_witness = int(self.result.exact_witness[target_index])
            if exact_witness >= 0 and not np.any(row == exact_witness):
                raise ReachabilityHierarchyError(
                    "the exact witness must belong to the routing leaf row"
                )

        object.__setattr__(self, "offsets", _immutable_array(offsets, dtype=np.int64))
        object.__setattr__(self, "leaf_count", leaf_count)
        object.__setattr__(
            self,
            "cell_flat_indices",
            _immutable_array(cells, dtype=np.int64),
        )

    @property
    def target_count(self) -> int:
        return int(self.result.valid.size)

    @property
    def pair_count(self) -> int:
        return int(self.cell_flat_indices.size)

    def cells_for(self, target_index: int) -> np.ndarray:
        if isinstance(target_index, (bool, np.bool_)) or not isinstance(
            target_index, (int, np.integer)
        ):
            raise ReachabilityHierarchyError("target_index must be an integer")
        index = int(target_index)
        if index < 0 or index >= self.target_count:
            raise ReachabilityHierarchyError(
                "target_index is outside the hierarchy localization"
            )
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return self.cell_flat_indices[start:stop]


def _routing_matches_to_csr(
    target_count: int,
    query_parts: Sequence[np.ndarray],
    cell_parts: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize complete routing matches to sorted target-major CSR."""

    if len(query_parts) != len(cell_parts):
        raise ReachabilityHierarchyError(
            "routing match query and cell parts disagree"
        )
    if query_parts:
        queries = np.concatenate(query_parts).astype(np.int64, copy=False)
        cells = np.concatenate(cell_parts).astype(np.int64, copy=False)
    else:
        queries = np.empty(0, dtype=np.int64)
        cells = np.empty(0, dtype=np.int64)
    if queries.shape != cells.shape or queries.ndim != 1:
        raise ReachabilityHierarchyError(
            "routing match query and cell arrays must be matching vectors"
        )
    if np.any(queries < 0) or np.any(queries >= target_count):
        raise ReachabilityHierarchyError(
            "routing match query ID is outside the target batch"
        )
    if queries.size:
        order = np.lexsort((cells, queries))
        queries = queries[order]
        cells = cells[order]
        duplicates = (queries[1:] == queries[:-1]) & (cells[1:] == cells[:-1])
        if np.any(duplicates):
            raise ReachabilityHierarchyError(
                "hierarchy produced a duplicate target/leaf routing match"
            )
    counts = np.bincount(queries, minlength=target_count).astype(
        np.int64, copy=False
    )
    offsets = np.empty(target_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, cells


def _child_block_ids(
    parent: RoutingHierarchyLevel,
    child: RoutingHierarchyLevel,
    parent_block_id: int,
) -> np.ndarray:
    parent_multi = np.asarray(
        np.unravel_index(parent_block_id, parent.block_shape, order="C"),
        dtype=np.int64,
    )
    starts, stops = _child_ranges(parent, child)
    ranges = [
        range(
            int(starts[axis][parent_multi[axis]]),
            int(stops[axis][parent_multi[axis]]),
        )
        for axis in range(4)
    ]
    child_multi = np.asarray(list(itertools.product(*ranges)), dtype=np.int64)
    return np.asarray(
        np.ravel_multi_index(
            tuple(child_multi[:, axis] for axis in range(4)),
            child.block_shape,
            order="C",
        ),
        dtype=np.int64,
    )


def _leaf_id_chunks(
    level: RoutingHierarchyLevel,
    block_id: int,
    cell_shape: Tuple[int, int, int, int],
    leaf_batch_size: int,
):
    block_multi = np.asarray(
        np.unravel_index(block_id, level.block_shape, order="C"),
        dtype=np.int64,
    )
    lower, upper = level.source_cell_bounds(block_multi)
    for first in range(int(lower[0]), int(upper[0])):
        for second in range(int(lower[1]), int(upper[1])):
            for third in range(int(lower[2]), int(upper[2])):
                row_start = int(
                    np.ravel_multi_index(
                        (first, second, third, int(lower[3])),
                        cell_shape,
                        order="C",
                    )
                )
                row_stop = row_start + int(upper[3] - lower[3])
                for start in range(row_start, row_stop, leaf_batch_size):
                    yield np.arange(
                        start,
                        min(row_stop, start + leaf_batch_size),
                        dtype=np.int64,
                    )


def _classify_full_leaf_level(
    hierarchy: RoutingHierarchy,
    target_values: np.ndarray,
    valid: np.ndarray,
    exact_lo: np.ndarray,
    exact_hi: np.ndarray,
    routing_lo: np.ndarray,
    routing_hi: np.ndarray,
    query_batch_size: int,
    maximum_pairwise_entries: int,
    collect_routing_matches: bool,
) -> Tuple[
    HierarchyLeafResolution,
    Optional[Tuple[np.ndarray, np.ndarray]],
]:
    """Vectorized terminal resolution when deepest blocks are fine cells."""

    target_count = target_values.shape[0]
    leaf_count = hierarchy.leaf_count
    routing_min = np.full(target_count, leaf_count, dtype=np.int64)
    exact_min = np.full(target_count, leaf_count, dtype=np.int64)
    actual_pairs = np.zeros(target_count, dtype=np.int64)
    classification_pairs = np.zeros(target_count, dtype=np.int64)
    first_routing_rank = np.full(target_count, leaf_count + 1, dtype=np.int64)
    first_exact_rank = np.full(target_count, leaf_count + 1, dtype=np.int64)
    coarse_pairs_tested = 0
    coarse_seconds = 0.0
    leaf_seconds = 0.0
    routing_query_parts: List[np.ndarray] = []
    routing_cell_parts: List[np.ndarray] = []
    valid_indices = np.flatnonzero(valid).astype(np.int64)
    pending: List[np.ndarray] = [
        valid_indices[start : start + query_batch_size]
        for start in range(0, valid_indices.size, query_batch_size)
    ]
    while pending:
        positions = pending.pop(0)
        coarse_started = time.perf_counter()
        try:
            traversal = _traverse_batch(
                hierarchy,
                target_values[positions],
                maximum_pairwise_entries,
            )
        except HierarchyPairLimitError:
            coarse_seconds += time.perf_counter() - coarse_started
            if positions.size == 1:
                raise HierarchyPairLimitError(
                    "one production target exceeds maximum_pairwise_entries; "
                    "increase the explicit bounded-pair configuration"
                )
            midpoint = positions.size // 2
            pending[0:0] = [positions[:midpoint], positions[midpoint:]]
            continue
        coarse_seconds += time.perf_counter() - coarse_started
        coarse_pairs_tested += int(sum(traversal.level_tested))
        local_queries = traversal.surviving_query_ids
        leaf_ids = traversal.surviving_block_ids
        if local_queries.size == 0:
            continue
        global_queries = positions[local_queries]
        local_counts = np.bincount(
            local_queries, minlength=positions.size
        ).astype(np.int64, copy=False)
        np.add.at(actual_pairs, positions, local_counts)
        local_starts = np.cumsum(
            np.concatenate((np.asarray([0], dtype=np.int64), local_counts[:-1]))
        )
        ranks = np.arange(local_queries.size, dtype=np.int64) - local_starts[
            local_queries
        ]
        for start in range(0, local_queries.size, maximum_pairwise_entries):
            stop = min(local_queries.size, start + maximum_pairwise_entries)
            selected_queries = global_queries[start:stop]
            selected_leaves = leaf_ids[start:stop]
            leaf_started = time.perf_counter()
            exact_contains = np.all(
                (target_values[selected_queries] >= exact_lo[selected_leaves])
                & (target_values[selected_queries] <= exact_hi[selected_leaves]),
                axis=1,
            )
            routing_contains = np.all(
                (target_values[selected_queries] >= routing_lo[selected_leaves])
                & (target_values[selected_queries] <= routing_hi[selected_leaves]),
                axis=1,
            )
            leaf_seconds += time.perf_counter() - leaf_started
            if np.any(routing_contains):
                matching_queries = selected_queries[routing_contains]
                matching_leaves = selected_leaves[routing_contains]
                np.minimum.at(
                    routing_min,
                    matching_queries,
                    matching_leaves,
                )
                np.minimum.at(
                    first_routing_rank,
                    matching_queries,
                    ranks[start:stop][routing_contains] + 1,
                )
                if collect_routing_matches:
                    routing_query_parts.append(matching_queries)
                    routing_cell_parts.append(matching_leaves)
            if np.any(exact_contains):
                np.minimum.at(
                    exact_min,
                    selected_queries[exact_contains],
                    selected_leaves[exact_contains],
                )
                np.minimum.at(
                    first_exact_rank,
                    selected_queries[exact_contains],
                    ranks[start:stop][exact_contains] + 1,
                )

    exact_found = exact_min < leaf_count
    routing_found = routing_min < leaf_count
    classification_pairs[:] = actual_pairs
    classification_pairs[routing_found] = first_routing_rank[routing_found]
    # Fine-cell pairs are sorted by (query, global C-order leaf ID), so the
    # first routing/exact match is already its global minimum witness.  Work
    # through the first exact match (or all pairs when no exact match exists)
    # is attributable to preserving the exact flag/witness contract; only the
    # remainder of the vector batch is batching overhead.
    witness_complete_pairs = actual_pairs.copy()
    witness_complete_pairs[exact_found] = first_exact_rank[exact_found]
    witness_pairs = witness_complete_pairs - classification_pairs
    batch_pairs = actual_pairs - witness_complete_pairs
    routing_witness = np.where(routing_found, routing_min, -1)
    exact_witness = np.where(exact_found, exact_min, -1)
    diagnostics = HierarchyLeafDiagnostics(
        actual_leaf_pairs=int(np.sum(actual_pairs, dtype=np.int64)),
        classification_leaf_pairs=int(
            np.sum(classification_pairs, dtype=np.int64)
        ),
        additional_minimum_witness_leaf_pairs=int(
            np.sum(witness_pairs, dtype=np.int64)
        ),
        additional_batch_leaf_pairs=int(np.sum(batch_pairs, dtype=np.int64)),
        coarse_pairs_tested=coarse_pairs_tested,
        coarse_seconds=float(coarse_seconds),
        leaf_seconds=float(leaf_seconds),
        actual_leaf_pairs_by_target=actual_pairs,
        classification_leaf_pairs_by_target=classification_pairs,
        additional_minimum_witness_leaf_pairs_by_target=witness_pairs,
        additional_batch_leaf_pairs_by_target=batch_pairs,
    )
    result = HierarchyLeafResolution(
        valid=valid,
        routing_candidate=routing_found,
        exact_candidate=exact_found,
        routing_witness=routing_witness,
        exact_witness=exact_witness,
        diagnostics=diagnostics,
    )
    routing_csr = None
    if collect_routing_matches:
        routing_csr = _routing_matches_to_csr(
            target_count,
            routing_query_parts,
            routing_cell_parts,
        )
    return result, routing_csr


def _resolve_hierarchy_leaves(
    hierarchy: RoutingHierarchy,
    targets: Any,
    exact_lower: Any,
    exact_upper: Any,
    routing_lower: Any,
    routing_upper: Any,
    *,
    valid_mask: Optional[Any] = None,
    query_batch_size: int = 256,
    leaf_batch_size: int = 4096,
    maximum_pairwise_entries: int = 1_048_576,
    collect_routing_matches: bool = False,
) -> Tuple[
    HierarchyLeafResolution,
    Optional[Tuple[np.ndarray, np.ndarray]],
]:
    """Complete exact/routing leaf checks below every surviving branch.

    This private method has no estimated-time cutoff. It scans all required
    leaves and preserves the global minimum C-order exact/routing witnesses.
    The pair limit bounds temporary comparisons; it does not terminate a
    scientific classification early.
    """

    if not isinstance(hierarchy, RoutingHierarchy):
        raise ReachabilityHierarchyError("hierarchy must be a RoutingHierarchy")
    query_batch = _positive_integer(query_batch_size, "query_batch_size")
    leaf_batch = _positive_integer(leaf_batch_size, "leaf_batch_size")
    pair_limit = _positive_integer(
        maximum_pairwise_entries, "maximum_pairwise_entries"
    )
    if not isinstance(collect_routing_matches, (bool, np.bool_)):
        raise ReachabilityHierarchyError(
            "collect_routing_matches must be a boolean"
        )
    if leaf_batch > pair_limit:
        raise ReachabilityHierarchyError(
            "leaf_batch_size cannot exceed maximum_pairwise_entries"
        )
    target_values, valid = _validate_profile_targets(targets, valid_mask)
    exact_lo, exact_hi = _validate_leaf_bounds(
        hierarchy.cell_shape, exact_lower, exact_upper
    )
    routing_lo, routing_hi = _validate_leaf_bounds(
        hierarchy.cell_shape, routing_lower, routing_upper
    )
    if np.any(routing_lo > exact_lo) or np.any(routing_hi < exact_hi):
        raise ReachabilityHierarchyError(
            "routing leaf boxes must contain exact leaf boxes"
        )
    leaf_count = hierarchy.leaf_count
    if hierarchy.reaches_leaf:
        try:
            return _classify_full_leaf_level(
                hierarchy,
                target_values,
                valid,
                exact_lo,
                exact_hi,
                routing_lo,
                routing_hi,
                query_batch,
                pair_limit,
                bool(collect_routing_matches),
            )
        except HierarchyPairLimitError:
            # The guard controls temporary vector storage, not scientific
            # classification.  Fall back to the bounded per-block traversal
            # when even one target has more terminal pairs than the limit.
            pass
    target_count = target_values.shape[0]
    routing_candidate = np.zeros(target_count, dtype=bool)
    exact_candidate = np.zeros(target_count, dtype=bool)
    routing_witness = np.full(target_count, -1, dtype=np.int64)
    exact_witness = np.full(target_count, -1, dtype=np.int64)
    actual_pairs = np.zeros(target_count, dtype=np.int64)
    classification_pairs = np.zeros(target_count, dtype=np.int64)
    coarse_pairs_tested = 0
    coarse_started = time.perf_counter()
    leaf_seconds = 0.0
    routing_query_parts: List[np.ndarray] = []
    routing_cell_parts: List[np.ndarray] = []

    valid_indices = np.flatnonzero(valid)
    for query_start in range(0, valid_indices.size, query_batch):
        positions = valid_indices[query_start : query_start + query_batch]
        for position in positions:
            target = target_values[position]
            stack = [(0, 0)]
            classification_known_at: Optional[int] = None
            while stack:
                level_index, block_id = stack.pop()
                level = hierarchy.levels[level_index]
                coarse_pairs_tested += 1
                if not np.all(
                    (target >= level.routing_lower[block_id])
                    & (target <= level.routing_upper[block_id])
                ):
                    continue
                if level_index + 1 < len(hierarchy.levels):
                    child = hierarchy.levels[level_index + 1]
                    children = _child_block_ids(level, child, block_id)
                    stack.extend(
                        (level_index + 1, int(child_id))
                        for child_id in children[::-1]
                    )
                    continue

                for leaf_ids in _leaf_id_chunks(
                    level,
                    block_id,
                    hierarchy.cell_shape,
                    leaf_batch,
                ):
                    started = time.perf_counter()
                    exact_contains = np.all(
                        (target >= exact_lo[leaf_ids])
                        & (target <= exact_hi[leaf_ids]),
                        axis=1,
                    )
                    routing_contains = np.all(
                        (target >= routing_lo[leaf_ids])
                        & (target <= routing_hi[leaf_ids]),
                        axis=1,
                    )
                    leaf_seconds += time.perf_counter() - started
                    before = int(actual_pairs[position])
                    actual_pairs[position] += leaf_ids.size
                    if np.any(routing_contains):
                        candidate_ids = leaf_ids[routing_contains]
                        if collect_routing_matches:
                            routing_query_parts.append(
                                np.full(
                                    candidate_ids.size,
                                    int(position),
                                    dtype=np.int64,
                                )
                            )
                            routing_cell_parts.append(candidate_ids)
                        candidate_minimum = int(np.min(candidate_ids))
                        if (
                            routing_witness[position] < 0
                            or candidate_minimum < routing_witness[position]
                        ):
                            routing_witness[position] = candidate_minimum
                        routing_candidate[position] = True
                        if classification_known_at is None:
                            first_routing = int(
                                np.flatnonzero(routing_contains)[0]
                            )
                            classification_known_at = (
                                before + first_routing + 1
                            )
                    if np.any(exact_contains):
                        candidate_ids = leaf_ids[exact_contains]
                        candidate_minimum = int(np.min(candidate_ids))
                        if (
                            exact_witness[position] < 0
                            or candidate_minimum < exact_witness[position]
                        ):
                            exact_witness[position] = candidate_minimum
                        exact_candidate[position] = True

            if classification_known_at is None:
                # Proving routing_candidate=False requires every routed leaf.
                classification_known_at = int(actual_pairs[position])
            classification_pairs[position] = classification_known_at

    if np.any((routing_witness >= leaf_count) | (exact_witness >= leaf_count)):
        raise ReachabilityHierarchyError("hierarchy produced an invalid leaf witness")
    additional_pairs = actual_pairs - classification_pairs
    coarse_seconds = time.perf_counter() - coarse_started - leaf_seconds
    batch_pairs = np.zeros(target_count, dtype=np.int64)
    diagnostics = HierarchyLeafDiagnostics(
        actual_leaf_pairs=int(np.sum(actual_pairs, dtype=np.int64)),
        classification_leaf_pairs=int(
            np.sum(classification_pairs, dtype=np.int64)
        ),
        additional_minimum_witness_leaf_pairs=int(
            np.sum(additional_pairs, dtype=np.int64)
        ),
        additional_batch_leaf_pairs=0,
        coarse_pairs_tested=int(coarse_pairs_tested),
        coarse_seconds=float(max(0.0, coarse_seconds)),
        leaf_seconds=float(leaf_seconds),
        actual_leaf_pairs_by_target=actual_pairs,
        classification_leaf_pairs_by_target=classification_pairs,
        additional_minimum_witness_leaf_pairs_by_target=additional_pairs,
        additional_batch_leaf_pairs_by_target=batch_pairs,
    )
    result = HierarchyLeafResolution(
        valid=valid,
        routing_candidate=routing_candidate,
        exact_candidate=exact_candidate,
        routing_witness=routing_witness,
        exact_witness=exact_witness,
        diagnostics=diagnostics,
    )
    routing_csr = None
    if collect_routing_matches:
        routing_csr = _routing_matches_to_csr(
            target_count,
            routing_query_parts,
            routing_cell_parts,
        )
    return result, routing_csr


def classify_hierarchy_leaves(
    hierarchy: RoutingHierarchy,
    targets: Any,
    exact_lower: Any,
    exact_upper: Any,
    routing_lower: Any,
    routing_upper: Any,
    *,
    valid_mask: Optional[Any] = None,
    query_batch_size: int = 256,
    leaf_batch_size: int = 4096,
    maximum_pairwise_entries: int = 1_048_576,
) -> HierarchyLeafResolution:
    """Classify completely while retaining only flags and witnesses."""

    result, routing_csr = _resolve_hierarchy_leaves(
        hierarchy,
        targets,
        exact_lower,
        exact_upper,
        routing_lower,
        routing_upper,
        valid_mask=valid_mask,
        query_batch_size=query_batch_size,
        leaf_batch_size=leaf_batch_size,
        maximum_pairwise_entries=maximum_pairwise_entries,
        collect_routing_matches=False,
    )
    if routing_csr is not None:  # Defensive: classification must stay compact.
        raise ReachabilityHierarchyError(
            "classification unexpectedly retained routing leaf matches"
        )
    return result


def localize_hierarchy_leaves(
    hierarchy: RoutingHierarchy,
    targets: Any,
    exact_lower: Any,
    exact_upper: Any,
    routing_lower: Any,
    routing_upper: Any,
    *,
    valid_mask: Optional[Any] = None,
    query_batch_size: int = 256,
    leaf_batch_size: int = 4096,
    maximum_pairwise_entries: int = 1_048_576,
) -> HierarchyLeafLocalization:
    """Return complete fine routing matches from the classification pass.

    The CSR rows contain actual individual source-cell routing AABB matches,
    never coarse hierarchy blocks.  No benchmark stopping policy is accepted
    here: every surviving branch is resolved to fine leaves before return.
    """

    result, routing_csr = _resolve_hierarchy_leaves(
        hierarchy,
        targets,
        exact_lower,
        exact_upper,
        routing_lower,
        routing_upper,
        valid_mask=valid_mask,
        query_batch_size=query_batch_size,
        leaf_batch_size=leaf_batch_size,
        maximum_pairwise_entries=maximum_pairwise_entries,
        collect_routing_matches=True,
    )
    if routing_csr is None:  # Defensive: localization must always be complete.
        raise ReachabilityHierarchyError(
            "complete hierarchy localization did not retain routing matches"
        )
    offsets, cell_flat_indices = routing_csr
    return HierarchyLeafLocalization(
        result=result,
        leaf_count=hierarchy.leaf_count,
        offsets=offsets,
        cell_flat_indices=cell_flat_indices,
    )


def _leaf_summary(values: np.ndarray) -> Tuple[int, int, int, float, float, float, int]:
    if values.size == 0:
        return (0, 0, 0, 0.0, 0.0, 0.0, 0)
    requiring = int(np.count_nonzero(values))
    rejected = int(values.size - requiring)
    return (
        rejected,
        requiring,
        int(np.sum(values, dtype=np.int64)),
        float(np.mean(values)),
        float(np.median(values)),
        float(np.percentile(values, 95)),
        int(np.max(values)),
    )


def profile_hierarchy_pruning(
    hierarchy: RoutingHierarchy,
    targets: Any,
    *,
    maximum_estimated_leaf_seconds: Any,
    leaf_pairs_per_second: Any,
    query_batch_size: int = 256,
    maximum_pairwise_entries: int = 1_048_576,
    valid_mask: Optional[Any] = None,
    categories: Optional[Sequence[Any]] = None,
    hierarchy_construction_seconds: Any = 0.0,
) -> HierarchyPruningReport:
    """Profile conservative routing without returning scientific statuses.

    ``maximum_estimated_leaf_seconds`` is deliberately mandatory and belongs
    only to this benchmark operation. It controls whether a later benchmark
    may attempt leaf completion; it never truncates production classification.
    """

    if not isinstance(hierarchy, RoutingHierarchy):
        raise ReachabilityHierarchyError("hierarchy must be a RoutingHierarchy")
    maximum_seconds = _finite_nonnegative(
        maximum_estimated_leaf_seconds, "maximum_estimated_leaf_seconds"
    )
    throughput = _finite_positive(leaf_pairs_per_second, "leaf_pairs_per_second")
    construction_seconds = _finite_nonnegative(
        hierarchy_construction_seconds, "hierarchy_construction_seconds"
    )
    batch_size = _positive_integer(query_batch_size, "query_batch_size")
    pair_limit = _positive_integer(
        maximum_pairwise_entries, "maximum_pairwise_entries"
    )
    target_values, valid = _validate_profile_targets(targets, valid_mask)
    valid_indices = np.flatnonzero(valid).astype(np.int64)
    if categories is None:
        category_values = np.full(target_values.shape[0], "all", dtype=object)
    else:
        category_values = np.asarray(categories, dtype=object)
        if category_values.shape != (target_values.shape[0],):
            raise ReachabilityHierarchyError(
                "categories must contain one value for every target"
            )
        category_values = np.asarray([str(value) for value in category_values])

    level_count = len(hierarchy.levels)
    tested = np.zeros(level_count, dtype=np.int64)
    surviving = np.zeros(level_count, dtype=np.int64)
    distinct = np.zeros(level_count, dtype=np.int64)
    seconds = np.zeros(level_count, dtype=float)
    profiled_indices: List[np.ndarray] = []
    profiled_leaf_pairs: List[np.ndarray] = []
    retry_seconds = 0.0
    stopped_reason: Optional[str] = None
    traversal_started = time.perf_counter()

    pending: List[np.ndarray] = [
        valid_indices[start : start + batch_size]
        for start in range(0, valid_indices.size, batch_size)
    ]
    while pending:
        target_indices = pending.pop(0)
        attempted = time.perf_counter()
        try:
            result = _traverse_batch(
                hierarchy, target_values[target_indices], pair_limit
            )
        except HierarchyPairLimitError:
            retry_seconds += time.perf_counter() - attempted
            if target_indices.size == 1:
                stopped_reason = "maximum_pairwise_entries_exceeded_by_one_target"
                break
            midpoint = target_indices.size // 2
            pending[0:0] = [target_indices[:midpoint], target_indices[midpoint:]]
            continue
        profiled_indices.append(target_indices)
        profiled_leaf_pairs.append(result.leaf_pairs_by_target)
        tested += np.asarray(result.level_tested, dtype=np.int64)
        surviving += np.asarray(result.level_surviving, dtype=np.int64)
        distinct += np.asarray(result.level_distinct, dtype=np.int64)
        seconds += np.asarray(result.level_seconds, dtype=float)

    coarse_seconds = time.perf_counter() - traversal_started
    if profiled_indices:
        completed_indices = np.concatenate(profiled_indices)
        leaf_pairs = np.concatenate(profiled_leaf_pairs)
        order = np.argsort(completed_indices, kind="stable")
        completed_indices = completed_indices[order]
        leaf_pairs = leaf_pairs[order]
    else:
        completed_indices = np.empty(0, dtype=np.int64)
        leaf_pairs = np.empty(0, dtype=np.int64)
    traversal_completed = stopped_reason is None
    rejected, requiring, leaf_sum, mean, median, p95, maximum = _leaf_summary(
        leaf_pairs
    )
    estimated_seconds = float(leaf_sum / throughput)
    completion_allowed = bool(
        traversal_completed and estimated_seconds <= maximum_seconds
    )
    if traversal_completed and not completion_allowed:
        stopped_reason = "estimated_leaf_seconds_exceeds_configured_limit"

    level_profiles = tuple(
        HierarchyLevelProfile(
            logical_level=level.logical_level,
            block_shape=level.block_shape,
            stored_blocks=level.block_count,
            tested_pairs=int(tested[index]),
            surviving_pairs=int(surviving[index]),
            distinct_targets_surviving=int(distinct[index]),
            traversal_seconds=float(seconds[index]),
        )
        for index, level in enumerate(hierarchy.levels)
    )
    category_profiles: Dict[str, HierarchyCategoryProfile] = {}
    completed_categories = category_values[completed_indices]
    for category in sorted(set(str(value) for value in completed_categories)):
        category_leaf_pairs = leaf_pairs[completed_categories == category]
        (
            category_rejected,
            category_requiring,
            category_sum,
            category_mean,
            category_median,
            category_p95,
            category_maximum,
        ) = _leaf_summary(category_leaf_pairs)
        category_profiles[category] = HierarchyCategoryProfile(
            target_count=int(category_leaf_pairs.size),
            rejected_before_leaf=category_rejected,
            requiring_leaf=category_requiring,
            leaf_pairs_sum=category_sum,
            leaf_pairs_mean=category_mean,
            leaf_pairs_median=category_median,
            leaf_pairs_p95=category_p95,
            leaf_pairs_maximum=category_maximum,
        )

    return HierarchyPruningReport(
        target_count=int(target_values.shape[0]),
        valid_target_count=int(valid_indices.size),
        profiled_valid_target_count=int(completed_indices.size),
        hierarchy_levels=hierarchy.requested_levels,
        level_profiles=level_profiles,
        traversal_completed=traversal_completed,
        complete_classification=False,
        stopped_reason=stopped_reason,
        targets_rejected_before_leaf=rejected,
        targets_requiring_leaf=requiring,
        descendant_leaf_pairs=leaf_sum,
        leaf_pairs_mean=mean,
        leaf_pairs_median=median,
        leaf_pairs_p95=p95,
        leaf_pairs_maximum=maximum,
        leaf_pairs_per_second=throughput,
        estimated_leaf_seconds=estimated_seconds,
        maximum_estimated_leaf_seconds=maximum_seconds,
        leaf_completion_allowed=completion_allowed,
        hierarchy_construction_seconds=construction_seconds,
        coarse_traversal_seconds=coarse_seconds,
        hierarchy_owned_bytes=hierarchy.owned_nbytes,
        hierarchy_referenced_leaf_bytes=hierarchy.referenced_leaf_nbytes,
        query_batch_size=batch_size,
        maximum_pairwise_entries=pair_limit,
        retry_seconds=retry_seconds,
        category_profiles=category_profiles,
        profiled_target_indices=completed_indices,
        descendant_leaf_pairs_by_profiled_target=leaf_pairs,
    )


def _validate_leaf_bounds(
    cell_shape: Tuple[int, int, int, int],
    routing_lower: Any,
    routing_upper: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    leaf_count = math.prod(cell_shape)
    lower = np.asarray(routing_lower, dtype=float)
    upper = np.asarray(routing_upper, dtype=float)
    if (
        lower.shape != (leaf_count, 4)
        or upper.shape != (leaf_count, 4)
        or np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower > upper)
    ):
        raise ReachabilityHierarchyError(
            "leaf routing bounds must be ordered (prod(cell_shape), 4) arrays"
        )
    return lower, upper


def _estimated_owned_bytes(
    cell_shape: Tuple[int, int, int, int],
    requested_levels: int,
    *,
    leaf_can_alias: bool,
) -> int:
    total = 0
    previous_shape = None
    for logical_level in range(1, requested_levels + 1):
        shape = _logical_block_shape(cell_shape, logical_level)
        if shape == previous_shape:
            break
        aliases_leaf = shape == cell_shape
        if not aliases_leaf or not leaf_can_alias:
            total += math.prod(shape) * 2 * 4 * np.dtype(np.float64).itemsize
        total += sum(
            (blocks + 1) * np.dtype(np.int64).itemsize for blocks in shape
        )
        previous_shape = shape
        if aliases_leaf:
            break
    return int(total)


def build_routing_hierarchy(
    cell_shape: Sequence[int],
    routing_lower: Any,
    routing_upper: Any,
    *,
    hierarchy_levels: int,
    max_hierarchy_bytes: int = DEFAULT_MAX_HIERARCHY_BYTES,
) -> RoutingHierarchy:
    """Build an immutable requested prefix from conservative leaf boxes."""

    resolved_shape = _cell_shape(cell_shape)
    levels = _positive_integer(hierarchy_levels, "hierarchy_levels")
    if levels not in SUPPORTED_HIERARCHY_LEVELS:
        raise ReachabilityHierarchyError(
            "hierarchy_levels must be one of %s"
            % (SUPPORTED_HIERARCHY_LEVELS,)
        )
    maximum = _positive_integer(max_hierarchy_bytes, "max_hierarchy_bytes")
    leaf_lower, leaf_upper = _validate_leaf_bounds(
        resolved_shape, routing_lower, routing_upper
    )
    leaf_can_alias = bool(
        leaf_lower.dtype == np.dtype(np.float64)
        and leaf_upper.dtype == np.dtype(np.float64)
        and leaf_lower.flags.c_contiguous
        and leaf_upper.flags.c_contiguous
        and not leaf_lower.flags.writeable
        and not leaf_upper.flags.writeable
    )
    estimated = _estimated_owned_bytes(
        resolved_shape, levels, leaf_can_alias=leaf_can_alias
    )
    if estimated > maximum:
        raise ReachabilityHierarchyError(
            "hierarchy requires at least %d owned bytes, exceeding "
            "max_hierarchy_bytes=%d" % (estimated, maximum)
        )

    level_shapes = []
    for logical_level in range(1, levels + 1):
        shape = _logical_block_shape(resolved_shape, logical_level)
        if level_shapes and shape == level_shapes[-1]:
            break
        level_shapes.append(shape)
        if shape == resolved_shape:
            break
    boundaries_by_level = [
        tuple(
            _axis_boundaries(cells, blocks)
            for cells, blocks in zip(resolved_shape, block_shape)
        )
        for block_shape in level_shapes
    ]

    deepest_shape = level_shapes[-1]
    deepest_boundaries = boundaries_by_level[-1]
    aliases_leaf = deepest_shape == resolved_shape and leaf_can_alias
    if aliases_leaf:
        child_lower = np.asarray(leaf_lower)
        child_upper = np.asarray(leaf_upper)
    else:
        child_lower = _aggregate_from_leaf(
            leaf_lower, resolved_shape, deepest_boundaries, "minimum"
        )
        child_upper = _aggregate_from_leaf(
            leaf_upper, resolved_shape, deepest_boundaries, "maximum"
        )
        child_lower, child_upper = _outward_bounds(child_lower, child_upper)

    built_reversed = [
        RoutingHierarchyLevel(
            logical_level=len(level_shapes),
            cell_shape=resolved_shape,
            block_shape=deepest_shape,
            axis_boundaries=deepest_boundaries,
            routing_lower=child_lower,
            routing_upper=child_upper,
            aliases_leaf=aliases_leaf,
        )
    ]
    child_shape = deepest_shape
    child_boundaries = deepest_boundaries
    for offset in range(len(level_shapes) - 2, -1, -1):
        parent_shape = level_shapes[offset]
        parent_boundaries = boundaries_by_level[offset]
        parent_lower = _aggregate_from_child(
            child_lower,
            child_shape,
            child_boundaries,
            parent_boundaries,
            "minimum",
        )
        parent_upper = _aggregate_from_child(
            child_upper,
            child_shape,
            child_boundaries,
            parent_boundaries,
            "maximum",
        )
        parent_lower, parent_upper = _outward_bounds(parent_lower, parent_upper)
        built_reversed.append(
            RoutingHierarchyLevel(
                logical_level=offset + 1,
                cell_shape=resolved_shape,
                block_shape=parent_shape,
                axis_boundaries=parent_boundaries,
                routing_lower=parent_lower,
                routing_upper=parent_upper,
                aliases_leaf=False,
            )
        )
        child_lower = parent_lower
        child_upper = parent_upper
        child_shape = parent_shape
        child_boundaries = parent_boundaries

    built = tuple(reversed(built_reversed))
    return RoutingHierarchy(
        cell_shape=resolved_shape,
        requested_levels=levels,
        levels=built,
        max_hierarchy_bytes=maximum,
    )


__all__ = [
    "DEFAULT_MAX_HIERARCHY_BYTES",
    "HierarchyCategoryProfile",
    "HierarchyLeafDiagnostics",
    "HierarchyLeafLocalization",
    "HierarchyLeafResolution",
    "HierarchyLevelProfile",
    "HierarchyPairLimitError",
    "HierarchyPruningReport",
    "ReachabilityHierarchyError",
    "RoutingHierarchy",
    "RoutingHierarchyLevel",
    "SUPPORTED_HIERARCHY_LEVELS",
    "build_routing_hierarchy",
    "classify_hierarchy_leaves",
    "localize_hierarchy_leaves",
    "profile_hierarchy_pruning",
]
