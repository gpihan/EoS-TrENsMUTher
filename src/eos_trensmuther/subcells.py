"""Proof-safe target-local dyadic subcell topology.

This private module owns only source-box geometry and completeness accounting.
It deliberately does not implement multilinear interval arithmetic.  A caller
must provide conservative routing enclosures through ``SubcellEvaluator``;
the production solver does so with its existing ``_RigorousCellModel``.

Every box is expressed in the unit coordinates of one original source cell.
All active axes are bisected together, giving ``2**dimension`` children in
deterministic C order.  Empty *complete* refinement is a no-root certificate.
Resource-guarded or otherwise incomplete work never is.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import numpy as np

from ._array_storage import immutable_array


MAX_SUBCELL_DIMENSION = 4
MAX_SUBCELL_DEPTH = 7
DEFAULT_MAX_SUBCELL_BOXES = 1_000_000
_EVALUATION_CHILD_BATCH_SIZE = 4096


class SubcellError(ValueError):
    """Base error for target-local subcell construction."""


class SubcellInputError(SubcellError):
    """Raised when target, cell, or configuration input is malformed."""


class SubcellInvariantError(SubcellError):
    """Raised when an enclosure callback violates the proof contract."""


class SubcellCompleteness(str, Enum):
    """Whether every branch required by the requested refinement was tested."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class SubcellStopReason(str, Enum):
    """Why target-local refinement returned."""

    EMPTY_UNION = "empty_union"
    MAXIMUM_DEPTH = "maximum_depth"
    ACTIVE_BOX_LIMIT = "active_box_limit"


def _integer(value: Any, name: str, *, minimum: int, maximum: Optional[int] = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise SubcellInputError("%s must be an integer" % name)
    converted = int(value)
    if converted < minimum or (maximum is not None and converted > maximum):
        if maximum is None:
            message = "%s must be at least %d" % (name, minimum)
        else:
            message = "%s must lie in [%d, %d]" % (name, minimum, maximum)
        raise SubcellInputError(message)
    return converted


@dataclass(frozen=True)
class SubcellBoxes:
    """Immutable equal-depth boxes in original-cell unit coordinates."""

    dimension: int
    depth: int
    cell_flat_indices: np.ndarray
    unit_lower: np.ndarray
    unit_upper: np.ndarray

    def __post_init__(self) -> None:
        dimension = _integer(
            self.dimension,
            "dimension",
            minimum=1,
            maximum=MAX_SUBCELL_DIMENSION,
        )
        depth = _integer(
            self.depth,
            "depth",
            minimum=0,
            maximum=MAX_SUBCELL_DEPTH,
        )
        raw_cells = np.asarray(self.cell_flat_indices)
        if raw_cells.ndim != 1 or not np.issubdtype(raw_cells.dtype, np.integer):
            raise SubcellInputError("cell_flat_indices must be a one-dimensional integer array")
        cells = np.asarray(raw_cells, dtype=np.int64)
        if np.any(cells < 0):
            raise SubcellInputError("cell_flat_indices cannot contain negative values")
        lower = np.asarray(self.unit_lower, dtype=np.float64)
        upper = np.asarray(self.unit_upper, dtype=np.float64)
        expected = (cells.size, dimension)
        if lower.shape != expected or upper.shape != expected:
            raise SubcellInputError(
                "unit bounds must both have shape %s" % (expected,)
            )
        if (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower < 0.0)
            or np.any(upper > 1.0)
            or np.any(lower >= upper)
        ):
            raise SubcellInputError(
                "unit bounds must be finite positive-width boxes inside [0, 1]"
            )

        resolution = 1 << depth
        scaled_lower = lower * float(resolution)
        local_indices = np.rint(scaled_lower).astype(np.int64)
        expected_lower = local_indices / float(resolution)
        expected_upper = (local_indices + 1) / float(resolution)
        if (
            np.any(local_indices < 0)
            or np.any(local_indices >= resolution)
            or not np.array_equal(lower, expected_lower)
            or not np.array_equal(upper, expected_upper)
        ):
            raise SubcellInputError(
                "unit bounds must describe exact dyadic boxes at the declared depth"
            )

        if cells.size:
            # np.lexsort uses the last key as primary.  This gives global cell
            # C order followed by local dyadic C order within each cell.
            keys = tuple(
                [local_indices[:, axis] for axis in range(dimension - 1, -1, -1)]
                + [cells]
            )
            order = np.lexsort(keys)
            cells = cells[order]
            lower = lower[order]
            upper = upper[order]
            local_indices = local_indices[order]
            duplicate = (cells[1:] == cells[:-1]) & np.all(
                local_indices[1:] == local_indices[:-1], axis=1
            )
            if np.any(duplicate):
                raise SubcellInputError("subcell boxes must be unique")

        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(
            self, "cell_flat_indices", immutable_array(cells, dtype=np.int64)
        )
        object.__setattr__(self, "unit_lower", immutable_array(lower, dtype=float))
        object.__setattr__(self, "unit_upper", immutable_array(upper, dtype=float))

    @classmethod
    def original_cells(cls, cell_flat_indices: Any, dimension: int) -> "SubcellBoxes":
        """Construct depth-zero unit boxes for complete original candidates."""

        raw = np.asarray(cell_flat_indices)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise SubcellInputError(
                "cell_flat_indices must be a one-dimensional integer array"
            )
        cells = np.asarray(raw, dtype=np.int64)
        return cls(
            dimension=dimension,
            depth=0,
            cell_flat_indices=cells,
            unit_lower=np.zeros((cells.size, int(dimension)), dtype=float),
            unit_upper=np.ones((cells.size, int(dimension)), dtype=float),
        )

    @property
    def count(self) -> int:
        return int(self.cell_flat_indices.size)

    @property
    def normalized_volume(self) -> float:
        """Total unit-cell volume, counting each original cell separately."""

        return float(self.count) / float((1 << self.depth) ** self.dimension)

    @property
    def nbytes(self) -> int:
        return int(
            self.cell_flat_indices.nbytes
            + self.unit_lower.nbytes
            + self.unit_upper.nbytes
        )

    def dyadic_children(self) -> "SubcellBoxes":
        """Bisect every axis of every box in deterministic parent/C order."""

        if self.depth >= MAX_SUBCELL_DEPTH:
            raise SubcellInputError(
                "cannot refine beyond maximum subcell depth %d" % MAX_SUBCELL_DEPTH
            )
        child_offsets = np.asarray(
            tuple(itertools.product((0, 1), repeat=self.dimension)),
            dtype=np.int64,
        )
        midpoint = self.unit_lower / 2.0 + self.unit_upper / 2.0
        lower = np.where(
            child_offsets[None, :, :] == 0,
            self.unit_lower[:, None, :],
            midpoint[:, None, :],
        )
        upper = np.where(
            child_offsets[None, :, :] == 0,
            midpoint[:, None, :],
            self.unit_upper[:, None, :],
        )
        children_per_parent = 1 << self.dimension
        return SubcellBoxes(
            dimension=self.dimension,
            depth=self.depth + 1,
            cell_flat_indices=np.repeat(
                self.cell_flat_indices, children_per_parent
            ),
            unit_lower=lower.reshape((-1, self.dimension)),
            unit_upper=upper.reshape((-1, self.dimension)),
        )

    def select(self, mask: Any) -> "SubcellBoxes":
        """Return the deterministically ordered subset selected by *mask*."""

        selected = np.asarray(mask)
        if selected.shape != (self.count,) or selected.dtype != np.dtype(bool):
            raise SubcellInputError(
                "subcell selection mask must be boolean with shape (%d,)"
                % self.count
            )
        return SubcellBoxes(
            dimension=self.dimension,
            depth=self.depth,
            cell_flat_indices=self.cell_flat_indices[selected],
            unit_lower=self.unit_lower[selected],
            unit_upper=self.unit_upper[selected],
        )


@dataclass(frozen=True)
class SubcellEnclosures:
    """Conservative target-space routing boxes for one subcell batch."""

    routing_lower: np.ndarray
    routing_upper: np.ndarray
    defined: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.routing_lower, dtype=np.float64)
        upper = np.asarray(self.routing_upper, dtype=np.float64)
        defined = np.asarray(self.defined)
        if lower.ndim != 2 or upper.shape != lower.shape:
            raise SubcellInvariantError(
                "routing enclosure bounds must be equal two-dimensional arrays"
            )
        if defined.shape != (lower.shape[0],) or defined.dtype != np.dtype(bool):
            raise SubcellInvariantError(
                "defined must be boolean with one entry per routing enclosure"
            )
        if np.any(
            defined
            & (
                ~np.all(np.isfinite(lower) & np.isfinite(upper), axis=1)
                | np.any(lower > upper, axis=1)
            )
        ):
            raise SubcellInvariantError(
                "defined routing enclosures must have finite ordered bounds"
            )
        object.__setattr__(self, "routing_lower", immutable_array(lower, dtype=float))
        object.__setattr__(self, "routing_upper", immutable_array(upper, dtype=float))
        object.__setattr__(self, "defined", immutable_array(defined, dtype=bool))


SubcellEvaluator = Callable[[SubcellBoxes], SubcellEnclosures]


@dataclass(frozen=True)
class SubcellLevelReport:
    """Work and surviving geometry for one requested dyadic level."""

    depth: int
    complete: bool
    boxes_entering: int
    boxes_generated: int
    boxes_evaluated: int
    boxes_surviving: int
    undefined_boxes_retained: int
    step_volume_fraction: Optional[float]
    original_volume_fraction: Optional[float]
    elapsed_seconds: float


@dataclass(frozen=True)
class SubcellRefinementReport:
    """Completeness-bearing diagnostics for one target-local refinement."""

    dimension: int
    requested_depth: int
    completed_depth: int
    completeness: SubcellCompleteness
    stop_reason: SubcellStopReason
    initial_cell_count: int
    total_boxes_evaluated: int
    first_exclusion_depth: Optional[int]
    levels: Tuple[SubcellLevelReport, ...]

    @property
    def complete(self) -> bool:
        return self.completeness is SubcellCompleteness.COMPLETE


@dataclass(frozen=True)
class SubcellRefinementResult:
    """Last complete box union plus diagnostics.

    If a guard interrupts a level, ``boxes`` is the preceding complete union.
    This intentionally discards partial-level pruning so the returned geometry
    never pretends to cover work that was not finished.
    """

    boxes: SubcellBoxes
    report: SubcellRefinementReport

    def __post_init__(self) -> None:
        if self.boxes.dimension != self.report.dimension:
            raise SubcellInvariantError(
                "subcell result dimension does not match its report"
            )
        if self.boxes.depth != self.report.completed_depth:
            raise SubcellInvariantError(
                "subcell result depth must be the last complete level"
            )
        if self.report.stop_reason is SubcellStopReason.EMPTY_UNION:
            if not self.report.complete or self.boxes.count != 0:
                raise SubcellInvariantError(
                    "empty-union termination requires complete empty geometry"
                )
        elif self.boxes.count == 0:
            raise SubcellInvariantError(
                "only complete empty-union termination may return no boxes"
            )

    @property
    def certifies_no_root(self) -> bool:
        """Whether complete routing-enclosure exclusion proves no root."""

        return bool(
            self.report.complete
            and self.report.stop_reason is SubcellStopReason.EMPTY_UNION
            and self.boxes.count == 0
        )


def _validate_evaluator_result(
    result: Any, boxes: SubcellBoxes
) -> SubcellEnclosures:
    if not isinstance(result, SubcellEnclosures):
        raise SubcellInvariantError(
            "subcell evaluator must return SubcellEnclosures"
        )
    expected = (boxes.count, boxes.dimension)
    if result.routing_lower.shape != expected:
        raise SubcellInvariantError(
            "subcell evaluator returned shape %s; expected %s"
            % (result.routing_lower.shape, expected)
        )
    return result


def refine_subcells(
    target: Any,
    candidate_cell_flat_indices: Any,
    dimension: int,
    evaluator: SubcellEvaluator,
    *,
    maximum_depth: int = MAX_SUBCELL_DEPTH,
    maximum_active_boxes: int = DEFAULT_MAX_SUBCELL_BOXES,
) -> SubcellRefinementResult:
    """Refine a complete original-cell candidate set toward one target.

    Undefined/non-finite proof enclosures are retained conservatively.  A
    resource limit stops before publishing a partial level and returns the
    preceding complete union with ``INCOMPLETE`` authority.
    """

    dimension = _integer(
        dimension, "dimension", minimum=1, maximum=MAX_SUBCELL_DIMENSION
    )
    maximum_depth = _integer(
        maximum_depth,
        "maximum_depth",
        minimum=1,
        maximum=MAX_SUBCELL_DEPTH,
    )
    maximum_active_boxes = _integer(
        maximum_active_boxes, "maximum_active_boxes", minimum=1
    )
    if not callable(evaluator):
        raise SubcellInputError("evaluator must be callable")
    query = np.asarray(target, dtype=np.float64)
    if query.shape != (dimension,) or not np.all(np.isfinite(query)):
        raise SubcellInputError(
            "target must be a finite vector with shape (%d,)" % dimension
        )
    active = SubcellBoxes.original_cells(candidate_cell_flat_indices, dimension)
    if active.count == 0:
        raise SubcellInputError(
            "subcell refinement requires at least one complete candidate cell"
        )
    initial_count = active.count
    if active.count > maximum_active_boxes:
        report = SubcellRefinementReport(
            dimension=dimension,
            requested_depth=maximum_depth,
            completed_depth=0,
            completeness=SubcellCompleteness.INCOMPLETE,
            stop_reason=SubcellStopReason.ACTIVE_BOX_LIMIT,
            initial_cell_count=initial_count,
            total_boxes_evaluated=0,
            first_exclusion_depth=None,
            levels=(),
        )
        return SubcellRefinementResult(active, report)

    level_reports = []
    total_evaluated = 0
    children_per_parent = 1 << dimension
    parents_per_batch = max(1, _EVALUATION_CHILD_BATCH_SIZE // children_per_parent)

    for depth in range(1, maximum_depth + 1):
        started = time.perf_counter()
        entering = active.count
        generated = entering * children_per_parent
        retained_parts = []
        survivors_seen = 0
        undefined_seen = 0
        evaluated = 0
        guarded = False

        for start in range(0, entering, parents_per_batch):
            stop = min(entering, start + parents_per_batch)
            parent_batch = SubcellBoxes(
                dimension=dimension,
                depth=active.depth,
                cell_flat_indices=active.cell_flat_indices[start:stop],
                unit_lower=active.unit_lower[start:stop],
                unit_upper=active.unit_upper[start:stop],
            )
            children = parent_batch.dyadic_children()
            shared_child_evaluator = getattr(
                evaluator, "evaluate_dyadic_children", None
            )
            if callable(shared_child_evaluator):
                evaluated_children = shared_child_evaluator(
                    parent_batch, children
                )
            else:
                evaluated_children = evaluator(children)
            enclosures = _validate_evaluator_result(
                evaluated_children, children
            )
            defined = np.asarray(enclosures.defined, dtype=bool)
            contains = defined & np.all(
                (query[None, :] >= enclosures.routing_lower)
                & (query[None, :] <= enclosures.routing_upper),
                axis=1,
            )
            # An undefined proof enclosure cannot exclude its source box.
            contains |= ~defined
            selected = children.select(contains)
            retained_parts.append(selected)
            survivors_seen += selected.count
            undefined_seen += int(np.count_nonzero(~defined))
            evaluated += children.count
            if survivors_seen > maximum_active_boxes:
                guarded = True
                break

        total_evaluated += evaluated
        elapsed = float(time.perf_counter() - started)
        if guarded:
            level_reports.append(
                SubcellLevelReport(
                    depth=depth,
                    complete=False,
                    boxes_entering=entering,
                    boxes_generated=generated,
                    boxes_evaluated=evaluated,
                    boxes_surviving=survivors_seen,
                    undefined_boxes_retained=undefined_seen,
                    step_volume_fraction=None,
                    original_volume_fraction=None,
                    elapsed_seconds=elapsed,
                )
            )
            report = SubcellRefinementReport(
                dimension=dimension,
                requested_depth=maximum_depth,
                completed_depth=depth - 1,
                completeness=SubcellCompleteness.INCOMPLETE,
                stop_reason=SubcellStopReason.ACTIVE_BOX_LIMIT,
                initial_cell_count=initial_count,
                total_boxes_evaluated=total_evaluated,
                first_exclusion_depth=None,
                levels=tuple(level_reports),
            )
            return SubcellRefinementResult(active, report)

        if retained_parts:
            cells = np.concatenate(
                [part.cell_flat_indices for part in retained_parts]
            )
            lower = np.concatenate([part.unit_lower for part in retained_parts])
            upper = np.concatenate([part.unit_upper for part in retained_parts])
        else:
            cells = np.empty(0, dtype=np.int64)
            lower = np.empty((0, dimension), dtype=float)
            upper = np.empty((0, dimension), dtype=float)
        active = SubcellBoxes(
            dimension=dimension,
            depth=depth,
            cell_flat_indices=cells,
            unit_lower=lower,
            unit_upper=upper,
        )
        step_fraction = active.count / float(generated)
        original_fraction = active.normalized_volume / float(initial_count)
        level_reports.append(
            SubcellLevelReport(
                depth=depth,
                complete=True,
                boxes_entering=entering,
                boxes_generated=generated,
                boxes_evaluated=evaluated,
                boxes_surviving=active.count,
                undefined_boxes_retained=undefined_seen,
                step_volume_fraction=step_fraction,
                original_volume_fraction=original_fraction,
                elapsed_seconds=elapsed,
            )
        )
        if active.count == 0:
            report = SubcellRefinementReport(
                dimension=dimension,
                requested_depth=maximum_depth,
                completed_depth=depth,
                completeness=SubcellCompleteness.COMPLETE,
                stop_reason=SubcellStopReason.EMPTY_UNION,
                initial_cell_count=initial_count,
                total_boxes_evaluated=total_evaluated,
                first_exclusion_depth=depth,
                levels=tuple(level_reports),
            )
            return SubcellRefinementResult(active, report)

    report = SubcellRefinementReport(
        dimension=dimension,
        requested_depth=maximum_depth,
        completed_depth=maximum_depth,
        completeness=SubcellCompleteness.COMPLETE,
        stop_reason=SubcellStopReason.MAXIMUM_DEPTH,
        initial_cell_count=initial_count,
        total_boxes_evaluated=total_evaluated,
        first_exclusion_depth=None,
        levels=tuple(level_reports),
    )
    return SubcellRefinementResult(active, report)


__all__ = [
    "DEFAULT_MAX_SUBCELL_BOXES",
    "MAX_SUBCELL_DEPTH",
    "MAX_SUBCELL_DIMENSION",
    "SubcellBoxes",
    "SubcellCompleteness",
    "SubcellEnclosures",
    "SubcellError",
    "SubcellInputError",
    "SubcellInvariantError",
    "SubcellLevelReport",
    "SubcellRefinementReport",
    "SubcellRefinementResult",
    "SubcellStopReason",
    "refine_subcells",
]
