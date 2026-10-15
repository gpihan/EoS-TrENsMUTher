"""Conservative serial reachability screening for multilinear EoS tables.

For each active source-grid cell, the mathematical multilinear image is
contained in the axis-aligned box formed by component-wise corner extrema.
The union of those boxes is a conservative superset of the reachable image:
being outside every box proves absence, while being inside a box proves
nothing about existence.

This module implements the deliberately simple Phase-2 reference algorithm.
It scans bounded query/cell batches and is intended as the correctness oracle
for future rasterized or BVH accelerators.  Exact cell-image boxes are kept
separate from outward-expanded routing boxes that account for the solver's
nonzero scaled residual tolerance.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum, IntEnum
from fractions import Fraction
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._array_storage import immutable_array as _storage_immutable_array
from .interpolation import MultilinearInterpolator
from .pipeline import (
    GridChunk,
    PipelinePlan,
    PrescreenStatus,
    pipeline_status_code,
)
from .tables import EoSError
from .transition import BinodalBand, MixedPhaseStrip


DEFAULT_REACHABILITY_CELL_BATCH_SIZE = 4096
DEFAULT_REACHABILITY_QUERY_BATCH_SIZE = 256
DEFAULT_MAX_ENCLOSURE_BYTES = 2 * 1024**3
DEFAULT_MAX_PAIRWISE_ENTRIES = 1_048_576
DEFAULT_MAX_HIERARCHY_BYTES = 2 * 1024**3
REACHABILITY_METHODS = ("exhaustive", "hierarchical")
# Production selection is intentionally limited to the validated HRG4D depth.
# The private builder/benchmark tooling retains shallower experimental depths.
REACHABILITY_HIERARCHY_LEVELS = (7,)
REACHABILITY_ROUTING_CONTRACT_VERSION = 1


class ReachabilityError(EoSError):
    """Base exception for conservative reachability screening."""


class ReachabilityConfigurationError(ReachabilityError):
    """Raised when an interpolator or screening policy is inconsistent."""


class ReachabilityInputError(ReachabilityError):
    """Raised when target queries cannot be interpreted."""


class ReachabilityLocalizationError(ReachabilityError):
    """Raised when candidate localization cannot support its claimed proof."""


class LocalizationCompleteness(str, Enum):
    """Authority carried by one candidate-cell localization."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class CellImageKind(IntEnum):
    """Bit-valued provenance tag for one conservative target-space image."""

    ORDINARY = 1
    LINE_REGION_S = 2
    BINODALS_INTERIOR = 4
    BINODALS_BOUNDARY = 8


_CELL_IMAGE_KIND_MASK = sum(int(value) for value in CellImageKind)


def _contains_unknown_cell_image_kind_bits(values: np.ndarray) -> bool:
    """Return whether an integer array contains bits outside the v1 mask."""

    uint16_max = np.iinfo(np.uint16).max
    if np.any(values < 0) or np.any(values > uint16_max):
        return True
    encoded = values.astype(np.uint16, copy=False)
    allowed = np.asarray(_CELL_IMAGE_KIND_MASK, dtype=np.uint16)
    unknown = np.bitwise_not(allowed)
    return bool(np.any(np.bitwise_and(encoded, unknown) != 0))


@dataclass(frozen=True)
class BinodalSubdivisionConfig:
    """Controls conservative source-box subdivision along binodal curves.

    A boundary box is retained when ``maximum_depth`` is reached or when both
    of its physical widths are no larger than their configured tolerances.
    Such a retained box is a candidate only; exact final phase membership is
    always checked with :meth:`BinodalBand.contains_points`.
    """

    maximum_depth: int = 12
    temperature_tolerance: float = 1.0e-4
    chemical_potential_tolerance: float = 1.0e-4
    maximum_images: int = 1_000_000

    def __post_init__(self) -> None:
        for name in ("maximum_depth", "maximum_images"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ReachabilityConfigurationError(
                    "%s must be a non-negative integer" % name
                )
            converted = int(value)
            if converted < 0 or (name == "maximum_images" and converted == 0):
                qualifier = "positive" if name == "maximum_images" else "non-negative"
                raise ReachabilityConfigurationError(
                    "%s must be a %s integer" % (name, qualifier)
                )
            if name == "maximum_depth" and converted > 60:
                raise ReachabilityConfigurationError(
                    "maximum_depth cannot exceed 60 binary subdivisions"
                )
            object.__setattr__(self, name, converted)
        object.__setattr__(
            self,
            "temperature_tolerance",
            _finite_nonnegative(
                self.temperature_tolerance, "temperature_tolerance"
            ),
        )
        object.__setattr__(
            self,
            "chemical_potential_tolerance",
            _finite_nonnegative(
                self.chemical_potential_tolerance,
                "chemical_potential_tolerance",
            ),
        )


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ReachabilityConfigurationError("%s must be a positive integer" % name)
    converted = int(value)
    if converted <= 0:
        raise ReachabilityConfigurationError("%s must be a positive integer" % name)
    return converted


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ReachabilityConfigurationError(
            "%s must be a finite non-negative number" % name
        )
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReachabilityConfigurationError(
            "%s must be a finite non-negative number" % name
        ) from exc
    if not math.isfinite(converted) or converted < 0.0:
        raise ReachabilityConfigurationError(
            "%s must be a finite non-negative number" % name
        )
    return converted


def _immutable_array(value: Any, dtype: Optional[Any] = None) -> np.ndarray:
    return _storage_immutable_array(value, dtype=dtype)


def _active_cell_domain_fingerprint(
    cell_shape: Tuple[int, ...], cell_flat_indices: np.ndarray
) -> str:
    """Return a stable identity for one declared active source-cell domain."""

    digest = hashlib.sha256()
    digest.update(b"EoSInverter.active_cell_domain.v1\0")
    digest.update(np.asarray(cell_shape, dtype="<i8").tobytes(order="C"))
    digest.update(
        np.ascontiguousarray(cell_flat_indices, dtype="<i8").tobytes(order="C")
    )
    return digest.hexdigest()


def _resolve_scales(
    value: Any, field_names: Tuple[str, ...]
) -> np.ndarray:
    if isinstance(value, Mapping):
        if set(value) != set(field_names):
            missing = sorted(set(field_names) - set(value), key=str)
            extra = sorted(set(value) - set(field_names), key=str)
            details = []
            if missing:
                details.append("missing %s" % ", ".join(map(str, missing)))
            if extra:
                details.append("unexpected %s" % ", ".join(map(str, extra)))
            raise ReachabilityConfigurationError(
                "residual scales must exactly match target fields (%s)"
                % "; ".join(details)
            )
        selected = [value[name] for name in field_names]
        if any(isinstance(item, (bool, np.bool_)) for item in selected):
            raise ReachabilityConfigurationError(
                "residual scales must be numeric and cannot be boolean"
            )
        try:
            array = np.asarray(selected, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReachabilityConfigurationError(
                "residual scales must be numeric"
            ) from exc
    else:
        try:
            if not isinstance(value, np.ndarray):
                supplied_values = tuple(value)
                if any(
                    isinstance(item, (bool, np.bool_))
                    for item in supplied_values
                ):
                    raise ReachabilityConfigurationError(
                        "residual scales must be numeric and cannot be boolean"
                    )
                value = supplied_values
            raw = np.asarray(value)
            if raw.dtype.kind == "b" or any(
                isinstance(item, (bool, np.bool_)) for item in raw.reshape(-1)
            ):
                raise ReachabilityConfigurationError(
                    "residual scales must be numeric and cannot be boolean"
                )
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReachabilityConfigurationError(
                "residual scales must be numeric"
            ) from exc
    if array.shape != (len(field_names),):
        raise ReachabilityConfigurationError(
            "residual scales must have one entry per target field"
        )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        reciprocal = 1.0 / array
    if (
        not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
        or not np.all(np.isfinite(reciprocal))
    ):
        raise ReachabilityConfigurationError(
            "residual scales must be finite, positive, and have finite reciprocals"
        )
    return array


@dataclass(frozen=True)
class ReachabilityConfig:
    """Reachability method selection and bounded temporary-memory limits."""

    cell_batch_size: int = DEFAULT_REACHABILITY_CELL_BATCH_SIZE
    query_batch_size: int = DEFAULT_REACHABILITY_QUERY_BATCH_SIZE
    max_enclosure_bytes: int = DEFAULT_MAX_ENCLOSURE_BYTES
    max_pairwise_entries: int = DEFAULT_MAX_PAIRWISE_ENTRIES
    method: str = "exhaustive"
    hierarchy_levels: Optional[int] = None
    max_hierarchy_bytes: int = DEFAULT_MAX_HIERARCHY_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cell_batch_size",
            _positive_integer(self.cell_batch_size, "cell_batch_size"),
        )
        object.__setattr__(
            self,
            "query_batch_size",
            _positive_integer(self.query_batch_size, "query_batch_size"),
        )
        object.__setattr__(
            self,
            "max_enclosure_bytes",
            _positive_integer(self.max_enclosure_bytes, "max_enclosure_bytes"),
        )
        object.__setattr__(
            self,
            "max_pairwise_entries",
            _positive_integer(self.max_pairwise_entries, "max_pairwise_entries"),
        )
        object.__setattr__(
            self,
            "max_hierarchy_bytes",
            _positive_integer(self.max_hierarchy_bytes, "max_hierarchy_bytes"),
        )
        if not isinstance(self.method, str) or self.method not in REACHABILITY_METHODS:
            raise ReachabilityConfigurationError(
                "method must be one of %s" % (REACHABILITY_METHODS,)
            )
        if self.method == "exhaustive":
            if self.hierarchy_levels is not None:
                raise ReachabilityConfigurationError(
                    "hierarchy_levels must be omitted when method='exhaustive'"
                )
        else:
            if (
                isinstance(self.hierarchy_levels, (bool, np.bool_))
                or not isinstance(self.hierarchy_levels, (int, np.integer))
                or int(self.hierarchy_levels) not in REACHABILITY_HIERARCHY_LEVELS
            ):
                raise ReachabilityConfigurationError(
                    "hierarchical reachability requires hierarchy_levels in %s"
                    % (REACHABILITY_HIERARCHY_LEVELS,)
                )
            object.__setattr__(self, "hierarchy_levels", int(self.hierarchy_levels))
        if self.query_batch_size > self.max_pairwise_entries // self.cell_batch_size:
            raise ReachabilityConfigurationError(
                "query_batch_size * cell_batch_size exceeds "
                "max_pairwise_entries"
            )


def _expanded_target_boxes(
    exact_lower: np.ndarray,
    exact_upper: np.ndarray,
    scales: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand exact target boxes for the solver's accepted residual shell."""

    if tolerance == 0.0:
        padding = np.zeros_like(scales)
        return padding, exact_lower.copy(), exact_upper.copy()
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        tolerance_upper = np.nextafter(np.float64(tolerance), np.inf)
        nearest_padding = tolerance_upper * scales
    padding = np.nextafter(nearest_padding, np.inf)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lower = exact_lower - padding
        upper = exact_upper + padding
    lower = np.nextafter(lower, -np.inf)
    upper = np.nextafter(upper, np.inf)
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ReachabilityConfigurationError(
            "routing-box expansion produced NaN"
        )
    return padding, lower, upper


@dataclass(frozen=True)
class CellImageCollection:
    """Immutable, phase-tagged conservative images of source-space pieces.

    Each row describes one positive-volume source rectangle and a rigorous
    component-wise target box containing its complete multilinear image.
    ``kind_codes`` records whether the row came from an ordinary cell, a LINE
    Region S cell, a fully mixed BINODALS piece, or an unresolved binodal
    boundary piece.  The latter remains only a candidate until exact
    source-space membership is verified.
    """

    coordinate_names: Tuple[str, ...]
    field_names: Tuple[str, ...]
    parent_cell_indices: np.ndarray
    source_lower: np.ndarray
    source_upper: np.ndarray
    exact_lower: np.ndarray
    exact_upper: np.ndarray
    routing_lower: np.ndarray
    routing_upper: np.ndarray
    kind_codes: np.ndarray
    residual_scales: np.ndarray
    routing_padding: np.ndarray
    root_residual_tolerance: float
    rigorous: bool = True
    outward_rounded: bool = True

    def __post_init__(self) -> None:
        coordinates = tuple(self.coordinate_names)
        fields = tuple(self.field_names)
        dimension = len(coordinates)
        if dimension not in range(1, 5) or len(fields) != dimension:
            raise ValueError(
                "cell-image collections require a square dimension 1--4 map"
            )
        if (
            any(not isinstance(name, str) or not name for name in coordinates + fields)
            or len(set(coordinates)) != dimension
            or len(set(fields)) != dimension
        ):
            raise ValueError("cell-image coordinate and field names must be unique")

        parents = np.asarray(self.parent_cell_indices)
        if (
            parents.ndim != 2
            or parents.shape[1] != dimension
            or not np.issubdtype(parents.dtype, np.integer)
            or np.any(parents < 0)
        ):
            raise ValueError(
                "parent_cell_indices must be a non-negative integer "
                "(image_count, dimension) array"
            )
        image_count = parents.shape[0]
        expected = (image_count, dimension)
        arrays = {}
        for name in (
            "source_lower",
            "source_upper",
            "exact_lower",
            "exact_upper",
            "routing_lower",
            "routing_upper",
        ):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != expected or np.any(np.isnan(array)):
                raise ValueError("%s must be a finite-shaped image array" % name)
            arrays[name] = array
        if not all(
            np.all(np.isfinite(arrays[name]))
            for name in ("source_lower", "source_upper", "exact_lower", "exact_upper")
        ):
            raise ValueError("source and exact target boxes must be finite")
        if np.any(arrays["source_lower"] >= arrays["source_upper"]):
            raise ValueError("every source-space image piece must have positive volume")
        if np.any(arrays["exact_lower"] > arrays["exact_upper"]):
            raise ValueError("exact target lower bounds cannot exceed upper bounds")
        if (
            np.any(arrays["routing_lower"] > arrays["exact_lower"])
            or np.any(arrays["routing_upper"] < arrays["exact_upper"])
        ):
            raise ValueError("routing target boxes must contain exact target boxes")

        kinds = np.asarray(self.kind_codes)
        if kinds.shape != (image_count,) or not np.issubdtype(kinds.dtype, np.integer):
            raise ValueError("kind_codes must be one integer per image")
        valid_kinds = np.asarray([int(value) for value in CellImageKind])
        if not np.all(np.isin(kinds, valid_kinds)):
            raise ValueError("kind_codes contains an unknown cell-image kind")
        scales = _resolve_scales(self.residual_scales, fields)
        padding = np.asarray(self.routing_padding, dtype=float)
        if (
            padding.shape != (dimension,)
            or np.any(np.isnan(padding))
            or np.any(padding < 0.0)
        ):
            raise ValueError("routing_padding must be a non-negative target vector")
        tolerance = _finite_nonnegative(
            self.root_residual_tolerance, "root_residual_tolerance"
        )
        if not self.rigorous:
            raise ValueError("cell-image target boxes must be rigorous")
        if tolerance > 0.0 and not self.outward_rounded:
            raise ValueError("nonzero routing expansion must be outward rounded")

        object.__setattr__(self, "coordinate_names", coordinates)
        object.__setattr__(self, "field_names", fields)
        object.__setattr__(
            self, "parent_cell_indices", _immutable_array(parents, dtype=np.int64)
        )
        for name, array in arrays.items():
            object.__setattr__(self, name, _immutable_array(array))
        object.__setattr__(
            self, "kind_codes", _immutable_array(kinds, dtype=np.uint8)
        )
        object.__setattr__(self, "residual_scales", _immutable_array(scales))
        object.__setattr__(self, "routing_padding", _immutable_array(padding))
        object.__setattr__(self, "root_residual_tolerance", tolerance)

    @property
    def image_count(self) -> int:
        return int(self.kind_codes.size)

    @property
    def dimension(self) -> int:
        return len(self.coordinate_names)

    @property
    def boundary_candidate_mask(self) -> np.ndarray:
        return _immutable_array(
            self.kind_codes == int(CellImageKind.BINODALS_BOUNDARY), dtype=bool
        )

    @property
    def nbytes(self) -> int:
        return int(
            self.parent_cell_indices.nbytes
            + self.source_lower.nbytes
            + self.source_upper.nbytes
            + self.exact_lower.nbytes
            + self.exact_upper.nbytes
            + self.routing_lower.nbytes
            + self.routing_upper.nbytes
            + self.kind_codes.nbytes
            + self.residual_scales.nbytes
            + self.routing_padding.nbytes
        )


def build_rectilinear_cell_images(
    interpolator: MultilinearInterpolator,
    residual_scales: Any,
    root_residual_tolerance: float,
    *,
    kind: CellImageKind = CellImageKind.ORDINARY,
    config: Optional[ReachabilityConfig] = None,
) -> CellImageCollection:
    """Build tagged exact corner images for every full rectilinear cell."""

    if not isinstance(interpolator, MultilinearInterpolator):
        raise TypeError("interpolator must be a MultilinearInterpolator")
    if (
        interpolator.field_names != interpolator.target_names
        or len(interpolator.field_names) != interpolator.dimension
        or not np.all(interpolator.node_value_bounds.defined)
    ):
        raise ReachabilityConfigurationError(
            "cell images require a finite target-only square interpolator"
        )
    try:
        resolved_kind = CellImageKind(kind)
    except (TypeError, ValueError) as exc:
        raise ReachabilityConfigurationError("unknown cell-image kind") from exc
    resolved_config = ReachabilityConfig() if config is None else config
    if not isinstance(resolved_config, ReachabilityConfig):
        raise TypeError("config must be a ReachabilityConfig")
    scales = _resolve_scales(residual_scales, interpolator.field_names)
    tolerance = _finite_nonnegative(
        root_residual_tolerance, "root_residual_tolerance"
    )
    image_count = math.prod(interpolator.cell_shape)
    dimension = interpolator.dimension
    estimated_bytes = image_count * (
        dimension * np.dtype(np.int64).itemsize
        + 6 * dimension * np.dtype(np.float64).itemsize
        + np.dtype(np.uint8).itemsize
    )
    if estimated_bytes > resolved_config.max_enclosure_bytes:
        raise ReachabilityConfigurationError(
            "tagged cell images require %d bytes, exceeding "
            "max_enclosure_bytes=%d"
            % (estimated_bytes, resolved_config.max_enclosure_bytes)
        )

    flat = np.arange(image_count, dtype=np.int64)
    parents = np.column_stack(
        np.unravel_index(flat, interpolator.cell_shape, order="C")
    )
    source_lower, source_upper = interpolator.cell_coordinate_bounds(parents)
    exact_lower = np.empty((image_count, dimension), dtype=float)
    exact_upper = np.empty_like(exact_lower)
    for start in range(0, image_count, resolved_config.cell_batch_size):
        stop = min(image_count, start + resolved_config.cell_batch_size)
        bounds = interpolator.cell_value_enclosures(parents[start:stop])
        if not bounds.rigorous or not np.all(bounds.defined):
            raise ReachabilityConfigurationError(
                "cell-image target enclosures must be rigorous and finite"
            )
        exact_lower[start:stop] = bounds.lower
        exact_upper[start:stop] = bounds.upper
    padding, routing_lower, routing_upper = _expanded_target_boxes(
        exact_lower, exact_upper, scales, tolerance
    )
    return CellImageCollection(
        coordinate_names=interpolator.coordinate_names,
        field_names=interpolator.field_names,
        parent_cell_indices=parents,
        source_lower=source_lower,
        source_upper=source_upper,
        exact_lower=exact_lower,
        exact_upper=exact_upper,
        routing_lower=routing_lower,
        routing_upper=routing_upper,
        kind_codes=np.full(image_count, int(resolved_kind), dtype=np.uint8),
        residual_scales=scales,
        routing_padding=padding,
        root_residual_tolerance=tolerance,
    )


def _fraction_lower(value: Fraction) -> float:
    nearest = float(value)
    if Fraction.from_float(nearest) > value:
        nearest = float(np.nextafter(nearest, -np.inf))
    return nearest


def _fraction_upper(value: Fraction) -> float:
    nearest = float(value)
    if Fraction.from_float(nearest) < value:
        nearest = float(np.nextafter(nearest, np.inf))
    return nearest


def _exact_bilinear_subcell_enclosure(
    interpolator: MultilinearInterpolator,
    parent_cell: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return outward-rounded exact-real extrema on one 2-D subrectangle."""

    cell_lower, cell_upper = interpolator.cell_coordinate_bounds(parent_cell)
    cell_lower = np.asarray(cell_lower, dtype=float)
    cell_upper = np.asarray(cell_upper, dtype=float)
    source_lower = np.asarray(source_lower, dtype=float)
    source_upper = np.asarray(source_upper, dtype=float)
    if (
        source_lower.shape != (2,)
        or source_upper.shape != (2,)
        or np.any(source_lower < cell_lower)
        or np.any(source_upper > cell_upper)
        or np.any(source_lower >= source_upper)
    ):
        raise ReachabilityConfigurationError(
            "a binodal source piece must be a positive subrectangle of its "
            "parent interpolation cell"
        )
    if np.array_equal(source_lower, cell_lower) and np.array_equal(
        source_upper, cell_upper
    ):
        bounds = interpolator.cell_value_enclosures(parent_cell)
        return np.asarray(bounds.lower), np.asarray(bounds.upper)

    normalized = []
    for axis in range(2):
        axis_lower = Fraction.from_float(float(cell_lower[axis]))
        width = Fraction.from_float(float(cell_upper[axis])) - axis_lower
        normalized.append(
            tuple(
                (
                    Fraction.from_float(float(value)) - axis_lower
                ) / width
                for value in (source_lower[axis], source_upper[axis])
            )
        )
    corners = np.asarray(
        interpolator.cell_corner_values(parent_cell), dtype=float
    ).reshape(2, 2, 2)
    exact_corners = np.empty(corners.shape, dtype=object)
    for index in np.ndindex(corners.shape):
        exact_corners[index] = Fraction.from_float(float(corners[index]))

    values: List[List[Fraction]] = [[], []]
    for first in normalized[0]:
        for second in normalized[1]:
            one_first = Fraction(1) - first
            one_second = Fraction(1) - second
            for field in range(2):
                value = (
                    exact_corners[0, 0, field] * one_first * one_second
                    + exact_corners[0, 1, field] * one_first * second
                    + exact_corners[1, 0, field] * first * one_second
                    + exact_corners[1, 1, field] * first * second
                )
                values[field].append(value)
    lower = np.asarray(
        [_fraction_lower(min(field_values)) for field_values in values]
    )
    upper = np.asarray(
        [_fraction_upper(max(field_values)) for field_values in values]
    )
    return lower, upper


def _binodal_interval_breakpoints(
    band: BinodalBand, lower: float, upper: float
) -> np.ndarray:
    points = [float(lower), float(upper)]
    knots = band.muB[(band.muB > lower) & (band.muB < upper)]
    points.extend(float(value) for value in knots)
    base = np.asarray(sorted(set(points)), dtype=float)
    crossings = []
    for left, right in zip(base[:-1], base[1:]):
        curves = np.asarray(band.evaluate([left, right]), dtype=float)
        difference = curves[:, 0] - curves[:, 1]
        if difference[0] * difference[1] < 0.0:
            fraction = -difference[0] / (difference[1] - difference[0])
            crossing = left + fraction * (right - left)
            if left < crossing < right:
                crossings.append(float(crossing))
    return np.asarray(sorted(set(points + crossings)), dtype=float)


def _classify_binodal_source_box(
    band: BinodalBand, lower: np.ndarray, upper: np.ndarray
) -> str:
    bounds = np.asarray(band.temperature_bounds([lower[1], upper[1]]))
    lower_curve = bounds[:, 0]
    upper_curve = bounds[:, 1]
    minimum_lower = np.nextafter(np.min(lower_curve), -np.inf)
    maximum_lower = np.nextafter(np.max(lower_curve), np.inf)
    minimum_upper = np.nextafter(np.min(upper_curve), -np.inf)
    maximum_upper = np.nextafter(np.max(upper_curve), np.inf)
    if upper[0] < minimum_lower or lower[0] > maximum_upper:
        return "outside"
    if lower[0] >= maximum_lower and upper[0] <= minimum_upper:
        return "inside"
    return "boundary"


def build_binodal_cell_images(
    interpolator: MultilinearInterpolator,
    band: BinodalBand,
    residual_scales: Any,
    root_residual_tolerance: float,
    *,
    subdivision: Optional[BinodalSubdivisionConfig] = None,
    config: Optional[ReachabilityConfig] = None,
) -> CellImageCollection:
    """Conservatively cover the BINODALS subset of ordinary source cells."""

    if not isinstance(interpolator, MultilinearInterpolator):
        raise TypeError("interpolator must be a MultilinearInterpolator")
    if not isinstance(band, BinodalBand):
        raise TypeError("band must be a BinodalBand")
    if (
        interpolator.coordinate_names != ("T", "muB")
        or interpolator.field_names != ("e", "nB")
        or interpolator.target_names != ("e", "nB")
        or interpolator.dimension != 2
        or not np.all(interpolator.node_value_bounds.defined)
    ):
        raise ReachabilityConfigurationError(
            "BINODALS images require a finite canonical "
            "(T, muB) -> (e, nB) interpolator"
        )
    subdivision = (
        BinodalSubdivisionConfig() if subdivision is None else subdivision
    )
    if not isinstance(subdivision, BinodalSubdivisionConfig):
        raise TypeError("subdivision must be a BinodalSubdivisionConfig")
    resolved_config = ReachabilityConfig() if config is None else config
    if not isinstance(resolved_config, ReachabilityConfig):
        raise TypeError("config must be a ReachabilityConfig")
    scales = _resolve_scales(residual_scales, interpolator.field_names)
    tolerance = _finite_nonnegative(
        root_residual_tolerance, "root_residual_tolerance"
    )

    axes = interpolator.axes
    parents: List[np.ndarray] = []
    source_lowers: List[np.ndarray] = []
    source_uppers: List[np.ndarray] = []
    kinds: List[int] = []

    def retain(
        parent: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        kind: CellImageKind,
    ) -> None:
        if len(kinds) >= subdivision.maximum_images:
            raise ReachabilityConfigurationError(
                "binodal subdivision exceeds maximum_images=%d"
                % subdivision.maximum_images
            )
        parents.append(np.asarray(parent, dtype=np.int64))
        source_lowers.append(np.asarray(lower, dtype=float))
        source_uppers.append(np.asarray(upper, dtype=float))
        kinds.append(int(kind))

    for temperature_cell in range(interpolator.cell_shape[0]):
        temperature_lower = float(axes[0][temperature_cell])
        temperature_upper = float(axes[0][temperature_cell + 1])
        for chemical_cell in range(interpolator.cell_shape[1]):
            cell_muB_lower = float(axes[1][chemical_cell])
            cell_muB_upper = float(axes[1][chemical_cell + 1])
            muB_lower = max(cell_muB_lower, band.muB_bounds[0])
            muB_upper = min(cell_muB_upper, band.muB_bounds[1])
            if muB_lower >= muB_upper:
                continue
            breakpoints = _binodal_interval_breakpoints(
                band, muB_lower, muB_upper
            )
            parent = np.asarray([temperature_cell, chemical_cell], dtype=np.int64)
            parent_widths = np.asarray(
                [
                    temperature_upper - temperature_lower,
                    cell_muB_upper - cell_muB_lower,
                ]
            )
            for left_muB, right_muB in zip(breakpoints[:-1], breakpoints[1:]):
                initial_lower = np.asarray([temperature_lower, left_muB])
                initial_upper = np.asarray([temperature_upper, right_muB])
                stack = [(initial_lower, initial_upper, 0)]
                while stack:
                    lower, upper, depth = stack.pop()
                    classification = _classify_binodal_source_box(
                        band, lower, upper
                    )
                    if classification == "outside":
                        continue
                    if classification == "inside":
                        retain(
                            parent,
                            lower,
                            upper,
                            CellImageKind.BINODALS_INTERIOR,
                        )
                        continue
                    widths = upper - lower
                    at_tolerance = (
                        widths[0] <= subdivision.temperature_tolerance
                        and widths[1]
                        <= subdivision.chemical_potential_tolerance
                    )
                    if depth >= subdivision.maximum_depth or at_tolerance:
                        retain(
                            parent,
                            lower,
                            upper,
                            CellImageKind.BINODALS_BOUNDARY,
                        )
                        continue
                    resolution = np.asarray(
                        [
                            max(
                                subdivision.temperature_tolerance,
                                parent_widths[0] / (2 ** subdivision.maximum_depth),
                            ),
                            max(
                                subdivision.chemical_potential_tolerance,
                                parent_widths[1] / (2 ** subdivision.maximum_depth),
                            ),
                        ]
                    )
                    splittable = widths > resolution
                    if not np.any(splittable):
                        retain(
                            parent,
                            lower,
                            upper,
                            CellImageKind.BINODALS_BOUNDARY,
                        )
                        continue
                    scores = np.where(splittable, widths / resolution, -np.inf)
                    axis = int(np.argmax(scores))
                    midpoint = lower[axis] / 2.0 + upper[axis] / 2.0
                    if not lower[axis] < midpoint < upper[axis]:
                        retain(
                            parent,
                            lower,
                            upper,
                            CellImageKind.BINODALS_BOUNDARY,
                        )
                        continue
                    lower_piece_upper = np.array(upper, copy=True)
                    lower_piece_upper[axis] = midpoint
                    upper_piece_lower = np.array(lower, copy=True)
                    upper_piece_lower[axis] = midpoint
                    stack.append((upper_piece_lower, upper, depth + 1))
                    stack.append((lower, lower_piece_upper, depth + 1))

    image_count = len(kinds)
    if image_count:
        parent_array = np.stack(parents)
        source_lower_array = np.stack(source_lowers)
        source_upper_array = np.stack(source_uppers)
    else:
        parent_array = np.empty((0, 2), dtype=np.int64)
        source_lower_array = np.empty((0, 2), dtype=float)
        source_upper_array = np.empty((0, 2), dtype=float)
    estimated_bytes = image_count * (
        2 * np.dtype(np.int64).itemsize
        + 12 * np.dtype(np.float64).itemsize
        + np.dtype(np.uint8).itemsize
    )
    if estimated_bytes > resolved_config.max_enclosure_bytes:
        raise ReachabilityConfigurationError(
            "binodal cell images require %d bytes, exceeding "
            "max_enclosure_bytes=%d"
            % (estimated_bytes, resolved_config.max_enclosure_bytes)
        )
    exact_lower = np.empty((image_count, 2), dtype=float)
    exact_upper = np.empty_like(exact_lower)
    for index in range(image_count):
        exact_lower[index], exact_upper[index] = (
            _exact_bilinear_subcell_enclosure(
                interpolator,
                parent_array[index],
                source_lower_array[index],
                source_upper_array[index],
            )
        )
    padding, routing_lower, routing_upper = _expanded_target_boxes(
        exact_lower, exact_upper, scales, tolerance
    )
    return CellImageCollection(
        coordinate_names=interpolator.coordinate_names,
        field_names=interpolator.field_names,
        parent_cell_indices=parent_array,
        source_lower=source_lower_array,
        source_upper=source_upper_array,
        exact_lower=exact_lower,
        exact_upper=exact_upper,
        routing_lower=routing_lower,
        routing_upper=routing_upper,
        kind_codes=np.asarray(kinds, dtype=np.uint8),
        residual_scales=scales,
        routing_padding=padding,
        root_residual_tolerance=tolerance,
    )


@dataclass(frozen=True)
class CellTargetEnclosures:
    """Exact and tolerance-expanded boxes for all active source cells."""

    field_names: Tuple[str, ...]
    cell_shape: Tuple[int, ...]
    cell_flat_indices: np.ndarray
    exact_lower: np.ndarray
    exact_upper: np.ndarray
    routing_lower: np.ndarray
    routing_upper: np.ndarray
    residual_scales: np.ndarray
    routing_padding: np.ndarray
    root_residual_tolerance: float
    rigorous: bool = True
    outward_rounded: bool = True

    def __post_init__(self) -> None:
        names = tuple(self.field_names)
        cell_shape = tuple(int(size) for size in self.cell_shape)
        dimension = len(names)
        if dimension not in range(1, 5) or len(cell_shape) != dimension:
            raise ValueError("field names and cell shape must have dimension 1--4")
        if any(size <= 0 for size in cell_shape):
            raise ValueError("every cell-shape axis must be positive")
        flat = np.asarray(self.cell_flat_indices)
        if flat.ndim != 1 or not np.issubdtype(flat.dtype, np.integer):
            raise ValueError("cell_flat_indices must be a one-dimensional integer array")
        total_cells = math.prod(cell_shape)
        if (
            np.any(flat < 0)
            or np.any(flat >= total_cells)
            or np.any(np.diff(flat.astype(np.int64, copy=False)) <= 0)
        ):
            raise ValueError("active cell indices must be unique, sorted, and in range")
        expected_shape = (flat.size, dimension)
        exact_lower = np.asarray(self.exact_lower, dtype=float)
        exact_upper = np.asarray(self.exact_upper, dtype=float)
        routing_lower = np.asarray(self.routing_lower, dtype=float)
        routing_upper = np.asarray(self.routing_upper, dtype=float)
        for label, array in (
            ("exact_lower", exact_lower),
            ("exact_upper", exact_upper),
            ("routing_lower", routing_lower),
            ("routing_upper", routing_upper),
        ):
            if array.shape != expected_shape:
                raise ValueError("%s has the wrong shape" % label)
            if np.any(np.isnan(array)):
                raise ValueError("%s cannot contain NaN" % label)
        if not np.all(np.isfinite(exact_lower)) or not np.all(np.isfinite(exact_upper)):
            raise ValueError("exact cell enclosures must be finite")
        if np.any(exact_lower > exact_upper):
            raise ValueError("exact cell lower bounds cannot exceed upper bounds")
        if np.any(routing_lower > exact_lower) or np.any(routing_upper < exact_upper):
            raise ValueError("routing boxes must contain the exact boxes")
        scales = _resolve_scales(self.residual_scales, names)
        padding = np.asarray(self.routing_padding, dtype=float)
        if padding.shape != (dimension,) or np.any(np.isnan(padding)) or np.any(padding < 0.0):
            raise ValueError("routing padding must be a non-negative target vector")
        tolerance = _finite_nonnegative(
            self.root_residual_tolerance, "root_residual_tolerance"
        )
        if not self.rigorous:
            raise ValueError("CellTargetEnclosures requires a rigorous exact box")
        if tolerance > 0.0 and not self.outward_rounded:
            raise ValueError("nonzero routing expansion must be outward rounded")
        object.__setattr__(self, "field_names", names)
        object.__setattr__(self, "cell_shape", cell_shape)
        object.__setattr__(
            self, "cell_flat_indices", _immutable_array(flat, dtype=np.int64)
        )
        object.__setattr__(self, "exact_lower", _immutable_array(exact_lower))
        object.__setattr__(self, "exact_upper", _immutable_array(exact_upper))
        object.__setattr__(self, "routing_lower", _immutable_array(routing_lower))
        object.__setattr__(self, "routing_upper", _immutable_array(routing_upper))
        object.__setattr__(self, "residual_scales", _immutable_array(scales))
        object.__setattr__(self, "routing_padding", _immutable_array(padding))
        object.__setattr__(self, "root_residual_tolerance", tolerance)

    @property
    def dimension(self) -> int:
        return len(self.field_names)

    @property
    def active_cell_count(self) -> int:
        return int(self.cell_flat_indices.size)

    @property
    def nbytes(self) -> int:
        return int(
            self.cell_flat_indices.nbytes
            + self.exact_lower.nbytes
            + self.exact_upper.nbytes
            + self.routing_lower.nbytes
            + self.routing_upper.nbytes
            + self.residual_scales.nbytes
            + self.routing_padding.nbytes
        )

    def unravel_cells(self, flat_indices: Any) -> np.ndarray:
        """Convert valid C-order flat cell indices to ``(..., dimension)``."""

        array = np.asarray(flat_indices)
        if not np.issubdtype(array.dtype, np.integer):
            raise ReachabilityInputError("cell flat indices must be integers")
        if np.any(array < 0) or np.any(array >= math.prod(self.cell_shape)):
            raise ReachabilityInputError("cell flat index is outside the cell grid")
        flat = array.reshape(-1).astype(np.int64, copy=False)
        if flat.size:
            coordinates = np.column_stack(
                np.unravel_index(flat, self.cell_shape, order="C")
            )
        else:
            coordinates = np.empty((0, self.dimension), dtype=np.int64)
        return _immutable_array(
            coordinates.reshape(array.shape + (self.dimension,)), dtype=np.int64
        )


@dataclass(frozen=True, eq=False)
class ReachabilityProvenance:
    """Identity of the source domain and routing-box proof contract.

    Candidate cell IDs authorize global exclusion only when this identity
    matches the consumer's expected table, active domain, fields, tolerance,
    and reachability method exactly.
    """

    table_fingerprint: str
    active_cell_fingerprint: str
    coordinate_names: Tuple[str, ...]
    field_names: Tuple[str, ...]
    cell_shape: Tuple[int, ...]
    active_cell_count: int
    residual_scales: np.ndarray
    root_residual_tolerance: float
    method: str
    hierarchy_levels: Optional[int]
    routing_contract_version: int = REACHABILITY_ROUTING_CONTRACT_VERSION
    cell_order: str = "C"

    def __post_init__(self) -> None:
        for label in ("table_fingerprint", "active_cell_fingerprint"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ReachabilityLocalizationError(
                    "%s must be a non-empty string" % label
                )

        try:
            coordinate_names = tuple(self.coordinate_names)
            field_names = tuple(self.field_names)
        except TypeError as exc:
            raise ReachabilityLocalizationError(
                "coordinate_names and field_names must be iterable"
            ) from exc
        if not coordinate_names or len(coordinate_names) != len(field_names):
            raise ReachabilityLocalizationError(
                "coordinate_names and field_names must have the same positive length"
            )
        for label, names in (
            ("coordinate_names", coordinate_names),
            ("field_names", field_names),
        ):
            if any(not isinstance(name, str) or not name for name in names):
                raise ReachabilityLocalizationError(
                    "%s must contain non-empty strings" % label
                )
            if len(set(names)) != len(names):
                raise ReachabilityLocalizationError(
                    "%s cannot contain duplicates" % label
                )

        try:
            raw_shape = tuple(self.cell_shape)
        except TypeError as exc:
            raise ReachabilityLocalizationError(
                "cell_shape must be iterable"
            ) from exc
        if len(raw_shape) != len(field_names):
            raise ReachabilityLocalizationError(
                "cell_shape must have one entry per target field"
            )
        cell_shape = []
        for size in raw_shape:
            if isinstance(size, (bool, np.bool_)) or not isinstance(
                size, (int, np.integer)
            ):
                raise ReachabilityLocalizationError(
                    "cell_shape entries must be positive integers"
                )
            converted = int(size)
            if converted <= 0:
                raise ReachabilityLocalizationError(
                    "cell_shape entries must be positive integers"
                )
            cell_shape.append(converted)
        cell_shape_tuple = tuple(cell_shape)

        active_count = self.active_cell_count
        if isinstance(active_count, (bool, np.bool_)) or not isinstance(
            active_count, (int, np.integer)
        ):
            raise ReachabilityLocalizationError(
                "active_cell_count must be a positive integer"
            )
        active_count = int(active_count)
        if active_count <= 0 or active_count > math.prod(cell_shape_tuple):
            raise ReachabilityLocalizationError(
                "active_cell_count is outside the source-cell domain"
            )

        try:
            scale_values = self.residual_scales
            if not isinstance(scale_values, np.ndarray):
                scale_values = tuple(scale_values)
                if any(
                    isinstance(value, (bool, np.bool_))
                    for value in scale_values
                ):
                    raise ReachabilityLocalizationError(
                        "residual_scales cannot contain booleans"
                    )
            raw_scales = np.asarray(scale_values)
            if raw_scales.dtype.kind == "b" or any(
                isinstance(value, (bool, np.bool_))
                for value in raw_scales.reshape(-1)
            ):
                raise ReachabilityLocalizationError(
                    "residual_scales cannot contain booleans"
                )
            scales = np.asarray(scale_values, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReachabilityLocalizationError(
                "residual_scales must be numeric"
            ) from exc
        if scales.shape != (len(field_names),):
            raise ReachabilityLocalizationError(
                "residual_scales must have one entry per target field"
            )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            reciprocal = 1.0 / scales
        if (
            not np.all(np.isfinite(scales))
            or np.any(scales <= 0.0)
            or not np.all(np.isfinite(reciprocal))
        ):
            raise ReachabilityLocalizationError(
                "residual_scales must be finite, positive, and invertible"
            )

        tolerance = self.root_residual_tolerance
        if isinstance(tolerance, (bool, np.bool_)):
            raise ReachabilityLocalizationError(
                "root_residual_tolerance must be finite and non-negative"
            )
        try:
            tolerance = float(tolerance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReachabilityLocalizationError(
                "root_residual_tolerance must be finite and non-negative"
            ) from exc
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ReachabilityLocalizationError(
                "root_residual_tolerance must be finite and non-negative"
            )

        method = self.method
        if method not in REACHABILITY_METHODS:
            raise ReachabilityLocalizationError(
                "method must be one of %s" % ", ".join(REACHABILITY_METHODS)
            )
        levels = self.hierarchy_levels
        if method == "exhaustive":
            if levels is not None:
                raise ReachabilityLocalizationError(
                    "exhaustive provenance cannot have hierarchy_levels"
                )
        else:
            if isinstance(levels, (bool, np.bool_)) or not isinstance(
                levels, (int, np.integer)
            ):
                raise ReachabilityLocalizationError(
                    "hierarchical provenance requires positive hierarchy_levels"
                )
            levels = int(levels)
            if levels <= 0:
                raise ReachabilityLocalizationError(
                    "hierarchical provenance requires positive hierarchy_levels"
                )

        version = self.routing_contract_version
        if isinstance(version, (bool, np.bool_)) or not isinstance(
            version, (int, np.integer)
        ):
            raise ReachabilityLocalizationError(
                "routing_contract_version must be an integer"
            )
        version = int(version)
        if version != REACHABILITY_ROUTING_CONTRACT_VERSION:
            raise ReachabilityLocalizationError(
                "unsupported routing_contract_version %d" % version
            )
        if self.cell_order != "C":
            raise ReachabilityLocalizationError(
                "candidate cell IDs currently require C-order"
            )

        object.__setattr__(self, "coordinate_names", coordinate_names)
        object.__setattr__(self, "field_names", field_names)
        object.__setattr__(self, "cell_shape", cell_shape_tuple)
        object.__setattr__(self, "active_cell_count", active_count)
        object.__setattr__(self, "residual_scales", _immutable_array(scales))
        object.__setattr__(self, "root_residual_tolerance", tolerance)
        object.__setattr__(self, "hierarchy_levels", levels)
        object.__setattr__(self, "routing_contract_version", version)

    def compatibility_mismatches(
        self, expected: "ReachabilityProvenance"
    ) -> Tuple[str, ...]:
        """Return proof-relevant fields that differ from ``expected``."""

        if not isinstance(expected, ReachabilityProvenance):
            return ("type",)
        mismatches = []
        for name in (
            "table_fingerprint",
            "active_cell_fingerprint",
            "coordinate_names",
            "field_names",
            "cell_shape",
            "active_cell_count",
            "root_residual_tolerance",
            "method",
            "hierarchy_levels",
            "routing_contract_version",
            "cell_order",
        ):
            if getattr(self, name) != getattr(expected, name):
                mismatches.append(name)
        if not np.array_equal(self.residual_scales, expected.residual_scales):
            mismatches.append("residual_scales")
        return tuple(mismatches)

    def is_compatible_with(self, expected: "ReachabilityProvenance") -> bool:
        return not self.compatibility_mismatches(expected)

    def require_compatible_with(
        self, expected: "ReachabilityProvenance"
    ) -> None:
        mismatches = self.compatibility_mismatches(expected)
        if mismatches:
            raise ReachabilityLocalizationError(
                "reachability provenance mismatch: %s"
                % ", ".join(mismatches)
            )


@dataclass(frozen=True)
class ReachabilityMicrobatch:
    """One contiguous target range whose complete CSR rows stay together."""

    target_start: int
    target_stop: int
    pair_start: int
    pair_stop: int

    def __post_init__(self) -> None:
        values = []
        for name in ("target_start", "target_stop", "pair_start", "pair_stop"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ReachabilityLocalizationError(
                    "%s must be a non-negative integer" % name
                )
            converted = int(value)
            if converted < 0:
                raise ReachabilityLocalizationError(
                    "%s must be a non-negative integer" % name
                )
            values.append(converted)
        target_start, target_stop, pair_start, pair_stop = values
        if target_stop <= target_start:
            raise ReachabilityLocalizationError(
                "a localization microbatch must contain at least one target"
            )
        if pair_stop < pair_start:
            raise ReachabilityLocalizationError(
                "microbatch pair bounds are reversed"
            )
        object.__setattr__(self, "target_start", target_start)
        object.__setattr__(self, "target_stop", target_stop)
        object.__setattr__(self, "pair_start", pair_start)
        object.__setattr__(self, "pair_stop", pair_stop)

    @property
    def target_count(self) -> int:
        return self.target_stop - self.target_start

    @property
    def pair_count(self) -> int:
        return self.pair_stop - self.pair_start

    @property
    def csr_nbytes(self) -> int:
        return int(
            (self.target_count + 1 + self.pair_count) * np.dtype(np.int64).itemsize
        )


@dataclass(frozen=True)
class ReachabilityResult:
    """Tri-state pre-screen results for one target query batch."""

    field_names: Tuple[str, ...]
    cell_shape: Tuple[int, ...]
    targets: np.ndarray
    prescreen_codes: np.ndarray
    exact_candidate: np.ndarray
    routing_witness_flat: np.ndarray
    exact_witness_flat: np.ndarray

    def __post_init__(self) -> None:
        names = tuple(self.field_names)
        cell_shape = tuple(int(size) for size in self.cell_shape)
        dimension = len(names)
        targets = np.asarray(self.targets, dtype=float)
        if targets.ndim < 1 or targets.shape[-1] != dimension:
            raise ValueError("targets must end with the target dimension")
        query_shape = targets.shape[:-1]
        prescreen = np.asarray(self.prescreen_codes)
        exact = np.asarray(self.exact_candidate, dtype=bool)
        routing_witness = np.asarray(self.routing_witness_flat)
        exact_witness = np.asarray(self.exact_witness_flat)
        for label, array in (
            ("prescreen_codes", prescreen),
            ("exact_candidate", exact),
            ("routing_witness_flat", routing_witness),
            ("exact_witness_flat", exact_witness),
        ):
            if array.shape != query_shape:
                raise ValueError("%s does not match the target query shape" % label)
        if not np.issubdtype(prescreen.dtype, np.integer):
            raise ValueError("prescreen codes must be integers")
        if not np.issubdtype(routing_witness.dtype, np.integer) or not np.issubdtype(
            exact_witness.dtype, np.integer
        ):
            raise ValueError("witness cell indices must be integers")
        not_run = pipeline_status_code(PrescreenStatus.NOT_RUN)
        candidate = pipeline_status_code(PrescreenStatus.CANDIDATE)
        excluded = pipeline_status_code(PrescreenStatus.DEFINITELY_UNREACHABLE)
        if not np.all(np.isin(prescreen, (not_run, candidate, excluded))):
            raise ValueError("prescreen result contains an unknown status code")
        evaluated = prescreen != not_run
        candidate_mask = prescreen == candidate
        excluded_mask = prescreen == excluded
        total_cells = math.prod(cell_shape)
        if np.any((routing_witness < -1) | (routing_witness >= total_cells)):
            raise ValueError("routing witness is outside the cell grid")
        if np.any((exact_witness < -1) | (exact_witness >= total_cells)):
            raise ValueError("exact witness is outside the cell grid")
        if np.any(candidate_mask != (routing_witness >= 0)):
            raise ValueError("routing candidate and witness columns disagree")
        if np.any(exact != (exact_witness >= 0)):
            raise ValueError("exact candidate and witness columns disagree")
        if np.any(exact & ~candidate_mask):
            raise ValueError("an exact candidate must also be a routing candidate")
        if np.any(excluded_mask & (routing_witness >= 0)):
            raise ValueError("an excluded target cannot have a witness cell")
        if np.any(~evaluated & ((routing_witness >= 0) | (exact_witness >= 0))):
            raise ValueError("a non-evaluated target cannot have a witness cell")
        object.__setattr__(self, "field_names", names)
        object.__setattr__(self, "cell_shape", cell_shape)
        object.__setattr__(self, "targets", _immutable_array(targets))
        object.__setattr__(
            self, "prescreen_codes", _immutable_array(prescreen, dtype=np.uint8)
        )
        object.__setattr__(self, "exact_candidate", _immutable_array(exact))
        object.__setattr__(
            self,
            "routing_witness_flat",
            _immutable_array(routing_witness, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "exact_witness_flat",
            _immutable_array(exact_witness, dtype=np.int64),
        )

    @property
    def query_shape(self) -> Tuple[int, ...]:
        return self.targets.shape[:-1]

    def _mask(self, value: Any) -> np.ndarray:
        return _immutable_array(np.asarray(value, dtype=bool))

    @property
    def evaluated_mask(self) -> np.ndarray:
        return self._mask(
            self.prescreen_codes
            != pipeline_status_code(PrescreenStatus.NOT_RUN)
        )

    @property
    def candidate_mask(self) -> np.ndarray:
        """Membership in at least one tolerance-expanded routing box."""

        return self._mask(
            self.prescreen_codes
            == pipeline_status_code(PrescreenStatus.CANDIDATE)
        )

    @property
    def definitely_unreachable_mask(self) -> np.ndarray:
        """Targets outside every tolerance-expanded active-cell box."""

        return self._mask(
            self.prescreen_codes
            == pipeline_status_code(PrescreenStatus.DEFINITELY_UNREACHABLE)
        )

    @property
    def exact_candidate_mask(self) -> np.ndarray:
        return self._mask(self.exact_candidate)

    @property
    def exact_excluded_mask(self) -> np.ndarray:
        return self._mask(self.evaluated_mask & ~self.exact_candidate)

    @property
    def tolerance_shell_mask(self) -> np.ndarray:
        """Routing candidates lying outside every exact cell-image box."""

        return self._mask(self.candidate_mask & ~self.exact_candidate)

    def _witness_cells(self, flat_witness: np.ndarray) -> np.ndarray:
        flat = np.asarray(flat_witness).reshape(-1)
        result = np.full((flat.size, len(self.cell_shape)), -1, dtype=np.int64)
        valid = flat >= 0
        if np.any(valid):
            result[valid] = np.column_stack(
                np.unravel_index(flat[valid], self.cell_shape, order="C")
            )
        return _immutable_array(
            result.reshape(self.query_shape + (len(self.cell_shape),)),
            dtype=np.int64,
        )

    @property
    def routing_witness_cells(self) -> np.ndarray:
        return self._witness_cells(self.routing_witness_flat)

    @property
    def exact_witness_cells(self) -> np.ndarray:
        return self._witness_cells(self.exact_witness_flat)


@dataclass(frozen=True)
class ReachabilityLocalization:
    """Complete or explicitly incomplete CSR routing-cell localization.

    Rows follow flattened target C-order.  Within each row, global source-cell
    IDs are unique and strictly increasing.  Only ``COMPLETE`` localization
    with matching provenance can authorize a global no-root conclusion.
    """

    result: ReachabilityResult
    offsets: np.ndarray
    cell_flat_indices: np.ndarray
    completeness: LocalizationCompleteness
    provenance: ReachabilityProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.result, ReachabilityResult):
            raise ReachabilityLocalizationError(
                "result must be a ReachabilityResult"
            )
        if not isinstance(self.provenance, ReachabilityProvenance):
            raise ReachabilityLocalizationError(
                "provenance must be ReachabilityProvenance"
            )
        try:
            completeness = LocalizationCompleteness(self.completeness)
        except (TypeError, ValueError) as exc:
            raise ReachabilityLocalizationError(
                "unknown localization completeness"
            ) from exc

        if self.result.field_names != self.provenance.field_names:
            raise ReachabilityLocalizationError(
                "localization result fields do not match provenance"
            )
        if self.result.cell_shape != self.provenance.cell_shape:
            raise ReachabilityLocalizationError(
                "localization result cell shape does not match provenance"
            )

        raw_offsets = np.asarray(self.offsets)
        raw_cells = np.asarray(self.cell_flat_indices)
        if raw_offsets.dtype.kind not in "iu" or raw_offsets.dtype.kind == "b":
            raise ReachabilityLocalizationError("CSR offsets must be integers")
        if raw_cells.dtype.kind not in "iu" or raw_cells.dtype.kind == "b":
            raise ReachabilityLocalizationError(
                "candidate cell flat indices must be integers"
            )
        offsets = np.asarray(raw_offsets, dtype=np.int64)
        cells = np.asarray(raw_cells, dtype=np.int64)
        target_count = math.prod(self.result.query_shape)
        if offsets.shape != (target_count + 1,):
            raise ReachabilityLocalizationError(
                "CSR offsets must have one entry beyond every flattened target"
            )
        if cells.ndim != 1:
            raise ReachabilityLocalizationError(
                "candidate cell flat indices must be one-dimensional"
            )
        if int(offsets[0]) != 0:
            raise ReachabilityLocalizationError("CSR offsets must start at zero")
        if np.any(np.diff(offsets) < 0):
            raise ReachabilityLocalizationError(
                "CSR offsets must be nondecreasing"
            )
        if int(offsets[-1]) != int(cells.size):
            raise ReachabilityLocalizationError(
                "the final CSR offset must equal the candidate-pair count"
            )
        total_cells = math.prod(self.result.cell_shape)
        if np.any(cells < 0) or np.any(cells >= total_cells):
            raise ReachabilityLocalizationError(
                "candidate cell ID is outside the source-cell grid"
            )

        counts = np.diff(offsets)
        candidate_mask = self.result.candidate_mask.reshape(-1)
        if np.any(candidate_mask & (counts == 0)):
            raise ReachabilityLocalizationError(
                "every candidate target must retain at least one cell"
            )
        if np.any(~candidate_mask & (counts != 0)):
            raise ReachabilityLocalizationError(
                "non-candidate targets cannot retain candidate cells"
            )

        routing_witness = self.result.routing_witness_flat.reshape(-1)
        exact_witness = self.result.exact_witness_flat.reshape(-1)
        for target_index in range(target_count):
            start = int(offsets[target_index])
            stop = int(offsets[target_index + 1])
            row = cells[start:stop]
            if row.size == 0:
                continue
            if row.size > 1 and np.any(np.diff(row) <= 0):
                raise ReachabilityLocalizationError(
                    "candidate cell IDs must be unique and strictly increasing "
                    "within each target row"
                )
            if int(row[0]) != int(routing_witness[target_index]):
                raise ReachabilityLocalizationError(
                    "the first candidate cell must equal the routing witness"
                )
            exact = int(exact_witness[target_index])
            if exact >= 0 and not np.any(row == exact):
                raise ReachabilityLocalizationError(
                    "the exact witness must belong to the routing candidate row"
                )

        object.__setattr__(self, "offsets", _immutable_array(offsets, dtype=np.int64))
        object.__setattr__(
            self,
            "cell_flat_indices",
            _immutable_array(cells, dtype=np.int64),
        )
        object.__setattr__(self, "completeness", completeness)

    @property
    def target_count(self) -> int:
        return math.prod(self.result.query_shape)

    @property
    def pair_count(self) -> int:
        return int(self.cell_flat_indices.size)

    @property
    def candidate_counts(self) -> np.ndarray:
        return _immutable_array(np.diff(self.offsets), dtype=np.int64)

    @property
    def csr_nbytes(self) -> int:
        return int(self.offsets.nbytes + self.cell_flat_indices.nbytes)

    @property
    def nbytes(self) -> int:
        """Bytes owned by the ragged CSR candidate representation."""

        return self.csr_nbytes

    @property
    def is_complete(self) -> bool:
        return self.completeness is LocalizationCompleteness.COMPLETE

    def cells_for(self, flat_target_index: int) -> np.ndarray:
        """Return a zero-copy immutable candidate-ID row for one target."""

        if isinstance(flat_target_index, (bool, np.bool_)) or not isinstance(
            flat_target_index, (int, np.integer)
        ):
            raise ReachabilityLocalizationError(
                "flat_target_index must be an integer"
            )
        index = int(flat_target_index)
        if index < 0 or index >= self.target_count:
            raise ReachabilityLocalizationError(
                "flat_target_index is outside the localization"
            )
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return self.cell_flat_indices[start:stop]

    def is_authoritative_for(self, expected: ReachabilityProvenance) -> bool:
        return self.is_complete and self.provenance.is_compatible_with(expected)

    def require_complete_for(self, expected: ReachabilityProvenance) -> None:
        """Require proof authority before making a global exclusion claim."""

        if not self.is_complete:
            raise ReachabilityLocalizationError(
                "incomplete candidate localization cannot authorize global exclusion"
            )
        self.provenance.require_compatible_with(expected)

    def plan_microbatches(
        self,
        *,
        maximum_candidate_pairs: int,
        maximum_csr_bytes: int,
    ) -> Tuple[ReachabilityMicrobatch, ...]:
        """Partition target rows without splitting or truncating any row.

        A single target that exceeds either limit raises instead of returning
        a partial candidate row.
        """

        limits = []
        for name, value in (
            ("maximum_candidate_pairs", maximum_candidate_pairs),
            ("maximum_csr_bytes", maximum_csr_bytes),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ReachabilityLocalizationError(
                    "%s must be a positive integer" % name
                )
            converted = int(value)
            if converted <= 0:
                raise ReachabilityLocalizationError(
                    "%s must be a positive integer" % name
                )
            limits.append(converted)
        pair_limit, byte_limit = limits

        batches = []
        start = 0
        while start < self.target_count:
            pair_start = int(self.offsets[start])
            stop = start
            while stop < self.target_count:
                proposed_stop = stop + 1
                pair_stop = int(self.offsets[proposed_stop])
                pair_count = pair_stop - pair_start
                byte_count = (
                    proposed_stop - start + 1 + pair_count
                ) * np.dtype(np.int64).itemsize
                if pair_count > pair_limit or byte_count > byte_limit:
                    break
                stop = proposed_stop
            if stop == start:
                candidate_count = int(self.offsets[start + 1] - self.offsets[start])
                required_bytes = (
                    2 + candidate_count
                ) * np.dtype(np.int64).itemsize
                raise ReachabilityLocalizationError(
                    "target %d requires %d candidate pairs and %d CSR bytes; "
                    "a complete row cannot fit the configured microbatch limits"
                    % (start, candidate_count, required_bytes)
                )
            batches.append(
                ReachabilityMicrobatch(
                    target_start=start,
                    target_stop=stop,
                    pair_start=pair_start,
                    pair_stop=int(self.offsets[stop]),
                )
            )
            start = stop
        return tuple(batches)


@dataclass(frozen=True)
class _BoxScanResult:
    routing_candidate: np.ndarray
    exact_candidate: np.ndarray
    routing_witness: np.ndarray
    exact_witness: np.ndarray
    kind_masks: np.ndarray
    exact_kind_masks: np.ndarray


def _scan_target_boxes(
    flat_targets: np.ndarray,
    valid: np.ndarray,
    exact_lower: np.ndarray,
    exact_upper: np.ndarray,
    routing_lower: np.ndarray,
    routing_upper: np.ndarray,
    witness_ids: np.ndarray,
    config: ReachabilityConfig,
    *,
    kind_codes: Optional[np.ndarray] = None,
) -> _BoxScanResult:
    """Shared bounded-memory AABB scan for ordinary and tagged indexes."""

    target_count = flat_targets.shape[0]
    image_count = exact_lower.shape[0]
    routing_candidate = np.zeros(target_count, dtype=bool)
    exact_candidate = np.zeros(target_count, dtype=bool)
    routing_witness = np.full(target_count, -1, dtype=np.int64)
    exact_witness = np.full(target_count, -1, dtype=np.int64)
    kind_masks = np.zeros(target_count, dtype=np.uint16)
    exact_kind_masks = np.zeros(target_count, dtype=np.uint16)
    valid_positions = np.flatnonzero(valid)
    collect_kinds = kind_codes is not None
    if kind_codes is None:
        kind_codes = np.zeros(image_count, dtype=np.uint8)

    for query_start in range(0, valid_positions.size, config.query_batch_size):
        query_stop = min(
            valid_positions.size, query_start + config.query_batch_size
        )
        positions = valid_positions[query_start:query_stop]
        query = flat_targets[positions]
        routing_found = np.zeros(positions.size, dtype=bool)
        exact_found = np.zeros(positions.size, dtype=bool)

        for image_start in range(0, image_count, config.cell_batch_size):
            image_stop = min(image_count, image_start + config.cell_batch_size)
            exact_contains = np.all(
                (query[:, np.newaxis, :] >= exact_lower[None, image_start:image_stop])
                & (query[:, np.newaxis, :] <= exact_upper[None, image_start:image_stop]),
                axis=2,
            )
            routing_contains = np.all(
                (query[:, np.newaxis, :] >= routing_lower[None, image_start:image_stop])
                & (query[:, np.newaxis, :] <= routing_upper[None, image_start:image_stop]),
                axis=2,
            )

            newly_exact = ~exact_found & np.any(exact_contains, axis=1)
            if np.any(newly_exact):
                rows = np.flatnonzero(newly_exact)
                first = np.argmax(exact_contains[rows], axis=1)
                exact_witness[positions[rows]] = witness_ids[image_start + first]
                exact_found[rows] = True
            newly_routing = ~routing_found & np.any(routing_contains, axis=1)
            if np.any(newly_routing):
                rows = np.flatnonzero(newly_routing)
                first = np.argmax(routing_contains[rows], axis=1)
                routing_witness[positions[rows]] = witness_ids[image_start + first]
                routing_found[rows] = True

            if collect_kinds:
                batch_kinds = kind_codes[image_start:image_stop]
                exact_kind_masks[positions] |= np.bitwise_or.reduce(
                    np.where(exact_contains, batch_kinds[None, :], 0), axis=1
                ).astype(np.uint16)
                kind_masks[positions] |= np.bitwise_or.reduce(
                    np.where(routing_contains, batch_kinds[None, :], 0), axis=1
                ).astype(np.uint16)
            elif np.all(routing_found) and np.all(exact_found):
                break

        routing_candidate[positions] = routing_found
        exact_candidate[positions] = exact_found

    return _BoxScanResult(
        routing_candidate=routing_candidate,
        exact_candidate=exact_candidate,
        routing_witness=routing_witness,
        exact_witness=exact_witness,
        kind_masks=kind_masks,
        exact_kind_masks=exact_kind_masks,
    )


def _routing_pairs_to_csr(
    target_count: int,
    query_parts: Sequence[np.ndarray],
    cell_parts: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize complete routing matches to sorted target-major CSR."""

    if len(query_parts) != len(cell_parts):
        raise ReachabilityLocalizationError(
            "routing match query and cell parts disagree"
        )
    if query_parts:
        queries = np.concatenate(query_parts).astype(np.int64, copy=False)
        cells = np.concatenate(cell_parts).astype(np.int64, copy=False)
    else:
        queries = np.empty(0, dtype=np.int64)
        cells = np.empty(0, dtype=np.int64)
    if queries.shape != cells.shape or queries.ndim != 1:
        raise ReachabilityLocalizationError(
            "routing match query and cell arrays must be matching vectors"
        )
    if np.any(queries < 0) or np.any(queries >= target_count):
        raise ReachabilityLocalizationError(
            "routing match query ID is outside the target batch"
        )
    if queries.size:
        order = np.lexsort((cells, queries))
        queries = queries[order]
        cells = cells[order]
        duplicates = (queries[1:] == queries[:-1]) & (cells[1:] == cells[:-1])
        if np.any(duplicates):
            raise ReachabilityLocalizationError(
                "exhaustive localization produced a duplicate target/cell pair"
            )
    counts = np.bincount(queries, minlength=target_count).astype(
        np.int64, copy=False
    )
    offsets = np.empty(target_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, cells


def _localize_target_boxes(
    flat_targets: np.ndarray,
    valid: np.ndarray,
    exact_lower: np.ndarray,
    exact_upper: np.ndarray,
    routing_lower: np.ndarray,
    routing_upper: np.ndarray,
    witness_ids: np.ndarray,
    config: ReachabilityConfig,
) -> Tuple[_BoxScanResult, np.ndarray, np.ndarray]:
    """Collect every ordinary routing match in one complete bounded scan.

    This is deliberately separate from :func:`_scan_target_boxes`, whose
    early-exit boolean/witness oracle remains unchanged. Temporary pairwise
    comparisons obey the same query/cell batching limits; output CSR storage
    is the complete scientific result and is never truncated.
    """

    target_count = int(flat_targets.shape[0])
    image_count = int(exact_lower.shape[0])
    routing_candidate = np.zeros(target_count, dtype=bool)
    exact_candidate = np.zeros(target_count, dtype=bool)
    maximum_cell_id = np.iinfo(np.int64).max
    routing_minimum = np.full(target_count, maximum_cell_id, dtype=np.int64)
    exact_minimum = np.full(target_count, maximum_cell_id, dtype=np.int64)
    routing_query_parts: List[np.ndarray] = []
    routing_cell_parts: List[np.ndarray] = []
    valid_positions = np.flatnonzero(valid)

    for query_start in range(0, valid_positions.size, config.query_batch_size):
        query_stop = min(
            valid_positions.size, query_start + config.query_batch_size
        )
        positions = valid_positions[query_start:query_stop]
        query = flat_targets[positions]
        for image_start in range(0, image_count, config.cell_batch_size):
            image_stop = min(image_count, image_start + config.cell_batch_size)
            exact_contains = np.all(
                (query[:, np.newaxis, :] >= exact_lower[None, image_start:image_stop])
                & (query[:, np.newaxis, :] <= exact_upper[None, image_start:image_stop]),
                axis=2,
            )
            routing_contains = np.all(
                (
                    query[:, np.newaxis, :]
                    >= routing_lower[None, image_start:image_stop]
                )
                & (
                    query[:, np.newaxis, :]
                    <= routing_upper[None, image_start:image_stop]
                ),
                axis=2,
            )

            routing_rows, routing_columns = np.nonzero(routing_contains)
            if routing_rows.size:
                global_queries = positions[routing_rows]
                global_cells = witness_ids[image_start + routing_columns]
                routing_query_parts.append(global_queries)
                routing_cell_parts.append(global_cells)
                routing_candidate[global_queries] = True
                np.minimum.at(routing_minimum, global_queries, global_cells)

            exact_rows, exact_columns = np.nonzero(exact_contains)
            if exact_rows.size:
                global_queries = positions[exact_rows]
                global_cells = witness_ids[image_start + exact_columns]
                exact_candidate[global_queries] = True
                np.minimum.at(exact_minimum, global_queries, global_cells)

    routing_witness = np.where(
        routing_candidate, routing_minimum, -1
    ).astype(np.int64, copy=False)
    exact_witness = np.where(
        exact_candidate, exact_minimum, -1
    ).astype(np.int64, copy=False)
    offsets, cells = _routing_pairs_to_csr(
        target_count,
        routing_query_parts,
        routing_cell_parts,
    )
    scan = _BoxScanResult(
        routing_candidate=routing_candidate,
        exact_candidate=exact_candidate,
        routing_witness=routing_witness,
        exact_witness=exact_witness,
        kind_masks=np.zeros(target_count, dtype=np.uint16),
        exact_kind_masks=np.zeros(target_count, dtype=np.uint16),
    )
    return scan, offsets, cells


@dataclass(frozen=True)
class CellImageQueryResult:
    """Candidate membership in one or more tagged cell-image collections."""

    field_names: Tuple[str, ...]
    targets: np.ndarray
    evaluated: np.ndarray
    candidate: np.ndarray
    exact_candidate: np.ndarray
    kind_masks: np.ndarray
    exact_kind_masks: np.ndarray
    witness_collection: np.ndarray
    witness_entry: np.ndarray

    def __post_init__(self) -> None:
        names = tuple(self.field_names)
        targets = np.asarray(self.targets, dtype=float)
        if targets.ndim < 1 or targets.shape[-1] != len(names):
            raise ValueError("targets must end with the cell-image field dimension")
        query_shape = targets.shape[:-1]
        evaluated = np.asarray(self.evaluated, dtype=bool)
        candidate = np.asarray(self.candidate, dtype=bool)
        exact = np.asarray(self.exact_candidate, dtype=bool)
        kinds = np.asarray(self.kind_masks)
        exact_kinds = np.asarray(self.exact_kind_masks)
        witness_collection = np.asarray(self.witness_collection)
        witness_entry = np.asarray(self.witness_entry)
        for label, array in (
            ("evaluated", evaluated),
            ("candidate", candidate),
            ("exact_candidate", exact),
            ("kind_masks", kinds),
            ("exact_kind_masks", exact_kinds),
            ("witness_collection", witness_collection),
            ("witness_entry", witness_entry),
        ):
            if array.shape != query_shape:
                raise ValueError("%s does not match the target query shape" % label)
        if not np.issubdtype(kinds.dtype, np.integer) or not np.issubdtype(
            exact_kinds.dtype, np.integer
        ):
            raise ValueError("cell-image kind masks must be integers")
        if _contains_unknown_cell_image_kind_bits(
            kinds
        ) or _contains_unknown_cell_image_kind_bits(exact_kinds):
            raise ValueError("cell-image query contains an unknown kind bit")
        if not np.issubdtype(witness_collection.dtype, np.integer) or not np.issubdtype(
            witness_entry.dtype, np.integer
        ):
            raise ValueError("cell-image witnesses must be integers")
        if np.any(exact & ~candidate):
            raise ValueError("an exact-box candidate must also be a routing candidate")
        if np.any(candidate != (witness_entry >= 0)):
            raise ValueError("candidate and witness columns disagree")
        if np.any((witness_collection >= 0) != (witness_entry >= 0)):
            raise ValueError("collection and entry witnesses disagree")
        if np.any(~evaluated & candidate):
            raise ValueError("a suppressed target cannot be a candidate")
        if np.any((kinds != 0) != candidate) or np.any((exact_kinds != 0) != exact):
            raise ValueError("candidate and kind-mask columns disagree")
        object.__setattr__(self, "field_names", names)
        object.__setattr__(self, "targets", _immutable_array(targets))
        object.__setattr__(self, "evaluated", _immutable_array(evaluated, dtype=bool))
        object.__setattr__(self, "candidate", _immutable_array(candidate, dtype=bool))
        object.__setattr__(
            self, "exact_candidate", _immutable_array(exact, dtype=bool)
        )
        object.__setattr__(self, "kind_masks", _immutable_array(kinds, dtype=np.uint16))
        object.__setattr__(
            self, "exact_kind_masks", _immutable_array(exact_kinds, dtype=np.uint16)
        )
        object.__setattr__(
            self,
            "witness_collection",
            _immutable_array(witness_collection, dtype=np.int64),
        )
        object.__setattr__(
            self, "witness_entry", _immutable_array(witness_entry, dtype=np.int64)
        )

    @property
    def query_shape(self) -> Tuple[int, ...]:
        return self.targets.shape[:-1]

    def has_kind(self, kind: CellImageKind, *, exact: bool = False) -> np.ndarray:
        try:
            bit = int(CellImageKind(kind))
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown cell-image kind") from exc
        masks = self.exact_kind_masks if exact else self.kind_masks
        selected = (masks & bit) != 0
        return _immutable_array(selected, dtype=bool)


class CellImageIndex:
    """Batched AABB query index over one or more immutable collections."""

    def __init__(
        self,
        collections: Sequence[CellImageCollection],
        *,
        config: Optional[ReachabilityConfig] = None,
    ) -> None:
        try:
            resolved = tuple(collections)
        except TypeError as exc:
            raise TypeError(
                "collections must be a sequence of CellImageCollection objects"
            ) from exc
        if not resolved or any(
            not isinstance(collection, CellImageCollection)
            for collection in resolved
        ):
            raise ReachabilityConfigurationError(
                "at least one CellImageCollection is required"
            )
        first = resolved[0]
        for collection in resolved[1:]:
            if (
                collection.field_names != first.field_names
                or not np.array_equal(
                    collection.residual_scales, first.residual_scales
                )
                or collection.root_residual_tolerance
                != first.root_residual_tolerance
            ):
                raise ReachabilityConfigurationError(
                    "combined cell-image collections must use identical fields, "
                    "residual scales, and routing tolerance"
                )
        self._collections = resolved
        self._config = ReachabilityConfig() if config is None else config
        if not isinstance(self._config, ReachabilityConfig):
            raise TypeError("config must be a ReachabilityConfig")
        self._exact_lower = np.concatenate(
            [item.exact_lower for item in resolved], axis=0
        )
        self._exact_upper = np.concatenate(
            [item.exact_upper for item in resolved], axis=0
        )
        self._routing_lower = np.concatenate(
            [item.routing_lower for item in resolved], axis=0
        )
        self._routing_upper = np.concatenate(
            [item.routing_upper for item in resolved], axis=0
        )
        self._kind_codes = np.concatenate(
            [item.kind_codes for item in resolved], axis=0
        )
        self._collection_ids = np.concatenate(
            [
                np.full(item.image_count, index, dtype=np.int64)
                for index, item in enumerate(resolved)
            ]
        )
        self._entry_ids = np.concatenate(
            [np.arange(item.image_count, dtype=np.int64) for item in resolved]
        )

    @property
    def collections(self) -> Tuple[CellImageCollection, ...]:
        return self._collections

    @property
    def config(self) -> ReachabilityConfig:
        return self._config

    @property
    def field_names(self) -> Tuple[str, ...]:
        return self._collections[0].field_names

    @property
    def image_count(self) -> int:
        return int(self._kind_codes.size)

    def classify(
        self, targets: Any, *, valid_mask: Optional[Any] = None
    ) -> CellImageQueryResult:
        try:
            target_array = np.asarray(targets, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReachabilityInputError("targets must be numeric") from exc
        dimension = len(self.field_names)
        if target_array.ndim == 0 or target_array.shape[-1] != dimension:
            raise ReachabilityInputError(
                "targets must have shape (..., %d)" % dimension
            )
        query_shape = target_array.shape[:-1]
        flat_targets = target_array.reshape(-1, dimension)
        finite = np.all(np.isfinite(flat_targets), axis=1)
        if valid_mask is None:
            valid = finite
        else:
            supplied = np.asarray(valid_mask)
            if supplied.shape != query_shape or supplied.dtype.kind != "b":
                raise ReachabilityInputError(
                    "valid_mask must be boolean with shape %r" % (query_shape,)
                )
            valid = supplied.reshape(-1) & finite

        scan = _scan_target_boxes(
            flat_targets,
            valid,
            self._exact_lower,
            self._exact_upper,
            self._routing_lower,
            self._routing_upper,
            np.arange(self.image_count, dtype=np.int64),
            self.config,
            kind_codes=self._kind_codes,
        )
        witness_collection = np.full(flat_targets.shape[0], -1, dtype=np.int64)
        witness_entry = np.full(flat_targets.shape[0], -1, dtype=np.int64)
        witnessed = scan.routing_witness >= 0
        if np.any(witnessed):
            witness_collection[witnessed] = self._collection_ids[
                scan.routing_witness[witnessed]
            ]
            witness_entry[witnessed] = self._entry_ids[
                scan.routing_witness[witnessed]
            ]
        return CellImageQueryResult(
            field_names=self.field_names,
            targets=target_array,
            evaluated=valid.reshape(query_shape),
            candidate=scan.routing_candidate.reshape(query_shape),
            exact_candidate=scan.exact_candidate.reshape(query_shape),
            kind_masks=scan.kind_masks.reshape(query_shape),
            exact_kind_masks=scan.exact_kind_masks.reshape(query_shape),
            witness_collection=witness_collection.reshape(query_shape),
            witness_entry=witness_entry.reshape(query_shape),
        )


class MixedPhaseCandidateIndex:
    """Mixed target candidates plus exact representation-specific verifier."""

    def __init__(
        self,
        model: Any,
        interpolator: MultilinearInterpolator,
        images: CellImageCollection,
        *,
        config: Optional[ReachabilityConfig] = None,
    ) -> None:
        if not isinstance(model, (MixedPhaseStrip, BinodalBand)):
            raise TypeError("model must be a MixedPhaseStrip or BinodalBand")
        if not isinstance(interpolator, MultilinearInterpolator):
            raise TypeError("interpolator must be a MultilinearInterpolator")
        if not isinstance(images, CellImageCollection):
            raise TypeError("images must be a CellImageCollection")
        if images.coordinate_names != interpolator.coordinate_names:
            raise ReachabilityConfigurationError(
                "cell-image coordinates do not match their interpolator"
            )
        if images.field_names != interpolator.field_names:
            raise ReachabilityConfigurationError(
                "cell-image targets do not match their interpolator"
            )
        if isinstance(model, MixedPhaseStrip):
            valid_kinds = (int(CellImageKind.LINE_REGION_S),)
        else:
            valid_kinds = (
                int(CellImageKind.BINODALS_INTERIOR),
                int(CellImageKind.BINODALS_BOUNDARY),
            )
        if not np.all(np.isin(images.kind_codes, valid_kinds)):
            raise ReachabilityConfigurationError(
                "cell-image provenance does not match the mixed-phase model"
            )
        self.model = model
        self.interpolator = interpolator
        self.images = images
        self.index = CellImageIndex((images,), config=config)

    @classmethod
    def for_line(
        cls,
        strip: MixedPhaseStrip,
        residual_scales: Any,
        root_residual_tolerance: float,
        *,
        config: Optional[ReachabilityConfig] = None,
    ) -> "MixedPhaseCandidateIndex":
        if not isinstance(strip, MixedPhaseStrip):
            raise TypeError("strip must be a MixedPhaseStrip")
        interpolator = strip.to_interpolator()
        images = build_rectilinear_cell_images(
            interpolator,
            residual_scales,
            root_residual_tolerance,
            kind=CellImageKind.LINE_REGION_S,
            config=config,
        )
        return cls(strip, interpolator, images, config=config)

    @classmethod
    def for_binodals(
        cls,
        interpolator: MultilinearInterpolator,
        band: BinodalBand,
        residual_scales: Any,
        root_residual_tolerance: float,
        *,
        subdivision: Optional[BinodalSubdivisionConfig] = None,
        config: Optional[ReachabilityConfig] = None,
    ) -> "MixedPhaseCandidateIndex":
        images = build_binodal_cell_images(
            interpolator,
            band,
            residual_scales,
            root_residual_tolerance,
            subdivision=subdivision,
            config=config,
        )
        return cls(band, interpolator, images, config=config)

    def classify(
        self, targets: Any, *, valid_mask: Optional[Any] = None
    ) -> CellImageQueryResult:
        return self.index.classify(targets, valid_mask=valid_mask)

    def matching_entries(self, target: Any, *, exact: bool = False) -> np.ndarray:
        """Return every deterministic image-row match for one finite target.

        The regular classifier intentionally returns only one witness because
        reachability needs a Boolean candidate decision.  Phase-restricted
        inversion must instead inspect *all* matching source pieces so that
        neither image order nor a first witness silently chooses an inverse
        branch.  Routing boxes are used by default; ``exact=True`` selects
        the unpadded mathematical enclosures.
        """

        try:
            array = np.asarray(target, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReachabilityInputError(
                "target must be a finite numeric vector"
            ) from exc
        if array.shape != (self.images.dimension,):
            raise ReachabilityInputError(
                "target must have shape (%d,)" % self.images.dimension
            )
        if not np.all(np.isfinite(array)):
            raise ReachabilityInputError("target must contain only finite values")
        lower = self.images.exact_lower if exact else self.images.routing_lower
        upper = self.images.exact_upper if exact else self.images.routing_upper
        matches = np.all((array >= lower) & (array <= upper), axis=1)
        return _immutable_array(np.flatnonzero(matches), dtype=np.int64)

    def verify_source(self, points: Any) -> Any:
        """Apply the exact physical membership rule after a candidate solve."""

        try:
            array = np.asarray(points, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReachabilityInputError("source points must be numeric") from exc
        if array.ndim == 0 or array.shape[-1] != 2:
            raise ReachabilityInputError("source points must have shape (..., 2)")
        if not np.all(np.isfinite(array)):
            raise ReachabilityInputError("source points must be finite")
        if isinstance(self.model, BinodalBand):
            return self.model.contains_points(array)
        contained = (
            (array[..., 0] >= self.interpolator.bounds[0, 0])
            & (array[..., 0] <= self.interpolator.bounds[0, 1])
            & (array[..., 1] >= 0.0)
            & (array[..., 1] <= 1.0)
        )
        return bool(contained) if contained.ndim == 0 else _immutable_array(
            contained, dtype=bool
        )


class ReachabilityIndex:
    """Reusable exact cell boxes and a slow batched reference classifier."""

    def __init__(
        self,
        interpolator: MultilinearInterpolator,
        residual_scales: Any,
        root_residual_tolerance: float,
        *,
        active_cell_mask: Optional[Any] = None,
        config: Optional[ReachabilityConfig] = None,
    ) -> None:
        if not isinstance(interpolator, MultilinearInterpolator):
            raise TypeError("interpolator must be a MultilinearInterpolator")
        if (
            interpolator.field_names != interpolator.target_names
            or len(interpolator.field_names) != interpolator.dimension
        ):
            raise ReachabilityConfigurationError(
                "reachability requires the canonical target-only interpolator "
                "in metadata order"
            )
        self._interpolator = interpolator
        self._config = ReachabilityConfig() if config is None else config
        if not isinstance(self._config, ReachabilityConfig):
            raise TypeError("config must be a ReachabilityConfig")
        scales = _resolve_scales(residual_scales, interpolator.field_names)
        tolerance = _finite_nonnegative(
            root_residual_tolerance, "root_residual_tolerance"
        )
        dimension = interpolator.dimension
        normalized_active_mask, active_count = self._validate_active_cell_mask(
            active_cell_mask
        )
        if self._config.method == "hierarchical":
            if dimension != 4:
                raise ReachabilityConfigurationError(
                    "hierarchical reachability is initially supported only in 4D"
                )
            if normalized_active_mask is not None and not np.all(
                normalized_active_mask
            ):
                raise ReachabilityConfigurationError(
                    "hierarchical reachability requires the full active-cell lattice"
                )
        estimated_bytes = active_count * (
            np.dtype(np.int64).itemsize
            + 4 * dimension * np.dtype(np.float64).itemsize
        )
        if estimated_bytes > self._config.max_enclosure_bytes:
            raise ReachabilityConfigurationError(
                "cell enclosures require %d bytes, exceeding "
                "max_enclosure_bytes=%d"
                % (estimated_bytes, self._config.max_enclosure_bytes)
            )
        active_flat = self._resolve_active_cells(normalized_active_mask)
        exact_lower = np.empty((active_flat.size, dimension), dtype=float)
        exact_upper = np.empty_like(exact_lower)
        for start in range(0, active_flat.size, self._config.cell_batch_size):
            stop = min(active_flat.size, start + self._config.cell_batch_size)
            cell_indices = np.column_stack(
                np.unravel_index(
                    active_flat[start:stop], interpolator.cell_shape, order="C"
                )
            )
            bounds = interpolator.cell_value_enclosures(cell_indices)
            if not bounds.rigorous or not np.all(bounds.defined):
                raise ReachabilityConfigurationError(
                    "canonical target cell enclosures must be rigorous and finite"
                )
            exact_lower[start:stop] = bounds.lower
            exact_upper[start:stop] = bounds.upper

        padding, routing_lower, routing_upper = self._expanded_boxes(
            exact_lower, exact_upper, scales, tolerance
        )
        self._enclosures = CellTargetEnclosures(
            field_names=interpolator.field_names,
            cell_shape=interpolator.cell_shape,
            cell_flat_indices=active_flat,
            exact_lower=exact_lower,
            exact_upper=exact_upper,
            routing_lower=routing_lower,
            routing_upper=routing_upper,
            residual_scales=scales,
            routing_padding=padding,
            root_residual_tolerance=tolerance,
            rigorous=True,
            outward_rounded=tolerance > 0.0,
        )
        self._hierarchy = None
        if self._config.method == "hierarchical":
            from .reachability_hierarchy import (
                ReachabilityHierarchyError,
                build_routing_hierarchy,
            )

            try:
                self._hierarchy = build_routing_hierarchy(
                    interpolator.cell_shape,
                    self._enclosures.routing_lower,
                    self._enclosures.routing_upper,
                    hierarchy_levels=self._config.hierarchy_levels,
                    max_hierarchy_bytes=self._config.max_hierarchy_bytes,
                )
            except ReachabilityHierarchyError as exc:
                raise ReachabilityConfigurationError(
                    "could not build the configured reachability hierarchy: %s"
                    % exc
                ) from exc
        self._provenance = ReachabilityProvenance(
            table_fingerprint=interpolator.source_snapshot_fingerprint,
            active_cell_fingerprint=_active_cell_domain_fingerprint(
                interpolator.cell_shape,
                self._enclosures.cell_flat_indices,
            ),
            coordinate_names=interpolator.coordinate_names,
            field_names=interpolator.field_names,
            cell_shape=interpolator.cell_shape,
            active_cell_count=int(self._enclosures.cell_flat_indices.size),
            residual_scales=self._enclosures.residual_scales,
            root_residual_tolerance=(
                self._enclosures.root_residual_tolerance
            ),
            method=self._config.method,
            hierarchy_levels=self._config.hierarchy_levels,
        )

    @classmethod
    def from_plan(
        cls,
        interpolator: MultilinearInterpolator,
        plan: PipelinePlan,
        *,
        active_cell_mask: Optional[Any] = None,
        config: Optional[ReachabilityConfig] = None,
    ) -> "ReachabilityIndex":
        """Build an index using one resolved pipeline plan's fixed scales."""

        if not isinstance(plan, PipelinePlan):
            raise TypeError("plan must be a PipelinePlan")
        if interpolator.coordinate_names != plan.source_names:
            raise ReachabilityConfigurationError(
                "interpolator coordinates do not match the pipeline plan"
            )
        if interpolator.field_names != plan.target_names:
            raise ReachabilityConfigurationError(
                "interpolator target fields do not match the pipeline plan"
            )
        return cls(
            interpolator,
            plan.residual_scales,
            plan.config.solver.root_residual_tolerance,
            active_cell_mask=active_cell_mask,
            config=config,
        )

    @classmethod
    def from_precomputed(
        cls,
        interpolator: MultilinearInterpolator,
        plan: PipelinePlan,
        enclosures: CellTargetEnclosures,
        *,
        hierarchy: Optional[Any] = None,
        config: Optional[ReachabilityConfig] = None,
    ) -> "ReachabilityIndex":
        """Attach immutable precomputed boxes without rebuilding large arrays.

        The entry point is intentionally strict and is used by spawned workers
        loading a versioned read-only mmap snapshot.  It preserves the normal
        provenance and proof contract; arbitrary cell-ID arrays cannot enter
        through this API.
        """

        if not isinstance(interpolator, MultilinearInterpolator):
            raise TypeError("interpolator must be a MultilinearInterpolator")
        if not isinstance(plan, PipelinePlan):
            raise TypeError("plan must be a PipelinePlan")
        if not isinstance(enclosures, CellTargetEnclosures):
            raise TypeError("enclosures must be CellTargetEnclosures")
        resolved_config = ReachabilityConfig() if config is None else config
        if not isinstance(resolved_config, ReachabilityConfig):
            raise TypeError("config must be a ReachabilityConfig")
        if interpolator.coordinate_names != plan.source_names:
            raise ReachabilityConfigurationError(
                "interpolator coordinates do not match the pipeline plan"
            )
        if interpolator.field_names != plan.target_names:
            raise ReachabilityConfigurationError(
                "interpolator target fields do not match the pipeline plan"
            )
        if (
            enclosures.field_names != interpolator.field_names
            or enclosures.cell_shape != interpolator.cell_shape
        ):
            raise ReachabilityConfigurationError(
                "precomputed cell enclosures do not match the interpolator"
            )
        expected_scales = _resolve_scales(
            plan.residual_scales, interpolator.field_names
        )
        expected_tolerance = _finite_nonnegative(
            plan.config.solver.root_residual_tolerance,
            "root_residual_tolerance",
        )
        if not np.array_equal(enclosures.residual_scales, expected_scales):
            raise ReachabilityConfigurationError(
                "precomputed enclosure residual scales do not match the plan"
            )
        if enclosures.root_residual_tolerance != expected_tolerance:
            raise ReachabilityConfigurationError(
                "precomputed enclosure tolerance does not match the plan"
            )
        total_cells = math.prod(interpolator.cell_shape)
        if resolved_config.method == "hierarchical":
            from .reachability_hierarchy import RoutingHierarchy

            if interpolator.dimension != 4:
                raise ReachabilityConfigurationError(
                    "hierarchical reachability is initially supported only in 4D"
                )
            if (
                enclosures.active_cell_count != total_cells
                or int(enclosures.cell_flat_indices[0]) != 0
                or int(enclosures.cell_flat_indices[-1]) != total_cells - 1
            ):
                raise ReachabilityConfigurationError(
                    "hierarchical precomputed reachability requires the full cell lattice"
                )
            if (
                not isinstance(hierarchy, RoutingHierarchy)
                or hierarchy.cell_shape != interpolator.cell_shape
                or hierarchy.requested_levels
                != resolved_config.hierarchy_levels
                or not hierarchy.reaches_leaf
            ):
                raise ReachabilityConfigurationError(
                    "precomputed hierarchy does not match the configured fine-cell lattice"
                )
            leaf = hierarchy.levels[-1]
            if (
                not leaf.aliases_leaf
                or not np.shares_memory(
                    leaf.routing_lower, enclosures.routing_lower
                )
                or not np.shares_memory(
                    leaf.routing_upper, enclosures.routing_upper
                )
            ):
                raise ReachabilityConfigurationError(
                    "precomputed hierarchy leaf must alias the enclosure routing boxes"
                )
        elif hierarchy is not None:
            raise ReachabilityConfigurationError(
                "exhaustive reachability cannot attach a hierarchy"
            )

        result = cls.__new__(cls)
        result._interpolator = interpolator
        result._config = resolved_config
        result._enclosures = enclosures
        result._hierarchy = hierarchy
        result._provenance = ReachabilityProvenance(
            table_fingerprint=interpolator.source_snapshot_fingerprint,
            active_cell_fingerprint=_active_cell_domain_fingerprint(
                interpolator.cell_shape,
                enclosures.cell_flat_indices,
            ),
            coordinate_names=interpolator.coordinate_names,
            field_names=interpolator.field_names,
            cell_shape=interpolator.cell_shape,
            active_cell_count=int(enclosures.cell_flat_indices.size),
            residual_scales=enclosures.residual_scales,
            root_residual_tolerance=enclosures.root_residual_tolerance,
            method=resolved_config.method,
            hierarchy_levels=resolved_config.hierarchy_levels,
        )
        return result

    @property
    def interpolator(self) -> MultilinearInterpolator:
        return self._interpolator

    @property
    def config(self) -> ReachabilityConfig:
        return self._config

    @property
    def enclosures(self) -> CellTargetEnclosures:
        return self._enclosures

    @property
    def hierarchy(self) -> Optional[Any]:
        """Private hierarchy backing explicit 4D hierarchical mode, if any."""

        return self._hierarchy

    @property
    def provenance(self) -> ReachabilityProvenance:
        """Proof identity required by complete candidate localizations."""

        return self._provenance

    @property
    def field_names(self) -> Tuple[str, ...]:
        return self.enclosures.field_names

    @property
    def estimated_enclosure_bytes(self) -> int:
        # Exclude the two tiny dimension-length scale/padding vectors so this
        # matches the pre-allocation memory guard exactly.
        return int(
            self.enclosures.cell_flat_indices.nbytes
            + self.enclosures.exact_lower.nbytes
            + self.enclosures.exact_upper.nbytes
            + self.enclosures.routing_lower.nbytes
            + self.enclosures.routing_upper.nbytes
        )

    def _validate_active_cell_mask(
        self, active_cell_mask: Optional[Any]
    ) -> Tuple[Optional[np.ndarray], int]:
        shape = self._interpolator.cell_shape
        total = math.prod(shape)
        if active_cell_mask is None:
            return None, total
        mask = np.asarray(active_cell_mask)
        if mask.shape != shape or mask.dtype.kind != "b":
            raise ReachabilityConfigurationError(
                "active_cell_mask must be a boolean array with shape %r"
                % (shape,)
            )
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise ReachabilityConfigurationError(
                "the active-cell domain cannot be empty"
            )
        return np.ascontiguousarray(mask, dtype=bool), count

    def _resolve_active_cells(
        self, active_cell_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        shape = self._interpolator.cell_shape
        if active_cell_mask is None:
            flat = np.arange(math.prod(shape), dtype=np.int64)
        else:
            flat = np.flatnonzero(active_cell_mask.reshape(-1, order="C")).astype(
                np.int64, copy=False
            )
        return flat

    @staticmethod
    def _expanded_boxes(
        exact_lower: np.ndarray,
        exact_upper: np.ndarray,
        scales: np.ndarray,
        tolerance: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _expanded_target_boxes(
            exact_lower, exact_upper, scales, tolerance
        )

    def _prepare_targets(
        self, targets: Any, valid_mask: Optional[Any]
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...], np.ndarray]:
        try:
            target_array = np.asarray(targets, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReachabilityInputError("targets must be numeric") from exc
        dimension = len(self.field_names)
        if target_array.ndim == 0 or target_array.shape[-1] != dimension:
            raise ReachabilityInputError(
                "targets must have shape (..., %d)" % dimension
            )
        query_shape = target_array.shape[:-1]
        flat_targets = target_array.reshape(-1, dimension)
        finite = np.all(np.isfinite(flat_targets), axis=1)
        if valid_mask is None:
            valid = finite
        else:
            supplied = np.asarray(valid_mask)
            if supplied.shape != query_shape or supplied.dtype.kind != "b":
                raise ReachabilityInputError(
                    "valid_mask must be boolean with shape %r" % (query_shape,)
                )
            valid = supplied.reshape(-1) & finite
        return target_array, flat_targets, query_shape, valid

    def _result_from_scan(
        self,
        target_array: np.ndarray,
        flat_targets: np.ndarray,
        query_shape: Tuple[int, ...],
        valid: np.ndarray,
        scan: Any,
    ) -> ReachabilityResult:
        """Build the stable public result from either reachability backend."""

        not_run = pipeline_status_code(PrescreenStatus.NOT_RUN)
        candidate_code = pipeline_status_code(PrescreenStatus.CANDIDATE)
        excluded_code = pipeline_status_code(
            PrescreenStatus.DEFINITELY_UNREACHABLE
        )
        prescreen = np.full(flat_targets.shape[0], not_run, dtype=np.uint8)
        valid_positions = np.flatnonzero(valid)
        prescreen[valid_positions] = np.where(
            scan.routing_candidate[valid_positions],
            candidate_code,
            excluded_code,
        ).astype(np.uint8, copy=False)

        return ReachabilityResult(
            field_names=self.field_names,
            cell_shape=self.enclosures.cell_shape,
            targets=target_array,
            prescreen_codes=prescreen.reshape(query_shape),
            exact_candidate=scan.exact_candidate.reshape(query_shape),
            routing_witness_flat=scan.routing_witness.reshape(query_shape),
            exact_witness_flat=scan.exact_witness.reshape(query_shape),
        )

    def _evaluate(
        self,
        targets: Any,
        valid_mask: Optional[Any],
        *,
        collect_routing_matches: bool,
    ) -> Any:
        target_array, flat_targets, query_shape, valid = self._prepare_targets(
            targets, valid_mask
        )
        boxes = self.enclosures
        offsets = None
        cells = None
        if self.config.method == "hierarchical":
            from .reachability_hierarchy import (
                classify_hierarchy_leaves,
                localize_hierarchy_leaves,
            )

            if self._hierarchy is None:
                raise ReachabilityConfigurationError(
                    "hierarchical reachability index was not constructed"
                )
            arguments = (
                self._hierarchy,
                flat_targets,
                boxes.exact_lower,
                boxes.exact_upper,
                boxes.routing_lower,
                boxes.routing_upper,
            )
            keywords = {
                "valid_mask": valid,
                "query_batch_size": self.config.query_batch_size,
                "leaf_batch_size": self.config.cell_batch_size,
                "maximum_pairwise_entries": self.config.max_pairwise_entries,
            }
            if collect_routing_matches:
                backend = localize_hierarchy_leaves(*arguments, **keywords)
                scan = backend.result
                offsets = backend.offsets
                cells = backend.cell_flat_indices
            else:
                scan = classify_hierarchy_leaves(*arguments, **keywords)
        elif collect_routing_matches:
            scan, offsets, cells = _localize_target_boxes(
                flat_targets,
                valid,
                boxes.exact_lower,
                boxes.exact_upper,
                boxes.routing_lower,
                boxes.routing_upper,
                boxes.cell_flat_indices,
                self.config,
            )
        else:
            scan = _scan_target_boxes(
                flat_targets,
                valid,
                boxes.exact_lower,
                boxes.exact_upper,
                boxes.routing_lower,
                boxes.routing_upper,
                boxes.cell_flat_indices,
                self.config,
            )

        result = self._result_from_scan(
            target_array,
            flat_targets,
            query_shape,
            valid,
            scan,
        )
        if not collect_routing_matches:
            return result
        if offsets is None or cells is None:
            raise ReachabilityLocalizationError(
                "complete reachability localization did not return CSR pairs"
            )
        return ReachabilityLocalization(
            result=result,
            offsets=offsets,
            cell_flat_indices=cells,
            completeness=LocalizationCompleteness.COMPLETE,
            provenance=self.provenance,
        )

    def classify(
        self, targets: Any, *, valid_mask: Optional[Any] = None
    ) -> ReachabilityResult:
        """Classify targets without retaining every matching routing cell.

        Finite targets are evaluated by default. ``valid_mask`` can suppress
        rows that failed an earlier tilde-to-canonical transform; suppressed or
        non-finite rows receive ``PrescreenStatus.NOT_RUN`` rather than a false
        no-root classification.
        """

        return self._evaluate(
            targets,
            valid_mask,
            collect_routing_matches=False,
        )

    def localize(
        self, targets: Any, *, valid_mask: Optional[Any] = None
    ) -> ReachabilityLocalization:
        """Classify and retain every matching active routing cell as CSR."""

        return self._evaluate(
            targets,
            valid_mask,
            collect_routing_matches=True,
        )

    def classify_chunk(self, chunk: GridChunk) -> ReachabilityResult:
        """Classify a Phase-1 grid chunk in canonical target space."""

        if not isinstance(chunk, GridChunk):
            raise TypeError("chunk must be a GridChunk")
        return self.classify(
            chunk.canonical_targets, valid_mask=chunk.transform_valid
        )

    def localize_chunk(self, chunk: GridChunk) -> ReachabilityLocalization:
        """Completely localize one grid chunk in canonical target space."""

        if not isinstance(chunk, GridChunk):
            raise TypeError("chunk must be a GridChunk")
        return self.localize(
            chunk.canonical_targets, valid_mask=chunk.transform_valid
        )


__all__ = [
    "BinodalSubdivisionConfig",
    "CellImageCollection",
    "CellImageIndex",
    "CellImageKind",
    "CellImageQueryResult",
    "DEFAULT_MAX_ENCLOSURE_BYTES",
    "DEFAULT_MAX_HIERARCHY_BYTES",
    "DEFAULT_MAX_PAIRWISE_ENTRIES",
    "DEFAULT_REACHABILITY_CELL_BATCH_SIZE",
    "DEFAULT_REACHABILITY_QUERY_BATCH_SIZE",
    "CellTargetEnclosures",
    "LocalizationCompleteness",
    "MixedPhaseCandidateIndex",
    "ReachabilityConfig",
    "ReachabilityConfigurationError",
    "ReachabilityError",
    "ReachabilityIndex",
    "ReachabilityInputError",
    "ReachabilityLocalization",
    "ReachabilityLocalizationError",
    "ReachabilityMicrobatch",
    "ReachabilityProvenance",
    "ReachabilityResult",
    "REACHABILITY_HIERARCHY_LEVELS",
    "REACHABILITY_METHODS",
    "REACHABILITY_ROUTING_CONTRACT_VERSION",
    "build_binodal_cell_images",
    "build_rectilinear_cell_images",
]
