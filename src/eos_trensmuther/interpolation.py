"""Boundary-safe multilinear interpolation on validated EoS tables.

The interpolator in this module is deliberately restricted to complete
rectilinear grids.  It consumes the canonical axes and shaped fields produced
by :class:`~eos_trensmuther.tables.ForwardTable`; it does not read files, repair
tables, extrapolate, or decide which physical inverse branch should be used.

Values, point Jacobians, and whole-cell component bounds all use the same cell
lookup and corner ordering.  This is important for the later inversion and
interval/Krawczyk stages, where disagreeing interpolation and derivative
implementations would be particularly difficult to diagnose.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._array_storage import immutable_array as _storage_immutable_array
from .tables import EoSError, ForwardTable


class InterpolationError(EoSError):
    """Base exception for interpolation failures."""


class InterpolationConfigurationError(InterpolationError):
    """Raised when a validated table cannot configure this interpolator."""


class InterpolationInputError(InterpolationError):
    """Raised when query points or cell indices have an invalid form."""


class InterpolationDomainError(InterpolationInputError):
    """Raised when one or more query points are outside the source domain.

    The exception retains the invalid points and their indices in the original
    batch.  Domain failure is therefore distinguishable from an in-domain
    auxiliary field whose interpolated value is ``NaN``.
    """

    def __init__(
        self,
        coordinate_names: Sequence[str],
        bounds: np.ndarray,
        points: np.ndarray,
        flat_indices: Sequence[int],
        query_shape: Tuple[int, ...],
    ) -> None:
        self.coordinate_names = tuple(coordinate_names)
        self.bounds = _immutable_array(bounds, dtype=float)
        self.points = _immutable_array(points, dtype=float)
        self.flat_indices = tuple(int(index) for index in flat_indices)
        self.query_shape = tuple(int(size) for size in query_shape)
        if self.query_shape:
            self.query_indices = tuple(
                tuple(int(value) for value in np.unravel_index(index, self.query_shape))
                for index in self.flat_indices
            )
        else:
            self.query_indices = ((),)

        count = len(self.flat_indices)
        first = self.points[0] if count else np.empty(0, dtype=float)
        message = (
            "%d query point(s) lie outside the closed interpolation domain; "
            "first invalid point is %s" % (count, np.array2string(first))
        )
        super().__init__(message)


class InterpolationCellError(InterpolationInputError):
    """Raised when an interpolation-cell index is invalid."""


class InterpolationDerivativeError(InterpolationError):
    """Raised when a grid-plane Jacobian may not be uniquely defined."""

    def __init__(self, location: "InterpolationLocation") -> None:
        self.location = location
        mask = np.any(location.on_internal_grid_plane, axis=-1)
        count = int(np.count_nonzero(mask))
        super().__init__(
            "%d query point(s) lie on an internal grid plane where the "
            "multilinear Jacobian can be discontinuous" % count
        )


def _immutable_array(value: Any, dtype: Optional[Any] = None) -> np.ndarray:
    """Return a C-contiguous ndarray backed by immutable bytes."""

    return _storage_immutable_array(value, dtype=dtype)


def _derived_map_fingerprint(
    coordinate_names: Tuple[str, ...],
    target_names: Tuple[str, ...],
    coordinate_units: Tuple[str, ...],
    target_units: Tuple[str, ...],
    axes: Tuple[np.ndarray, ...],
    field_arrays: Tuple[np.ndarray, ...],
    boundary_absolute_tolerance: float,
    boundary_relative_tolerance: float,
) -> str:
    """Return a process-stable identity for an immutable derived map."""

    digest = hashlib.sha256()
    digest.update(b"EoSInverter.derived_rectilinear_map.v1\0")
    for values in (
        coordinate_names,
        target_names,
        coordinate_units,
        target_units,
    ):
        digest.update(len(values).to_bytes(8, "little", signed=False))
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little", signed=False))
            digest.update(encoded)
    for array in axes + field_arrays:
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(canonical.ndim.to_bytes(8, "little", signed=False))
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes(order="C"))
        digest.update(canonical.tobytes(order="C"))
    digest.update(
        np.asarray(
            [boundary_absolute_tolerance, boundary_relative_tolerance],
            dtype="<f8",
        ).tobytes(order="C")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class InterpolationLocation:
    """Cell lookup and boundary information for one point or a point batch.

    All arrays have shape ``query_shape + (dimension,)``.  ``requested_points``
    contains the caller's coordinates, while ``points`` contains the actually
    evaluated coordinates.  They differ only when an explicitly configured
    outer-boundary tolerance snaps a tiny outside displacement to an endpoint.
    ``sides`` records the one-sided cell convention for each coordinate.
    """

    coordinate_names: Tuple[str, ...]
    requested_points: np.ndarray
    points: np.ndarray
    cell_indices: np.ndarray
    normalized_coordinates: np.ndarray
    cell_lower: np.ndarray
    cell_upper: np.ndarray
    on_domain_boundary: np.ndarray
    on_internal_grid_plane: np.ndarray
    snapped_to_boundary: np.ndarray
    sides: Tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(self.coordinate_names)
        dimension = len(names)
        requested = np.asarray(self.requested_points, dtype=float)
        points = np.asarray(self.points, dtype=float)
        indices = np.asarray(self.cell_indices)
        normalized = np.asarray(self.normalized_coordinates, dtype=float)
        lower = np.asarray(self.cell_lower, dtype=float)
        upper = np.asarray(self.cell_upper, dtype=float)
        domain_boundary = np.asarray(self.on_domain_boundary, dtype=bool)
        internal_boundary = np.asarray(self.on_internal_grid_plane, dtype=bool)
        snapped = np.asarray(self.snapped_to_boundary, dtype=bool)

        expected_shape = points.shape
        if points.ndim < 1 or points.shape[-1] != dimension:
            raise ValueError("located point arrays must end with the coordinate dimension")
        for array in (
            requested,
            indices,
            normalized,
            lower,
            upper,
            domain_boundary,
            internal_boundary,
            snapped,
        ):
            if array.shape != expected_shape:
                raise ValueError("all interpolation-location arrays must have equal shapes")
        sides = tuple(self.sides)
        if len(sides) != dimension or any(
            side not in {"left", "right"} for side in sides
        ):
            raise ValueError(
                "interpolation-location sides must contain one 'left' or "
                "'right' entry per coordinate"
            )

        object.__setattr__(self, "coordinate_names", names)
        object.__setattr__(self, "requested_points", _immutable_array(requested))
        object.__setattr__(self, "points", _immutable_array(points))
        object.__setattr__(self, "cell_indices", _immutable_array(indices, dtype=np.intp))
        object.__setattr__(
            self, "normalized_coordinates", _immutable_array(normalized)
        )
        object.__setattr__(self, "cell_lower", _immutable_array(lower))
        object.__setattr__(self, "cell_upper", _immutable_array(upper))
        object.__setattr__(
            self, "on_domain_boundary", _immutable_array(domain_boundary, dtype=bool)
        )
        object.__setattr__(
            self,
            "on_internal_grid_plane",
            _immutable_array(internal_boundary, dtype=bool),
        )
        object.__setattr__(
            self, "snapped_to_boundary", _immutable_array(snapped, dtype=bool)
        )
        object.__setattr__(self, "sides", sides)

    @property
    def query_shape(self) -> Tuple[int, ...]:
        return self.points.shape[:-1]

    @property
    def dimension(self) -> int:
        return len(self.coordinate_names)

    @property
    def was_snapped(self) -> bool:
        return bool(np.any(self.snapped_to_boundary))

    @property
    def touches_internal_grid_plane(self) -> bool:
        return bool(np.any(self.on_internal_grid_plane))


@dataclass(frozen=True)
class InterpolationEvaluation:
    """Values and local Jacobians evaluated from one shared cell lookup."""

    field_names: Tuple[str, ...]
    values: np.ndarray
    jacobians: np.ndarray
    location: InterpolationLocation

    def __post_init__(self) -> None:
        names = tuple(self.field_names)
        values = np.asarray(self.values, dtype=float)
        jacobians = np.asarray(self.jacobians, dtype=float)
        expected_values = self.location.query_shape + (len(names),)
        expected_jacobians = expected_values + (self.location.dimension,)
        if values.shape != expected_values:
            raise ValueError("evaluation values do not match point and field shapes")
        if jacobians.shape != expected_jacobians:
            raise ValueError("evaluation Jacobians do not match point and field shapes")
        object.__setattr__(self, "field_names", names)
        object.__setattr__(self, "values", _immutable_array(values))
        object.__setattr__(self, "jacobians", _immutable_array(jacobians))

    @property
    def values_defined(self) -> np.ndarray:
        """Per-field mask; allowed auxiliary ``NaN`` values remain local."""

        return np.isfinite(self.values)

    @property
    def jacobians_defined(self) -> np.ndarray:
        """Per-field/per-coordinate finite-Jacobian mask."""

        return np.isfinite(self.jacobians)


@dataclass(frozen=True)
class IntervalBounds:
    """Component-wise bounds produced from one multilinear cell.

    ``rigorous`` records whether the producing operation supplied the directed
    rounding (or exact comparisons) needed for a mathematical enclosure.  The
    ordinary whole-cell diagnostic methods below deliberately return false;
    exact extrema over stored nodes can return true.
    """

    field_names: Tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    coordinate_names: Tuple[str, ...] = ()
    rigorous: bool = False

    def __post_init__(self) -> None:
        fields = tuple(self.field_names)
        coordinates = tuple(self.coordinate_names)
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.shape != upper.shape:
            raise ValueError("interval lower and upper arrays must have equal shapes")
        trailing = (len(fields), len(coordinates)) if coordinates else (len(fields),)
        if lower.ndim < len(trailing) or lower.shape[-len(trailing) :] != trailing:
            raise ValueError("interval arrays do not match their field/coordinate labels")
        finite_pairs = np.isfinite(lower) & np.isfinite(upper)
        if np.any(lower[finite_pairs] > upper[finite_pairs]):
            raise ValueError("an interval lower bound cannot exceed its upper bound")
        object.__setattr__(self, "field_names", fields)
        object.__setattr__(self, "coordinate_names", coordinates)
        object.__setattr__(self, "lower", _immutable_array(lower))
        object.__setattr__(self, "upper", _immutable_array(upper))

    @property
    def defined(self) -> np.ndarray:
        """Mask identifying components with finite lower and upper bounds."""

        return np.isfinite(self.lower) & np.isfinite(self.upper)


class MultilinearInterpolator:
    """Interpolate selected fields of one validated rectilinear EoS table.

    Parameters
    ----------
    table:
        A :class:`ForwardTable` that has already passed validation.  The
        constructor performs no file I/O and captures an immutable snapshot of
        its axes and selected fields.
    field_names:
        Ordered dependent fields to interpolate.  The default is the canonical
        inversion target tuple (``e``, ``nB``, ...).  Coordinate fields are
        inputs and cannot be selected.  Every selected field must declare
        ``"interpolation": "linear"`` in metadata.
    boundary_absolute_tolerance, boundary_relative_tolerance:
        Explicit tolerances for snapping a point lying just *outside* an outer
        domain face to that face.  Both default to zero.  Interior points are
        never moved, and points beyond the tolerance raise
        :class:`InterpolationDomainError`.
    chunk_size:
        Maximum number of points whose corner arrays are materialized at once.

    Notes
    -----
    Internal grid planes belong to the cell on their positive/right side by
    default; the global upper endpoint necessarily belongs to the last cell.
    Values are continuous there, but Jacobians can be discontinuous.  Use
    ``plane_policy="left"`` for the opposite one-sided diagnostic,
    ``plane_policy=("left", "right", ...)`` to select a mixed adjacent cell at
    a multidimensional knot, or ``plane_policy="raise"`` when the caller wants
    to reject every grid-plane point whose differentiability has not been
    established separately.
    """

    def __init__(
        self,
        table: ForwardTable,
        field_names: Optional[Sequence[str]] = None,
        *,
        boundary_absolute_tolerance: float = 0.0,
        boundary_relative_tolerance: float = 0.0,
        chunk_size: int = 65_536,
    ) -> None:
        if not isinstance(table, ForwardTable):
            raise TypeError("table must be a ForwardTable")
        table.require_validated()
        metadata = table.metadata
        if metadata is None:  # Defensive: require_validated() should make this impossible.
            raise InterpolationConfigurationError("validated table has no metadata")

        absolute = self._finite_nonnegative_tolerance(
            boundary_absolute_tolerance, "boundary_absolute_tolerance"
        )
        relative = self._finite_nonnegative_tolerance(
            boundary_relative_tolerance, "boundary_relative_tolerance"
        )
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
            raise TypeError("chunk_size must be a positive integer")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be a positive integer")

        self._metadata = metadata
        self._source_snapshot_fingerprint = table.snapshot_fingerprint()
        self._dimension = metadata.dimension
        self._coordinate_names = metadata.coordinate_names
        self._target_names = metadata.target_names
        self._shape = tuple(table.shape)
        self._cell_shape = tuple(size - 1 for size in self._shape)
        self._axes = tuple(_immutable_array(axis, dtype=float) for axis in table.axes)
        self._boundary_absolute_tolerance = absolute
        self._boundary_relative_tolerance = relative
        self._chunk_size = int(chunk_size)

        inverse_widths = []
        for name, axis in zip(self._coordinate_names, self._axes):
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                widths = np.diff(axis)
                inverse = 1.0 / widths
            if (
                widths.size != axis.size - 1
                or np.any(widths <= 0.0)
                or not np.all(np.isfinite(widths))
                or not np.all(np.isfinite(inverse))
            ):
                raise InterpolationConfigurationError(
                    "axis '%s' does not have finite, strictly increasing, "
                    "numerically usable cells" % name
                )
            inverse_widths.append(_immutable_array(inverse))
        self._inverse_widths = tuple(inverse_widths)

        offsets = np.asarray(
            list(itertools.product((0, 1), repeat=self._dimension)), dtype=np.intp
        )
        self._corner_offsets = _immutable_array(offsets, dtype=np.intp)
        edge_pairs = []
        for axis_index in range(self._dimension):
            lower = np.flatnonzero(offsets[:, axis_index] == 0)
            upper_offsets = offsets[lower].copy()
            upper_offsets[:, axis_index] = 1
            lookup = {tuple(offset): index for index, offset in enumerate(offsets)}
            upper = np.asarray(
                [lookup[tuple(offset)] for offset in upper_offsets], dtype=np.intp
            )
            edge_pairs.append(
                (
                    _immutable_array(lower, dtype=np.intp),
                    _immutable_array(upper, dtype=np.intp),
                )
            )
        self._edge_pairs = tuple(edge_pairs)

        selected_names = self._resolve_field_names(field_names)
        grid_fields = table.grid_fields
        selected_arrays = []
        for name in selected_names:
            values = grid_fields[name]
            if values.shape != self._shape:
                raise InterpolationConfigurationError(
                    "field '%s' shape %s does not match grid shape %s"
                    % (name, values.shape, self._shape)
                )
            selected_arrays.append(values)
        self._field_names = selected_names
        self._field_arrays = tuple(selected_arrays)
        self._field_indices = MappingProxyType(
            {name: index for index, name in enumerate(self._field_names)}
        )
        self._coordinate_units = tuple(
            metadata.fields[name].canonical_unit for name in self._coordinate_names
        )
        self._field_units = tuple(
            metadata.fields[name].canonical_unit for name in self._field_names
        )
        node_lower = []
        node_upper = []
        for values in self._field_arrays:
            with np.errstate(invalid="ignore"):
                node_lower.append(float(np.min(values)))
                node_upper.append(float(np.max(values)))
        self._node_value_bounds = IntervalBounds(
            field_names=self._field_names,
            lower=np.asarray(node_lower, dtype=float),
            upper=np.asarray(node_upper, dtype=float),
            rigorous=True,
        )
        self._bounds = _immutable_array(
            np.asarray([(axis[0], axis[-1]) for axis in self._axes], dtype=float)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            endpoint_tolerances = self._boundary_tolerance(self._bounds)
        if not np.all(np.isfinite(endpoint_tolerances)):
            raise InterpolationConfigurationError(
                "boundary tolerances overflow for the source-coordinate domain"
            )

    @classmethod
    def from_rectilinear_map(
        cls,
        axes: Sequence[Any],
        values: Mapping[str, Any],
        *,
        coordinate_names: Sequence[str],
        target_names: Sequence[str],
        coordinate_units: Sequence[str],
        target_units: Sequence[str],
        boundary_absolute_tolerance: float = 0.0,
        boundary_relative_tolerance: float = 0.0,
        chunk_size: int = 65_536,
    ) -> "MultilinearInterpolator":
        """Construct a strict square target map from in-memory grid arrays.

        This constructor is intended for validated, derived maps such as the
        physical LINE Region S strip.  It deliberately accepts only a finite
        target-only square map: one target per coordinate, complete
        rectilinear axes, and no auxiliary ``NaN`` policy.  Once constructed,
        it uses exactly the same location, value, Jacobian, corner, and
        interval-enclosure implementation as a table-backed interpolator.

        ``values`` maps every target name to an array with shape
        ``tuple(len(axis) for axis in axes)``.  All inputs are copied into an
        immutable snapshot; subsequent changes to caller-owned arrays cannot
        change the map.
        """

        def names_tuple(value: Sequence[str], label: str) -> Tuple[str, ...]:
            if isinstance(value, str):
                raise InterpolationConfigurationError(
                    "%s must be a sequence of names" % label
                )
            try:
                result = tuple(value)
            except TypeError as exc:
                raise InterpolationConfigurationError(
                    "%s must be a sequence of names" % label
                ) from exc
            if not result or any(
                not isinstance(name, str) or not name for name in result
            ):
                raise InterpolationConfigurationError(
                    "%s must contain non-empty strings" % label
                )
            if len(set(result)) != len(result):
                raise InterpolationConfigurationError(
                    "%s contains duplicate names" % label
                )
            return result

        def units_tuple(
            value: Sequence[str], label: str, expected: int
        ) -> Tuple[str, ...]:
            if isinstance(value, str):
                raise InterpolationConfigurationError(
                    "%s must contain one unit per field" % label
                )
            try:
                result = tuple(value)
            except TypeError as exc:
                raise InterpolationConfigurationError(
                    "%s must contain one unit per field" % label
                ) from exc
            if len(result) != expected or any(
                not isinstance(unit, str) or not unit for unit in result
            ):
                raise InterpolationConfigurationError(
                    "%s must contain %d non-empty string units"
                    % (label, expected)
                )
            return result

        coordinates = names_tuple(coordinate_names, "coordinate_names")
        targets = names_tuple(target_names, "target_names")
        dimension = len(coordinates)
        if dimension not in range(1, 5):
            raise InterpolationConfigurationError(
                "rectilinear target maps require dimension 1--4"
            )
        if len(targets) != dimension:
            raise InterpolationConfigurationError(
                "rectilinear target maps must be square with one target per "
                "coordinate"
            )
        overlap = sorted(set(coordinates) & set(targets))
        if overlap:
            raise InterpolationConfigurationError(
                "coordinate and target names must be disjoint; overlap=%s"
                % overlap
            )

        try:
            supplied_axes = tuple(axes)
        except TypeError as exc:
            raise InterpolationConfigurationError(
                "axes must contain one one-dimensional array per coordinate"
            ) from exc
        if len(supplied_axes) != dimension:
            raise InterpolationConfigurationError(
                "axes must contain exactly %d arrays" % dimension
            )
        resolved_axes = []
        inverse_widths = []
        for name, supplied in zip(coordinates, supplied_axes):
            try:
                axis = np.asarray(supplied, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise InterpolationConfigurationError(
                    "axis '%s' must be numeric" % name
                ) from exc
            if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
                raise InterpolationConfigurationError(
                    "axis '%s' must be a finite one-dimensional array with at "
                    "least two nodes" % name
                )
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                widths = np.diff(axis)
                inverse = 1.0 / widths
            if (
                np.any(widths <= 0.0)
                or not np.all(np.isfinite(widths))
                or not np.all(np.isfinite(inverse))
            ):
                raise InterpolationConfigurationError(
                    "axis '%s' does not have finite, strictly increasing, "
                    "numerically usable cells" % name
                )
            resolved_axes.append(_immutable_array(axis))
            inverse_widths.append(_immutable_array(inverse))
        resolved_axes_tuple = tuple(resolved_axes)
        shape = tuple(axis.size for axis in resolved_axes_tuple)

        if not isinstance(values, Mapping):
            raise InterpolationConfigurationError(
                "values must map every target name to one grid array"
            )
        supplied_names = set(values)
        expected_names = set(targets)
        if supplied_names != expected_names:
            missing = sorted(expected_names - supplied_names)
            extra = sorted(supplied_names - expected_names, key=str)
            raise InterpolationConfigurationError(
                "values must contain exactly the target names; missing=%s, "
                "extra=%s" % (missing, extra)
            )
        field_arrays = []
        for name in targets:
            try:
                field = np.asarray(values[name], dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise InterpolationConfigurationError(
                    "target field '%s' must be numeric" % name
                ) from exc
            if field.shape != shape:
                raise InterpolationConfigurationError(
                    "target field '%s' shape %s does not match grid shape %s"
                    % (name, field.shape, shape)
                )
            if not np.all(np.isfinite(field)):
                raise InterpolationConfigurationError(
                    "target field '%s' must be finite at every grid node" % name
                )
            field_arrays.append(_immutable_array(field))

        coordinate_unit_tuple = units_tuple(
            coordinate_units, "coordinate_units", dimension
        )
        target_unit_tuple = units_tuple(target_units, "target_units", dimension)
        absolute = cls._finite_nonnegative_tolerance(
            boundary_absolute_tolerance, "boundary_absolute_tolerance"
        )
        relative = cls._finite_nonnegative_tolerance(
            boundary_relative_tolerance, "boundary_relative_tolerance"
        )
        if isinstance(chunk_size, bool) or not isinstance(
            chunk_size, (int, np.integer)
        ):
            raise TypeError("chunk_size must be a positive integer")
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be a positive integer")

        instance = cls.__new__(cls)
        instance._metadata = None
        instance._dimension = dimension
        instance._coordinate_names = coordinates
        instance._target_names = targets
        instance._shape = shape
        instance._cell_shape = tuple(size - 1 for size in shape)
        instance._axes = resolved_axes_tuple
        instance._boundary_absolute_tolerance = absolute
        instance._boundary_relative_tolerance = relative
        instance._chunk_size = int(chunk_size)
        instance._inverse_widths = tuple(inverse_widths)

        offsets = np.asarray(
            list(itertools.product((0, 1), repeat=dimension)), dtype=np.intp
        )
        instance._corner_offsets = _immutable_array(offsets, dtype=np.intp)
        lookup = {tuple(offset): index for index, offset in enumerate(offsets)}
        edge_pairs = []
        for axis_index in range(dimension):
            lower = np.flatnonzero(offsets[:, axis_index] == 0)
            upper_offsets = offsets[lower].copy()
            upper_offsets[:, axis_index] = 1
            upper = np.asarray(
                [lookup[tuple(offset)] for offset in upper_offsets], dtype=np.intp
            )
            edge_pairs.append(
                (
                    _immutable_array(lower, dtype=np.intp),
                    _immutable_array(upper, dtype=np.intp),
                )
            )
        instance._edge_pairs = tuple(edge_pairs)
        instance._field_names = targets
        instance._field_arrays = tuple(field_arrays)
        instance._field_indices = MappingProxyType(
            {name: index for index, name in enumerate(targets)}
        )
        instance._coordinate_units = coordinate_unit_tuple
        instance._field_units = target_unit_tuple
        instance._node_value_bounds = IntervalBounds(
            field_names=targets,
            lower=np.asarray([np.min(field) for field in field_arrays]),
            upper=np.asarray([np.max(field) for field in field_arrays]),
            rigorous=True,
        )
        instance._bounds = _immutable_array(
            np.asarray([(axis[0], axis[-1]) for axis in resolved_axes_tuple])
        )
        with np.errstate(over="ignore", invalid="ignore"):
            endpoint_tolerances = instance._boundary_tolerance(instance._bounds)
        if not np.all(np.isfinite(endpoint_tolerances)):
            raise InterpolationConfigurationError(
                "boundary tolerances overflow for the source-coordinate domain"
            )
        instance._source_snapshot_fingerprint = _derived_map_fingerprint(
            coordinates,
            targets,
            coordinate_unit_tuple,
            target_unit_tuple,
            resolved_axes_tuple,
            instance._field_arrays,
            absolute,
            relative,
        )
        return instance

    @staticmethod
    def _finite_nonnegative_tolerance(value: Any, name: str) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("%s must be a finite non-negative number" % name)
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("%s must be a finite non-negative number" % name) from exc
        if not math.isfinite(converted) or converted < 0.0:
            raise ValueError("%s must be a finite non-negative number" % name)
        return converted

    def _resolve_field_names(
        self, field_names: Optional[Sequence[str]]
    ) -> Tuple[str, ...]:
        if field_names is None:
            names = tuple(self._target_names)
        elif isinstance(field_names, str):
            names = (field_names,)
        else:
            try:
                names = tuple(field_names)
            except TypeError as exc:
                raise TypeError("field_names must be a sequence of field names") from exc
        if not names:
            raise InterpolationConfigurationError(
                "at least one dependent field must be selected"
            )
        if any(not isinstance(name, str) or not name for name in names):
            raise InterpolationConfigurationError(
                "field_names must contain non-empty strings"
            )
        if len(set(names)) != len(names):
            raise InterpolationConfigurationError("field_names contains duplicates")

        for name in names:
            spec = self._metadata.fields.get(name)
            if spec is None:
                raise InterpolationConfigurationError(
                    "unknown EoS field '%s'" % name
                )
            if spec.role == "coordinate":
                raise InterpolationConfigurationError(
                    "coordinate field '%s' is an interpolation input, not an output"
                    % name
                )
            if spec.interpolation != "linear":
                declared = "null" if spec.interpolation is None else spec.interpolation
                raise InterpolationConfigurationError(
                    "field '%s' declares interpolation=%s; "
                    "MultilinearInterpolator requires 'linear'" % (name, declared)
                )
        return names

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def coordinate_names(self) -> Tuple[str, ...]:
        return self._coordinate_names

    @property
    def target_names(self) -> Tuple[str, ...]:
        return self._target_names

    @property
    def field_names(self) -> Tuple[str, ...]:
        return self._field_names

    @property
    def source_snapshot_fingerprint(self) -> str:
        """Stable identity of the immutable table or derived-map snapshot."""

        return self._source_snapshot_fingerprint

    @property
    def coordinate_units(self) -> Tuple[str, ...]:
        return self._coordinate_units

    @property
    def field_units(self) -> Tuple[str, ...]:
        return self._field_units

    @property
    def axes(self) -> Tuple[np.ndarray, ...]:
        return self._axes

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def cell_shape(self) -> Tuple[int, ...]:
        return self._cell_shape

    @property
    def corner_offsets(self) -> np.ndarray:
        """Binary corner offsets; the last coordinate varies fastest."""

        return self._corner_offsets

    @property
    def bounds(self) -> np.ndarray:
        """Immutable ``(dimension, 2)`` source-coordinate domain."""

        return self._bounds

    @property
    def node_value_bounds(self) -> IntervalBounds:
        """Global extrema of each selected field over stored grid nodes.

        A multilinear scalar field cannot exceed the extrema of its cell
        corners.  Therefore finite node extrema also enclose that field over
        the complete interpolation domain.  These bounds use only exact
        comparisons between stored binary64 values and are marked rigorous;
        fields containing an allowed ``NaN`` remain explicitly undefined.
        """

        return self._node_value_bounds

    @property
    def domain(self) -> Mapping[str, Tuple[float, float]]:
        return MappingProxyType(
            {
                name: (float(bounds[0]), float(bounds[1]))
                for name, bounds in zip(self._coordinate_names, self._bounds)
            }
        )

    @property
    def boundary_absolute_tolerance(self) -> float:
        return self._boundary_absolute_tolerance

    @property
    def boundary_relative_tolerance(self) -> float:
        return self._boundary_relative_tolerance

    def contains(
        self, points: Any, *, include_boundary_tolerance: bool = True
    ) -> Any:
        """Return whether each query point lies in the closed source domain."""

        if not isinstance(include_boundary_tolerance, (bool, np.bool_)):
            raise TypeError("include_boundary_tolerance must be a boolean")
        flat, query_shape = self._prepare_points(points, require_finite=False)
        finite = np.all(np.isfinite(flat), axis=1)
        lower = self._bounds[:, 0]
        upper = self._bounds[:, 1]
        if include_boundary_tolerance:
            lower_tolerance = self._boundary_tolerance(lower)
            upper_tolerance = self._boundary_tolerance(upper)
        else:
            lower_tolerance = np.zeros(self._dimension, dtype=float)
            upper_tolerance = np.zeros(self._dimension, dtype=float)
        inside = finite & np.all(
            (flat >= lower - lower_tolerance)
            & (flat <= upper + upper_tolerance),
            axis=1,
        )
        shaped = inside.reshape(query_shape)
        return bool(shaped) if query_shape == () else shaped

    def locate(self, points: Any, *, side: Any = "right") -> InterpolationLocation:
        """Locate points without extrapolation.

        ``side`` may be one global ``"left"``/``"right"`` convention or one
        convention per coordinate, allowing every adjacent cell at a
        multidimensional knot to be addressed explicitly.
        """

        sides = self._normalize_sides(side, argument_name="side")
        requested, query_shape = self._prepare_points(points, require_finite=True)
        evaluated = requested.copy()
        snapped = np.zeros(evaluated.shape, dtype=bool)
        outside = np.zeros(evaluated.shape, dtype=bool)

        for axis_index, axis in enumerate(self._axes):
            lower = float(axis[0])
            upper = float(axis[-1])
            values = evaluated[:, axis_index]
            below = values < lower
            above = values > upper
            lower_tolerance = float(self._boundary_tolerance(np.asarray(lower)))
            upper_tolerance = float(self._boundary_tolerance(np.asarray(upper)))
            snap_lower = below & ((lower - values) <= lower_tolerance)
            snap_upper = above & ((values - upper) <= upper_tolerance)
            evaluated[snap_lower, axis_index] = lower
            evaluated[snap_upper, axis_index] = upper
            snapped[snap_lower | snap_upper, axis_index] = True
            outside[:, axis_index] = (below & ~snap_lower) | (above & ~snap_upper)

        invalid = np.any(outside, axis=1)
        if np.any(invalid):
            invalid_indices = np.flatnonzero(invalid)
            raise InterpolationDomainError(
                self._coordinate_names,
                self._bounds,
                requested[invalid],
                invalid_indices,
                query_shape,
            )

        point_count = evaluated.shape[0]
        cell_indices = np.empty((point_count, self._dimension), dtype=np.intp)
        normalized = np.empty((point_count, self._dimension), dtype=float)
        cell_lower = np.empty((point_count, self._dimension), dtype=float)
        cell_upper = np.empty((point_count, self._dimension), dtype=float)
        on_domain = np.zeros((point_count, self._dimension), dtype=bool)
        on_internal = np.zeros((point_count, self._dimension), dtype=bool)

        for axis_index, axis in enumerate(self._axes):
            values = evaluated[:, axis_index]
            indices = np.searchsorted(axis, values, side=sides[axis_index]) - 1
            indices = np.clip(indices, 0, axis.size - 2).astype(np.intp, copy=False)
            lower_values = axis[indices]
            upper_values = axis[indices + 1]
            fractions = (values - lower_values) * self._inverse_widths[axis_index][indices]

            # Preserve exact face weights.  The product ``cell_width *
            # inverse_width`` is not guaranteed to round to exactly one, and a
            # tiny spurious weight could otherwise activate an off-face NaN.
            fractions[values == lower_values] = 0.0
            fractions[values == upper_values] = 1.0

            # Validation above established domain membership.  Clipping here
            # only removes normalization roundoff; it never admits a point.
            fractions = np.clip(fractions, 0.0, 1.0)
            cell_indices[:, axis_index] = indices
            normalized[:, axis_index] = fractions
            cell_lower[:, axis_index] = lower_values
            cell_upper[:, axis_index] = upper_values
            on_domain[:, axis_index] = (values == axis[0]) | (values == axis[-1])

            insertion = np.searchsorted(axis, values, side="left")
            clipped = np.clip(insertion, 0, axis.size - 1)
            is_node = axis[clipped] == values
            on_internal[:, axis_index] = (
                is_node & (insertion > 0) & (insertion < axis.size - 1)
            )

        point_shape = query_shape + (self._dimension,)
        return InterpolationLocation(
            coordinate_names=self._coordinate_names,
            requested_points=requested.reshape(point_shape),
            points=evaluated.reshape(point_shape),
            cell_indices=cell_indices.reshape(point_shape),
            normalized_coordinates=normalized.reshape(point_shape),
            cell_lower=cell_lower.reshape(point_shape),
            cell_upper=cell_upper.reshape(point_shape),
            on_domain_boundary=on_domain.reshape(point_shape),
            on_internal_grid_plane=on_internal.reshape(point_shape),
            snapped_to_boundary=snapped.reshape(point_shape),
            sides=sides,
        )

    def evaluate(self, points: Any, *, side: Any = "right") -> np.ndarray:
        """Evaluate all configured fields at one point or a point batch."""

        location = self.locate(points, side=side)
        values, _ = self._evaluate_location(location, include_jacobian=False)
        return values

    __call__ = evaluate

    def evaluate_field(
        self, name: str, points: Any, *, side: Any = "right"
    ) -> Any:
        """Evaluate one field configured on this interpolator."""

        try:
            field_index = self._field_indices[name]
        except (KeyError, TypeError) as exc:
            raise KeyError("field '%s' is not configured on this interpolator" % name) from exc
        values = self.evaluate(points, side=side)[..., field_index]
        return float(values) if np.asarray(values).ndim == 0 else values

    def evaluate_dict(
        self, points: Any, *, side: Any = "right"
    ) -> Mapping[str, Any]:
        """Evaluate fields and return a read-only name-to-result mapping."""

        values = self.evaluate(points, side=side)
        result: Dict[str, Any] = {}
        for field_index, name in enumerate(self._field_names):
            selected = values[..., field_index]
            result[name] = float(selected) if np.asarray(selected).ndim == 0 else selected
        return MappingProxyType(result)

    def jacobian(
        self, points: Any, *, plane_policy: Any = "right"
    ) -> np.ndarray:
        """Return local Jacobians with shape ``(..., fields, coordinates)``.

        ``plane_policy`` accepts ``"left"``, ``"right"``, ``"raise"``, or a
        per-coordinate sequence of left/right choices. ``"raise"`` is
        deliberately conservative: it rejects an internal-plane point without
        first checking whether all adjacent derivatives happen to agree.
        """

        location = self._location_for_derivative(points, plane_policy)
        _, jacobians = self._evaluate_location(location, include_jacobian=True)
        assert jacobians is not None
        return jacobians

    def evaluate_with_jacobian(
        self, points: Any, *, plane_policy: Any = "right"
    ) -> InterpolationEvaluation:
        """Evaluate values and Jacobians from one shared point lookup."""

        location = self._location_for_derivative(points, plane_policy)
        values, jacobians = self._evaluate_location(location, include_jacobian=True)
        assert jacobians is not None
        return InterpolationEvaluation(
            field_names=self._field_names,
            values=values,
            jacobians=jacobians,
            location=location,
        )

    def _location_for_derivative(
        self, points: Any, plane_policy: Any
    ) -> InterpolationLocation:
        raise_on_plane = isinstance(plane_policy, str) and plane_policy == "raise"
        if raise_on_plane:
            sides = ("right",) * self._dimension
        else:
            sides = self._normalize_sides(
                plane_policy, argument_name="plane_policy"
            )
        location = self.locate(points, side=sides)
        if raise_on_plane and location.touches_internal_grid_plane:
            raise InterpolationDerivativeError(location)
        return location

    def _evaluate_location(
        self, location: InterpolationLocation, *, include_jacobian: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        query_shape = location.query_shape
        cell_indices = location.cell_indices.reshape(-1, self._dimension)
        normalized = location.normalized_coordinates.reshape(-1, self._dimension)
        point_count = cell_indices.shape[0]
        field_count = len(self._field_names)
        values = np.empty((point_count, field_count), dtype=float)
        jacobians = (
            np.empty((point_count, field_count, self._dimension), dtype=float)
            if include_jacobian
            else None
        )

        for start in range(0, point_count, self._chunk_size):
            stop = min(start + self._chunk_size, point_count)
            chunk_cells = cell_indices[start:stop]
            chunk_normalized = normalized[start:stop]
            corners = self._gather_corner_values(chunk_cells)
            values[start:stop] = self._multilinear_reduce(
                corners, chunk_normalized
            )

            if jacobians is not None:
                for axis_index, (lower_ids, upper_ids) in enumerate(
                    self._edge_pairs
                ):
                    inverse_width = self._inverse_widths[axis_index][
                        chunk_cells[:, axis_index]
                    ]
                    slopes = self._edge_slopes(
                        corners[:, lower_ids, :],
                        corners[:, upper_ids, :],
                        inverse_width,
                    )
                    other_coordinates = np.delete(
                        chunk_normalized, axis_index, axis=1
                    )
                    jacobians[start:stop, :, axis_index] = (
                        self._multilinear_reduce(slopes, other_coordinates)
                    )

        value_shape = query_shape + (field_count,)
        reshaped_values = values.reshape(value_shape)
        if jacobians is None:
            return reshaped_values, None
        jacobian_shape = value_shape + (self._dimension,)
        return reshaped_values, jacobians.reshape(jacobian_shape)

    @staticmethod
    def _multilinear_reduce(
        corner_values: np.ndarray, normalized_coordinates: np.ndarray
    ) -> np.ndarray:
        """Contract corners one coordinate at a time with exact face semantics.

        A staged contraction avoids explicitly forming products of several tiny
        positive weights.  Such a product can underflow before multiplication
        by a large corner value even when the final contribution is
        representable.  Exact zero/one coordinates select one face directly,
        so a NaN on a mathematically inactive face cannot contaminate a value.
        """

        point_count, corner_count, field_count = corner_values.shape
        dimension = normalized_coordinates.shape[1]
        if corner_count != 1 << dimension:
            raise InterpolationConfigurationError(
                "corner array does not match the interpolation dimension"
            )
        work = corner_values.reshape(
            (point_count,) + (2,) * dimension + (field_count,)
        )
        for axis_index in range(dimension):
            lower = work[:, 0, ...]
            upper = work[:, 1, ...]
            factor_shape = (point_count,) + (1,) * (lower.ndim - 1)
            factor = normalized_coordinates[:, axis_index].reshape(factor_shape)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                work = lower * (1.0 - factor) + upper * factor
            lower_face = np.broadcast_to(factor == 0.0, work.shape)
            upper_face = np.broadcast_to(factor == 1.0, work.shape)
            np.copyto(work, lower, where=lower_face)
            np.copyto(work, upper, where=upper_face)
        return work.reshape(point_count, field_count)

    @staticmethod
    def _edge_slopes(
        lower: np.ndarray, upper: np.ndarray, inverse_width: np.ndarray
    ) -> np.ndarray:
        """Compute divided differences with an overflow-safe fallback."""

        scale = inverse_width[:, None, None]
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            slopes = (upper - lower) * scale
        recoverable = (
            ~np.isfinite(slopes) & np.isfinite(lower) & np.isfinite(upper)
        )
        if np.any(recoverable):
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                scaled_slopes = upper * scale - lower * scale
            use_scaled = recoverable & np.isfinite(scaled_slopes)
            slopes = np.where(use_scaled, scaled_slopes, slopes)
        return slopes

    def _gather_corner_values(self, cell_indices: np.ndarray) -> np.ndarray:
        point_count = cell_indices.shape[0]
        corner_count = self._corner_offsets.shape[0]
        index = tuple(
            cell_indices[:, axis_index, None]
            + self._corner_offsets[None, :, axis_index]
            for axis_index in range(self._dimension)
        )
        if not self._field_arrays:
            return np.empty((point_count, corner_count, 0), dtype=float)
        return np.stack([values[index] for values in self._field_arrays], axis=2)

    def cell_coordinate_bounds(
        self, cell_indices: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower/upper coordinates for one cell or a cell batch."""

        flat, query_shape = self._prepare_cell_indices(cell_indices)
        lower = np.empty((flat.shape[0], self._dimension), dtype=float)
        upper = np.empty_like(lower)
        for axis_index, axis in enumerate(self._axes):
            indices = flat[:, axis_index]
            lower[:, axis_index] = axis[indices]
            upper[:, axis_index] = axis[indices + 1]
        shape = query_shape + (self._dimension,)
        return lower.reshape(shape), upper.reshape(shape)

    def cell_corner_values(self, cell_indices: Any) -> np.ndarray:
        """Return all field values at a cell's ``2**dimension`` corners."""

        flat, query_shape = self._prepare_cell_indices(cell_indices)
        corners = self._gather_corner_values(flat)
        shape = query_shape + (self._corner_offsets.shape[0], len(self._field_names))
        return corners.reshape(shape)

    def cell_value_bounds(self, cell_indices: Any) -> IntervalBounds:
        """Return diagnostic component extrema over complete cells.

        If any required corner of an auxiliary field is ``NaN``, both bounds
        for that field are ``NaN``.  ``nanmin``/``nanmax`` are intentionally not
        used because they would manufacture an apparently valid range.  This
        legacy diagnostic API remains marked non-rigorous; proof-bearing
        callers should use :meth:`cell_value_enclosures`.
        """

        corners = self.cell_corner_values(cell_indices)
        with np.errstate(invalid="ignore"):
            lower = np.min(corners, axis=-2)
            upper = np.max(corners, axis=-2)
        return IntervalBounds(
            field_names=self._field_names,
            lower=lower,
            upper=upper,
            rigorous=False,
        )

    def cell_value_enclosures(self, cell_indices: Any) -> IntervalBounds:
        """Return proof-safe cell-image boxes from stored corner comparisons.

        A scalar multilinear field on a complete cell is a convex combination
        of its ``2**dimension`` stored corner values.  Its mathematical range
        is therefore contained between the smallest and largest corner, and
        obtaining those extrema requires comparisons only—no rounded interval
        arithmetic.  Finite components are consequently marked rigorous.

        This method does *not* add a numerical root-acceptance tolerance.
        Table-wide reachability code must expand these exact image boxes
        outward separately when deciding whether a target could satisfy a
        nonzero residual tolerance.

        Fields with a required ``NaN`` corner remain undefined: both returned
        bounds are ``NaN`` for that component.
        """

        corners = self.cell_corner_values(cell_indices)
        with np.errstate(invalid="ignore"):
            lower = np.min(corners, axis=-2)
            upper = np.max(corners, axis=-2)
        return IntervalBounds(
            field_names=self._field_names,
            lower=lower,
            upper=upper,
            rigorous=True,
        )

    def cell_jacobian_bounds(self, cell_indices: Any) -> IntervalBounds:
        """Return tight component-wise Jacobian ranges over whole cells.

        Each derivative is multilinear in the other coordinates, so its extrema
        are the extrema of the parallel edge slopes.  The returned floating
        bounds are useful for pruning and diagnostics but are not directed-
        rounding interval certificates.
        """

        flat, query_shape = self._prepare_cell_indices(cell_indices)
        corners = self._gather_corner_values(flat)
        point_count = flat.shape[0]
        field_count = len(self._field_names)
        lower = np.empty((point_count, field_count, self._dimension), dtype=float)
        upper = np.empty_like(lower)

        for axis_index, (lower_ids, upper_ids) in enumerate(self._edge_pairs):
            inverse_width = self._inverse_widths[axis_index][flat[:, axis_index]]
            with np.errstate(invalid="ignore"):
                slopes = self._edge_slopes(
                    corners[:, lower_ids, :],
                    corners[:, upper_ids, :],
                    inverse_width,
                )
                lower[:, :, axis_index] = np.min(slopes, axis=1)
                upper[:, :, axis_index] = np.max(slopes, axis=1)

        shape = query_shape + (field_count, self._dimension)
        return IntervalBounds(
            field_names=self._field_names,
            coordinate_names=self._coordinate_names,
            lower=lower.reshape(shape),
            upper=upper.reshape(shape),
            rigorous=False,
        )

    def _prepare_points(
        self, points: Any, *, require_finite: bool
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        try:
            array = np.asarray(points, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InterpolationInputError("query points must be numeric") from exc
        if array.ndim == 0 or array.shape[-1] != self._dimension:
            raise InterpolationInputError(
                "query points must have shape (..., %d)" % self._dimension
            )
        query_shape = tuple(int(size) for size in array.shape[:-1])
        flat = array.reshape(-1, self._dimension)
        if require_finite and not np.all(np.isfinite(flat)):
            invalid = np.flatnonzero(~np.all(np.isfinite(flat), axis=1))
            raise InterpolationInputError(
                "%d query point(s) contain NaN or infinity" % invalid.size
            )
        return flat, query_shape

    def _prepare_cell_indices(
        self, cell_indices: Any
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        array = np.asarray(cell_indices)
        if (
            array.ndim == 0
            or array.shape[-1] != self._dimension
            or array.dtype.kind not in {"i", "u"}
        ):
            raise InterpolationCellError(
                "cell indices must be an integer array with shape (..., %d)"
                % self._dimension
            )
        query_shape = tuple(int(size) for size in array.shape[:-1])
        flat = array.reshape(-1, self._dimension).astype(np.intp, copy=False)
        invalid = np.zeros(flat.shape, dtype=bool)
        for axis_index, cell_count in enumerate(self._cell_shape):
            invalid[:, axis_index] = (
                (flat[:, axis_index] < 0) | (flat[:, axis_index] >= cell_count)
            )
        if np.any(invalid):
            first = flat[np.flatnonzero(np.any(invalid, axis=1))[0]]
            raise InterpolationCellError(
                "cell index %s is outside cell shape %s"
                % (np.array2string(first), self._cell_shape)
            )
        return flat, query_shape

    def _boundary_tolerance(self, boundary: np.ndarray) -> np.ndarray:
        return self._boundary_absolute_tolerance + (
            self._boundary_relative_tolerance * np.abs(boundary)
        )

    def _normalize_sides(
        self, value: Any, *, argument_name: str
    ) -> Tuple[str, ...]:
        if isinstance(value, str):
            sides = (value,) * self._dimension
        else:
            try:
                sides = tuple(value)
            except TypeError as exc:
                raise ValueError(
                    "%s must be 'left', 'right', or a per-coordinate sequence"
                    % argument_name
                ) from exc
        if len(sides) != self._dimension or any(
            side not in {"left", "right"} for side in sides
        ):
            raise ValueError(
                "%s must be 'left', 'right', or a length-%d sequence of "
                "those values" % (argument_name, self._dimension)
            )
        return sides


__all__ = [
    "InterpolationCellError",
    "InterpolationConfigurationError",
    "InterpolationDerivativeError",
    "InterpolationDomainError",
    "InterpolationError",
    "InterpolationEvaluation",
    "InterpolationInputError",
    "InterpolationLocation",
    "IntervalBounds",
    "MultilinearInterpolator",
]
