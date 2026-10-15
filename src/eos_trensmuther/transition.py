"""Standalone physical first-order-transition domain models.

This module consumes the canonical table and sidecar snapshots validated by
``ForwardTable``.  It defines bounded piecewise-linear LINE and BINODALS
queries plus corrected physical LINE RegionS endpoint reconstruction.  It
deliberately has no solver, reachability, output-format, or
transformed-coordinate dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

from .tables import (
    CoexistenceTable,
    EoSError,
    FirstOrderTransitionSpec,
    ForwardTable,
)


class TransitionError(EoSError):
    """Base exception for first-order transition-domain errors."""


class TransitionConfigurationError(TransitionError):
    """Raised when validated metadata and canonical sidecar data disagree."""


class TransitionInputError(TransitionError):
    """Raised when a query does not have the required finite numeric shape."""


class TransitionDomainError(TransitionError):
    """Raised when bounded evaluation would require extrapolation."""


class RegionSReconstructionError(TransitionError):
    """Raised when a physical LINE endpoint curve cannot be reconstructed."""


class FirstOrderRepresentation(str, Enum):
    """Supported physical representations of an equilibrium transition."""

    LINE = "line"
    BINODALS = "binodals"


class PhaseLabel(str, Enum):
    """Thermodynamic phase label, independent of numerical solver status."""

    NOT_APPLICABLE = "not_applicable"
    SINGLE_PHASE = "single_phase"
    MIXED_PHASE = "mixed_phase"


class PhaseRootSelectionReason(str, Enum):
    """Reason a phase-aware solver accepted its selected source root."""

    NOT_APPLICABLE = "not_applicable"
    UNAMBIGUOUS = "unambiguous"
    SELECTED_FROM_JACOBIAN = "SELECTED_FROM_JACOBIAN"


_REGION_S_ALGORITHM = "physical_side_spline_v3"
_CRITICAL_ENDPOINT_POLICY = "numerical_physical_center_separation_closure"
_UNMERGED_ENDPOINT_POLICY = "unmerged_one_sided_endpoints"
_CRITICAL_CLOSURE_WEIGHT = "smoothstep_3s2_minus_2s3"


def _immutable_array(value: Any, dtype: Optional[Any] = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    contiguous = np.ascontiguousarray(array)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return immutable.reshape(contiguous.shape)


def _finite_vector(value: Any, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind == "b":
            raise TypeError
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TransitionConfigurationError(
            "%s must be a finite one-dimensional numeric array" % name
        ) from exc
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise TransitionConfigurationError(
            "%s must be a finite one-dimensional numeric array with at least two values"
            % name
        )
    return array


def _finite_query(value: Any, name: str) -> Tuple[np.ndarray, bool]:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind == "b":
            raise TypeError
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TransitionInputError("%s must be finite and numeric" % name) from exc
    if not np.all(np.isfinite(array)):
        raise TransitionInputError("%s must contain only finite values" % name)
    return array, array.ndim == 0


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TransitionInputError("%s must be finite and non-negative" % name)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TransitionInputError(
            "%s must be finite and non-negative" % name
        ) from exc
    if not math.isfinite(converted) or converted < 0.0:
        raise TransitionInputError("%s must be finite and non-negative" % name)
    return converted


def _restore_numeric(array: np.ndarray, scalar: bool) -> Any:
    if scalar:
        return float(np.asarray(array).reshape(()))
    return _immutable_array(array, dtype=float)


def _restore_boolean(array: np.ndarray, scalar: bool) -> Any:
    if scalar:
        return bool(np.asarray(array).reshape(()))
    return _immutable_array(array, dtype=bool)


def _strictly_increasing(axis: np.ndarray, name: str) -> None:
    if np.any(np.diff(axis) <= 0.0):
        raise TransitionConfigurationError("%s must be strictly increasing" % name)


def _bounded_interpolate(
    axis: np.ndarray,
    values: np.ndarray,
    query: Any,
    *,
    query_name: str = "muB",
) -> Any:
    requested, scalar = _finite_query(query, query_name)
    outside = (requested < axis[0]) | (requested > axis[-1])
    if np.any(outside):
        flat_index = int(np.flatnonzero(outside.reshape(-1))[0])
        offending = float(requested.reshape(-1)[flat_index])
        raise TransitionDomainError(
            "%s=%r lies outside the closed transition domain [%r, %r]; "
            "extrapolation is forbidden"
            % (query_name, offending, float(axis[0]), float(axis[-1]))
        )
    interpolated = np.interp(requested.reshape(-1), axis, values).reshape(
        requested.shape
    )
    return _restore_numeric(interpolated, scalar)


@dataclass(frozen=True)
class TransitionDiagnostics:
    """Immutable provenance and bounds shared by transition-domain models."""

    representation: FirstOrderRepresentation
    point_count: int
    muB_bounds: Tuple[float, float]
    temperature_bounds: Mapping[str, Tuple[float, float]]
    sidecar_fingerprint: str
    critical_endpoint: Optional[str] = None
    interpolation: str = "piecewise_linear"
    extrapolation: str = "forbidden"
    source_membership_boundary: str = "closed"

    def __post_init__(self) -> None:
        if self.point_count < 2:
            raise ValueError("transition diagnostics require at least two points")
        lower, upper = self.muB_bounds
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("transition diagnostics require finite muB bounds")
        frozen_bounds = {
            str(name): (float(bounds[0]), float(bounds[1]))
            for name, bounds in self.temperature_bounds.items()
        }
        object.__setattr__(
            self, "temperature_bounds", MappingProxyType(frozen_bounds)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "representation": self.representation.value,
            "point_count": self.point_count,
            "muB_bounds": list(self.muB_bounds),
            "temperature_bounds": {
                name: list(bounds)
                for name, bounds in self.temperature_bounds.items()
            },
            "sidecar_fingerprint": self.sidecar_fingerprint,
            "critical_endpoint": self.critical_endpoint,
            "interpolation": self.interpolation,
            "extrapolation": self.extrapolation,
            "source_membership_boundary": self.source_membership_boundary,
        }


@dataclass(frozen=True)
class TransitionLine:
    """Bounded piecewise-linear physical coexistence line ``T(muB)``."""

    muB: np.ndarray
    temperature: np.ndarray
    sidecar_fingerprint: str
    critical_endpoint: str
    critical_point_name: str
    kind: str
    phase_label: PhaseLabel = PhaseLabel.MIXED_PHASE
    diagnostics: TransitionDiagnostics = field(init=False)

    def __post_init__(self) -> None:
        muB = _finite_vector(self.muB, "muB")
        temperature = _finite_vector(self.temperature, "temperature")
        if temperature.shape != muB.shape:
            raise TransitionConfigurationError(
                "transition temperature and muB arrays must have equal shape"
            )
        _strictly_increasing(muB, "muB")
        if self.critical_endpoint not in {"first", "last"}:
            raise TransitionConfigurationError(
                "critical_endpoint must be 'first' or 'last'"
            )
        for name, value in (
            ("sidecar_fingerprint", self.sidecar_fingerprint),
            ("critical_point_name", self.critical_point_name),
            ("kind", self.kind),
        ):
            if not isinstance(value, str) or not value:
                raise TransitionConfigurationError("%s must be non-empty" % name)
        if self.phase_label is not PhaseLabel.MIXED_PHASE:
            raise TransitionConfigurationError(
                "a transition line must carry the MIXED_PHASE label"
            )
        immutable_muB = _immutable_array(muB, dtype=float)
        immutable_temperature = _immutable_array(temperature, dtype=float)
        object.__setattr__(self, "muB", immutable_muB)
        object.__setattr__(self, "temperature", immutable_temperature)
        object.__setattr__(
            self,
            "diagnostics",
            TransitionDiagnostics(
                representation=FirstOrderRepresentation.LINE,
                point_count=int(muB.size),
                muB_bounds=(float(muB[0]), float(muB[-1])),
                temperature_bounds={
                    "T": (float(np.min(temperature)), float(np.max(temperature)))
                },
                sidecar_fingerprint=self.sidecar_fingerprint,
                critical_endpoint=self.critical_endpoint,
            ),
        )

    @classmethod
    def from_validated_data(
        cls,
        spec: FirstOrderTransitionSpec,
        data: CoexistenceTable,
    ) -> "TransitionLine":
        if spec.representation != FirstOrderRepresentation.LINE.value:
            raise TransitionConfigurationError("metadata do not describe a LINE")
        if data.representation != FirstOrderRepresentation.LINE.value:
            raise TransitionConfigurationError("sidecar data do not describe a LINE")
        if data.names != ("muB", "T") or data.units != ("GeV", "GeV"):
            raise TransitionConfigurationError(
                "LINE sidecar must provide canonical (muB, T) in GeV"
            )
        if data.critical_endpoint not in {"first", "last"}:
            raise TransitionConfigurationError(
                "validated LINE data lack a unique critical endpoint"
            )
        if not spec.critical_point_name:
            raise TransitionConfigurationError("LINE metadata lack a critical point")
        return cls(
            muB=data.field("muB"),
            temperature=data.field("T"),
            sidecar_fingerprint=data.fingerprint,
            critical_endpoint=data.critical_endpoint,
            critical_point_name=spec.critical_point_name,
            kind=spec.kind,
        )

    @property
    def representation(self) -> FirstOrderRepresentation:
        return FirstOrderRepresentation.LINE

    @property
    def muB_bounds(self) -> Tuple[float, float]:
        return self.diagnostics.muB_bounds

    @property
    def critical_index(self) -> int:
        return 0 if self.critical_endpoint == "first" else self.muB.size - 1

    @property
    def critical_point(self) -> Mapping[str, float]:
        index = self.critical_index
        return MappingProxyType(
            {"T": float(self.temperature[index]), "muB": float(self.muB[index])}
        )

    @property
    def first_order_muB_direction(self) -> str:
        return "increasing" if self.critical_endpoint == "first" else "decreasing"

    def contains_muB(self, muB: Any) -> Any:
        requested, scalar = _finite_query(muB, "muB")
        contained = (requested >= self.muB[0]) & (requested <= self.muB[-1])
        return _restore_boolean(contained, scalar)

    def evaluate(self, muB: Any) -> Any:
        """Return physical ``T(muB)`` without endpoint extrapolation."""

        return _bounded_interpolate(self.muB, self.temperature, muB)

    temperature_at = evaluate

    def contains(
        self,
        temperature: Any,
        muB: Any,
        *,
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 0.0,
    ) -> Any:
        """Test closed source-space membership on the coexistence line."""

        supplied_temperature, _ = _finite_query(temperature, "temperature")
        supplied_muB, _ = _finite_query(muB, "muB")
        try:
            temperatures, chemical_potentials = np.broadcast_arrays(
                supplied_temperature, supplied_muB
            )
        except ValueError as exc:
            raise TransitionInputError(
                "temperature and muB queries must be broadcast-compatible"
            ) from exc
        scalar = temperatures.ndim == 0
        absolute = _finite_nonnegative(absolute_tolerance, "absolute_tolerance")
        relative = _finite_nonnegative(relative_tolerance, "relative_tolerance")
        in_muB = (chemical_potentials >= self.muB[0]) & (
            chemical_potentials <= self.muB[-1]
        )
        result = np.zeros(temperatures.shape, dtype=bool)
        if np.any(in_muB):
            expected = np.interp(
                chemical_potentials[in_muB], self.muB, self.temperature
            )
            result[in_muB] = np.isclose(
                temperatures[in_muB],
                expected,
                rtol=relative,
                atol=absolute,
            )
        return _restore_boolean(result, scalar)

    def contains_points(
        self,
        points: Any,
        *,
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 0.0,
    ) -> Any:
        """Test points whose last axis is canonical ``(T, muB)``."""

        try:
            array = np.asarray(points, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TransitionInputError("points must be finite and numeric") from exc
        if array.ndim == 0 or array.shape[-1] != 2:
            raise TransitionInputError("points must have shape (..., 2) for (T, muB)")
        if not np.all(np.isfinite(array)):
            raise TransitionInputError("points must contain only finite values")
        return self.contains(
            array[..., 0],
            array[..., 1],
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )


@dataclass(frozen=True)
class BinodalBand:
    """Bounded physical mixed-phase band between two binodal curves."""

    muB: np.ndarray
    dilute_temperature: np.ndarray
    dense_temperature: np.ndarray
    sidecar_fingerprint: str
    kind: str
    phase_fraction_field: Optional[str] = None
    mixed_phase_condition: Optional[str] = None
    phase_label: PhaseLabel = PhaseLabel.MIXED_PHASE
    diagnostics: TransitionDiagnostics = field(init=False)

    def __post_init__(self) -> None:
        muB = _finite_vector(self.muB, "muB")
        dilute = _finite_vector(self.dilute_temperature, "dilute_temperature")
        dense = _finite_vector(self.dense_temperature, "dense_temperature")
        if dilute.shape != muB.shape or dense.shape != muB.shape:
            raise TransitionConfigurationError(
                "binodal temperature and muB arrays must have equal shape"
            )
        _strictly_increasing(muB, "muB")
        for name, value in (
            ("sidecar_fingerprint", self.sidecar_fingerprint),
            ("kind", self.kind),
        ):
            if not isinstance(value, str) or not value:
                raise TransitionConfigurationError("%s must be non-empty" % name)
        if self.phase_fraction_field is not None and (
            not isinstance(self.phase_fraction_field, str)
            or not self.phase_fraction_field
        ):
            raise TransitionConfigurationError(
                "phase_fraction_field must be a non-empty string or None"
            )
        if self.mixed_phase_condition is not None and (
            not isinstance(self.mixed_phase_condition, str)
            or not self.mixed_phase_condition
        ):
            raise TransitionConfigurationError(
                "mixed_phase_condition must be a non-empty string or None"
            )
        if self.phase_label is not PhaseLabel.MIXED_PHASE:
            raise TransitionConfigurationError(
                "a binodal band must carry the MIXED_PHASE label"
            )
        immutable_muB = _immutable_array(muB, dtype=float)
        immutable_dilute = _immutable_array(dilute, dtype=float)
        immutable_dense = _immutable_array(dense, dtype=float)
        object.__setattr__(self, "muB", immutable_muB)
        object.__setattr__(self, "dilute_temperature", immutable_dilute)
        object.__setattr__(self, "dense_temperature", immutable_dense)
        lower = np.minimum(dilute, dense)
        upper = np.maximum(dilute, dense)
        object.__setattr__(
            self,
            "diagnostics",
            TransitionDiagnostics(
                representation=FirstOrderRepresentation.BINODALS,
                point_count=int(muB.size),
                muB_bounds=(float(muB[0]), float(muB[-1])),
                temperature_bounds={
                    "T_dilute": (float(np.min(dilute)), float(np.max(dilute))),
                    "T_dense": (float(np.min(dense)), float(np.max(dense))),
                    "T_lower": (float(np.min(lower)), float(np.max(lower))),
                    "T_upper": (float(np.min(upper)), float(np.max(upper))),
                },
                sidecar_fingerprint=self.sidecar_fingerprint,
            ),
        )

    @classmethod
    def from_validated_data(
        cls,
        spec: FirstOrderTransitionSpec,
        data: CoexistenceTable,
    ) -> "BinodalBand":
        if spec.representation != FirstOrderRepresentation.BINODALS.value:
            raise TransitionConfigurationError("metadata do not describe BINODALS")
        if data.representation != FirstOrderRepresentation.BINODALS.value:
            raise TransitionConfigurationError("sidecar data do not describe BINODALS")
        expected_names = ("muB", "T_dilute", "T_dense")
        if data.names != expected_names or data.units != ("GeV", "GeV", "GeV"):
            raise TransitionConfigurationError(
                "BINODALS sidecar must provide canonical "
                "(muB, T_dilute, T_dense) in GeV"
            )
        return cls(
            muB=data.field("muB"),
            dilute_temperature=data.field("T_dilute"),
            dense_temperature=data.field("T_dense"),
            sidecar_fingerprint=data.fingerprint,
            kind=spec.kind,
            phase_fraction_field=spec.phase_fraction_field,
            mixed_phase_condition=spec.mixed_phase_condition,
        )

    @property
    def representation(self) -> FirstOrderRepresentation:
        return FirstOrderRepresentation.BINODALS

    @property
    def muB_bounds(self) -> Tuple[float, float]:
        return self.diagnostics.muB_bounds

    def contains_muB(self, muB: Any) -> Any:
        requested, scalar = _finite_query(muB, "muB")
        contained = (requested >= self.muB[0]) & (requested <= self.muB[-1])
        return _restore_boolean(contained, scalar)

    def evaluate(self, muB: Any) -> np.ndarray:
        """Return ``[..., (T_dilute, T_dense)]`` without extrapolation."""

        dilute = _bounded_interpolate(
            self.muB, self.dilute_temperature, muB
        )
        dense = _bounded_interpolate(self.muB, self.dense_temperature, muB)
        if np.isscalar(dilute):
            return _immutable_array([dilute, dense], dtype=float)
        return _immutable_array(np.stack((dilute, dense), axis=-1), dtype=float)

    def temperature_bounds(self, muB: Any) -> np.ndarray:
        """Return sorted closed ``[..., (T_lower, T_upper)]`` bounds."""

        curves = self.evaluate(muB)
        return _immutable_array(
            np.stack(
                (
                    np.minimum(curves[..., 0], curves[..., 1]),
                    np.maximum(curves[..., 0], curves[..., 1]),
                ),
                axis=-1,
            ),
            dtype=float,
        )

    def contains(self, temperature: Any, muB: Any) -> Any:
        """Return exact inclusive mixed-band membership in physical space."""

        supplied_temperature, _ = _finite_query(temperature, "temperature")
        supplied_muB, _ = _finite_query(muB, "muB")
        try:
            temperatures, chemical_potentials = np.broadcast_arrays(
                supplied_temperature, supplied_muB
            )
        except ValueError as exc:
            raise TransitionInputError(
                "temperature and muB queries must be broadcast-compatible"
            ) from exc
        scalar = temperatures.ndim == 0
        in_muB = (chemical_potentials >= self.muB[0]) & (
            chemical_potentials <= self.muB[-1]
        )
        result = np.zeros(temperatures.shape, dtype=bool)
        if np.any(in_muB):
            valid_muB = chemical_potentials[in_muB]
            dilute = np.interp(valid_muB, self.muB, self.dilute_temperature)
            dense = np.interp(valid_muB, self.muB, self.dense_temperature)
            lower = np.minimum(dilute, dense)
            upper = np.maximum(dilute, dense)
            selected_temperature = temperatures[in_muB]
            result[in_muB] = (selected_temperature >= lower) & (
                selected_temperature <= upper
            )
        return _restore_boolean(result, scalar)

    def contains_points(self, points: Any) -> Any:
        """Test points whose last axis is canonical ``(T, muB)``."""

        try:
            array = np.asarray(points, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TransitionInputError("points must be finite and numeric") from exc
        if array.ndim == 0 or array.shape[-1] != 2:
            raise TransitionInputError("points must have shape (..., 2) for (T, muB)")
        if not np.all(np.isfinite(array)):
            raise TransitionInputError("points must contain only finite values")
        return self.contains(array[..., 0], array[..., 1])


TransitionModel = Union[TransitionLine, BinodalBand]


def transition_model_from_table(table: ForwardTable) -> Optional[TransitionModel]:
    """Build the standalone physical transition model for a validated table.

    A table with no enabled computational first-order transition returns
    ``None``.  The function performs no file I/O: the immutable canonical
    sidecar snapshot must already have been loaded by ``ForwardTable.validate``.
    """

    if not isinstance(table, ForwardTable):
        raise TypeError("table must be a ForwardTable")
    table.require_validated()
    spec = table.first_order_transition
    if spec is None:
        return None
    data = table.coexistence_table
    if data is None:
        raise TransitionConfigurationError(
            "validated first-order metadata lack canonical coexistence data"
        )
    if spec.representation == FirstOrderRepresentation.LINE.value:
        return TransitionLine.from_validated_data(spec, data)
    if spec.representation == FirstOrderRepresentation.BINODALS.value:
        return BinodalBand.from_validated_data(spec, data)
    raise TransitionConfigurationError(
        "unsupported first-order representation: %s" % spec.representation
    )


def _configuration_number(
    value: Any,
    name: str,
    *,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TransitionConfigurationError("%s must be a finite number" % name)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TransitionConfigurationError(
            "%s must be a finite number" % name
        ) from exc
    valid = converted > 0.0 if strictly_positive else converted >= 0.0
    if not math.isfinite(converted) or not valid:
        qualifier = "positive" if strictly_positive else "non-negative"
        raise TransitionConfigurationError(
            "%s must be finite and %s" % (name, qualifier)
        )
    return converted


@dataclass(frozen=True)
class RegionSConfig:
    """Resolved physical-side fitting policy for LINE RegionS construction.

    The defaults reproduce the per-slice part of the legacy ``FindVals.py``
    algorithm.  They deliberately do not include its later cross-slice
    coordinate transformation and smoothing stage.  The declared critical
    endpoint is not fitted from either side: a local numerical closure in
    canonical physical target space brings both phase endpoints to one
    linearly interpolated ordinary-table state.
    """

    hadronic_temperature_offset: float = 0.002
    qgp_temperature_offset: float = 0.002
    qgp_energy_gaussian_sigma: float = 0.0008
    gaussian_truncate: float = 4.0
    spline_degree: int = 3
    hadronic_energy_smoothing: float = 0.0
    hadronic_density_smoothing: float = 0.0
    qgp_energy_smoothing: float = 0.01
    qgp_density_smoothing: float = 0.0
    critical_closure_enabled: bool = True
    critical_closure_width: float = 0.005
    critical_closure_max_slice_expansions: int = 32
    minimum_target_segment_norm: float = 0.0
    uniformity_absolute_tolerance: float = 1.0e-14
    uniformity_relative_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if not isinstance(self.critical_closure_enabled, (bool, np.bool_)):
            raise TransitionConfigurationError(
                "critical_closure_enabled must be a boolean"
            )
        object.__setattr__(
            self,
            "critical_closure_enabled",
            bool(self.critical_closure_enabled),
        )
        positive = (
            "hadronic_temperature_offset",
            "qgp_temperature_offset",
            "gaussian_truncate",
            "critical_closure_width",
        )
        nonnegative = (
            "qgp_energy_gaussian_sigma",
            "hadronic_energy_smoothing",
            "hadronic_density_smoothing",
            "qgp_energy_smoothing",
            "qgp_density_smoothing",
            "minimum_target_segment_norm",
            "uniformity_absolute_tolerance",
            "uniformity_relative_tolerance",
        )
        for name in positive:
            object.__setattr__(
                self,
                name,
                _configuration_number(
                    getattr(self, name), name, strictly_positive=True
                ),
            )
        for name in nonnegative:
            object.__setattr__(
                self,
                name,
                _configuration_number(getattr(self, name), name),
            )
        degree = self.spline_degree
        if isinstance(degree, (bool, np.bool_)) or not isinstance(
            degree, (int, np.integer)
        ):
            raise TransitionConfigurationError(
                "spline_degree must be an integer from 1 through 5"
            )
        degree = int(degree)
        if degree < 1 or degree > 5:
            raise TransitionConfigurationError(
                "spline_degree must be an integer from 1 through 5"
            )
        object.__setattr__(self, "spline_degree", degree)
        expansions = self.critical_closure_max_slice_expansions
        if isinstance(expansions, (bool, np.bool_)) or not isinstance(
            expansions, (int, np.integer)
        ):
            raise TransitionConfigurationError(
                "critical_closure_max_slice_expansions must be a "
                "non-negative integer"
            )
        expansions = int(expansions)
        if expansions < 0:
            raise TransitionConfigurationError(
                "critical_closure_max_slice_expansions must be a "
                "non-negative integer"
            )
        object.__setattr__(
            self, "critical_closure_max_slice_expansions", expansions
        )

    @property
    def minimum_side_points(self) -> int:
        return self.spline_degree + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": _REGION_S_ALGORITHM,
            "hadronic_temperature_offset_GeV": self.hadronic_temperature_offset,
            "qgp_temperature_offset_GeV": self.qgp_temperature_offset,
            "qgp_energy_gaussian_sigma_GeV": self.qgp_energy_gaussian_sigma,
            "gaussian_mode": "nearest",
            "gaussian_truncate": self.gaussian_truncate,
            "spline_degree": self.spline_degree,
            "minimum_side_points": self.minimum_side_points,
            "hadronic_energy_smoothing": self.hadronic_energy_smoothing,
            "hadronic_density_smoothing": self.hadronic_density_smoothing,
            "qgp_energy_smoothing": self.qgp_energy_smoothing,
            "qgp_density_smoothing": self.qgp_density_smoothing,
            "critical_closure_enabled": self.critical_closure_enabled,
            "critical_closure_requested_width_GeV": (
                self.critical_closure_width
            ),
            "critical_closure_max_slice_expansions": (
                self.critical_closure_max_slice_expansions
            ),
            "minimum_target_segment_norm": self.minimum_target_segment_norm,
            "uniformity_absolute_tolerance": self.uniformity_absolute_tolerance,
            "uniformity_relative_tolerance": self.uniformity_relative_tolerance,
            "energy_fit_quantity": "e/T^4",
            "density_fit_quantity": "nB/T^3",
            "endpoint_space": "canonical_physical",
            "cross_slice_smoothing": "none",
            "critical_endpoint_policy": (
                _CRITICAL_ENDPOINT_POLICY
                if self.critical_closure_enabled
                else _UNMERGED_ENDPOINT_POLICY
            ),
            "critical_closure_space": "canonical_physical_e_nB",
            "critical_closure_weight_function": _CRITICAL_CLOSURE_WEIGHT,
            "critical_closure_asymptotic_model": False,
        }


@dataclass(frozen=True)
class RegionSEndpoints:
    """Corrected physical hadronic/QGP endpoint curves for a LINE."""

    temperature: np.ndarray
    muB: np.ndarray
    e_hadronic: np.ndarray
    nB_hadronic: np.ndarray
    e_qgp: np.ndarray
    nB_qgp: np.ndarray
    table_fingerprint: str
    sidecar_fingerprint: str

    def __post_init__(self) -> None:
        names = (
            "temperature",
            "muB",
            "e_hadronic",
            "nB_hadronic",
            "e_qgp",
            "nB_qgp",
        )
        arrays = {name: _finite_vector(getattr(self, name), name) for name in names}
        shape = arrays["muB"].shape
        if any(array.shape != shape for array in arrays.values()):
            raise TransitionConfigurationError(
                "all RegionS endpoint arrays must have equal shape"
            )
        _strictly_increasing(arrays["muB"], "RegionS muB")
        if np.any(arrays["temperature"] <= 0.0):
            raise TransitionConfigurationError(
                "RegionS temperatures must be strictly positive"
            )
        for name in ("table_fingerprint", "sidecar_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TransitionConfigurationError("%s must be non-empty" % name)
        for name, array in arrays.items():
            object.__setattr__(self, name, _immutable_array(array, dtype=float))

    @property
    def point_count(self) -> int:
        return int(self.muB.size)

    @property
    def muB_bounds(self) -> Tuple[float, float]:
        return float(self.muB[0]), float(self.muB[-1])

    @property
    def hadronic_targets(self) -> np.ndarray:
        return _immutable_array(
            np.column_stack((self.e_hadronic, self.nB_hadronic)), dtype=float
        )

    @property
    def qgp_targets(self) -> np.ndarray:
        return _immutable_array(
            np.column_stack((self.e_qgp, self.nB_qgp)), dtype=float
        )

    @property
    def target_segments(self) -> np.ndarray:
        return _immutable_array(
            np.stack((self.hadronic_targets, self.qgp_targets), axis=1),
            dtype=float,
        )

    @property
    def target_deltas(self) -> np.ndarray:
        return _immutable_array(self.qgp_targets - self.hadronic_targets)

    @property
    def target_segment_norms(self) -> np.ndarray:
        delta = self.target_deltas
        return _immutable_array(np.hypot(delta[:, 0], delta[:, 1]))

    def as_array(self) -> np.ndarray:
        """Return legacy-compatible physical columns without writing a file."""

        return _immutable_array(
            np.column_stack(
                (
                    self.temperature,
                    self.muB,
                    self.e_hadronic,
                    self.nB_hadronic,
                    self.e_qgp,
                    self.nB_qgp,
                )
            ),
            dtype=float,
        )


@dataclass(frozen=True)
class RegionSAuxiliaryEndpoints:
    """Generic physical auxiliary endpoint curves for a reconstructed LINE."""

    field_names: Tuple[str, ...]
    field_units: Tuple[str, ...]
    temperature_powers: Tuple[int, ...]
    muB: np.ndarray
    hadronic_values: np.ndarray
    qgp_values: np.ndarray
    table_fingerprint: str
    sidecar_fingerprint: str

    def __post_init__(self) -> None:
        names = tuple(self.field_names)
        units = tuple(self.field_units)
        powers = tuple(self.temperature_powers)
        if any(not isinstance(name, str) or not name for name in names):
            raise TransitionConfigurationError(
                "auxiliary endpoint field names must be non-empty strings"
            )
        if len(set(names)) != len(names):
            raise TransitionConfigurationError(
                "auxiliary endpoint field names cannot contain duplicates"
            )
        if len(units) != len(names) or any(
            not isinstance(unit, str) or not unit for unit in units
        ):
            raise TransitionConfigurationError(
                "auxiliary endpoint units must align with field names"
            )
        if len(powers) != len(names) or any(
            isinstance(power, (bool, np.bool_))
            or not isinstance(power, (int, np.integer))
            for power in powers
        ):
            raise TransitionConfigurationError(
                "auxiliary endpoint temperature powers must be integers"
            )
        chemical_potential = _finite_vector(self.muB, "auxiliary endpoint muB")
        expected_shape = (chemical_potential.size, len(names))
        arrays = {}
        for label in ("hadronic_values", "qgp_values"):
            try:
                values = np.asarray(getattr(self, label), dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TransitionConfigurationError(
                    "%s must be finite numeric endpoint values" % label
                ) from exc
            if values.shape != expected_shape or not np.all(np.isfinite(values)):
                raise TransitionConfigurationError(
                    "%s must be finite with shape %r" % (label, expected_shape)
                )
            arrays[label] = values
        _strictly_increasing(chemical_potential, "auxiliary endpoint muB")
        for label in ("table_fingerprint", "sidecar_fingerprint"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise TransitionConfigurationError("%s must be non-empty" % label)
        object.__setattr__(self, "field_names", names)
        object.__setattr__(self, "field_units", units)
        object.__setattr__(self, "temperature_powers", tuple(int(p) for p in powers))
        object.__setattr__(self, "muB", _immutable_array(chemical_potential))
        object.__setattr__(
            self, "hadronic_values", _immutable_array(arrays["hadronic_values"])
        )
        object.__setattr__(self, "qgp_values", _immutable_array(arrays["qgp_values"]))

    @property
    def field_count(self) -> int:
        return len(self.field_names)

    def evaluate_endpoints(self, muB: Any) -> np.ndarray:
        """Return piecewise-linear ``[..., (H/Q), field]`` values."""

        chemical_potential, scalar = _finite_query(muB, "muB")
        outside = (chemical_potential < self.muB[0]) | (
            chemical_potential > self.muB[-1]
        )
        if np.any(outside):
            offending = float(chemical_potential[outside].reshape(-1)[0])
            raise TransitionDomainError(
                "muB=%r lies outside the closed auxiliary endpoint domain "
                "[%r, %r]" % (offending, self.muB[0], self.muB[-1])
            )
        flat = chemical_potential.reshape(-1)
        values = np.empty((flat.size, 2, self.field_count), dtype=float)
        for index in range(self.field_count):
            values[:, 0, index] = np.interp(
                flat, self.muB, self.hadronic_values[:, index]
            )
            values[:, 1, index] = np.interp(
                flat, self.muB, self.qgp_values[:, index]
            )
        shape = (
            (2, self.field_count)
            if scalar
            else chemical_potential.shape + (2, self.field_count)
        )
        return _immutable_array(values.reshape(shape))

    def evaluate(self, muB: Any, mixture_fraction: Any) -> np.ndarray:
        """Apply the LINE lever rule using the physical H/Q orientation."""

        chemical_potential, _ = _finite_query(muB, "muB")
        fraction, _ = _finite_query(mixture_fraction, "mixture_fraction")
        try:
            chemical_potential, fraction = np.broadcast_arrays(
                chemical_potential, fraction
            )
        except ValueError as exc:
            raise TransitionInputError(
                "muB and mixture_fraction queries must be broadcast-compatible"
            ) from exc
        outside_fraction = (fraction < 0.0) | (fraction > 1.0)
        if np.any(outside_fraction):
            offending = float(fraction[outside_fraction].reshape(-1)[0])
            raise TransitionDomainError(
                "mixture_fraction=%r lies outside the closed interval [0, 1]"
                % offending
            )
        endpoints = self.evaluate_endpoints(chemical_potential)
        values = (
            (1.0 - fraction)[..., np.newaxis] * endpoints[..., 0, :]
            + fraction[..., np.newaxis] * endpoints[..., 1, :]
        )
        shape = (
            (self.field_count,)
            if chemical_potential.ndim == 0
            else chemical_potential.shape + (self.field_count,)
        )
        return _immutable_array(values.reshape(shape))


@dataclass(frozen=True)
class RegionSReconstructionDiagnostics:
    """Auditable physical fitting statistics for a completed RegionS strip."""

    algorithm: str
    table_fingerprint: str
    sidecar_fingerprint: str
    endpoint_fingerprint: str
    config: Mapping[str, Any]
    candidate_slice_count: int
    reconstructed_slice_count: int
    muB_bounds: Tuple[float, float]
    temperature_bounds: Tuple[float, float]
    hadronic_sample_count_bounds: Tuple[int, int]
    qgp_sample_count_bounds: Tuple[int, int]
    target_segment_norm_bounds: Tuple[float, float]
    energy_orientation_counts: Mapping[str, int]
    density_orientation_counts: Mapping[str, int]
    rejected_slices: Tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.algorithm != _REGION_S_ALGORITHM:
            raise ValueError("unknown RegionS reconstruction algorithm")
        for name in (
            "table_fingerprint",
            "sidecar_fingerprint",
            "endpoint_fingerprint",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError("%s must be non-empty" % name)
        if self.candidate_slice_count < 2 or self.reconstructed_slice_count < 2:
            raise ValueError("RegionS diagnostics require at least two slices")
        if self.reconstructed_slice_count > self.candidate_slice_count:
            raise ValueError("reconstructed slices cannot exceed candidates")
        numeric_bounds = (
            ("muB_bounds", self.muB_bounds, True),
            ("temperature_bounds", self.temperature_bounds, False),
            ("target_segment_norm_bounds", self.target_segment_norm_bounds, False),
        )
        for name, bounds, strict in numeric_bounds:
            if len(bounds) != 2:
                raise ValueError("%s must contain two bounds" % name)
            lower, upper = float(bounds[0]), float(bounds[1])
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError("%s must be finite" % name)
            if lower > upper or (strict and lower == upper):
                raise ValueError("%s are not ordered" % name)
        for name, bounds in (
            ("hadronic_sample_count_bounds", self.hadronic_sample_count_bounds),
            ("qgp_sample_count_bounds", self.qgp_sample_count_bounds),
        ):
            if len(bounds) != 2 or bounds[0] <= 0 or bounds[0] > bounds[1]:
                raise ValueError("%s are not positive ordered counts" % name)
        orientation_keys = {"qgp_greater", "equal", "hadronic_greater"}
        for name, counts in (
            ("energy_orientation_counts", self.energy_orientation_counts),
            ("density_orientation_counts", self.density_orientation_counts),
        ):
            if set(counts) != orientation_keys:
                raise ValueError("%s have the wrong labels" % name)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
                for value in counts.values()
            ):
                raise ValueError("%s must be non-negative integer counts" % name)
            if (
                sum(int(value) for value in counts.values())
                != self.reconstructed_slice_count
            ):
                raise ValueError("%s do not cover every endpoint" % name)
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))
        object.__setattr__(
            self,
            "energy_orientation_counts",
            MappingProxyType(dict(self.energy_orientation_counts)),
        )
        object.__setattr__(
            self,
            "density_orientation_counts",
            MappingProxyType(dict(self.density_orientation_counts)),
        )
        object.__setattr__(
            self,
            "rejected_slices",
            tuple(MappingProxyType(dict(item)) for item in self.rejected_slices),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "table_fingerprint": self.table_fingerprint,
            "sidecar_fingerprint": self.sidecar_fingerprint,
            "endpoint_fingerprint": self.endpoint_fingerprint,
            "config": dict(self.config),
            "candidate_slice_count": self.candidate_slice_count,
            "reconstructed_slice_count": self.reconstructed_slice_count,
            "muB_bounds": list(self.muB_bounds),
            "temperature_bounds": list(self.temperature_bounds),
            "hadronic_sample_count_bounds": list(
                self.hadronic_sample_count_bounds
            ),
            "qgp_sample_count_bounds": list(self.qgp_sample_count_bounds),
            "target_segment_norm_bounds": list(
                self.target_segment_norm_bounds
            ),
            "energy_orientation_counts": dict(self.energy_orientation_counts),
            "density_orientation_counts": dict(self.density_orientation_counts),
            "rejected_slices": [dict(item) for item in self.rejected_slices],
        }


@dataclass(frozen=True)
class MixedPhaseCellEnclosures:
    """Exact component-wise target boxes for physical RegionS cells."""

    muB_intervals: np.ndarray
    corners: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    target_names: Tuple[str, str] = ("e", "nB")
    rigorous: bool = True

    def __post_init__(self) -> None:
        intervals = np.asarray(self.muB_intervals, dtype=float)
        corners = np.asarray(self.corners, dtype=float)
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if intervals.ndim != 2 or intervals.shape[1] != 2:
            raise ValueError("muB_intervals must have shape (cell_count, 2)")
        cell_count = intervals.shape[0]
        if corners.shape != (cell_count, 4, 2):
            raise ValueError("corners must have shape (cell_count, 4, 2)")
        if lower.shape != (cell_count, 2) or upper.shape != (cell_count, 2):
            raise ValueError("cell bounds must have shape (cell_count, 2)")
        if not all(
            np.all(np.isfinite(item))
            for item in (intervals, corners, lower, upper)
        ):
            raise ValueError("RegionS cell enclosures must be finite")
        if np.any(intervals[:, 0] >= intervals[:, 1]):
            raise ValueError("RegionS cell intervals must be strictly increasing")
        if np.any(lower > upper):
            raise ValueError("RegionS cell lower bounds cannot exceed upper bounds")
        if tuple(self.target_names) != ("e", "nB"):
            raise ValueError("RegionS cell targets must be canonical (e, nB)")
        if not self.rigorous:
            raise ValueError("RegionS corner enclosures must be rigorous")
        object.__setattr__(self, "muB_intervals", _immutable_array(intervals))
        object.__setattr__(self, "corners", _immutable_array(corners))
        object.__setattr__(self, "lower", _immutable_array(lower))
        object.__setattr__(self, "upper", _immutable_array(upper))
        object.__setattr__(self, "target_names", ("e", "nB"))

    @property
    def cell_count(self) -> int:
        return int(self.muB_intervals.shape[0])


@dataclass(frozen=True)
class MixedPhaseStrip:
    """Physical bilinear LINE strip parameterized by ``(muB, lambda)``."""

    line: TransitionLine
    endpoints: RegionSEndpoints
    diagnostics: RegionSReconstructionDiagnostics
    auxiliary_endpoints: Optional[RegionSAuxiliaryEndpoints] = None

    def __post_init__(self) -> None:
        if not isinstance(self.line, TransitionLine):
            raise TypeError("line must be a TransitionLine")
        if not isinstance(self.endpoints, RegionSEndpoints):
            raise TypeError("endpoints must be RegionSEndpoints")
        if not isinstance(self.diagnostics, RegionSReconstructionDiagnostics):
            raise TypeError(
                "diagnostics must be RegionSReconstructionDiagnostics"
            )
        if self.auxiliary_endpoints is not None and not isinstance(
            self.auxiliary_endpoints, RegionSAuxiliaryEndpoints
        ):
            raise TypeError(
                "auxiliary_endpoints must be RegionSAuxiliaryEndpoints or None"
            )
        expected_temperature = np.asarray(self.line.evaluate(self.endpoints.muB))
        if not np.array_equal(expected_temperature, self.endpoints.temperature):
            raise TransitionConfigurationError(
                "RegionS endpoint temperatures do not lie on the transition line"
            )
        if self.line.sidecar_fingerprint != self.endpoints.sidecar_fingerprint:
            raise TransitionConfigurationError(
                "RegionS endpoints and line have different sidecar fingerprints"
            )
        if self.auxiliary_endpoints is not None:
            auxiliary = self.auxiliary_endpoints
            if not np.array_equal(auxiliary.muB, self.endpoints.muB):
                raise TransitionConfigurationError(
                    "target and auxiliary RegionS endpoints use different muB nodes"
                )
            if (
                auxiliary.table_fingerprint != self.endpoints.table_fingerprint
                or auxiliary.sidecar_fingerprint
                != self.endpoints.sidecar_fingerprint
            ):
                raise TransitionConfigurationError(
                    "target and auxiliary RegionS endpoints describe different snapshots"
                )
        if (
            self.diagnostics.table_fingerprint
            != self.endpoints.table_fingerprint
            or self.diagnostics.sidecar_fingerprint
            != self.endpoints.sidecar_fingerprint
            or self.diagnostics.reconstructed_slice_count
            != self.endpoints.point_count
        ):
            raise TransitionConfigurationError(
                "RegionS diagnostics and endpoints describe different snapshots"
            )
        if self.diagnostics.endpoint_fingerprint != _endpoint_fingerprint(
            self.endpoints,
            self.diagnostics.config,
            self.auxiliary_endpoints,
        ):
            raise TransitionConfigurationError(
                "RegionS diagnostics do not identify the supplied endpoints"
            )

    @property
    def auxiliary_names(self) -> Tuple[str, ...]:
        return (
            ()
            if self.auxiliary_endpoints is None
            else self.auxiliary_endpoints.field_names
        )

    def evaluate_auxiliaries(
        self, muB: Any, mixture_fraction: Any
    ) -> np.ndarray:
        """Evaluate every reconstructed LINE auxiliary with the lever rule."""

        if self.auxiliary_endpoints is None:
            raise TransitionConfigurationError(
                "this RegionS strip has no reconstructed auxiliary endpoints"
            )
        return self.auxiliary_endpoints.evaluate(muB, mixture_fraction)

    @property
    def coordinate_names(self) -> Tuple[str, str]:
        return "muB", "lambda"

    @property
    def target_names(self) -> Tuple[str, str]:
        return "e", "nB"

    @property
    def axes(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.endpoints.muB, _immutable_array([0.0, 1.0])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.endpoints.point_count, 2

    @property
    def cell_shape(self) -> Tuple[int, int]:
        return self.endpoints.point_count - 1, 1

    @property
    def muB_bounds(self) -> Tuple[float, float]:
        return self.endpoints.muB_bounds

    def to_interpolator(
        self,
        *,
        boundary_absolute_tolerance: float = 0.0,
        boundary_relative_tolerance: float = 0.0,
        chunk_size: int = 65_536,
    ) -> "MultilinearInterpolator":
        """Return an immutable solver-ready ``(muB, lambda)`` target map.

        The map is built directly from the corrected physical endpoint nodes.
        It therefore evaluates the same bilinear function as :meth:`evaluate`
        while exposing the shared Jacobian, cell-corner, and rigorous
        enclosure APIs used by the ordinary-table solver and reachability
        code.  No tilde coordinate enters this conversion.
        """

        # Local import keeps the standalone transition metadata/evaluator
        # types usable without creating an interpolation import cycle.
        from .interpolation import MultilinearInterpolator

        segments = self.endpoints.target_segments
        return MultilinearInterpolator.from_rectilinear_map(
            self.axes,
            {
                "e": segments[:, :, 0],
                "nB": segments[:, :, 1],
            },
            coordinate_names=self.coordinate_names,
            target_names=self.target_names,
            coordinate_units=("GeV", "1"),
            target_units=("GeV^4", "GeV^3"),
            boundary_absolute_tolerance=boundary_absolute_tolerance,
            boundary_relative_tolerance=boundary_relative_tolerance,
            chunk_size=chunk_size,
        )

    def _queries(
        self, muB: Any, mixture_fraction: Any
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        chemical_potential, _ = _finite_query(muB, "muB")
        fraction, _ = _finite_query(mixture_fraction, "mixture_fraction")
        try:
            chemical_potential, fraction = np.broadcast_arrays(
                chemical_potential, fraction
            )
        except ValueError as exc:
            raise TransitionInputError(
                "muB and mixture_fraction queries must be broadcast-compatible"
            ) from exc
        outside_muB = (chemical_potential < self.endpoints.muB[0]) | (
            chemical_potential > self.endpoints.muB[-1]
        )
        if np.any(outside_muB):
            offending = float(chemical_potential[outside_muB].reshape(-1)[0])
            raise TransitionDomainError(
                "muB=%r lies outside the closed RegionS domain [%r, %r]"
                % (offending, self.endpoints.muB[0], self.endpoints.muB[-1])
            )
        outside_fraction = (fraction < 0.0) | (fraction > 1.0)
        if np.any(outside_fraction):
            offending = float(fraction[outside_fraction].reshape(-1)[0])
            raise TransitionDomainError(
                "mixture_fraction=%r lies outside the closed interval [0, 1]"
                % offending
            )
        return chemical_potential, fraction, chemical_potential.ndim == 0

    def evaluate_endpoints(self, muB: Any) -> np.ndarray:
        """Return piecewise-linear ``[..., (H/Q), (e/nB)]`` endpoints."""

        chemical_potential, scalar = _finite_query(muB, "muB")
        outside = (chemical_potential < self.endpoints.muB[0]) | (
            chemical_potential > self.endpoints.muB[-1]
        )
        if np.any(outside):
            offending = float(chemical_potential[outside].reshape(-1)[0])
            raise TransitionDomainError(
                "muB=%r lies outside the closed RegionS domain [%r, %r]"
                % (offending, self.endpoints.muB[0], self.endpoints.muB[-1])
            )
        flat = chemical_potential.reshape(-1)
        values = np.empty((flat.size, 2, 2), dtype=float)
        curves = (
            self.endpoints.e_hadronic,
            self.endpoints.nB_hadronic,
            self.endpoints.e_qgp,
            self.endpoints.nB_qgp,
        )
        values[:, 0, 0] = np.interp(flat, self.endpoints.muB, curves[0])
        values[:, 0, 1] = np.interp(flat, self.endpoints.muB, curves[1])
        values[:, 1, 0] = np.interp(flat, self.endpoints.muB, curves[2])
        values[:, 1, 1] = np.interp(flat, self.endpoints.muB, curves[3])
        shape = (2, 2) if scalar else chemical_potential.shape + (2, 2)
        return _immutable_array(values.reshape(shape))

    def evaluate(self, muB: Any, mixture_fraction: Any) -> np.ndarray:
        """Evaluate canonical ``(e, nB)`` on the closed physical strip."""

        chemical_potential, fraction, scalar = self._queries(
            muB, mixture_fraction
        )
        endpoints = self.evaluate_endpoints(chemical_potential)
        result = (
            (1.0 - fraction)[..., np.newaxis] * endpoints[..., 0, :]
            + fraction[..., np.newaxis] * endpoints[..., 1, :]
        )
        shape = (2,) if scalar else chemical_potential.shape + (2,)
        return _immutable_array(result.reshape(shape))

    def source_coordinates(
        self, muB: Any, mixture_fraction: Any = 0.0
    ) -> np.ndarray:
        """Return physical coordinates, which do not vary along a segment."""

        chemical_potential, _, scalar = self._queries(muB, mixture_fraction)
        temperature = np.asarray(self.line.evaluate(chemical_potential))
        result = np.stack((temperature, chemical_potential), axis=-1)
        shape = (2,) if scalar else chemical_potential.shape + (2,)
        return _immutable_array(result.reshape(shape))

    def cell_target_enclosures(self) -> MixedPhaseCellEnclosures:
        """Return exact corner-extrema boxes for every ``muB`` strip cell."""

        segments = self.endpoints.target_segments
        corners = np.stack(
            (segments[:-1, 0], segments[:-1, 1], segments[1:, 0], segments[1:, 1]),
            axis=1,
        )
        lower = np.min(corners, axis=1)
        upper = np.max(corners, axis=1)
        intervals = np.column_stack(
            (self.endpoints.muB[:-1], self.endpoints.muB[1:])
        )
        return MixedPhaseCellEnclosures(
            muB_intervals=intervals,
            corners=corners,
            lower=lower,
            upper=upper,
        )


def _endpoint_fingerprint(
    endpoints: RegionSEndpoints,
    config: Mapping[str, Any],
    auxiliary_endpoints: Optional[RegionSAuxiliaryEndpoints] = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"EoSInverter:PhysicalRegionSEndpoints:v3\0")
    descriptor = json.dumps(
        {
            "table_fingerprint": endpoints.table_fingerprint,
            "sidecar_fingerprint": endpoints.sidecar_fingerprint,
            "config": dict(config),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    for array in (
        endpoints.temperature,
        endpoints.muB,
        endpoints.e_hadronic,
        endpoints.nB_hadronic,
        endpoints.e_qgp,
        endpoints.nB_qgp,
    ):
        contiguous = np.ascontiguousarray(array, dtype="<f8")
        digest.update(contiguous.size.to_bytes(8, "big"))
        digest.update(contiguous.tobytes(order="C"))
    if auxiliary_endpoints is not None:
        auxiliary_descriptor = json.dumps(
            {
                "field_names": auxiliary_endpoints.field_names,
                "field_units": auxiliary_endpoints.field_units,
                "temperature_powers": auxiliary_endpoints.temperature_powers,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(auxiliary_descriptor).to_bytes(8, "big"))
        digest.update(auxiliary_descriptor)
        for array in (
            auxiliary_endpoints.hadronic_values,
            auxiliary_endpoints.qgp_values,
        ):
            contiguous = np.ascontiguousarray(array, dtype="<f8")
            digest.update(contiguous.size.to_bytes(8, "big"))
            digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _orientation_counts(delta: np.ndarray) -> Mapping[str, int]:
    return {
        "qgp_greater": int(np.count_nonzero(delta > 0.0)),
        "equal": int(np.count_nonzero(delta == 0.0)),
        "hadronic_greater": int(np.count_nonzero(delta < 0.0)),
    }


class RegionSBuilder:
    """Reconstruct a physical mixed-phase strip from a validated LINE table."""

    def __init__(
        self,
        table: ForwardTable,
        config: Optional[RegionSConfig] = None,
        *,
        auxiliary_fields: Sequence[str] = (),
    ) -> None:
        if not isinstance(table, ForwardTable):
            raise TypeError("table must be a ForwardTable")
        table.require_validated()
        resolved = RegionSConfig() if config is None else config
        if not isinstance(resolved, RegionSConfig):
            raise TypeError("config must be a RegionSConfig")
        model = transition_model_from_table(table)
        if not isinstance(model, TransitionLine):
            representation = "none" if model is None else model.representation.value
            raise TransitionConfigurationError(
                "RegionS construction requires a validated LINE transition; "
                "received %s" % representation
            )
        if table.coordinate_names != ("T", "muB") or table.target_names != (
            "e",
            "nB",
        ):
            raise TransitionConfigurationError(
                "RegionS construction requires canonical (T, muB) -> (e, nB)"
            )
        if isinstance(auxiliary_fields, (str, bytes)) or not isinstance(
            auxiliary_fields, Sequence
        ):
            raise TypeError("auxiliary_fields must be a sequence of field names")
        auxiliary_names = tuple(auxiliary_fields)
        if any(not isinstance(name, str) or not name for name in auxiliary_names):
            raise TransitionConfigurationError(
                "RegionS auxiliary field names must be non-empty strings"
            )
        if len(set(auxiliary_names)) != len(auxiliary_names):
            raise TransitionConfigurationError(
                "RegionS auxiliary fields cannot contain duplicates"
            )
        assert table.metadata is not None
        invalid_auxiliary = [
            name
            for name in auxiliary_names
            if name not in table.metadata.fields
            or table.metadata.fields[name].role != "auxiliary"
            or table.metadata.fields[name].interpolation != "linear"
        ]
        if invalid_auxiliary:
            raise TransitionConfigurationError(
                "RegionS auxiliary fields must be declared linear auxiliaries: %s"
                % ", ".join(invalid_auxiliary)
            )
        self._table = table
        self._line = model
        self._config = resolved
        self._auxiliary_fields = auxiliary_names

    @property
    def table(self) -> ForwardTable:
        return self._table

    @property
    def line(self) -> TransitionLine:
        return self._line

    @property
    def config(self) -> RegionSConfig:
        return self._config

    @property
    def auxiliary_fields(self) -> Tuple[str, ...]:
        return self._auxiliary_fields

    def _validate_temperature_axis(self, temperature: np.ndarray) -> float:
        if np.any(temperature <= 0.0):
            raise RegionSReconstructionError(
                "RegionS scaled endpoint fits require strictly positive T"
            )
        spacing = np.diff(temperature)
        if np.any(spacing <= 0.0) or not np.all(np.isfinite(spacing)):
            raise RegionSReconstructionError(
                "RegionS requires a finite strictly increasing T axis"
            )
        reference = float(spacing[0])
        if self.config.qgp_energy_gaussian_sigma > 0.0 and not np.allclose(
            spacing,
            reference,
            rtol=self.config.uniformity_relative_tolerance,
            atol=self.config.uniformity_absolute_tolerance,
        ):
            raise RegionSReconstructionError(
                "the configured index-space Gaussian reproduces the legacy "
                "physical-side fit only on a uniform T axis; set "
                "qgp_energy_gaussian_sigma=0 or preprocess the axis"
            )
        return reference

    def _fit(
        self,
        temperature: np.ndarray,
        values: np.ndarray,
        transition_temperature: float,
        smoothing: float,
        label: str,
    ) -> float:
        try:
            spline = UnivariateSpline(
                temperature,
                values,
                k=self.config.spline_degree,
                s=smoothing,
                ext=0,
                check_finite=True,
            )
            result = float(spline(transition_temperature))
        except (ValueError, FloatingPointError, OverflowError) as exc:
            raise RegionSReconstructionError(
                "%s endpoint fit failed: %s" % (label, exc)
            ) from exc
        if not math.isfinite(result):
            raise RegionSReconstructionError(
                "%s endpoint fit produced a nonfinite value" % label
            )
        return result

    @staticmethod
    def _cross_2d(left: np.ndarray, right: np.ndarray) -> float:
        return float(left[0] * right[1] - left[1] * right[0])

    @classmethod
    def _cell_geometry(
        cls,
        hadronic_start: np.ndarray,
        qgp_start: np.ndarray,
        hadronic_stop: np.ndarray,
        qgp_stop: np.ndarray,
    ) -> Tuple[Optional[int], Optional[str]]:
        """Return polygon orientation, rejecting a folded RegionS cell."""

        polygon = np.asarray(
            (hadronic_start, qgp_start, qgp_stop, hadronic_stop),
            dtype=float,
        )
        edges = np.roll(polygon, -1, axis=0) - polygon
        scale = max(float(np.max(np.linalg.norm(edges, axis=1))), 1.0e-300)
        tolerance = 64.0 * np.finfo(float).eps * scale * scale
        turns = np.asarray(
            [
                cls._cross_2d(edges[index], edges[(index + 1) % 4])
                for index in range(4)
            ]
        )
        nonzero_turns = turns[np.abs(turns) > tolerance]
        if nonzero_turns.size == 0:
            return None, "target cell has zero area"
        if np.any(nonzero_turns > 0.0) and np.any(nonzero_turns < 0.0):
            return None, "target cell is concave or folded"

        twice_area = sum(
            cls._cross_2d(polygon[index], polygon[(index + 1) % 4])
            for index in range(4)
        )
        if abs(twice_area) <= tolerance:
            return None, "target cell has zero area"
        orientation = 1 if twice_area > 0.0 else -1
        turn_orientation = 1 if nonzero_turns[0] > 0.0 else -1
        if orientation != turn_orientation:
            return None, "target cell polygon is self-intersecting"
        return orientation, None

    def _closure_geometry_error(
        self,
        raw_hadronic: np.ndarray,
        raw_qgp: np.ndarray,
        corrected_hadronic: np.ndarray,
        corrected_qgp: np.ndarray,
        selected_muB: np.ndarray,
        path_indices: np.ndarray,
        match_position: int,
        critical_state: np.ndarray,
    ) -> Optional[str]:
        critical_index = int(path_indices[0])
        match_index = int(path_indices[match_position])
        if not np.array_equal(corrected_hadronic[critical_index], critical_state):
            return "hadronic CEP is not the exact ordinary-table state"
        if not np.array_equal(corrected_qgp[critical_index], critical_state):
            return "QGP CEP is not the exact ordinary-table state"
        if not np.array_equal(
            corrected_hadronic[match_index], raw_hadronic[match_index]
        ) or not np.array_equal(corrected_qgp[match_index], raw_qgp[match_index]):
            return "outer matching slice does not recover the raw endpoints"

        closure_indices = path_indices[: match_position + 1]
        raw_delta = raw_qgp[closure_indices] - raw_hadronic[closure_indices]
        corrected_delta = (
            corrected_qgp[closure_indices] - corrected_hadronic[closure_indices]
        )
        raw_norm = np.linalg.norm(raw_delta, axis=1)
        corrected_norm = np.linalg.norm(corrected_delta, axis=1)
        norm_scale = max(float(np.max(raw_norm)), 1.0)
        tolerance = 64.0 * np.finfo(float).eps * norm_scale
        if abs(float(corrected_norm[0])) > tolerance:
            return "phase separation does not vanish at the CEP"
        if np.any(np.diff(corrected_norm) < -tolerance):
            return "phase separation closes again inside the closure interval"

        raw_center = 0.5 * (raw_hadronic + raw_qgp)
        distances = np.abs(
            selected_muB[closure_indices] - selected_muB[critical_index]
        )
        match_distance = float(distances[-1])
        for local_index, source_index in enumerate(closure_indices[:-1]):
            s = float(distances[local_index] / match_distance)
            weight = 3.0 * s**2 - 2.0 * s**3
            expected_delta = weight * (
                raw_qgp[source_index] - raw_hadronic[source_index]
            )
            if not np.allclose(
                corrected_qgp[source_index] - corrected_hadronic[source_index],
                expected_delta,
                rtol=2.0e-13,
                atol=tolerance,
            ):
                return "phase-separation vector changed orientation"
            expected_center = raw_center[source_index] + (1.0 - weight) * (
                critical_state - raw_center[critical_index]
            )
            if not np.allclose(
                0.5
                * (
                    corrected_hadronic[source_index]
                    + corrected_qgp[source_index]
                ),
                expected_center,
                rtol=2.0e-13,
                atol=tolerance,
            ):
                return "closure centerline does not follow the configured formula"

        cell_orientations = []
        for local_index in range(match_position):
            start = int(path_indices[local_index])
            stop = int(path_indices[local_index + 1])
            orientation, error = self._cell_geometry(
                corrected_hadronic[start],
                corrected_qgp[start],
                corrected_hadronic[stop],
                corrected_qgp[stop],
            )
            if error is not None:
                return "%s at closure cell %d" % (error, local_index)
            assert orientation is not None
            cell_orientations.append(orientation)

        if match_position + 1 < path_indices.size:
            outside_start = int(path_indices[match_position])
            outside_stop = int(path_indices[match_position + 1])
            reference_orientation, error = self._cell_geometry(
                raw_hadronic[outside_start],
                raw_qgp[outside_start],
                raw_hadronic[outside_stop],
                raw_qgp[outside_stop],
            )
            if error is not None:
                return "first untouched target cell is invalid: %s" % error
            assert reference_orientation is not None
        else:
            reference_orientation = cell_orientations[-1]
        if any(
            orientation != reference_orientation
            for orientation in cell_orientations
        ):
            return "target-cell orientation flips inside the closure interval"
        return None

    def _apply_critical_closure(
        self,
        raw_hadronic: np.ndarray,
        raw_qgp: np.ndarray,
        selected_muB: np.ndarray,
        critical_output_index: int,
        critical_state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
        """Apply the local numerical CEP closure in canonical ``(e, nB)``."""

        if critical_output_index == 0:
            path_indices = np.arange(selected_muB.size, dtype=int)
        elif critical_output_index == selected_muB.size - 1:
            path_indices = np.arange(selected_muB.size - 1, -1, -1, dtype=int)
        else:
            raise RegionSReconstructionError(
                "the critical endpoint must bound the reconstructed LINE domain"
            )
        distances = np.abs(
            selected_muB[path_indices] - selected_muB[critical_output_index]
        )
        width_tolerance = max(
            self.table.metadata.grid.absolute_tolerance,
            self.config.uniformity_absolute_tolerance,
        )
        candidates = np.flatnonzero(
            distances >= self.config.critical_closure_width - width_tolerance
        )
        if candidates.size == 0:
            raise RegionSReconstructionError(
                "the first-order LINE domain is shorter than the requested "
                "critical closure width %r GeV"
                % self.config.critical_closure_width
            )
        initial_match_position = int(candidates[0])
        maximum_match_position = min(
            path_indices.size - 1,
            initial_match_position
            + self.config.critical_closure_max_slice_expansions,
        )
        raw_center = 0.5 * (raw_hadronic + raw_qgp)
        raw_separation = 0.5 * (raw_qgp - raw_hadronic)
        last_error = "unknown closure geometry failure"

        for match_position in range(
            initial_match_position, maximum_match_position + 1
        ):
            match_distance = float(distances[match_position])
            corrected_hadronic = np.array(raw_hadronic, copy=True)
            corrected_qgp = np.array(raw_qgp, copy=True)
            for local_index in range(match_position):
                source_index = int(path_indices[local_index])
                s = float(distances[local_index] / match_distance)
                weight = 3.0 * s**2 - 2.0 * s**3
                corrected_center = raw_center[source_index] + (1.0 - weight) * (
                    critical_state - raw_center[critical_output_index]
                )
                corrected_separation = weight * raw_separation[source_index]
                corrected_hadronic[source_index] = (
                    corrected_center - corrected_separation
                )
                corrected_qgp[source_index] = (
                    corrected_center + corrected_separation
                )
            corrected_hadronic[critical_output_index] = critical_state
            corrected_qgp[critical_output_index] = critical_state
            error = self._closure_geometry_error(
                raw_hadronic,
                raw_qgp,
                corrected_hadronic,
                corrected_qgp,
                selected_muB,
                path_indices,
                match_position,
                critical_state,
            )
            if error is None:
                match_index = int(path_indices[match_position])
                return corrected_hadronic, corrected_qgp, {
                    "critical_closure_match_muB_GeV": float(
                        selected_muB[match_index]
                    ),
                    "critical_closure_effective_width_GeV": match_distance,
                    "critical_closure_slice_count": match_position + 1,
                    "critical_closure_adaptive_expansions": (
                        match_position - initial_match_position
                    ),
                    "critical_closure_geometry_validated": True,
                }
            last_error = error

        raise RegionSReconstructionError(
            "near-CEP numerical closure failed after %d candidate matching "
            "slices; last failure: %s"
            % (
                maximum_match_position - initial_match_position + 1,
                last_error,
            )
        )

    @staticmethod
    def _apply_auxiliary_critical_closure(
        raw_hadronic: np.ndarray,
        raw_qgp: np.ndarray,
        selected_muB: np.ndarray,
        critical_output_index: int,
        critical_state: np.ndarray,
        closure_slice_count: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the target RegionS closure weights to auxiliary endpoints."""

        if critical_output_index == 0:
            path_indices = np.arange(selected_muB.size, dtype=int)
        elif critical_output_index == selected_muB.size - 1:
            path_indices = np.arange(selected_muB.size - 1, -1, -1, dtype=int)
        else:
            raise RegionSReconstructionError(
                "the critical endpoint must bound auxiliary LINE endpoints"
            )
        match_position = int(closure_slice_count) - 1
        if match_position < 1 or match_position >= path_indices.size:
            raise RegionSReconstructionError(
                "auxiliary closure received an invalid target closure interval"
            )
        distances = np.abs(
            selected_muB[path_indices] - selected_muB[critical_output_index]
        )
        match_distance = float(distances[match_position])
        raw_center = 0.5 * (raw_hadronic + raw_qgp)
        raw_separation = 0.5 * (raw_qgp - raw_hadronic)
        corrected_hadronic = np.array(raw_hadronic, copy=True)
        corrected_qgp = np.array(raw_qgp, copy=True)
        for local_index in range(match_position):
            source_index = int(path_indices[local_index])
            scaled_distance = float(distances[local_index] / match_distance)
            weight = 3.0 * scaled_distance**2 - 2.0 * scaled_distance**3
            corrected_center = raw_center[source_index] + (1.0 - weight) * (
                critical_state - raw_center[critical_output_index]
            )
            corrected_separation = weight * raw_separation[source_index]
            corrected_hadronic[source_index] = (
                corrected_center - corrected_separation
            )
            corrected_qgp[source_index] = corrected_center + corrected_separation
        corrected_hadronic[critical_output_index] = critical_state
        corrected_qgp[critical_output_index] = critical_state
        return corrected_hadronic, corrected_qgp

    def build(self) -> MixedPhaseStrip:
        temperature_axis = np.asarray(self.table.axes[0], dtype=float)
        muB_axis = np.asarray(self.table.axes[1], dtype=float)
        spacing = self._validate_temperature_axis(temperature_axis)
        lower_muB, upper_muB = self.line.muB_bounds
        selected_indices = np.flatnonzero(
            (muB_axis >= lower_muB) & (muB_axis <= upper_muB)
        )
        if selected_indices.size < 2:
            raise RegionSReconstructionError(
                "the table has fewer than two muB slices in the transition domain"
            )
        selected_muB = muB_axis[selected_indices]
        transition_temperature = np.asarray(
            self.line.evaluate(selected_muB), dtype=float
        )
        critical_muB = float(self.line.critical_point["muB"])
        assert self.table.metadata is not None
        grid_tolerance = self.table.metadata.grid
        critical_matches = np.flatnonzero(
            np.isclose(
                selected_muB,
                critical_muB,
                rtol=grid_tolerance.relative_tolerance,
                atol=grid_tolerance.absolute_tolerance,
            )
        )
        if critical_matches.size != 1:
            raise RegionSReconstructionError(
                "the declared critical muB=%r GeV must coincide with exactly "
                "one source-table muB slice; found %d"
                % (critical_muB, int(critical_matches.size))
            )
        critical_output_index = int(critical_matches[0])
        energy = np.asarray(self.table.grid_fields["e"], dtype=float)
        density = np.asarray(self.table.grid_fields["nB"], dtype=float)
        auxiliary_specs = tuple(
            self.table.metadata.fields[name] for name in self.auxiliary_fields
        )
        auxiliary_grids = tuple(
            np.asarray(self.table.grid_fields[name], dtype=float)
            for name in self.auxiliary_fields
        )
        critical_source_index = int(selected_indices[critical_output_index])
        critical_temperature = float(
            transition_temperature[critical_output_index]
        )
        critical_state = np.asarray(
            (
                np.interp(
                    critical_temperature,
                    temperature_axis,
                    energy[:, critical_source_index],
                ),
                np.interp(
                    critical_temperature,
                    temperature_axis,
                    density[:, critical_source_index],
                ),
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(critical_state)):
            raise RegionSReconstructionError(
                "critical-endpoint table interpolation produced a "
                "nonfinite target"
            )
        critical_auxiliary_state = np.asarray(
            [
                np.interp(
                    critical_temperature,
                    temperature_axis,
                    values[:, critical_source_index],
                )
                for values in auxiliary_grids
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(critical_auxiliary_state)):
            raise RegionSReconstructionError(
                "critical-endpoint table interpolation produced a nonfinite "
                "auxiliary value"
            )
        if self.config.qgp_energy_gaussian_sigma > 0.0:
            sigma_index = self.config.qgp_energy_gaussian_sigma / spacing
            smoothed_energy = gaussian_filter1d(
                energy,
                sigma=sigma_index,
                axis=0,
                mode="nearest",
                truncate=self.config.gaussian_truncate,
            )
        else:
            smoothed_energy = energy

        count = selected_indices.size
        e_hadronic = np.empty(count, dtype=float)
        nB_hadronic = np.empty(count, dtype=float)
        e_qgp = np.empty(count, dtype=float)
        nB_qgp = np.empty(count, dtype=float)
        hadronic_counts = np.empty(count, dtype=np.int64)
        qgp_counts = np.empty(count, dtype=np.int64)
        auxiliary_hadronic = np.empty(
            (count, len(self.auxiliary_fields)), dtype=float
        )
        auxiliary_qgp = np.empty(
            (count, len(self.auxiliary_fields)), dtype=float
        )
        required = self.config.minimum_side_points

        for output_index, (source_index, muB, critical_temperature) in enumerate(
            zip(selected_indices, selected_muB, transition_temperature)
        ):
            hadronic_mask = temperature_axis <= (
                critical_temperature - self.config.hadronic_temperature_offset
            )
            qgp_mask = temperature_axis >= (
                critical_temperature + self.config.qgp_temperature_offset
            )
            hadronic_count = int(np.count_nonzero(hadronic_mask))
            qgp_count = int(np.count_nonzero(qgp_mask))
            hadronic_counts[output_index] = hadronic_count
            qgp_counts[output_index] = qgp_count
            if hadronic_count < required or qgp_count < required:
                raise RegionSReconstructionError(
                    "muB=%r GeV has insufficient side samples: hadronic=%d, "
                    "qgp=%d, required=%d"
                    % (float(muB), hadronic_count, qgp_count, required)
                )

            hadronic_temperature = temperature_axis[hadronic_mask]
            qgp_temperature = temperature_axis[qgp_mask]
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                try:
                    hadronic_energy_scaled = (
                        energy[hadronic_mask, source_index]
                        / hadronic_temperature**4
                    )
                    hadronic_density_scaled = (
                        density[hadronic_mask, source_index]
                        / hadronic_temperature**3
                    )
                    qgp_energy_scaled = (
                        smoothed_energy[qgp_mask, source_index]
                        / qgp_temperature**4
                    )
                    qgp_density_scaled = (
                        density[qgp_mask, source_index] / qgp_temperature**3
                    )
                except FloatingPointError as exc:
                    raise RegionSReconstructionError(
                        "scaled endpoint fields overflow at muB=%r GeV"
                        % float(muB)
                    ) from exc

            e_hadronic[output_index] = self._fit(
                hadronic_temperature,
                hadronic_energy_scaled,
                float(critical_temperature),
                self.config.hadronic_energy_smoothing,
                "hadronic energy",
            ) * critical_temperature**4
            nB_hadronic[output_index] = self._fit(
                hadronic_temperature,
                hadronic_density_scaled,
                float(critical_temperature),
                self.config.hadronic_density_smoothing,
                "hadronic density",
            ) * critical_temperature**3
            e_qgp[output_index] = self._fit(
                qgp_temperature,
                qgp_energy_scaled,
                float(critical_temperature),
                self.config.qgp_energy_smoothing,
                "QGP energy",
            ) * critical_temperature**4
            nB_qgp[output_index] = self._fit(
                qgp_temperature,
                qgp_density_scaled,
                float(critical_temperature),
                self.config.qgp_density_smoothing,
                "QGP density",
            ) * critical_temperature**3

            for field_index, (name, spec, values) in enumerate(
                zip(self.auxiliary_fields, auxiliary_specs, auxiliary_grids)
            ):
                power = spec.transform.temperature_power
                with np.errstate(over="raise", divide="raise", invalid="raise"):
                    try:
                        hadronic_scaled = (
                            values[hadronic_mask, source_index]
                            / hadronic_temperature**power
                        )
                        qgp_scaled = (
                            values[qgp_mask, source_index]
                            / qgp_temperature**power
                        )
                    except FloatingPointError as exc:
                        raise RegionSReconstructionError(
                            "%s scaled endpoint field overflows at muB=%r GeV"
                            % (name, float(muB))
                        ) from exc
                auxiliary_hadronic[output_index, field_index] = self._fit(
                    hadronic_temperature,
                    hadronic_scaled,
                    float(critical_temperature),
                    0.0,
                    "hadronic auxiliary %s" % name,
                ) * critical_temperature**power
                auxiliary_qgp[output_index, field_index] = self._fit(
                    qgp_temperature,
                    qgp_scaled,
                    float(critical_temperature),
                    0.0,
                    "QGP auxiliary %s" % name,
                ) * critical_temperature**power

        raw_hadronic = np.column_stack((e_hadronic, nB_hadronic))
        raw_qgp = np.column_stack((e_qgp, nB_qgp))
        if self.config.critical_closure_enabled:
            corrected_hadronic, corrected_qgp, closure_diagnostics = (
                self._apply_critical_closure(
                    raw_hadronic,
                    raw_qgp,
                    selected_muB,
                    critical_output_index,
                    critical_state,
                )
            )
            closure_diagnostics = {
                "critical_closure_applied": True,
                **closure_diagnostics,
            }
        else:
            corrected_hadronic = np.array(raw_hadronic, copy=True)
            corrected_qgp = np.array(raw_qgp, copy=True)
            closure_diagnostics = {
                "critical_closure_applied": False,
                "critical_closure_match_muB_GeV": None,
                "critical_closure_effective_width_GeV": None,
                "critical_closure_slice_count": 0,
                "critical_closure_adaptive_expansions": 0,
                "critical_closure_geometry_validated": None,
            }
        if self.auxiliary_fields and self.config.critical_closure_enabled:
            auxiliary_hadronic, auxiliary_qgp = (
                self._apply_auxiliary_critical_closure(
                    auxiliary_hadronic,
                    auxiliary_qgp,
                    selected_muB,
                    critical_output_index,
                    critical_auxiliary_state,
                    int(closure_diagnostics["critical_closure_slice_count"]),
                )
            )
        e_hadronic = corrected_hadronic[:, 0]
        nB_hadronic = corrected_hadronic[:, 1]
        e_qgp = corrected_qgp[:, 0]
        nB_qgp = corrected_qgp[:, 1]

        endpoints = RegionSEndpoints(
            temperature=transition_temperature,
            muB=selected_muB,
            e_hadronic=e_hadronic,
            nB_hadronic=nB_hadronic,
            e_qgp=e_qgp,
            nB_qgp=nB_qgp,
            table_fingerprint=self.table.snapshot_fingerprint(),
            sidecar_fingerprint=self.line.sidecar_fingerprint,
        )
        auxiliary_endpoints = (
            RegionSAuxiliaryEndpoints(
                field_names=self.auxiliary_fields,
                field_units=tuple(spec.canonical_unit for spec in auxiliary_specs),
                temperature_powers=tuple(
                    spec.transform.temperature_power for spec in auxiliary_specs
                ),
                muB=selected_muB,
                hadronic_values=auxiliary_hadronic,
                qgp_values=auxiliary_qgp,
                table_fingerprint=endpoints.table_fingerprint,
                sidecar_fingerprint=endpoints.sidecar_fingerprint,
            )
            if self.auxiliary_fields
            else None
        )
        segment_norms = endpoints.target_segment_norms
        degenerate = segment_norms <= self.config.minimum_target_segment_norm
        at_critical_endpoint = np.zeros(endpoints.point_count, dtype=bool)
        at_critical_endpoint[critical_output_index] = True
        invalid_degenerate = degenerate & ~at_critical_endpoint
        if np.any(invalid_degenerate):
            first = int(np.flatnonzero(invalid_degenerate)[0])
            raise RegionSReconstructionError(
                "reconstructed target segment is degenerate at muB=%r GeV"
                % float(endpoints.muB[first])
            )
        config = self.config.to_dict()
        config.update(closure_diagnostics)
        if auxiliary_endpoints is not None:
            config["auxiliary_endpoint_fields"] = list(
                auxiliary_endpoints.field_names
            )
            config["auxiliary_endpoint_fit"] = {
                "side_masks": "same_as_target_endpoints",
                "spline_degree": self.config.spline_degree,
                "smoothing": 0.0,
                "fit_quantity": "canonical_field_divided_by_T_power_from_metadata",
                "critical_closure": "same_center_separation_weight_and_match_slice",
                "mixture_rule": "linear_lever_rule",
            }
        endpoint_fingerprint = _endpoint_fingerprint(
            endpoints, config, auxiliary_endpoints
        )
        delta = endpoints.target_deltas
        diagnostics = RegionSReconstructionDiagnostics(
            algorithm=_REGION_S_ALGORITHM,
            table_fingerprint=endpoints.table_fingerprint,
            sidecar_fingerprint=endpoints.sidecar_fingerprint,
            endpoint_fingerprint=endpoint_fingerprint,
            config=config,
            candidate_slice_count=int(count),
            reconstructed_slice_count=int(count),
            muB_bounds=endpoints.muB_bounds,
            temperature_bounds=(
                float(np.min(endpoints.temperature)),
                float(np.max(endpoints.temperature)),
            ),
            hadronic_sample_count_bounds=(
                int(np.min(hadronic_counts)),
                int(np.max(hadronic_counts)),
            ),
            qgp_sample_count_bounds=(
                int(np.min(qgp_counts)),
                int(np.max(qgp_counts)),
            ),
            target_segment_norm_bounds=(
                float(np.min(segment_norms)),
                float(np.max(segment_norms)),
            ),
            energy_orientation_counts=_orientation_counts(delta[:, 0]),
            density_orientation_counts=_orientation_counts(delta[:, 1]),
        )
        return MixedPhaseStrip(
            line=self.line,
            endpoints=endpoints,
            diagnostics=diagnostics,
            auxiliary_endpoints=auxiliary_endpoints,
        )


def build_region_s(
    table: ForwardTable,
    config: Optional[RegionSConfig] = None,
    *,
    auxiliary_fields: Sequence[str] = (),
) -> MixedPhaseStrip:
    """Build the corrected physical LINE RegionS strip."""

    return RegionSBuilder(
        table,
        config=config,
        auxiliary_fields=auxiliary_fields,
    ).build()


__all__ = [
    "BinodalBand",
    "FirstOrderRepresentation",
    "MixedPhaseCellEnclosures",
    "MixedPhaseStrip",
    "PhaseLabel",
    "PhaseRootSelectionReason",
    "RegionSBuilder",
    "RegionSAuxiliaryEndpoints",
    "RegionSConfig",
    "RegionSEndpoints",
    "RegionSReconstructionDiagnostics",
    "RegionSReconstructionError",
    "TransitionConfigurationError",
    "TransitionDiagnostics",
    "TransitionDomainError",
    "TransitionError",
    "TransitionInputError",
    "TransitionLine",
    "TransitionModel",
    "build_region_s",
    "transition_model_from_table",
]
