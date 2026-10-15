"""Robust inversion of a validated multilinear EoS interpolant.

The single-target solver contract is documented in ``docs/SOLVER.md``:

* bounded, scaled least-squares localization;
* safeguarded Newton refinement; and
* a cell-local Krawczyk fallback using outward-rounded interval arithmetic.

The fast path is deliberately numerical rather than evidentiary.  Only the
fallback emits proof-bearing results, and it distinguishes a unique root in
one box from global uniqueness.  Exhausting a resource limit is reported as
unresolved, never as a proof that no root exists.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .interpolation import (
    InterpolationDomainError,
    MultilinearInterpolator,
)
from .subcells import (
    DEFAULT_MAX_SUBCELL_BOXES,
    MAX_SUBCELL_DEPTH,
    SubcellBoxes,
    SubcellEnclosures,
    SubcellRefinementReport,
    SubcellRefinementResult,
    refine_subcells,
)
from .tables import EoSError


class SolverError(EoSError):
    """Base exception for inversion-solver failures."""


class SolverConfigurationError(SolverError):
    """Raised when a solver cannot be configured for an interpolator."""


class SolverInputError(SolverError):
    """Raised when a target, initial guess, or strategy is malformed."""


class SolverInvariantError(SolverError):
    """Raised when a bound-safe internal stage attempts extrapolation."""


class SolveStrategy(str, Enum):
    """Top-level pipeline selection."""

    AUTO = "auto"
    FAST = "fast"
    FALLBACK = "fallback"


class SolveStatus(str, Enum):
    """Final inversion outcome."""

    CONVERGED_FAST = "converged_fast"
    CONVERGED_FALLBACK = "converged_fallback"
    CERTIFIED_ROOT_UNREFINED = "certified_root_unrefined"
    CERTIFIED_NO_ROOT = "certified_no_root"
    UNRESOLVED = "unresolved"
    FAST_PATH_FAILED = "fast_path_failed"


class ProofStatus(str, Enum):
    """Mathematical information established independently of convergence."""

    NONE = "none"
    UNIQUE_IN_BOX = "unique_in_box"
    NO_ROOT_IN_DOMAIN = "no_root_in_domain"
    INCONCLUSIVE = "inconclusive"


class FallbackStatus(str, Enum):
    """Outcome of the interval search itself."""

    UNIQUE_IN_BOX = "unique_in_box"
    NO_ROOT_IN_DOMAIN = "no_root_in_domain"
    UNRESOLVED = "unresolved"


def _immutable_array(value: Any, dtype: Optional[Any] = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
    return immutable.reshape(array.shape)


def _optional_immutable_array(value: Optional[Any]) -> Optional[np.ndarray]:
    return None if value is None else _immutable_array(value, dtype=float)


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a finite positive number" % name)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a finite positive number" % name) from exc
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError("%s must be a finite positive number" % name)
    return converted


def _automatic_residual_scales_from_bounds(
    lower: Any, upper: Any
) -> np.ndarray:
    """Return deterministic target-independent scales from validated bounds.

    A nonzero component uses its maximum absolute table-node magnitude.  An
    exactly zero component uses one canonical unit so the normalization can
    never divide by zero.  Positive subnormal magnitudes are raised to the
    smallest normal binary64 value so their reciprocals remain finite.
    """

    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if (
        lower_array.ndim != 1
        or upper_array.shape != lower_array.shape
        or not np.all(np.isfinite(lower_array))
        or not np.all(np.isfinite(upper_array))
        or np.any(lower_array > upper_array)
    ):
        raise ValueError(
            "automatic residual scaling requires finite ordered component bounds"
        )
    characteristic = np.maximum(np.abs(lower_array), np.abs(upper_array))
    scales = np.maximum(characteristic, np.finfo(float).tiny)
    scales = np.where(characteristic == 0.0, 1.0, scales)
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        reciprocals = 1.0 / scales
    if (
        not np.all(np.isfinite(scales))
        or np.any(scales <= 0.0)
        or not np.all(np.isfinite(reciprocals))
        or np.any(reciprocals <= 0.0)
    ):
        raise ValueError("automatic residual scales are not safely invertible")
    return _immutable_array(scales, dtype=float)


def _finite_unit_interval(value: Any, name: str, *, open_lower: bool = False) -> float:
    converted = _finite_positive(value, name) if open_lower else float(value)
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(converted):
        raise TypeError("%s must be a finite number" % name)
    lower_ok = converted > 0.0 if open_lower else converted >= 0.0
    if not lower_ok or converted >= 1.0:
        relation = "0 < value < 1" if open_lower else "0 <= value < 1"
        raise ValueError("%s must satisfy %s" % (name, relation))
    return converted


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("%s must be a positive integer" % name)
    converted = int(value)
    if converted <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return converted


def _bounded_midpoint(lower: Any, upper: Any) -> np.ndarray:
    """Return a representable midpoint guaranteed to remain in its box."""

    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        midpoint = lower_array + (upper_array - lower_array) * np.float64(0.5)
    return np.minimum(np.maximum(midpoint, lower_array), upper_array)


def _supports_rigorous_binary64_intervals() -> bool:
    """Check the floating-point properties required by the proof kernel."""

    if np.dtype(np.float64).itemsize != 8:
        return False
    info = np.finfo(np.float64)
    zero = np.float64(0.0)
    one = np.float64(1.0)
    smallest = np.nextafter(zero, one)
    with np.errstate(under="ignore", invalid="ignore"):
        half_min_normal = np.float64(info.tiny) * np.float64(0.5)
        doubled_subnormal = smallest + smallest
        half_ulp_sum = one + np.float64(info.eps) * np.float64(0.5)
        one_ulp_sum = one + np.float64(info.eps)
    return bool(
        info.bits == 64
        and smallest > 0.0
        and half_min_normal > 0.0
        and doubled_subnormal > smallest
        and half_ulp_sum == one
        and one_ulp_sum == np.nextafter(one, np.inf)
    )


@dataclass(frozen=True)
class SolverConfig:
    """Numerical tolerances and explicit work limits for inversion.

    Residual tolerances are dimensionless because every stage solves the same
    row-scaled system. Automatic row scales are deterministic characteristic
    magnitudes derived only from the validated global node bounds; callers may
    instead supply fixed physical scales to :class:`EoSRootSolver`.
    """

    localization_residual_tolerance: float = 1.0e-6
    root_residual_tolerance: float = 1.0e-10
    least_squares_max_evaluations: int = 200
    least_squares_ftol: float = 1.0e-10
    least_squares_xtol: float = 1.0e-10
    # A bounded solution can have a small projected gradient while its
    # residual is still of order sqrt(gtol).  Keep this below the square of
    # the default Newton-entry tolerance so valid face roots are handed off.
    least_squares_gtol: float = 1.0e-12
    newton_max_iterations: int = 24
    newton_condition_limit: float = 1.0e12
    newton_step_tolerance: float = 1.0e-13
    newton_armijo: float = 1.0e-4
    newton_backtrack_factor: float = 0.5
    newton_min_step_fraction: float = 2.0 ** -24
    fallback_max_scanned_cells: int = 2_000_000
    fallback_max_candidate_cells: int = 100_000
    fallback_scan_chunk_size: int = 65_536
    fallback_max_boxes: int = 100_000
    fallback_max_depth: int = 40
    fallback_min_normalized_width: float = 2.0 ** -40
    fallback_contraction_volume_ratio: float = 0.8
    fallback_cell_cache_size: int = 64
    enable_subcell_fallback: bool = True
    max_subcell_depth: int = MAX_SUBCELL_DEPTH
    max_subcell_active_boxes: int = DEFAULT_MAX_SUBCELL_BOXES

    def __post_init__(self) -> None:
        positive_floats = (
            "localization_residual_tolerance",
            "root_residual_tolerance",
            "least_squares_ftol",
            "least_squares_xtol",
            "least_squares_gtol",
            "newton_condition_limit",
            "newton_step_tolerance",
            "newton_min_step_fraction",
            "fallback_min_normalized_width",
        )
        for name in positive_floats:
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))
        for name in (
            "least_squares_ftol",
            "least_squares_xtol",
            "least_squares_gtol",
        ):
            if getattr(self, name) <= np.finfo(float).eps:
                raise ValueError(
                    "%s must be greater than binary64 machine epsilon" % name
                )
        if self.root_residual_tolerance > self.localization_residual_tolerance:
            raise ValueError(
                "root_residual_tolerance cannot exceed "
                "localization_residual_tolerance"
            )
        for name in (
            "least_squares_max_evaluations",
            "newton_max_iterations",
            "fallback_max_scanned_cells",
            "fallback_max_candidate_cells",
            "fallback_scan_chunk_size",
            "fallback_max_boxes",
            "fallback_max_depth",
            "fallback_cell_cache_size",
            "max_subcell_depth",
            "max_subcell_active_boxes",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if self.max_subcell_depth > MAX_SUBCELL_DEPTH:
            raise ValueError(
                "max_subcell_depth cannot exceed %d" % MAX_SUBCELL_DEPTH
            )
        if not isinstance(self.enable_subcell_fallback, (bool, np.bool_)):
            raise TypeError("enable_subcell_fallback must be boolean")
        object.__setattr__(
            self, "enable_subcell_fallback", bool(self.enable_subcell_fallback)
        )
        object.__setattr__(
            self,
            "newton_armijo",
            _finite_unit_interval(self.newton_armijo, "newton_armijo", open_lower=True),
        )
        object.__setattr__(
            self,
            "newton_backtrack_factor",
            _finite_unit_interval(
                self.newton_backtrack_factor,
                "newton_backtrack_factor",
                open_lower=True,
            ),
        )
        object.__setattr__(
            self,
            "fallback_contraction_volume_ratio",
            _finite_unit_interval(
                self.fallback_contraction_volume_ratio,
                "fallback_contraction_volume_ratio",
                open_lower=True,
            ),
        )


@dataclass(frozen=True)
class LocalizationReport:
    """Diagnostics from bounded trust-region localization."""

    accepted_for_newton: bool
    optimizer_success: bool
    point: np.ndarray
    forward_values: np.ndarray
    residual: np.ndarray
    scaled_residual_norm: float
    optimizer_status: int
    message: str
    function_evaluations: int
    jacobian_evaluations: Optional[int]
    cost: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _immutable_array(self.point, dtype=float))
        object.__setattr__(
            self, "forward_values", _immutable_array(self.forward_values, dtype=float)
        )
        object.__setattr__(self, "residual", _immutable_array(self.residual, dtype=float))


@dataclass(frozen=True)
class NewtonReport:
    """Diagnostics from safeguarded Newton refinement."""

    converged: bool
    point: np.ndarray
    forward_values: np.ndarray
    residual: np.ndarray
    scaled_residual_norm: float
    iterations: int
    function_evaluations: int
    jacobian_evaluations: int
    condition_estimate: Optional[float]
    reason: str
    normalized_bounds: np.ndarray
    normalized_coordinate_system: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _immutable_array(self.point, dtype=float))
        object.__setattr__(
            self, "forward_values", _immutable_array(self.forward_values, dtype=float)
        )
        object.__setattr__(self, "residual", _immutable_array(self.residual, dtype=float))
        object.__setattr__(
            self, "normalized_bounds", _immutable_array(self.normalized_bounds, dtype=float)
        )


@dataclass(frozen=True)
class CertifiedRootBox:
    """A rigorous unique-root enclosure within one interpolation cell."""

    coordinate_names: Tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    cell_indices: np.ndarray
    cell_normalized_lower: np.ndarray
    cell_normalized_upper: np.ndarray
    rigorous: bool = True
    uniqueness_scope: str = "box_only"

    def __post_init__(self) -> None:
        names = tuple(self.coordinate_names)
        dimension = len(names)
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        indices = np.asarray(self.cell_indices)
        unit_lower = np.asarray(self.cell_normalized_lower, dtype=float)
        unit_upper = np.asarray(self.cell_normalized_upper, dtype=float)
        for array in (lower, upper, indices, unit_lower, unit_upper):
            if array.shape != (dimension,):
                raise ValueError("certified root-box arrays must match the dimension")
        if np.any(lower > upper) or np.any(unit_lower > unit_upper):
            raise ValueError("certified root-box lower bounds cannot exceed upper bounds")
        if not self.rigorous:
            raise ValueError("CertifiedRootBox requires rigorous arithmetic")
        if self.uniqueness_scope != "box_only":
            raise ValueError("only box-local uniqueness is currently supported")
        object.__setattr__(self, "coordinate_names", names)
        object.__setattr__(self, "lower", _immutable_array(lower))
        object.__setattr__(self, "upper", _immutable_array(upper))
        object.__setattr__(
            self, "cell_indices", _immutable_array(indices, dtype=np.intp)
        )
        object.__setattr__(
            self, "cell_normalized_lower", _immutable_array(unit_lower)
        )
        object.__setattr__(
            self, "cell_normalized_upper", _immutable_array(unit_upper)
        )


@dataclass(frozen=True)
class FallbackReport:
    """Diagnostics and proof outcome from the cell-local interval search."""

    status: FallbackStatus
    message: str
    seed_source: str
    total_cells: int
    cells_scanned: int
    scan_complete: bool
    candidate_cells: int
    retained_candidate_cells: int
    cells_excluded_by_range: int
    boxes_processed: int
    boxes_excluded: int
    boxes_contracted: int
    boxes_subdivided: int
    unresolved_boxes: int
    maximum_depth_reached: int
    certificate: Optional[CertifiedRootBox] = None
    outward_rounded: bool = False
    candidate_source: str = "global_scan"
    localization_complete: bool = False
    supplied_candidate_cells: int = 0
    cells_excluded_by_reachability: int = 0
    supplied_cells_excluded_by_range: int = 0
    unprocessed_candidate_cells: int = 0
    subcell_report: Optional[SubcellRefinementReport] = None
    subcell_residual_box_count: int = 0


@dataclass(frozen=True)
class InversionResult:
    """Immutable result from one target inversion."""

    status: SolveStatus
    proof_status: ProofStatus
    strategy: SolveStrategy
    target: np.ndarray
    solution: Optional[np.ndarray]
    best_candidate: Optional[np.ndarray]
    forward_values: Optional[np.ndarray]
    residual: Optional[np.ndarray]
    scaled_residual: Optional[np.ndarray]
    residual_scales: np.ndarray
    initial_guess: np.ndarray
    initial_guess_source: str
    path: Tuple[str, ...]
    message: str
    localization: Optional[LocalizationReport] = None
    newton: Optional[NewtonReport] = None
    fallback: Optional[FallbackReport] = None
    fallback_newton: Optional[NewtonReport] = None
    certificate: Optional[CertifiedRootBox] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _immutable_array(self.target, dtype=float))
        object.__setattr__(self, "solution", _optional_immutable_array(self.solution))
        object.__setattr__(
            self, "best_candidate", _optional_immutable_array(self.best_candidate)
        )
        object.__setattr__(
            self, "forward_values", _optional_immutable_array(self.forward_values)
        )
        object.__setattr__(self, "residual", _optional_immutable_array(self.residual))
        object.__setattr__(
            self, "scaled_residual", _optional_immutable_array(self.scaled_residual)
        )
        object.__setattr__(
            self, "residual_scales", _immutable_array(self.residual_scales, dtype=float)
        )
        object.__setattr__(
            self, "initial_guess", _immutable_array(self.initial_guess, dtype=float)
        )
        object.__setattr__(self, "path", tuple(self.path))

    @property
    def converged(self) -> bool:
        return self.status in {
            SolveStatus.CONVERGED_FAST,
            SolveStatus.CONVERGED_FALLBACK,
        }

    @property
    def certified(self) -> bool:
        return self.proof_status in {
            ProofStatus.UNIQUE_IN_BOX,
            ProofStatus.NO_ROOT_IN_DOMAIN,
        }

    @property
    def scaled_residual_norm(self) -> Optional[float]:
        if self.scaled_residual is None:
            return None
        return float(np.max(np.abs(self.scaled_residual)))


class _ScaledInverseProblem:
    """One target expressed in normalized source and residual coordinates."""

    def __init__(
        self,
        interpolator: MultilinearInterpolator,
        target: np.ndarray,
        residual_scales: np.ndarray,
        *,
        source_lower: Optional[np.ndarray] = None,
        source_upper: Optional[np.ndarray] = None,
        normalized_coordinate_system: str = "global_domain",
    ) -> None:
        self.interpolator = interpolator
        self.target = np.asarray(target, dtype=float)
        self.residual_scales = np.asarray(residual_scales, dtype=float)
        self.residual_factors = 1.0 / self.residual_scales
        if (
            not np.all(np.isfinite(self.residual_factors))
            or np.any(self.residual_factors <= 0.0)
        ):
            raise SolverConfigurationError(
                "residual scales must have finite positive reciprocals"
            )
        self.lower = np.asarray(
            interpolator.bounds[:, 0] if source_lower is None else source_lower,
            dtype=float,
        )
        self.upper = np.asarray(
            interpolator.bounds[:, 1] if source_upper is None else source_upper,
            dtype=float,
        )
        if (
            self.lower.shape != (interpolator.dimension,)
            or self.upper.shape != (interpolator.dimension,)
            or not np.all(np.isfinite(self.lower))
            or not np.all(np.isfinite(self.upper))
            or np.any(self.lower < interpolator.bounds[:, 0])
            or np.any(self.upper > interpolator.bounds[:, 1])
        ):
            raise SolverConfigurationError(
                "restricted source bounds must lie inside the interpolation domain"
            )
        self.span = self.upper - self.lower
        if not np.all(np.isfinite(self.span)) or np.any(self.span <= 0.0):
            raise SolverConfigurationError(
                "source-coordinate bounds do not have finite positive spans"
            )
        self.dimension = interpolator.dimension
        if not isinstance(normalized_coordinate_system, str) or not normalized_coordinate_system:
            raise SolverConfigurationError(
                "normalized_coordinate_system must be a non-empty string"
            )
        self._normalized_coordinate_system = normalized_coordinate_system

    def to_normalized(self, point: np.ndarray) -> np.ndarray:
        physical = np.asarray(point, dtype=float)
        normalized = (physical - self.lower) / self.span
        normalized = np.minimum(np.maximum(normalized, 0.0), 1.0)
        normalized = np.where(physical == self.lower, 0.0, normalized)
        return np.where(physical == self.upper, 1.0, normalized)

    def to_physical(self, normalized: np.ndarray) -> np.ndarray:
        unit = np.asarray(normalized, dtype=float)
        physical = self.lower + self.span * unit
        physical = np.minimum(np.maximum(physical, self.lower), self.upper)
        physical = np.where(unit == 0.0, self.lower, physical)
        return np.where(unit == 1.0, self.upper, physical)

    def values_and_residual(self, normalized: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        point = self.to_physical(normalized)
        try:
            values = np.asarray(self.interpolator.evaluate(point), dtype=float)
        except InterpolationDomainError as exc:
            raise SolverInvariantError(
                "a bound-safe solver stage attempted interpolation outside the domain"
            ) from exc
        residual = (values - self.target) * self.residual_factors
        return values, residual

    def jacobian(self, normalized: np.ndarray, sides: Any = "right") -> np.ndarray:
        point = self.to_physical(normalized)
        try:
            physical = np.asarray(
                self.interpolator.jacobian(point, plane_policy=sides), dtype=float
            )
        except InterpolationDomainError as exc:
            raise SolverInvariantError(
                "a bound-safe solver stage attempted interpolation outside the domain"
            ) from exc
        return (
            self.residual_factors[:, np.newaxis]
            * physical
            * self.span[np.newaxis, :]
        )

    def newton_jacobians(self, normalized: np.ndarray) -> Tuple[np.ndarray, ...]:
        physical = self.to_physical(normalized)
        location = self.interpolator.locate(physical)
        plane_mask = np.asarray(location.on_internal_grid_plane, dtype=bool)
        jacobians = []
        for sides_at_planes in itertools.product(
            ("left", "right"), repeat=int(np.count_nonzero(plane_mask))
        ):
            sides = []
            selected = iter(sides_at_planes)
            for on_plane in plane_mask:
                sides.append(next(selected) if on_plane else "right")
            jacobians.append(self.jacobian(normalized, tuple(sides)))
        return tuple(jacobians)

    @property
    def normalized_coordinate_system(self) -> str:
        return self._normalized_coordinate_system


class _LeastSquaresAdapter:
    """Cache the shared interpolation requested separately by SciPy."""

    def __init__(self, problem: _ScaledInverseProblem) -> None:
        self.problem = problem
        self._point: Optional[np.ndarray] = None
        self._residual: Optional[np.ndarray] = None
        self._jacobian: Optional[np.ndarray] = None

    def _evaluate(self, point: np.ndarray) -> None:
        if self._point is not None and np.array_equal(point, self._point):
            return
        physical = self.problem.to_physical(point)
        try:
            evaluation = self.problem.interpolator.evaluate_with_jacobian(physical)
        except InterpolationDomainError as exc:
            raise SolverInvariantError(
                "bounded least squares attempted interpolation outside the domain"
            ) from exc
        values = np.asarray(evaluation.values, dtype=float)
        jacobian = np.asarray(evaluation.jacobians, dtype=float)
        residual = (values - self.problem.target) * self.problem.residual_factors
        scaled_jacobian = (
            self.problem.residual_factors[:, np.newaxis]
            * jacobian
            * self.problem.span[np.newaxis, :]
        )
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(scaled_jacobian)):
            raise FloatingPointError("the scaled residual or Jacobian is non-finite")
        self._point = np.asarray(point, dtype=float).copy()
        self._residual = residual
        self._jacobian = scaled_jacobian

    def residual(self, point: np.ndarray) -> np.ndarray:
        self._evaluate(point)
        assert self._residual is not None
        return self._residual

    def jacobian(self, point: np.ndarray) -> np.ndarray:
        self._evaluate(point)
        assert self._jacobian is not None
        return self._jacobian


@dataclass(frozen=True)
class _Interval:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if lower.shape != upper.shape:
            raise ValueError("interval endpoints must have equal shapes")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


def _interval_point(value: Any) -> _Interval:
    array = np.asarray(value, dtype=np.float64)
    return _Interval(array, array)


def _round_down(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.nextafter(array, -np.inf)


def _round_up(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.nextafter(array, np.inf)


def _interval_add(left: _Interval, right: _Interval) -> _Interval:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return _Interval(
            _round_down(left.lower + right.lower),
            _round_up(left.upper + right.upper),
        )


def _interval_subtract(left: _Interval, right: _Interval) -> _Interval:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return _Interval(
            _round_down(left.lower - right.upper),
            _round_up(left.upper - right.lower),
        )


def _interval_multiply(left: _Interval, right: _Interval) -> _Interval:
    a, b, c, d = np.broadcast_arrays(
        left.lower, left.upper, right.lower, right.upper
    )
    lower = np.empty(a.shape, dtype=np.float64)
    upper = np.empty_like(lower)
    exact_zero = ((a == 0.0) & (b == 0.0)) | ((c == 0.0) & (d == 0.0))
    left_positive = a >= 0.0
    left_negative = b <= 0.0
    right_positive = c >= 0.0
    right_negative = d <= 0.0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        mask = left_positive & right_positive & ~exact_zero
        lower[mask], upper[mask] = a[mask] * c[mask], b[mask] * d[mask]
        mask = left_positive & right_negative & ~exact_zero
        lower[mask], upper[mask] = b[mask] * c[mask], a[mask] * d[mask]
        mask = left_positive & ~(right_positive | right_negative) & ~exact_zero
        lower[mask], upper[mask] = b[mask] * c[mask], b[mask] * d[mask]
        mask = left_negative & right_positive & ~exact_zero
        lower[mask], upper[mask] = a[mask] * d[mask], b[mask] * c[mask]
        mask = left_negative & right_negative & ~exact_zero
        lower[mask], upper[mask] = b[mask] * d[mask], a[mask] * c[mask]
        mask = left_negative & ~(right_positive | right_negative) & ~exact_zero
        lower[mask], upper[mask] = a[mask] * d[mask], a[mask] * c[mask]
        left_mixed = ~(left_positive | left_negative)
        mask = left_mixed & right_positive & ~exact_zero
        lower[mask], upper[mask] = a[mask] * d[mask], b[mask] * d[mask]
        mask = left_mixed & right_negative & ~exact_zero
        lower[mask], upper[mask] = b[mask] * c[mask], a[mask] * c[mask]
        mask = left_mixed & ~(right_positive | right_negative) & ~exact_zero
        lower[mask] = np.minimum(a[mask] * d[mask], b[mask] * c[mask])
        upper[mask] = np.maximum(a[mask] * c[mask], b[mask] * d[mask])
    lower[exact_zero] = 0.0
    upper[exact_zero] = 0.0
    rounded_lower = _round_down(lower)
    rounded_upper = _round_up(upper)
    rounded_lower = np.where(exact_zero, 0.0, rounded_lower)
    rounded_upper = np.where(exact_zero, 0.0, rounded_upper)
    return _Interval(rounded_lower, rounded_upper)


def _interval_is_finite(value: _Interval) -> bool:
    return bool(
        np.all(np.isfinite(value.lower))
        and np.all(np.isfinite(value.upper))
        and np.all(value.lower <= value.upper)
    )


def _interval_lerp(lower: _Interval, upper: _Interval, factor: float) -> _Interval:
    # Exact faces matter: they keep inactive corners inactive and preserve the
    # mathematical table-node value even when another corner is very large.
    if factor == 0.0:
        return lower
    if factor == 1.0:
        return upper
    t = _interval_point(factor)
    one_minus_t = _interval_subtract(_interval_point(1.0), t)
    return _interval_add(
        _interval_multiply(lower, one_minus_t),
        _interval_multiply(upper, t),
    )


def _interval_lerp_many(
    lower: _Interval, upper: _Interval, factors: np.ndarray
) -> _Interval:
    """Vectorized interval lerp while preserving exact zero/one faces."""

    factor = np.asarray(factors, dtype=np.float64)
    t = _interval_point(factor)
    one_minus_t = _interval_subtract(_interval_point(1.0), t)
    result = _interval_add(
        _interval_multiply(lower, one_minus_t),
        _interval_multiply(upper, t),
    )
    at_lower = factor == 0.0
    at_upper = factor == 1.0
    return _Interval(
        np.where(at_lower, lower.lower, np.where(at_upper, upper.lower, result.lower)),
        np.where(at_lower, lower.upper, np.where(at_upper, upper.upper, result.upper)),
    )


def _evaluate_interval_tensor(corners: _Interval, point: np.ndarray) -> _Interval:
    """Evaluate a multilinear corner tensor at one exact binary64 point."""

    work = corners
    for factor in np.asarray(point, dtype=float):
        lower = _Interval(work.lower[0, ...], work.upper[0, ...])
        upper = _Interval(work.lower[1, ...], work.upper[1, ...])
        work = _interval_lerp(lower, upper, float(factor))
    return work


def _evaluate_interval_tensor_points(
    corners: _Interval, points: np.ndarray
) -> _Interval:
    """Evaluate one multilinear corner tensor at exact point batches."""

    query = np.asarray(points, dtype=np.float64)
    dimension = corners.lower.ndim - 1
    if query.ndim != 2 or query.shape[1] != dimension:
        raise ValueError(
            "interval tensor points must have shape (n, %d)" % dimension
        )
    count = query.shape[0]
    work = _Interval(
        np.broadcast_to(corners.lower, (count,) + corners.lower.shape),
        np.broadcast_to(corners.upper, (count,) + corners.upper.shape),
    )
    for axis in range(dimension):
        lower = _Interval(work.lower[:, 0, ...], work.upper[:, 0, ...])
        upper = _Interval(work.lower[:, 1, ...], work.upper[:, 1, ...])
        factor_shape = (count,) + (1,) * (lower.lower.ndim - 1)
        work = _interval_lerp_many(
            lower,
            upper,
            query[:, axis].reshape(factor_shape),
        )
    return work


def _evaluate_interval_tensor_point_pairs(
    corners: _Interval, points: np.ndarray
) -> _Interval:
    """Evaluate one independently supplied corner tensor per exact point."""

    query = np.asarray(points, dtype=np.float64)
    dimension = corners.lower.ndim - 2
    if (
        query.ndim != 2
        or query.shape[1] != dimension
        or corners.lower.shape[0] != query.shape[0]
    ):
        raise ValueError(
            "paired interval tensors/points must share shape (n, %d)"
            % dimension
        )
    count = query.shape[0]
    work = corners
    for axis in range(dimension):
        lower = _Interval(work.lower[:, 0, ...], work.upper[:, 0, ...])
        upper = _Interval(work.lower[:, 1, ...], work.upper[:, 1, ...])
        factor_shape = (count,) + (1,) * (lower.lower.ndim - 1)
        work = _interval_lerp_many(
            lower,
            upper,
            query[:, axis].reshape(factor_shape),
        )
    return work


def _evaluate_interval_tensor_point_batches(
    corners: _Interval, points: np.ndarray
) -> _Interval:
    """Evaluate one corner tensor at many exact points for every batch row.

    ``corners`` has shape ``(batch, 2, ..., 2, fields)`` and ``points`` has
    shape ``(batch, point_count, dimension)``.  This is the same outward-
    rounded interpolation kernel as the scalar and paired evaluators, with a
    point axis added so shared dyadic child vertices are evaluated once.
    """

    query = np.asarray(points, dtype=np.float64)
    dimension = corners.lower.ndim - 2
    if (
        query.ndim != 3
        or query.shape[2] != dimension
        or corners.lower.shape[0] != query.shape[0]
    ):
        raise ValueError(
            "batched interval tensors/points must share shape "
            "(batch, point_count, %d)" % dimension
        )
    batch, point_count, _ = query.shape
    expanded_shape = (
        (batch, point_count) + tuple(corners.lower.shape[1:])
    )
    work = _Interval(
        np.broadcast_to(corners.lower[:, None, ...], expanded_shape),
        np.broadcast_to(corners.upper[:, None, ...], expanded_shape),
    )
    for axis in range(dimension):
        lower = _Interval(work.lower[:, :, 0, ...], work.upper[:, :, 0, ...])
        upper = _Interval(work.lower[:, :, 1, ...], work.upper[:, :, 1, ...])
        factor_shape = (batch, point_count) + (1,) * (lower.lower.ndim - 2)
        work = _interval_lerp_many(
            lower,
            upper,
            query[:, :, axis].reshape(factor_shape),
        )
    return work


class _RigorousCellModel:
    """Outward-rounded polynomial model in one cell's unit coordinates."""

    def __init__(
        self,
        interpolator: MultilinearInterpolator,
        cell_indices: np.ndarray,
        target: np.ndarray,
        residual_factors: np.ndarray,
    ) -> None:
        self.interpolator = interpolator
        self.cell_indices = np.asarray(cell_indices, dtype=np.intp)
        self.dimension = interpolator.dimension
        corner_values = np.asarray(
            interpolator.cell_corner_values(self.cell_indices), dtype=np.float64
        )
        tensor_shape = (2,) * self.dimension + (self.dimension,)
        self.corner_values = corner_values.reshape(tensor_shape)
        self.corners = _interval_point(self.corner_values)
        self.target = np.asarray(target, dtype=np.float64)
        self.residual_factors = np.asarray(residual_factors, dtype=np.float64)
        lower, upper = interpolator.cell_coordinate_bounds(self.cell_indices)
        self.physical_lower = np.asarray(lower, dtype=float)
        self.physical_upper = np.asarray(upper, dtype=float)

    def value_at(self, point: np.ndarray) -> _Interval:
        return _evaluate_interval_tensor(self.corners, point)

    def scaled_residual_at(self, point: np.ndarray) -> _Interval:
        residual = _interval_subtract(self.value_at(point), _interval_point(self.target))
        return _interval_multiply(
            residual, _interval_point(self.residual_factors)
        )

    def value_enclosure(self, lower: np.ndarray, upper: np.ndarray) -> _Interval:
        values = []
        for offsets in itertools.product((0, 1), repeat=self.dimension):
            point = np.asarray(
                [upper[j] if offsets[j] else lower[j] for j in range(self.dimension)],
                dtype=float,
            )
            values.append(self.value_at(point))
        return _Interval(
            np.min(np.stack([value.lower for value in values]), axis=0),
            np.max(np.stack([value.upper for value in values]), axis=0),
        )

    def value_enclosures(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> _Interval:
        """Vectorized outward-safe enclosures for boxes in this cell."""

        lower_array = np.asarray(lower, dtype=np.float64)
        upper_array = np.asarray(upper, dtype=np.float64)
        if (
            lower_array.ndim != 2
            or lower_array.shape != upper_array.shape
            or lower_array.shape[1] != self.dimension
        ):
            raise ValueError(
                "batched unit bounds must both have shape (n, %d)"
                % self.dimension
            )
        count = lower_array.shape[0]
        if count == 0:
            empty = np.empty((0, self.dimension), dtype=float)
            return _Interval(empty, empty.copy())
        offsets = np.asarray(
            tuple(itertools.product((0, 1), repeat=self.dimension)),
            dtype=bool,
        )
        points = np.where(
            offsets[None, :, :],
            upper_array[:, None, :],
            lower_array[:, None, :],
        )
        values = _evaluate_interval_tensor_points(
            self.corners,
            points.reshape((-1, self.dimension)),
        )
        vertex_count = 1 << self.dimension
        value_lower = values.lower.reshape(
            (count, vertex_count, self.dimension)
        )
        value_upper = values.upper.reshape(
            (count, vertex_count, self.dimension)
        )
        return _Interval(
            np.min(value_lower, axis=1),
            np.max(value_upper, axis=1),
        )

    @staticmethod
    def _corners_for_cells(
        interpolator: MultilinearInterpolator,
        cell_indices: np.ndarray,
    ) -> _Interval:
        indices = np.asarray(cell_indices)
        dimension = interpolator.dimension
        if indices.ndim != 2 or indices.shape[1] != dimension:
            raise ValueError(
                "cell indices must have shape (n, %d)" % dimension
            )
        corner_values = np.asarray(
            interpolator.cell_corner_values(indices), dtype=np.float64
        )
        return _interval_point(
            corner_values.reshape(
                (indices.shape[0],) + (2,) * dimension + (dimension,)
            )
        )

    @staticmethod
    def values_at_for_cells(
        interpolator: MultilinearInterpolator,
        cell_indices: np.ndarray,
        points: np.ndarray,
    ) -> _Interval:
        """Vectorized proof-kernel evaluation for cell/point pairs."""

        corners = _RigorousCellModel._corners_for_cells(
            interpolator, cell_indices
        )
        return _evaluate_interval_tensor_point_pairs(corners, points)

    @staticmethod
    def value_enclosures_for_cells(
        interpolator: MultilinearInterpolator,
        cell_indices: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> _Interval:
        """Vectorized proof kernel for boxes owned by different cells."""

        indices = np.asarray(cell_indices)
        lower_array = np.asarray(lower, dtype=np.float64)
        upper_array = np.asarray(upper, dtype=np.float64)
        dimension = interpolator.dimension
        if (
            indices.ndim != 2
            or indices.shape[1] != dimension
            or lower_array.shape != indices.shape
            or upper_array.shape != indices.shape
        ):
            raise ValueError(
                "cell indices and unit bounds must all have shape (n, %d)"
                % dimension
            )
        count = indices.shape[0]
        if count == 0:
            empty = np.empty((0, dimension), dtype=float)
            return _Interval(empty, empty.copy())
        corners = _RigorousCellModel._corners_for_cells(
            interpolator, indices
        )
        offsets = tuple(itertools.product((0, 1), repeat=dimension))
        value_lowers = []
        value_uppers = []
        for offset in offsets:
            selected = np.asarray(offset, dtype=bool)
            points = np.where(selected[None, :], upper_array, lower_array)
            values = _evaluate_interval_tensor_point_pairs(corners, points)
            value_lowers.append(values.lower)
            value_uppers.append(values.upper)
        return _Interval(
            np.min(np.stack(value_lowers, axis=1), axis=1),
            np.max(np.stack(value_uppers, axis=1), axis=1),
        )

    @staticmethod
    def dyadic_child_value_enclosures_for_cells(
        interpolator: MultilinearInterpolator,
        cell_indices: np.ndarray,
        parent_lower: np.ndarray,
        parent_upper: np.ndarray,
    ) -> _Interval:
        """Evaluate all children from their parents' shared three-point grid.

        Bisecting every source axis gives ``2**d`` children whose corners are
        the ``3**d`` Cartesian combinations of parent lower/mid/upper values.
        Evaluating that shared lattice once is proof-equivalent to evaluating
        the ``4**d`` child-corner occurrences independently, while avoiding
        the repeated sibling work that dominated the Phase-6 benchmark.
        """

        indices = np.asarray(cell_indices)
        lower_array = np.asarray(parent_lower, dtype=np.float64)
        upper_array = np.asarray(parent_upper, dtype=np.float64)
        dimension = interpolator.dimension
        if (
            indices.ndim != 2
            or indices.shape[1] != dimension
            or lower_array.shape != indices.shape
            or upper_array.shape != indices.shape
        ):
            raise ValueError(
                "parent cell indices and unit bounds must all have shape "
                "(n, %d)" % dimension
            )
        count = indices.shape[0]
        child_count = 1 << dimension
        if count == 0:
            empty = np.empty((0, dimension), dtype=float)
            return _Interval(empty, empty.copy())
        grid_offsets = np.asarray(
            tuple(itertools.product((0.0, 0.5, 1.0), repeat=dimension)),
            dtype=np.float64,
        )
        points = lower_array[:, None, :] + (
            upper_array - lower_array
        )[:, None, :] * grid_offsets[None, :, :]
        corners = _RigorousCellModel._corners_for_cells(interpolator, indices)
        values = _evaluate_interval_tensor_point_batches(corners, points)

        child_offsets = np.asarray(
            tuple(itertools.product((0, 1), repeat=dimension)),
            dtype=np.int64,
        )
        vertex_offsets = child_offsets
        grid_coordinates = (
            child_offsets[:, None, :] + vertex_offsets[None, :, :]
        )
        grid_indices = np.ravel_multi_index(
            grid_coordinates.reshape((-1, dimension)).T,
            (3,) * dimension,
            order="C",
        ).reshape((child_count, child_count))
        child_lower = np.min(
            np.take(values.lower, grid_indices, axis=1), axis=2
        )
        child_upper = np.max(
            np.take(values.upper, grid_indices, axis=1), axis=2
        )
        midpoint = lower_array / 2.0 + upper_array / 2.0
        child_unit_lower = np.where(
            child_offsets[None, :, :] == 0,
            lower_array[:, None, :],
            midpoint[:, None, :],
        ).reshape((count * child_count, dimension))
        flat_cells = np.ravel_multi_index(
            indices.T, interpolator.cell_shape, order="C"
        )
        repeated_cells = np.repeat(flat_cells, child_count)
        keys = tuple(
            [
                child_unit_lower[:, axis]
                for axis in range(dimension - 1, -1, -1)
            ]
            + [repeated_cells]
        )
        canonical_order = np.lexsort(keys)
        return _Interval(
            child_lower.reshape((count * child_count, dimension))[
                canonical_order
            ],
            child_upper.reshape((count * child_count, dimension))[
                canonical_order
            ],
        )

    @staticmethod
    def scaled_jacobian_enclosures_for_cells(
        interpolator: MultilinearInterpolator,
        cell_indices: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        residual_factors: np.ndarray,
    ) -> _Interval:
        """Vectorized scaled Jacobian enclosures for cell/box rows."""

        indices = np.asarray(cell_indices)
        lower_array = np.asarray(lower, dtype=np.float64)
        upper_array = np.asarray(upper, dtype=np.float64)
        factors = np.asarray(residual_factors, dtype=np.float64)
        dimension = interpolator.dimension
        if (
            indices.ndim != 2
            or indices.shape[1] != dimension
            or lower_array.shape != indices.shape
            or upper_array.shape != indices.shape
            or factors.shape != (dimension,)
        ):
            raise ValueError("invalid batched Jacobian-enclosure inputs")
        count = indices.shape[0]
        corners = _RigorousCellModel._corners_for_cells(
            interpolator, indices
        )
        result_lower = np.empty((count, dimension, dimension), dtype=float)
        result_upper = np.empty_like(result_lower)
        for axis in range(dimension):
            low_face = _Interval(
                np.take(corners.lower, 0, axis=axis + 1),
                np.take(corners.upper, 0, axis=axis + 1),
            )
            high_face = _Interval(
                np.take(corners.lower, 1, axis=axis + 1),
                np.take(corners.upper, 1, axis=axis + 1),
            )
            differences = _interval_subtract(high_face, low_face)
            other_axes = [item for item in range(dimension) if item != axis]
            derivative_lowers = []
            derivative_uppers = []
            for offset in itertools.product((0, 1), repeat=len(other_axes)):
                if other_axes:
                    points = np.asarray(
                        [
                            np.where(
                                bool(offset[position]),
                                upper_array[:, source_axis],
                                lower_array[:, source_axis],
                            )
                            for position, source_axis in enumerate(other_axes)
                        ],
                        dtype=float,
                    ).T
                else:
                    points = np.empty((count, 0), dtype=float)
                values = _evaluate_interval_tensor_point_pairs(
                    differences, points
                )
                derivative_lowers.append(values.lower)
                derivative_uppers.append(values.upper)
            derivative = _Interval(
                np.min(np.stack(derivative_lowers, axis=1), axis=1),
                np.max(np.stack(derivative_uppers, axis=1), axis=1),
            )
            scaled = _interval_multiply(
                derivative,
                _interval_point(factors),
            )
            result_lower[:, :, axis] = scaled.lower
            result_upper[:, :, axis] = scaled.upper
        return _Interval(result_lower, result_upper)

    def scaled_jacobian_enclosure(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> _Interval:
        result_lower = np.empty((self.dimension, self.dimension), dtype=float)
        result_upper = np.empty_like(result_lower)
        for axis in range(self.dimension):
            low_face = _Interval(
                np.take(self.corners.lower, 0, axis=axis),
                np.take(self.corners.upper, 0, axis=axis),
            )
            high_face = _Interval(
                np.take(self.corners.lower, 1, axis=axis),
                np.take(self.corners.upper, 1, axis=axis),
            )
            edge_differences = _interval_subtract(high_face, low_face)
            other_axes = [index for index in range(self.dimension) if index != axis]
            values = []
            for offsets in itertools.product((0, 1), repeat=len(other_axes)):
                point = np.asarray(
                    [
                        upper[index] if offsets[position] else lower[index]
                        for position, index in enumerate(other_axes)
                    ],
                    dtype=float,
                )
                values.append(_evaluate_interval_tensor(edge_differences, point))
            derivative = _Interval(
                np.min(np.stack([value.lower for value in values]), axis=0),
                np.max(np.stack([value.upper for value in values]), axis=0),
            )
            scaled = _interval_multiply(
                derivative, _interval_point(self.residual_factors)
            )
            result_lower[:, axis] = scaled.lower
            result_upper[:, axis] = scaled.upper
        return _Interval(result_lower, result_upper)

    def physical_box(
        self, unit_lower: np.ndarray, unit_upper: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        widths = _interval_subtract(
            _interval_point(self.physical_upper),
            _interval_point(self.physical_lower),
        )
        mapped = _interval_add(
            _interval_point(self.physical_lower),
            _interval_multiply(
                widths, _Interval(np.asarray(unit_lower), np.asarray(unit_upper))
            ),
        )
        # Intersect with the owning cell.  Outward rounding in the affine map
        # may otherwise extend one ulp beyond an exact face.
        return (
            np.maximum(mapped.lower, self.physical_lower),
            np.minimum(mapped.upper, self.physical_upper),
        )


class _RigorousCellModelCache:
    """One target/problem cache of rigorous original-cell models."""

    def __init__(
        self,
        interpolator: MultilinearInterpolator,
        target: np.ndarray,
        residual_factors: np.ndarray,
    ) -> None:
        self.interpolator = interpolator
        self.target = np.asarray(target, dtype=np.float64)
        self.residual_factors = np.asarray(residual_factors, dtype=np.float64)
        self.dimension = interpolator.dimension
        self.cell_count = int(np.prod(interpolator.cell_shape, dtype=np.int64))
        self._models = {}
        self.construction_count = 0

    def for_flat_id(self, flat_cell_id: int) -> _RigorousCellModel:
        cell_id = int(flat_cell_id)
        if cell_id < 0 or cell_id >= self.cell_count:
            raise SolverInputError(
                "original-cell ID %d lies outside the interpolation grid"
                % cell_id
            )
        model = self._models.get(cell_id)
        if model is None:
            indices = np.asarray(
                np.unravel_index(cell_id, self.interpolator.cell_shape, order="C"),
                dtype=np.intp,
            )
            model = _RigorousCellModel(
                self.interpolator,
                indices,
                self.target,
                self.residual_factors,
            )
            self._models[cell_id] = model
            self.construction_count += 1
        return model


class _RigorousSubcellEnclosureEvaluator:
    """Evaluate subcell routing boxes with the existing proof kernel.

    The target-local topology lives in :mod:`eos_trensmuther.subcells`; this
    adapter intentionally keeps all multilinear interval arithmetic in
    ``_RigorousCellModel``.  Models are cached by original C-order source-cell
    ID so descendants of one cell reuse the same corner tensor.
    """

    def __init__(
        self,
        interpolator: MultilinearInterpolator,
        target: np.ndarray,
        residual_scales: np.ndarray,
        root_residual_tolerance: float,
        *,
        model_cache: Optional[_RigorousCellModelCache] = None,
    ) -> None:
        self.interpolator = interpolator
        self.dimension = interpolator.dimension
        self.target = np.asarray(target, dtype=np.float64)
        self.residual_scales = np.asarray(residual_scales, dtype=np.float64)
        self.root_residual_tolerance = float(root_residual_tolerance)
        if (
            self.target.shape != (self.dimension,)
            or self.residual_scales.shape != (self.dimension,)
            or not np.all(np.isfinite(self.target))
            or not np.all(np.isfinite(self.residual_scales))
            or np.any(self.residual_scales <= 0.0)
            or not math.isfinite(self.root_residual_tolerance)
            or self.root_residual_tolerance <= 0.0
        ):
            raise SolverConfigurationError(
                "subcell proof evaluation requires finite target/scales and a "
                "positive residual tolerance"
            )
        self._residual_factors = 1.0 / self.residual_scales
        if not np.all(np.isfinite(self._residual_factors)):
            raise SolverConfigurationError(
                "subcell residual scales must have finite reciprocals"
            )
        if model_cache is None:
            model_cache = _RigorousCellModelCache(
                interpolator,
                self.target,
                self._residual_factors,
            )
        elif (
            not isinstance(model_cache, _RigorousCellModelCache)
            or model_cache.interpolator is not interpolator
            or not np.array_equal(model_cache.target, self.target)
            or not np.array_equal(
                model_cache.residual_factors, self._residual_factors
            )
        ):
            raise SolverConfigurationError(
                "subcell model cache does not match the evaluator problem"
            )
        self.model_cache = model_cache

    @property
    def cached_model_count(self) -> int:
        return len(self.model_cache._models)

    def _model(self, flat_cell_id: int) -> _RigorousCellModel:
        return self.model_cache.for_flat_id(flat_cell_id)

    def _routing_enclosures(self, enclosure: _Interval) -> SubcellEnclosures:
        exact_lower = np.asarray(enclosure.lower, dtype=float)
        exact_upper = np.asarray(enclosure.upper, dtype=float)
        if (
            exact_lower.ndim != 2
            or exact_lower.shape != exact_upper.shape
            or exact_lower.shape[1] != self.dimension
        ):
            raise SolverInvariantError(
                "subcell proof kernel returned invalid enclosure arrays"
            )
        defined = np.all(
            np.isfinite(exact_lower) & np.isfinite(exact_upper), axis=1
        ) & np.all(exact_lower <= exact_upper, axis=1)
        routing_lower = np.full_like(exact_lower, np.nan)
        routing_upper = np.full_like(exact_upper, np.nan)
        if np.any(defined):
            # Import lazily because reachability imports the pipeline, which in
            # turn imports this solver module.  Reusing this helper preserves
            # the exact production tolerance expansion and outward rounding.
            from .reachability import _expanded_target_boxes

            _, expanded_lower, expanded_upper = _expanded_target_boxes(
                exact_lower[defined],
                exact_upper[defined],
                self.residual_scales,
                self.root_residual_tolerance,
            )
            routing_lower[defined] = expanded_lower
            routing_upper[defined] = expanded_upper
        return SubcellEnclosures(routing_lower, routing_upper, defined)

    def __call__(self, boxes: SubcellBoxes) -> SubcellEnclosures:
        if not isinstance(boxes, SubcellBoxes):
            raise TypeError("boxes must be SubcellBoxes")
        if boxes.dimension != self.dimension:
            raise SolverInputError(
                "subcell dimension does not match the interpolator"
            )
        indices = np.column_stack(
            np.unravel_index(
                boxes.cell_flat_indices,
                self.interpolator.cell_shape,
                order="C",
            )
        ).astype(np.intp, copy=False)
        enclosure = _RigorousCellModel.value_enclosures_for_cells(
            self.interpolator,
            indices,
            boxes.unit_lower,
            boxes.unit_upper,
        )
        return self._routing_enclosures(enclosure)

    def evaluate_dyadic_children(
        self, parents: SubcellBoxes, children: SubcellBoxes
    ) -> SubcellEnclosures:
        """Evaluate sibling children through one shared dyadic vertex grid."""

        if not isinstance(parents, SubcellBoxes) or not isinstance(
            children, SubcellBoxes
        ):
            raise TypeError("parents and children must be SubcellBoxes")
        expected_children = parents.count * (1 << self.dimension)
        if (
            parents.dimension != self.dimension
            or children.dimension != self.dimension
            or children.depth != parents.depth + 1
            or children.count != expected_children
        ):
            raise SolverInputError("dyadic parent/child topology is inconsistent")
        indices = np.column_stack(
            np.unravel_index(
                parents.cell_flat_indices,
                self.interpolator.cell_shape,
                order="C",
            )
        ).astype(np.intp, copy=False)
        enclosure = (
            _RigorousCellModel.dyadic_child_value_enclosures_for_cells(
                self.interpolator,
                indices,
                parents.unit_lower,
                parents.unit_upper,
            )
        )
        return self._routing_enclosures(enclosure)

    def physical_bounds(
        self, boxes: SubcellBoxes
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map normalized subcells outward-safely into source coordinates."""

        if boxes.dimension != self.dimension:
            raise SolverInputError(
                "subcell dimension does not match the interpolator"
            )
        lower = np.empty((boxes.count, self.dimension), dtype=float)
        upper = np.empty_like(lower)
        for row, flat_cell_id in enumerate(boxes.cell_flat_indices):
            lower[row], upper[row] = self._model(int(flat_cell_id)).physical_box(
                boxes.unit_lower[row], boxes.unit_upper[row]
            )
        return lower, upper


class _CellRefinementProblem:
    """Numerical root problem in one certified cell's unit coordinates."""

    def __init__(
        self,
        interpolator: MultilinearInterpolator,
        cell_indices: np.ndarray,
        target: np.ndarray,
        residual_scales: np.ndarray,
    ) -> None:
        self.dimension = interpolator.dimension
        self.target = np.asarray(target, dtype=float)
        self.residual_scales = np.asarray(residual_scales, dtype=float)
        self.residual_factors = 1.0 / self.residual_scales
        lower, upper = interpolator.cell_coordinate_bounds(cell_indices)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.span = self.upper - self.lower
        corners = np.asarray(interpolator.cell_corner_values(cell_indices), dtype=float)
        self.corners = corners.reshape((2,) * self.dimension + (self.dimension,))

    @staticmethod
    def _evaluate_tensor(corners: np.ndarray, point: np.ndarray) -> np.ndarray:
        work = np.asarray(corners, dtype=float)
        for factor in np.asarray(point, dtype=float):
            lower = work[0, ...]
            upper = work[1, ...]
            if factor == 0.0:
                work = lower
            elif factor == 1.0:
                work = upper
            else:
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    work = lower * (1.0 - factor) + upper * factor
        return np.asarray(work, dtype=float)

    def to_physical(self, normalized: np.ndarray) -> np.ndarray:
        unit = np.asarray(normalized, dtype=float)
        physical = self.lower + self.span * unit
        physical = np.minimum(np.maximum(physical, self.lower), self.upper)
        physical = np.where(unit == 0.0, self.lower, physical)
        return np.where(unit == 1.0, self.upper, physical)

    def values_and_residual(self, normalized: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values = self._evaluate_tensor(self.corners, normalized)
        return values, (values - self.target) * self.residual_factors

    def newton_jacobians(self, normalized: np.ndarray) -> Tuple[np.ndarray, ...]:
        jacobian = np.empty((self.dimension, self.dimension), dtype=float)
        for axis in range(self.dimension):
            low_face = np.take(self.corners, 0, axis=axis)
            high_face = np.take(self.corners, 1, axis=axis)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                edge_differences = high_face - low_face
            other_point = np.delete(np.asarray(normalized, dtype=float), axis)
            derivative = self._evaluate_tensor(edge_differences, other_point)
            jacobian[:, axis] = self.residual_factors * derivative
        return (jacobian,)

    @property
    def normalized_coordinate_system(self) -> str:
        return "interpolation_cell"


def _constant_matrix_times_interval(
    matrix: np.ndarray, interval: _Interval
) -> _Interval:
    rows, shared = matrix.shape
    if interval.lower.shape[0] != shared:
        raise ValueError("interval matrix dimensions do not align")
    trailing = interval.lower.shape[1:]
    lower = np.empty((rows,) + trailing, dtype=float)
    upper = np.empty_like(lower)
    for row in range(rows):
        total = _interval_point(np.zeros(trailing, dtype=float))
        for index in range(shared):
            term = _interval_multiply(
                _interval_point(matrix[row, index]),
                _Interval(interval.lower[index], interval.upper[index]),
            )
            total = _interval_add(total, term)
        lower[row] = total.lower
        upper[row] = total.upper
    return _Interval(lower, upper)


def _interval_matrix_times_vector(matrix: _Interval, vector: _Interval) -> _Interval:
    rows, shared = matrix.lower.shape
    if vector.lower.shape != (shared,):
        raise ValueError("interval matrix and vector dimensions do not align")
    lower = np.empty(rows, dtype=float)
    upper = np.empty(rows, dtype=float)
    for row in range(rows):
        total = _interval_point(0.0)
        for index in range(shared):
            term = _interval_multiply(
                _Interval(matrix.lower[row, index], matrix.upper[row, index]),
                _Interval(vector.lower[index], vector.upper[index]),
            )
            total = _interval_add(total, term)
        lower[row] = total.lower
        upper[row] = total.upper
    return _Interval(lower, upper)


def _exactly_full_rank(matrix: np.ndarray) -> bool:
    """Prove that a small binary64 matrix is nonsingular over the rationals."""

    values = [
        [Fraction.from_float(float(entry)) for entry in row]
        for row in np.asarray(matrix, dtype=float)
    ]
    size = len(values)
    pivot_row = 0
    for column in range(size):
        pivot = next(
            (row for row in range(pivot_row, size) if values[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        values[pivot_row], values[pivot] = values[pivot], values[pivot_row]
        pivot_value = values[pivot_row][column]
        for row in range(pivot_row + 1, size):
            if values[row][column] == 0:
                continue
            factor = values[row][column] / pivot_value
            for remaining in range(column, size):
                values[row][remaining] -= factor * values[pivot_row][remaining]
        pivot_row += 1
    return pivot_row == size


@dataclass(frozen=True)
class _KrawczykStep:
    kind: str
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None


def _krawczyk_step(
    model: _RigorousCellModel, lower: np.ndarray, upper: np.ndarray
) -> _KrawczykStep:
    value_range = model.value_enclosure(lower, upper)
    residual_range = _interval_subtract(value_range, _interval_point(model.target))
    if not _interval_is_finite(residual_range):
        return _KrawczykStep("inconclusive")
    if np.any((residual_range.lower > 0.0) | (residual_range.upper < 0.0)):
        return _KrawczykStep("excluded")

    center = _bounded_midpoint(lower, upper)
    if np.any(center < lower) or np.any(center > upper):
        return _KrawczykStep("inconclusive")
    jacobian = model.scaled_jacobian_enclosure(lower, upper)
    center_residual = model.scaled_residual_at(center)
    if not _interval_is_finite(jacobian) or not _interval_is_finite(center_residual):
        return _KrawczykStep("inconclusive")
    midpoint_jacobian = jacobian.lower / 2.0 + jacobian.upper / 2.0
    if not np.all(np.isfinite(midpoint_jacobian)):
        return _KrawczykStep("inconclusive")
    try:
        preconditioner = np.linalg.inv(midpoint_jacobian)
    except np.linalg.LinAlgError:
        return _KrawczykStep("inconclusive")
    if not np.all(np.isfinite(preconditioner)):
        return _KrawczykStep("inconclusive")

    predicted = _interval_subtract(
        _interval_point(center),
        _constant_matrix_times_interval(preconditioner, center_residual),
    )
    product = _constant_matrix_times_interval(preconditioner, jacobian)
    identity_minus = _interval_subtract(
        _interval_point(np.eye(model.dimension, dtype=float)), product
    )
    displacement = _interval_subtract(
        _Interval(lower, upper), _interval_point(center)
    )
    correction = _interval_matrix_times_vector(identity_minus, displacement)
    krawczyk = _interval_add(predicted, correction)
    if not _interval_is_finite(krawczyk):
        return _KrawczykStep("inconclusive")

    intersection_lower = np.maximum(lower, krawczyk.lower)
    intersection_upper = np.minimum(upper, krawczyk.upper)
    if np.any(intersection_lower > intersection_upper):
        return _KrawczykStep("excluded")
    if np.all(krawczyk.lower > lower) and np.all(krawczyk.upper < upper):
        if _exactly_full_rank(preconditioner):
            return _KrawczykStep(
                "certified", intersection_lower, intersection_upper
            )
        return _KrawczykStep("inconclusive")
    return _KrawczykStep("contract", intersection_lower, intersection_upper)


def _interval_rows_finite(value: _Interval) -> np.ndarray:
    axes = tuple(range(1, value.lower.ndim))
    return (
        np.all(np.isfinite(value.lower), axis=axes)
        & np.all(np.isfinite(value.upper), axis=axes)
        & np.all(value.lower <= value.upper, axis=axes)
    )


def _constant_matrices_times_interval_vectors(
    matrices: np.ndarray, vectors: _Interval
) -> _Interval:
    products = _interval_multiply(
        _interval_point(np.asarray(matrices)),
        _Interval(vectors.lower[:, None, :], vectors.upper[:, None, :]),
    )
    count, rows, shared = products.lower.shape
    total = _interval_point(np.zeros((count, rows), dtype=float))
    for index in range(shared):
        total = _interval_add(
            total,
            _Interval(
                products.lower[:, :, index],
                products.upper[:, :, index],
            ),
        )
    return total


def _constant_matrices_times_interval_matrices(
    matrices: np.ndarray, intervals: _Interval
) -> _Interval:
    exact = np.asarray(matrices, dtype=float)
    products = _interval_multiply(
        _interval_point(exact[:, :, :, None]),
        _Interval(
            intervals.lower[:, None, :, :],
            intervals.upper[:, None, :, :],
        ),
    )
    count, rows, shared, columns = products.lower.shape
    total = _interval_point(np.zeros((count, rows, columns), dtype=float))
    for index in range(shared):
        total = _interval_add(
            total,
            _Interval(
                products.lower[:, :, index, :],
                products.upper[:, :, index, :],
            ),
        )
    return total


def _interval_matrices_times_interval_vectors(
    matrices: _Interval, vectors: _Interval
) -> _Interval:
    products = _interval_multiply(
        matrices,
        _Interval(vectors.lower[:, None, :], vectors.upper[:, None, :]),
    )
    count, rows, shared = products.lower.shape
    total = _interval_point(np.zeros((count, rows), dtype=float))
    for index in range(shared):
        total = _interval_add(
            total,
            _Interval(
                products.lower[:, :, index],
                products.upper[:, :, index],
            ),
        )
    return total


_BATCH_KRAWCZYK_EXCLUDED = np.int8(0)
_BATCH_KRAWCZYK_CERTIFIED = np.int8(1)
_BATCH_KRAWCZYK_CONTRACT = np.int8(2)
_BATCH_KRAWCZYK_INCONCLUSIVE = np.int8(3)


@dataclass(frozen=True)
class _KrawczykBatchSteps:
    kinds: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def _krawczyk_steps_for_cells(
    interpolator: MultilinearInterpolator,
    cell_indices: np.ndarray,
    target: np.ndarray,
    residual_factors: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> _KrawczykBatchSteps:
    """Vectorized first Krawczyk evaluation for independent cell boxes."""

    indices = np.asarray(cell_indices)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    query_target = np.asarray(target, dtype=float)
    factors = np.asarray(residual_factors, dtype=float)
    dimension = interpolator.dimension
    if (
        indices.ndim != 2
        or indices.shape[1] != dimension
        or lower_array.shape != indices.shape
        or upper_array.shape != indices.shape
        or query_target.shape != (dimension,)
        or factors.shape != (dimension,)
    ):
        raise ValueError("invalid batched Krawczyk inputs")
    count = indices.shape[0]
    kinds = np.full(count, _BATCH_KRAWCZYK_INCONCLUSIVE, dtype=np.int8)
    result_lower = np.full((count, dimension), np.nan, dtype=float)
    result_upper = np.full_like(result_lower, np.nan)
    if count == 0:
        return _KrawczykBatchSteps(kinds, result_lower, result_upper)

    value_range = _RigorousCellModel.value_enclosures_for_cells(
        interpolator, indices, lower_array, upper_array
    )
    residual_range = _interval_subtract(
        value_range, _interval_point(query_target)
    )
    finite_residual = _interval_rows_finite(residual_range)
    excluded = finite_residual & np.any(
        (residual_range.lower > 0.0) | (residual_range.upper < 0.0),
        axis=1,
    )
    kinds[excluded] = _BATCH_KRAWCZYK_EXCLUDED
    active = np.flatnonzero(finite_residual & ~excluded)
    if active.size == 0:
        return _KrawczykBatchSteps(kinds, result_lower, result_upper)

    active_indices = indices[active]
    active_lower = lower_array[active]
    active_upper = upper_array[active]
    center = _bounded_midpoint(active_lower, active_upper)
    jacobian = _RigorousCellModel.scaled_jacobian_enclosures_for_cells(
        interpolator,
        active_indices,
        active_lower,
        active_upper,
        factors,
    )
    center_values = _RigorousCellModel.values_at_for_cells(
        interpolator, active_indices, center
    )
    center_residual = _interval_multiply(
        _interval_subtract(center_values, _interval_point(query_target)),
        _interval_point(factors),
    )
    usable = _interval_rows_finite(jacobian) & _interval_rows_finite(
        center_residual
    )
    midpoint_jacobian = jacobian.lower / 2.0 + jacobian.upper / 2.0
    usable &= np.all(np.isfinite(midpoint_jacobian), axis=(1, 2))
    usable_positions = np.flatnonzero(usable)
    if usable_positions.size == 0:
        return _KrawczykBatchSteps(kinds, result_lower, result_upper)

    inverse_rows = []
    inverse_positions = []
    for position in usable_positions:
        try:
            inverse = np.linalg.inv(midpoint_jacobian[position])
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(inverse)):
            inverse_positions.append(int(position))
            inverse_rows.append(inverse)
    if not inverse_positions:
        return _KrawczykBatchSteps(kinds, result_lower, result_upper)

    local = np.asarray(inverse_positions, dtype=np.int64)
    preconditioner = np.stack(inverse_rows)
    selected_center = center[local]
    selected_lower = active_lower[local]
    selected_upper = active_upper[local]
    predicted = _interval_subtract(
        _interval_point(selected_center),
        _constant_matrices_times_interval_vectors(
            preconditioner,
            _Interval(
                center_residual.lower[local], center_residual.upper[local]
            ),
        ),
    )
    product = _constant_matrices_times_interval_matrices(
        preconditioner,
        _Interval(jacobian.lower[local], jacobian.upper[local]),
    )
    identity_minus = _interval_subtract(
        _interval_point(
            np.broadcast_to(
                np.eye(dimension, dtype=float),
                (local.size, dimension, dimension),
            )
        ),
        product,
    )
    displacement = _interval_subtract(
        _Interval(selected_lower, selected_upper),
        _interval_point(selected_center),
    )
    correction = _interval_matrices_times_interval_vectors(
        identity_minus, displacement
    )
    krawczyk = _interval_add(predicted, correction)
    finite_krawczyk = _interval_rows_finite(krawczyk)
    intersection_lower = np.maximum(selected_lower, krawczyk.lower)
    intersection_upper = np.minimum(selected_upper, krawczyk.upper)
    empty = finite_krawczyk & np.any(
        intersection_lower > intersection_upper, axis=1
    )
    global_rows = active[local]
    kinds[global_rows[empty]] = _BATCH_KRAWCZYK_EXCLUDED

    nonempty = finite_krawczyk & ~empty
    strict = nonempty & np.all(krawczyk.lower > selected_lower, axis=1) & np.all(
        krawczyk.upper < selected_upper, axis=1
    )
    for position in np.flatnonzero(strict):
        global_row = global_rows[position]
        if _exactly_full_rank(preconditioner[position]):
            kinds[global_row] = _BATCH_KRAWCZYK_CERTIFIED
            result_lower[global_row] = intersection_lower[position]
            result_upper[global_row] = intersection_upper[position]
    contract = nonempty & ~strict
    contract_rows = global_rows[contract]
    kinds[contract_rows] = _BATCH_KRAWCZYK_CONTRACT
    result_lower[contract_rows] = intersection_lower[contract]
    result_upper[contract_rows] = intersection_upper[contract]
    return _KrawczykBatchSteps(kinds, result_lower, result_upper)


@dataclass(frozen=True)
class _FallbackOutcome:
    report: FallbackReport
    certificate: Optional[CertifiedRootBox]
    box_search: Optional["_KrawczykBoxSearchReport"] = None


@dataclass(frozen=True)
class _KrawczykBoxSearchReport:
    """Internal accounting for one shared arbitrary-box Krawczyk context."""

    initial_boxes: int
    one_shot_boxes_processed: int
    one_shot_boxes_excluded: int
    one_shot_boxes_certified: int
    recursive_boxes_processed: int
    boxes_excluded: int
    boxes_contracted: int
    boxes_subdivided: int
    unresolved_boxes: int
    unprocessed_initial_boxes: int
    queued_boxes_remaining: int
    maximum_depth_reached: int
    cell_models_constructed: int
    cell_models_total: int

    @property
    def boxes_processed(self) -> int:
        return self.one_shot_boxes_processed + self.recursive_boxes_processed

    @property
    def complete_no_root(self) -> bool:
        return bool(
            self.one_shot_boxes_certified == 0
            and self.unresolved_boxes == 0
            and self.unprocessed_initial_boxes == 0
            and self.queued_boxes_remaining == 0
            and self.one_shot_boxes_processed == self.initial_boxes
        )


class EoSRootSolver:
    """Invert the canonical target fields of a multilinear EoS table.

    The object is stateless between individual :meth:`solve` calls.  Use
    :meth:`solve_path` for deterministic continuation: only a converged point
    becomes the next target's initial guess, and a failure resets continuation.

    Parameters
    ----------
    interpolator:
        A target-only :class:`MultilinearInterpolator`.  Its selected fields
        must exactly equal ``target_names`` in canonical order, producing a
        square system.
    config:
        Solver tolerances and resource limits.
    residual_scales:
        Optional fixed positive scales, either one per target in canonical
        order or a mapping keyed by every target name. When omitted, one
        deterministic scale per component is derived from the global node
        extrema and reused for every target.
    """

    def __init__(
        self,
        interpolator: MultilinearInterpolator,
        config: Optional[SolverConfig] = None,
        *,
        residual_scales: Optional[Any] = None,
    ) -> None:
        if not isinstance(interpolator, MultilinearInterpolator):
            raise TypeError("interpolator must be a MultilinearInterpolator")
        if interpolator.field_names != interpolator.target_names:
            raise SolverConfigurationError(
                "the solver requires exactly the canonical target fields in "
                "canonical order; auxiliary fields need a separate interpolator"
            )
        if len(interpolator.field_names) != interpolator.dimension:
            raise SolverConfigurationError(
                "inversion requires a square system with one target per coordinate"
            )
        if not np.all(interpolator.node_value_bounds.defined):
            raise SolverConfigurationError(
                "canonical target nodes must all be finite for inversion"
            )
        self.interpolator = interpolator
        self.config = SolverConfig() if config is None else config
        if not isinstance(self.config, SolverConfig):
            raise TypeError("config must be a SolverConfig")
        self._interval_certification_available = (
            _supports_rigorous_binary64_intervals()
        )
        self._fixed_residual_scales = self._parse_residual_scales(residual_scales)
        bounds = self.interpolator.node_value_bounds
        try:
            self._automatic_residual_scales = (
                _automatic_residual_scales_from_bounds(
                    bounds.lower, bounds.upper
                )
            )
        except ValueError as exc:
            raise SolverConfigurationError(str(exc)) from exc

    @property
    def interval_certification_available(self) -> bool:
        """Whether this runtime passed the binary64 proof-backend checks."""

        return self._interval_certification_available

    def _parse_residual_scales(self, value: Optional[Any]) -> Optional[np.ndarray]:
        if value is None:
            return None
        names = self.interpolator.target_names
        if isinstance(value, Mapping):
            missing = [name for name in names if name not in value]
            extra = [name for name in value if name not in names]
            if missing or extra:
                raise SolverConfigurationError(
                    "residual-scale mapping must contain exactly %s; missing=%s, extra=%s"
                    % (names, missing, extra)
                )
            raw = [value[name] for name in names]
        else:
            try:
                raw = list(value)
            except TypeError as exc:
                raise TypeError(
                    "residual_scales must be a sequence or target-name mapping"
                ) from exc
            if len(raw) != len(names):
                raise SolverConfigurationError(
                    "residual_scales must have one entry per target"
                )
        converted = np.asarray(
            [_finite_positive(entry, "residual scale") for entry in raw],
            dtype=float,
        )
        with np.errstate(over="ignore", divide="ignore", under="ignore"):
            reciprocals = 1.0 / converted
        if not np.all(np.isfinite(reciprocals)) or np.any(reciprocals <= 0.0):
            raise SolverConfigurationError(
                "every residual scale must have a finite positive reciprocal"
            )
        return _immutable_array(converted)

    def _target_array(self, target: Any) -> np.ndarray:
        try:
            array = np.asarray(target, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SolverInputError("target must be a finite numeric vector") from exc
        if array.shape != (self.interpolator.dimension,):
            raise SolverInputError(
                "target must have shape (%d,)" % self.interpolator.dimension
            )
        if not np.all(np.isfinite(array)):
            raise SolverInputError("target must contain only finite values")
        return array

    def _initial_array(self, initial_guess: Optional[Any]) -> Tuple[np.ndarray, str]:
        if initial_guess is None:
            bounds = self.interpolator.bounds
            point = bounds[:, 0] / 2.0 + bounds[:, 1] / 2.0
            return point, "domain_midpoint"
        try:
            point = np.asarray(initial_guess, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SolverInputError(
                "initial_guess must be a finite numeric vector"
            ) from exc
        if point.shape != (self.interpolator.dimension,):
            raise SolverInputError(
                "initial_guess must have shape (%d,)" % self.interpolator.dimension
            )
        if not np.all(np.isfinite(point)):
            raise SolverInputError("initial_guess must contain only finite values")
        if not self.interpolator.contains(point, include_boundary_tolerance=False):
            raise SolverInputError(
                "initial_guess lies outside the closed source-coordinate domain"
            )
        return point, "provided"

    @staticmethod
    def _strategy(value: Any) -> SolveStrategy:
        if isinstance(value, SolveStrategy):
            return value
        aliases = {"fast_only": "fast", "fallback_only": "fallback"}
        if isinstance(value, str):
            value = aliases.get(value, value)
        try:
            return SolveStrategy(value)
        except (TypeError, ValueError) as exc:
            raise SolverInputError(
                "strategy must be 'auto', 'fast', or 'fallback'"
            ) from exc

    def _residual_scales(self, target: np.ndarray) -> np.ndarray:
        if self._fixed_residual_scales is not None:
            return np.asarray(self._fixed_residual_scales, dtype=float)
        return np.asarray(self._automatic_residual_scales, dtype=float)

    def _refine_candidate_subcells(
        self,
        target: Any,
        candidate_cell_flat_indices: Any,
        *,
        maximum_depth: int = MAX_SUBCELL_DEPTH,
        maximum_active_boxes: int = DEFAULT_MAX_SUBCELL_BOXES,
    ) -> SubcellRefinementResult:
        """Build proof-safe target-local subcells without changing solve flow.

        This private Phase-2 entry point is intentionally not called by
        :meth:`solve`.  Later integration phases can reuse it after the shared
        residual-box Krawczyk queue exists.  Until then it provides a focused
        contract and test surface without changing production behavior.
        """

        if not self.interval_certification_available:
            raise SolverConfigurationError(
                "subcell proof evaluation requires the rigorous binary64 "
                "interval backend"
            )
        target_array = self._target_array(target)
        scales = self._residual_scales(target_array)
        evaluator = _RigorousSubcellEnclosureEvaluator(
            self.interpolator,
            target_array,
            scales,
            self.config.root_residual_tolerance,
        )
        return refine_subcells(
            target_array,
            candidate_cell_flat_indices,
            self.interpolator.dimension,
            evaluator,
            maximum_depth=maximum_depth,
            maximum_active_boxes=maximum_active_boxes,
        )

    def _refine_and_fallback_candidate_subcells(
        self,
        target: Any,
        candidate_cell_flat_indices: Any,
        initial_guess: Optional[Any] = None,
        *,
        complete_source_coverage: bool,
        active_cell_count: Optional[int] = None,
        maximum_depth: int = MAX_SUBCELL_DEPTH,
        maximum_active_boxes: int = DEFAULT_MAX_SUBCELL_BOXES,
    ) -> Tuple[SubcellRefinementResult, _FallbackOutcome]:
        """Private Phase-3 subcell plus shared-Krawczyk diagnostic path.

        This method performs no FAST solve and is not wired into the pipeline.
        ``complete_source_coverage`` is explicit because a refined box set is
        complete only relative to its supplied original candidates.  Phase 4
        will replace this private boolean with validated reachability
        localization provenance.
        """

        if not isinstance(complete_source_coverage, (bool, np.bool_)):
            raise TypeError("complete_source_coverage must be boolean")
        target_array = self._target_array(target)
        initial, initial_source = self._initial_array(initial_guess)
        scales = self._residual_scales(target_array)
        problem = _ScaledInverseProblem(self.interpolator, target_array, scales)
        cache = _RigorousCellModelCache(
            self.interpolator,
            target_array,
            problem.residual_factors,
        )
        evaluator = _RigorousSubcellEnclosureEvaluator(
            self.interpolator,
            target_array,
            scales,
            self.config.root_residual_tolerance,
            model_cache=cache,
        )
        refinement = refine_subcells(
            target_array,
            candidate_cell_flat_indices,
            self.interpolator.dimension,
            evaluator,
            maximum_depth=maximum_depth,
            maximum_active_boxes=maximum_active_boxes,
        )
        full_cell_count = int(math.prod(self.interpolator.cell_shape))
        if active_cell_count is None:
            active_cell_count = full_cell_count
        elif (
            isinstance(active_cell_count, (bool, np.bool_))
            or not isinstance(active_cell_count, (int, np.integer))
        ):
            raise TypeError("active_cell_count must be an integer")
        outcome = self._fallback_refined_subcells(
            problem,
            problem.to_normalized(initial),
            refinement,
            complete_source_coverage=bool(complete_source_coverage),
            active_cell_count=int(active_cell_count),
            seed_source=initial_source,
            model_cache=cache,
        )
        return refinement, outcome

    def solve(
        self,
        target: Any,
        initial_guess: Optional[Any] = None,
        *,
        strategy: Any = SolveStrategy.AUTO,
    ) -> InversionResult:
        """Invert one target using the selected pipeline."""

        target_array = self._target_array(target)
        initial, initial_source = self._initial_array(initial_guess)
        return self._solve(
            target_array,
            initial,
            initial_source=initial_source,
            strategy=self._strategy(strategy),
        )

    def solve_with_candidate_cells(
        self,
        target: Any,
        localization: Any,
        flat_target_index: int,
        initial_guess: Optional[Any] = None,
        *,
        expected_provenance: Any,
        strategy: Any = SolveStrategy.FALLBACK,
    ) -> InversionResult:
        """Solve using one proof-complete reachability candidate-cell row.

        Unlike :meth:`solve`, FALLBACK does not rescan the source-cell domain.
        The supplied row must be a complete
        :class:`~eos_trensmuther.reachability.ReachabilityLocalization` produced
        under the caller's expected reachability provenance.  A plain array of
        cell IDs is deliberately not accepted because it cannot prove that
        cells outside the array were examined and excluded.

        ``strategy='auto'`` first runs the normal FAST path and enters the
        restricted FALLBACK only after FAST fails.  ``strategy='fallback'``
        enters it directly.  ``strategy='fast'`` is rejected because candidate
        localization has no role in a FAST-only solve.
        """

        # Import lazily: reachability owns pipeline-facing types and imports
        # pipeline enums, while pipeline imports this solver module.
        from .reachability import (
            ReachabilityLocalization,
            ReachabilityLocalizationError,
            ReachabilityProvenance,
        )

        if not isinstance(localization, ReachabilityLocalization):
            raise SolverInputError(
                "localization must be a ReachabilityLocalization; bare cell IDs "
                "cannot authorize restricted FALLBACK"
            )
        if not isinstance(expected_provenance, ReachabilityProvenance):
            raise SolverInputError(
                "expected_provenance must be a ReachabilityProvenance"
            )
        try:
            localization.require_complete_for(expected_provenance)
        except ReachabilityLocalizationError as exc:
            raise SolverInputError(
                "restricted FALLBACK requires complete matching reachability "
                "provenance: %s" % exc
            ) from exc

        target_array = self._target_array(target)
        if isinstance(flat_target_index, (bool, np.bool_)) or not isinstance(
            flat_target_index, (int, np.integer)
        ):
            raise SolverInputError("flat_target_index must be an integer")
        target_index = int(flat_target_index)
        try:
            candidate_cells = localization.cells_for(target_index)
        except ReachabilityLocalizationError as exc:
            raise SolverInputError(str(exc)) from exc
        localized_targets = np.asarray(localization.result.targets, dtype=float).reshape(
            (-1, self.interpolator.dimension)
        )
        evaluated = localization.result.evaluated_mask.reshape(-1)
        if not bool(evaluated[target_index]):
            raise SolverInputError(
                "restricted FALLBACK cannot consume a non-evaluated target row"
            )
        if not np.array_equal(localized_targets[target_index], target_array):
            raise SolverInputError(
                "target does not equal the selected localization row"
            )

        provenance = localization.provenance
        mismatches = []
        if provenance.table_fingerprint != self.interpolator.source_snapshot_fingerprint:
            mismatches.append("table_fingerprint")
        if provenance.coordinate_names != self.interpolator.coordinate_names:
            mismatches.append("coordinate_names")
        if provenance.field_names != self.interpolator.field_names:
            mismatches.append("field_names")
        if provenance.cell_shape != self.interpolator.cell_shape:
            mismatches.append("cell_shape")
        scales = self._residual_scales(target_array)
        if not np.array_equal(provenance.residual_scales, scales):
            mismatches.append("residual_scales")
        if provenance.root_residual_tolerance != self.config.root_residual_tolerance:
            mismatches.append("root_residual_tolerance")
        if provenance.cell_order != "C":
            mismatches.append("cell_order")
        if mismatches:
            raise SolverInputError(
                "reachability localization does not match this solver problem: %s"
                % ", ".join(mismatches)
            )

        selected_strategy = self._strategy(strategy)
        if selected_strategy is SolveStrategy.FAST:
            raise SolverInputError(
                "solve_with_candidate_cells requires strategy 'auto' or 'fallback'"
            )
        initial, initial_source = self._initial_array(initial_guess)
        return self._solve(
            target_array,
            initial,
            initial_source=initial_source,
            strategy=selected_strategy,
            fallback_cell_flat_indices=candidate_cells,
            fallback_active_cell_count=provenance.active_cell_count,
        )

    def solve_in_cell_box(
        self,
        target: Any,
        cell_indices: Any,
        source_lower: Any,
        source_upper: Any,
        initial_guess: Optional[Any] = None,
        *,
        strategy: Any = SolveStrategy.AUTO,
    ) -> InversionResult:
        """Invert one target inside a closed subrectangle of one source cell.

        This is the phase-domain counterpart of :meth:`solve`.  FAST stages
        are bounded by the supplied physical box, while FALLBACK starts from
        that same box in the owning cell's exact multilinear polynomial.  It
        therefore cannot silently leave a BINODALS/LINE candidate piece.

        The caller remains responsible for applying any curved physical
        membership predicate after convergence; a rectangular boundary
        image is intentionally only a conservative candidate.
        """

        target_array = self._target_array(target)
        dimension = self.interpolator.dimension
        raw_indices = np.asarray(cell_indices)
        if (
            raw_indices.shape != (dimension,)
            or not np.issubdtype(raw_indices.dtype, np.integer)
        ):
            raise SolverInputError(
                "cell_indices must be an integer vector with shape (%d,)"
                % dimension
            )
        indices = np.asarray(raw_indices, dtype=np.intp)
        if np.any(indices < 0) or np.any(indices >= self.interpolator.cell_shape):
            raise SolverInputError("cell_indices lie outside the interpolation grid")
        try:
            lower = np.asarray(source_lower, dtype=float)
            upper = np.asarray(source_upper, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SolverInputError(
                "source_lower and source_upper must be finite numeric vectors"
            ) from exc
        if (
            lower.shape != (dimension,)
            or upper.shape != (dimension,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise SolverInputError(
                "restricted source bounds must be finite positive-width vectors"
            )
        cell_lower, cell_upper = self.interpolator.cell_coordinate_bounds(indices)
        if np.any(lower < cell_lower) or np.any(upper > cell_upper):
            raise SolverInputError(
                "restricted source bounds must lie inside the declared cell"
            )
        if initial_guess is None:
            initial = _bounded_midpoint(lower, upper)
            initial_source = "restricted_box_midpoint"
        else:
            try:
                initial = np.asarray(initial_guess, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SolverInputError(
                    "initial_guess must be a finite numeric vector"
                ) from exc
            if (
                initial.shape != (dimension,)
                or not np.all(np.isfinite(initial))
                or np.any(initial < lower)
                or np.any(initial > upper)
            ):
                raise SolverInputError(
                    "initial_guess must lie inside the closed restricted source box"
                )
            initial_source = "provided_restricted"
        return self._solve(
            target_array,
            initial,
            initial_source=initial_source,
            strategy=self._strategy(strategy),
            restricted_cell_indices=indices,
            restricted_source_lower=lower,
            restricted_source_upper=upper,
        )

    def solve_path(
        self,
        targets: Any,
        initial_guess: Optional[Any] = None,
        *,
        strategy: Any = SolveStrategy.AUTO,
    ) -> Tuple[InversionResult, ...]:
        """Solve an ordered target path with success-only continuation."""

        try:
            array = np.asarray(targets, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SolverInputError(
                "targets must be a finite array with shape (count, dimension)"
            ) from exc
        if array.ndim != 2 or array.shape[1] != self.interpolator.dimension:
            raise SolverInputError(
                "targets must have shape (count, %d)" % self.interpolator.dimension
            )
        if not np.all(np.isfinite(array)):
            raise SolverInputError("targets must contain only finite values")
        selected_strategy = self._strategy(strategy)
        first, first_source = self._initial_array(initial_guess)
        continuation: Optional[np.ndarray] = None
        results = []
        for index, target_array in enumerate(array):
            if continuation is not None:
                guess = continuation
                source = "continuation"
            elif index == 0:
                guess = first
                source = first_source
            else:
                guess, source = self._initial_array(None)
            result = self._solve(
                target_array,
                guess,
                initial_source=source,
                strategy=selected_strategy,
            )
            results.append(result)
            continuation = (
                np.asarray(result.solution, dtype=float)
                if result.converged and result.solution is not None
                else None
            )
        return tuple(results)

    def _solve(
        self,
        target: np.ndarray,
        initial: np.ndarray,
        *,
        initial_source: str,
        strategy: SolveStrategy,
        restricted_cell_indices: Optional[np.ndarray] = None,
        restricted_source_lower: Optional[np.ndarray] = None,
        restricted_source_upper: Optional[np.ndarray] = None,
        fallback_cell_flat_indices: Optional[np.ndarray] = None,
        fallback_active_cell_count: Optional[int] = None,
    ) -> InversionResult:
        scales = self._residual_scales(target)
        restricted = restricted_cell_indices is not None
        localized_fallback = fallback_cell_flat_indices is not None
        if restricted and localized_fallback:
            raise SolverInvariantError(
                "phase-box and ordinary localized FALLBACK cannot be combined"
            )
        problem = _ScaledInverseProblem(
            self.interpolator,
            target,
            scales,
            source_lower=restricted_source_lower,
            source_upper=restricted_source_upper,
            normalized_coordinate_system=(
                "restricted_source_box" if restricted else "global_domain"
            ),
        )
        normalized_initial = problem.to_normalized(initial)
        path = []
        localization: Optional[LocalizationReport] = None
        newton: Optional[NewtonReport] = None
        best_candidate: Optional[np.ndarray] = None

        if strategy in {SolveStrategy.AUTO, SolveStrategy.FAST}:
            path.append("bounded_least_squares")
            localization = self._localize(problem, normalized_initial)
            best_candidate = localization.point
            if localization.accepted_for_newton:
                path.append("safeguarded_newton")
                newton = self._newton(
                    problem,
                    problem.to_normalized(localization.point),
                    np.column_stack(
                        (
                            np.zeros(problem.dimension),
                            np.ones(problem.dimension),
                        )
                    ),
                )
                best_candidate = newton.point
                if newton.converged:
                    scaled = newton.residual * problem.residual_factors
                    return InversionResult(
                        status=SolveStatus.CONVERGED_FAST,
                        proof_status=ProofStatus.NONE,
                        strategy=strategy,
                        target=target,
                        solution=newton.point,
                        best_candidate=newton.point,
                        forward_values=newton.forward_values,
                        residual=newton.residual,
                        scaled_residual=scaled,
                        residual_scales=scales,
                        initial_guess=initial,
                        initial_guess_source=initial_source,
                        path=tuple(path),
                        message="fast path converged to the requested residual tolerance",
                        localization=localization,
                        newton=newton,
                    )
            if strategy is SolveStrategy.FAST:
                report = newton
                values = (
                    report.forward_values
                    if report is not None
                    else localization.forward_values
                )
                residual = report.residual if report is not None else localization.residual
                scaled = residual * problem.residual_factors
                return InversionResult(
                    status=SolveStatus.FAST_PATH_FAILED,
                    proof_status=ProofStatus.NONE,
                    strategy=strategy,
                    target=target,
                    solution=None,
                    best_candidate=best_candidate,
                    forward_values=values,
                    residual=residual,
                    scaled_residual=scaled,
                    residual_scales=scales,
                    initial_guess=initial,
                    initial_guess_source=initial_source,
                    path=tuple(path),
                    message="fast path did not meet the root residual tolerance",
                    localization=localization,
                    newton=newton,
                )

        if localized_fallback and self.config.enable_subcell_fallback:
            path.append("subcell_refinement")
        path.append("interval_krawczyk")
        fallback_seed = normalized_initial
        fallback_seed_source = "initial_guess"
        if best_candidate is not None and np.all(np.isfinite(best_candidate)):
            candidate_seed = problem.to_normalized(best_candidate)
            if np.all((candidate_seed >= 0.0) & (candidate_seed <= 1.0)):
                fallback_seed = candidate_seed
                fallback_seed_source = "fast_candidate"
        if restricted:
            assert restricted_source_lower is not None
            assert restricted_source_upper is not None
            fallback = self._fallback_restricted_cell_box(
                problem,
                restricted_cell_indices,
                restricted_source_lower,
                restricted_source_upper,
                seed_source=fallback_seed_source,
            )
        elif localized_fallback:
            if fallback_active_cell_count is None:
                raise SolverInvariantError(
                    "localized FALLBACK requires the active source-cell count"
                )
            if self.config.enable_subcell_fallback:
                fallback = self._fallback_candidate_subcells(
                    problem,
                    fallback_seed,
                    np.asarray(fallback_cell_flat_indices, dtype=np.int64),
                    active_cell_count=int(fallback_active_cell_count),
                    seed_source=fallback_seed_source,
                )
            else:
                fallback = self._fallback_candidate_localization(
                    problem,
                    fallback_seed,
                    np.asarray(fallback_cell_flat_indices, dtype=np.int64),
                    active_cell_count=int(fallback_active_cell_count),
                    seed_source=fallback_seed_source,
                )
        else:
            fallback = self._fallback(
                problem, fallback_seed, seed_source=fallback_seed_source
            )
        certificate = fallback.certificate
        if fallback.report.status is FallbackStatus.NO_ROOT_IN_DOMAIN:
            return InversionResult(
                status=SolveStatus.CERTIFIED_NO_ROOT,
                proof_status=ProofStatus.NO_ROOT_IN_DOMAIN,
                strategy=strategy,
                target=target,
                solution=None,
                best_candidate=best_candidate,
                forward_values=None,
                residual=None,
                scaled_residual=None,
                residual_scales=scales,
                initial_guess=initial,
                initial_guess_source=initial_source,
                path=tuple(path),
                message=fallback.report.message,
                localization=localization,
                newton=newton,
                fallback=fallback.report,
            )
        if certificate is None:
            return InversionResult(
                status=SolveStatus.UNRESOLVED,
                proof_status=ProofStatus.INCONCLUSIVE,
                strategy=strategy,
                target=target,
                solution=None,
                best_candidate=best_candidate,
                forward_values=None,
                residual=None,
                scaled_residual=None,
                residual_scales=scales,
                initial_guess=initial,
                initial_guess_source=initial_source,
                path=tuple(path),
                message=fallback.report.message,
                localization=localization,
                newton=newton,
                fallback=fallback.report,
            )

        path.append("certified_box_newton")
        cell_problem = _CellRefinementProblem(
            self.interpolator,
            certificate.cell_indices,
            target,
            scales,
        )
        normalized_center = _bounded_midpoint(
            certificate.cell_normalized_lower,
            certificate.cell_normalized_upper,
        )
        normalized_bounds = np.column_stack(
            (
                certificate.cell_normalized_lower,
                certificate.cell_normalized_upper,
            )
        )
        certified_newton = self._newton(
            cell_problem, normalized_center, normalized_bounds
        )
        candidate = certified_newton.point
        inside_certificate = bool(
            np.all(candidate >= certificate.lower)
            and np.all(candidate <= certificate.upper)
        )
        try:
            verified_values = np.asarray(
                self.interpolator.evaluate(candidate), dtype=float
            )
        except InterpolationDomainError as exc:
            raise SolverInvariantError(
                "certified-box Newton returned a point outside the source domain"
            ) from exc
        verified_residual = verified_values - target
        verified_scaled = verified_residual * problem.residual_factors
        verified_norm = float(np.max(np.abs(verified_scaled)))
        refinement_verified = bool(
            certified_newton.converged
            and inside_certificate
            and np.all(np.isfinite(verified_scaled))
            and verified_norm <= self.config.root_residual_tolerance
        )
        if refinement_verified:
            return InversionResult(
                status=SolveStatus.CONVERGED_FALLBACK,
                proof_status=ProofStatus.UNIQUE_IN_BOX,
                strategy=strategy,
                target=target,
                solution=candidate,
                best_candidate=candidate,
                forward_values=verified_values,
                residual=verified_residual,
                scaled_residual=verified_scaled,
                residual_scales=scales,
                initial_guess=initial,
                initial_guess_source=initial_source,
                path=tuple(path),
                message=(
                    "fallback certified a unique root in one box and Newton "
                    "refined it"
                ),
                localization=localization,
                newton=newton,
                fallback=fallback.report,
                fallback_newton=certified_newton,
                certificate=certificate,
            )
        return InversionResult(
            status=SolveStatus.CERTIFIED_ROOT_UNREFINED,
            proof_status=ProofStatus.UNIQUE_IN_BOX,
            strategy=strategy,
            target=target,
            solution=None,
            best_candidate=certified_newton.point,
            forward_values=certified_newton.forward_values,
            residual=certified_newton.residual,
            scaled_residual=certified_newton.residual * problem.residual_factors,
            residual_scales=scales,
            initial_guess=initial,
            initial_guess_source=initial_source,
            path=tuple(path),
            message=(
                "a unique root is certified inside the reported box, but "
                "Newton did not meet the requested point tolerance"
            ),
            localization=localization,
            newton=newton,
            fallback=fallback.report,
            fallback_newton=certified_newton,
            certificate=certificate,
        )

    def _localize(
        self, problem: _ScaledInverseProblem, initial: np.ndarray
    ) -> LocalizationReport:
        adapter = _LeastSquaresAdapter(problem)
        optimizer_success = False
        optimizer_status = -1
        message = "least-squares localization failed before returning a result"
        nfev = 0
        njev: Optional[int] = 0
        cost = math.inf
        point = np.asarray(initial, dtype=float)
        try:
            result = least_squares(
                adapter.residual,
                point,
                jac=adapter.jacobian,
                bounds=(np.zeros(problem.dimension), np.ones(problem.dimension)),
                method="trf",
                tr_solver="exact",
                loss="linear",
                x_scale=1.0,
                ftol=self.config.least_squares_ftol,
                xtol=self.config.least_squares_xtol,
                gtol=self.config.least_squares_gtol,
                max_nfev=self.config.least_squares_max_evaluations,
            )
            point = np.asarray(result.x, dtype=float)
            optimizer_success = bool(result.success)
            optimizer_status = int(result.status)
            message = str(result.message)
            nfev = int(result.nfev)
            njev = None if result.njev is None else int(result.njev)
            cost = float(result.cost)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            message = "least-squares numerical failure: %s" % exc
        values, scaled_residual = problem.values_and_residual(point)
        residual = values - problem.target
        norm = float(np.max(np.abs(scaled_residual)))
        accepted = bool(
            np.all(np.isfinite(point))
            and np.all((point >= 0.0) & (point <= 1.0))
            and math.isfinite(norm)
            and norm <= self.config.localization_residual_tolerance
        )
        return LocalizationReport(
            accepted_for_newton=accepted,
            optimizer_success=optimizer_success,
            point=problem.to_physical(point),
            forward_values=values,
            residual=residual,
            scaled_residual_norm=norm,
            optimizer_status=optimizer_status,
            message=message,
            function_evaluations=nfev,
            jacobian_evaluations=njev,
            cost=cost,
        )

    def _newton(
        self,
        problem: Any,
        initial: np.ndarray,
        bounds: np.ndarray,
    ) -> NewtonReport:
        lower = np.asarray(bounds[:, 0], dtype=float)
        upper = np.asarray(bounds[:, 1], dtype=float)
        point = np.asarray(initial, dtype=float).copy()
        function_evaluations = 0
        jacobian_evaluations = 0
        condition_estimate: Optional[float] = None
        reason = "maximum Newton iterations reached"
        converged = False
        values = np.full(problem.dimension, np.nan)
        scaled_residual = np.full(problem.dimension, np.nan)

        for iteration in range(self.config.newton_max_iterations + 1):
            values, scaled_residual = problem.values_and_residual(point)
            function_evaluations += 1
            norm = float(np.max(np.abs(scaled_residual)))
            if math.isfinite(norm) and norm <= self.config.root_residual_tolerance:
                converged = True
                reason = "residual tolerance satisfied"
                break
            if iteration == self.config.newton_max_iterations:
                break
            if not np.all(np.isfinite(scaled_residual)):
                reason = "non-finite residual"
                break

            phi = 0.5 * float(np.dot(scaled_residual, scaled_residual))
            accepted_trials = []
            smallest_step = math.inf
            usable_jacobian = False
            for jacobian in problem.newton_jacobians(point):
                jacobian_evaluations += 1
                if not np.all(np.isfinite(jacobian)):
                    continue
                try:
                    condition = float(np.linalg.cond(jacobian))
                except np.linalg.LinAlgError:
                    continue
                if not math.isfinite(condition):
                    continue
                condition_estimate = (
                    condition
                    if condition_estimate is None
                    else max(condition_estimate, condition)
                )
                if condition > self.config.newton_condition_limit:
                    continue
                try:
                    direction = np.linalg.solve(jacobian, -scaled_residual)
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(direction)):
                    continue
                usable_jacobian = True
                smallest_step = min(
                    smallest_step, float(np.max(np.abs(direction)))
                )
                alpha = 1.0
                for coordinate, delta in enumerate(direction):
                    if delta > 0.0:
                        alpha = min(alpha, (upper[coordinate] - point[coordinate]) / delta)
                    elif delta < 0.0:
                        alpha = min(alpha, (lower[coordinate] - point[coordinate]) / delta)
                if not math.isfinite(alpha) or alpha <= 0.0:
                    continue
                alpha = min(1.0, alpha)
                while alpha >= self.config.newton_min_step_fraction:
                    trial = point + alpha * direction
                    if np.array_equal(trial, point):
                        break
                    if np.any(trial < lower) or np.any(trial > upper):
                        alpha *= self.config.newton_backtrack_factor
                        continue
                    trial_values, trial_residual = problem.values_and_residual(trial)
                    function_evaluations += 1
                    if not np.all(np.isfinite(trial_residual)):
                        alpha *= self.config.newton_backtrack_factor
                        continue
                    trial_phi = 0.5 * float(np.dot(trial_residual, trial_residual))
                    required = phi * (1.0 - 2.0 * self.config.newton_armijo * alpha)
                    if trial_phi <= required:
                        accepted_trials.append(
                            (trial_phi, trial, trial_values, trial_residual)
                        )
                        break
                    alpha *= self.config.newton_backtrack_factor

            if not accepted_trials:
                if not usable_jacobian:
                    reason = "Jacobian is singular, non-finite, or ill-conditioned"
                elif smallest_step <= self.config.newton_step_tolerance:
                    reason = "Newton stagnated with an unacceptable residual"
                else:
                    reason = "Newton line search could not find a decreasing in-domain step"
                break
            _, point, values, scaled_residual = min(
                accepted_trials, key=lambda item: item[0]
            )

        residual = values - problem.target
        final_norm = float(np.max(np.abs(scaled_residual)))
        return NewtonReport(
            converged=converged,
            point=problem.to_physical(point),
            forward_values=values,
            residual=residual,
            scaled_residual_norm=final_norm,
            iterations=iteration,
            function_evaluations=function_evaluations,
            jacobian_evaluations=jacobian_evaluations,
            condition_estimate=condition_estimate,
            reason=reason,
            normalized_bounds=np.column_stack((lower, upper)),
            normalized_coordinate_system=problem.normalized_coordinate_system,
        )

    def _run_krawczyk_subcell_boxes(
        self,
        problem: _ScaledInverseProblem,
        seed: np.ndarray,
        boxes: SubcellBoxes,
        model_cache: _RigorousCellModelCache,
    ) -> Tuple[_KrawczykBoxSearchReport, Optional[CertifiedRootBox]]:
        """Run one shared Krawczyk context over arbitrary normalized boxes.

        Every initial residual box receives one Krawczyk evaluation before
        recursive work begins (unless the global box-evaluation resource limit
        prevents it).  Only inconclusive or useful contracted boxes enter the
        shared priority queue.  This is the Phase-3 replacement for invoking
        :meth:`solve_in_cell_box` once per refined box.
        """

        if not isinstance(boxes, SubcellBoxes):
            raise TypeError("boxes must be SubcellBoxes")
        if boxes.dimension != problem.dimension:
            raise SolverInvariantError(
                "subcell boxes do not match the inverse-problem dimension"
            )
        if (
            not isinstance(model_cache, _RigorousCellModelCache)
            or model_cache.interpolator is not self.interpolator
            or not np.array_equal(model_cache.target, problem.target)
            or not np.array_equal(
                model_cache.residual_factors, problem.residual_factors
            )
        ):
            raise SolverInvariantError(
                "Krawczyk model cache does not match the inverse problem"
            )

        normalized_seed = np.asarray(seed, dtype=float)
        if (
            normalized_seed.shape != (problem.dimension,)
            or not np.all(np.isfinite(normalized_seed))
            or np.any(normalized_seed < 0.0)
            or np.any(normalized_seed > 1.0)
        ):
            raise SolverInvariantError(
                "Krawczyk box priority seed must lie in the normalized domain"
            )

        model_constructions_before = model_cache.construction_count
        cell_indices = np.column_stack(
            np.unravel_index(
                boxes.cell_flat_indices,
                self.interpolator.cell_shape,
                order="C",
            )
        ).astype(np.intp, copy=False)
        physical_lower, physical_upper = self.interpolator.cell_coordinate_bounds(
            cell_indices
        )
        unit_centers = _bounded_midpoint(boxes.unit_lower, boxes.unit_upper)
        physical_centers = physical_lower + (
            physical_upper - physical_lower
        ) * unit_centers
        normalized_centers = (
            physical_centers - problem.lower[None, :]
        ) / problem.span[None, :]
        with np.errstate(over="ignore", invalid="ignore"):
            priorities = np.sum(
                (normalized_centers - normalized_seed[None, :]) ** 2,
                axis=1,
            )
        priorities = np.where(np.isfinite(priorities), priorities, math.inf)
        initial_order = np.argsort(priorities, kind="stable")

        queue = []
        counter = 0
        one_shot_processed = 0
        one_shot_excluded = 0
        one_shot_certified = 0
        recursive_processed = 0
        excluded = 0
        contracted = 0
        subdivided = 0
        terminal_unresolved = 0
        maximum_depth = 0
        certificate: Optional[CertifiedRootBox] = None

        def priority(
            model: _RigorousCellModel,
            lower: np.ndarray,
            upper: np.ndarray,
        ) -> float:
            center = _bounded_midpoint(lower, upper)
            physical = model.physical_lower + (
                model.physical_upper - model.physical_lower
            ) * center
            normalized = (physical - problem.lower) / problem.span
            value = float(np.sum((normalized - normalized_seed) ** 2))
            return value if math.isfinite(value) else math.inf

        def certificate_for(
            model: _RigorousCellModel, step: _KrawczykStep
        ) -> CertifiedRootBox:
            assert step.lower is not None and step.upper is not None
            physical_lower, physical_upper = model.physical_box(
                step.lower, step.upper
            )
            return CertifiedRootBox(
                coordinate_names=self.interpolator.coordinate_names,
                lower=physical_lower,
                upper=physical_upper,
                cell_indices=model.cell_indices,
                cell_normalized_lower=step.lower,
                cell_normalized_upper=step.upper,
            )

        def schedule_after_step(
            model: _RigorousCellModel,
            flat_cell_id: int,
            lower: np.ndarray,
            upper: np.ndarray,
            depth: int,
            step: _KrawczykStep,
        ) -> None:
            nonlocal counter, contracted, subdivided, terminal_unresolved
            next_lower = lower
            next_upper = upper
            if step.kind == "contract":
                assert step.lower is not None and step.upper is not None
                old_width = upper - lower
                new_width = step.upper - step.lower
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = new_width / old_width
                volume_ratio = float(np.prod(ratios))
                representable_progress = bool(
                    np.any(step.lower > lower) or np.any(step.upper < upper)
                )
                if (
                    representable_progress
                    and math.isfinite(volume_ratio)
                    and volume_ratio
                    <= self.config.fallback_contraction_volume_ratio
                ):
                    contracted += 1
                    heapq.heappush(
                        queue,
                        (
                            priority(model, step.lower, step.upper),
                            counter,
                            int(flat_cell_id),
                            step.lower,
                            step.upper,
                            depth + 1,
                        ),
                    )
                    counter += 1
                    return
                next_lower = step.lower
                next_upper = step.upper

            widths = next_upper - next_lower
            if (
                depth >= self.config.fallback_max_depth
                or float(np.max(widths))
                <= self.config.fallback_min_normalized_width
            ):
                terminal_unresolved += 1
                return
            split_axis = int(np.argmax(widths))
            midpoint = (
                next_lower[split_axis] / 2.0
                + next_upper[split_axis] / 2.0
            )
            if not next_lower[split_axis] < midpoint < next_upper[split_axis]:
                terminal_unresolved += 1
                return
            first_upper = next_upper.copy()
            first_upper[split_axis] = midpoint
            second_lower = next_lower.copy()
            second_lower[split_axis] = midpoint
            for child_lower, child_upper in (
                (next_lower.copy(), first_upper),
                (second_lower, next_upper.copy()),
            ):
                heapq.heappush(
                    queue,
                    (
                        priority(model, child_lower, child_upper),
                        counter,
                        int(flat_cell_id),
                        child_lower,
                        child_upper,
                        depth + 1,
                    ),
                )
                counter += 1
            subdivided += 1

        # One-shot phase: evaluate every permitted initial residual box as one
        # target-level batch before recursing.  This is the scalar proof
        # formula lifted over independent cell/box rows, not a second interval
        # interpolation implementation.
        permitted = min(boxes.count, self.config.fallback_max_boxes)
        selected_rows = initial_order[:permitted]
        batch_steps = _krawczyk_steps_for_cells(
            self.interpolator,
            cell_indices[selected_rows],
            problem.target,
            problem.residual_factors,
            boxes.unit_lower[selected_rows],
            boxes.unit_upper[selected_rows],
        )
        for position, row in enumerate(selected_rows):
            flat_cell_id = int(boxes.cell_flat_indices[row])
            lower = np.asarray(boxes.unit_lower[row], dtype=float)
            upper = np.asarray(boxes.unit_upper[row], dtype=float)
            kind = batch_steps.kinds[position]
            one_shot_processed += 1
            if kind == _BATCH_KRAWCZYK_EXCLUDED:
                one_shot_excluded += 1
                excluded += 1
                continue
            model = model_cache.for_flat_id(flat_cell_id)
            if kind == _BATCH_KRAWCZYK_CERTIFIED:
                one_shot_certified += 1
                if certificate is None:
                    certificate = certificate_for(
                        model,
                        _KrawczykStep(
                            "certified",
                            batch_steps.lower[position],
                            batch_steps.upper[position],
                        ),
                    )
                continue
            if kind == _BATCH_KRAWCZYK_CONTRACT:
                step = _KrawczykStep(
                    "contract",
                    batch_steps.lower[position],
                    batch_steps.upper[position],
                )
            else:
                step = _KrawczykStep("inconclusive")
            schedule_after_step(model, flat_cell_id, lower, upper, 0, step)

        unprocessed_initial = boxes.count - one_shot_processed

        # A box-local root certificate is already sufficient for the current
        # solver contract.  Complete the promised initial one-shot phase, then
        # avoid unrelated recursive work.
        if certificate is None and unprocessed_initial == 0:
            while (
                queue
                and one_shot_processed + recursive_processed
                < self.config.fallback_max_boxes
            ):
                _, _, flat_cell_id, lower, upper, depth = heapq.heappop(queue)
                maximum_depth = max(maximum_depth, depth)
                recursive_processed += 1
                model = model_cache.for_flat_id(flat_cell_id)
                step = _krawczyk_step(model, lower, upper)
                if step.kind == "excluded":
                    excluded += 1
                    continue
                if step.kind == "certified":
                    certificate = certificate_for(model, step)
                    break
                schedule_after_step(
                    model, flat_cell_id, lower, upper, depth, step
                )

        queued_remaining = len(queue)
        unresolved = (
            terminal_unresolved + unprocessed_initial + queued_remaining
        )
        report = _KrawczykBoxSearchReport(
            initial_boxes=boxes.count,
            one_shot_boxes_processed=one_shot_processed,
            one_shot_boxes_excluded=one_shot_excluded,
            one_shot_boxes_certified=one_shot_certified,
            recursive_boxes_processed=recursive_processed,
            boxes_excluded=excluded,
            boxes_contracted=contracted,
            boxes_subdivided=subdivided,
            unresolved_boxes=unresolved,
            unprocessed_initial_boxes=unprocessed_initial,
            queued_boxes_remaining=queued_remaining,
            maximum_depth_reached=maximum_depth,
            cell_models_constructed=(
                model_cache.construction_count - model_constructions_before
            ),
            cell_models_total=model_cache.construction_count,
        )
        return report, certificate

    def _fallback_refined_subcells(
        self,
        problem: _ScaledInverseProblem,
        seed: np.ndarray,
        refinement: SubcellRefinementResult,
        *,
        complete_source_coverage: bool,
        active_cell_count: int,
        seed_source: str,
        model_cache: _RigorousCellModelCache,
    ) -> _FallbackOutcome:
        """Resolve one subcell result under explicit source-coverage authority."""

        if not isinstance(refinement, SubcellRefinementResult):
            raise TypeError("refinement must be a SubcellRefinementResult")
        if not isinstance(complete_source_coverage, (bool, np.bool_)):
            raise TypeError("complete_source_coverage must be boolean")
        complete_source_coverage = bool(complete_source_coverage)
        total_cells = int(active_cell_count)
        supplied_cells = int(refinement.report.initial_cell_count)
        if (
            total_cells <= 0
            or total_cells > model_cache.cell_count
            or supplied_cells > total_cells
        ):
            raise SolverInvariantError(
                "subcell source-cell coverage is inconsistent"
            )
        cells_excluded_by_reachability = (
            total_cells - supplied_cells if complete_source_coverage else 0
        )
        candidate_source = (
            "complete_subcell_refinement"
            if complete_source_coverage
            else "unproven_subcell_refinement"
        )

        def outcome(
            status: FallbackStatus,
            message: str,
            *,
            search: Optional[_KrawczykBoxSearchReport] = None,
            certificate: Optional[CertifiedRootBox] = None,
            unresolved_boxes: Optional[int] = None,
            outward_rounded: bool = True,
        ) -> _FallbackOutcome:
            processed = 0 if search is None else search.boxes_processed
            unresolved = (
                refinement.boxes.count
                if unresolved_boxes is None and search is None
                else (
                    search.unresolved_boxes
                    if unresolved_boxes is None
                    else int(unresolved_boxes)
                )
            )
            report = FallbackReport(
                status=status,
                message=message,
                seed_source=seed_source,
                total_cells=total_cells,
                cells_scanned=supplied_cells,
                scan_complete=complete_source_coverage,
                candidate_cells=supplied_cells,
                retained_candidate_cells=supplied_cells,
                cells_excluded_by_range=cells_excluded_by_reachability,
                boxes_processed=processed,
                boxes_excluded=0 if search is None else search.boxes_excluded,
                boxes_contracted=0 if search is None else search.boxes_contracted,
                boxes_subdivided=0 if search is None else search.boxes_subdivided,
                unresolved_boxes=unresolved,
                maximum_depth_reached=(
                    0 if search is None else search.maximum_depth_reached
                ),
                certificate=certificate,
                outward_rounded=outward_rounded,
                candidate_source=candidate_source,
                localization_complete=complete_source_coverage,
                supplied_candidate_cells=supplied_cells,
                cells_excluded_by_reachability=cells_excluded_by_reachability,
                supplied_cells_excluded_by_range=0,
                unprocessed_candidate_cells=0,
                subcell_report=refinement.report,
                subcell_residual_box_count=refinement.boxes.count,
            )
            return _FallbackOutcome(report, certificate, search)

        if not self.interval_certification_available:
            return outcome(
                FallbackStatus.UNRESOLVED,
                "rigorous binary64 interval proofs are unavailable",
                outward_rounded=False,
            )
        if not refinement.report.complete:
            return outcome(
                FallbackStatus.UNRESOLVED,
                "subcell refinement is incomplete after reaching its active-box guard",
            )
        if refinement.certifies_no_root:
            if complete_source_coverage:
                return outcome(
                    FallbackStatus.NO_ROOT_IN_DOMAIN,
                    "complete source coverage was eliminated by rigorous subcell ranges",
                    unresolved_boxes=0,
                )
            return outcome(
                FallbackStatus.UNRESOLVED,
                "the supplied subcells were excluded but source coverage is unproven",
                unresolved_boxes=0,
            )

        search, certificate = self._run_krawczyk_subcell_boxes(
            problem,
            seed,
            refinement.boxes,
            model_cache,
        )
        if certificate is not None:
            return outcome(
                FallbackStatus.UNIQUE_IN_BOX,
                "shared residual-box Krawczyk search certified one unique root in the reported box",
                search=search,
                certificate=certificate,
            )
        if search.complete_no_root and complete_source_coverage:
            return outcome(
                FallbackStatus.NO_ROOT_IN_DOMAIN,
                "every box in the complete refined source union was rigorously excluded",
                search=search,
                unresolved_boxes=0,
            )
        message = "shared residual-box Krawczyk search is unresolved"
        if search.complete_no_root and not complete_source_coverage:
            message += " because source coverage is unproven"
        else:
            message += " after a box, depth, resolution, or processing limit"
        return outcome(
            FallbackStatus.UNRESOLVED,
            message,
            search=search,
        )

    def _fallback_restricted_cell_box(
        self,
        problem: _ScaledInverseProblem,
        cell_indices: np.ndarray,
        source_lower: np.ndarray,
        source_upper: np.ndarray,
        *,
        seed_source: str,
    ) -> _FallbackOutcome:
        """Run the proof kernel only inside one physical cell subrectangle."""

        if not self.interval_certification_available:
            report = FallbackReport(
                status=FallbackStatus.UNRESOLVED,
                message=(
                    "the runtime did not pass the IEEE-754 binary64 gradual-"
                    "underflow checks required for rigorous interval proofs"
                ),
                seed_source=seed_source,
                total_cells=1,
                cells_scanned=0,
                scan_complete=False,
                candidate_cells=1,
                retained_candidate_cells=1,
                cells_excluded_by_range=0,
                boxes_processed=0,
                boxes_excluded=0,
                boxes_contracted=0,
                boxes_subdivided=0,
                unresolved_boxes=1,
                maximum_depth_reached=0,
                outward_rounded=False,
            )
            return _FallbackOutcome(report, None)

        model = _RigorousCellModel(
            self.interpolator,
            np.asarray(cell_indices, dtype=np.intp),
            problem.target,
            problem.residual_factors,
        )
        cell_span = model.physical_upper - model.physical_lower
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            lower = (np.asarray(source_lower) - model.physical_lower) / cell_span
            upper = (np.asarray(source_upper) - model.physical_lower) / cell_span
        # Include the requested physical box despite binary64 division.  The
        # exact curved phase predicate is applied after any root is refined.
        lower = np.where(
            np.asarray(source_lower) == model.physical_lower,
            0.0,
            np.nextafter(lower, -np.inf),
        )
        upper = np.where(
            np.asarray(source_upper) == model.physical_upper,
            1.0,
            np.nextafter(upper, np.inf),
        )
        lower = np.maximum(lower, 0.0)
        upper = np.minimum(upper, 1.0)
        enclosure = model.value_enclosure(lower, upper)
        if np.any(
            (problem.target < enclosure.lower) | (problem.target > enclosure.upper)
        ):
            report = FallbackReport(
                status=FallbackStatus.NO_ROOT_IN_DOMAIN,
                message=(
                    "the restricted cell box was excluded by its rigorous "
                    "component ranges"
                ),
                seed_source=seed_source,
                total_cells=1,
                cells_scanned=1,
                scan_complete=True,
                candidate_cells=0,
                retained_candidate_cells=0,
                cells_excluded_by_range=1,
                boxes_processed=0,
                boxes_excluded=0,
                boxes_contracted=0,
                boxes_subdivided=0,
                unresolved_boxes=0,
                maximum_depth_reached=0,
                outward_rounded=True,
            )
            return _FallbackOutcome(report, None)

        queue = [(0, 0, lower, upper)]
        counter = 1
        processed = 0
        excluded = 0
        contracted = 0
        subdivided = 0
        unresolved = 0
        maximum_depth = 0
        certificate: Optional[CertifiedRootBox] = None

        while queue and processed < self.config.fallback_max_boxes:
            depth, _, box_lower, box_upper = heapq.heappop(queue)
            maximum_depth = max(maximum_depth, depth)
            processed += 1
            step = _krawczyk_step(model, box_lower, box_upper)
            if step.kind == "excluded":
                excluded += 1
                continue
            if step.kind == "certified":
                assert step.lower is not None and step.upper is not None
                physical_lower, physical_upper = model.physical_box(
                    step.lower, step.upper
                )
                certificate = CertifiedRootBox(
                    coordinate_names=self.interpolator.coordinate_names,
                    lower=physical_lower,
                    upper=physical_upper,
                    cell_indices=cell_indices,
                    cell_normalized_lower=step.lower,
                    cell_normalized_upper=step.upper,
                )
                break

            next_lower = box_lower
            next_upper = box_upper
            if step.kind == "contract":
                assert step.lower is not None and step.upper is not None
                old_width = box_upper - box_lower
                new_width = step.upper - step.lower
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = new_width / old_width
                volume_ratio = float(np.prod(ratios))
                progress = bool(
                    np.any(step.lower > box_lower)
                    or np.any(step.upper < box_upper)
                )
                if (
                    progress
                    and math.isfinite(volume_ratio)
                    and volume_ratio <= self.config.fallback_contraction_volume_ratio
                ):
                    contracted += 1
                    heapq.heappush(
                        queue,
                        (depth + 1, counter, step.lower, step.upper),
                    )
                    counter += 1
                    continue
                next_lower = step.lower
                next_upper = step.upper

            widths = next_upper - next_lower
            if (
                depth >= self.config.fallback_max_depth
                or float(np.max(widths))
                <= self.config.fallback_min_normalized_width
            ):
                unresolved += 1
                continue
            split_axis = int(np.argmax(widths))
            midpoint = (
                next_lower[split_axis] / 2.0
                + next_upper[split_axis] / 2.0
            )
            if not next_lower[split_axis] < midpoint < next_upper[split_axis]:
                unresolved += 1
                continue
            first_upper = next_upper.copy()
            first_upper[split_axis] = midpoint
            second_lower = next_lower.copy()
            second_lower[split_axis] = midpoint
            for child_lower, child_upper in (
                (next_lower.copy(), first_upper),
                (second_lower, next_upper.copy()),
            ):
                heapq.heappush(
                    queue,
                    (depth + 1, counter, child_lower, child_upper),
                )
                counter += 1
            subdivided += 1

        if certificate is not None:
            report = FallbackReport(
                status=FallbackStatus.UNIQUE_IN_BOX,
                message=(
                    "strict outward-rounded Krawczyk inclusion certified one "
                    "unique root inside the restricted cell box"
                ),
                seed_source=seed_source,
                total_cells=1,
                cells_scanned=1,
                scan_complete=True,
                candidate_cells=1,
                retained_candidate_cells=1,
                cells_excluded_by_range=0,
                boxes_processed=processed,
                boxes_excluded=excluded,
                boxes_contracted=contracted,
                boxes_subdivided=subdivided,
                unresolved_boxes=unresolved + len(queue),
                maximum_depth_reached=maximum_depth,
                certificate=certificate,
                outward_rounded=True,
            )
            return _FallbackOutcome(report, certificate)

        unresolved += len(queue)
        if unresolved == 0:
            status = FallbackStatus.NO_ROOT_IN_DOMAIN
            message = (
                "the complete restricted cell box was rigorously excluded; "
                "no root exists in that box"
            )
        else:
            status = FallbackStatus.UNRESOLVED
            message = (
                "restricted fallback is unresolved after reaching a box, "
                "depth, resolution, singularity, or boundary limit"
            )
        report = FallbackReport(
            status=status,
            message=message,
            seed_source=seed_source,
            total_cells=1,
            cells_scanned=1,
            scan_complete=True,
            candidate_cells=1,
            retained_candidate_cells=1,
            cells_excluded_by_range=0,
            boxes_processed=processed,
            boxes_excluded=excluded,
            boxes_contracted=contracted,
            boxes_subdivided=subdivided,
            unresolved_boxes=unresolved,
            maximum_depth_reached=maximum_depth,
            outward_rounded=True,
        )
        return _FallbackOutcome(report, None)

    def _fallback_candidate_subcells(
        self,
        problem: _ScaledInverseProblem,
        seed: np.ndarray,
        candidate_flat_ids: np.ndarray,
        *,
        active_cell_count: int,
        seed_source: str,
    ) -> _FallbackOutcome:
        """Refine one proof-complete ordinary localization before Krawczyk."""

        cell_shape = self.interpolator.cell_shape
        full_cell_count = int(math.prod(cell_shape))
        flat_ids = np.asarray(candidate_flat_ids, dtype=np.int64)
        if flat_ids.ndim != 1:
            raise SolverInvariantError(
                "localized subcell candidates must be one-dimensional"
            )
        if (
            active_cell_count <= 0
            or active_cell_count > full_cell_count
            or flat_ids.size > active_cell_count
            or np.any(flat_ids < 0)
            or np.any(flat_ids >= full_cell_count)
            or (flat_ids.size > 1 and np.any(np.diff(flat_ids) <= 0))
        ):
            raise SolverInvariantError(
                "subcell FALLBACK requires a unique, increasing, "
                "proof-complete localization row"
            )
        if flat_ids.size == 0:
            # The complete localization itself already excluded every active
            # cell; there is no geometry to refine. Reuse the established
            # complete-localization proof/report path.
            return self._fallback_candidate_localization(
                problem,
                seed,
                flat_ids,
                active_cell_count=active_cell_count,
                seed_source=seed_source,
            )
        if not self.interval_certification_available:
            return self._fallback_candidate_localization(
                problem,
                seed,
                flat_ids,
                active_cell_count=active_cell_count,
                seed_source=seed_source,
            )

        model_cache = _RigorousCellModelCache(
            self.interpolator,
            problem.target,
            problem.residual_factors,
        )
        evaluator = _RigorousSubcellEnclosureEvaluator(
            self.interpolator,
            problem.target,
            problem.residual_scales,
            self.config.root_residual_tolerance,
            model_cache=model_cache,
        )
        refinement = refine_subcells(
            problem.target,
            flat_ids,
            problem.dimension,
            evaluator,
            maximum_depth=self.config.max_subcell_depth,
            maximum_active_boxes=self.config.max_subcell_active_boxes,
        )
        return self._fallback_refined_subcells(
            problem,
            seed,
            refinement,
            complete_source_coverage=True,
            active_cell_count=active_cell_count,
            seed_source=seed_source,
            model_cache=model_cache,
        )

    def _fallback_candidate_localization(
        self,
        problem: _ScaledInverseProblem,
        seed: np.ndarray,
        candidate_flat_ids: np.ndarray,
        *,
        active_cell_count: int,
        seed_source: str,
    ) -> _FallbackOutcome:
        """Search only a proof-complete row of routing-AABB cell matches."""

        cell_shape = self.interpolator.cell_shape
        full_cell_count = int(math.prod(cell_shape))
        flat_ids = np.asarray(candidate_flat_ids, dtype=np.int64)
        if flat_ids.ndim != 1:
            raise SolverInvariantError(
                "localized FALLBACK candidate IDs must be one-dimensional"
            )
        if (
            active_cell_count <= 0
            or active_cell_count > full_cell_count
            or flat_ids.size > active_cell_count
        ):
            raise SolverInvariantError(
                "localized FALLBACK active-cell count is inconsistent"
            )
        if (
            np.any(flat_ids < 0)
            or np.any(flat_ids >= full_cell_count)
            or (flat_ids.size > 1 and np.any(np.diff(flat_ids) <= 0))
        ):
            raise SolverInvariantError(
                "localized FALLBACK requires unique increasing C-order cell IDs"
            )

        supplied_count = int(flat_ids.size)
        reachability_excluded = active_cell_count - supplied_count
        if not self.interval_certification_available:
            report = FallbackReport(
                status=FallbackStatus.UNRESOLVED,
                message=(
                    "the runtime did not pass the IEEE-754 binary64 gradual-"
                    "underflow checks required for rigorous interval proofs"
                ),
                seed_source=seed_source,
                total_cells=active_cell_count,
                cells_scanned=0,
                scan_complete=True,
                candidate_cells=supplied_count,
                retained_candidate_cells=0,
                cells_excluded_by_range=reachability_excluded,
                boxes_processed=0,
                boxes_excluded=0,
                boxes_contracted=0,
                boxes_subdivided=0,
                unresolved_boxes=supplied_count,
                maximum_depth_reached=0,
                outward_rounded=False,
                candidate_source="complete_reachability_localization",
                localization_complete=True,
                supplied_candidate_cells=supplied_count,
                cells_excluded_by_reachability=reachability_excluded,
                unprocessed_candidate_cells=supplied_count,
            )
            return _FallbackOutcome(report, None)

        candidate_chunks = []
        undefined_cells = 0
        for start in range(0, supplied_count, self.config.fallback_scan_chunk_size):
            selected_ids = flat_ids[
                start : start + self.config.fallback_scan_chunk_size
            ]
            indices = np.column_stack(
                np.unravel_index(selected_ids, cell_shape, order="C")
            ).astype(np.intp, copy=False)
            corners = np.asarray(
                self.interpolator.cell_corner_values(indices), dtype=float
            )
            finite = np.all(np.isfinite(corners), axis=(1, 2))
            undefined_cells += int(np.count_nonzero(~finite))
            with np.errstate(invalid="ignore"):
                lower = np.min(corners, axis=1)
                upper = np.max(corners, axis=1)
            exact_matches = finite & np.all(
                (problem.target >= lower) & (problem.target <= upper), axis=1
            )
            if np.any(exact_matches):
                candidate_chunks.append(indices[exact_matches])
        candidates = (
            np.concatenate(candidate_chunks, axis=0)
            if candidate_chunks
            else np.empty((0, problem.dimension), dtype=np.intp)
        )
        candidate_count = int(candidates.shape[0])
        supplied_range_excluded = (
            supplied_count - candidate_count - undefined_cells
        )
        cells_excluded_by_range = (
            reachability_excluded + supplied_range_excluded
        )

        if not candidate_count:
            if undefined_cells == 0:
                report = FallbackReport(
                    status=FallbackStatus.NO_ROOT_IN_DOMAIN,
                    message=(
                        "every active source cell was excluded by the complete "
                        "reachability localization and rigorous component ranges"
                    ),
                    seed_source=seed_source,
                    total_cells=active_cell_count,
                    cells_scanned=supplied_count,
                    scan_complete=True,
                    candidate_cells=0,
                    retained_candidate_cells=0,
                    cells_excluded_by_range=active_cell_count,
                    boxes_processed=0,
                    boxes_excluded=0,
                    boxes_contracted=0,
                    boxes_subdivided=0,
                    unresolved_boxes=0,
                    maximum_depth_reached=0,
                    outward_rounded=True,
                    candidate_source="complete_reachability_localization",
                    localization_complete=True,
                    supplied_candidate_cells=supplied_count,
                    cells_excluded_by_reachability=reachability_excluded,
                    supplied_cells_excluded_by_range=supplied_range_excluded,
                )
                return _FallbackOutcome(report, None)
            report = FallbackReport(
                status=FallbackStatus.UNRESOLVED,
                message=(
                    "complete reachability localization contains non-finite "
                    "source cells that cannot be rigorously excluded"
                ),
                seed_source=seed_source,
                total_cells=active_cell_count,
                cells_scanned=supplied_count,
                scan_complete=True,
                candidate_cells=0,
                retained_candidate_cells=0,
                cells_excluded_by_range=cells_excluded_by_range,
                boxes_processed=0,
                boxes_excluded=0,
                boxes_contracted=0,
                boxes_subdivided=0,
                unresolved_boxes=undefined_cells,
                maximum_depth_reached=0,
                outward_rounded=True,
                candidate_source="complete_reachability_localization",
                localization_complete=True,
                supplied_candidate_cells=supplied_count,
                cells_excluded_by_reachability=reachability_excluded,
                supplied_cells_excluded_by_range=supplied_range_excluded,
            )
            return _FallbackOutcome(report, None)

        truncated = candidate_count > self.config.fallback_max_candidate_cells
        if candidate_count:
            lower, upper = self.interpolator.cell_coordinate_bounds(candidates)
            centers = lower / 2.0 + upper / 2.0
            normalized_centers = (centers - problem.lower) / problem.span
            distances = np.sum((normalized_centers - seed) ** 2, axis=1)
            order = np.argsort(distances, kind="stable")
            candidates = candidates[
                order[: self.config.fallback_max_candidate_cells]
            ]
        retained_candidate_count = int(candidates.shape[0])
        omitted_candidates = candidate_count - retained_candidate_count
        return self._fallback_selected_cells(
            problem,
            seed,
            candidates,
            seed_source=seed_source,
            total_cells=active_cell_count,
            cells_scanned=supplied_count,
            scan_complete=True,
            candidate_count=candidate_count,
            retained_candidate_count=retained_candidate_count,
            range_excluded_cells=cells_excluded_by_range,
            truncated=truncated,
            initial_unresolved=undefined_cells + omitted_candidates,
            candidate_source="complete_reachability_localization",
            localization_complete=True,
            supplied_candidate_cells=supplied_count,
            cells_excluded_by_reachability=reachability_excluded,
            supplied_cells_excluded_by_range=supplied_range_excluded,
            unprocessed_candidate_cells=omitted_candidates,
        )

    def _fallback(
        self,
        problem: _ScaledInverseProblem,
        seed: np.ndarray,
        *,
        seed_source: str,
    ) -> _FallbackOutcome:
        cell_shape = self.interpolator.cell_shape
        total_cells = int(math.prod(cell_shape))
        node_bounds = self.interpolator.node_value_bounds
        if np.any(
            (problem.target < node_bounds.lower)
            | (problem.target > node_bounds.upper)
        ):
            report = FallbackReport(
                status=FallbackStatus.NO_ROOT_IN_DOMAIN,
                message=(
                    "the target is outside a rigorous global component range; "
                    "no inverse exists in the interpolation domain"
                ),
                seed_source=seed_source,
                total_cells=total_cells,
                cells_scanned=0,
                scan_complete=True,
                candidate_cells=0,
                retained_candidate_cells=0,
                cells_excluded_by_range=total_cells,
                boxes_processed=0,
                boxes_excluded=0,
                boxes_contracted=0,
                boxes_subdivided=0,
                unresolved_boxes=0,
                maximum_depth_reached=0,
                outward_rounded=self.interval_certification_available,
            )
            return _FallbackOutcome(report, None)

        if not self.interval_certification_available:
            report = FallbackReport(
                status=FallbackStatus.UNRESOLVED,
                message=(
                    "the runtime did not pass the IEEE-754 binary64 gradual-"
                    "underflow checks required for rigorous interval proofs"
                ),
                seed_source=seed_source,
                total_cells=total_cells,
                cells_scanned=0,
                scan_complete=False,
                candidate_cells=0,
                retained_candidate_cells=0,
                cells_excluded_by_range=0,
                boxes_processed=0,
                boxes_excluded=0,
                boxes_contracted=0,
                boxes_subdivided=0,
                unresolved_boxes=total_cells,
                maximum_depth_reached=0,
                outward_rounded=False,
            )
            return _FallbackOutcome(report, None)

        scan_count = min(total_cells, self.config.fallback_max_scanned_cells)
        flat_ids = np.arange(scan_count, dtype=np.int64)
        seed_physical = problem.to_physical(seed)
        seed_cell = np.asarray(
            self.interpolator.locate(seed_physical).cell_indices, dtype=np.intp
        )
        seed_flat = int(np.ravel_multi_index(tuple(seed_cell), cell_shape))
        if seed_flat >= scan_count:
            flat_ids = np.concatenate((flat_ids, np.asarray([seed_flat], dtype=np.int64)))
        candidate_chunks = []
        undefined_cells = 0
        for start in range(0, flat_ids.size, self.config.fallback_scan_chunk_size):
            selected_ids = flat_ids[start : start + self.config.fallback_scan_chunk_size]
            indices = np.column_stack(
                np.unravel_index(selected_ids, cell_shape)
            ).astype(np.intp, copy=False)
            corners = np.asarray(self.interpolator.cell_corner_values(indices), dtype=float)
            finite = np.all(np.isfinite(corners), axis=(1, 2))
            undefined_cells += int(np.count_nonzero(~finite))
            with np.errstate(invalid="ignore"):
                lower = np.min(corners, axis=1)
                upper = np.max(corners, axis=1)
            matches = finite & np.all(
                (problem.target >= lower) & (problem.target <= upper), axis=1
            )
            if np.any(matches):
                candidate_chunks.append(indices[matches])
        candidates = (
            np.concatenate(candidate_chunks, axis=0)
            if candidate_chunks
            else np.empty((0, problem.dimension), dtype=np.intp)
        )
        scan_complete = bool(scan_count == total_cells and undefined_cells == 0)
        candidate_count = int(candidates.shape[0])
        truncated = candidate_count > self.config.fallback_max_candidate_cells
        if candidate_count:
            lower, upper = self.interpolator.cell_coordinate_bounds(candidates)
            centers = lower / 2.0 + upper / 2.0
            normalized_centers = (centers - problem.lower) / problem.span
            distances = np.sum((normalized_centers - seed) ** 2, axis=1)
            order = np.argsort(distances, kind="stable")
            candidates = candidates[
                order[: self.config.fallback_max_candidate_cells]
            ]
        retained_candidate_count = int(candidates.shape[0])
        range_excluded_cells = int(
            flat_ids.size - candidate_count - undefined_cells
        )
        omitted_candidates = candidate_count - retained_candidate_count
        unscanned_cells = total_cells - int(flat_ids.size)

        if not candidate_count:
            if scan_complete:
                report = FallbackReport(
                    status=FallbackStatus.NO_ROOT_IN_DOMAIN,
                    message=(
                        "every interpolation cell was excluded by rigorous "
                        "component ranges"
                    ),
                    seed_source=seed_source,
                    total_cells=total_cells,
                    cells_scanned=int(flat_ids.size),
                    scan_complete=True,
                    candidate_cells=0,
                    retained_candidate_cells=0,
                    cells_excluded_by_range=total_cells,
                    boxes_processed=0,
                    boxes_excluded=0,
                    boxes_contracted=0,
                    boxes_subdivided=0,
                    unresolved_boxes=0,
                    maximum_depth_reached=0,
                    outward_rounded=True,
                )
                return _FallbackOutcome(report, None)
            report = FallbackReport(
                status=FallbackStatus.UNRESOLVED,
                message="the configured cell-scan limit was reached before full coverage",
                seed_source=seed_source,
                total_cells=total_cells,
                cells_scanned=int(flat_ids.size),
                scan_complete=False,
                candidate_cells=0,
                retained_candidate_cells=0,
                cells_excluded_by_range=range_excluded_cells,
                boxes_processed=0,
                boxes_excluded=0,
                boxes_contracted=0,
                boxes_subdivided=0,
                unresolved_boxes=unscanned_cells + undefined_cells,
                maximum_depth_reached=0,
                outward_rounded=True,
            )
            return _FallbackOutcome(report, None)

        return self._fallback_selected_cells(
            problem,
            seed,
            candidates,
            seed_source=seed_source,
            total_cells=total_cells,
            cells_scanned=int(flat_ids.size),
            scan_complete=scan_complete,
            candidate_count=candidate_count,
            retained_candidate_count=retained_candidate_count,
            range_excluded_cells=range_excluded_cells,
            truncated=truncated,
            initial_unresolved=(
                undefined_cells + omitted_candidates + unscanned_cells
            ),
            candidate_source="global_scan",
            localization_complete=False,
            supplied_candidate_cells=0,
            cells_excluded_by_reachability=0,
            supplied_cells_excluded_by_range=0,
            unprocessed_candidate_cells=omitted_candidates,
        )

    def _fallback_selected_cells(
        self,
        problem: _ScaledInverseProblem,
        seed: np.ndarray,
        candidates: np.ndarray,
        *,
        seed_source: str,
        total_cells: int,
        cells_scanned: int,
        scan_complete: bool,
        candidate_count: int,
        retained_candidate_count: int,
        range_excluded_cells: int,
        truncated: bool,
        initial_unresolved: int,
        candidate_source: str,
        localization_complete: bool,
        supplied_candidate_cells: int,
        cells_excluded_by_reachability: int,
        supplied_cells_excluded_by_range: int,
        unprocessed_candidate_cells: int,
    ) -> _FallbackOutcome:
        """Run the shared interval queue over already-selected exact cells."""

        model_cache: "OrderedDict[Tuple[int, ...], _RigorousCellModel]" = OrderedDict()

        def model_for(indices: np.ndarray) -> _RigorousCellModel:
            key = tuple(int(value) for value in indices)
            model = model_cache.pop(key, None)
            if model is None:
                model = _RigorousCellModel(
                    self.interpolator,
                    np.asarray(key, dtype=np.intp),
                    problem.target,
                    problem.residual_factors,
                )
            model_cache[key] = model
            while len(model_cache) > self.config.fallback_cell_cache_size:
                model_cache.popitem(last=False)
            return model

        queue = []
        counter = 0

        def priority(
            model: _RigorousCellModel, lower: np.ndarray, upper: np.ndarray
        ) -> float:
            unit_center = lower / 2.0 + upper / 2.0
            physical = model.physical_lower + (
                model.physical_upper - model.physical_lower
            ) * unit_center
            normalized = (physical - problem.lower) / problem.span
            return float(np.sum((normalized - seed) ** 2))

        for indices in candidates:
            model = model_for(indices)
            lower = np.zeros(problem.dimension, dtype=float)
            upper = np.ones(problem.dimension, dtype=float)
            heapq.heappush(
                queue,
                (priority(model, lower, upper), counter, indices.copy(), lower, upper, 0),
            )
            counter += 1

        processed = 0
        excluded = 0
        contracted = 0
        subdivided = 0
        unresolved = int(initial_unresolved)
        maximum_depth = 0
        certificate: Optional[CertifiedRootBox] = None

        while queue and processed < self.config.fallback_max_boxes:
            _, _, indices, lower, upper, depth = heapq.heappop(queue)
            maximum_depth = max(maximum_depth, depth)
            processed += 1
            model = model_for(indices)
            step = _krawczyk_step(model, lower, upper)
            if step.kind == "excluded":
                excluded += 1
                continue
            if step.kind == "certified":
                assert step.lower is not None and step.upper is not None
                physical_lower, physical_upper = model.physical_box(
                    step.lower, step.upper
                )
                certificate = CertifiedRootBox(
                    coordinate_names=self.interpolator.coordinate_names,
                    lower=physical_lower,
                    upper=physical_upper,
                    cell_indices=indices,
                    cell_normalized_lower=step.lower,
                    cell_normalized_upper=step.upper,
                )
                break

            next_lower = lower
            next_upper = upper
            if step.kind == "contract":
                assert step.lower is not None and step.upper is not None
                old_width = upper - lower
                new_width = step.upper - step.lower
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = new_width / old_width
                volume_ratio = float(np.prod(ratios))
                representable_progress = bool(
                    np.any(step.lower > lower) or np.any(step.upper < upper)
                )
                if (
                    representable_progress
                    and math.isfinite(volume_ratio)
                    and volume_ratio <= self.config.fallback_contraction_volume_ratio
                ):
                    contracted += 1
                    heapq.heappush(
                        queue,
                        (
                            priority(model, step.lower, step.upper),
                            counter,
                            indices.copy(),
                            step.lower,
                            step.upper,
                            depth + 1,
                        ),
                    )
                    counter += 1
                    continue
                next_lower = step.lower
                next_upper = step.upper

            widths = next_upper - next_lower
            if (
                depth >= self.config.fallback_max_depth
                or float(np.max(widths))
                <= self.config.fallback_min_normalized_width
            ):
                unresolved += 1
                continue
            split_axis = int(np.argmax(widths))
            midpoint = (
                next_lower[split_axis] / 2.0
                + next_upper[split_axis] / 2.0
            )
            if (
                midpoint <= next_lower[split_axis]
                or midpoint >= next_upper[split_axis]
            ):
                unresolved += 1
                continue
            first_upper = next_upper.copy()
            first_upper[split_axis] = midpoint
            second_lower = next_lower.copy()
            second_lower[split_axis] = midpoint
            for child_lower, child_upper in (
                (next_lower.copy(), first_upper),
                (second_lower, next_upper.copy()),
            ):
                heapq.heappush(
                    queue,
                    (
                        priority(model, child_lower, child_upper),
                        counter,
                        indices.copy(),
                        child_lower,
                        child_upper,
                        depth + 1,
                    ),
                )
                counter += 1
            subdivided += 1

        if certificate is not None:
            report = FallbackReport(
                status=FallbackStatus.UNIQUE_IN_BOX,
                message=(
                    "strict outward-rounded Krawczyk inclusion certified one "
                    "unique root in the reported box; other cells were not "
                    "necessarily excluded"
                ),
                seed_source=seed_source,
                total_cells=total_cells,
                cells_scanned=cells_scanned,
                scan_complete=scan_complete,
                candidate_cells=candidate_count,
                retained_candidate_cells=retained_candidate_count,
                cells_excluded_by_range=range_excluded_cells,
                boxes_processed=processed,
                boxes_excluded=excluded,
                boxes_contracted=contracted,
                boxes_subdivided=subdivided,
                unresolved_boxes=unresolved + len(queue),
                maximum_depth_reached=maximum_depth,
                certificate=certificate,
                outward_rounded=True,
                candidate_source=candidate_source,
                localization_complete=localization_complete,
                supplied_candidate_cells=supplied_candidate_cells,
                cells_excluded_by_reachability=cells_excluded_by_reachability,
                supplied_cells_excluded_by_range=(
                    supplied_cells_excluded_by_range
                ),
                unprocessed_candidate_cells=unprocessed_candidate_cells,
            )
            return _FallbackOutcome(report, certificate)

        unresolved += len(queue)
        complete_search = bool(
            scan_complete and not truncated and unresolved == 0 and not queue
        )
        if complete_search:
            status = FallbackStatus.NO_ROOT_IN_DOMAIN
            message = (
                "every cell in the source domain was rigorously excluded; "
                "no root exists for this target"
            )
        else:
            status = FallbackStatus.UNRESOLVED
            reasons = []
            if not scan_complete:
                reasons.append("cell-scan limit")
            if truncated:
                reasons.append("candidate-cell limit")
            if queue:
                reasons.append("box-processing limit")
            if unresolved:
                reasons.append("depth, resolution, singularity, or boundary candidates")
            message = "fallback is unresolved after reaching " + ", ".join(reasons)
        report = FallbackReport(
            status=status,
            message=message,
            seed_source=seed_source,
            total_cells=total_cells,
            cells_scanned=cells_scanned,
            scan_complete=scan_complete,
            candidate_cells=candidate_count,
            retained_candidate_cells=retained_candidate_count,
            cells_excluded_by_range=range_excluded_cells,
            boxes_processed=processed,
            boxes_excluded=excluded,
            boxes_contracted=contracted,
            boxes_subdivided=subdivided,
            unresolved_boxes=unresolved,
            maximum_depth_reached=maximum_depth,
            outward_rounded=True,
            candidate_source=candidate_source,
            localization_complete=localization_complete,
            supplied_candidate_cells=supplied_candidate_cells,
            cells_excluded_by_reachability=cells_excluded_by_reachability,
            supplied_cells_excluded_by_range=supplied_cells_excluded_by_range,
            unprocessed_candidate_cells=unprocessed_candidate_cells,
        )
        return _FallbackOutcome(report, None)


__all__ = [
    "CertifiedRootBox",
    "EoSRootSolver",
    "FallbackReport",
    "FallbackStatus",
    "InversionResult",
    "LocalizationReport",
    "NewtonReport",
    "ProofStatus",
    "SolveStatus",
    "SolveStrategy",
    "SolverConfig",
    "SolverConfigurationError",
    "SolverError",
    "SolverInputError",
    "SolverInvariantError",
]
