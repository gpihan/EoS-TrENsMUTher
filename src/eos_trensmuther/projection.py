"""Deterministic bounded nearest-state projection for EoS tables.

Projection answers a different question from root solving.  Given a canonical
target ``y``, it returns the best feasible state found for the fixed run-wide
metric ``||(F(x) - y) / scales||_2``.  A nonzero residual is expected and is
never relabeled as a root.  The implementation is a deterministic local
multistart reference; it makes no global-optimality claim.

The nearest active table node is installed before numerical optimization as a
guaranteed feasible incumbent.  When an active-cell mask excludes part of the
rectilinear domain, every optimizer trial is bounded to one active cell so it
cannot cross an inadmissible gap.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from .interpolation import (
    InterpolationDomainError,
    MultilinearInterpolator,
)
from .tables import EoSError

if TYPE_CHECKING:  # Avoid a pipeline -> projection -> pipeline import cycle.
    from .pipeline import PipelinePlan
    from .reachability import CellTargetEnclosures, ReachabilityIndex


DEFAULT_PROJECTION_CELL_BATCH_SIZE = 4096
DEFAULT_MAX_PROJECTION_INDEX_BYTES = 2 * 1024**3


class ProjectionError(EoSError):
    """Base exception for nearest-state projection."""


class ProjectionConfigurationError(ProjectionError):
    """Raised when a projector cannot represent the requested domain."""


class ProjectionInputError(ProjectionError):
    """Raised when a target or additional seed is malformed."""


class ProjectionInvariantError(ProjectionError):
    """Raised when the guaranteed feasible projection contract is broken."""


class ProjectionQuality(str, Enum):
    """Strength of the numerical statement made by projection."""

    OPTIMIZER_CONVERGED = "optimizer_converged"
    BEST_FEASIBLE_ENCOUNTERED = "best_feasible_encountered"
    DISCRETE_NODE_FALLBACK = "discrete_node_fallback"
    GLOBAL_CERTIFIED = "global_certified"


class ProjectionSeedSource(str, Enum):
    """Deterministic provenance of one projection optimizer start."""

    NEAREST_ACTIVE_NODE = "nearest_active_node"
    DOMAIN_MIDPOINT = "domain_midpoint"
    NEAREST_CELL_CENTER = "nearest_cell_center"
    FAST_BEST_CANDIDATE = "fast_best_candidate"
    USER_PROVIDED = "user_provided"


def _immutable_array(value: Any, dtype: Optional[Any] = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
    return immutable.reshape(array.shape)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ProjectionConfigurationError("%s must be a positive integer" % name)
    converted = int(value)
    if converted <= 0:
        raise ProjectionConfigurationError("%s must be a positive integer" % name)
    return converted


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ProjectionConfigurationError(
            "%s must be a non-negative integer" % name
        )
    converted = int(value)
    if converted < 0:
        raise ProjectionConfigurationError(
            "%s must be a non-negative integer" % name
        )
    return converted


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ProjectionConfigurationError("%s must be a finite positive number" % name)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProjectionConfigurationError(
            "%s must be a finite positive number" % name
        ) from exc
    if not math.isfinite(converted) or converted <= 0.0:
        raise ProjectionConfigurationError(
            "%s must be a finite positive number" % name
        )
    return converted


def _coerce_seed_source(value: Any) -> ProjectionSeedSource:
    if isinstance(value, ProjectionSeedSource):
        return value
    try:
        return ProjectionSeedSource(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in ProjectionSeedSource)
        raise ProjectionInputError(
            "projection seed source must be one of: %s" % choices
        ) from exc


def _resolve_scales(value: Any, names: Tuple[str, ...]) -> np.ndarray:
    if isinstance(value, Mapping):
        if set(value) != set(names):
            missing = sorted(set(names) - set(value), key=str)
            extra = sorted(set(value) - set(names), key=str)
            raise ProjectionConfigurationError(
                "residual-scale mapping must contain exactly %s; missing=%s, extra=%s"
                % (names, missing, extra)
            )
        supplied = tuple(value[name] for name in names)
    else:
        try:
            supplied = tuple(value)
        except TypeError as exc:
            raise ProjectionConfigurationError(
                "residual scales must be a sequence or target-name mapping"
            ) from exc
    if len(supplied) != len(names):
        raise ProjectionConfigurationError(
            "residual scales must have one entry per canonical target"
        )
    if any(isinstance(item, (bool, np.bool_)) for item in supplied):
        raise ProjectionConfigurationError("residual scales cannot be boolean")
    try:
        scales = np.asarray(supplied, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProjectionConfigurationError("residual scales must be numeric") from exc
    if scales.shape != (len(names),):
        raise ProjectionConfigurationError(
            "residual scales must have shape (%d,)" % len(names)
        )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        reciprocal = 1.0 / scales
    if (
        not np.all(np.isfinite(scales))
        or np.any(scales <= 0.0)
        or not np.all(np.isfinite(reciprocal))
    ):
        raise ProjectionConfigurationError(
            "residual scales must be finite, positive, and have finite reciprocals"
        )
    return scales


@dataclass(frozen=True)
class ProjectionConfig:
    """Deterministic seed policy, optimizer tolerances, and memory limits."""

    enabled: bool = True
    nearest_cell_starts: int = 8
    max_evaluations_per_start: int = 200
    least_squares_ftol: float = 1.0e-10
    least_squares_xtol: float = 1.0e-10
    least_squares_gtol: float = 1.0e-12
    cell_batch_size: int = DEFAULT_PROJECTION_CELL_BATCH_SIZE
    max_index_bytes: int = DEFAULT_MAX_PROJECTION_INDEX_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ProjectionConfigurationError("projection enabled must be boolean")
        object.__setattr__(
            self,
            "nearest_cell_starts",
            _nonnegative_integer(self.nearest_cell_starts, "nearest_cell_starts"),
        )
        for name in (
            "max_evaluations_per_start",
            "cell_batch_size",
            "max_index_bytes",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        for name in (
            "least_squares_ftol",
            "least_squares_xtol",
            "least_squares_gtol",
        ):
            converted = _finite_positive(getattr(self, name), name)
            if converted <= np.finfo(float).eps:
                raise ProjectionConfigurationError(
                    "%s must be greater than binary64 machine epsilon" % name
                )
            object.__setattr__(self, name, converted)


@dataclass(frozen=True)
class ProjectionSeed:
    """One optional caller-supplied feasible projection start."""

    point: np.ndarray
    source: ProjectionSeedSource = ProjectionSeedSource.USER_PROVIDED

    def __post_init__(self) -> None:
        try:
            point = np.asarray(self.point, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProjectionInputError("projection seed point must be numeric") from exc
        if point.ndim != 1 or not np.all(np.isfinite(point)):
            raise ProjectionInputError(
                "projection seed point must be a finite one-dimensional vector"
            )
        object.__setattr__(self, "point", _immutable_array(point))
        object.__setattr__(self, "source", _coerce_seed_source(self.source))


@dataclass(frozen=True)
class ProjectionAttempt:
    """Immutable diagnostics for one bounded least-squares start."""

    seed_source: ProjectionSeedSource
    seed_point: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    origin_cell_flat_index: Optional[int]
    admissible_cell_flat_index: Optional[int]
    optimizer_success: bool
    optimizer_status: Optional[int]
    message: str
    function_evaluations: int
    jacobian_evaluations: Optional[int]
    candidate_point: Optional[np.ndarray] = None
    achieved_target: Optional[np.ndarray] = None
    candidate_from_optimizer: bool = False
    scaled_distance_l2: Optional[float] = None
    objective: Optional[float] = None

    def __post_init__(self) -> None:
        source = _coerce_seed_source(self.seed_source)
        seed = np.asarray(self.seed_point, dtype=float)
        lower = np.asarray(self.lower_bounds, dtype=float)
        upper = np.asarray(self.upper_bounds, dtype=float)
        if seed.ndim != 1 or lower.shape != seed.shape or upper.shape != seed.shape:
            raise ValueError("projection-attempt vectors must share one dimension")
        if (
            not np.all(np.isfinite(seed))
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
            or np.any(seed < lower)
            or np.any(seed > upper)
        ):
            raise ValueError("projection attempt has invalid bounds or seed")
        if not isinstance(self.optimizer_success, (bool, np.bool_)):
            raise ValueError("optimizer_success must be boolean")
        if not isinstance(self.function_evaluations, (int, np.integer)) or int(
            self.function_evaluations
        ) < 0:
            raise ValueError("function_evaluations must be non-negative")
        if self.jacobian_evaluations is not None and (
            not isinstance(self.jacobian_evaluations, (int, np.integer))
            or int(self.jacobian_evaluations) < 0
        ):
            raise ValueError("jacobian_evaluations must be non-negative or None")
        for name in ("origin_cell_flat_index", "admissible_cell_flat_index"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, np.integer)) or int(value) < 0
            ):
                raise ValueError("%s must be non-negative or None" % name)
        candidate = self.candidate_point
        achieved = self.achieved_target
        if (candidate is None) != (achieved is None):
            raise ValueError("candidate point and achieved target must be paired")
        if candidate is None:
            if self.candidate_from_optimizer:
                raise ValueError(
                    "an unusable attempt cannot select an optimizer candidate"
                )
            if self.scaled_distance_l2 is not None or self.objective is not None:
                raise ValueError("an unusable attempt cannot carry objective values")
        else:
            candidate_array = np.asarray(candidate, dtype=float)
            achieved_array = np.asarray(achieved, dtype=float)
            if (
                candidate_array.shape != seed.shape
                or achieved_array.shape != seed.shape
                or not np.all(np.isfinite(candidate_array))
                or not np.all(np.isfinite(achieved_array))
            ):
                raise ValueError("projection candidate arrays are invalid")
            if self.scaled_distance_l2 is None or self.objective is None:
                raise ValueError("a usable attempt requires distance and objective")
            if (
                not math.isfinite(float(self.scaled_distance_l2))
                or float(self.scaled_distance_l2) < 0.0
                or not math.isfinite(float(self.objective))
                or float(self.objective) < 0.0
            ):
                raise ValueError("projection attempt metrics must be finite")
            object.__setattr__(self, "candidate_point", _immutable_array(candidate_array))
            object.__setattr__(self, "achieved_target", _immutable_array(achieved_array))
        if not isinstance(self.candidate_from_optimizer, (bool, np.bool_)):
            raise ValueError("candidate_from_optimizer must be boolean")
        object.__setattr__(self, "seed_source", source)
        object.__setattr__(self, "seed_point", _immutable_array(seed))
        object.__setattr__(self, "lower_bounds", _immutable_array(lower))
        object.__setattr__(self, "upper_bounds", _immutable_array(upper))
        object.__setattr__(self, "optimizer_success", bool(self.optimizer_success))
        object.__setattr__(
            self, "candidate_from_optimizer", bool(self.candidate_from_optimizer)
        )
        object.__setattr__(self, "function_evaluations", int(self.function_evaluations))
        if self.jacobian_evaluations is not None:
            object.__setattr__(
                self, "jacobian_evaluations", int(self.jacobian_evaluations)
            )

    @property
    def usable(self) -> bool:
        return self.candidate_point is not None


@dataclass(frozen=True)
class ProjectionResult:
    """Final reevaluated nearest-state result from deterministic multistart."""

    requested_target: np.ndarray
    source_coordinates: np.ndarray
    achieved_target: np.ndarray
    residual: np.ndarray
    scaled_residual: np.ndarray
    residual_scales: np.ndarray
    scaled_residual_norm_inf: float
    scaled_distance_l2: float
    objective: float
    quality: ProjectionQuality
    selected_seed_source: ProjectionSeedSource
    selected_attempt_index: Optional[int]
    nearest_node_flat_index: int
    attempts: Tuple[ProjectionAttempt, ...]
    method: str = "deterministic_local_multistart"

    def __post_init__(self) -> None:
        arrays = {
            "requested_target": np.asarray(self.requested_target, dtype=float),
            "source_coordinates": np.asarray(self.source_coordinates, dtype=float),
            "achieved_target": np.asarray(self.achieved_target, dtype=float),
            "residual": np.asarray(self.residual, dtype=float),
            "scaled_residual": np.asarray(self.scaled_residual, dtype=float),
            "residual_scales": np.asarray(self.residual_scales, dtype=float),
        }
        dimension = arrays["requested_target"].size
        if arrays["requested_target"].shape != (dimension,) or dimension == 0:
            raise ValueError("projection result target must be one-dimensional")
        for name, array in arrays.items():
            if array.shape != (dimension,) or not np.all(np.isfinite(array)):
                raise ValueError("projection result %s is invalid" % name)
        if np.any(arrays["residual_scales"] <= 0.0):
            raise ValueError("projection result scales must be positive")
        expected_residual = arrays["achieved_target"] - arrays["requested_target"]
        expected_scaled = expected_residual / arrays["residual_scales"]
        if not np.array_equal(arrays["residual"], expected_residual):
            raise ValueError("projection physical residual is inconsistent")
        if not np.array_equal(arrays["scaled_residual"], expected_scaled):
            raise ValueError("projection scaled residual is inconsistent")
        norm_inf = float(np.max(np.abs(expected_scaled)))
        distance = float(np.hypot.reduce(expected_scaled))
        objective = 0.5 * distance * distance
        metrics = (
            float(self.scaled_residual_norm_inf),
            float(self.scaled_distance_l2),
            float(self.objective),
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in metrics):
            raise ValueError("projection result metrics must be finite")
        if metrics != (norm_inf, distance, objective):
            raise ValueError("projection result metrics are inconsistent")
        try:
            quality = (
                self.quality
                if isinstance(self.quality, ProjectionQuality)
                else ProjectionQuality(self.quality)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("projection result quality is invalid") from exc
        seed_source = _coerce_seed_source(self.selected_seed_source)
        attempts = tuple(self.attempts)
        if not all(isinstance(item, ProjectionAttempt) for item in attempts):
            raise ValueError("attempts must contain ProjectionAttempt values")
        selected = self.selected_attempt_index
        if quality is ProjectionQuality.DISCRETE_NODE_FALLBACK:
            if selected is not None or seed_source is not ProjectionSeedSource.NEAREST_ACTIVE_NODE:
                raise ValueError("discrete-node fallback provenance is inconsistent")
        else:
            if selected is None or not 0 <= int(selected) < len(attempts):
                raise ValueError("optimizer-selected projection requires a valid attempt")
            attempt = attempts[int(selected)]
            if not attempt.usable:
                raise ValueError("selected projection attempt must be usable")
            selected_converged_optimizer = (
                attempt.optimizer_success and attempt.candidate_from_optimizer
            )
            if (
                quality is ProjectionQuality.OPTIMIZER_CONVERGED
                and not selected_converged_optimizer
            ):
                raise ValueError(
                    "converged quality requires a selected point from a "
                    "converged optimizer"
                )
            if (
                quality is ProjectionQuality.BEST_FEASIBLE_ENCOUNTERED
                and selected_converged_optimizer
            ):
                raise ValueError(
                    "best-feasible quality cannot select a converged optimizer point"
                )
        if (
            not isinstance(self.nearest_node_flat_index, (int, np.integer))
            or int(self.nearest_node_flat_index) < 0
        ):
            raise ValueError("nearest_node_flat_index must be non-negative")
        for name, array in arrays.items():
            object.__setattr__(self, name, _immutable_array(array))
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "selected_seed_source", seed_source)
        object.__setattr__(self, "attempts", attempts)
        if selected is not None:
            object.__setattr__(self, "selected_attempt_index", int(selected))
        object.__setattr__(
            self, "nearest_node_flat_index", int(self.nearest_node_flat_index)
        )

    @property
    def optimizer_attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def converged_attempt_count(self) -> int:
        return sum(item.optimizer_success for item in self.attempts)

    @property
    def usable_attempt_count(self) -> int:
        return sum(item.usable for item in self.attempts)

    @property
    def globally_certified(self) -> bool:
        return self.quality is ProjectionQuality.GLOBAL_CERTIFIED


@dataclass(frozen=True)
class _TrialSeed:
    point: np.ndarray
    source: ProjectionSeedSource
    origin_cell_flat_index: Optional[int]
    admissible_cell_flat_index: Optional[int]


class _BoundedProjectionProblem:
    """Scaled objective in one global box or one active interpolation cell."""

    def __init__(
        self,
        interpolator: MultilinearInterpolator,
        target: np.ndarray,
        scales: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        cell_bounded: bool,
    ) -> None:
        self.interpolator = interpolator
        self.target = target
        self.scales = scales
        self.factors = 1.0 / scales
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.span = self.upper - self.lower
        self.cell_bounded = cell_bounded
        self._point: Optional[np.ndarray] = None
        self._residual: Optional[np.ndarray] = None
        self._jacobian: Optional[np.ndarray] = None

    def to_normalized(self, point: np.ndarray) -> np.ndarray:
        unit = (np.asarray(point, dtype=float) - self.lower) / self.span
        return np.minimum(np.maximum(unit, 0.0), 1.0)

    def to_physical(self, normalized: np.ndarray) -> np.ndarray:
        unit = np.asarray(normalized, dtype=float)
        point = self.lower + self.span * unit
        point = np.minimum(np.maximum(point, self.lower), self.upper)
        point = np.where(unit == 0.0, self.lower, point)
        return np.where(unit == 1.0, self.upper, point)

    def _evaluate(self, normalized: np.ndarray) -> None:
        unit = np.asarray(normalized, dtype=float)
        if self._point is not None and np.array_equal(unit, self._point):
            return
        point = self.to_physical(unit)
        sides: Any = "right"
        if self.cell_bounded:
            sides = tuple("left" if value == 1.0 else "right" for value in unit)
        try:
            evaluation = self.interpolator.evaluate_with_jacobian(
                point, plane_policy=sides
            )
        except InterpolationDomainError as exc:
            raise ProjectionInvariantError(
                "a bounded projection trial attempted extrapolation"
            ) from exc
        values = np.asarray(evaluation.values, dtype=float)
        physical_jacobian = np.asarray(evaluation.jacobians, dtype=float)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            residual = (values - self.target) * self.factors
            jacobian = (
                self.factors[:, np.newaxis]
                * physical_jacobian
                * self.span[np.newaxis, :]
            )
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
            raise FloatingPointError(
                "projection produced a non-finite scaled residual or Jacobian"
            )
        self._point = unit.copy()
        self._residual = residual
        self._jacobian = jacobian

    def residual(self, normalized: np.ndarray) -> np.ndarray:
        self._evaluate(normalized)
        assert self._residual is not None
        return self._residual

    def jacobian(self, normalized: np.ndarray) -> np.ndarray:
        self._evaluate(normalized)
        assert self._jacobian is not None
        return self._jacobian


class EoSProjector:
    """Project canonical targets onto a validated active EoS domain."""

    def __init__(
        self,
        interpolator: MultilinearInterpolator,
        residual_scales: Any,
        config: Optional[ProjectionConfig] = None,
        *,
        active_cell_mask: Optional[Any] = None,
        _cell_enclosures: Optional["CellTargetEnclosures"] = None,
    ) -> None:
        if not isinstance(interpolator, MultilinearInterpolator):
            raise TypeError("interpolator must be a MultilinearInterpolator")
        if (
            interpolator.field_names != interpolator.target_names
            or len(interpolator.field_names) != interpolator.dimension
        ):
            raise ProjectionConfigurationError(
                "projection requires the canonical target-only square interpolator"
            )
        if not np.all(interpolator.node_value_bounds.defined):
            raise ProjectionConfigurationError(
                "canonical target nodes must be finite for projection"
            )
        self._interpolator = interpolator
        self._config = ProjectionConfig() if config is None else config
        if not isinstance(self._config, ProjectionConfig):
            raise TypeError("config must be a ProjectionConfig")
        if not self._config.enabled:
            raise ProjectionConfigurationError(
                "cannot construct EoSProjector when projection is disabled"
            )
        self._residual_scales = _immutable_array(
            _resolve_scales(residual_scales, interpolator.target_names)
        )
        minimum_mask_bytes = (
            math.prod(interpolator.cell_shape)
            + math.prod(interpolator.shape)
        ) * np.dtype(np.bool_).itemsize
        if minimum_mask_bytes > self._config.max_index_bytes:
            raise ProjectionConfigurationError(
                "projection domain masks require at least %d bytes, exceeding "
                "max_index_bytes=%d"
                % (minimum_mask_bytes, self._config.max_index_bytes)
            )
        self._active_cell_mask, shared_enclosures = self._resolve_active_domain(
            active_cell_mask, _cell_enclosures
        )
        active_cell_count = int(np.count_nonzero(self._active_cell_mask))
        cell_only_estimate = self._estimate_index_bytes(active_cell_count, 0)
        if cell_only_estimate > self._config.max_index_bytes:
            raise ProjectionConfigurationError(
                "projection active-cell index requires at least %d bytes, "
                "exceeding max_index_bytes=%d"
                % (cell_only_estimate, self._config.max_index_bytes)
            )

        node_mask = self._active_node_mask(self._active_cell_mask)
        active_node_count = int(np.count_nonzero(node_mask))
        if active_node_count == 0:
            raise ProjectionConfigurationError(
                "the active-cell domain has no feasible table node"
            )
        estimated = self._estimate_index_bytes(active_cell_count, active_node_count)
        if estimated > self._config.max_index_bytes:
            raise ProjectionConfigurationError(
                "projection index requires approximately %d bytes, exceeding "
                "max_index_bytes=%d" % (estimated, self._config.max_index_bytes)
            )
        self._estimated_index_bytes = estimated
        active_flat = np.flatnonzero(
            self._active_cell_mask.reshape(-1, order="C")
        ).astype(np.int64, copy=False)
        active_nodes = np.flatnonzero(node_mask.reshape(-1, order="C")).astype(
            np.int64, copy=False
        )
        self._active_cell_flat_indices = _immutable_array(active_flat, dtype=np.int64)
        self._restricted_domain = active_flat.size != math.prod(interpolator.cell_shape)
        self._active_node_flat_indices = _immutable_array(active_nodes, dtype=np.int64)

        if shared_enclosures is None:
            exact_lower, exact_upper = self._build_cell_enclosures(active_flat)
            self._cell_exact_lower = _immutable_array(exact_lower)
            self._cell_exact_upper = _immutable_array(exact_upper)
        else:
            # CellTargetEnclosures already owns defensive read-only arrays.
            # Retain those buffers instead of duplicating the largest part of
            # the reference reachability index.
            self._cell_exact_lower = shared_enclosures.exact_lower
            self._cell_exact_upper = shared_enclosures.exact_upper
        self._active_node_targets = _immutable_array(
            self._build_active_node_targets(active_nodes)
        )

    @classmethod
    def from_plan(
        cls,
        interpolator: MultilinearInterpolator,
        plan: "PipelinePlan",
        *,
        active_cell_mask: Optional[Any] = None,
    ) -> "EoSProjector":
        """Construct with one resolved pipeline plan's fixed metric."""

        from .pipeline import PipelinePlan

        if not isinstance(plan, PipelinePlan):
            raise TypeError("plan must be a PipelinePlan")
        if interpolator.coordinate_names != plan.source_names:
            raise ProjectionConfigurationError(
                "interpolator coordinates do not match the pipeline plan"
            )
        if interpolator.field_names != plan.target_names:
            raise ProjectionConfigurationError(
                "interpolator targets do not match the pipeline plan"
            )
        return cls(
            interpolator,
            plan.residual_scales,
            plan.config.projection,
            active_cell_mask=active_cell_mask,
        )

    @classmethod
    def from_reachability(
        cls,
        index: "ReachabilityIndex",
        config: Optional[ProjectionConfig] = None,
    ) -> "EoSProjector":
        """Reuse a reachability index's exact active cells and metric scales."""

        from .reachability import ReachabilityIndex

        if not isinstance(index, ReachabilityIndex):
            raise TypeError("index must be a ReachabilityIndex")
        return cls(
            index.interpolator,
            index.enclosures.residual_scales,
            config,
            _cell_enclosures=index.enclosures,
        )

    @property
    def interpolator(self) -> MultilinearInterpolator:
        return self._interpolator

    @property
    def config(self) -> ProjectionConfig:
        return self._config

    @property
    def residual_scales(self) -> np.ndarray:
        return self._residual_scales

    @property
    def active_cell_flat_indices(self) -> np.ndarray:
        return self._active_cell_flat_indices

    @property
    def active_node_flat_indices(self) -> np.ndarray:
        return self._active_node_flat_indices

    @property
    def restricted_domain(self) -> bool:
        return self._restricted_domain

    @property
    def estimated_index_bytes(self) -> int:
        return self._estimated_index_bytes

    def _resolve_active_domain(
        self,
        active_cell_mask: Optional[Any],
        enclosures: Optional["CellTargetEnclosures"],
    ) -> Tuple[np.ndarray, Optional["CellTargetEnclosures"]]:
        shape = self.interpolator.cell_shape
        if enclosures is not None:
            if active_cell_mask is not None:
                raise ProjectionConfigurationError(
                    "active_cell_mask cannot accompany shared cell enclosures"
                )
            from .reachability import CellTargetEnclosures

            if not isinstance(enclosures, CellTargetEnclosures):
                raise TypeError("_cell_enclosures must be CellTargetEnclosures")
            if enclosures.field_names != self.interpolator.field_names:
                raise ProjectionConfigurationError(
                    "shared enclosure targets do not match the interpolator"
                )
            if enclosures.cell_shape != shape:
                raise ProjectionConfigurationError(
                    "shared enclosure cells do not match the interpolator"
                )
            if not np.array_equal(enclosures.residual_scales, self.residual_scales):
                raise ProjectionConfigurationError(
                    "shared enclosure scales do not match projection scales"
                )
            mask = np.zeros(shape, dtype=bool)
            mask.reshape(-1, order="C")[enclosures.cell_flat_indices] = True
            if not np.any(mask):
                raise ProjectionConfigurationError("the active-cell domain is empty")
            return _immutable_array(mask, dtype=bool), enclosures

        if active_cell_mask is None:
            return _immutable_array(np.ones(shape, dtype=bool)), None
        mask = np.asarray(active_cell_mask)
        if mask.shape != shape or mask.dtype.kind != "b":
            raise ProjectionConfigurationError(
                "active_cell_mask must be boolean with shape %r" % (shape,)
            )
        if not np.any(mask):
            raise ProjectionConfigurationError("the active-cell domain is empty")
        return _immutable_array(mask, dtype=bool), None

    def _active_node_mask(self, cell_mask: np.ndarray) -> np.ndarray:
        node_mask = np.zeros(self.interpolator.shape, dtype=bool)
        for offset in self.interpolator.corner_offsets:
            slices = tuple(
                slice(int(value), int(value) + size)
                for value, size in zip(offset, self.interpolator.cell_shape)
            )
            node_mask[slices] |= cell_mask
        return node_mask

    def _estimate_index_bytes(self, active_cells: int, active_nodes: int) -> int:
        dimension = self.interpolator.dimension
        return int(
            math.prod(self.interpolator.cell_shape) * np.dtype(np.bool_).itemsize
            + math.prod(self.interpolator.shape) * np.dtype(np.bool_).itemsize
            + active_cells
            * (
                np.dtype(np.int64).itemsize
                + 2 * dimension * np.dtype(np.float64).itemsize
            )
            + active_nodes
            * (
                np.dtype(np.int64).itemsize
                + dimension * np.dtype(np.float64).itemsize
            )
        )

    def _unravel(self, flat: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        if flat.size == 0:
            return np.empty((0, len(shape)), dtype=np.int64)
        return np.column_stack(np.unravel_index(flat, shape, order="C"))

    def _node_coordinates(self, flat: np.ndarray) -> np.ndarray:
        indices = self._unravel(np.asarray(flat, dtype=np.int64), self.interpolator.shape)
        return np.column_stack(
            [axis[indices[:, position]] for position, axis in enumerate(self.interpolator.axes)]
        )

    def _build_active_node_targets(self, flat: np.ndarray) -> np.ndarray:
        result = np.empty((flat.size, self.interpolator.dimension), dtype=float)
        for start in range(0, flat.size, self.config.cell_batch_size):
            stop = min(flat.size, start + self.config.cell_batch_size)
            points = self._node_coordinates(flat[start:stop])
            values = np.asarray(self.interpolator.evaluate(points), dtype=float)
            if not np.all(np.isfinite(values)):
                raise ProjectionConfigurationError(
                    "active target nodes must evaluate to finite values"
                )
            result[start:stop] = values
        return result

    def _build_cell_enclosures(
        self, active_flat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        dimension = self.interpolator.dimension
        lower = np.empty((active_flat.size, dimension), dtype=float)
        upper = np.empty_like(lower)
        for start in range(0, active_flat.size, self.config.cell_batch_size):
            stop = min(active_flat.size, start + self.config.cell_batch_size)
            cells = self._unravel(
                active_flat[start:stop], self.interpolator.cell_shape
            )
            bounds = self.interpolator.cell_value_enclosures(cells)
            if not bounds.rigorous or not np.all(bounds.defined):
                raise ProjectionConfigurationError(
                    "active cell target enclosures must be rigorous and finite"
                )
            lower[start:stop] = bounds.lower
            upper[start:stop] = bounds.upper
        return lower, upper

    def _target_array(self, target: Any) -> np.ndarray:
        try:
            array = np.asarray(target, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProjectionInputError("target must be a finite numeric vector") from exc
        if array.shape != (self.interpolator.dimension,) or not np.all(
            np.isfinite(array)
        ):
            raise ProjectionInputError(
                "target must be finite with shape (%d,)" % self.interpolator.dimension
            )
        return array

    def _metrics(
        self, achieved: np.ndarray, target: np.ndarray, *, input_context: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            residual = np.asarray(achieved, dtype=float) - target
            scaled = residual / self.residual_scales
        distance = float(np.hypot.reduce(scaled))
        norm_inf = float(np.max(np.abs(scaled)))
        objective = 0.5 * distance * distance
        if (
            not np.all(np.isfinite(residual))
            or not np.all(np.isfinite(scaled))
            or not math.isfinite(distance)
            or not math.isfinite(norm_inf)
            or not math.isfinite(objective)
        ):
            error = ProjectionInputError if input_context else ProjectionInvariantError
            raise error("projection metric is not representable as finite binary64")
        return residual, scaled, norm_inf, distance, objective

    def _nearest_active_node(
        self, target: np.ndarray
    ) -> Tuple[int, np.ndarray, np.ndarray, float]:
        best_position = -1
        best_distance = math.inf
        count = self.active_node_flat_indices.size
        for start in range(0, count, self.config.cell_batch_size):
            stop = min(count, start + self.config.cell_batch_size)
            values = self._active_node_targets[start:stop]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                scaled = (values - target) / self.residual_scales
            distances = np.hypot.reduce(scaled, axis=1)
            for relative in np.argsort(distances, kind="stable"):
                distance = float(distances[relative])
                if distance < best_distance:
                    best_distance = distance
                    best_position = start + int(relative)
                break
        if best_position < 0 or not math.isfinite(best_distance):
            raise ProjectionInputError(
                "target is too large for the configured projection metric"
            )
        flat = int(self.active_node_flat_indices[best_position])
        point = self._node_coordinates(np.asarray([flat], dtype=np.int64))[0]
        achieved = np.asarray(self._active_node_targets[best_position], dtype=float)
        return flat, point, achieved, best_distance

    def _cell_distance(self, rows: np.ndarray, target: np.ndarray) -> np.ndarray:
        lower = self._cell_exact_lower[rows]
        upper = self._cell_exact_upper[rows]
        delta = np.where(target < lower, lower - target, 0.0)
        delta = np.where(target > upper, target - upper, delta)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scaled = delta / self.residual_scales
        return np.hypot.reduce(scaled, axis=1)

    def _nearest_cells(self, target: np.ndarray, count: int) -> Tuple[int, ...]:
        if count <= 0:
            return ()
        candidates = []
        total = self.active_cell_flat_indices.size
        for start in range(0, total, self.config.cell_batch_size):
            stop = min(total, start + self.config.cell_batch_size)
            rows = np.arange(start, stop, dtype=np.int64)
            distances = self._cell_distance(rows, target)
            order = np.lexsort(
                (self.active_cell_flat_indices[start:stop], distances)
            )[:count]
            candidates.extend(
                (
                    float(distances[position]),
                    int(self.active_cell_flat_indices[start + int(position)]),
                )
                for position in order
            )
            candidates = sorted(candidates)[:count]
        if any(not math.isfinite(item[0]) for item in candidates):
            raise ProjectionInputError(
                "target is too large for cell-distance seed ranking"
            )
        return tuple(item[1] for item in candidates)

    def _cell_row(self, flat: int) -> int:
        position = int(np.searchsorted(self.active_cell_flat_indices, flat))
        if (
            position >= self.active_cell_flat_indices.size
            or int(self.active_cell_flat_indices[position]) != flat
        ):
            raise ProjectionInvariantError("requested projection cell is not active")
        return position

    def _cell_source_bounds(self, flat: int) -> Tuple[np.ndarray, np.ndarray]:
        cell = self._unravel(
            np.asarray([flat], dtype=np.int64), self.interpolator.cell_shape
        )[0]
        lower, upper = self.interpolator.cell_coordinate_bounds(cell)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def _containing_active_cells(self, point: np.ndarray) -> Tuple[int, ...]:
        choices = []
        for value, axis in zip(point, self.interpolator.axes):
            if value < axis[0] or value > axis[-1]:
                return ()
            insertion = int(np.searchsorted(axis, value, side="left"))
            if insertion < axis.size and axis[insertion] == value:
                adjacent = []
                if insertion > 0:
                    adjacent.append(insertion - 1)
                if insertion < axis.size - 1:
                    adjacent.append(insertion)
                choices.append(tuple(adjacent))
            else:
                cell = int(np.searchsorted(axis, value, side="right") - 1)
                choices.append((cell,))
        result = []
        for cell in itertools.product(*choices):
            flat = int(np.ravel_multi_index(cell, self.interpolator.cell_shape, order="C"))
            if self._active_cell_mask[cell]:
                result.append(flat)
        return tuple(sorted(result))

    def _select_containing_cell(self, point: np.ndarray, target: np.ndarray) -> int:
        cells = self._containing_active_cells(point)
        if not cells:
            raise ProjectionInputError(
                "projection seed does not belong to the active-cell domain"
            )
        rows = np.asarray([self._cell_row(flat) for flat in cells], dtype=np.int64)
        distances = self._cell_distance(rows, target)
        order = np.lexsort((np.asarray(cells, dtype=np.int64), distances))
        return int(cells[int(order[0])])

    def _seed_definitions(
        self,
        target: np.ndarray,
        nearest_node: np.ndarray,
        extra_seeds: Sequence[ProjectionSeed],
    ) -> Tuple[_TrialSeed, ...]:
        seeds = []

        def add(
            point: np.ndarray,
            source: ProjectionSeedSource,
            *,
            origin_cell: Optional[int] = None,
            admissible_cell: Optional[int] = None,
        ) -> None:
            candidate = np.asarray(point, dtype=float)
            key = (candidate.tobytes(order="C"), admissible_cell)
            if any(
                (item.point.tobytes(order="C"), item.admissible_cell_flat_index)
                == key
                for item in seeds
            ):
                return
            seeds.append(
                _TrialSeed(
                    point=candidate.copy(),
                    source=source,
                    origin_cell_flat_index=origin_cell,
                    admissible_cell_flat_index=admissible_cell,
                )
            )

        nearest_cell = (
            self._select_containing_cell(nearest_node, target)
            if self.restricted_domain
            else None
        )
        add(
            nearest_node,
            ProjectionSeedSource.NEAREST_ACTIVE_NODE,
            origin_cell=nearest_cell,
            admissible_cell=nearest_cell,
        )

        bounds = self.interpolator.bounds
        if not self.restricted_domain:
            midpoint = bounds[:, 0] + 0.5 * (bounds[:, 1] - bounds[:, 0])
            add(midpoint, ProjectionSeedSource.DOMAIN_MIDPOINT)

        for flat in self._nearest_cells(target, self.config.nearest_cell_starts):
            lower, upper = self._cell_source_bounds(flat)
            center = lower + 0.5 * (upper - lower)
            add(
                center,
                ProjectionSeedSource.NEAREST_CELL_CENTER,
                origin_cell=flat,
                admissible_cell=flat if self.restricted_domain else None,
            )

        for supplied in extra_seeds:
            point = np.asarray(supplied.point, dtype=float)
            if point.shape != (self.interpolator.dimension,):
                raise ProjectionInputError(
                    "projection seed must have shape (%d,)"
                    % self.interpolator.dimension
                )
            if not self.interpolator.contains(
                point, include_boundary_tolerance=False
            ):
                raise ProjectionInputError(
                    "projection seed lies outside the closed source domain"
                )
            cell = (
                self._select_containing_cell(point, target)
                if self.restricted_domain
                else None
            )
            add(point, supplied.source, admissible_cell=cell)
        return tuple(seeds)

    def _normalize_extra_seeds(self, seeds: Any) -> Tuple[ProjectionSeed, ...]:
        if seeds is None:
            return ()
        if isinstance(seeds, ProjectionSeed):
            return (seeds,)
        if isinstance(seeds, (str, bytes)):
            raise ProjectionInputError("seeds must contain ProjectionSeed values")
        try:
            normalized = tuple(seeds)
        except TypeError as exc:
            raise ProjectionInputError(
                "seeds must be a sequence of ProjectionSeed values"
            ) from exc
        if not all(isinstance(item, ProjectionSeed) for item in normalized):
            raise ProjectionInputError("seeds must contain ProjectionSeed values")
        return normalized

    def _run_attempt(
        self, seed: _TrialSeed, target: np.ndarray
    ) -> ProjectionAttempt:
        if seed.admissible_cell_flat_index is None:
            bounds = self.interpolator.bounds
            lower = np.asarray(bounds[:, 0], dtype=float)
            upper = np.asarray(bounds[:, 1], dtype=float)
            cell_bounded = False
        else:
            lower, upper = self._cell_source_bounds(
                seed.admissible_cell_flat_index
            )
            cell_bounded = True
        problem = _BoundedProjectionProblem(
            self.interpolator,
            target,
            np.asarray(self.residual_scales),
            lower,
            upper,
            cell_bounded=cell_bounded,
        )
        initial = problem.to_normalized(seed.point)
        try:
            seed_achieved = np.asarray(
                self.interpolator.evaluate(seed.point), dtype=float
            )
            _, _, _, seed_distance, seed_objective = self._metrics(
                seed_achieved, target
            )
        except ProjectionInvariantError as exc:
            return ProjectionAttempt(
                seed_source=seed.source,
                seed_point=seed.point,
                lower_bounds=lower,
                upper_bounds=upper,
                origin_cell_flat_index=seed.origin_cell_flat_index,
                admissible_cell_flat_index=seed.admissible_cell_flat_index,
                optimizer_success=False,
                optimizer_status=None,
                message="seed metric was unusable: %s" % exc,
                function_evaluations=0,
                jacobian_evaluations=None,
            )
        except Exception as exc:
            raise ProjectionInvariantError(
                "a validated feasible projection seed could not be reevaluated"
            ) from exc
        try:
            optimizer = least_squares(
                problem.residual,
                initial,
                jac=problem.jacobian,
                bounds=(
                    np.zeros(self.interpolator.dimension),
                    np.ones(self.interpolator.dimension),
                ),
                ftol=self.config.least_squares_ftol,
                xtol=self.config.least_squares_xtol,
                gtol=self.config.least_squares_gtol,
                max_nfev=self.config.max_evaluations_per_start,
            )
        except ProjectionInvariantError:
            raise
        except Exception as exc:
            return ProjectionAttempt(
                seed_source=seed.source,
                seed_point=seed.point,
                lower_bounds=lower,
                upper_bounds=upper,
                origin_cell_flat_index=seed.origin_cell_flat_index,
                admissible_cell_flat_index=seed.admissible_cell_flat_index,
                optimizer_success=False,
                optimizer_status=None,
                message="optimizer raised %s: %s" % (type(exc).__name__, exc),
                function_evaluations=0,
                jacobian_evaluations=None,
                candidate_point=seed.point,
                achieved_target=seed_achieved,
                candidate_from_optimizer=False,
                scaled_distance_l2=seed_distance,
                objective=seed_objective,
            )

        try:
            normalized = np.asarray(optimizer.x, dtype=float)
            if normalized.shape != (self.interpolator.dimension,) or not np.all(
                np.isfinite(normalized)
            ):
                raise ValueError("optimizer returned an invalid normalized point")
            coordinate_tolerance = 8.0 * np.finfo(float).eps
            if np.any(normalized < -coordinate_tolerance) or np.any(
                normalized > 1.0 + coordinate_tolerance
            ):
                raise ValueError("optimizer returned a normalized point outside [0, 1]")
            normalized = np.clip(normalized, 0.0, 1.0)
            candidate = problem.to_physical(normalized)
            if np.any(candidate < lower) or np.any(candidate > upper):
                raise ValueError("optimizer returned a point outside its bounds")
            achieved = np.asarray(self.interpolator.evaluate(candidate), dtype=float)
            _, _, _, distance, objective = self._metrics(achieved, target)
        except ProjectionInvariantError:
            raise
        except Exception as exc:
            return ProjectionAttempt(
                seed_source=seed.source,
                seed_point=seed.point,
                lower_bounds=lower,
                upper_bounds=upper,
                origin_cell_flat_index=seed.origin_cell_flat_index,
                admissible_cell_flat_index=seed.admissible_cell_flat_index,
                optimizer_success=False,
                optimizer_status=int(getattr(optimizer, "status", 0)),
                message="optimizer result was unusable: %s" % exc,
                function_evaluations=int(getattr(optimizer, "nfev", 0)),
                jacobian_evaluations=(
                    None
                    if getattr(optimizer, "njev", None) is None
                    else int(optimizer.njev)
                ),
                candidate_point=seed.point,
                achieved_target=seed_achieved,
                candidate_from_optimizer=False,
                scaled_distance_l2=seed_distance,
                objective=seed_objective,
            )
        if objective < seed_objective:
            selected_point = candidate
            selected_achieved = achieved
            selected_distance = distance
            selected_objective = objective
            selected_from_optimizer = True
        else:
            selected_point = seed.point
            selected_achieved = seed_achieved
            selected_distance = seed_distance
            selected_objective = seed_objective
            selected_from_optimizer = False
        return ProjectionAttempt(
            seed_source=seed.source,
            seed_point=seed.point,
            lower_bounds=lower,
            upper_bounds=upper,
            origin_cell_flat_index=seed.origin_cell_flat_index,
            admissible_cell_flat_index=seed.admissible_cell_flat_index,
            optimizer_success=bool(optimizer.success),
            optimizer_status=int(optimizer.status),
            message=str(optimizer.message),
            function_evaluations=int(optimizer.nfev),
            jacobian_evaluations=(
                None if getattr(optimizer, "njev", None) is None else int(optimizer.njev)
            ),
            candidate_point=selected_point,
            achieved_target=selected_achieved,
            candidate_from_optimizer=selected_from_optimizer,
            scaled_distance_l2=selected_distance,
            objective=selected_objective,
        )

    def project(
        self, target: Any, *, seeds: Any = ()
    ) -> ProjectionResult:
        """Return the best deterministic feasible state found for one target.

        The result is always a projection result, even when its residual is
        numerically zero.  Root existence and acceptance remain the solver's
        responsibility.
        """

        requested = self._target_array(target)
        extra_seeds = self._normalize_extra_seeds(seeds)
        nearest_flat, nearest_point, _, _ = self._nearest_active_node(requested)
        try:
            incumbent_achieved = np.asarray(
                self.interpolator.evaluate(nearest_point), dtype=float
            )
        except Exception as exc:
            raise ProjectionInvariantError(
                "the guaranteed active-node incumbent could not be reevaluated"
            ) from exc
        (
            _,
            _,
            _,
            _,
            incumbent_objective,
        ) = self._metrics(incumbent_achieved, requested, input_context=True)
        best_point = nearest_point
        best_objective = incumbent_objective
        best_attempt: Optional[int] = None
        best_source = ProjectionSeedSource.NEAREST_ACTIVE_NODE

        definitions = self._seed_definitions(
            requested, nearest_point, extra_seeds
        )
        attempts = []
        for definition in definitions:
            attempt = self._run_attempt(definition, requested)
            attempts.append(attempt)
            if attempt.usable and float(attempt.objective) < best_objective:
                best_point = np.asarray(attempt.candidate_point, dtype=float)
                best_objective = float(attempt.objective)
                best_attempt = len(attempts) - 1
                best_source = attempt.seed_source

        try:
            achieved = np.asarray(self.interpolator.evaluate(best_point), dtype=float)
        except Exception as exc:
            raise ProjectionInvariantError(
                "the selected projection state could not be reevaluated"
            ) from exc
        residual, scaled, norm_inf, distance, objective = self._metrics(
            achieved, requested
        )
        if best_attempt is None:
            quality = ProjectionQuality.DISCRETE_NODE_FALLBACK
        elif (
            attempts[best_attempt].optimizer_success
            and attempts[best_attempt].candidate_from_optimizer
        ):
            quality = ProjectionQuality.OPTIMIZER_CONVERGED
        else:
            quality = ProjectionQuality.BEST_FEASIBLE_ENCOUNTERED
        return ProjectionResult(
            requested_target=requested,
            source_coordinates=best_point,
            achieved_target=achieved,
            residual=residual,
            scaled_residual=scaled,
            residual_scales=self.residual_scales,
            scaled_residual_norm_inf=norm_inf,
            scaled_distance_l2=distance,
            objective=objective,
            quality=quality,
            selected_seed_source=best_source,
            selected_attempt_index=best_attempt,
            nearest_node_flat_index=nearest_flat,
            attempts=tuple(attempts),
        )


__all__ = [
    "DEFAULT_MAX_PROJECTION_INDEX_BYTES",
    "DEFAULT_PROJECTION_CELL_BATCH_SIZE",
    "EoSProjector",
    "ProjectionAttempt",
    "ProjectionConfig",
    "ProjectionConfigurationError",
    "ProjectionError",
    "ProjectionInputError",
    "ProjectionInvariantError",
    "ProjectionQuality",
    "ProjectionResult",
    "ProjectionSeed",
    "ProjectionSeedSource",
]
