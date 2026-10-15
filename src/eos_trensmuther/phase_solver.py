"""Deterministic first-order-aware routing for single-target inversion.

The ordinary root solver remains unchanged for tables without first-order
metadata and for targets outside mixed-phase candidate images.  Candidate
images never assign a phase by themselves: LINE and BINODALS roots are solved
inside their matching source boxes, checked against the original interpolant,
and then checked by the exact representation-specific physical predicate.

This module deliberately stops at the immutable per-target decision.  Dense
pipeline arrays, checkpoint schemas, and process-worker transport belong to
the following integration step.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np

from .reachability import (
    BinodalSubdivisionConfig,
    MixedPhaseCandidateIndex,
    ReachabilityConfig,
)
from .solver import (
    EoSRootSolver,
    InversionResult,
    SolveStrategy,
    SolverConfig,
    SolverInputError,
)
from .tables import EoSError
from .transition import (
    BinodalBand,
    MixedPhaseStrip,
    PhaseLabel,
    PhaseRootSelectionReason,
)


class PhaseSolverError(EoSError):
    """Base exception for phase-aware root routing."""


class PhaseSolverConfigurationError(PhaseSolverError):
    """Raised when ordinary and mixed solver domains are inconsistent."""


class PhaseSolverResourceError(PhaseSolverError):
    """Raised instead of silently truncating matching phase source pieces."""


class PhaseSolverAmbiguityError(PhaseSolverError):
    """Raised when more than one distinct verified mixed root exists."""


class PhaseRootRoute(str, Enum):
    """Physical mapping that supplied the selected root."""

    NO_TRANSITION = "no_transition"
    ORDINARY = "ordinary"
    LINE_REGION_S = "line_region_s"
    BINODALS_ORDINARY = "binodals_ordinary"


class _JacobianCellOrientation(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    SIGN_CHANGING_OR_SINGULAR = "sign_changing_or_singular"
    INVALID = "invalid"


@dataclass(frozen=True)
class _LocalOrientationDecision:
    expected_orientation: Optional[_JacobianCellOrientation]
    candidate_orientations: Tuple[_JacobianCellOrientation, ...]
    selected_index: Optional[int]
    candidate_cells: Tuple[Tuple[int, int], ...]
    surrounding_cell_counts: Tuple[Tuple[str, int], ...]


_JACOBIAN_ROUNDOFF_FACTOR = 64.0


def _scaled_determinant_orientation(
    jacobian: Any,
) -> _JacobianCellOrientation:
    """Classify a 2x2 Jacobian without a tolerance in raw physical units."""

    matrix = np.asarray(jacobian, dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        return _JacobianCellOrientation.INVALID
    determinant = float(np.linalg.det(matrix))
    column_scale = float(
        np.linalg.norm(matrix[:, 0], ord=2)
        * np.linalg.norm(matrix[:, 1], ord=2)
    )
    if not math.isfinite(determinant) or not math.isfinite(column_scale):
        return _JacobianCellOrientation.INVALID
    threshold = _JACOBIAN_ROUNDOFF_FACTOR * np.finfo(float).eps * column_scale
    if determinant > threshold:
        return _JacobianCellOrientation.POSITIVE
    if determinant < -threshold:
        return _JacobianCellOrientation.NEGATIVE
    return _JacobianCellOrientation.SIGN_CHANGING_OR_SINGULAR


def _bilinear_cell_orientation(
    interpolator: Any, cell: Tuple[int, int]
) -> _JacobianCellOrientation:
    """Classify the exact affine determinant range of one bilinear cell."""

    try:
        corners = np.asarray(interpolator.cell_corner_values(cell), dtype=float)
        lower, upper = interpolator.cell_coordinate_bounds(cell)
    except (TypeError, ValueError, IndexError):
        return _JacobianCellOrientation.INVALID
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return _JacobianCellOrientation.INVALID
    widths = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    if widths.shape != (2,) or not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        return _JacobianCellOrientation.INVALID

    # Corner order is (0,0), (0,1), (1,0), (1,1).  For a bilinear
    # map each Jacobian column is affine in the other normalized coordinate,
    # so det(J) is affine and its exact extrema occur at these four corners.
    f00, f01, f10, f11 = corners
    cross = f11 - f10 - f01 + f00
    determinants = []
    scales = []
    for u, v in itertools.product((0.0, 1.0), repeat=2):
        d_t = (f10 - f00 + cross * v) / widths[0]
        d_mu = (f01 - f00 + cross * u) / widths[1]
        jacobian = np.column_stack((d_t, d_mu))
        if not np.all(np.isfinite(jacobian)):
            return _JacobianCellOrientation.INVALID
        determinants.append(float(np.linalg.det(jacobian)))
        scales.append(
            float(np.linalg.norm(d_t, ord=2) * np.linalg.norm(d_mu, ord=2))
        )
    determinant_array = np.asarray(determinants, dtype=float)
    scale = float(np.max(scales))
    if not np.all(np.isfinite(determinant_array)) or not math.isfinite(scale):
        return _JacobianCellOrientation.INVALID
    threshold = _JACOBIAN_ROUNDOFF_FACTOR * np.finfo(float).eps * scale
    if float(np.min(determinant_array)) > threshold:
        return _JacobianCellOrientation.POSITIVE
    if float(np.max(determinant_array)) < -threshold:
        return _JacobianCellOrientation.NEGATIVE
    return _JacobianCellOrientation.SIGN_CHANGING_OR_SINGULAR


def _local_orientation_decision(
    interpolator: Any, candidate_sources: Any
) -> _LocalOrientationDecision:
    """Apply the deterministic one-cell-halo orientation rule."""

    sources = np.asarray(candidate_sources, dtype=float)
    if sources.ndim != 2 or sources.shape[1] != 2 or sources.shape[0] < 2:
        raise ValueError("candidate_sources must have shape (N, 2), N >= 2")
    if (
        interpolator.dimension != 2
        or interpolator.coordinate_names != ("T", "muB")
        or interpolator.field_names != ("e", "nB")
    ):
        raise ValueError(
            "local Jacobian selection requires canonical (T, muB) -> (e, nB)"
        )

    locations = interpolator.locate(sources, side="right")
    raw_cells = np.asarray(locations.cell_indices, dtype=np.intp)
    candidate_cells = tuple(sorted({tuple(map(int, row)) for row in raw_cells}))
    candidate_cell_set = set(candidate_cells)
    surrounding = set()
    for cell in candidate_cells:
        ranges = (
            range(max(0, cell[axis] - 1), min(interpolator.cell_shape[axis], cell[axis] + 2))
            for axis in range(2)
        )
        surrounding.update(itertools.product(*ranges))
    surrounding.difference_update(candidate_cell_set)

    counts = {orientation.value: 0 for orientation in _JacobianCellOrientation}
    usable = set()
    for cell in sorted(surrounding):
        orientation = _bilinear_cell_orientation(interpolator, cell)
        counts[orientation.value] += 1
        if orientation in (
            _JacobianCellOrientation.POSITIVE,
            _JacobianCellOrientation.NEGATIVE,
        ):
            usable.add(orientation)
    expected = next(iter(usable)) if len(usable) == 1 else None

    candidate_orientations = tuple(
        _scaled_determinant_orientation(
            interpolator.jacobian(source, plane_policy="right")
        )
        for source in sources
    )
    matching = (
        []
        if expected is None
        else [
            index
            for index, orientation in enumerate(candidate_orientations)
            if orientation is expected
        ]
    )
    selected_index = matching[0] if len(matching) == 1 else None
    return _LocalOrientationDecision(
        expected_orientation=expected,
        candidate_orientations=candidate_orientations,
        selected_index=selected_index,
        candidate_cells=candidate_cells,
        surrounding_cell_counts=tuple(sorted(counts.items())),
    )


def _immutable_array(value: Any, dtype: Optional[Any] = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
    return immutable.reshape(array.shape)


def _optional_immutable_array(value: Optional[Any]) -> Optional[np.ndarray]:
    return None if value is None else _immutable_array(value, dtype=float)


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a finite non-negative number" % name)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a finite non-negative number" % name) from exc
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError("%s must be a finite non-negative number" % name)
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


@dataclass(frozen=True)
class PhaseSolverConfig:
    """Deterministic comparison tolerances and explicit routing work limit."""

    source_absolute_tolerance: float = 1.0e-10
    source_relative_tolerance: float = 1.0e-9
    duplicate_source_absolute_tolerance: float = 1.0e-8
    duplicate_source_relative_tolerance: float = 1.0e-8
    maximum_candidate_images_per_target: int = 100_000

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_absolute_tolerance",
            _finite_nonnegative(
                self.source_absolute_tolerance, "source_absolute_tolerance"
            ),
        )
        object.__setattr__(
            self,
            "source_relative_tolerance",
            _finite_nonnegative(
                self.source_relative_tolerance, "source_relative_tolerance"
            ),
        )
        object.__setattr__(
            self,
            "duplicate_source_absolute_tolerance",
            _finite_nonnegative(
                self.duplicate_source_absolute_tolerance,
                "duplicate_source_absolute_tolerance",
            ),
        )
        object.__setattr__(
            self,
            "duplicate_source_relative_tolerance",
            _finite_nonnegative(
                self.duplicate_source_relative_tolerance,
                "duplicate_source_relative_tolerance",
            ),
        )
        object.__setattr__(
            self,
            "maximum_candidate_images_per_target",
            _positive_integer(
                self.maximum_candidate_images_per_target,
                "maximum_candidate_images_per_target",
            ),
        )


@dataclass(frozen=True)
class PhaseRootResult:
    """Orthogonal numerical/thermodynamic result for one physical target."""

    target: np.ndarray
    phase_label: PhaseLabel
    route: PhaseRootRoute
    selection_reason: PhaseRootSelectionReason
    source_coordinates: Optional[np.ndarray]
    parameter_coordinates: Optional[np.ndarray]
    selected_result: InversionResult
    ordinary_result: InversionResult
    mixed_result: Optional[InversionResult]
    mixed_candidate: bool
    exact_mixed_candidate: bool
    mixed_candidate_entry_count: int
    mixed_verified_root_count: int
    phase_boundary: bool
    multivalued: bool
    alternate_single_phase_root: Optional[np.ndarray]
    line_mixture_fraction: Optional[float]
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.phase_label, PhaseLabel):
            raise TypeError("phase_label must be a PhaseLabel")
        if not isinstance(self.route, PhaseRootRoute):
            raise TypeError("route must be a PhaseRootRoute")
        if not isinstance(self.selection_reason, PhaseRootSelectionReason):
            raise TypeError("selection_reason must be a PhaseRootSelectionReason")
        if not isinstance(self.selected_result, InversionResult):
            raise TypeError("selected_result must be an InversionResult")
        if not isinstance(self.ordinary_result, InversionResult):
            raise TypeError("ordinary_result must be an InversionResult")
        if self.mixed_result is not None and not isinstance(
            self.mixed_result, InversionResult
        ):
            raise TypeError("mixed_result must be an InversionResult or None")
        object.__setattr__(self, "target", _immutable_array(self.target, dtype=float))
        object.__setattr__(
            self,
            "source_coordinates",
            _optional_immutable_array(self.source_coordinates),
        )
        object.__setattr__(
            self,
            "parameter_coordinates",
            _optional_immutable_array(self.parameter_coordinates),
        )
        object.__setattr__(
            self,
            "alternate_single_phase_root",
            _optional_immutable_array(self.alternate_single_phase_root),
        )
        for name in ("mixed_candidate_entry_count", "mixed_verified_root_count"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or int(value) < 0:
                raise ValueError("%s must be a non-negative integer" % name)
            object.__setattr__(self, name, int(value))
        if self.line_mixture_fraction is not None:
            fraction = float(self.line_mixture_fraction)
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError("line_mixture_fraction must lie in [0, 1]")
            object.__setattr__(self, "line_mixture_fraction", fraction)
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty string")

    @property
    def solved(self) -> bool:
        return self.source_coordinates is not None and self.selected_result.converged


@dataclass(frozen=True)
class _VerifiedMixedRoot:
    result: InversionResult
    parameters: np.ndarray
    source: np.ndarray
    boundary: bool
    restricted: bool


class PhaseAwareRootSolver:
    """Compare verified mixed and ordinary roots using one fixed policy.

    A mixed root takes equilibrium precedence over a distinct ordinary
    single-phase root.  The latter is retained in
    ``alternate_single_phase_root``.  Matching-box and root iteration order
    cannot choose the answer: every matching source piece is attempted,
    duplicate physical roots are collapsed, and multiple distinct mixed roots
    use only the conservative local Jacobian-orientation rule.  If it cannot
    identify exactly one locally consistent root, ambiguity still fails
    explicitly.
    """

    def __init__(
        self,
        ordinary_solver: EoSRootSolver,
        mixed_candidates: Optional[MixedPhaseCandidateIndex] = None,
        config: Optional[PhaseSolverConfig] = None,
    ) -> None:
        if not isinstance(ordinary_solver, EoSRootSolver):
            raise TypeError("ordinary_solver must be an EoSRootSolver")
        if mixed_candidates is not None and not isinstance(
            mixed_candidates, MixedPhaseCandidateIndex
        ):
            raise TypeError(
                "mixed_candidates must be a MixedPhaseCandidateIndex or None"
            )
        self.ordinary_solver = ordinary_solver
        self.mixed_candidates = mixed_candidates
        self.config = PhaseSolverConfig() if config is None else config
        if not isinstance(self.config, PhaseSolverConfig):
            raise TypeError("config must be a PhaseSolverConfig")
        self._mixed_solver: Optional[EoSRootSolver] = None
        if mixed_candidates is not None:
            ordinary = ordinary_solver.interpolator
            mixed = mixed_candidates.interpolator
            if (
                ordinary.dimension != 2
                or ordinary.coordinate_names != ("T", "muB")
                or ordinary.target_names != ("e", "nB")
                or ordinary.field_names != ("e", "nB")
                or mixed.target_names != ("e", "nB")
            ):
                raise PhaseSolverConfigurationError(
                    "phase-aware inversion requires canonical two-dimensional "
                    "(T, muB) -> (e, nB) ordinary data"
                )
            if isinstance(mixed_candidates.model, BinodalBand):
                if mixed is not ordinary:
                    for left, right in zip(mixed.axes, ordinary.axes):
                        if not np.array_equal(left, right):
                            raise PhaseSolverConfigurationError(
                                "BINODALS candidates and the ordinary solver "
                                "must describe the same source grid"
                            )
            elif not isinstance(mixed_candidates.model, MixedPhaseStrip):
                raise PhaseSolverConfigurationError(
                    "unsupported mixed-phase candidate model"
                )
            if (
                mixed_candidates.images.root_residual_tolerance
                != ordinary_solver.config.root_residual_tolerance
            ):
                raise PhaseSolverConfigurationError(
                    "mixed routing padding and the ordinary solver must use "
                    "the same root_residual_tolerance"
                )
            self._mixed_solver = EoSRootSolver(
                mixed,
                ordinary_solver.config,
                residual_scales=mixed_candidates.images.residual_scales,
            )

    @classmethod
    def for_line(
        cls,
        ordinary_interpolator: Any,
        strip: MixedPhaseStrip,
        residual_scales: Any,
        solver_config: Optional[SolverConfig] = None,
        *,
        reachability_config: Optional[ReachabilityConfig] = None,
        config: Optional[PhaseSolverConfig] = None,
    ) -> "PhaseAwareRootSolver":
        ordinary = EoSRootSolver(
            ordinary_interpolator, solver_config, residual_scales=residual_scales
        )
        candidates = MixedPhaseCandidateIndex.for_line(
            strip,
            residual_scales,
            ordinary.config.root_residual_tolerance,
            config=reachability_config,
        )
        return cls(ordinary, candidates, config=config)

    @classmethod
    def for_binodals(
        cls,
        ordinary_interpolator: Any,
        band: BinodalBand,
        residual_scales: Any,
        solver_config: Optional[SolverConfig] = None,
        *,
        subdivision: Optional[BinodalSubdivisionConfig] = None,
        reachability_config: Optional[ReachabilityConfig] = None,
        config: Optional[PhaseSolverConfig] = None,
    ) -> "PhaseAwareRootSolver":
        ordinary = EoSRootSolver(
            ordinary_interpolator, solver_config, residual_scales=residual_scales
        )
        candidates = MixedPhaseCandidateIndex.for_binodals(
            ordinary_interpolator,
            band,
            residual_scales,
            ordinary.config.root_residual_tolerance,
            subdivision=subdivision,
            config=reachability_config,
        )
        return cls(ordinary, candidates, config=config)

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

    def _target(self, target: Any) -> np.ndarray:
        try:
            array = np.asarray(target, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SolverInputError("target must be a finite numeric vector") from exc
        dimension = self.ordinary_solver.interpolator.dimension
        if array.shape != (dimension,) or not np.all(np.isfinite(array)):
            raise SolverInputError(
                "target must be a finite vector with shape (%d,)" % dimension
            )
        return array

    def _validate_initial(self, initial_guess: Optional[Any]) -> None:
        if initial_guess is None:
            return
        try:
            point = np.asarray(initial_guess, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SolverInputError(
                "initial_guess must be a finite numeric vector"
            ) from exc
        dimension = self.ordinary_solver.interpolator.dimension
        if (
            point.shape != (dimension,)
            or not np.all(np.isfinite(point))
            or not self.ordinary_solver.interpolator.contains(
                point, include_boundary_tolerance=False
            )
        ):
            raise SolverInputError(
                "initial_guess must lie in the closed ordinary source domain"
            )

    def _source_close(self, left: np.ndarray, right: np.ndarray) -> bool:
        return bool(
            np.allclose(
                left,
                right,
                rtol=self.config.duplicate_source_relative_tolerance,
                atol=self.config.duplicate_source_absolute_tolerance,
            )
        )

    def _scaled_target_verified(
        self,
        interpolator: Any,
        source: np.ndarray,
        target: np.ndarray,
    ) -> bool:
        values = np.asarray(interpolator.evaluate(source), dtype=float)
        scales = self.mixed_candidates.images.residual_scales
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scaled = (values - target) / scales
        return bool(
            np.all(np.isfinite(scaled))
            and float(np.max(np.abs(scaled)))
            <= self.ordinary_solver.config.root_residual_tolerance
        )

    def _phase_boundary(self, parameters: np.ndarray, source: np.ndarray) -> bool:
        assert self.mixed_candidates is not None
        model = self.mixed_candidates.model
        atol = self.config.source_absolute_tolerance
        rtol = self.config.source_relative_tolerance
        if isinstance(model, MixedPhaseStrip):
            fraction_boundary = bool(
                np.isclose(parameters[1], 0.0, rtol=rtol, atol=atol)
                or np.isclose(parameters[1], 1.0, rtol=rtol, atol=atol)
            )
            critical_muB = float(model.line.critical_point["muB"])
            critical_boundary = bool(
                np.isclose(parameters[0], critical_muB, rtol=rtol, atol=atol)
            )
            return fraction_boundary or critical_boundary
        bounds = np.asarray(model.temperature_bounds(source[1]), dtype=float)
        return bool(
            np.isclose(source[0], bounds[0], rtol=rtol, atol=atol)
            or np.isclose(source[0], bounds[1], rtol=rtol, atol=atol)
        )

    def _physical_source(self, parameters: np.ndarray) -> np.ndarray:
        assert self.mixed_candidates is not None
        model = self.mixed_candidates.model
        if isinstance(model, MixedPhaseStrip):
            return np.asarray(
                model.source_coordinates(parameters[0], parameters[1]), dtype=float
            )
        return np.asarray(parameters, dtype=float)

    def _solve_matching_images(
        self,
        target: np.ndarray,
        entry_indices: np.ndarray,
        strategy: SolveStrategy,
    ) -> Tuple[_VerifiedMixedRoot, ...]:
        assert self.mixed_candidates is not None
        assert self._mixed_solver is not None
        images = self.mixed_candidates.images
        roots = []
        for raw_entry in entry_indices:
            entry = int(raw_entry)
            result = self._mixed_solver.solve_in_cell_box(
                target,
                images.parent_cell_indices[entry],
                images.source_lower[entry],
                images.source_upper[entry],
                strategy=strategy,
            )
            if not result.converged or result.solution is None:
                continue
            parameters = np.asarray(result.solution, dtype=float)
            if not bool(self.mixed_candidates.verify_source(parameters)):
                continue
            if not self._scaled_target_verified(
                self.mixed_candidates.interpolator, parameters, target
            ):
                continue
            source = self._physical_source(parameters)
            roots.append(
                _VerifiedMixedRoot(
                    result=result,
                    parameters=_immutable_array(parameters),
                    source=_immutable_array(source),
                    boundary=self._phase_boundary(parameters, source),
                    restricted=True,
                )
            )
        roots.sort(
            key=lambda root: tuple(float(value) for value in root.source)
            + tuple(float(value) for value in root.parameters)
            + (0 if root.restricted else 1,)
        )
        unique = []
        for root in roots:
            if not any(self._source_close(root.source, kept.source) for kept in unique):
                unique.append(root)
        return tuple(unique)

    def _ordinary_as_mixed(
        self,
        target: np.ndarray,
        ordinary: InversionResult,
    ) -> Optional[_VerifiedMixedRoot]:
        if (
            self.mixed_candidates is None
            or not ordinary.converged
            or ordinary.solution is None
        ):
            return None
        source = np.asarray(ordinary.solution, dtype=float)
        model = self.mixed_candidates.model
        if isinstance(model, BinodalBand):
            if not bool(model.contains_points(source)):
                return None
            parameters = source
        else:
            if not bool(
                model.line.contains_points(
                    source,
                    absolute_tolerance=self.config.source_absolute_tolerance,
                    relative_tolerance=self.config.source_relative_tolerance,
                )
            ):
                return None
            muB = float(source[1])
            endpoints = np.asarray(model.evaluate_endpoints(muB), dtype=float)
            delta = endpoints[1] - endpoints[0]
            scales = self.mixed_candidates.images.residual_scales
            scaled_delta = delta / scales
            denominator = float(np.dot(scaled_delta, scaled_delta))
            if denominator == 0.0:
                fraction = 0.5
            else:
                fraction = float(
                    np.dot((target - endpoints[0]) / scales, scaled_delta)
                    / denominator
                )
            tolerance = self.ordinary_solver.config.root_residual_tolerance
            if fraction < -tolerance or fraction > 1.0 + tolerance:
                return None
            fraction = min(max(fraction, 0.0), 1.0)
            parameters = np.asarray([muB, fraction], dtype=float)
            if not self._scaled_target_verified(
                self.mixed_candidates.interpolator, parameters, target
            ):
                return None
        return _VerifiedMixedRoot(
            result=ordinary,
            parameters=_immutable_array(parameters),
            source=_immutable_array(source),
            boundary=self._phase_boundary(parameters, source),
            restricted=False,
        )

    def solve(
        self,
        target: Any,
        initial_guess: Optional[Any] = None,
        *,
        strategy: Any = SolveStrategy.AUTO,
    ) -> PhaseRootResult:
        """Solve and route one target without conflating phase and status."""

        requested = self._target(target)
        self._validate_initial(initial_guess)
        selected_strategy = self._strategy(strategy)

        if self.mixed_candidates is None:
            ordinary = self.ordinary_solver.solve(
                requested, initial_guess, strategy=selected_strategy
            )
            source = ordinary.solution if ordinary.converged else None
            return PhaseRootResult(
                target=requested,
                phase_label=PhaseLabel.NOT_APPLICABLE,
                route=PhaseRootRoute.NO_TRANSITION,
                selection_reason=PhaseRootSelectionReason.NOT_APPLICABLE,
                source_coordinates=source,
                parameter_coordinates=source,
                selected_result=ordinary,
                ordinary_result=ordinary,
                mixed_result=None,
                mixed_candidate=False,
                exact_mixed_candidate=False,
                mixed_candidate_entry_count=0,
                mixed_verified_root_count=0,
                phase_boundary=False,
                multivalued=False,
                alternate_single_phase_root=None,
                line_mixture_fraction=None,
                message="ordinary inversion used because no first-order model is configured",
            )

        classification = self.mixed_candidates.classify(requested)
        is_candidate = bool(classification.candidate)
        exact_candidate = bool(classification.exact_candidate)
        entries = (
            self.mixed_candidates.matching_entries(requested)
            if is_candidate
            else np.empty(0, dtype=np.int64)
        )
        if entries.size > self.config.maximum_candidate_images_per_target:
            raise PhaseSolverResourceError(
                "target matches %d mixed source images, exceeding "
                "maximum_candidate_images_per_target=%d"
                % (
                    entries.size,
                    self.config.maximum_candidate_images_per_target,
                )
            )
        mixed_roots = list(
            self._solve_matching_images(requested, entries, selected_strategy)
        )

        # Always evaluate the ordinary route for a mixed candidate so a
        # distinct branch is detected rather than chosen by an initial guess.
        ordinary = self.ordinary_solver.solve(
            requested, initial_guess, strategy=selected_strategy
        )
        promoted = self._ordinary_as_mixed(requested, ordinary)
        if promoted is not None and not any(
            self._source_close(promoted.source, root.source)
            for root in mixed_roots
        ):
            mixed_roots.append(promoted)
        mixed_roots.sort(
            key=lambda root: tuple(float(value) for value in root.source)
            + tuple(float(value) for value in root.parameters)
            + (0 if root.restricted else 1,)
        )

        selected_by_jacobian = False
        if len(mixed_roots) > 1 and isinstance(
            self.mixed_candidates.model, BinodalBand
        ):
            decision = _local_orientation_decision(
                self.ordinary_solver.interpolator,
                np.asarray([root.source for root in mixed_roots], dtype=float),
            )
            if decision.selected_index is not None:
                selected_root = mixed_roots[decision.selected_index]
                mixed_roots = [selected_root] + [
                    root
                    for index, root in enumerate(mixed_roots)
                    if index != decision.selected_index
                ]
                selected_by_jacobian = True

        verified_mixed_root_count = len(mixed_roots)
        if len(mixed_roots) > 1 and not selected_by_jacobian:
            coordinates = [root.source.tolist() for root in mixed_roots]
            raise PhaseSolverAmbiguityError(
                "target has %d distinct verified mixed roots at %r; no "
                "mixed-vs-mixed branch precedence is configured"
                % (len(mixed_roots), coordinates)
            )

        if mixed_roots:
            selected = mixed_roots[0]
            alternate = None
            if ordinary.converged and ordinary.solution is not None:
                ordinary_source = np.asarray(ordinary.solution, dtype=float)
                ordinary_is_mixed = promoted is not None
                if (
                    not ordinary_is_mixed
                    and not self._source_close(ordinary_source, selected.source)
                ):
                    alternate = ordinary_source
            model = self.mixed_candidates.model
            line = isinstance(model, MixedPhaseStrip)
            return PhaseRootResult(
                target=requested,
                phase_label=PhaseLabel.MIXED_PHASE,
                route=(
                    PhaseRootRoute.LINE_REGION_S
                    if line
                    else PhaseRootRoute.BINODALS_ORDINARY
                ),
                selection_reason=(
                    PhaseRootSelectionReason.SELECTED_FROM_JACOBIAN
                    if selected_by_jacobian
                    else PhaseRootSelectionReason.UNAMBIGUOUS
                ),
                source_coordinates=selected.source,
                parameter_coordinates=selected.parameters,
                selected_result=selected.result,
                ordinary_result=ordinary,
                mixed_result=selected.result,
                mixed_candidate=is_candidate,
                exact_mixed_candidate=exact_candidate,
                mixed_candidate_entry_count=int(entries.size),
                mixed_verified_root_count=verified_mixed_root_count,
                phase_boundary=selected.boundary,
                multivalued=(
                    verified_mixed_root_count > 1 or alternate is not None
                ),
                alternate_single_phase_root=alternate,
                line_mixture_fraction=(
                    float(selected.parameters[1]) if line else None
                ),
                message=(
                    "one of multiple verified mixed roots selected by unanimous "
                    "local Jacobian-orientation consistency"
                    if selected_by_jacobian
                    else (
                        "verified mixed root selected by equilibrium precedence; "
                        "a distinct ordinary root is retained diagnostically"
                        if alternate is not None
                        else "verified mixed root selected after exact phase validation"
                    )
                ),
            )

        source = ordinary.solution if ordinary.converged else None
        return PhaseRootResult(
            target=requested,
            phase_label=(
                PhaseLabel.SINGLE_PHASE
                if ordinary.converged
                else PhaseLabel.NOT_APPLICABLE
            ),
            route=PhaseRootRoute.ORDINARY,
            selection_reason=(
                PhaseRootSelectionReason.UNAMBIGUOUS
                if ordinary.converged
                else PhaseRootSelectionReason.NOT_APPLICABLE
            ),
            source_coordinates=source,
            parameter_coordinates=source,
            selected_result=ordinary,
            ordinary_result=ordinary,
            mixed_result=None,
            mixed_candidate=is_candidate,
            exact_mixed_candidate=exact_candidate,
            mixed_candidate_entry_count=int(entries.size),
            mixed_verified_root_count=0,
            phase_boundary=False,
            multivalued=False,
            alternate_single_phase_root=None,
            line_mixture_fraction=None,
            message=(
                "ordinary root selected after conservative mixed candidates "
                "failed exact solve-and-membership verification"
                if is_candidate
                else "ordinary inversion used outside all mixed candidate images"
            ),
        )


__all__ = [
    "PhaseAwareRootSolver",
    "PhaseRootResult",
    "PhaseRootRoute",
    "PhaseRootSelectionReason",
    "PhaseSolverAmbiguityError",
    "PhaseSolverConfig",
    "PhaseSolverConfigurationError",
    "PhaseSolverError",
    "PhaseSolverResourceError",
]
