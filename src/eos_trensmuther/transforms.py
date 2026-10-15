"""Constraint-aware coordinate transforms shared by boundary and inversion code.

The tilde variables are based on the full three-charge ideal-QGP convention.
Lower-dimensional tables are reductions of that one convention: metadata
describes the omitted charge densities as homogeneous linear combinations of
the independent density prefix. This keeps the forward and inverse maps
algebraically consistent in every supported dimension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


TILDE_TRANSFORM_NAME = "constrained_ideal_qgp_v2"

CHARGE_DENSITY_NAMES = ("nB", "nQ", "nS")
TILDE_CHEMICAL_NAMES = ("muB_tilde", "muQ_tilde", "muS_tilde")
TILDE_NAMES = ("T_tilde",) + TILDE_CHEMICAL_NAMES

# mu_tilde = A n / T_tilde**2. Tuples keep the module constant immutable.
IDEAL_QGP_DENSITY_TO_CHEMICAL = (
    (5.0, -1.0, 2.0),
    (-1.0, 2.0, -1.0),
    (2.0, -1.0, 2.0),
)

_ENERGY_COEFFICIENT = 19.0 * np.pi**2 / 12.0
_MIN_REDUCED_SINGULAR_VALUE = 1e-12


class TransformDomainError(ValueError):
    """Raised when physical values are outside a transform's domain."""


class TransformConstraintError(ValueError):
    """Raised when charge constraints cannot define stable tilde axes."""


def _immutable_float_array(value: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=float)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=float)
    return immutable.reshape(contiguous.shape)


def _validate_dimension(dimension: Any) -> int:
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TransformConstraintError("dimension must be one of 1, 2, 3, or 4")
    if dimension not in range(1, 5):
        raise TransformConstraintError("dimension must be one of 1, 2, 3, or 4")
    return dimension


@dataclass(frozen=True)
class ChargeConstraintSpec:
    """Resolved homogeneous relations for densities omitted by ``dimension``.

    ``dependent_densities`` maps each omitted density to coefficients of the
    independent density prefix. For example, a two-dimensional heavy-ion
    table uses ``{"nQ": {"nB": 0.4}, "nS": {}}`` for
    ``nQ = 0.4*nB`` and ``nS = 0``. An empty right-hand side means zero.
    """

    dimension: int
    dependent_densities: Mapping[str, Mapping[str, float]]
    source: str = "metadata"
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-8
    model: str = "linear_homogeneous"
    _embedding: np.ndarray = field(init=False, repr=False, compare=False)
    _chemical_map: np.ndarray = field(init=False, repr=False, compare=False)
    _reduced_map: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        dimension = _validate_dimension(self.dimension)
        if self.model != "linear_homogeneous":
            raise TransformConstraintError(
                "charge constraint model must be 'linear_homogeneous'"
            )
        if self.source not in {"default", "metadata"}:
            raise TransformConstraintError(
                "charge constraint source must be 'default' or 'metadata'"
            )
        for name, tolerance in (
            ("absolute_tolerance", self.absolute_tolerance),
            ("relative_tolerance", self.relative_tolerance),
        ):
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or not math.isfinite(float(tolerance))
                or float(tolerance) < 0.0
            ):
                raise TransformConstraintError(
                    "%s must be a finite non-negative number" % name
                )

        independent_count = dimension - 1
        independent = CHARGE_DENSITY_NAMES[:independent_count]
        expected_dependent = CHARGE_DENSITY_NAMES[independent_count:]
        if not isinstance(self.dependent_densities, Mapping):
            raise TransformConstraintError("dependent_densities must be a mapping")
        if not all(isinstance(name, str) for name in self.dependent_densities):
            raise TransformConstraintError(
                "dependent density names must be strings"
            )
        actual_dependent = set(self.dependent_densities)
        expected_set = set(expected_dependent)
        if actual_dependent != expected_set:
            missing = sorted(expected_set - actual_dependent)
            extra = sorted(actual_dependent - expected_set)
            details = []
            if missing:
                details.append("missing %s" % ", ".join(missing))
            if extra:
                details.append("unexpected %s" % ", ".join(extra))
            raise TransformConstraintError(
                "dependent density relations must match dimension %d (%s)"
                % (dimension, "; ".join(details))
            )

        frozen_relations: Dict[str, Mapping[str, float]] = {}
        embedding = np.zeros((len(CHARGE_DENSITY_NAMES), independent_count))
        if independent_count:
            embedding[:independent_count, :] = np.eye(independent_count)
        for row, dependent_name in enumerate(
            expected_dependent, start=independent_count
        ):
            relation = self.dependent_densities[dependent_name]
            if not isinstance(relation, Mapping):
                raise TransformConstraintError(
                    "relation for %s must be a mapping" % dependent_name
                )
            if not all(isinstance(name, str) for name in relation):
                raise TransformConstraintError(
                    "relation source names for %s must be strings"
                    % dependent_name
                )
            unknown = sorted(set(relation) - set(independent))
            if unknown:
                raise TransformConstraintError(
                    "relation for %s references non-independent densities: %s"
                    % (dependent_name, ", ".join(unknown))
                )
            frozen_relation: Dict[str, float] = {}
            for source_name, coefficient in relation.items():
                if (
                    isinstance(coefficient, bool)
                    or not isinstance(coefficient, (int, float))
                    or not math.isfinite(float(coefficient))
                ):
                    raise TransformConstraintError(
                        "coefficient %s <- %s must be finite"
                        % (dependent_name, source_name)
                    )
                numeric = float(coefficient)
                frozen_relation[source_name] = numeric
                embedding[row, independent.index(source_name)] = numeric
            frozen_relations[dependent_name] = MappingProxyType(frozen_relation)

        density_to_chemical = np.asarray(
            IDEAL_QGP_DENSITY_TO_CHEMICAL, dtype=float
        )
        with np.errstate(over="ignore", invalid="ignore"):
            chemical_map = density_to_chemical @ embedding
        reduced_map = chemical_map[:independent_count, :]
        if not np.all(np.isfinite(embedding)) or not np.all(np.isfinite(chemical_map)):
            raise TransformConstraintError(
                "charge constraint coefficients overflow the tilde transform"
            )
        if independent_count:
            try:
                singular_values = np.linalg.svd(reduced_map, compute_uv=False)
            except np.linalg.LinAlgError as exc:
                raise TransformConstraintError(
                    "charge constraints could not be reduced numerically"
                ) from exc
            scale = max(1.0, float(singular_values[0]))
            if float(singular_values[-1]) <= _MIN_REDUCED_SINGULAR_VALUE * scale:
                raise TransformConstraintError(
                    "charge constraints make the independent tilde transform "
                    "singular or numerically unstable"
                )

        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(
            self, "dependent_densities", MappingProxyType(frozen_relations)
        )
        object.__setattr__(self, "absolute_tolerance", float(self.absolute_tolerance))
        object.__setattr__(self, "relative_tolerance", float(self.relative_tolerance))
        object.__setattr__(self, "_embedding", _immutable_float_array(embedding))
        object.__setattr__(
            self, "_chemical_map", _immutable_float_array(chemical_map)
        )
        object.__setattr__(self, "_reduced_map", _immutable_float_array(reduced_map))

    @classmethod
    def default(cls, dimension: int) -> "ChargeConstraintSpec":
        """Return zero-density constraints for every omitted charge."""

        checked_dimension = _validate_dimension(dimension)
        dependent = CHARGE_DENSITY_NAMES[checked_dimension - 1 :]
        return cls(
            dimension=checked_dimension,
            dependent_densities={name: {} for name in dependent},
            source="default",
        )

    @property
    def independent_density_names(self) -> Tuple[str, ...]:
        return CHARGE_DENSITY_NAMES[: self.dimension - 1]

    @property
    def dependent_density_names(self) -> Tuple[str, ...]:
        return CHARGE_DENSITY_NAMES[self.dimension - 1 :]

    @property
    def independent_tilde_names(self) -> Tuple[str, ...]:
        return TILDE_NAMES[: self.dimension]

    @property
    def dependent_tilde_names(self) -> Tuple[str, ...]:
        return TILDE_CHEMICAL_NAMES[self.dimension - 1 :]

    @property
    def density_embedding(self) -> np.ndarray:
        return self._embedding

    @property
    def chemical_map(self) -> np.ndarray:
        return self._chemical_map

    @property
    def reduced_map(self) -> np.ndarray:
        return self._reduced_map

    @property
    def dependent_tilde_relations(self) -> Mapping[str, Mapping[str, float]]:
        """Linear derived-potential relations in independent tilde variables."""

        independent_count = self.dimension - 1
        if independent_count:
            dependent_map = self.chemical_map[independent_count:, :]
            try:
                relation_matrix = np.linalg.solve(
                    self.reduced_map.T, dependent_map.T
                ).T
            except np.linalg.LinAlgError as exc:
                raise TransformConstraintError(
                    "charge constraints could not derive dependent potentials"
                ) from exc
        else:
            relation_matrix = np.zeros((3, 0), dtype=float)
        relations: Dict[str, Mapping[str, float]] = {}
        independent_names = TILDE_CHEMICAL_NAMES[:independent_count]
        for name, coefficients in zip(
            self.dependent_tilde_names, relation_matrix
        ):
            relation = {
                source_name: float(coefficient)
                for source_name, coefficient in zip(
                    independent_names, coefficients
                )
                if coefficient != 0.0
            }
            relations[name] = MappingProxyType(relation)
        return MappingProxyType(relations)

    @property
    def identifier(self) -> str:
        equations = []
        for dependent_name in self.dependent_density_names:
            relation = self.dependent_densities[dependent_name]
            terms = [
                "%s*%s" % (repr(relation[source_name]), source_name)
                for source_name in self.independent_density_names
                if source_name in relation and relation[source_name] != 0.0
            ]
            right = "+".join(terms) if terms else "0"
            equations.append("%s=%s" % (dependent_name, right))
        suffix = ",".join(equations) if equations else "full"
        return "%s:%s" % (TILDE_TRANSFORM_NAME, suffix)


@dataclass(frozen=True)
class TildeCoordinateSet:
    """Independent inversion coordinates plus derived tilde output columns."""

    independent_values: np.ndarray
    independent_names: Tuple[str, ...]
    dependent_values: np.ndarray
    dependent_names: Tuple[str, ...]
    dependent_relations: Mapping[str, Mapping[str, float]]
    transform_id: str

    def __post_init__(self) -> None:
        independent = np.asarray(self.independent_values, dtype=float)
        dependent = np.asarray(self.dependent_values, dtype=float)
        if independent.ndim != 2 or dependent.ndim != 2:
            raise ValueError("tilde coordinate arrays must be two-dimensional")
        if independent.shape[0] != dependent.shape[0]:
            raise ValueError("independent and dependent coordinates must align")
        if independent.shape[1] != len(self.independent_names):
            raise ValueError("independent coordinate names do not match values")
        if dependent.shape[1] != len(self.dependent_names):
            raise ValueError("dependent coordinate names do not match values")
        if not np.all(np.isfinite(independent)) or not np.all(np.isfinite(dependent)):
            raise TransformDomainError(
                "tilde transformation produced a non-finite coordinate"
            )
        object.__setattr__(
            self, "independent_values", _immutable_float_array(independent)
        )
        object.__setattr__(self, "dependent_values", _immutable_float_array(dependent))
        object.__setattr__(self, "independent_names", tuple(self.independent_names))
        object.__setattr__(self, "dependent_names", tuple(self.dependent_names))
        frozen_relations = {
            name: MappingProxyType(dict(relation))
            for name, relation in self.dependent_relations.items()
        }
        object.__setattr__(
            self, "dependent_relations", MappingProxyType(frozen_relations)
        )

    @property
    def all_values(self) -> np.ndarray:
        """Return ``T_tilde`` and all three tilde chemical potentials."""

        return _immutable_float_array(
            np.column_stack((self.independent_values, self.dependent_values))
        )

    @property
    def all_names(self) -> Tuple[str, ...]:
        return self.independent_names + self.dependent_names


class ConstrainedTildeTransform:
    """Bidirectional ideal-QGP transform reduced by density constraints."""

    def __init__(self, constraints: ChargeConstraintSpec) -> None:
        if not isinstance(constraints, ChargeConstraintSpec):
            raise TypeError("constraints must be a ChargeConstraintSpec")
        self._constraints = constraints

    @property
    def constraints(self) -> ChargeConstraintSpec:
        return self._constraints

    @property
    def identifier(self) -> str:
        return self.constraints.identifier

    def targets_to_coordinates(self, targets: np.ndarray) -> TildeCoordinateSet:
        """Convert target prefixes to independent and derived tilde values."""

        values = self._validate_matrix(targets, self.constraints.dimension, "targets")
        energy = values[:, 0]
        if np.any(energy < 0.0):
            raise TransformDomainError(
                "tilde temperature is undefined for negative energy"
            )
        independent_count = self.constraints.dimension - 1
        if independent_count and np.any(energy <= 0.0):
            raise TransformDomainError(
                "tilde chemical potentials require strictly positive energy"
            )

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            temperature = np.power(energy / _ENERGY_COEFFICIENT, 0.25)
            if independent_count:
                independent_densities = values[:, 1:]
                full_chemical = (
                    independent_densities @ self.constraints.chemical_map.T
                ) / temperature[:, np.newaxis] ** 2
            else:
                full_chemical = np.zeros((values.shape[0], 3), dtype=float)
            independent = np.column_stack(
                (temperature, full_chemical[:, :independent_count])
            )
            dependent = full_chemical[:, independent_count:]
        return TildeCoordinateSet(
            independent_values=independent,
            independent_names=self.constraints.independent_tilde_names,
            dependent_values=dependent,
            dependent_names=self.constraints.dependent_tilde_names,
            dependent_relations=self.constraints.dependent_tilde_relations,
            transform_id=self.identifier,
        )

    def independent_to_targets(self, independent: np.ndarray) -> np.ndarray:
        """Invert independent tilde coordinates to the target-density prefix."""

        values = self._validate_matrix(
            independent, self.constraints.dimension, "independent tilde coordinates"
        )
        temperature = values[:, 0]
        if np.any(temperature < 0.0):
            raise TransformDomainError("T_tilde cannot be negative")
        independent_count = self.constraints.dimension - 1
        if independent_count and np.any(temperature <= 0.0):
            raise TransformDomainError(
                "tilde chemical potentials require strictly positive T_tilde"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            energy = _ENERGY_COEFFICIENT * temperature**4
            if independent_count:
                scaled_chemical = values[:, 1:] * temperature[:, np.newaxis] ** 2
                densities = np.linalg.solve(
                    self.constraints.reduced_map, scaled_chemical.T
                ).T
                targets = np.column_stack((energy, densities))
            else:
                targets = energy[:, np.newaxis]
        if not np.all(np.isfinite(targets)):
            raise TransformDomainError(
                "inverse tilde transformation produced a non-finite target"
            )
        return _immutable_float_array(targets)

    def expand_independent(self, independent: np.ndarray) -> TildeCoordinateSet:
        """Add derived tilde chemical potentials without adding grid axes."""

        values = self._validate_matrix(
            independent, self.constraints.dimension, "independent tilde coordinates"
        )
        temperature = values[:, :1]
        independent_chemical = values[:, 1:]
        if np.any(temperature < 0.0):
            raise TransformDomainError("T_tilde cannot be negative")
        if independent_chemical.shape[1] and np.any(temperature <= 0.0):
            raise TransformDomainError(
                "tilde chemical potentials require strictly positive T_tilde"
            )
        relation_matrix = self._dependent_relation_matrix()
        dependent = independent_chemical @ relation_matrix.T
        return TildeCoordinateSet(
            independent_values=values,
            independent_names=self.constraints.independent_tilde_names,
            dependent_values=dependent,
            dependent_names=self.constraints.dependent_tilde_names,
            dependent_relations=self.constraints.dependent_tilde_relations,
            transform_id=self.identifier,
        )

    def _dependent_relation_matrix(self) -> np.ndarray:
        relations = self.constraints.dependent_tilde_relations
        independent_names = TILDE_CHEMICAL_NAMES[: self.constraints.dimension - 1]
        return np.asarray(
            [
                [relations[name].get(source_name, 0.0) for source_name in independent_names]
                for name in self.constraints.dependent_tilde_names
            ],
            dtype=float,
        ).reshape(len(self.constraints.dependent_tilde_names), len(independent_names))

    @staticmethod
    def _validate_matrix(values: Any, columns: int, label: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != columns:
            raise ValueError(
                "%s must have shape (number_of_points, %d)" % (label, columns)
            )
        if not np.all(np.isfinite(array)):
            raise TransformDomainError("%s must be finite" % label)
        return array


def _resolve_constraints(
    dimension: int, constraints: Optional[ChargeConstraintSpec]
) -> ChargeConstraintSpec:
    checked_dimension = _validate_dimension(dimension)
    if constraints is None:
        return ChargeConstraintSpec.default(checked_dimension)
    if not isinstance(constraints, ChargeConstraintSpec):
        raise TypeError("constraints must be a ChargeConstraintSpec or None")
    if constraints.dimension != checked_dimension:
        raise TransformConstraintError(
            "constraint dimension %d does not match requested dimension %d"
            % (constraints.dimension, checked_dimension)
        )
    return constraints


def transform_targets_to_tilde(
    targets: np.ndarray,
    dimension: int,
    constraints: Optional[ChargeConstraintSpec] = None,
) -> TildeCoordinateSet:
    """Return independent inversion axes and all derived tilde outputs."""

    resolved = _resolve_constraints(dimension, constraints)
    return ConstrainedTildeTransform(resolved).targets_to_coordinates(targets)


def targets_to_tilde(
    targets: np.ndarray,
    dimension: int,
    constraints: Optional[ChargeConstraintSpec] = None,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Compatibility wrapper returning only independent tilde coordinates."""

    result = transform_targets_to_tilde(targets, dimension, constraints)
    return result.independent_values, result.independent_names


def tilde_to_targets(
    independent: np.ndarray,
    dimension: int,
    constraints: Optional[ChargeConstraintSpec] = None,
) -> np.ndarray:
    """Convert independent tilde coordinates back to inversion targets."""

    resolved = _resolve_constraints(dimension, constraints)
    return ConstrainedTildeTransform(resolved).independent_to_targets(independent)


def expand_independent_tilde(
    independent: np.ndarray,
    dimension: int,
    constraints: Optional[ChargeConstraintSpec] = None,
) -> TildeCoordinateSet:
    """Compute dependent tilde potentials for future output serialization."""

    resolved = _resolve_constraints(dimension, constraints)
    return ConstrainedTildeTransform(resolved).expand_independent(independent)


__all__ = [
    "CHARGE_DENSITY_NAMES",
    "ChargeConstraintSpec",
    "ConstrainedTildeTransform",
    "IDEAL_QGP_DENSITY_TO_CHEMICAL",
    "TILDE_CHEMICAL_NAMES",
    "TILDE_NAMES",
    "TILDE_TRANSFORM_NAME",
    "TildeCoordinateSet",
    "TransformConstraintError",
    "TransformDomainError",
    "expand_independent_tilde",
    "targets_to_tilde",
    "tilde_to_targets",
    "transform_targets_to_tilde",
]
