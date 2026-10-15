"""Metadata-driven loading and validation of equation-of-state tables.

This module deliberately stops before interpolation and inversion.  Its job is
to turn a user-described text table into one deterministic, unit-aware,
complete rectilinear grid on which those later operations can safely depend.

The validation flag is runtime state.  It is never read from metadata: only a
successful validation of the currently loaded arrays can set
``ForwardTable.is_validated`` to ``True``.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import math
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._array_storage import (
    immutable_array as _storage_immutable_array,
    is_read_only_mmap_backed,
)
from ._table_sanity import INVALID, NEGATIVE, POSITIVE, SINGULAR, scan_table_geometry
from .transforms import (
    ChargeConstraintSpec,
    TransformConstraintError,
    TransformDomainError,
    transform_targets_to_tilde,
)


HBARC_GEV_FM = 0.1973269804
"""The reduced Planck constant times c, in GeV fm."""

METADATA_SCHEMA_VERSION = 1
TABLE_SNAPSHOT_FINGERPRINT_VERSION = 2
MAX_ABSOLUTE_TRANSFORM_POWER = 64
MAX_TABLE_COLUMNS = 10_000

_COORDINATE_NAMES = ("T", "muB", "muQ", "muS")
_TARGET_NAMES = ("e", "nB", "nQ", "nS")
_SEVERITIES = ("error", "warning", "info")
_FIRST_ORDER_REPRESENTATIONS = ("line", "binodals")
_FIRST_ORDER_REQUIRED_COLUMNS = MappingProxyType(
    {
        "line": ("muB", "T"),
        "binodals": ("muB", "T_dilute", "T_dense"),
    }
)
_CANONICAL_ENERGY_UNIT_SCALES = MappingProxyType(
    {
        "GeV": 1.0,
        "MeV": 1.0e-3,
    }
)


class EoSError(Exception):
    """Base exception for EoS ingestion errors."""


class MetadataError(EoSError):
    """Raised when a metadata file cannot be parsed or is invalid."""


class DataLoadError(EoSError):
    """Raised when a numerical table cannot be loaded or interpreted."""


class BoundaryComputationError(EoSError):
    """Raised when table values cannot define the requested boundary space."""


class EoSStateError(EoSError):
    """Raised when an operation is requested before its prerequisites."""


class EoSNotValidatedError(EoSStateError):
    """Raised when validated data are requested from an invalid EoS."""


class EoSValidationError(EoSError):
    """Raised when strict validation finds one or more errors."""

    def __init__(self, report: "ValidationReport") -> None:
        self.report = report
        codes = ", ".join(issue.code for issue in report.errors)
        message = "EoS validation failed"
        if codes:
            message += ": " + codes
        super().__init__(message)


def _freeze(value: Any) -> Any:
    """Recursively make JSON-like data immutable."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    """Convert immutable/numpy values to JSON-serializable Python values."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _snapshot_plain(value: Any) -> Any:
    """Return a deterministic JSON value for validated schema semantics."""

    if is_dataclass(value):
        return {
            descriptor.name: _snapshot_plain(getattr(value, descriptor.name))
            for descriptor in dataclass_fields(value)
            if not descriptor.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(key): _snapshot_plain(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_snapshot_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _update_digest_array(digest: Any, value: Any) -> None:
    array = np.asarray(value)
    canonical_dtype = array.dtype.newbyteorder("<")
    if array.dtype != canonical_dtype:
        array = array.astype(canonical_dtype, copy=True)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating) and np.any(np.isnan(array)):
        array = array.copy()
        array[np.isnan(array)] = np.nan
    descriptor = "%s:%s" % (
        canonical_dtype.str,
        ",".join(str(size) for size in array.shape),
    )
    encoded = descriptor.encode("ascii")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(memoryview(array).cast("B"))


def _immutable_array(value: Any) -> np.ndarray:
    """Return a C-contiguous ndarray backed by immutable ``bytes``.

    Merely calling ``array.setflags(write=False)`` is insufficient when an
    exposed array owns its memory: a caller can turn the flag back on without
    notifying the EoS object.  A bytes-backed array cannot be made writable,
    so the runtime validation state cannot silently become stale.
    """

    return _storage_immutable_array(value)


@dataclass(frozen=True)
class ValidationIssue:
    """One stable, machine-readable validation diagnostic."""

    severity: str
    code: str
    message: str
    path: Optional[str] = None
    context: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITIES:
            raise ValueError("invalid validation severity: %s" % self.severity)
        object.__setattr__(self, "context", _freeze(dict(self.context)))

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "context": _plain(self.context),
        }
        if self.path is not None:
            result["path"] = self.path
        return result


@dataclass(frozen=True)
class ValidationReport:
    """All diagnostics and measurements produced by one validation run."""

    issues: Tuple[ValidationIssue, ...] = ()
    statistics: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "statistics", _freeze(dict(self.statistics)))

    @property
    def errors(self) -> Tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> Tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def info(self) -> Tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "info")

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if not self.is_valid:
            raise EoSValidationError(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "statistics": _plain(self.statistics),
        }


@dataclass(frozen=True)
class TransformSpec:
    """Explicit conversion from a raw column to its canonical unit."""

    scale_factor: float = 1.0
    temperature_power: int = 0
    hbarc_power: int = 0


@dataclass(frozen=True)
class FieldSpec:
    """Description of one selected table column."""

    name: str
    role: str
    column: int
    source_unit: str
    canonical_unit: str
    transform: TransformSpec
    nonfinite_policy: str
    output: bool = True
    interpolation: Optional[str] = None
    source_label: Optional[str] = None
    stored_as: Optional[str] = None
    description: Optional[str] = None

    @property
    def uses_temperature_scaling(self) -> bool:
        return self.transform.temperature_power != 0


@dataclass(frozen=True)
class FormatSpec:
    """Rules for reading the numerical text file."""

    delimiter: Optional[str]
    comment_prefix: Optional[str]
    skip_rows: int
    encoding: str
    expected_columns: Optional[int]
    missing_values: Tuple[str, ...]


@dataclass(frozen=True)
class GridSpec:
    """Validation policy for a complete rectilinear grid."""

    kind: str
    require_complete: bool
    require_uniform: bool
    reorder: bool
    duplicate_policy: str
    absolute_tolerance: float
    relative_tolerance: float
    min_axis_points: int
    expected_shape: Optional[Tuple[int, ...]]
    expected_bounds: Mapping[str, Tuple[float, float]]
    expected_spacing: Mapping[str, float]


@dataclass(frozen=True)
class CoexistenceColumnSpec:
    """One semantic column selected from a first-order sidecar."""

    name: str
    column: int
    source_unit: str
    canonical_unit: str
    scale_factor: float


@dataclass(frozen=True)
class CoexistenceDataSpec:
    """Validated text format and semantic columns of a coexistence sidecar."""

    file: str
    delimiter: Optional[str]
    comment_prefix: Optional[str]
    skip_rows: int
    encoding: str
    expected_columns: int
    columns: Mapping[str, CoexistenceColumnSpec]
    fields: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", MappingProxyType(dict(self.columns)))
        object.__setattr__(self, "fields", tuple(self.fields))


@dataclass(frozen=True)
class FirstOrderTransitionSpec:
    """Computational first-order-transition metadata for a 2D table."""

    representation: str
    kind: str
    coexistence_data: CoexistenceDataSpec
    phase_fraction_field: Optional[str] = None
    mixed_phase_condition: Optional[str] = None
    critical_point_name: Optional[str] = None
    critical_point_coordinates: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    critical_point_absolute_tolerance: float = 0.0
    critical_point_relative_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.representation not in _FIRST_ORDER_REPRESENTATIONS:
            raise ValueError("unsupported first-order representation")
        object.__setattr__(
            self,
            "critical_point_coordinates",
            MappingProxyType(dict(self.critical_point_coordinates)),
        )


@dataclass(frozen=True)
class CoexistenceTable:
    """Canonical, immutable semantic data loaded from a coexistence sidecar."""

    representation: str
    names: Tuple[str, ...]
    units: Tuple[str, ...]
    values: np.ndarray
    source_row_count: int
    source_column_count: int
    fingerprint: str
    critical_endpoint: Optional[str] = None

    def __post_init__(self) -> None:
        names = tuple(self.names)
        units = tuple(self.units)
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("coexistence values must match semantic names")
        if len(units) != len(names):
            raise ValueError("coexistence units must match semantic names")
        if len(set(names)) != len(names):
            raise ValueError("coexistence semantic names must be unique")
        if not np.all(np.isfinite(values)):
            raise ValueError("coexistence values must be finite")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "values", _immutable_array(values))

    def field(self, name: str) -> np.ndarray:
        """Return one immutable canonical semantic column."""

        try:
            index = self.names.index(name)
        except ValueError as exc:
            raise KeyError("unknown coexistence field: %s" % name) from exc
        return self.values[:, index]

    @property
    def bounds(self) -> Mapping[str, Tuple[float, float]]:
        return MappingProxyType(
            {
                name: (float(np.min(column)), float(np.max(column)))
                for name, column in zip(self.names, self.values.T)
            }
        )


@dataclass(frozen=True)
class TableSchema:
    """Validated version-1 metadata."""

    schema_version: int
    name: str
    table_revision: Optional[str]
    description: Optional[str]
    dimension: int
    data_file: str
    format: FormatSpec
    fields: Mapping[str, FieldSpec]
    grid: GridSpec
    charge_constraints: ChargeConstraintSpec
    scientific: Mapping[str, Any]
    first_order_transition: Optional[FirstOrderTransitionSpec] = None

    @property
    def coordinate_names(self) -> Tuple[str, ...]:
        return _COORDINATE_NAMES[: self.dimension]

    @property
    def target_names(self) -> Tuple[str, ...]:
        return _TARGET_NAMES[: self.dimension]

    @property
    def canonical_names(self) -> Tuple[str, ...]:
        return self.coordinate_names + self.target_names

    @property
    def canonical_columns(self) -> Tuple[int, ...]:
        return tuple(self.fields[name].column for name in self.canonical_names)

    @property
    def target_temperature_powers(self) -> Tuple[int, ...]:
        return tuple(
            self.fields[name].transform.temperature_power for name in self.target_names
        )

    @property
    def auxiliary_names(self) -> Tuple[str, ...]:
        return tuple(
            sorted(name for name, spec in self.fields.items() if spec.role == "auxiliary")
        )


@dataclass(frozen=True)
class RectilinearGrid:
    """Canonical coordinate axes and row ordering for a validated table."""

    coordinate_names: Tuple[str, ...]
    axes: Tuple[np.ndarray, ...]
    shape: Tuple[int, ...]
    uniform_by_axis: Tuple[bool, ...]
    row_permutation: np.ndarray
    was_reordered: bool
    expected_cells: int
    duplicate_count: int
    missing_count: int

    @property
    def bounds(self) -> Mapping[str, Tuple[float, float]]:
        return MappingProxyType(
            {
                name: (float(axis[0]), float(axis[-1]))
                for name, axis in zip(self.coordinate_names, self.axes)
            }
        )


@dataclass(frozen=True)
class BoundaryEnvelope:
    """Component-wise envelope of the validated forward-table nodes."""

    coordinate_system: str
    names: Tuple[str, ...]
    units: Tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    sample_count: int
    coverage: str = "validated_grid_nodes"
    transform_id: Optional[str] = None
    degenerate_names: Tuple[str, ...] = ()
    dependent_names: Tuple[str, ...] = ()
    dependent_lower: Optional[np.ndarray] = None
    dependent_upper: Optional[np.ndarray] = None
    dependent_relations: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        lower = _immutable_array(np.asarray(self.lower, dtype=float))
        upper = _immutable_array(np.asarray(self.upper, dtype=float))
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("boundary arrays must be one-dimensional and equally sized")
        if len(self.names) != lower.size or len(self.units) != lower.size:
            raise ValueError("boundary names and units must match the array dimension")
        if np.any(lower > upper):
            raise ValueError("a lower boundary cannot exceed its upper boundary")
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "units", tuple(self.units))
        object.__setattr__(self, "degenerate_names", tuple(self.degenerate_names))
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        dependent_names = tuple(self.dependent_names)
        if self.dependent_lower is None:
            dependent_lower = np.empty(0, dtype=float)
        else:
            dependent_lower = np.asarray(self.dependent_lower, dtype=float)
        if self.dependent_upper is None:
            dependent_upper = np.empty(0, dtype=float)
        else:
            dependent_upper = np.asarray(self.dependent_upper, dtype=float)
        if (
            dependent_lower.ndim != 1
            or dependent_upper.shape != dependent_lower.shape
            or dependent_lower.size != len(dependent_names)
        ):
            raise ValueError(
                "dependent boundary names and arrays must have matching lengths"
            )
        if np.any(dependent_lower > dependent_upper):
            raise ValueError(
                "a dependent lower boundary cannot exceed its upper boundary"
            )
        relations = {
            name: MappingProxyType(dict(relation))
            for name, relation in self.dependent_relations.items()
        }
        if set(relations) != set(dependent_names):
            raise ValueError(
                "dependent relations must match the dependent coordinate names"
            )
        object.__setattr__(self, "dependent_names", dependent_names)
        object.__setattr__(
            self, "dependent_lower", _immutable_array(dependent_lower)
        )
        object.__setattr__(
            self, "dependent_upper", _immutable_array(dependent_upper)
        )
        object.__setattr__(
            self, "dependent_relations", MappingProxyType(relations)
        )

    @property
    def ranges(self) -> np.ndarray:
        """Return an immutable ``(dimension, 2)`` lower/upper array."""

        return _immutable_array(np.column_stack((self.lower, self.upper)))

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "coordinate_system": self.coordinate_system,
            "transform_id": self.transform_id,
            "sample_count": self.sample_count,
            "coverage": self.coverage,
            "degenerate_names": list(self.degenerate_names),
            "cartesian_product_reachability": "not_guaranteed",
            "coordinates": {
                name: {
                    "minimum": float(lower),
                    "maximum": float(upper),
                    "unit": unit,
                }
                for name, unit, lower, upper in zip(
                    self.names, self.units, self.lower, self.upper
                )
            },
        }
        result["dependent_coordinates"] = {
            name: {
                "minimum": float(lower),
                "maximum": float(upper),
                "unit": "GeV",
                "kind": "derived_output",
                "linear_combination": dict(self.dependent_relations[name]),
            }
            for name, lower, upper in zip(
                self.dependent_names, self.dependent_lower, self.dependent_upper
            )
        }
        return result


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-standard JSON numeric constant: %s" % value)


def _issue(
    issues: List[ValidationIssue],
    severity: str,
    code: str,
    message: str,
    path: Optional[str] = None,
    **context: Any,
) -> None:
    issues.append(
        ValidationIssue(
            severity=severity,
            code=code,
            message=message,
            path=path,
            context=context,
        )
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_float(value: Any) -> Optional[float]:
    """Safely coerce a JSON number, including arbitrarily large integers."""

    if not _is_number(value):
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _check_unknown_keys(
    value: Mapping[str, Any],
    allowed: Iterable[str],
    path: str,
    issues: List[ValidationIssue],
) -> None:
    unknown = sorted(set(value) - set(allowed))
    for key in unknown:
        _issue(
            issues,
            "error",
            "metadata.unknown_key",
            "Unknown metadata key '%s'." % key,
            "%s.%s" % (path, key) if path else key,
        )


class ForwardTable:
    """Read, interpret, and validate one metadata-described EoS table.

    Parameters
    ----------
    metadata_path:
        Path to the JSON metadata.  ``data_file`` and scientific sidecars are
        resolved relative to this file, not relative to the process cwd.
    data_path:
        Optional explicit override for the numerical table, useful for tests
        and controlled deployments.  Relative overrides are resolved from the
        process cwd.

    Notes
    -----
    Construction performs no heavy I/O.  Call :meth:`validate`, or use
    :meth:`load_validated` for strict production loading.
    """

    def __init__(
        self,
        metadata_path: Any,
        data_path: Optional[Any] = None,
        *,
        _allow_unsuitable_geometry: bool = False,
    ) -> None:
        if not isinstance(_allow_unsuitable_geometry, bool):
            raise TypeError("_allow_unsuitable_geometry must be a boolean")
        self._metadata_path = Path(metadata_path).expanduser().resolve()
        self._data_path_override = (
            Path(data_path).expanduser().resolve() if data_path is not None else None
        )
        self._metadata: Optional[TableSchema] = None
        self._metadata_issues: Tuple[ValidationIssue, ...] = ()
        self._raw_data: Optional[np.ndarray] = None
        self._input_fields: Optional[Mapping[str, np.ndarray]] = None
        self._fields: Optional[Mapping[str, np.ndarray]] = None
        self._grid_fields: Optional[Mapping[str, np.ndarray]] = None
        self._grid: Optional[RectilinearGrid] = None
        self._coexistence_table: Optional[CoexistenceTable] = None
        self._report: Optional[ValidationReport] = None
        self._is_validated = False
        self._snapshot_fingerprint: Optional[str] = None
        # Internal analytical-test escape hatch. Production loading remains
        # strict: a wholly degenerate or non-finite local Jacobian is fatal.
        # Tests of forward interpolation and proof-safe handling occasionally
        # require deliberately non-invertible maps, which opt in explicitly.
        self._allow_unsuitable_geometry = _allow_unsuitable_geometry

    @classmethod
    def load_validated(
        cls,
        metadata_path: Any,
        data_path: Optional[Any] = None,
        *,
        _allow_unsuitable_geometry: bool = False,
    ) -> "ForwardTable":
        """Load an EoS and raise if its metadata or table is invalid."""

        eos = cls(
            metadata_path,
            data_path=data_path,
            _allow_unsuitable_geometry=_allow_unsuitable_geometry,
        )
        eos.validate(raise_on_error=True)
        return eos

    def _attach_validated_mmap_snapshot(
        self,
        *,
        axes: Sequence[np.ndarray],
        fields: Mapping[str, np.ndarray],
        row_permutation: np.ndarray,
        uniform_by_axis: Sequence[bool],
        was_reordered: bool,
        duplicate_count: int,
        missing_count: int,
        snapshot_fingerprint: str,
    ) -> None:
        """Replace ordinary table storage with a validated read-only mmap view.

        This is an internal spawn-worker/storage hook.  It deliberately does
        not support first-order sidecars, whose validated phase-aware path is
        outside the shared ordinary-table rollout.  The serialized snapshot
        is created only from an already validated :class:`ForwardTable`.
        """

        if self._metadata is None:
            self.load_metadata()
        assert self._metadata is not None
        if self._metadata.first_order_transition is not None:
            raise EoSStateError(
                "memory-mapped ordinary snapshots do not support first-order tables"
            )
        coordinate_names = self._metadata.coordinate_names
        mapped_axes = tuple(np.asarray(axis) for axis in axes)
        if len(mapped_axes) != self._metadata.dimension:
            raise EoSStateError("mmap snapshot has the wrong number of axes")
        for axis in mapped_axes:
            if (
                axis.ndim != 1
                or axis.size < 2
                or axis.dtype != np.dtype(np.float64)
                or not axis.flags.c_contiguous
                or axis.flags.writeable
                or not is_read_only_mmap_backed(axis)
                or not np.all(np.isfinite(axis))
                or np.any(np.diff(axis) <= 0.0)
            ):
                raise EoSStateError(
                    "mmap snapshot axes must be finite increasing read-only float64 mappings"
                )
        shape = tuple(int(axis.size) for axis in mapped_axes)
        expected_cells = math.prod(shape)
        expected_names = set(self._metadata.fields)
        if set(fields) != expected_names:
            raise EoSStateError(
                "mmap snapshot field names do not match table metadata"
            )
        mapped_fields: Dict[str, np.ndarray] = {}
        shaped_fields: Dict[str, np.ndarray] = {}
        for name in sorted(fields):
            values = np.asarray(fields[name])
            if (
                values.shape != (expected_cells,)
                or values.dtype != np.dtype(np.float64)
                or not values.flags.c_contiguous
                or values.flags.writeable
                or not is_read_only_mmap_backed(values)
            ):
                raise EoSStateError(
                    "mmap snapshot field '%s' is not a complete read-only float64 mapping"
                    % name
                )
            mapped_fields[name] = values
            shaped_fields[name] = values.reshape(shape, order="C")
        permutation = np.asarray(row_permutation)
        if (
            permutation.shape != (expected_cells,)
            or permutation.dtype != np.dtype(np.int64)
            or not permutation.flags.c_contiguous
            or permutation.flags.writeable
            or not is_read_only_mmap_backed(permutation)
        ):
            raise EoSStateError(
                "mmap snapshot row permutation is not a complete read-only int64 mapping"
            )
        uniform = tuple(bool(value) for value in uniform_by_axis)
        if len(uniform) != len(shape):
            raise EoSStateError("mmap snapshot uniform-axis flags are invalid")
        if not isinstance(snapshot_fingerprint, str) or not snapshot_fingerprint:
            raise EoSStateError("mmap snapshot fingerprint is invalid")

        self._raw_data = None
        self._input_fields = None
        self._fields = MappingProxyType(mapped_fields)
        self._grid_fields = MappingProxyType(shaped_fields)
        self._grid = RectilinearGrid(
            coordinate_names=coordinate_names,
            axes=mapped_axes,
            shape=shape,
            uniform_by_axis=uniform,
            row_permutation=permutation,
            was_reordered=bool(was_reordered),
            expected_cells=expected_cells,
            duplicate_count=int(duplicate_count),
            missing_count=int(missing_count),
        )
        self._coexistence_table = None
        self._report = ValidationReport(
            (),
            {
                "storage": {
                    "backend": "numpy_mmap_read_only",
                    "source_snapshot_fingerprint": snapshot_fingerprint,
                }
            },
        )
        self._snapshot_fingerprint = snapshot_fingerprint
        self._is_validated = True

    @property
    def is_validated(self) -> bool:
        """Whether the current in-memory table snapshot passed validation."""

        return self._is_validated

    @property
    def metadata_path(self) -> Path:
        """Immutable path of the metadata snapshot owned by this object."""

        return self._metadata_path

    @property
    def metadata(self) -> Optional[TableSchema]:
        return self._metadata

    @property
    def data_path(self) -> Optional[Path]:
        if self._data_path_override is not None:
            return self._data_path_override
        if self._metadata is None:
            return None
        return self.metadata_path.parent / self._metadata.data_file

    @property
    def raw_data(self) -> Optional[np.ndarray]:
        """Raw input-order numeric data, exposed as a read-only array."""

        return self._raw_data

    @property
    def report(self) -> Optional[ValidationReport]:
        return self._report

    @property
    def grid(self) -> RectilinearGrid:
        self.require_validated()
        assert self._grid is not None
        return self._grid

    @property
    def axes(self) -> Tuple[np.ndarray, ...]:
        return self.grid.axes

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.grid.shape

    @property
    def fields(self) -> Mapping[str, np.ndarray]:
        """Canonical-unit fields flattened in canonical grid order."""

        self.require_validated()
        assert self._fields is not None
        return self._fields

    @property
    def grid_fields(self) -> Mapping[str, np.ndarray]:
        """Canonical-unit fields reshaped to :attr:`shape`."""

        self.require_validated()
        assert self._grid_fields is not None
        return self._grid_fields

    @property
    def coordinate_names(self) -> Tuple[str, ...]:
        if self._metadata is None:
            raise EoSStateError("metadata have not been loaded")
        return self._metadata.coordinate_names

    @property
    def target_names(self) -> Tuple[str, ...]:
        if self._metadata is None:
            raise EoSStateError("metadata have not been loaded")
        return self._metadata.target_names

    @property
    def auxiliary_names(self) -> Tuple[str, ...]:
        if self._metadata is None:
            raise EoSStateError("metadata have not been loaded")
        return self._metadata.auxiliary_names

    @property
    def charge_constraints(self) -> ChargeConstraintSpec:
        if self._metadata is None:
            raise EoSStateError("metadata have not been loaded")
        return self._metadata.charge_constraints

    @property
    def first_order_transition(self) -> Optional[FirstOrderTransitionSpec]:
        """Return typed computational transition metadata, when declared."""

        if self._metadata is None:
            raise EoSStateError("metadata have not been loaded")
        return self._metadata.first_order_transition

    @property
    def coexistence_table(self) -> Optional[CoexistenceTable]:
        """Return validated canonical sidecar data for a first-order table."""

        self.require_validated()
        return self._coexistence_table

    def snapshot_fingerprint(self) -> str:
        """Return the versioned fingerprint of this validated table snapshot.

        The identity is derived from normalized metadata semantics plus every
        canonical axis and selected field in canonical grid order. Absolute
        filesystem paths and row order are excluded. The algorithm is stable
        for :data:`TABLE_SNAPSHOT_FINGERPRINT_VERSION` and is suitable for
        artifact integrity and checkpoint compatibility checks.
        """

        self.require_validated()
        if self._snapshot_fingerprint is not None:
            return self._snapshot_fingerprint
        assert self._metadata is not None
        assert self._grid is not None
        assert self._fields is not None
        descriptor = {
            "fingerprint_schema": "forward_table_snapshot_v%d"
            % TABLE_SNAPSHOT_FINGERPRINT_VERSION,
            "metadata": _snapshot_plain(self._metadata),
            "canonical_grid": {
                "coordinate_names": list(self._grid.coordinate_names),
                "shape": list(self._grid.shape),
                "uniform_by_axis": list(self._grid.uniform_by_axis),
            },
            "coexistence": (
                None
                if self._coexistence_table is None
                else {
                    "representation": self._coexistence_table.representation,
                    "names": list(self._coexistence_table.names),
                    "units": list(self._coexistence_table.units),
                    "fingerprint": self._coexistence_table.fingerprint,
                    "critical_endpoint": self._coexistence_table.critical_endpoint,
                }
            ),
        }
        encoded = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(
            (
                "EoSInverter:ForwardTableSnapshot:v%d\0"
                % TABLE_SNAPSHOT_FINGERPRINT_VERSION
            ).encode("ascii")
        )
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        for name, axis in zip(self.coordinate_names, self.axes):
            label = ("axis:" + name).encode("utf-8")
            digest.update(len(label).to_bytes(8, "big"))
            digest.update(label)
            _update_digest_array(digest, axis)
        for name in sorted(self._fields):
            label = ("field:" + name).encode("utf-8")
            digest.update(len(label).to_bytes(8, "big"))
            digest.update(label)
            _update_digest_array(digest, self._fields[name])
        if self._coexistence_table is not None:
            label = b"coexistence:canonical_values"
            digest.update(len(label).to_bytes(8, "big"))
            digest.update(label)
            _update_digest_array(digest, self._coexistence_table.values)
        fingerprint = digest.hexdigest()
        self._snapshot_fingerprint = fingerprint
        return fingerprint

    def field(self, name: str, shaped: bool = True) -> np.ndarray:
        """Return a validated field by semantic name."""

        selected = self.grid_fields if shaped else self.fields
        try:
            return selected[name]
        except KeyError as exc:
            raise KeyError("unknown EoS field: %s" % name) from exc

    def find_boundaries(
        self, *, space: str = "target", require_non_degenerate: bool = True
    ) -> BoundaryEnvelope:
        """Find the tightest axis-aligned box containing all target points.

        ``space='target'`` computes component-wise extrema of the canonical
        dynamical variables ``(e, nB, nQ, nS)`` selected by metadata.
        ``space='tilde'`` first applies the constraint-aware ideal-QGP tilde
        transform and then computes extrema. Calling this method on a fresh
        object performs strict validation, so raw column selection, unit
        reconstruction, and density constraints always come from metadata.

        The result is the smallest *axis-aligned* rectangle/hyperrectangle.  It
        encloses the full output point cloud, but its interior may contain
        target combinations that the forward EoS never reaches.
        """

        if not isinstance(require_non_degenerate, bool):
            raise TypeError("require_non_degenerate must be a boolean")
        if space not in {"target", "tilde"}:
            raise ValueError("space must be 'target' or 'tilde'")
        if not self.is_validated:
            self.validate(raise_on_error=True)
        assert self._metadata is not None
        assert self._fields is not None

        targets = np.column_stack(
            [self._fields[name] for name in self._metadata.target_names]
        )
        if space == "target":
            values = targets
            names = self._metadata.target_names
            units = tuple(
                self._metadata.fields[name].canonical_unit for name in names
            )
            transform = None
            dependent_names: Tuple[str, ...] = ()
            dependent_lower = np.empty(0, dtype=float)
            dependent_upper = np.empty(0, dtype=float)
            dependent_relations: Mapping[str, Mapping[str, float]] = {}
        elif space == "tilde":
            try:
                transformed = transform_targets_to_tilde(
                    targets,
                    self._metadata.dimension,
                    self._metadata.charge_constraints,
                )
            except (
                TransformConstraintError,
                TransformDomainError,
                ValueError,
            ) as exc:
                raise BoundaryComputationError(str(exc)) from exc
            values = transformed.independent_values
            names = transformed.independent_names
            units = ("GeV",) * self._metadata.dimension
            transform = transformed.transform_id
            dependent_names = transformed.dependent_names
            if transformed.dependent_values.shape[1]:
                dependent_lower = np.min(transformed.dependent_values, axis=0)
                dependent_upper = np.max(transformed.dependent_values, axis=0)
            else:
                dependent_lower = np.empty(0, dtype=float)
                dependent_upper = np.empty(0, dtype=float)
            dependent_relations = transformed.dependent_relations

        lower = np.min(values, axis=0)
        upper = np.max(values, axis=0)
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise BoundaryComputationError(
                "boundary computation produced a non-finite value"
            )
        degenerate_names = tuple(
            name for name, low, high in zip(names, lower, upper) if low == high
        )
        if require_non_degenerate and degenerate_names:
            raise BoundaryComputationError(
                "Cannot form an inversion grid because target envelope axis/axes "
                "have zero span: %s" % ", ".join(degenerate_names)
            )
        return BoundaryEnvelope(
            coordinate_system=space,
            names=names,
            units=units,
            lower=lower,
            upper=upper,
            sample_count=int(values.shape[0]),
            transform_id=transform,
            degenerate_names=degenerate_names,
            dependent_names=dependent_names,
            dependent_lower=dependent_lower,
            dependent_upper=dependent_upper,
            dependent_relations=dependent_relations,
        )

    def require_validated(self) -> None:
        """Guard used by all future interpolation and inversion operations."""

        if not self._is_validated:
            if self._report is not None and not self._report.is_valid:
                raise EoSValidationError(self._report)
            raise EoSNotValidatedError(
                "EoS has not passed validation; call validate() first"
            )

    def _invalidate(self, keep_metadata: bool = True, keep_raw: bool = True) -> None:
        self._is_validated = False
        self._snapshot_fingerprint = None
        self._report = None
        self._input_fields = None
        self._fields = None
        self._grid_fields = None
        self._grid = None
        self._coexistence_table = None
        if not keep_raw:
            self._raw_data = None
        if not keep_metadata:
            self._metadata = None
            self._metadata_issues = ()

    def load_metadata(self) -> TableSchema:
        """Read and type-check the JSON metadata, without loading table data."""

        self._invalidate(keep_metadata=False, keep_raw=False)
        metadata, issues = self._read_and_parse_metadata()
        self._metadata_issues = tuple(issues)
        if metadata is None or any(issue.severity == "error" for issue in issues):
            details = "; ".join(issue.message for issue in issues)
            raise MetadataError(details or "invalid EoS metadata")
        self._metadata = metadata
        return metadata

    def read_data(self) -> np.ndarray:
        """Load the numerical table and invalidate any previous validation."""

        if self._metadata is None:
            self.load_metadata()
        self._invalidate(keep_metadata=True, keep_raw=False)
        assert self._metadata is not None
        path = self.data_path
        assert path is not None
        try:
            exists = path.is_file()
        except (OSError, ValueError) as exc:
            raise DataLoadError("Invalid EoS data path %s: %s" % (path, exc)) from exc
        if not exists:
            raise DataLoadError("EoS data file does not exist: %s" % path)

        delimiter = self._metadata.format.delimiter
        converters = None
        if self._metadata.format.missing_values:
            missing_values = set(self._metadata.format.missing_values)

            def convert_token(token: str) -> float:
                stripped = token.strip()
                if stripped in missing_values:
                    return float("nan")
                return float(stripped)

            assert self._metadata.format.expected_columns is not None
            converters = {
                index: convert_token
                for index in range(self._metadata.format.expected_columns)
            }
        try:
            data = np.loadtxt(
                str(path),
                dtype=float,
                comments=self._metadata.format.comment_prefix,
                delimiter=delimiter,
                skiprows=self._metadata.format.skip_rows,
                ndmin=2,
                encoding=self._metadata.format.encoding,
                converters=converters,
            )
        except (LookupError, OSError, OverflowError, TypeError, ValueError) as exc:
            raise DataLoadError("Could not read EoS data file %s: %s" % (path, exc)) from exc

        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
            raise DataLoadError("EoS data file contains no two-dimensional numeric table")
        self._raw_data = _immutable_array(data)
        return self._raw_data

    # ``read`` is a concise alias for interactive use.
    read = read_data

    def interpret_data(self) -> Mapping[str, np.ndarray]:
        """Select fields and apply all explicit unit/scaling transforms."""

        if self._metadata is None:
            self.load_metadata()
        if self._raw_data is None:
            self.read_data()
        self._invalidate(keep_metadata=True, keep_raw=True)
        assert self._metadata is not None
        assert self._raw_data is not None

        if not self._metadata.fields:
            raise MetadataError("metadata define no fields")
        maximum_column = max(spec.column for spec in self._metadata.fields.values())
        if maximum_column >= self._raw_data.shape[1]:
            raise DataLoadError(
                "Field column %d is outside a table with %d columns"
                % (maximum_column, self._raw_data.shape[1])
            )

        temperature_spec = self._metadata.fields["T"]
        temperature = self._raw_data[:, temperature_spec.column].astype(float, copy=True)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            temperature *= temperature_spec.transform.scale_factor
            temperature_hbarc_factor = float(
                np.power(HBARC_GEV_FM, temperature_spec.transform.hbarc_power)
            )
            temperature *= temperature_hbarc_factor
        if not math.isfinite(temperature_hbarc_factor) or temperature_hbarc_factor == 0:
            raise DataLoadError("The T hbarc_power produces an unusable conversion factor")

        interpreted: Dict[str, np.ndarray] = {}
        for name, spec in self._metadata.fields.items():
            values = self._raw_data[:, spec.column].astype(float, copy=True)
            transform = spec.transform
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                values *= transform.scale_factor
                if transform.temperature_power:
                    values *= np.power(temperature, transform.temperature_power)
                if transform.hbarc_power:
                    hbarc_factor = float(np.power(HBARC_GEV_FM, transform.hbarc_power))
                    if not math.isfinite(hbarc_factor) or hbarc_factor == 0:
                        raise DataLoadError(
                            "Field '%s' hbarc_power produces an unusable conversion factor"
                            % name
                        )
                    values *= hbarc_factor
            interpreted[name] = _immutable_array(values)

        self._input_fields = MappingProxyType(interpreted)
        return self._input_fields

    # ``interpret`` mirrors the terminology used in the design discussion.
    interpret = interpret_data

    def validate(
        self, *, reload: bool = False, raise_on_error: bool = False
    ) -> ValidationReport:
        """Validate metadata, transforms, finite values, and grid completeness.

        Expected input problems are aggregated in the returned report.  Set
        ``raise_on_error`` for strict loading.  ``reload=True`` forces a new
        metadata/table snapshot; otherwise already loaded raw data may be
        reused, but transformations and all checks are rerun.
        """

        self._is_validated = False
        self._report = None
        self._fields = None
        self._grid_fields = None
        self._grid = None
        self._coexistence_table = None

        issues: List[ValidationIssue] = []
        statistics: Dict[str, Any] = {}

        if reload:
            self._invalidate(keep_metadata=False, keep_raw=False)

        if self._metadata is None:
            metadata, metadata_issues = self._read_and_parse_metadata()
            self._metadata_issues = tuple(metadata_issues)
            issues.extend(metadata_issues)
            if metadata is not None and not any(
                issue.severity == "error" for issue in metadata_issues
            ):
                self._metadata = metadata
        else:
            issues.extend(self._metadata_issues)

        if self._metadata is None or any(issue.severity == "error" for issue in issues):
            return self._finish_validation(issues, statistics, raise_on_error)

        self._validate_scientific_files(issues)

        if self._raw_data is None:
            try:
                self.read_data()
            except DataLoadError as exc:
                _issue(
                    issues,
                    "error",
                    "data.load_failed",
                    str(exc),
                    str(self.data_path) if self.data_path else None,
                )
                return self._finish_validation(issues, statistics, raise_on_error)

        assert self._raw_data is not None
        statistics["data"] = {
            "rows": int(self._raw_data.shape[0]),
            "columns": int(self._raw_data.shape[1]),
            "path": str(self.data_path),
        }
        self._validate_table_shape(issues)

        if any(issue.code == "data.column_out_of_range" for issue in issues):
            return self._finish_validation(issues, statistics, raise_on_error)

        try:
            interpreted = self.interpret_data()
        except (DataLoadError, MetadataError, KeyError) as exc:
            _issue(
                issues,
                "error",
                "data.interpretation_failed",
                str(exc),
                str(self.data_path) if self.data_path else None,
            )
            return self._finish_validation(issues, statistics, raise_on_error)

        field_statistics = self._validate_field_values(interpreted, issues)
        statistics["fields"] = field_statistics
        statistics["charge_constraints"] = self._validate_charge_constraint_values(
            interpreted, issues
        )

        coordinate_error = any(
            issue.severity == "error" and issue.code.startswith("field.nonfinite.coordinate")
            for issue in issues
        )
        if not coordinate_error:
            grid, ordered_fields, shaped_fields, grid_statistics = self._analyze_grid(
                interpreted, issues
            )
            statistics["grid"] = grid_statistics
            if grid is not None:
                self._grid = grid
                self._fields = MappingProxyType(ordered_fields)
                self._grid_fields = MappingProxyType(shaped_fields)
                geometry_statistics = scan_table_geometry(
                    grid.axes,
                    shaped_fields,
                    self._metadata.target_names,
                )
                statistics["geometry"] = geometry_statistics
                self._record_geometry_issues(geometry_statistics, issues)
                transition_statistics = self._load_and_validate_coexistence_data(
                    issues
                )
                if transition_statistics is not None:
                    statistics["first_order_transition"] = transition_statistics
        self._input_fields = None

        return self._finish_validation(issues, statistics, raise_on_error)

    def _finish_validation(
        self,
        issues: Sequence[ValidationIssue],
        statistics: Mapping[str, Any],
        raise_on_error: bool,
    ) -> ValidationReport:
        report = ValidationReport(tuple(issues), statistics)
        self._report = report
        self._is_validated = bool(
            report.is_valid
            and self._metadata is not None
            and self._raw_data is not None
            and self._grid is not None
            and self._fields is not None
            and self._grid_fields is not None
            and (
                self._metadata.first_order_transition is None
                or self._coexistence_table is not None
            )
        )
        if raise_on_error and not self._is_validated:
            if report.is_valid:
                incomplete_issues = list(issues)
                _issue(
                    incomplete_issues,
                    "error",
                    "validation.incomplete",
                    "Validation did not produce a complete canonical grid.",
                )
                report = ValidationReport(tuple(incomplete_issues), statistics)
                self._report = report
            raise EoSValidationError(report)
        return report

    def _read_and_parse_metadata(
        self,
    ) -> Tuple[Optional[TableSchema], List[ValidationIssue]]:
        issues: List[ValidationIssue] = []
        try:
            text = self.metadata_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            _issue(
                issues,
                "error",
                "metadata.not_found",
                "Metadata file not found:\n  %s" % self.metadata_path,
                str(self.metadata_path),
            )
            return None, issues
        except (OSError, UnicodeError) as exc:
            _issue(
                issues,
                "error",
                "metadata.read_failed",
                "Could not read metadata: %s" % exc,
                str(self.metadata_path),
            )
            return None, issues

        try:
            raw = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as exc:
            _issue(
                issues,
                "error",
                "metadata.invalid_json",
                "Metadata JSON syntax error:\n"
                "  file: %s\n"
                "  line: %d\n"
                "  column: %d\n\n"
                "%s." % (
                    self.metadata_path,
                    exc.lineno,
                    exc.colno,
                    exc.msg,
                ),
                str(self.metadata_path),
            )
            return None, issues
        except (ValueError, _DuplicateJSONKey) as exc:
            _issue(
                issues,
                "error",
                "metadata.invalid_json",
                str(exc),
                str(self.metadata_path),
            )
            return None, issues

        if not isinstance(raw, dict):
            _issue(
                issues,
                "error",
                "metadata.root_type",
                "The metadata JSON root must be an object.",
                str(self.metadata_path),
            )
            return None, issues
        metadata = self._parse_metadata(raw, issues)
        return metadata, issues

    def _parse_metadata(
        self, raw: Mapping[str, Any], issues: List[ValidationIssue]
    ) -> Optional[TableSchema]:
        allowed_top = {
            "schema_version",
            "name",
            "table_revision",
            "description",
            "dimension",
            "data_file",
            "format",
            "fields",
            "grid",
            "charge_constraints",
            "scientific",
        }
        required_top = {
            "name",
            "dimension",
            "data_file",
            "fields",
        }
        _check_unknown_keys(raw, allowed_top, "", issues)
        for key in sorted(required_top - set(raw)):
            _issue(
                issues,
                "error",
                "metadata.missing_key",
                "Required metadata key '%s' is missing." % key,
                key,
            )

        schema_version = raw.get("schema_version", METADATA_SCHEMA_VERSION)
        if not _is_int(schema_version) or schema_version != METADATA_SCHEMA_VERSION:
            _issue(
                issues,
                "error",
                "metadata.schema_version",
                "schema_version must be integer %d." % METADATA_SCHEMA_VERSION,
                "schema_version",
            )

        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            _issue(issues, "error", "metadata.name", "name must be a non-empty string.", "name")

        revision = raw.get("table_revision")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            _issue(
                issues,
                "error",
                "metadata.table_revision",
                "table_revision must be a non-empty string or null.",
                "table_revision",
            )

        description = raw.get("description")
        if description is not None and not isinstance(description, str):
            _issue(
                issues,
                "error",
                "metadata.description",
                "description must be a string or null.",
                "description",
            )

        dimension = raw.get("dimension")
        if not _is_int(dimension) or dimension not in range(1, 5):
            _issue(
                issues,
                "error",
                "metadata.dimension",
                "dimension must be one of 1, 2, 3, or 4.",
                "dimension",
            )

        data_file = raw.get("data_file")
        if not isinstance(data_file, str) or not data_file.strip():
            _issue(
                issues,
                "error",
                "metadata.data_file",
                "data_file must be a non-empty relative path string.",
                "data_file",
            )
        elif Path(data_file).is_absolute():
            _issue(
                issues,
                "error",
                "metadata.data_file_absolute",
                "data_file must be relative to the metadata file.",
                "data_file",
            )
        elif "\x00" in data_file:
            _issue(
                issues,
                "error",
                "metadata.data_file_nul",
                "data_file cannot contain a NUL character.",
                "data_file",
            )

        format_spec = self._parse_format(raw.get("format", {}), issues)
        fields = self._parse_fields(raw.get("fields"), dimension, issues)
        grid = self._parse_grid(raw.get("grid", {}), dimension, issues)
        charge_constraints = self._parse_charge_constraints(
            raw.get("charge_constraints"), dimension, issues
        )

        first_order_transition: Optional[FirstOrderTransitionSpec] = None
        scientific = raw.get("scientific", {})
        if not isinstance(scientific, dict):
            _issue(
                issues,
                "error",
                "metadata.scientific_type",
                "scientific must be an object.",
                "scientific",
            )
            scientific = {}
        else:
            first_order_transition = self._validate_scientific_metadata(
                scientific, fields or {}, dimension, issues
            )

        if any(issue.severity == "error" for issue in issues):
            return None
        assert isinstance(schema_version, int)
        assert isinstance(name, str)
        assert isinstance(dimension, int)
        assert isinstance(data_file, str)
        assert format_spec is not None
        assert fields is not None
        assert grid is not None
        assert charge_constraints is not None
        return TableSchema(
            schema_version=schema_version,
            name=name,
            table_revision=revision,
            description=description,
            dimension=dimension,
            data_file=data_file,
            format=format_spec,
            fields=MappingProxyType(fields),
            grid=grid,
            charge_constraints=charge_constraints,
            scientific=_freeze(scientific),
            first_order_transition=first_order_transition,
        )

    def _parse_charge_constraints(
        self,
        raw: Any,
        dimension: Any,
        issues: List[ValidationIssue],
    ) -> Optional[ChargeConstraintSpec]:
        """Parse computational density relations used by the tilde transform."""

        if not _is_int(dimension) or dimension not in range(1, 5):
            return None
        if raw is None:
            return ChargeConstraintSpec.default(dimension)
        if not isinstance(raw, dict):
            _issue(
                issues,
                "error",
                "metadata.charge_constraints_type",
                "charge_constraints must be an object when provided.",
                "charge_constraints",
            )
            return None

        _check_unknown_keys(
            raw,
            {"model", "dependent_densities", "validation"},
            "charge_constraints",
            issues,
        )
        for key in ("model", "dependent_densities"):
            if key not in raw:
                _issue(
                    issues,
                    "error",
                    "metadata.charge_constraints_missing_key",
                    "charge_constraints.%s is required." % key,
                    "charge_constraints.%s" % key,
                )

        model = raw.get("model")
        if model != "linear_homogeneous":
            _issue(
                issues,
                "error",
                "metadata.charge_constraints_model",
                "charge_constraints.model must be 'linear_homogeneous'.",
                "charge_constraints.model",
            )

        independent_names = set(_TARGET_NAMES[1:dimension])
        expected_dependent = set(_TARGET_NAMES[dimension:])
        relations_raw = raw.get("dependent_densities")
        parsed_relations: Dict[str, Dict[str, float]] = {}
        if not isinstance(relations_raw, dict):
            _issue(
                issues,
                "error",
                "metadata.dependent_densities_type",
                "charge_constraints.dependent_densities must be an object.",
                "charge_constraints.dependent_densities",
            )
        else:
            actual_dependent = set(relations_raw)
            for missing in sorted(expected_dependent - actual_dependent):
                _issue(
                    issues,
                    "error",
                    "metadata.dependent_density_missing",
                    "A relation for omitted density '%s' is required." % missing,
                    "charge_constraints.dependent_densities.%s" % missing,
                )
            for extra in sorted(actual_dependent - expected_dependent):
                _issue(
                    issues,
                    "error",
                    "metadata.dependent_density_unexpected",
                    "Density '%s' is not dependent in dimension %d."
                    % (extra, dimension),
                    "charge_constraints.dependent_densities.%s" % extra,
                )
            for dependent_name in sorted(expected_dependent & actual_dependent):
                path = "charge_constraints.dependent_densities.%s" % dependent_name
                relation = relations_raw[dependent_name]
                if not isinstance(relation, dict):
                    _issue(
                        issues,
                        "error",
                        "metadata.density_relation_type",
                        "A density relation must be an object of coefficients.",
                        path,
                    )
                    continue
                parsed_relation: Dict[str, float] = {}
                for source_name, coefficient in relation.items():
                    coefficient_path = "%s.%s" % (path, source_name)
                    if source_name not in independent_names:
                        _issue(
                            issues,
                            "error",
                            "metadata.density_relation_source",
                            "'%s' is not an independent density for dimension %d."
                            % (source_name, dimension),
                            coefficient_path,
                        )
                        continue
                    numeric = _finite_float(coefficient)
                    if numeric is None:
                        _issue(
                            issues,
                            "error",
                            "metadata.density_relation_coefficient",
                            "Density-relation coefficients must be finite numbers.",
                            coefficient_path,
                        )
                        continue
                    parsed_relation[source_name] = numeric
                parsed_relations[dependent_name] = parsed_relation

        validation = raw.get("validation", {})
        absolute_tolerance = 1e-12
        relative_tolerance = 1e-8
        if not isinstance(validation, dict):
            _issue(
                issues,
                "error",
                "metadata.charge_constraint_validation_type",
                "charge_constraints.validation must be an object.",
                "charge_constraints.validation",
            )
        else:
            _check_unknown_keys(
                validation,
                {"absolute_tolerance", "relative_tolerance"},
                "charge_constraints.validation",
                issues,
            )
            for key, default in (
                ("absolute_tolerance", absolute_tolerance),
                ("relative_tolerance", relative_tolerance),
            ):
                value = validation.get(key, default)
                numeric = _finite_float(value)
                if numeric is None or numeric < 0.0:
                    _issue(
                        issues,
                        "error",
                        "metadata.charge_constraint_tolerance",
                        "%s must be a finite non-negative number." % key,
                        "charge_constraints.validation.%s" % key,
                    )
                elif key == "absolute_tolerance":
                    absolute_tolerance = numeric
                else:
                    relative_tolerance = numeric

        prefix = "charge_constraints"
        if any(
            issue.severity == "error"
            and issue.path
            and issue.path.startswith(prefix)
            for issue in issues
        ):
            return None
        try:
            return ChargeConstraintSpec(
                dimension=dimension,
                dependent_densities=parsed_relations,
                source="metadata",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                model=model,
            )
        except TransformConstraintError as exc:
            _issue(
                issues,
                "error",
                "metadata.charge_constraints_invalid",
                str(exc),
                "charge_constraints",
            )
            return None

    def _parse_format(
        self, raw: Any, issues: List[ValidationIssue]
    ) -> Optional[FormatSpec]:
        if not isinstance(raw, dict):
            _issue(
                issues,
                "error",
                "metadata.format_type",
                "format must be an object.",
                "format",
            )
            return None
        allowed = {
            "delimiter",
            "comment_prefix",
            "skip_rows",
            "encoding",
            "expected_columns",
            "missing_values",
        }
        _check_unknown_keys(raw, allowed, "format", issues)
        delimiter_value = raw.get("delimiter", "whitespace")
        delimiter: Optional[str]
        if delimiter_value == "whitespace":
            delimiter = None
        elif isinstance(delimiter_value, str) and len(delimiter_value) == 1:
            delimiter = delimiter_value
        else:
            delimiter = None
            _issue(
                issues,
                "error",
                "metadata.format_delimiter",
                "delimiter must be 'whitespace' or a single character.",
                "format.delimiter",
            )

        comment = raw.get("comment_prefix", "#")
        if comment is not None and (not isinstance(comment, str) or not comment):
            _issue(
                issues,
                "error",
                "metadata.format_comment",
                "comment_prefix must be a non-empty string or null.",
                "format.comment_prefix",
            )

        skip_rows = raw.get("skip_rows", 0)
        if not _is_int(skip_rows) or skip_rows < 0:
            _issue(
                issues,
                "error",
                "metadata.format_skip_rows",
                "skip_rows must be a non-negative integer.",
                "format.skip_rows",
            )

        encoding = raw.get("encoding", "utf-8")
        if not isinstance(encoding, str) or not encoding:
            _issue(
                issues,
                "error",
                "metadata.format_encoding",
                "encoding must be a non-empty string.",
                "format.encoding",
            )
        else:
            try:
                codecs.lookup(encoding)
            except LookupError:
                _issue(
                    issues,
                    "error",
                    "metadata.format_encoding",
                    "encoding is not recognized by Python.",
                    "format.encoding",
                    encoding=encoding,
                )

        expected_columns = raw.get("expected_columns")
        if expected_columns is not None and (
            not _is_int(expected_columns)
            or expected_columns < 1
            or expected_columns > MAX_TABLE_COLUMNS
        ):
            _issue(
                issues,
                "error",
                "metadata.format_expected_columns",
                "expected_columns must be between 1 and %d, or null."
                % MAX_TABLE_COLUMNS,
                "format.expected_columns",
            )

        missing_values_raw = raw.get("missing_values", [])
        missing_values: Tuple[str, ...] = ()
        if not isinstance(missing_values_raw, list) or not all(
            isinstance(value, str) for value in missing_values_raw
        ):
            _issue(
                issues,
                "error",
                "metadata.format_missing_values",
                "missing_values must be a list of strings.",
                "format.missing_values",
            )
        elif len(set(missing_values_raw)) != len(missing_values_raw):
            _issue(
                issues,
                "error",
                "metadata.format_missing_values",
                "missing_values cannot contain duplicate tokens.",
                "format.missing_values",
            )
        elif missing_values_raw and not (
            _is_int(expected_columns) and expected_columns > 0
        ):
            _issue(
                issues,
                "error",
                "metadata.format_missing_values_columns",
                "expected_columns is required when custom missing_values are used.",
                "format.expected_columns",
            )
        else:
            missing_values = tuple(missing_values_raw)

        if any(
            issue.severity == "error" and issue.path and issue.path.startswith("format")
            for issue in issues
        ):
            return None
        assert _is_int(skip_rows)
        assert isinstance(encoding, str)
        return FormatSpec(
            delimiter,
            comment,
            skip_rows,
            encoding,
            expected_columns,
            missing_values,
        )

    def _parse_fields(
        self, raw: Any, dimension: Any, issues: List[ValidationIssue]
    ) -> Optional[Dict[str, FieldSpec]]:
        if not isinstance(raw, dict) or not raw:
            _issue(
                issues,
                "error",
                "metadata.fields_type",
                "fields must be a non-empty object keyed by semantic field name.",
                "fields",
            )
            return None

        parsed: Dict[str, FieldSpec] = {}
        columns: Dict[int, str] = {}
        for name, value in raw.items():
            path = "fields.%s" % name
            if not isinstance(name, str) or not name:
                _issue(
                    issues,
                    "error",
                    "metadata.field_name",
                    "Field names must be strings.",
                    path,
                )
                continue
            if not isinstance(value, dict):
                _issue(
                    issues,
                    "error",
                    "metadata.field_type",
                    "Each field descriptor must be an object.",
                    path,
                )
                continue
            allowed = {
                "role",
                "column",
                "source_unit",
                "canonical_unit",
                "transform",
                "nonfinite_policy",
                "output",
                "interpolation",
                "source_label",
                "stored_as",
                "description",
            }
            _check_unknown_keys(value, allowed, path, issues)
            required = {"role", "column", "source_unit", "canonical_unit"}
            for key in sorted(required - set(value)):
                _issue(
                    issues,
                    "error",
                    "metadata.field_missing_key",
                    "Required field key '%s' is missing." % key,
                    "%s.%s" % (path, key),
                )

            role = value.get("role")
            if role not in {"coordinate", "target", "auxiliary"}:
                _issue(
                    issues,
                    "error",
                    "metadata.field_role",
                    "role must be coordinate, target, or auxiliary.",
                    path + ".role",
                )

            column = value.get("column")
            if not _is_int(column) or column < 0:
                _issue(
                    issues,
                    "error",
                    "metadata.field_column",
                    "column must be a non-negative integer.",
                    path + ".column",
                )
            elif column in columns:
                _issue(
                    issues,
                    "error",
                    "metadata.duplicate_column",
                    "Column %d is already assigned to field '%s'." % (column, columns[column]),
                    path + ".column",
                    column=column,
                    other_field=columns[column],
                )
            else:
                columns[column] = name

            source_unit = value.get("source_unit")
            canonical_unit = value.get("canonical_unit")
            if not isinstance(source_unit, str) or not source_unit:
                _issue(
                    issues,
                    "error",
                    "metadata.field_source_unit",
                    "source_unit must be a non-empty string.",
                    path + ".source_unit",
                )
            if not isinstance(canonical_unit, str) or not canonical_unit:
                _issue(
                    issues,
                    "error",
                    "metadata.field_canonical_unit",
                    "canonical_unit must be a non-empty string.",
                    path + ".canonical_unit",
                )

            transform = self._parse_transform(value.get("transform", {}), path, issues)
            policy = value.get("nonfinite_policy", "forbid")
            if policy not in {"forbid", "allow_nan"}:
                _issue(
                    issues,
                    "error",
                    "metadata.field_nonfinite_policy",
                    "nonfinite_policy must be 'forbid' or 'allow_nan'.",
                    path + ".nonfinite_policy",
                )
            if role in {"coordinate", "target"} and policy != "forbid":
                _issue(
                    issues,
                    "error",
                    "metadata.required_field_nonfinite_policy",
                    "Coordinate and target fields must forbid non-finite values.",
                    path + ".nonfinite_policy",
                )

            output = value.get("output", True)
            if not isinstance(output, bool):
                _issue(
                    issues,
                    "error",
                    "metadata.field_output",
                    "output must be a boolean.",
                    path + ".output",
                )

            interpolation = value.get("interpolation", "linear")
            if interpolation is not None and interpolation not in {"linear", "nearest"}:
                _issue(
                    issues,
                    "error",
                    "metadata.field_interpolation",
                    "interpolation must be linear, nearest, or null.",
                    path + ".interpolation",
                )
            source_label = value.get("source_label")
            if source_label is not None and not isinstance(source_label, str):
                _issue(
                    issues,
                    "error",
                    "metadata.field_source_label",
                    "source_label must be a string or null.",
                    path + ".source_label",
                )
            stored_as = value.get("stored_as")
            if stored_as is not None and (
                not isinstance(stored_as, str) or not stored_as
            ):
                _issue(
                    issues,
                    "error",
                    "metadata.field_stored_as",
                    "stored_as must be a non-empty string or null.",
                    path + ".stored_as",
                )
            field_description = value.get("description")
            if field_description is not None and not isinstance(field_description, str):
                _issue(
                    issues,
                    "error",
                    "metadata.field_description",
                    "description must be a string or null.",
                    path + ".description",
                )

            if (
                role in {"coordinate", "target", "auxiliary"}
                and _is_int(column)
                and column >= 0
                and isinstance(source_unit, str)
                and source_unit
                and isinstance(canonical_unit, str)
                and canonical_unit
                and transform is not None
                and policy in {"forbid", "allow_nan"}
                and isinstance(output, bool)
            ):
                parsed[name] = FieldSpec(
                    name=name,
                    role=role,
                    column=column,
                    source_unit=source_unit,
                    canonical_unit=canonical_unit,
                    transform=transform,
                    nonfinite_policy=policy,
                    output=output,
                    interpolation=interpolation,
                    source_label=source_label,
                    stored_as=stored_as,
                    description=field_description,
                )

        if _is_int(dimension) and dimension in range(1, 5):
            coordinates = _COORDINATE_NAMES[:dimension]
            targets = _TARGET_NAMES[:dimension]
            for name in coordinates:
                spec = parsed.get(name)
                if spec is None:
                    _issue(
                        issues,
                        "error",
                        "metadata.canonical_field_missing",
                        "Canonical coordinate field '%s' is required." % name,
                        "fields.%s" % name,
                    )
                elif spec.role != "coordinate":
                    _issue(
                        issues,
                        "error",
                        "metadata.canonical_field_role",
                        "Canonical field '%s' must have role coordinate." % name,
                        "fields.%s.role" % name,
                    )
            for name in targets:
                spec = parsed.get(name)
                if spec is None:
                    _issue(
                        issues,
                        "error",
                        "metadata.canonical_field_missing",
                        "Canonical target field '%s' is required." % name,
                        "fields.%s" % name,
                    )
                elif spec.role != "target":
                    _issue(
                        issues,
                        "error",
                        "metadata.canonical_field_role",
                        "Canonical field '%s' must have role target." % name,
                        "fields.%s.role" % name,
                    )
            canonical = set(coordinates + targets)
            for name, spec in parsed.items():
                if name not in canonical and spec.role != "auxiliary":
                    _issue(
                        issues,
                        "error",
                        "metadata.noncanonical_role",
                        "Non-canonical field '%s' must be auxiliary for dimension %d."
                        % (name, dimension),
                        "fields.%s.role" % name,
                    )

            canonical_units = {
                "T": "GeV",
                "muB": "GeV",
                "muQ": "GeV",
                "muS": "GeV",
                "e": "GeV^4",
                "nB": "GeV^3",
                "nQ": "GeV^3",
                "nS": "GeV^3",
            }
            for name, spec in parsed.items():
                if (
                    name in canonical_units
                    and spec.canonical_unit != canonical_units[name]
                ):
                    _issue(
                        issues,
                        "error",
                        "metadata.canonical_unit",
                        "Reserved physical field '%s' must use canonical unit '%s'."
                        % (name, canonical_units[name]),
                        "fields.%s.canonical_unit" % name,
                        actual=spec.canonical_unit,
                        expected=canonical_units[name],
                    )

        temperature = parsed.get("T")
        if temperature is not None and temperature.transform.temperature_power != 0:
            _issue(
                issues,
                "error",
                "metadata.temperature_self_scaling",
                "The T field cannot have a nonzero temperature_power.",
                "fields.T.transform.temperature_power",
            )

        if any(
            issue.severity == "error" and issue.path and issue.path.startswith("fields")
            for issue in issues
        ):
            return None
        return parsed

    def _parse_transform(
        self, raw: Any, field_path: str, issues: List[ValidationIssue]
    ) -> Optional[TransformSpec]:
        path = field_path + ".transform"
        if not isinstance(raw, dict):
            _issue(
                issues,
                "error",
                "metadata.transform_type",
                "transform must be an object.",
                path,
            )
            return None
        allowed = {"scale_factor", "temperature_power", "hbarc_power"}
        _check_unknown_keys(raw, allowed, path, issues)
        scale = raw.get("scale_factor", 1.0)
        temperature_power = raw.get("temperature_power", 0)
        hbarc_power = raw.get("hbarc_power", 0)
        numeric_scale = _finite_float(scale)
        if numeric_scale is None or numeric_scale <= 0:
            _issue(
                issues,
                "error",
                "metadata.transform_scale",
                "scale_factor must be a finite positive number.",
                path + ".scale_factor",
            )
        if (
            not _is_int(temperature_power)
            or abs(temperature_power) > MAX_ABSOLUTE_TRANSFORM_POWER
        ):
            _issue(
                issues,
                "error",
                "metadata.transform_temperature_power",
                "temperature_power must be an integer between -%d and %d."
                % (MAX_ABSOLUTE_TRANSFORM_POWER, MAX_ABSOLUTE_TRANSFORM_POWER),
                path + ".temperature_power",
            )
        if not _is_int(hbarc_power) or abs(hbarc_power) > MAX_ABSOLUTE_TRANSFORM_POWER:
            _issue(
                issues,
                "error",
                "metadata.transform_hbarc_power",
                "hbarc_power must be an integer between -%d and %d."
                % (MAX_ABSOLUTE_TRANSFORM_POWER, MAX_ABSOLUTE_TRANSFORM_POWER),
                path + ".hbarc_power",
            )
        if numeric_scale is None or numeric_scale <= 0:
            return None
        if (
            not _is_int(temperature_power)
            or abs(temperature_power) > MAX_ABSOLUTE_TRANSFORM_POWER
            or not _is_int(hbarc_power)
            or abs(hbarc_power) > MAX_ABSOLUTE_TRANSFORM_POWER
        ):
            return None
        return TransformSpec(numeric_scale, temperature_power, hbarc_power)

    def _parse_grid(
        self, raw: Any, dimension: Any, issues: List[ValidationIssue]
    ) -> Optional[GridSpec]:
        if not isinstance(raw, dict):
            _issue(
                issues,
                "error",
                "metadata.grid_type",
                "grid must be an object.",
                "grid",
            )
            return None
        allowed = {
            "kind",
            "require_complete",
            "require_uniform",
            "reorder",
            "duplicate_policy",
            "absolute_tolerance",
            "relative_tolerance",
            "min_axis_points",
            "expected_shape",
            "expected_bounds",
            "expected_spacing",
        }
        _check_unknown_keys(raw, allowed, "grid", issues)
        kind = raw.get("kind", "rectilinear")
        if kind != "rectilinear":
            _issue(
                issues,
                "error",
                "metadata.grid_kind",
                "Only kind='rectilinear' is supported in schema version 1.",
                "grid.kind",
            )
        require_complete = raw.get("require_complete", True)
        if require_complete is not True:
            _issue(
                issues,
                "error",
                "metadata.grid_complete",
                "Schema version 1 requires a complete rectilinear grid.",
                "grid.require_complete",
            )
        require_uniform = raw.get("require_uniform", False)
        reorder = raw.get("reorder", True)
        for key, value in (("require_uniform", require_uniform), ("reorder", reorder)):
            if not isinstance(value, bool):
                _issue(
                    issues,
                    "error",
                    "metadata.grid_boolean",
                    "%s must be a boolean." % key,
                    "grid.%s" % key,
                )
        duplicate_policy = raw.get("duplicate_policy", "error")
        if duplicate_policy != "error":
            _issue(
                issues,
                "error",
                "metadata.grid_duplicate_policy",
                "Only duplicate_policy='error' is currently supported.",
                "grid.duplicate_policy",
            )
        absolute_tolerance = raw.get("absolute_tolerance", 0.0)
        relative_tolerance = raw.get("relative_tolerance", 0.0)
        for key, value in (
            ("absolute_tolerance", absolute_tolerance),
            ("relative_tolerance", relative_tolerance),
        ):
            numeric_value = _finite_float(value)
            if numeric_value is None or numeric_value < 0:
                _issue(
                    issues,
                    "error",
                    "metadata.grid_tolerance",
                    "%s must be a finite non-negative number." % key,
                    "grid.%s" % key,
                )
        min_axis_points = raw.get("min_axis_points", 2)
        if not _is_int(min_axis_points) or min_axis_points < 2:
            _issue(
                issues,
                "error",
                "metadata.grid_min_axis_points",
                "min_axis_points must be an integer of at least 2.",
                "grid.min_axis_points",
            )

        expected_shape_value = raw.get("expected_shape")
        expected_shape: Optional[Tuple[int, ...]] = None
        if expected_shape_value is not None:
            valid_shape = (
                isinstance(expected_shape_value, list)
                and _is_int(dimension)
                and len(expected_shape_value) == dimension
                and all(_is_int(value) and value >= 1 for value in expected_shape_value)
            )
            if not valid_shape:
                _issue(
                    issues,
                    "error",
                    "metadata.grid_expected_shape",
                    "expected_shape must contain one positive integer per dimension.",
                    "grid.expected_shape",
                )
            else:
                expected_shape = tuple(expected_shape_value)

        expected_names = (
            set(_COORDINATE_NAMES[:dimension])
            if _is_int(dimension) and dimension in range(1, 5)
            else set()
        )
        expected_bounds = self._parse_named_bounds(
            raw.get("expected_bounds", {}), expected_names, issues
        )
        expected_spacing = self._parse_named_spacing(
            raw.get("expected_spacing", {}), expected_names, issues
        )

        if any(
            issue.severity == "error" and issue.path and issue.path.startswith("grid")
            for issue in issues
        ):
            return None
        numeric_absolute_tolerance = _finite_float(absolute_tolerance)
        numeric_relative_tolerance = _finite_float(relative_tolerance)
        assert numeric_absolute_tolerance is not None
        assert numeric_relative_tolerance is not None
        return GridSpec(
            kind=kind,
            require_complete=require_complete,
            require_uniform=require_uniform,
            reorder=reorder,
            duplicate_policy=duplicate_policy,
            absolute_tolerance=numeric_absolute_tolerance,
            relative_tolerance=numeric_relative_tolerance,
            min_axis_points=min_axis_points,
            expected_shape=expected_shape,
            expected_bounds=MappingProxyType(expected_bounds),
            expected_spacing=MappingProxyType(expected_spacing),
        )

    def _parse_named_bounds(
        self,
        raw: Any,
        expected_names: set,
        issues: List[ValidationIssue],
    ) -> Dict[str, Tuple[float, float]]:
        if not isinstance(raw, dict):
            _issue(
                issues,
                "error",
                "metadata.grid_expected_bounds",
                "expected_bounds must be an object.",
                "grid.expected_bounds",
            )
            return {}
        result: Dict[str, Tuple[float, float]] = {}
        for name, bounds in raw.items():
            path = "grid.expected_bounds.%s" % name
            if name not in expected_names:
                _issue(
                    issues,
                    "error",
                    "metadata.grid_axis_name",
                    "'%s' is not a canonical coordinate for this dimension." % name,
                    path,
                )
                continue
            numeric_bounds = (
                tuple(_finite_float(value) for value in bounds)
                if isinstance(bounds, list) and len(bounds) == 2
                else (None, None)
            )
            valid = (
                numeric_bounds[0] is not None
                and numeric_bounds[1] is not None
                and numeric_bounds[0] <= numeric_bounds[1]
            )
            if not valid:
                _issue(
                    issues,
                    "error",
                    "metadata.grid_expected_bounds",
                    "Each expected bound must be [finite minimum, finite maximum].",
                    path,
                )
                continue
            result[name] = (numeric_bounds[0], numeric_bounds[1])
        return result

    def _parse_named_spacing(
        self,
        raw: Any,
        expected_names: set,
        issues: List[ValidationIssue],
    ) -> Dict[str, float]:
        if not isinstance(raw, dict):
            _issue(
                issues,
                "error",
                "metadata.grid_expected_spacing",
                "expected_spacing must be an object.",
                "grid.expected_spacing",
            )
            return {}
        result: Dict[str, float] = {}
        for name, spacing in raw.items():
            path = "grid.expected_spacing.%s" % name
            if name not in expected_names:
                _issue(
                    issues,
                    "error",
                    "metadata.grid_axis_name",
                    "'%s' is not a canonical coordinate for this dimension." % name,
                    path,
                )
            else:
                numeric_spacing = _finite_float(spacing)
                if numeric_spacing is None or numeric_spacing <= 0:
                    _issue(
                        issues,
                        "error",
                        "metadata.grid_expected_spacing",
                        "Expected spacing must be a finite positive number.",
                        path,
                    )
                else:
                    result[name] = numeric_spacing
        return result

    def _validate_scientific_metadata(
        self,
        scientific: Mapping[str, Any],
        fields: Mapping[str, FieldSpec],
        dimension: Any,
        issues: List[ValidationIssue],
    ) -> Optional[FirstOrderTransitionSpec]:
        description = scientific.get("description")
        if description is not None and not isinstance(description, str):
            _issue(
                issues,
                "error",
                "scientific.description_type",
                "scientific.description must be a string or null.",
                "scientific.description",
            )

        publications = scientific.get("publications")
        if publications is None or publications == []:
            _issue(
                issues,
                "warning",
                "scientific.publications_missing",
                "No publications are recorded for this table.",
                "scientific.publications",
            )
        elif not isinstance(publications, list):
            _issue(
                issues,
                "error",
                "scientific.publications_type",
                "scientific.publications must be a list.",
                "scientific.publications",
            )
        else:
            for index, publication in enumerate(publications):
                path = "scientific.publications.%d" % index
                if not isinstance(publication, dict):
                    _issue(
                        issues,
                        "error",
                        "scientific.publication_type",
                        "Each publication must be an object.",
                        path,
                    )
                    continue
                for key in ("type", "title", "doi", "arxiv", "url"):
                    value = publication.get(key)
                    if value is not None and (not isinstance(value, str) or not value):
                        _issue(
                            issues,
                            "error",
                            "scientific.publication_value",
                            "Publication '%s' must be a non-empty string." % key,
                            "%s.%s" % (path, key),
                        )
                authors = publication.get("authors")
                if authors is not None and (
                    not isinstance(authors, list)
                    or not all(isinstance(author, str) and author for author in authors)
                ):
                    _issue(
                        issues,
                        "error",
                        "scientific.publication_authors",
                        "Publication authors must be a list of non-empty strings.",
                        path + ".authors",
                    )
                if not any(publication.get(key) for key in ("title", "doi", "arxiv", "url")):
                    _issue(
                        issues,
                        "error",
                        "scientific.publication_identifier",
                        "A publication needs a title, DOI, arXiv identifier, or URL.",
                        path,
                    )

        origin = scientific.get("origin")
        if origin is None:
            _issue(
                issues,
                "warning",
                "scientific.origin_missing",
                "No generator or origin is recorded for this table.",
                "scientific.origin",
            )
        elif not isinstance(origin, dict):
            _issue(
                issues,
                "error",
                "scientific.origin_type",
                "scientific.origin must be an object or null.",
                "scientific.origin",
            )
        else:
            for key in ("generator", "converter", "version", "url"):
                value = origin.get(key)
                if value is not None and (not isinstance(value, str) or not value):
                    _issue(
                        issues,
                        "error",
                        "scientific.origin_value",
                        "Origin '%s' must be a non-empty string." % key,
                        "scientific.origin.%s" % key,
                    )

        constraints = scientific.get("constraints", [])
        if not isinstance(constraints, list) or not all(
            isinstance(value, (str, dict)) for value in constraints
        ):
            _issue(
                issues,
                "error",
                "scientific.constraints_type",
                "constraints must be a list of strings or objects.",
                "scientific.constraints",
            )

        open_questions = scientific.get("open_questions", [])
        if not isinstance(open_questions, list) or not all(
            isinstance(value, str) and value for value in open_questions
        ):
            _issue(
                issues,
                "error",
                "scientific.open_questions_type",
                "open_questions must be a list of non-empty strings.",
                "scientific.open_questions",
            )

        validity = scientific.get("validity")
        if validity is not None and not isinstance(validity, dict):
            _issue(
                issues,
                "error",
                "scientific.validity_type",
                "scientific.validity must be an object or null.",
                "scientific.validity",
            )

        phase = scientific.get("phase_structure")
        if not isinstance(phase, dict):
            _issue(
                issues,
                "warning",
                "scientific.phase_unknown",
                "Phase-structure information is not documented.",
                "scientific.phase_structure",
            )
            return None
        status = phase.get("status")
        if status not in {"known", "none", "unknown"}:
            _issue(
                issues,
                "error",
                "scientific.phase_status",
                "phase_structure.status must be known, none, or unknown.",
                "scientific.phase_structure.status",
            )
        elif status == "unknown":
            _issue(
                issues,
                "warning",
                "scientific.phase_unknown",
                "Phase-structure information is explicitly marked unknown.",
                "scientific.phase_structure.status",
            )

        critical_points = phase.get("critical_points", [])
        if not isinstance(critical_points, list):
            _issue(
                issues,
                "error",
                "scientific.critical_points_type",
                "critical_points must be a list.",
                "scientific.phase_structure.critical_points",
            )
            critical_points = []
        else:
            for index, point in enumerate(critical_points):
                path = "scientific.phase_structure.critical_points.%d" % index
                if not isinstance(point, dict):
                    _issue(
                        issues,
                        "error",
                        "scientific.critical_point_type",
                        "Each critical point must be an object.",
                        path,
                    )
                    continue
                point_name = point.get("name")
                if not isinstance(point_name, str) or not point_name:
                    _issue(
                        issues,
                        "error",
                        "scientific.critical_point_name",
                        "A critical point needs a non-empty name.",
                        path + ".name",
                    )
                point_status = point.get("status")
                if point_status not in {"known", "approximate", "not_provided"}:
                    _issue(
                        issues,
                        "error",
                        "scientific.critical_point_status",
                        "Critical-point status must be known, approximate, or not_provided.",
                        path + ".status",
                    )
                location = point.get("location")
                if location is None:
                    if point_status in {"known", "approximate"}:
                        _issue(
                            issues,
                            "error",
                            "scientific.critical_point_location",
                            "A known or approximate critical point requires a location.",
                            path + ".location",
                        )
                    else:
                        _issue(
                            issues,
                            "warning",
                            "scientific.critical_point_location_missing",
                            "A critical point is documented but its location is not supplied.",
                            path + ".location",
                        )
                elif not isinstance(location, dict) or not location:
                    _issue(
                        issues,
                        "error",
                        "scientific.critical_point_location",
                        "location must be null or an object of named coordinates.",
                        path + ".location",
                    )
                else:
                    independent_coordinates = {
                        name for name, spec in fields.items() if spec.role == "coordinate"
                    }
                    for coordinate, descriptor in location.items():
                        coordinate_path = "%s.location.%s" % (path, coordinate)
                        if coordinate not in independent_coordinates:
                            _issue(
                                issues,
                                "error",
                                "scientific.critical_point_coordinate",
                                "'%s' is not an independent coordinate of this EoS."
                                % coordinate,
                                coordinate_path,
                            )
                        if not isinstance(descriptor, dict):
                            _issue(
                                issues,
                                "error",
                                "scientific.critical_point_coordinate",
                                "Each critical-point coordinate needs value and unit.",
                                coordinate_path,
                            )
                            continue
                        if _finite_float(descriptor.get("value")) is None:
                            _issue(
                                issues,
                                "error",
                                "scientific.critical_point_value",
                                "Critical-point coordinate value must be finite.",
                                coordinate_path + ".value",
                            )
                        unit = descriptor.get("unit")
                        if not isinstance(unit, str) or not unit:
                            _issue(
                                issues,
                                "error",
                                "scientific.critical_point_unit",
                                "Critical-point coordinate unit must be a non-empty string.",
                                coordinate_path + ".unit",
                            )
                    if (
                        point_status in {"known", "approximate"}
                        and set(location) != independent_coordinates
                    ):
                        _issue(
                            issues,
                            "error",
                            "scientific.critical_point_coordinates_missing",
                            "A located critical point must provide every independent coordinate.",
                            path + ".location",
                            expected=sorted(independent_coordinates),
                            actual=sorted(location),
                        )
                note = point.get("note")
                if note is not None and not isinstance(note, str):
                    _issue(
                        issues,
                        "error",
                        "scientific.critical_point_note",
                        "Critical-point note must be a string or null.",
                        path + ".note",
                    )

        transition = phase.get("first_order_transition")
        transition_spec = self._parse_first_order_transition(
            transition,
            critical_points,
            fields,
            dimension,
            issues,
        )

        transition_claimed = (
            isinstance(transition, dict) and transition.get("present") is not False
        )
        if status == "none" and (critical_points or transition_claimed):
            _issue(
                issues,
                "error",
                "scientific.phase_status_inconsistent",
                "status='none' cannot include critical points or a transition object.",
                "scientific.phase_structure",
            )
        if (
            status == "unknown"
            and isinstance(transition, dict)
            and transition.get("present") is True
        ):
            _issue(
                issues,
                "error",
                "scientific.phase_status_inconsistent",
                "A documented first-order transition requires phase status='known'.",
                "scientific.phase_structure.status",
            )

        return transition_spec

    def _parse_first_order_transition(
        self,
        transition: Any,
        critical_points: Sequence[Any],
        fields: Mapping[str, FieldSpec],
        dimension: Any,
        issues: List[ValidationIssue],
    ) -> Optional[FirstOrderTransitionSpec]:
        path = "scientific.phase_structure.first_order_transition"
        if transition is None:
            return None
        if not isinstance(transition, dict):
            _issue(
                issues,
                "error",
                "scientific.first_order_type",
                "first_order_transition must be an object or null.",
                path,
            )
            return None
        _check_unknown_keys(
            transition,
            {
                "present",
                "representation",
                "kind",
                "phase_fraction_field",
                "mixed_phase_condition",
                "critical_point",
                "critical_point_tolerance",
                "coexistence_data",
            },
            path,
            issues,
        )
        present = transition.get("present")
        if not isinstance(present, bool):
            _issue(
                issues,
                "error",
                "scientific.first_order_present",
                "first_order_transition.present must be a boolean.",
                path + ".present",
            )
            return None
        if not present:
            unexpected = sorted(set(transition) - {"present"})
            if unexpected:
                _issue(
                    issues,
                    "error",
                    "scientific.first_order_disabled_fields",
                    "A disabled transition may contain only present=false.",
                    path,
                    unexpected=unexpected,
                )
            return None

        error_count = sum(issue.severity == "error" for issue in issues)
        if dimension != 2:
            _issue(
                issues,
                "error",
                "scientific.first_order_dimension",
                "First-order transition handling is supported only for dimension=2.",
                path,
            )

        representation = transition.get("representation")
        if representation not in _FIRST_ORDER_REPRESENTATIONS:
            _issue(
                issues,
                "error",
                "scientific.first_order_representation",
                "representation must be 'line' or 'binodals'.",
                path + ".representation",
            )

        kind = transition.get("kind")
        if not isinstance(kind, str) or not kind:
            _issue(
                issues,
                "error",
                "scientific.first_order_kind",
                "A present first-order transition requires a kind.",
                path + ".kind",
            )

        phase_field = transition.get("phase_fraction_field")
        if phase_field is not None:
            if not isinstance(phase_field, str) or phase_field not in fields:
                _issue(
                    issues,
                    "error",
                    "scientific.phase_fraction_field",
                    "phase_fraction_field must name a metadata field.",
                    path + ".phase_fraction_field",
                )
            elif fields[phase_field].role != "auxiliary":
                _issue(
                    issues,
                    "error",
                    "scientific.phase_fraction_field",
                    "phase_fraction_field must be an auxiliary field.",
                    path + ".phase_fraction_field",
                )

        mixed_condition = transition.get("mixed_phase_condition")
        if mixed_condition is not None and (
            not isinstance(mixed_condition, str) or not mixed_condition
        ):
            _issue(
                issues,
                "error",
                "scientific.mixed_phase_condition",
                "mixed_phase_condition must be a non-empty string or null.",
                path + ".mixed_phase_condition",
            )

        coexistence = self._parse_coexistence_metadata(
            transition.get("coexistence_data"), representation, path, issues
        )

        critical_name = transition.get("critical_point")
        critical_coordinates: Dict[str, float] = {}
        absolute_tolerance = 0.0
        relative_tolerance = 0.0
        if representation == "line":
            if not isinstance(critical_name, str) or not critical_name:
                _issue(
                    issues,
                    "error",
                    "scientific.first_order_critical_point",
                    "The line representation requires a named critical point.",
                    path + ".critical_point",
                )
            else:
                matches = [
                    point
                    for point in critical_points
                    if isinstance(point, Mapping) and point.get("name") == critical_name
                ]
                if len(matches) != 1:
                    _issue(
                        issues,
                        "error",
                        "scientific.first_order_critical_point",
                        "critical_point must uniquely name a declared critical point.",
                        path + ".critical_point",
                    )
                else:
                    point = matches[0]
                    if point.get("status") not in {"known", "approximate"}:
                        _issue(
                            issues,
                            "error",
                            "scientific.first_order_critical_point",
                            "The LINE critical point must have known or approximate status.",
                            path + ".critical_point",
                        )
                    location = point.get("location")
                    if isinstance(location, Mapping):
                        for coordinate in ("T", "muB"):
                            descriptor = location.get(coordinate)
                            if not isinstance(descriptor, Mapping):
                                continue
                            numeric = _finite_float(descriptor.get("value"))
                            unit = descriptor.get("unit")
                            scale = _CANONICAL_ENERGY_UNIT_SCALES.get(unit)
                            if numeric is not None and scale is not None:
                                critical_coordinates[coordinate] = numeric * scale
                            elif numeric is not None:
                                _issue(
                                    issues,
                                    "error",
                                    "scientific.first_order_critical_point_unit",
                                    "LINE critical-point units must be GeV or MeV.",
                                    "scientific.phase_structure.critical_points",
                                    coordinate=coordinate,
                                    unit=unit,
                                )

            tolerance = transition.get("critical_point_tolerance")
            tolerance_path = path + ".critical_point_tolerance"
            if not isinstance(tolerance, dict):
                _issue(
                    issues,
                    "error",
                    "scientific.first_order_critical_tolerance",
                    "The line representation requires a tolerance object.",
                    tolerance_path,
                )
            else:
                _check_unknown_keys(
                    tolerance, {"absolute", "relative"}, tolerance_path, issues
                )
                for key in ("absolute", "relative"):
                    numeric = _finite_float(tolerance.get(key))
                    if numeric is None or numeric < 0.0:
                        _issue(
                            issues,
                            "error",
                            "scientific.first_order_critical_tolerance",
                            "Critical-point tolerances must be finite and non-negative.",
                            tolerance_path + "." + key,
                        )
                    elif key == "absolute":
                        absolute_tolerance = numeric
                    else:
                        relative_tolerance = numeric
        elif critical_name is not None:
            _issue(
                issues,
                "error",
                "scientific.first_order_critical_point",
                "critical_point is currently supported only for representation='line'.",
                path + ".critical_point",
            )

        if sum(issue.severity == "error" for issue in issues) != error_count:
            return None
        assert isinstance(representation, str)
        assert isinstance(kind, str)
        assert coexistence is not None
        return FirstOrderTransitionSpec(
            representation=representation,
            kind=kind,
            coexistence_data=coexistence,
            phase_fraction_field=phase_field,
            mixed_phase_condition=mixed_condition,
            critical_point_name=critical_name,
            critical_point_coordinates=critical_coordinates,
            critical_point_absolute_tolerance=absolute_tolerance,
            critical_point_relative_tolerance=relative_tolerance,
        )

    def _parse_coexistence_metadata(
        self,
        coexistence: Any,
        representation: Any,
        transition_path: str,
        issues: List[ValidationIssue],
    ) -> Optional[CoexistenceDataSpec]:
        path = transition_path + ".coexistence_data"
        if not isinstance(coexistence, dict):
            _issue(
                issues,
                "error",
                "scientific.coexistence_type",
                "A present transition requires a coexistence_data object.",
                path,
            )
            return None
        _check_unknown_keys(
            coexistence,
            {
                "file",
                "delimiter",
                "comment_prefix",
                "skip_rows",
                "encoding",
                "expected_columns",
                "fields",
                "columns",
            },
            path,
            issues,
        )
        error_count = sum(issue.severity == "error" for issue in issues)
        relative_path = coexistence.get("file")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or "\x00" in relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            _issue(
                issues,
                "error",
                "scientific.coexistence_file",
                "coexistence_data.file must be a non-empty relative path.",
                path + ".file",
            )

        delimiter_raw = coexistence.get("delimiter", "whitespace")
        if delimiter_raw == "whitespace":
            delimiter = None
        elif isinstance(delimiter_raw, str) and len(delimiter_raw) == 1:
            delimiter = delimiter_raw
        else:
            delimiter = None
            _issue(
                issues,
                "error",
                "scientific.coexistence_delimiter",
                "delimiter must be 'whitespace' or a single character.",
                path + ".delimiter",
            )

        comment = coexistence.get("comment_prefix", "#")
        if comment is not None and (not isinstance(comment, str) or not comment):
            _issue(
                issues,
                "error",
                "scientific.coexistence_comment",
                "comment_prefix must be a non-empty string or null.",
                path + ".comment_prefix",
            )

        skip_rows = coexistence.get("skip_rows", 0)
        if not _is_int(skip_rows) or skip_rows < 0:
            _issue(
                issues,
                "error",
                "scientific.coexistence_skip_rows",
                "skip_rows must be a non-negative integer.",
                path + ".skip_rows",
            )

        encoding = coexistence.get("encoding", "utf-8")
        if not isinstance(encoding, str) or not encoding:
            _issue(
                issues,
                "error",
                "scientific.coexistence_encoding",
                "encoding must be a non-empty string.",
                path + ".encoding",
            )
        else:
            try:
                codecs.lookup(encoding)
            except LookupError:
                _issue(
                    issues,
                    "error",
                    "scientific.coexistence_encoding",
                    "Unknown coexistence-data encoding '%s'." % encoding,
                    path + ".encoding",
                )

        expected_columns = coexistence.get("expected_columns")
        if (
            not _is_int(expected_columns)
            or expected_columns < 1
            or expected_columns > MAX_TABLE_COLUMNS
        ):
            _issue(
                issues,
                "error",
                "scientific.coexistence_columns",
                "expected_columns must be an integer from 1 through %d."
                % MAX_TABLE_COLUMNS,
                path + ".expected_columns",
            )

        sidecar_fields = coexistence.get("fields")
        parsed_fields: Tuple[str, ...] = ()
        if sidecar_fields is not None:
            valid_fields = (
                isinstance(sidecar_fields, list)
                and all(isinstance(value, str) and value for value in sidecar_fields)
                and len(set(sidecar_fields)) == len(sidecar_fields)
            )
            if not valid_fields:
                _issue(
                    issues,
                    "error",
                    "scientific.coexistence_fields",
                    "coexistence fields must be unique non-empty strings.",
                    path + ".fields",
                )
            elif _is_int(expected_columns) and len(sidecar_fields) != expected_columns:
                _issue(
                    issues,
                    "error",
                    "scientific.coexistence_fields",
                    "coexistence field count must match expected_columns.",
                    path + ".fields",
                )
            else:
                parsed_fields = tuple(sidecar_fields)

        required_names = _FIRST_ORDER_REQUIRED_COLUMNS.get(representation, ())
        raw_columns = coexistence.get("columns")
        parsed_columns: Dict[str, CoexistenceColumnSpec] = {}
        used_indices: Dict[int, str] = {}
        if not isinstance(raw_columns, dict):
            _issue(
                issues,
                "error",
                "scientific.coexistence_semantic_columns",
                "columns must be an object keyed by semantic field name.",
                path + ".columns",
            )
        else:
            missing = sorted(set(required_names) - set(raw_columns))
            extra = sorted(set(raw_columns) - set(required_names))
            if missing or extra:
                _issue(
                    issues,
                    "error",
                    "scientific.coexistence_semantic_columns",
                    "columns must exactly define the representation's required fields.",
                    path + ".columns",
                    required=list(required_names),
                    missing=missing,
                    unexpected=extra,
                )
            for name, descriptor in raw_columns.items():
                column_path = "%s.columns.%s" % (path, name)
                if not isinstance(descriptor, dict):
                    _issue(
                        issues,
                        "error",
                        "scientific.coexistence_column",
                        "Each semantic column must be an object.",
                        column_path,
                    )
                    continue
                _check_unknown_keys(
                    descriptor, {"index", "unit"}, column_path, issues
                )
                index = descriptor.get("index")
                if (
                    not _is_int(index)
                    or index < 0
                    or (_is_int(expected_columns) and index >= expected_columns)
                ):
                    _issue(
                        issues,
                        "error",
                        "scientific.coexistence_column_index",
                        "Column index must lie inside expected_columns.",
                        column_path + ".index",
                    )
                elif index in used_indices:
                    _issue(
                        issues,
                        "error",
                        "scientific.coexistence_duplicate_column",
                        "Column %d is already assigned to '%s'."
                        % (index, used_indices[index]),
                        column_path + ".index",
                    )
                else:
                    used_indices[index] = name
                unit = descriptor.get("unit")
                scale = _CANONICAL_ENERGY_UNIT_SCALES.get(unit)
                if scale is None:
                    _issue(
                        issues,
                        "error",
                        "scientific.coexistence_column_unit",
                        "First-order coordinate units must be GeV or MeV.",
                        column_path + ".unit",
                    )
                if _is_int(index) and index >= 0 and scale is not None:
                    parsed_columns[name] = CoexistenceColumnSpec(
                        name=name,
                        column=index,
                        source_unit=unit,
                        canonical_unit="GeV",
                        scale_factor=scale,
                    )

        if sum(issue.severity == "error" for issue in issues) != error_count:
            return None
        assert isinstance(relative_path, str)
        assert isinstance(comment, str) or comment is None
        assert _is_int(skip_rows)
        assert isinstance(encoding, str)
        assert _is_int(expected_columns)
        return CoexistenceDataSpec(
            file=relative_path,
            delimiter=delimiter,
            comment_prefix=comment,
            skip_rows=skip_rows,
            encoding=encoding,
            expected_columns=expected_columns,
            columns=parsed_columns,
            fields=parsed_fields,
        )

    def _validate_scientific_files(self, issues: List[ValidationIssue]) -> None:
        assert self._metadata is not None
        transition = self._metadata.first_order_transition
        if transition is None:
            return
        relative_path = transition.coexistence_data.file
        metadata_directory = self.metadata_path.parent.resolve()
        try:
            path = (metadata_directory / relative_path).resolve()
            contained = path == metadata_directory or metadata_directory in path.parents
            exists = path.is_file()
        except (OSError, ValueError) as exc:
            _issue(
                issues,
                "error",
                "scientific.coexistence_file",
                "Invalid coexistence data path: %s" % exc,
                "scientific.phase_structure.first_order_transition.coexistence_data.file",
            )
            return
        if not contained:
            _issue(
                issues,
                "error",
                "scientific.coexistence_file",
                "Coexistence data must remain inside the metadata directory.",
                "scientific.phase_structure.first_order_transition.coexistence_data.file",
            )
            return
        if not exists:
            _issue(
                issues,
                "error",
                "scientific.coexistence_file_missing",
                "Referenced coexistence data file does not exist: %s" % path,
                str(path),
            )

    def _load_and_validate_coexistence_data(
        self, issues: List[ValidationIssue]
    ) -> Optional[Mapping[str, Any]]:
        """Load one sidecar, convert semantic columns, and validate its domain."""

        assert self._metadata is not None
        assert self._grid is not None
        transition = self._metadata.first_order_transition
        if transition is None:
            return None
        spec = transition.coexistence_data
        path = (self.metadata_path.parent / spec.file).resolve()
        try:
            raw = np.loadtxt(
                str(path),
                dtype=float,
                comments=spec.comment_prefix,
                delimiter=spec.delimiter,
                skiprows=spec.skip_rows,
                ndmin=2,
                encoding=spec.encoding,
            )
        except (LookupError, OSError, OverflowError, TypeError, ValueError) as exc:
            _issue(
                issues,
                "error",
                "scientific.coexistence_load_failed",
                "Could not read coexistence data %s: %s" % (path, exc),
                str(path),
            )
            return None
        if raw.ndim != 2 or raw.shape[0] < 2:
            _issue(
                issues,
                "error",
                "scientific.coexistence_rows",
                "Coexistence data must contain at least two numeric rows.",
                str(path),
            )
            return None
        if raw.shape[1] != spec.expected_columns:
            _issue(
                issues,
                "error",
                "scientific.coexistence_column_count",
                "Expected %d coexistence columns, found %d."
                % (spec.expected_columns, raw.shape[1]),
                str(path),
            )
            return None
        if not np.all(np.isfinite(raw)):
            positions = np.argwhere(~np.isfinite(raw))
            _issue(
                issues,
                "error",
                "scientific.coexistence_nonfinite",
                "Coexistence data cannot contain NaN or infinity.",
                str(path),
                count=int(positions.shape[0]),
                first_position=[int(value) for value in positions[0]],
            )
            return None

        names = _FIRST_ORDER_REQUIRED_COLUMNS[transition.representation]
        columns = []
        units = []
        for name in names:
            column = spec.columns[name]
            with np.errstate(over="ignore", invalid="ignore"):
                values = raw[:, column.column] * column.scale_factor
            if not np.all(np.isfinite(values)):
                _issue(
                    issues,
                    "error",
                    "scientific.coexistence_conversion",
                    "Canonical conversion produced non-finite values for '%s'."
                    % name,
                    str(path),
                )
                return None
            columns.append(values)
            units.append(column.canonical_unit)
        canonical = np.column_stack(columns)

        mu_values = canonical[:, names.index("muB")]
        differences = np.diff(mu_values)
        if np.any(differences == 0.0):
            duplicate = int(np.flatnonzero(differences == 0.0)[0])
            _issue(
                issues,
                "error",
                "scientific.coexistence_muB_duplicate",
                "Coexistence muB values must be unique.",
                str(path),
                rows=[duplicate, duplicate + 1],
                value=float(mu_values[duplicate]),
            )
        if np.any(differences < 0.0):
            reversed_index = int(np.flatnonzero(differences < 0.0)[0])
            _issue(
                issues,
                "error",
                "scientific.coexistence_muB_order",
                "Coexistence muB values must be strictly increasing.",
                str(path),
                rows=[reversed_index, reversed_index + 1],
            )

        axis_by_name = dict(zip(self._grid.coordinate_names, self._grid.axes))
        tolerance = self._metadata.grid
        for semantic_name, values in zip(names, canonical.T):
            coordinate = "muB" if semantic_name == "muB" else "T"
            axis = axis_by_name[coordinate]
            lower = float(axis[0])
            upper = float(axis[-1])
            magnitude = np.maximum.reduce(
                (np.abs(values), np.full(values.shape, abs(lower)), np.full(values.shape, abs(upper)))
            )
            allowed = tolerance.absolute_tolerance + tolerance.relative_tolerance * magnitude
            outside = (values < lower - allowed) | (values > upper + allowed)
            if np.any(outside):
                index = int(np.flatnonzero(outside)[0])
                _issue(
                    issues,
                    "error",
                    "scientific.coexistence_out_of_domain",
                    "Coexistence field '%s' lies outside the validated %s axis."
                    % (semantic_name, coordinate),
                    str(path),
                    row=index,
                    value=float(values[index]),
                    bounds=[lower, upper],
                )

        critical_endpoint: Optional[str] = None
        if transition.representation == "line":
            coordinate_indices = {name: names.index(name) for name in ("T", "muB")}
            matches = []
            for endpoint_name, row_index in (("first", 0), ("last", -1)):
                endpoint_matches = []
                for coordinate, column_index in coordinate_indices.items():
                    expected = transition.critical_point_coordinates[coordinate]
                    actual = float(canonical[row_index, column_index])
                    endpoint_matches.append(
                        math.isclose(
                            actual,
                            expected,
                            rel_tol=transition.critical_point_relative_tolerance,
                            abs_tol=transition.critical_point_absolute_tolerance,
                        )
                    )
                if all(endpoint_matches):
                    matches.append(endpoint_name)
            if len(matches) != 1:
                _issue(
                    issues,
                    "error",
                    "scientific.coexistence_critical_endpoint",
                    "The declared critical point must match exactly one LINE endpoint.",
                    str(path),
                    matches=matches,
                    critical_point=dict(transition.critical_point_coordinates),
                )
            else:
                critical_endpoint = matches[0]

        if any(
            issue.severity == "error"
            and issue.code.startswith("scientific.coexistence_")
            for issue in issues
        ):
            return None

        fingerprint_descriptor = json.dumps(
            {
                "schema": "coexistence_table_v1",
                "representation": transition.representation,
                "names": list(names),
                "units": units,
                "source_shape": list(raw.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(b"EoSInverter:CoexistenceTable:v1\0")
        digest.update(len(fingerprint_descriptor).to_bytes(8, "big"))
        digest.update(fingerprint_descriptor)
        _update_digest_array(digest, raw)
        _update_digest_array(digest, canonical)
        fingerprint = digest.hexdigest()
        self._coexistence_table = CoexistenceTable(
            representation=transition.representation,
            names=names,
            units=tuple(units),
            values=canonical,
            source_row_count=int(raw.shape[0]),
            source_column_count=int(raw.shape[1]),
            fingerprint=fingerprint,
            critical_endpoint=critical_endpoint,
        )
        _issue(
            issues,
            "info",
            "scientific.coexistence_loaded",
            "Loaded and validated canonical coexistence data.",
            str(path),
            representation=transition.representation,
            rows=int(raw.shape[0]),
            fingerprint=fingerprint,
        )
        return {
            "representation": transition.representation,
            "rows": int(raw.shape[0]),
            "source_columns": int(raw.shape[1]),
            "semantic_fields": list(names),
            "canonical_units": list(units),
            "bounds": {
                name: [float(np.min(values)), float(np.max(values))]
                for name, values in zip(names, canonical.T)
            },
            "critical_endpoint": critical_endpoint,
            "fingerprint": fingerprint,
        }

    def _validate_table_shape(self, issues: List[ValidationIssue]) -> None:
        assert self._metadata is not None
        assert self._raw_data is not None
        columns = self._raw_data.shape[1]
        expected = self._metadata.format.expected_columns
        if expected is not None and columns != expected:
            _issue(
                issues,
                "error",
                "data.column_count",
                "Expected %d columns, found %d." % (expected, columns),
                str(self.data_path),
                expected=expected,
                actual=columns,
            )
        for name, spec in self._metadata.fields.items():
            if spec.column >= columns:
                _issue(
                    issues,
                    "error",
                    "data.column_out_of_range",
                    "Field '%s' selects column %d, but the table has %d columns."
                    % (name, spec.column, columns),
                    "fields.%s.column" % name,
                    field=name,
                    column=spec.column,
                    columns=columns,
                )
        unmapped = columns - len(self._metadata.fields)
        if unmapped > 0:
            _issue(
                issues,
                "info",
                "data.unmapped_columns",
                "%d table column(s) are intentionally not mapped by metadata." % unmapped,
                str(self.data_path),
                count=unmapped,
            )

    def _validate_field_values(
        self,
        fields: Mapping[str, np.ndarray],
        issues: List[ValidationIssue],
    ) -> Dict[str, Any]:
        assert self._metadata is not None
        statistics: Dict[str, Any] = {}
        for name, spec in self._metadata.fields.items():
            values = fields[name]
            nan_count = int(np.count_nonzero(np.isnan(values)))
            positive_inf = int(np.count_nonzero(np.isposinf(values)))
            negative_inf = int(np.count_nonzero(np.isneginf(values)))
            finite = values[np.isfinite(values)]
            statistics[name] = {
                "nan_count": nan_count,
                "positive_infinity_count": positive_inf,
                "negative_infinity_count": negative_inf,
                "finite_minimum": float(np.min(finite)) if finite.size else None,
                "finite_maximum": float(np.max(finite)) if finite.size else None,
                "source_unit": spec.source_unit,
                "canonical_unit": spec.canonical_unit,
            }

            infinity_count = positive_inf + negative_inf
            if infinity_count:
                code_role = "coordinate" if spec.role == "coordinate" else spec.role
                nonfinite_indices = np.flatnonzero(np.isinf(values))[:8]
                _issue(
                    issues,
                    "error",
                    "field.nonfinite.%s.infinity" % code_role,
                    "Field '%s' contains %d infinite value(s)." % (name, infinity_count),
                    "fields.%s" % name,
                    field=name,
                    count=infinity_count,
                    row_indices=[int(index) for index in nonfinite_indices],
                    values=[float(values[index]) for index in nonfinite_indices],
                )
            if nan_count and spec.nonfinite_policy == "forbid":
                code_role = "coordinate" if spec.role == "coordinate" else spec.role
                nonfinite_indices = np.flatnonzero(np.isnan(values))[:8]
                _issue(
                    issues,
                    "error",
                    "field.nonfinite.%s.nan" % code_role,
                    "Field '%s' contains %d forbidden NaN value(s)." % (name, nan_count),
                    "fields.%s" % name,
                    field=name,
                    count=nan_count,
                    row_indices=[int(index) for index in nonfinite_indices],
                    values=["nan" for _ in nonfinite_indices],
                )
            elif nan_count:
                _issue(
                    issues,
                    "warning",
                    "field.nan_allowed",
                    "Field '%s' retains %d explicitly allowed NaN value(s)."
                    % (name, nan_count),
                    "fields.%s" % name,
                    field=name,
                    count=nan_count,
                )
        return statistics

    def _record_geometry_issues(
        self, statistics: Mapping[str, Any], issues: List[ValidationIssue]
    ) -> None:
        """Translate local geometry diagnostics into conservative policy."""

        counts = statistics["counts"]
        positive = int(counts[POSITIVE])
        negative = int(counts[NEGATIVE])
        singular = int(counts[SINGULAR])
        invalid = int(counts[INVALID])
        ill_conditioned = int(counts["ill_conditioned"])
        sampled = int(statistics["sampled_cells"])
        location = str(self.data_path) if self.data_path is not None else "grid"

        if invalid:
            _issue(
                issues,
                "warning" if self._allow_unsuitable_geometry else "error",
                "geometry.invalid_jacobian",
                "The local geometry scan found %d cell(s) with a non-finite "
                "or invalid Jacobian." % invalid,
                location,
                count=invalid,
                sampled_cells=sampled,
                locations=statistics["locations"],
            )
        regular = positive + negative
        if sampled and regular == 0 and not invalid:
            _issue(
                issues,
                "warning" if self._allow_unsuitable_geometry else "error",
                "geometry.degenerate_map",
                "Every inspected source cell is locally singular; this table "
                "cannot define a stable inverse.",
                location,
                sampled_cells=sampled,
            )
        elif singular:
            isolated = statistics.get("isolated_defects")
            if isolated == singular:
                message = (
                    "Local geometry contains %d isolated sign-changing or "
                    "singular cell(s); inversion remains allowed and the "
                    "locations are recorded in preprocessing diagnostics."
                    % singular
                )
            else:
                message = (
                    "Local geometry contains %d sampled sign-changing or "
                    "singular cell(s); inspect preprocessing diagnostics "
                    "before relying on the inverse." % singular
                )
            _issue(
                issues,
                "warning",
                "geometry.sign_changing_cells",
                message,
                location,
                count=singular,
                sampled_cells=sampled,
                isolated_defects=isolated,
                locations=statistics["locations"],
            )
        if positive and negative:
            declared_first_order = (
                self._metadata is not None
                and self._metadata.first_order_transition is not None
            )
            allow_orientation_reversal = (
                declared_first_order or self._allow_unsuitable_geometry
            )
            if declared_first_order:
                message = (
                    "Both positive and negative regular local Jacobian "
                    "orientations were observed in a declared first-order "
                    "table. The reversal is retained as an explicit warning; "
                    "inversion must use the dedicated phase-aware route."
                )
            elif self._allow_unsuitable_geometry:
                message = (
                    "Both positive and negative regular local Jacobian "
                    "orientations were observed. The internal analytical-test "
                    "escape hatch keeps this as a warning."
                )
            else:
                message = (
                    "Both positive and negative regular local Jacobian "
                    "orientations were observed. This violates the local "
                    "orientation consistency required for an ordinary EoS "
                    "table and is substantial evidence of a fold or "
                    "non-injective forward map; inversion is refused. Either "
                    "uniform orientation sign is acceptable, but the sign "
                    "must not reverse across regular cells. Declare genuine "
                    "first-order structure in metadata so that the dedicated "
                    "phase-aware route is used."
                )
            _issue(
                issues,
                "warning" if allow_orientation_reversal else "error",
                "geometry.orientation_reversal",
                message,
                location,
                positive_cells=positive,
                negative_cells=negative,
                sign_changing_or_singular_cells=singular,
                invalid_cells=invalid,
                sampled_cells=sampled,
                declared_first_order=declared_first_order,
            )
        if ill_conditioned:
            _issue(
                issues,
                "warning",
                "geometry.ill_conditioned_cells",
                "%d sampled cell(s) have a center Jacobian condition number "
                "above %.3g." % (
                    ill_conditioned,
                    float(statistics.get("condition_warning", 0.0)),
                ),
                location,
                count=ill_conditioned,
                sampled_cells=sampled,
                locations=statistics["locations"],
            )

    def _validate_charge_constraint_values(
        self,
        fields: Mapping[str, np.ndarray],
        issues: List[ValidationIssue],
    ) -> Dict[str, Any]:
        """Check any provided dependent-density columns against metadata."""

        assert self._metadata is not None
        constraints = self._metadata.charge_constraints
        statistics: Dict[str, Any] = {
            "source": constraints.source,
            "model": constraints.model,
            "transform_id": constraints.identifier,
            "relations": {},
        }
        for dependent_name in constraints.dependent_density_names:
            relation = constraints.dependent_densities[dependent_name]
            relation_statistics: Dict[str, Any] = {
                "coefficients": dict(relation),
                "field_available": dependent_name in fields,
            }
            statistics["relations"][dependent_name] = relation_statistics
            if dependent_name not in fields:
                continue

            actual = np.asarray(fields[dependent_name], dtype=float)
            expected = np.zeros(actual.shape, dtype=float)
            with np.errstate(over="ignore", invalid="ignore"):
                for source_name, coefficient in relation.items():
                    expected += coefficient * np.asarray(
                        fields[source_name], dtype=float
                    )
            expected_nonfinite_count = int(
                np.count_nonzero(~np.isfinite(expected))
            )
            finite = np.isfinite(actual) & np.isfinite(expected)
            difference = np.abs(actual[finite] - expected[finite])
            threshold = constraints.absolute_tolerance + (
                constraints.relative_tolerance
                * np.maximum(np.abs(actual[finite]), np.abs(expected[finite]))
            )
            mismatch = difference > threshold
            mismatch_count = int(np.count_nonzero(mismatch))
            relation_statistics.update(
                {
                    "checked_rows": int(np.count_nonzero(finite)),
                    "nonfinite_expected_count": expected_nonfinite_count,
                    "mismatch_count": mismatch_count,
                    "maximum_absolute_difference": (
                        float(np.max(difference)) if difference.size else None
                    ),
                    "absolute_tolerance": constraints.absolute_tolerance,
                    "relative_tolerance": constraints.relative_tolerance,
                }
            )
            if expected_nonfinite_count:
                _issue(
                    issues,
                    "error",
                    "constraint.nonfinite_expected",
                    "The declared relation for '%s' produces %d non-finite value(s)."
                    % (dependent_name, expected_nonfinite_count),
                    "charge_constraints.dependent_densities.%s" % dependent_name,
                    field=dependent_name,
                    count=expected_nonfinite_count,
                )
            if mismatch_count:
                _issue(
                    issues,
                    "error",
                    "constraint.density_mismatch",
                    "Field '%s' violates its declared density relation in %d row(s)."
                    % (dependent_name, mismatch_count),
                    "charge_constraints.dependent_densities.%s" % dependent_name,
                    field=dependent_name,
                    mismatch_count=mismatch_count,
                    maximum_absolute_difference=(
                        float(np.max(difference)) if difference.size else None
                    ),
                )
        return statistics

    def _cluster_axis(
        self, values: np.ndarray, absolute_tolerance: float, relative_tolerance: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        unique, exact_inverse = np.unique(values, return_inverse=True)
        if unique.size < 2 or (absolute_tolerance == 0 and relative_tolerance == 0):
            axis = unique.astype(float, copy=True)
            inverse = exact_inverse.astype(np.intp, copy=False)
            return _immutable_array(axis), inverse

        groups = np.empty(unique.size, dtype=np.intp)
        centers: List[float] = []
        anchors: List[float] = []
        counts: List[int] = []
        for unique_index, value in enumerate(unique):
            numeric = float(value)
            if not centers:
                centers.append(numeric)
                anchors.append(numeric)
                counts.append(1)
                groups[unique_index] = 0
                continue
            anchor = anchors[-1]
            tolerance = absolute_tolerance + relative_tolerance * max(
                abs(anchor), abs(numeric)
            )
            if abs(numeric - anchor) <= tolerance:
                group = len(centers) - 1
                counts[group] += 1
                centers[group] += (numeric - centers[group]) / counts[group]
                groups[unique_index] = group
            else:
                centers.append(numeric)
                anchors.append(numeric)
                counts.append(1)
                groups[unique_index] = len(centers) - 1
        axis = np.asarray(centers, dtype=float)
        return _immutable_array(axis), groups[exact_inverse]

    def _analyze_grid(
        self,
        fields: Mapping[str, np.ndarray],
        issues: List[ValidationIssue],
    ) -> Tuple[
        Optional[RectilinearGrid],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, Any],
    ]:
        assert self._metadata is not None
        spec = self._metadata.grid
        coordinate_names = self._metadata.coordinate_names
        axes: List[np.ndarray] = []
        indices: List[np.ndarray] = []
        uniform_flags: List[bool] = []
        axis_statistics: Dict[str, Any] = {}

        for name in coordinate_names:
            axis, inverse = self._cluster_axis(
                fields[name], spec.absolute_tolerance, spec.relative_tolerance
            )
            axes.append(axis)
            indices.append(inverse)
            snapped = axis[inverse]
            snap_distance = np.abs(fields[name] - snapped)
            snapped_rows = int(np.count_nonzero(snap_distance))
            maximum_snap = float(np.max(snap_distance)) if snap_distance.size else 0.0
            differences = np.diff(axis)
            uniform = bool(
                differences.size <= 1
                or np.allclose(
                    differences,
                    differences[0],
                    atol=spec.absolute_tolerance,
                    rtol=spec.relative_tolerance,
                )
            )
            uniform_flags.append(uniform)
            axis_statistics[name] = {
                "count": int(axis.size),
                "minimum": float(axis[0]) if axis.size else None,
                "maximum": float(axis[-1]) if axis.size else None,
                "uniform": uniform,
                "minimum_spacing": float(np.min(differences)) if differences.size else None,
                "maximum_spacing": float(np.max(differences)) if differences.size else None,
                "clustered_rows": snapped_rows,
                "maximum_cluster_distance": maximum_snap,
            }
            if snapped_rows:
                _issue(
                    issues,
                    "info",
                    "grid.coordinates_clustered",
                    "%d value(s) on axis '%s' were snapped within coordinate tolerance."
                    % (snapped_rows, name),
                    "grid.%s" % name,
                    axis=name,
                    count=snapped_rows,
                    maximum_distance=maximum_snap,
                )
            if axis.size < spec.min_axis_points:
                _issue(
                    issues,
                    "error",
                    "grid.axis_too_short",
                    "Axis '%s' has %d point(s); at least %d are required."
                    % (name, axis.size, spec.min_axis_points),
                    "grid.%s" % name,
                    axis=name,
                    actual=int(axis.size),
                    required=spec.min_axis_points,
                )
            if not uniform:
                severity = "error" if spec.require_uniform else "info"
                _issue(
                    issues,
                    severity,
                    "grid.nonuniform",
                    "Axis '%s' is nonuniform%s."
                    % (name, " but uniform spacing is required" if spec.require_uniform else ""),
                    "grid.%s" % name,
                    axis=name,
                )

        shape = tuple(int(axis.size) for axis in axes)
        expected_cells = math.prod(shape)
        row_count = len(next(iter(fields.values())))
        grid_statistics: Dict[str, Any] = {
            "coordinate_order": list(coordinate_names),
            "shape": list(shape),
            "axes": axis_statistics,
            "expected_cells_from_axes": expected_cells,
            "rows": row_count,
        }

        if spec.expected_shape is not None and shape != spec.expected_shape:
            _issue(
                issues,
                "error",
                "grid.expected_shape_mismatch",
                "Grid shape %s does not match metadata expectation %s."
                % (shape, spec.expected_shape),
                "grid.expected_shape",
                actual=shape,
                expected=spec.expected_shape,
            )

        for name, axis in zip(coordinate_names, axes):
            tolerance = spec.absolute_tolerance
            relative = spec.relative_tolerance
            if name in spec.expected_bounds:
                expected_minimum, expected_maximum = spec.expected_bounds[name]
                minimum_matches = np.isclose(
                    axis[0], expected_minimum, atol=tolerance, rtol=relative
                )
                maximum_matches = np.isclose(
                    axis[-1], expected_maximum, atol=tolerance, rtol=relative
                )
                if not minimum_matches or not maximum_matches:
                    _issue(
                        issues,
                        "error",
                        "grid.expected_bounds_mismatch",
                        "Axis '%s' bounds do not match metadata." % name,
                        "grid.expected_bounds.%s" % name,
                        actual=(float(axis[0]), float(axis[-1])),
                        expected=(expected_minimum, expected_maximum),
                    )
            if name in spec.expected_spacing and axis.size > 1:
                expected_spacing = spec.expected_spacing[name]
                if not np.allclose(
                    np.diff(axis),
                    expected_spacing,
                    atol=tolerance,
                    rtol=relative,
                ):
                    _issue(
                        issues,
                        "error",
                        "grid.expected_spacing_mismatch",
                        "Axis '%s' spacing does not match metadata." % name,
                        "grid.expected_spacing.%s" % name,
                        expected=expected_spacing,
                    )

        try:
            flat_indices = np.ravel_multi_index(tuple(indices), shape)
        except (ValueError, OverflowError) as exc:
            _issue(
                issues,
                "error",
                "grid.indexing_failed",
                "Could not map coordinates to rectilinear cells: %s" % exc,
                "grid",
            )
            return None, {}, {}, grid_statistics

        unique_cells, counts = np.unique(flat_indices, return_counts=True)
        duplicate_count = int(np.sum(counts - 1))
        missing_count = int(expected_cells - unique_cells.size)
        grid_statistics["unique_cells"] = int(unique_cells.size)
        grid_statistics["duplicate_rows"] = duplicate_count
        grid_statistics["missing_cells"] = missing_count

        if duplicate_count:
            duplicate_ids = unique_cells[counts > 1][:5]
            samples = [
                {
                    name: float(axes[axis_index][coordinate_index])
                    for axis_index, (name, coordinate_index) in enumerate(
                        zip(coordinate_names, np.unravel_index(int(cell), shape))
                    )
                }
                for cell in duplicate_ids
            ]
            _issue(
                issues,
                "error",
                "grid.duplicate_coordinates",
                "The table contains %d duplicate coordinate row(s)." % duplicate_count,
                "grid",
                count=duplicate_count,
                samples=samples,
            )

        if missing_count:
            missing_ids = self._sample_missing_cells(unique_cells, expected_cells, limit=5)
            samples = [
                {
                    name: float(axes[axis_index][coordinate_index])
                    for axis_index, (name, coordinate_index) in enumerate(
                        zip(coordinate_names, np.unravel_index(int(cell), shape))
                    )
                }
                for cell in missing_ids
            ]
            _issue(
                issues,
                "error",
                "grid.missing_cells",
                "The Cartesian product contains %d cell(s) with no table row."
                % missing_count,
                "grid",
                count=missing_count,
                samples=samples,
            )

        permutation = np.argsort(flat_indices, kind="stable")
        canonical_order = bool(np.array_equal(permutation, np.arange(row_count)))
        was_reordered = not canonical_order
        grid_statistics["input_was_canonical_order"] = canonical_order
        grid_statistics["was_reordered"] = was_reordered
        if was_reordered and not spec.reorder:
            non_increasing = np.flatnonzero(np.diff(flat_indices) <= 0)
            offending = int(non_increasing[0]) if non_increasing.size else 0
            first_row = {
                name: float(fields[name][offending]) for name in coordinate_names
            }
            second_row = {
                name: float(fields[name][offending + 1])
                for name in coordinate_names
            }
            changed = next(
                (
                    name
                    for name in coordinate_names
                    if second_row[name] < first_row[name]
                ),
                coordinate_names[-1],
            )
            _issue(
                issues,
                "error",
                "grid.row_order",
                "Coordinate '%s' is not in canonical increasing table order at "
                "rows %d -> %d: %s[%d]=%s, %s[%d]=%s. Reorder the source "
                "rows or set grid.reorder=true in the metadata."
                % (
                    changed,
                    offending,
                    offending + 1,
                    changed,
                    offending,
                    first_row[changed],
                    changed,
                    offending + 1,
                    second_row[changed],
                ),
                "grid.reorder",
                coordinate=changed,
                row_indices=[offending, offending + 1],
                values=[first_row[changed], second_row[changed]],
                first_coordinates=first_row,
                second_coordinates=second_row,
            )
        elif was_reordered:
            _issue(
                issues,
                "info",
                "grid.rows_reordered",
                "Rows were reordered lexicographically in canonical coordinate order.",
                "grid",
            )

        can_build = (
            duplicate_count == 0
            and missing_count == 0
            and expected_cells == row_count
            and (canonical_order or spec.reorder)
        )
        if not can_build:
            return None, {}, {}, grid_statistics

        immutable_permutation = _immutable_array(permutation)
        ordered_fields: Dict[str, np.ndarray] = {}
        shaped_fields: Dict[str, np.ndarray] = {}
        for name, values in fields.items():
            if name in coordinate_names:
                coordinate_index = coordinate_names.index(name)
                values_to_order = axes[coordinate_index][indices[coordinate_index]]
            else:
                values_to_order = values
            if canonical_order and (
                values_to_order is values or np.array_equal(values_to_order, values)
            ):
                ordered = values
            else:
                ordered = _immutable_array(values_to_order[permutation])
            shaped = ordered.reshape(shape)
            ordered_fields[name] = ordered
            shaped_fields[name] = shaped

        grid = RectilinearGrid(
            coordinate_names=coordinate_names,
            axes=tuple(axes),
            shape=shape,
            uniform_by_axis=tuple(uniform_flags),
            row_permutation=immutable_permutation,
            was_reordered=was_reordered,
            expected_cells=expected_cells,
            duplicate_count=duplicate_count,
            missing_count=missing_count,
        )
        return grid, ordered_fields, shaped_fields, grid_statistics

    @staticmethod
    def _sample_missing_cells(
        occupied: np.ndarray, expected_cells: int, limit: int
    ) -> List[int]:
        result: List[int] = []
        previous = -1
        for cell in occupied:
            numeric = int(cell)
            candidate = previous + 1
            while candidate < numeric and len(result) < limit:
                result.append(candidate)
                candidate += 1
            if len(result) >= limit:
                return result
            previous = numeric
        candidate = previous + 1
        while candidate < expected_cells and len(result) < limit:
            result.append(candidate)
            candidate += 1
        return result


__all__ = [
    "BoundaryComputationError",
    "CoexistenceColumnSpec",
    "CoexistenceDataSpec",
    "CoexistenceTable",
    "DataLoadError",
    "ForwardTable",
    "EoSError",
    "TableSchema",
    "EoSNotValidatedError",
    "EoSStateError",
    "EoSValidationError",
    "FieldSpec",
    "FirstOrderTransitionSpec",
    "FormatSpec",
    "GridSpec",
    "HBARC_GEV_FM",
    "METADATA_SCHEMA_VERSION",
    "MAX_TABLE_COLUMNS",
    "MetadataError",
    "RectilinearGrid",
    "TABLE_SNAPSHOT_FINGERPRINT_VERSION",
    "BoundaryEnvelope",
    "TransformSpec",
    "ValidationIssue",
    "ValidationReport",
]
