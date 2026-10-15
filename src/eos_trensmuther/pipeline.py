"""Contracts, storage, and CPU orchestration for complete-table inversion.

The planning and result contracts were introduced in Phase 1 of
``docs/PIPELINE.md``.  :class:`EoSInversionPipeline` supplies the Phase-4
scientific executor and the Phase-7 dynamic process scheduler: conservative
reachability screening, staged FAST/FALLBACK root solving, explicitly
labelled nearest-state projection, and final auxiliary/tilde evaluation run
through one identical chunk engine in serial and process modes.

Requested grid coordinates remain separable axes plus ordering metadata.
They are reconstructed chunkwise and are never overwritten by achieved
targets.  Dense status columns use explicit, versioned ``uint8`` codebooks so
large inverse tables do not pay the cost of repeated Python strings.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, fields as dataclass_fields, replace
from enum import Enum
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Type, TypeVar

import numpy as np

from .interpolation import MultilinearInterpolator
from ._scheduler import DynamicScheduleReport, execute_dynamic_tasks
from .projection import (
    EoSProjector,
    ProjectionConfig,
    ProjectionError,
    ProjectionQuality,
    ProjectionSeed,
    ProjectionSeedSource,
)
from .solver import (
    EoSRootSolver,
    InversionResult,
    ProofStatus,
    SolveStatus,
    SolveStrategy,
    SolverConfig,
    _automatic_residual_scales_from_bounds,
)
from .tables import EoSError, ForwardTable, TABLE_SNAPSHOT_FINGERPRINT_VERSION
from .transition import PhaseLabel, PhaseRootSelectionReason
from .transforms import (
    ChargeConstraintSpec,
    ConstrainedTildeTransform,
    TransformConstraintError,
    TransformDomainError,
)


PIPELINE_RESULT_SCHEMA_VERSION = 4
PIPELINE_RUN_MANIFEST_VERSION = 11
PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION = 3
DEFAULT_MAX_GRID_POINTS = 10_000_000
DEFAULT_MAX_RESULT_BYTES = 2 * 1024**3
DEFAULT_WORKER_TASK_SIZE = 8
LEGACY_PROCESS_INTERNAL_BATCH_SIZE = 256
_WORKER_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_SEED_COUNTER_NAMES = (
    "seed_direct_neighbor",
    "seed_coarse_level",
    "seed_generic",
    "seed_retry_coarse_level",
    "seed_retry_generic",
    "seed_global_nearest_node",
) + tuple(
    "seed_direct_neighbor_axis%d_%s" % (axis, direction)
    for axis in range(4)
    for direction in ("minus", "plus")
)
_SUBCELL_COUNTER_NAMES = (
    "subcell_fallback_attempted",
    "subcell_certified_no_root",
    "subcell_guarded_incomplete",
    "subcell_residual_krawczyk",
    "subcell_boxes_evaluated",
    "subcell_residual_boxes",
    "subcell_krawczyk_boxes_processed",
) + tuple("subcell_certificate_depth_%d" % depth for depth in range(1, 8))
_THREAD_LIMIT_CONFIGURATION_LOCK = threading.RLock()


def _process_peak_rss_bytes() -> int:
    """Best-effort normalized process peak RSS, or zero when unavailable."""

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return 0
    if value <= 0:
        return 0
    # Darwin reports bytes; Linux and the BSDs exposed by CPython generally
    # report KiB. Keep this platform distinction explicit in the manifest.
    return value if sys.platform == "darwin" else value * 1024


def _process_memory_snapshot() -> Dict[str, Any]:
    """Best-effort current RSS/PSS/private-resident process measurements."""

    result: Dict[str, Any] = {
        "pid": os.getpid(),
        "rss_bytes": None,
        "pss_bytes": None,
        "private_resident_bytes": None,
        "method": "unavailable",
    }
    rollup = Path("/proc/self/smaps_rollup")
    if rollup.is_file():
        values: Dict[str, int] = {}
        try:
            for line in rollup.read_text(encoding="ascii").splitlines():
                if ":" not in line:
                    continue
                name, remainder = line.split(":", 1)
                token = remainder.strip().split()
                if token:
                    values[name] = int(token[0]) * 1024
        except (OSError, ValueError):
            values = {}
        if values:
            result.update(
                {
                    "rss_bytes": values.get("Rss"),
                    "pss_bytes": values.get("Pss"),
                    "private_resident_bytes": values.get("Private_Clean", 0)
                    + values.get("Private_Dirty", 0),
                    "method": "/proc/self/smaps_rollup",
                }
            )
            return result
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(
                ["ps", "-o", "rss=", "-o", "rprvt=", "-p", str(os.getpid())],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            tokens = output.split()
            if len(tokens) >= 2:
                result.update(
                    {
                        "rss_bytes": int(tokens[0]) * 1024,
                        "private_resident_bytes": int(tokens[1]) * 1024,
                        "method": "ps_rss_rprvt_kib",
                    }
                )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        if result["private_resident_bytes"] is None:
            def parse_vmmap_size(token: str) -> Optional[int]:
                text = token.strip()
                multipliers = {
                    "K": 1024,
                    "M": 1024**2,
                    "G": 1024**3,
                    "T": 1024**4,
                }
                try:
                    if text[-1:] in multipliers:
                        return int(float(text[:-1]) * multipliers[text[-1]])
                    return int(text)
                except (ValueError, OverflowError):
                    return None

            try:
                output = subprocess.check_output(
                    ["vmmap", "-summary", str(os.getpid())],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                footprint = None
                mapped_resident = None
                total_dirty = None
                for line in output.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("Physical footprint:"):
                        token = stripped.split(":", 1)[1].strip().split()[0]
                        footprint = parse_vmmap_size(token)
                    elif stripped.startswith("mapped file"):
                        tokens = stripped.split()
                        if len(tokens) >= 4:
                            mapped_resident = parse_vmmap_size(tokens[3])
                    elif stripped.startswith("TOTAL "):
                        tokens = stripped.split()
                        if len(tokens) >= 4:
                            total_dirty = parse_vmmap_size(tokens[3])
                if footprint is not None:
                    result.update(
                        {
                            "private_resident_bytes": footprint,
                            "mapped_file_resident_bytes": mapped_resident,
                            "dirty_resident_bytes": total_dirty,
                            "method": "vmmap_summary_physical_footprint",
                        }
                    )
            except (OSError, subprocess.SubprocessError):
                pass
    return result


def _duration_statistics(values: Sequence[float]) -> Dict[str, Any]:
    """Return deterministic descriptive statistics for measured durations."""

    samples = np.asarray(tuple(values), dtype=float)
    if samples.ndim != 1 or np.any(~np.isfinite(samples)) or np.any(samples < 0.0):
        raise PipelineStateError("performance durations must be finite and non-negative")
    if samples.size == 0:
        return {
            "count": 0,
            "sum": 0.0,
            "minimum": None,
            "mean": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(samples.size),
        "sum": float(np.sum(samples, dtype=np.float64)),
        "minimum": float(np.min(samples)),
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p95": float(np.quantile(samples, 0.95)),
        "maximum": float(np.max(samples)),
    }


def _fraction(numerator: int, denominator: int) -> float:
    """Return a finite fraction, using zero for an empty population."""

    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


class PipelineError(EoSError):
    """Base exception for table-wide pipeline failures."""


class PipelineConfigurationError(PipelineError):
    """Raised when a pipeline configuration cannot define a valid plan."""


class PipelineStateError(PipelineError):
    """Raised when result storage violates the pipeline state contract."""


class GridSpace(str, Enum):
    """Coordinate system sampled by the regular inverse grid."""

    TARGET = "target"
    TILDE = "tilde"


class InitialGuessPolicy(str, Enum):
    """Generic seed used after neighbor and coarse-grid seeds are unavailable."""

    DOMAIN_MIDPOINT = "domain_midpoint"


class ErrorPolicy(str, Enum):
    """How a future executor handles unexpected worker failures."""

    FAIL_FAST = "fail_fast"
    ACCUMULATE = "accumulate"


class ProcessStartMethod(str, Enum):
    """Supported multiprocessing start methods."""

    SPAWN = "spawn"
    FORK = "fork"
    FORKSERVER = "forkserver"


class PointDisposition(str, Enum):
    """Terminal scientific disposition of one requested grid point."""

    SOLVED = "solved"
    PROJECTED = "projected"
    CERTIFIED_UNREACHABLE = "certified_unreachable"
    ROOT_CERTIFIED_UNREFINED = "root_certified_unrefined"
    UNRESOLVED = "unresolved"
    INVALID_TARGET = "invalid_target"
    PROJECTION_FAILED = "projection_failed"
    ERROR = "error"


class PointRoute(str, Enum):
    """Route that produced one terminal point disposition."""

    FAST = "fast"
    FALLBACK_CERTIFIED = "fallback_certified"
    FALLBACK_ROOT_UNREFINED = "fallback_root_unrefined"
    FALLBACK_UNRESOLVED = "fallback_unresolved"
    PROJECTED_PRESCREEN = "projected_prescreen"
    PROJECTED_CERTIFIED_NO_ROOT = "projected_certified_no_root"
    CERTIFIED_UNREACHABLE_PRESCREEN = "certified_unreachable_prescreen"
    CERTIFIED_UNREACHABLE_FALLBACK = "certified_unreachable_fallback"
    PROJECTION_FAILED_PRESCREEN = "projection_failed_prescreen"
    PROJECTION_FAILED_CERTIFIED_NO_ROOT = (
        "projection_failed_certified_no_root"
    )
    INVALID_TRANSFORM = "invalid_transform"
    ERROR = "error"


class PrescreenStatus(str, Enum):
    """Tri-state output of the future conservative reachability screen."""

    NOT_RUN = "not_run"
    CANDIDATE = "candidate"
    DEFINITELY_UNREACHABLE = "definitely_unreachable"


class InitialGuessSource(str, Enum):
    """Recorded provenance for a point solver's starting value."""

    NEAREST_NODE = "nearest_node"
    DOMAIN_MIDPOINT = "domain_midpoint"
    PROVIDED = "provided"
    CONTAINING_CELL = "containing_cell"
    FAST_BEST_CANDIDATE = "fast_best_candidate"
    COARSE_LEVEL = "coarse_level"
    DIRECT_NEIGHBOR = "direct_neighbor"


class PhaseRoute(str, Enum):
    """Physical map used to interpret the selected numerical root."""

    NO_TRANSITION = "no_transition"
    ORDINARY = "ordinary"
    LINE_REGION_S = "line_region_s"
    BINODALS_ORDINARY = "binodals_ordinary"


class LineAuxiliaryPolicy(str, Enum):
    """Defined policies for an auxiliary on a reconstructed LINE state."""

    LINEAR = "linear"
    EMPTY = "empty"


# Codes are explicit and form part of result schema version 2.  Code zero is
# reserved for "unassigned" or "not run" and is never derived from enum order.
_DISPOSITION_CODES = MappingProxyType(
    {
        PointDisposition.SOLVED: 1,
        PointDisposition.PROJECTED: 2,
        PointDisposition.CERTIFIED_UNREACHABLE: 3,
        PointDisposition.ROOT_CERTIFIED_UNREFINED: 4,
        PointDisposition.UNRESOLVED: 5,
        PointDisposition.INVALID_TARGET: 6,
        PointDisposition.PROJECTION_FAILED: 7,
        PointDisposition.ERROR: 8,
    }
)
_ROUTE_CODES = MappingProxyType(
    {
        PointRoute.FAST: 1,
        PointRoute.FALLBACK_CERTIFIED: 2,
        PointRoute.FALLBACK_ROOT_UNREFINED: 3,
        PointRoute.FALLBACK_UNRESOLVED: 4,
        PointRoute.PROJECTED_PRESCREEN: 5,
        PointRoute.PROJECTED_CERTIFIED_NO_ROOT: 6,
        PointRoute.CERTIFIED_UNREACHABLE_PRESCREEN: 7,
        PointRoute.CERTIFIED_UNREACHABLE_FALLBACK: 8,
        PointRoute.PROJECTION_FAILED_PRESCREEN: 9,
        PointRoute.PROJECTION_FAILED_CERTIFIED_NO_ROOT: 10,
        PointRoute.INVALID_TRANSFORM: 11,
        PointRoute.ERROR: 12,
    }
)
_PRESCREEN_CODES = MappingProxyType(
    {
        PrescreenStatus.NOT_RUN: 0,
        PrescreenStatus.CANDIDATE: 1,
        PrescreenStatus.DEFINITELY_UNREACHABLE: 2,
    }
)
_PROJECTION_QUALITY_CODES = MappingProxyType(
    {
        ProjectionQuality.OPTIMIZER_CONVERGED: 1,
        ProjectionQuality.BEST_FEASIBLE_ENCOUNTERED: 2,
        ProjectionQuality.DISCRETE_NODE_FALLBACK: 3,
        ProjectionQuality.GLOBAL_CERTIFIED: 4,
    }
)
_INITIAL_GUESS_CODES = MappingProxyType(
    {
        InitialGuessSource.NEAREST_NODE: 1,
        InitialGuessSource.DOMAIN_MIDPOINT: 2,
        InitialGuessSource.PROVIDED: 3,
        InitialGuessSource.CONTAINING_CELL: 4,
        InitialGuessSource.FAST_BEST_CANDIDATE: 5,
        InitialGuessSource.COARSE_LEVEL: 6,
        InitialGuessSource.DIRECT_NEIGHBOR: 7,
    }
)
_SOLVE_STATUS_CODES = MappingProxyType(
    {
        SolveStatus.CONVERGED_FAST: 1,
        SolveStatus.CONVERGED_FALLBACK: 2,
        SolveStatus.CERTIFIED_ROOT_UNREFINED: 3,
        SolveStatus.CERTIFIED_NO_ROOT: 4,
        SolveStatus.UNRESOLVED: 5,
        SolveStatus.FAST_PATH_FAILED: 6,
    }
)
_PROOF_STATUS_CODES = MappingProxyType(
    {
        ProofStatus.NONE: 1,
        ProofStatus.UNIQUE_IN_BOX: 2,
        ProofStatus.NO_ROOT_IN_DOMAIN: 3,
        ProofStatus.INCONCLUSIVE: 4,
    }
)
_PHASE_CODES = MappingProxyType(
    {
        PhaseLabel.NOT_APPLICABLE: 1,
        PhaseLabel.SINGLE_PHASE: 2,
        PhaseLabel.MIXED_PHASE: 3,
    }
)
_PHASE_ROUTE_CODES = MappingProxyType(
    {
        PhaseRoute.NO_TRANSITION: 1,
        PhaseRoute.ORDINARY: 2,
        PhaseRoute.LINE_REGION_S: 3,
        PhaseRoute.BINODALS_ORDINARY: 4,
    }
)
_PHASE_SELECTION_REASON_CODES = MappingProxyType(
    {
        PhaseRootSelectionReason.NOT_APPLICABLE: 1,
        PhaseRootSelectionReason.UNAMBIGUOUS: 2,
        PhaseRootSelectionReason.SELECTED_FROM_JACOBIAN: 3,
    }
)


_E = TypeVar("_E", bound=Enum)


def _coerce_enum(value: Any, enum_type: Type[_E], name: str) -> _E:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise PipelineConfigurationError(
            "%s must be one of: %s" % (name, choices)
        ) from exc


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise PipelineConfigurationError("%s must be an integer" % name)
    converted = int(value)
    if converted < minimum:
        raise PipelineConfigurationError(
            "%s must be at least %d" % (name, minimum)
        )
    return converted


def _nonnegative_integer(value: Any, name: str) -> int:
    return _positive_integer(value, name, minimum=0)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise PipelineConfigurationError("%s must be a finite number" % name)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PipelineConfigurationError(
            "%s must be a finite number" % name
        ) from exc
    if not math.isfinite(converted):
        raise PipelineConfigurationError("%s must be a finite number" % name)
    return converted


def _positive_scale(value: Any, name: str) -> float:
    converted = _finite_float(value, name)
    if converted <= 0.0:
        raise PipelineConfigurationError("%s must be positive" % name)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        reciprocal = np.float64(1.0) / np.float64(converted)
    if not np.isfinite(reciprocal):
        raise PipelineConfigurationError(
            "%s is too small to define a finite reciprocal" % name
        )
    return converted


def _immutable_array(value: Any, dtype: Optional[Any] = None) -> np.ndarray:
    contiguous = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return immutable.reshape(contiguous.shape)


def _validated_table_fingerprint(table: ForwardTable) -> str:
    """Return the public stable fingerprint for a validated table snapshot."""

    return table.snapshot_fingerprint()


def _freeze_mapping_of_mappings(
    value: Mapping[str, Mapping[str, float]]
) -> Mapping[str, Mapping[str, float]]:
    return MappingProxyType(
        {
            str(name): MappingProxyType(
                {str(source): float(coefficient) for source, coefficient in relation.items()}
            )
            for name, relation in value.items()
        }
    )


def _freeze_manifest(value: Any, path: str = "manifest") -> Any:
    """Return an immutable, JSON-compatible copy of manifest data."""

    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PipelineStateError("%s keys must be strings" % path)
            result[key] = _freeze_manifest(item, "%s.%s" % (path, key))
        return MappingProxyType(result)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_manifest(item, "%s[]" % path) for item in value
        )
    if isinstance(value, np.generic):
        return _freeze_manifest(value.item(), path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PipelineStateError("%s contains a non-finite number" % path)
        return value
    raise PipelineStateError(
        "%s contains a non-JSON-compatible value of type %s"
        % (path, type(value).__name__)
    )


def _enum_code(enum_value: Enum, codebook: Mapping[Enum, int]) -> int:
    return int(codebook[enum_value])


def pipeline_status_code(value: Enum) -> int:
    """Return the current dense-result storage code for one status enum.

    ``SolveStatus`` and ``ProofStatus`` keep their solver-level string values;
    this helper maps them, and the pipeline enums, into compact result columns.
    Code zero remains reserved for an unassigned/not-run stage and therefore
    has no value for enums that do not explicitly define ``NOT_RUN``.
    """

    typed_codebooks: Tuple[Tuple[Type[Enum], Mapping[Enum, int]], ...] = (
        (PointDisposition, _DISPOSITION_CODES),
        (PointRoute, _ROUTE_CODES),
        (PrescreenStatus, _PRESCREEN_CODES),
        (ProjectionQuality, _PROJECTION_QUALITY_CODES),
        (InitialGuessSource, _INITIAL_GUESS_CODES),
        (SolveStatus, _SOLVE_STATUS_CODES),
        (ProofStatus, _PROOF_STATUS_CODES),
        (PhaseLabel, _PHASE_CODES),
        (PhaseRoute, _PHASE_ROUTE_CODES),
        (PhaseRootSelectionReason, _PHASE_SELECTION_REASON_CODES),
    )
    for enum_type, codebook in typed_codebooks:
        if isinstance(value, enum_type):
            return int(codebook[value])
    raise TypeError("unsupported pipeline status enum: %r" % (value,))


def _codebook_dict(
    codebook: Mapping[Enum, int], *, zero_label: str
) -> Dict[str, str]:
    result = {"0": zero_label}
    for member, code in codebook.items():
        result[str(code)] = str(member.value)
    return result


def _result_codebooks() -> Dict[str, Dict[str, str]]:
    return {
        "disposition": _codebook_dict(
            _DISPOSITION_CODES, zero_label="unassigned"
        ),
        "route": _codebook_dict(_ROUTE_CODES, zero_label="unassigned"),
        "prescreen": _codebook_dict(_PRESCREEN_CODES, zero_label="not_run"),
        "solve_status": _codebook_dict(
            _SOLVE_STATUS_CODES, zero_label="not_run"
        ),
        "proof_status": _codebook_dict(
            _PROOF_STATUS_CODES, zero_label="not_run"
        ),
        "projection_quality": _codebook_dict(
            _PROJECTION_QUALITY_CODES, zero_label="not_run"
        ),
        "initial_guess_source": _codebook_dict(
            _INITIAL_GUESS_CODES, zero_label="not_run"
        ),
        "phase": _codebook_dict(_PHASE_CODES, zero_label="unassigned"),
        "phase_route": _codebook_dict(
            _PHASE_ROUTE_CODES, zero_label="not_evaluated"
        ),
        "phase_selection_reason": _codebook_dict(
            _PHASE_SELECTION_REASON_CODES, zero_label="not_evaluated"
        ),
    }


@dataclass(frozen=True)
class GridAxisSpec:
    """Point count and optional bound overrides for one semantic grid axis."""

    count: int
    lower: Optional[float] = None
    upper: Optional[float] = None

    def __post_init__(self) -> None:
        count = _positive_integer(self.count, "axis count", minimum=2)
        lower = (
            None if self.lower is None else _finite_float(self.lower, "axis lower bound")
        )
        upper = (
            None if self.upper is None else _finite_float(self.upper, "axis upper bound")
        )
        if lower is not None and upper is not None and lower >= upper:
            raise PipelineConfigurationError(
                "an axis lower bound must be smaller than its upper bound"
            )
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class InversionGridSpec:
    """Unresolved named regular-grid request."""

    space: GridSpace
    axes: Mapping[str, GridAxisSpec]
    max_points: int = DEFAULT_MAX_GRID_POINTS
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES
    max_peak_result_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        space = _coerce_enum(self.space, GridSpace, "grid space")
        if not isinstance(self.axes, Mapping) or not self.axes:
            raise PipelineConfigurationError(
                "grid axes must be a non-empty mapping keyed by semantic name"
            )
        normalized: Dict[str, GridAxisSpec] = {}
        for name, specification in self.axes.items():
            if not isinstance(name, str) or not name:
                raise PipelineConfigurationError(
                    "grid axis names must be non-empty strings"
                )
            if not isinstance(specification, GridAxisSpec):
                raise PipelineConfigurationError(
                    "grid axis '%s' must use GridAxisSpec" % name
                )
            normalized[name] = specification
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "axes", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "max_points",
            _positive_integer(self.max_points, "max_points"),
        )
        max_result_bytes = _positive_integer(
            self.max_result_bytes, "max_result_bytes"
        )
        object.__setattr__(self, "max_result_bytes", max_result_bytes)
        peak_limit = self.max_peak_result_bytes
        if peak_limit is None:
            peak_limit = 2 * max_result_bytes
        object.__setattr__(
            self,
            "max_peak_result_bytes",
            _positive_integer(peak_limit, "max_peak_result_bytes"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "space": self.space.value,
            "axes": {
                name: {
                    "count": axis.count,
                    "lower": axis.lower,
                    "upper": axis.upper,
                }
                for name, axis in self.axes.items()
            },
            "max_points": self.max_points,
            "max_result_bytes": self.max_result_bytes,
            "max_peak_result_bytes": self.max_peak_result_bytes,
        }


@dataclass(frozen=True)
class ExecutionConfig:
    """Atomic checkpoint and process-task settings for table-wide execution."""

    workers: int = 1
    start_method: ProcessStartMethod = ProcessStartMethod.SPAWN
    chunk_size: int = 256
    worker_task_size: int = DEFAULT_WORKER_TASK_SIZE
    max_pending_chunks: Optional[int] = None
    error_policy: ErrorPolicy = ErrorPolicy.FAIL_FAST

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "workers", _positive_integer(self.workers, "workers")
        )
        object.__setattr__(
            self,
            "start_method",
            _coerce_enum(
                self.start_method, ProcessStartMethod, "process start method"
            ),
        )
        object.__setattr__(
            self,
            "chunk_size",
            _positive_integer(self.chunk_size, "chunk_size"),
        )
        object.__setattr__(
            self,
            "worker_task_size",
            _positive_integer(self.worker_task_size, "worker_task_size"),
        )
        if self.max_pending_chunks is not None:
            object.__setattr__(
                self,
                "max_pending_chunks",
                _positive_integer(
                    self.max_pending_chunks, "max_pending_chunks"
                ),
            )
        object.__setattr__(
            self,
            "error_policy",
            _coerce_enum(self.error_policy, ErrorPolicy, "error policy"),
        )


@dataclass(frozen=True)
class CoarseSeedConfig:
    """Deterministic reduced inverse-grid seeds for smooth ordinary EoSs."""

    enabled: bool = True
    points_per_axis: int = 3
    maximum_points: int = 4096

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise PipelineConfigurationError("coarse seed enabled must be boolean")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(
            self,
            "points_per_axis",
            _positive_integer(
                self.points_per_axis, "coarse seed points_per_axis", minimum=2
            ),
        )
        object.__setattr__(
            self,
            "maximum_points",
            _positive_integer(
                self.maximum_points, "coarse seed maximum_points"
            ),
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable unresolved configuration for one inverse-table run."""

    grid: InversionGridSpec
    residual_scales: Optional[Mapping[str, float]] = None
    solver: SolverConfig = field(default_factory=SolverConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    coarse_seed: CoarseSeedConfig = field(default_factory=CoarseSeedConfig)
    initial_guess: InitialGuessPolicy = InitialGuessPolicy.DOMAIN_MIDPOINT
    use_reachability_seed: bool = True
    auxiliary_fields: Optional[Tuple[str, ...]] = None
    line_auxiliary_default_policy: LineAuxiliaryPolicy = (
        LineAuxiliaryPolicy.LINEAR
    )
    line_auxiliary_policies: Mapping[str, LineAuxiliaryPolicy] = field(
        default_factory=dict
    )
    retain_detailed_diagnostics: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.grid, InversionGridSpec):
            raise PipelineConfigurationError("grid must be an InversionGridSpec")
        if self.residual_scales is None:
            scales = None
        else:
            if (
                not isinstance(self.residual_scales, Mapping)
                or not self.residual_scales
            ):
                raise PipelineConfigurationError(
                    "residual_scales must be None or a non-empty mapping by target name"
                )
            parsed_scales: Dict[str, float] = {}
            for name, value in self.residual_scales.items():
                if not isinstance(name, str) or not name:
                    raise PipelineConfigurationError(
                        "residual scale names must be non-empty strings"
                    )
                parsed_scales[name] = _positive_scale(
                    value, "residual scale for %s" % name
                )
            scales = MappingProxyType(parsed_scales)
        if not isinstance(self.solver, SolverConfig):
            raise PipelineConfigurationError("solver must be a SolverConfig")
        if not isinstance(self.projection, ProjectionConfig):
            raise PipelineConfigurationError(
                "projection must be a ProjectionConfig"
            )
        if not isinstance(self.execution, ExecutionConfig):
            raise PipelineConfigurationError(
                "execution must be an ExecutionConfig"
            )
        if not isinstance(self.coarse_seed, CoarseSeedConfig):
            raise PipelineConfigurationError(
                "coarse_seed must be a CoarseSeedConfig"
            )
        initial_guess = _coerce_enum(
            self.initial_guess, InitialGuessPolicy, "initial guess policy"
        )
        if not isinstance(self.use_reachability_seed, (bool, np.bool_)):
            raise PipelineConfigurationError(
                "use_reachability_seed must be boolean"
            )
        auxiliary_fields = self.auxiliary_fields
        if auxiliary_fields is not None:
            if isinstance(auxiliary_fields, str) or not isinstance(
                auxiliary_fields, Sequence
            ):
                raise PipelineConfigurationError(
                    "auxiliary_fields must be a sequence of names or None"
                )
            auxiliary_fields = tuple(auxiliary_fields)
            if not all(
                isinstance(name, str) and name for name in auxiliary_fields
            ):
                raise PipelineConfigurationError(
                    "auxiliary field names must be non-empty strings"
                )
            if len(set(auxiliary_fields)) != len(auxiliary_fields):
                raise PipelineConfigurationError(
                    "auxiliary_fields cannot contain duplicates"
                )
        if not isinstance(self.retain_detailed_diagnostics, bool):
            raise PipelineConfigurationError(
                "retain_detailed_diagnostics must be boolean"
            )
        default_policy = _coerce_enum(
            self.line_auxiliary_default_policy,
            LineAuxiliaryPolicy,
            "line_auxiliary_default_policy",
        )
        if not isinstance(self.line_auxiliary_policies, Mapping):
            raise PipelineConfigurationError(
                "line_auxiliary_policies must be a mapping by field name"
            )
        line_policies: Dict[str, LineAuxiliaryPolicy] = {}
        for name, policy in self.line_auxiliary_policies.items():
            if not isinstance(name, str) or not name:
                raise PipelineConfigurationError(
                    "LINE auxiliary policy names must be non-empty strings"
                )
            line_policies[name] = _coerce_enum(
                policy,
                LineAuxiliaryPolicy,
                "LINE auxiliary policy for %s" % name,
            )
        object.__setattr__(self, "residual_scales", scales)
        object.__setattr__(self, "initial_guess", initial_guess)
        object.__setattr__(
            self, "use_reachability_seed", bool(self.use_reachability_seed)
        )
        object.__setattr__(self, "auxiliary_fields", auxiliary_fields)
        object.__setattr__(self, "line_auxiliary_default_policy", default_policy)
        object.__setattr__(
            self,
            "line_auxiliary_policies",
            MappingProxyType(line_policies),
        )

    def to_dict(self) -> Dict[str, Any]:
        solver_values = {
            item.name: getattr(self.solver, item.name)
            for item in dataclass_fields(self.solver)
        }
        return {
            "numerical_routing_contract_version": (
                PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION
            ),
            "grid": self.grid.to_dict(),
            "residual_scales": (
                None
                if self.residual_scales is None
                else dict(self.residual_scales)
            ),
            "solver": solver_values,
            "projection": {
                item.name: getattr(self.projection, item.name)
                for item in dataclass_fields(self.projection)
            },
            "execution": {
                "workers": self.execution.workers,
                "start_method": self.execution.start_method.value,
                "chunk_size": self.execution.chunk_size,
                "worker_task_size": self.execution.worker_task_size,
                "max_pending_chunks": self.execution.max_pending_chunks,
                "error_policy": self.execution.error_policy.value,
            },
            "coarse_seed": {
                item.name: getattr(self.coarse_seed, item.name)
                for item in dataclass_fields(self.coarse_seed)
            },
            "initial_guess": self.initial_guess.value,
            "use_reachability_seed": self.use_reachability_seed,
            "auxiliary_fields": (
                None
                if self.auxiliary_fields is None
                else list(self.auxiliary_fields)
            ),
            "line_auxiliary_default_policy": (
                self.line_auxiliary_default_policy.value
            ),
            "line_auxiliary_policies": {
                name: policy.value
                for name, policy in self.line_auxiliary_policies.items()
            },
            "retain_detailed_diagnostics": self.retain_detailed_diagnostics,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineConfig":
        """Reconstruct validated configuration from its JSON representation."""

        if not isinstance(value, Mapping):
            raise PipelineConfigurationError(
                "pipeline configuration must be a mapping"
            )
        routing_version = value.get("numerical_routing_contract_version")
        if routing_version is not None and (
            isinstance(routing_version, (bool, np.bool_))
            or not isinstance(routing_version, (int, np.integer))
            or int(routing_version)
            != PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION
        ):
            raise PipelineConfigurationError(
                "unsupported numerical routing contract version %r"
                % routing_version
            )
        try:
            grid_value = value["grid"]
            axes = {
                name: GridAxisSpec(
                    count=specification["count"],
                    lower=specification.get("lower"),
                    upper=specification.get("upper"),
                )
                for name, specification in grid_value["axes"].items()
            }
            grid = InversionGridSpec(
                space=grid_value["space"],
                axes=axes,
                max_points=grid_value["max_points"],
                max_result_bytes=grid_value["max_result_bytes"],
                max_peak_result_bytes=grid_value.get(
                    "max_peak_result_bytes"
                ),
            )
            execution_value = value["execution"]
            execution = ExecutionConfig(
                workers=execution_value["workers"],
                start_method=execution_value["start_method"],
                chunk_size=execution_value["chunk_size"],
                worker_task_size=execution_value.get(
                    "worker_task_size", LEGACY_PROCESS_INTERNAL_BATCH_SIZE
                ),
                max_pending_chunks=execution_value.get("max_pending_chunks"),
                error_policy=execution_value["error_policy"],
            )
            solver_value = dict(value["solver"])
            subcell_defaults = {
                "enable_subcell_fallback": False,
                "max_subcell_depth": 7,
                "max_subcell_active_boxes": 1_000_000,
            }
            present_subcell_fields = {
                name for name in subcell_defaults if name in solver_value
            }
            if not present_subcell_fields:
                # Configurations serialized before the subcell fields existed
                # describe the historical complete-original-cell route.  Do
                # not reinterpret those artifacts using the newer default.
                solver_value.update(subcell_defaults)
            elif present_subcell_fields != set(subcell_defaults):
                raise PipelineConfigurationError(
                    "serialized solver subcell configuration must contain "
                    "all three subcell fields or none of them"
                )
            auxiliary = value.get("auxiliary_fields")
            return cls(
                grid=grid,
                residual_scales=value["residual_scales"],
                solver=SolverConfig(**solver_value),
                projection=ProjectionConfig(**value["projection"]),
                execution=execution,
                coarse_seed=CoarseSeedConfig(**value.get("coarse_seed", {})),
                initial_guess=value["initial_guess"],
                use_reachability_seed=value.get(
                    "use_reachability_seed", True
                ),
                auxiliary_fields=(
                    None if auxiliary is None else tuple(auxiliary)
                ),
                line_auxiliary_default_policy=value.get(
                    "line_auxiliary_default_policy",
                    LineAuxiliaryPolicy.LINEAR.value,
                ),
                line_auxiliary_policies=value.get(
                    "line_auxiliary_policies", {}
                ),
                retain_detailed_diagnostics=value.get(
                    "retain_detailed_diagnostics", False
                ),
            )
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineConfigurationError(
                "invalid serialized pipeline configuration"
            ) from exc


@dataclass(frozen=True)
class GridChunk:
    """One contiguous half-open flat-index range from an inversion grid."""

    start: int
    stop: int
    flat_indices: np.ndarray
    requested_values: np.ndarray
    canonical_targets: np.ndarray
    transform_valid: np.ndarray
    requested_dependent_tilde: np.ndarray

    def __post_init__(self) -> None:
        start = _nonnegative_integer(self.start, "chunk start")
        stop = _nonnegative_integer(self.stop, "chunk stop")
        if stop < start:
            raise PipelineConfigurationError("chunk stop cannot precede start")
        count = stop - start
        indices = np.asarray(self.flat_indices)
        requested = np.asarray(self.requested_values, dtype=float)
        targets = np.asarray(self.canonical_targets, dtype=float)
        valid = np.asarray(self.transform_valid, dtype=bool)
        dependent = np.asarray(self.requested_dependent_tilde, dtype=float)
        if indices.shape != (count,) or not np.issubdtype(indices.dtype, np.integer):
            raise PipelineStateError("chunk flat indices have an invalid shape")
        if requested.ndim != 2 or requested.shape[0] != count:
            raise PipelineStateError("chunk requested values must be two-dimensional")
        if targets.shape != requested.shape:
            raise PipelineStateError(
                "chunk canonical targets must match requested-value dimension"
            )
        if valid.shape != (count,):
            raise PipelineStateError("chunk transform_valid has an invalid shape")
        if dependent.ndim != 2 or dependent.shape[0] != count:
            raise PipelineStateError(
                "chunk dependent tilde values must be two-dimensional"
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(
            self, "flat_indices", _immutable_array(indices, dtype=np.int64)
        )
        object.__setattr__(self, "requested_values", _immutable_array(requested))
        object.__setattr__(self, "canonical_targets", _immutable_array(targets))
        object.__setattr__(self, "transform_valid", _immutable_array(valid))
        object.__setattr__(
            self, "requested_dependent_tilde", _immutable_array(dependent)
        )


@dataclass(frozen=True)
class InversionGrid:
    """Resolved regular grid with lazy C-order chunk generation."""

    space: GridSpace
    axis_names: Tuple[str, ...]
    axis_units: Tuple[str, ...]
    axes: Tuple[np.ndarray, ...]
    canonical_target_names: Tuple[str, ...]
    canonical_target_units: Tuple[str, ...]
    constraints: ChargeConstraintSpec = field(repr=False, compare=False)
    sampling_envelope_lower: np.ndarray = field(repr=False)
    sampling_envelope_upper: np.ndarray = field(repr=False)
    coverage: str = "validated_grid_nodes"
    transform_id: Optional[str] = None
    dependent_tilde_names: Tuple[str, ...] = ()
    dependent_tilde_relations: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    ordering: str = "C_last_axis_fastest"
    shape: Tuple[int, ...] = field(init=False)
    size: int = field(init=False)

    def __post_init__(self) -> None:
        space = _coerce_enum(self.space, GridSpace, "grid space")
        names = tuple(self.axis_names)
        units = tuple(self.axis_units)
        target_names = tuple(self.canonical_target_names)
        target_units = tuple(self.canonical_target_units)
        dimension = len(names)
        if dimension not in range(1, 5):
            raise PipelineStateError("grid dimension must be between one and four")
        if len(set(names)) != dimension or not all(names):
            raise PipelineStateError("grid axis names must be unique and non-empty")
        if len(units) != dimension or len(self.axes) != dimension:
            raise PipelineStateError("grid axis metadata does not match dimension")
        if len(target_names) != dimension or len(target_units) != dimension:
            raise PipelineStateError(
                "canonical target metadata does not match grid dimension"
            )
        if not isinstance(self.constraints, ChargeConstraintSpec):
            raise PipelineStateError("grid constraints are invalid")
        if self.constraints.dimension != dimension:
            raise PipelineStateError("grid constraints do not match dimension")
        immutable_axes = []
        for name, axis in zip(names, self.axes):
            values = np.asarray(axis, dtype=float)
            if values.ndim != 1 or values.size < 2:
                raise PipelineStateError(
                    "grid axis '%s' must contain at least two values" % name
                )
            if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
                raise PipelineStateError(
                    "grid axis '%s' must be finite and strictly increasing" % name
                )
            immutable_axes.append(_immutable_array(values))
        shape = tuple(int(axis.size) for axis in immutable_axes)
        size = math.prod(shape)
        lower = np.asarray(self.sampling_envelope_lower, dtype=float)
        upper = np.asarray(self.sampling_envelope_upper, dtype=float)
        if lower.shape != (dimension,) or upper.shape != (dimension,):
            raise PipelineStateError("sampling envelope does not match grid dimension")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise PipelineStateError("sampling envelope must be finite")
        if np.any(lower >= upper):
            raise PipelineStateError("sampling envelope axes must be non-degenerate")
        if self.ordering != "C_last_axis_fastest":
            raise PipelineStateError("unsupported grid ordering")
        dependent_names = tuple(self.dependent_tilde_names)
        relations = _freeze_mapping_of_mappings(self.dependent_tilde_relations)
        if set(relations) != set(dependent_names):
            raise PipelineStateError(
                "dependent tilde relations must match dependent names"
            )
        if space is GridSpace.TARGET:
            if self.transform_id is not None or dependent_names:
                raise PipelineStateError(
                    "target-space grids cannot declare requested tilde outputs"
                )
        else:
            if names != self.constraints.independent_tilde_names:
                raise PipelineStateError(
                    "tilde grid axes do not match the constraint reduction"
                )
            if dependent_names != self.constraints.dependent_tilde_names:
                raise PipelineStateError(
                    "dependent tilde names do not match the constraint reduction"
                )
            if self.transform_id != self.constraints.identifier:
                raise PipelineStateError("tilde transform identifier is inconsistent")
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "axis_names", names)
        object.__setattr__(self, "axis_units", units)
        object.__setattr__(self, "axes", tuple(immutable_axes))
        object.__setattr__(self, "canonical_target_names", target_names)
        object.__setattr__(self, "canonical_target_units", target_units)
        object.__setattr__(self, "sampling_envelope_lower", _immutable_array(lower))
        object.__setattr__(self, "sampling_envelope_upper", _immutable_array(upper))
        object.__setattr__(self, "dependent_tilde_names", dependent_names)
        object.__setattr__(self, "dependent_tilde_relations", relations)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "size", size)

    @property
    def dimension(self) -> int:
        return len(self.axis_names)

    @classmethod
    def from_table(
        cls, table: ForwardTable, specification: InversionGridSpec
    ) -> "InversionGrid":
        """Resolve a grid specification against one validated table."""

        if not isinstance(table, ForwardTable):
            raise TypeError("table must be a ForwardTable")
        if not isinstance(specification, InversionGridSpec):
            raise TypeError("specification must be an InversionGridSpec")
        table.require_validated()
        metadata = table.metadata
        assert metadata is not None
        envelope = table.find_boundaries(
            space=specification.space.value, require_non_degenerate=True
        )
        expected_names = tuple(envelope.names)
        actual_names = set(specification.axes)
        expected_set = set(expected_names)
        if actual_names != expected_set:
            missing = sorted(expected_set - actual_names)
            extra = sorted(actual_names - expected_set)
            details = []
            if missing:
                details.append("missing %s" % ", ".join(missing))
            if extra:
                details.append("unexpected %s" % ", ".join(extra))
            raise PipelineConfigurationError(
                "grid axes must exactly match %s (%s)"
                % (", ".join(expected_names), "; ".join(details))
            )

        # Check the product incrementally with Python integers before linspace
        # allocates any axis.  This cannot overflow like a fixed-width np.prod.
        point_count = 1
        for name in expected_names:
            count = specification.axes[name].count
            if point_count > specification.max_points // count:
                raise PipelineConfigurationError(
                    "grid has more than max_points=%d" % specification.max_points
                )
            point_count *= count

        resolved_bounds = []
        for index, name in enumerate(expected_names):
            axis_spec = specification.axes[name]
            envelope_lower = float(envelope.lower[index])
            envelope_upper = float(envelope.upper[index])
            lower = envelope_lower if axis_spec.lower is None else axis_spec.lower
            upper = envelope_upper if axis_spec.upper is None else axis_spec.upper
            assert lower is not None and upper is not None
            if lower < envelope_lower or upper > envelope_upper:
                raise PipelineConfigurationError(
                    "grid bounds for '%s' must remain inside [%r, %r]"
                    % (name, envelope_lower, envelope_upper)
                )
            if lower >= upper:
                raise PipelineConfigurationError(
                    "resolved grid bounds for '%s' are degenerate" % name
                )
            resolved_bounds.append((float(lower), float(upper)))

        axes = []
        for name, bounds in zip(expected_names, resolved_bounds):
            axis = np.linspace(
                bounds[0],
                bounds[1],
                specification.axes[name].count,
                endpoint=True,
                dtype=float,
            )
            if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
                raise PipelineConfigurationError(
                    "grid axis '%s' cannot be represented as a finite, "
                    "strictly increasing binary64 linspace" % name
                )
            axes.append(axis)

        target_names = metadata.target_names
        target_units = tuple(
            metadata.fields[name].canonical_unit for name in target_names
        )
        if specification.space is GridSpace.TILDE:
            dependent_names = envelope.dependent_names
            dependent_relations = envelope.dependent_relations
        else:
            dependent_names = ()
            dependent_relations = {}
        return cls(
            space=specification.space,
            axis_names=expected_names,
            axis_units=tuple(envelope.units),
            axes=tuple(axes),
            canonical_target_names=target_names,
            canonical_target_units=target_units,
            constraints=table.charge_constraints,
            sampling_envelope_lower=envelope.lower,
            sampling_envelope_upper=envelope.upper,
            coverage=envelope.coverage,
            transform_id=envelope.transform_id,
            dependent_tilde_names=dependent_names,
            dependent_tilde_relations=dependent_relations,
        )

    def chunk(self, start: int, stop: int) -> GridChunk:
        """Return one lazy half-open flat-index chunk in C order."""

        checked_start = _nonnegative_integer(start, "chunk start")
        checked_stop = _nonnegative_integer(stop, "chunk stop")
        if checked_start > checked_stop or checked_stop > self.size:
            raise PipelineConfigurationError(
                "chunk range must satisfy 0 <= start <= stop <= %d" % self.size
            )
        flat_indices = np.arange(checked_start, checked_stop, dtype=np.int64)
        if flat_indices.size:
            multi_indices = np.unravel_index(
                flat_indices, self.shape, order="C"
            )
            requested = np.column_stack(
                [axis[index] for axis, index in zip(self.axes, multi_indices)]
            )
        else:
            requested = np.empty((0, self.dimension), dtype=float)

        if self.space is GridSpace.TARGET:
            canonical = requested.copy()
            transform_valid = np.ones(requested.shape[0], dtype=bool)
            dependent = np.empty((requested.shape[0], 0), dtype=float)
        else:
            canonical, transform_valid, dependent = self._transform_tilde(requested)
        return GridChunk(
            start=checked_start,
            stop=checked_stop,
            flat_indices=flat_indices,
            requested_values=requested,
            canonical_targets=canonical,
            transform_valid=transform_valid,
            requested_dependent_tilde=dependent,
        )

    def iter_chunks(self, chunk_size: int) -> Iterator[GridChunk]:
        """Yield contiguous chunks covering every flat index exactly once."""

        checked_size = _positive_integer(chunk_size, "chunk_size")
        for start in range(0, self.size, checked_size):
            yield self.chunk(start, min(self.size, start + checked_size))

    def _transform_tilde(
        self, requested: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = requested.shape[0]
        dependent_count = len(self.dependent_tilde_names)
        canonical = np.full((count, self.dimension), np.nan, dtype=float)
        dependent = np.full((count, dependent_count), np.nan, dtype=float)
        valid = np.zeros(count, dtype=bool)
        if count == 0:
            return canonical, valid, dependent
        transform = ConstrainedTildeTransform(self.constraints)
        try:
            canonical[:, :] = transform.independent_to_targets(requested)
            dependent[:, :] = transform.expand_independent(
                requested
            ).dependent_values
            valid[:] = True
            return canonical, valid, dependent
        except (TransformDomainError, TransformConstraintError, ValueError):
            # A single transform-domain failure must not invalidate unrelated
            # Cartesian slots.  Retry rowwise so later pipeline stages can
            # assign INVALID_TARGET only to the affected points.
            pass
        for index in range(count):
            row = requested[index : index + 1]
            try:
                canonical[index] = transform.independent_to_targets(row)[0]
                dependent[index] = transform.expand_independent(
                    row
                ).dependent_values[0]
                valid[index] = True
            except (TransformDomainError, TransformConstraintError, ValueError):
                continue
        return canonical, valid, dependent


def _estimate_result_bytes(
    grid: InversionGrid, auxiliary_count: int
) -> int:
    points = grid.size
    dimension = grid.dimension
    # Four dense d-vectors: source, achieved targets, residual, scaled residual.
    per_point = 4 * dimension * np.dtype(np.float64).itemsize
    # Infinity root norm and L2 metric/projection distance.
    per_point += 2 * np.dtype(np.float64).itemsize
    # disposition, route, prescreen, final solve, transient fast solve, proof,
    # projection quality, initial-guess provenance, phase, phase route, and
    # phase selection reason, and verified mixed-root count.
    per_point += 12 * np.dtype(np.uint8).itemsize
    # Conservative mixed-image match count.
    per_point += np.dtype(np.uint32).itemsize
    # Candidate/exact-candidate, boundary, multivalued, LINE-fraction-defined,
    # and alternate-root-defined flags.
    per_point += 6 * np.dtype(np.bool_).itemsize
    # Optional LINE fraction and alternate physical single-phase root.
    per_point += (1 + dimension) * np.dtype(np.float64).itemsize
    # Auxiliary value plus a field-local defined flag.
    per_point += auxiliary_count * (
        np.dtype(np.float64).itemsize + np.dtype(np.bool_).itemsize
    )
    # A row-validity flag is retained even for target grids, where it is
    # identically false and the achieved-tilde trailing dimension is zero.
    per_point += np.dtype(np.bool_).itemsize
    # For a tilde grid, retain achieved independent+dependent values.
    # Requested values remain lazy in GridChunk.
    if grid.space is GridSpace.TILDE:
        per_point += 4 * np.dtype(np.float64).itemsize
    return int(points * per_point)


def _validated_target_bounds(
    table: ForwardTable, target_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Read target extrema already computed during table validation."""

    report = table.report
    if report is None:
        raise PipelineStateError(
            "validated table has no validation report for automatic residual scaling"
        )
    field_statistics = report.statistics.get("fields")
    if not isinstance(field_statistics, Mapping):
        raise PipelineStateError(
            "validated table has no field statistics for automatic residual scaling"
        )
    lower = []
    upper = []
    for name in target_names:
        statistics = field_statistics.get(name)
        if not isinstance(statistics, Mapping):
            raise PipelineStateError(
                "validated table has no extrema for target '%s'" % name
            )
        minimum = statistics.get("finite_minimum")
        maximum = statistics.get("finite_maximum")
        try:
            minimum_value = float(minimum)
            maximum_value = float(maximum)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PipelineStateError(
                "validated target '%s' has unusable extrema" % name
            ) from exc
        if (
            not math.isfinite(minimum_value)
            or not math.isfinite(maximum_value)
            or minimum_value > maximum_value
        ):
            raise PipelineStateError(
                "validated target '%s' has unusable extrema" % name
            )
        lower.append(minimum_value)
        upper.append(maximum_value)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


@dataclass(frozen=True)
class PipelinePlan:
    """Resolved, immutable Phase-1 plan for one validated table."""

    config: PipelineConfig
    grid: InversionGrid
    source_names: Tuple[str, ...]
    source_units: Tuple[str, ...]
    target_names: Tuple[str, ...]
    target_units: Tuple[str, ...]
    auxiliary_names: Tuple[str, ...]
    auxiliary_units: Tuple[str, ...]
    residual_scales: np.ndarray
    estimated_result_bytes: int
    estimated_peak_result_bytes: int
    table_name: str
    table_revision: Optional[str]
    metadata_path: str
    data_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, PipelineConfig):
            raise PipelineStateError("plan config is invalid")
        if not isinstance(self.grid, InversionGrid):
            raise PipelineStateError("plan grid is invalid")
        dimension = self.grid.dimension
        source_names = tuple(self.source_names)
        source_units = tuple(self.source_units)
        target_names = tuple(self.target_names)
        target_units = tuple(self.target_units)
        auxiliary_names = tuple(self.auxiliary_names)
        auxiliary_units = tuple(self.auxiliary_units)
        if len(source_names) != dimension or len(source_units) != dimension:
            raise PipelineStateError("source metadata does not match dimension")
        if len(target_names) != dimension or len(target_units) != dimension:
            raise PipelineStateError("target metadata does not match dimension")
        if len(auxiliary_names) != len(auxiliary_units):
            raise PipelineStateError("auxiliary names and units do not align")
        scales = np.asarray(self.residual_scales, dtype=float)
        if scales.shape != (dimension,) or not np.all(np.isfinite(scales)):
            raise PipelineStateError("plan residual scales are invalid")
        if np.any(scales <= 0.0) or not np.all(np.isfinite(1.0 / scales)):
            raise PipelineStateError("plan residual scales must be safely positive")
        estimated = _positive_integer(
            self.estimated_result_bytes, "estimated_result_bytes"
        )
        estimated_peak = _positive_integer(
            self.estimated_peak_result_bytes,
            "estimated_peak_result_bytes",
        )
        if estimated_peak < 2 * estimated:
            raise PipelineStateError(
                "estimated peak result storage must include mutable and "
                "immutable result buffers"
            )
        object.__setattr__(self, "source_names", source_names)
        object.__setattr__(self, "source_units", source_units)
        object.__setattr__(self, "target_names", target_names)
        object.__setattr__(self, "target_units", target_units)
        object.__setattr__(self, "auxiliary_names", auxiliary_names)
        object.__setattr__(self, "auxiliary_units", auxiliary_units)
        object.__setattr__(self, "residual_scales", _immutable_array(scales))
        object.__setattr__(self, "estimated_result_bytes", estimated)
        object.__setattr__(
            self, "estimated_peak_result_bytes", estimated_peak
        )

    @classmethod
    def from_table(
        cls, table: ForwardTable, config: PipelineConfig
    ) -> "PipelinePlan":
        """Resolve names, bounds, scales, and storage before any solve."""

        if not isinstance(table, ForwardTable):
            raise TypeError("table must be a ForwardTable")
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig")
        table.require_validated()
        metadata = table.metadata
        assert metadata is not None
        grid = InversionGrid.from_table(table, config.grid)

        expected_targets = tuple(metadata.target_names)
        expected_scale_names = set(expected_targets)
        if config.residual_scales is None:
            lower, upper = _validated_target_bounds(table, expected_targets)
            try:
                residual_scales = _automatic_residual_scales_from_bounds(
                    lower, upper
                )
            except ValueError as exc:
                raise PipelineConfigurationError(
                    "could not derive automatic residual scales: %s" % exc
                ) from exc
            resolved_scales = {
                name: float(value)
                for name, value in zip(expected_targets, residual_scales)
            }
            config = replace(config, residual_scales=resolved_scales)
        else:
            scale_names = set(config.residual_scales)
            if scale_names != expected_scale_names:
                missing = sorted(expected_scale_names - scale_names)
                extra = sorted(scale_names - expected_scale_names)
                details = []
                if missing:
                    details.append("missing %s" % ", ".join(missing))
                if extra:
                    details.append("unexpected %s" % ", ".join(extra))
                raise PipelineConfigurationError(
                    "residual_scales must exactly match canonical targets (%s)"
                    % "; ".join(details)
                )
            residual_scales = np.asarray(
                [config.residual_scales[name] for name in expected_targets],
                dtype=float,
            )

        non_linear_targets = [
            name
            for name in expected_targets
            if metadata.fields[name].interpolation != "linear"
        ]
        if non_linear_targets:
            raise PipelineConfigurationError(
                "canonical target fields must declare interpolation='linear': %s"
                % ", ".join(non_linear_targets)
            )

        if config.auxiliary_fields is None:
            auxiliary_names = tuple(
                name
                for name in metadata.auxiliary_names
                if metadata.fields[name].output
            )
        else:
            auxiliary_names = tuple(config.auxiliary_fields)
        unknown = [
            name
            for name in auxiliary_names
            if name not in metadata.fields
            or metadata.fields[name].role != "auxiliary"
        ]
        if unknown:
            raise PipelineConfigurationError(
                "requested auxiliary fields are unknown or not auxiliary: %s"
                % ", ".join(unknown)
            )
        non_linear_auxiliary = [
            name
            for name in auxiliary_names
            if metadata.fields[name].interpolation != "linear"
        ]
        if non_linear_auxiliary:
            raise PipelineConfigurationError(
                "requested auxiliary fields must declare interpolation='linear': %s"
                % ", ".join(non_linear_auxiliary)
            )
        unused_line_policies = sorted(
            set(config.line_auxiliary_policies) - set(auxiliary_names)
        )
        if unused_line_policies:
            raise PipelineConfigurationError(
                "LINE auxiliary policies refer to fields not retained by this "
                "run: %s" % ", ".join(unused_line_policies)
            )

        source_names = tuple(metadata.coordinate_names)
        source_units = tuple(
            metadata.fields[name].canonical_unit for name in source_names
        )
        target_units = tuple(
            metadata.fields[name].canonical_unit for name in expected_targets
        )
        auxiliary_units = tuple(
            metadata.fields[name].canonical_unit for name in auxiliary_names
        )
        estimated = _estimate_result_bytes(grid, len(auxiliary_names))
        estimated_peak = 2 * estimated
        if estimated > config.grid.max_result_bytes:
            raise PipelineConfigurationError(
                "estimated result storage %d bytes exceeds max_result_bytes=%d"
                % (estimated, config.grid.max_result_bytes)
            )
        assert config.grid.max_peak_result_bytes is not None
        if estimated_peak > config.grid.max_peak_result_bytes:
            raise PipelineConfigurationError(
                "estimated result finalization peak %d bytes exceeds "
                "max_peak_result_bytes=%d"
                % (estimated_peak, config.grid.max_peak_result_bytes)
            )
        data_path = table.data_path
        if data_path is None:
            raise PipelineStateError("validated table has no resolved data path")
        return cls(
            config=config,
            grid=grid,
            source_names=source_names,
            source_units=source_units,
            target_names=expected_targets,
            target_units=target_units,
            auxiliary_names=auxiliary_names,
            auxiliary_units=auxiliary_units,
            residual_scales=residual_scales,
            estimated_result_bytes=estimated,
            estimated_peak_result_bytes=estimated_peak,
            table_name=metadata.name,
            table_revision=metadata.table_revision,
            metadata_path=str(table.metadata_path),
            data_path=str(data_path),
        )

    def iter_chunks(self) -> Iterator[GridChunk]:
        return self.grid.iter_chunks(self.config.execution.chunk_size)

    def allocate_result_buffer(self) -> "_PipelineResultBuffer":
        """Allocate the private mutable accumulator used by future stages."""

        return _PipelineResultBuffer(self)

    def numerical_routing_contract(self) -> Dict[str, Any]:
        """Return the immutable numerical routing policy for this plan."""

        contract = {
            "version": PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
            "ordinary_dimensions": [1, 2, 3, 4],
            "complete_localization": "required_for_restricted_fallback",
            "reachability_seed_enabled": self.config.use_reachability_seed,
            "seed_precedence": (
                [
                    "robust_direct_solved_inverse_grid_neighbor",
                    "best_complete_candidate_cell_center",
                ]
                if self.config.use_reachability_seed
                else [
                    "robust_direct_solved_inverse_grid_neighbor",
                    "precomputed_coarse_grid_seed",
                    "generic_configured_initial_guess",
                ]
            ),
            "fast_attempt_policy": "exactly_one_selected_seed",
            "fallback_domain": "complete_reachability_routing_cell_set",
            "candidate_localization_lifetime": "worker_chunk_only",
            "candidate_localization_persisted": False,
            "first_order_policy": "separate_phase_aware_path",
        }
        # Preserve the established disabled-route structure.  The enabled
        # scientific route carries its additional contract explicitly and
        # also differs through serialized SolverConfig.
        if self.config.solver.enable_subcell_fallback:
            contract.update(
                {
                    "fallback_cell_resolution": (
                        "dyadic_subcells_then_shared_krawczyk"
                    ),
                    "subcell_fallback": {
                        "enabled": True,
                        "maximum_depth": self.config.solver.max_subcell_depth,
                        "maximum_active_boxes": (
                            self.config.solver.max_subcell_active_boxes
                        ),
                        "incomplete_semantics": "unresolved",
                    },
                }
            )
        return contract

    def manifest(self) -> Dict[str, Any]:
        """Return a JSON-compatible reproducibility manifest skeleton."""

        return {
            "pipeline_result_schema_version": PIPELINE_RESULT_SCHEMA_VERSION,
            "numerical_routing_contract": self.numerical_routing_contract(),
            "table": {
                "name": self.table_name,
                "revision": self.table_revision,
                "metadata_path": self.metadata_path,
                "data_path": self.data_path,
            },
            "grid": {
                "space": self.grid.space.value,
                "axis_names": list(self.grid.axis_names),
                "axis_units": list(self.grid.axis_units),
                "axes": [axis.tolist() for axis in self.grid.axes],
                "shape": list(self.grid.shape),
                "size": self.grid.size,
                "ordering": self.grid.ordering,
                "coverage": self.grid.coverage,
                "cartesian_product_reachability": "not_guaranteed",
                "transform_id": self.grid.transform_id,
                "dependent_tilde_names": list(
                    self.grid.dependent_tilde_names
                ),
            },
            "source": {
                "names": list(self.source_names),
                "units": list(self.source_units),
            },
            "target": {
                "names": list(self.target_names),
                "units": list(self.target_units),
                "residual_scales": {
                    name: float(scale)
                    for name, scale in zip(
                        self.target_names, self.residual_scales
                    )
                },
                "root_acceptance_norm": "scaled_linf",
                "nearest_and_projection_norm": "scaled_l2",
            },
            "auxiliary": {
                "names": list(self.auxiliary_names),
                "units": list(self.auxiliary_units),
                "line_mixed_phase": {
                    "default_policy": (
                        self.config.line_auxiliary_default_policy.value
                    ),
                    "field_policies": {
                        name: self.config.line_auxiliary_policies.get(
                            name,
                            self.config.line_auxiliary_default_policy,
                        ).value
                        for name in self.auxiliary_names
                    },
                    "lambda_semantics": (
                        "lambda=0 hadronic endpoint; lambda=1 qgp endpoint"
                    ),
                    "linear_rule": "(1-lambda)*X_H + lambda*X_Q",
                    "empty_encoding": "nan_with_auxiliary_defined_false",
                    "scientific_status": (
                        "explicit_lever_rule_or_effective_numerical_prescription"
                    ),
                },
            },
            "configuration": self.config.to_dict(),
            "estimated_result_bytes": self.estimated_result_bytes,
            "estimated_peak_result_bytes": self.estimated_peak_result_bytes,
            "status_codebooks": _result_codebooks(),
        }


class _PipelineResultBuffer:
    """Private mutable columnar accumulator, finalized exactly once."""

    def __init__(self, plan: PipelinePlan) -> None:
        if not isinstance(plan, PipelinePlan):
            raise TypeError("plan must be a PipelinePlan")
        self.plan = plan
        shape = plan.grid.shape
        vector_shape = shape + (plan.grid.dimension,)
        auxiliary_shape = shape + (len(plan.auxiliary_names),)
        tilde_columns = 4 if plan.grid.space is GridSpace.TILDE else 0
        tilde_shape = shape + (tilde_columns,)
        self.source_coordinates = np.full(vector_shape, np.nan, dtype=float)
        self.achieved_targets = np.full(vector_shape, np.nan, dtype=float)
        self.residual = np.full(vector_shape, np.nan, dtype=float)
        self.scaled_residual = np.full(vector_shape, np.nan, dtype=float)
        self.scaled_residual_norm_inf = np.full(shape, np.nan, dtype=float)
        self.scaled_distance_l2 = np.full(shape, np.nan, dtype=float)
        self.auxiliary_values = np.full(auxiliary_shape, np.nan, dtype=float)
        self.auxiliary_defined = np.zeros(auxiliary_shape, dtype=bool)
        self.achieved_tilde_values = np.full(tilde_shape, np.nan, dtype=float)
        self.achieved_tilde_defined = np.zeros(shape, dtype=bool)
        self.disposition_codes = np.zeros(shape, dtype=np.uint8)
        self.route_codes = np.zeros(shape, dtype=np.uint8)
        self.prescreen_codes = np.zeros(shape, dtype=np.uint8)
        self.solve_status_codes = np.zeros(shape, dtype=np.uint8)
        self.fast_status_codes = np.zeros(shape, dtype=np.uint8)
        self.proof_status_codes = np.zeros(shape, dtype=np.uint8)
        self.projection_quality_codes = np.zeros(shape, dtype=np.uint8)
        self.initial_guess_source_codes = np.zeros(shape, dtype=np.uint8)
        self.phase_codes = np.full(
            shape,
            _enum_code(PhaseLabel.NOT_APPLICABLE, _PHASE_CODES),
            dtype=np.uint8,
        )
        self.phase_route_codes = np.zeros(shape, dtype=np.uint8)
        self.phase_selection_reason_codes = np.full(
            shape,
            _enum_code(
                PhaseRootSelectionReason.NOT_APPLICABLE,
                _PHASE_SELECTION_REASON_CODES,
            ),
            dtype=np.uint8,
        )
        self.mixed_candidate = np.zeros(shape, dtype=bool)
        self.exact_mixed_candidate = np.zeros(shape, dtype=bool)
        self.mixed_candidate_entry_count = np.zeros(shape, dtype=np.uint32)
        self.mixed_verified_root_count = np.zeros(shape, dtype=np.uint8)
        self.phase_boundary = np.zeros(shape, dtype=bool)
        self.phase_multivalued = np.zeros(shape, dtype=bool)
        self.line_mixture_fraction = np.full(shape, np.nan, dtype=float)
        self.line_mixture_fraction_defined = np.zeros(shape, dtype=bool)
        self.alternate_single_phase_source = np.full(
            vector_shape, np.nan, dtype=float
        )
        self.alternate_single_phase_source_defined = np.zeros(shape, dtype=bool)
        self._closed = False
        if self.nbytes != plan.estimated_result_bytes:
            raise PipelineStateError(
                "result allocation disagrees with the planned byte estimate"
            )

    @property
    def nbytes(self) -> int:
        arrays = (
            self.source_coordinates,
            self.achieved_targets,
            self.residual,
            self.scaled_residual,
            self.scaled_residual_norm_inf,
            self.scaled_distance_l2,
            self.auxiliary_values,
            self.auxiliary_defined,
            self.achieved_tilde_values,
            self.achieved_tilde_defined,
            self.disposition_codes,
            self.route_codes,
            self.prescreen_codes,
            self.solve_status_codes,
            self.fast_status_codes,
            self.proof_status_codes,
            self.projection_quality_codes,
            self.initial_guess_source_codes,
            self.phase_codes,
            self.phase_route_codes,
            self.phase_selection_reason_codes,
            self.mixed_candidate,
            self.exact_mixed_candidate,
            self.mixed_candidate_entry_count,
            self.mixed_verified_root_count,
            self.phase_boundary,
            self.phase_multivalued,
            self.line_mixture_fraction,
            self.line_mixture_fraction_defined,
            self.alternate_single_phase_source,
            self.alternate_single_phase_source_defined,
        )
        return int(sum(array.nbytes for array in arrays))

    def finalize(
        self, manifest_updates: Optional[Mapping[str, Any]] = None
    ) -> "PipelineResult":
        """Validate terminal invariants and return an immutable result."""

        if self._closed:
            raise PipelineStateError("result buffer has already been finalized")
        manifest = self.plan.manifest()
        if manifest_updates is not None:
            if not isinstance(manifest_updates, Mapping):
                raise PipelineStateError("manifest_updates must be a mapping")
            manifest["run"] = dict(manifest_updates)
        result = PipelineResult(
            plan=self.plan,
            source_coordinates=self.source_coordinates,
            achieved_targets=self.achieved_targets,
            residual=self.residual,
            scaled_residual=self.scaled_residual,
            scaled_residual_norm_inf=self.scaled_residual_norm_inf,
            scaled_distance_l2=self.scaled_distance_l2,
            auxiliary_values=self.auxiliary_values,
            auxiliary_defined=self.auxiliary_defined,
            achieved_tilde_values=self.achieved_tilde_values,
            achieved_tilde_defined=self.achieved_tilde_defined,
            disposition_codes=self.disposition_codes,
            route_codes=self.route_codes,
            prescreen_codes=self.prescreen_codes,
            solve_status_codes=self.solve_status_codes,
            fast_status_codes=self.fast_status_codes,
            proof_status_codes=self.proof_status_codes,
            projection_quality_codes=self.projection_quality_codes,
            initial_guess_source_codes=self.initial_guess_source_codes,
            phase_codes=self.phase_codes,
            phase_route_codes=self.phase_route_codes,
            phase_selection_reason_codes=self.phase_selection_reason_codes,
            mixed_candidate=self.mixed_candidate,
            exact_mixed_candidate=self.exact_mixed_candidate,
            mixed_candidate_entry_count=self.mixed_candidate_entry_count,
            mixed_verified_root_count=self.mixed_verified_root_count,
            phase_boundary=self.phase_boundary,
            phase_multivalued=self.phase_multivalued,
            line_mixture_fraction=self.line_mixture_fraction,
            line_mixture_fraction_defined=self.line_mixture_fraction_defined,
            alternate_single_phase_source=self.alternate_single_phase_source,
            alternate_single_phase_source_defined=(
                self.alternate_single_phase_source_defined
            ),
            manifest=manifest,
        )
        self._closed = True
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        return result


def _status_array(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise PipelineStateError("%s must be an integer array of shape %r" % (name, shape))
    if np.any(array < 0) or np.any(array > np.iinfo(np.uint8).max):
        raise PipelineStateError("%s contains an invalid code" % name)
    return np.ascontiguousarray(array, dtype=np.uint8)


def _count_array(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise PipelineStateError("%s must be an integer array of shape %r" % (name, shape))
    if np.any(array < 0) or np.any(array > np.iinfo(np.uint32).max):
        raise PipelineStateError("%s contains an invalid count" % name)
    return np.ascontiguousarray(array, dtype=np.uint32)


def _numeric_array(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=float))
    if array.shape != shape:
        raise PipelineStateError("%s must have shape %r" % (name, shape))
    return array


def _boolean_array(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=bool))
    if array.shape != shape:
        raise PipelineStateError("%s must have shape %r" % (name, shape))
    return array


def _codes_are_valid(array: np.ndarray, codes: Sequence[int]) -> bool:
    return bool(np.all(np.isin(array, np.asarray(tuple(codes), dtype=np.uint8))))


@dataclass(frozen=True)
class PipelineResult:
    """Immutable, columnar terminal result for every requested grid slot."""

    plan: PipelinePlan
    source_coordinates: np.ndarray
    achieved_targets: np.ndarray
    residual: np.ndarray
    scaled_residual: np.ndarray
    scaled_residual_norm_inf: np.ndarray
    scaled_distance_l2: np.ndarray
    auxiliary_values: np.ndarray
    auxiliary_defined: np.ndarray
    achieved_tilde_values: np.ndarray
    achieved_tilde_defined: np.ndarray
    disposition_codes: np.ndarray
    route_codes: np.ndarray
    prescreen_codes: np.ndarray
    solve_status_codes: np.ndarray
    fast_status_codes: np.ndarray
    proof_status_codes: np.ndarray
    projection_quality_codes: np.ndarray
    initial_guess_source_codes: np.ndarray
    phase_codes: np.ndarray
    phase_route_codes: np.ndarray
    phase_selection_reason_codes: np.ndarray
    mixed_candidate: np.ndarray
    exact_mixed_candidate: np.ndarray
    mixed_candidate_entry_count: np.ndarray
    mixed_verified_root_count: np.ndarray
    phase_boundary: np.ndarray
    phase_multivalued: np.ndarray
    line_mixture_fraction: np.ndarray
    line_mixture_fraction_defined: np.ndarray
    alternate_single_phase_source: np.ndarray
    alternate_single_phase_source_defined: np.ndarray
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PipelinePlan):
            raise PipelineStateError("result plan is invalid")
        shape = self.plan.grid.shape
        vector_shape = shape + (self.plan.grid.dimension,)
        auxiliary_shape = shape + (len(self.plan.auxiliary_names),)
        tilde_columns = 4 if self.plan.grid.space is GridSpace.TILDE else 0
        tilde_shape = shape + (tilde_columns,)

        numeric_vectors = {
            "source_coordinates": _numeric_array(
                self.source_coordinates, vector_shape, "source_coordinates"
            ),
            "achieved_targets": _numeric_array(
                self.achieved_targets, vector_shape, "achieved_targets"
            ),
            "residual": _numeric_array(self.residual, vector_shape, "residual"),
            "scaled_residual": _numeric_array(
                self.scaled_residual, vector_shape, "scaled_residual"
            ),
            "auxiliary_values": _numeric_array(
                self.auxiliary_values, auxiliary_shape, "auxiliary_values"
            ),
            "achieved_tilde_values": _numeric_array(
                self.achieved_tilde_values,
                tilde_shape,
                "achieved_tilde_values",
            ),
            "alternate_single_phase_source": _numeric_array(
                self.alternate_single_phase_source,
                vector_shape,
                "alternate_single_phase_source",
            ),
        }
        numeric_scalars = {
            "scaled_residual_norm_inf": _numeric_array(
                self.scaled_residual_norm_inf,
                shape,
                "scaled_residual_norm_inf",
            ),
            "scaled_distance_l2": _numeric_array(
                self.scaled_distance_l2, shape, "scaled_distance_l2"
            ),
            "line_mixture_fraction": _numeric_array(
                self.line_mixture_fraction, shape, "line_mixture_fraction"
            ),
        }
        booleans = {
            "auxiliary_defined": _boolean_array(
                self.auxiliary_defined, auxiliary_shape, "auxiliary_defined"
            ),
            "achieved_tilde_defined": _boolean_array(
                self.achieved_tilde_defined,
                shape,
                "achieved_tilde_defined",
            ),
            "mixed_candidate": _boolean_array(
                self.mixed_candidate, shape, "mixed_candidate"
            ),
            "exact_mixed_candidate": _boolean_array(
                self.exact_mixed_candidate, shape, "exact_mixed_candidate"
            ),
            "phase_boundary": _boolean_array(
                self.phase_boundary, shape, "phase_boundary"
            ),
            "phase_multivalued": _boolean_array(
                self.phase_multivalued, shape, "phase_multivalued"
            ),
            "line_mixture_fraction_defined": _boolean_array(
                self.line_mixture_fraction_defined,
                shape,
                "line_mixture_fraction_defined",
            ),
            "alternate_single_phase_source_defined": _boolean_array(
                self.alternate_single_phase_source_defined,
                shape,
                "alternate_single_phase_source_defined",
            ),
        }
        statuses = {
            "disposition_codes": _status_array(
                self.disposition_codes, shape, "disposition_codes"
            ),
            "route_codes": _status_array(self.route_codes, shape, "route_codes"),
            "prescreen_codes": _status_array(
                self.prescreen_codes, shape, "prescreen_codes"
            ),
            "solve_status_codes": _status_array(
                self.solve_status_codes, shape, "solve_status_codes"
            ),
            "fast_status_codes": _status_array(
                self.fast_status_codes, shape, "fast_status_codes"
            ),
            "proof_status_codes": _status_array(
                self.proof_status_codes, shape, "proof_status_codes"
            ),
            "projection_quality_codes": _status_array(
                self.projection_quality_codes,
                shape,
                "projection_quality_codes",
            ),
            "initial_guess_source_codes": _status_array(
                self.initial_guess_source_codes,
                shape,
                "initial_guess_source_codes",
            ),
            "phase_codes": _status_array(
                self.phase_codes, shape, "phase_codes"
            ),
            "phase_route_codes": _status_array(
                self.phase_route_codes, shape, "phase_route_codes"
            ),
            "phase_selection_reason_codes": _status_array(
                self.phase_selection_reason_codes,
                shape,
                "phase_selection_reason_codes",
            ),
            "mixed_verified_root_count": _status_array(
                self.mixed_verified_root_count,
                shape,
                "mixed_verified_root_count",
            ),
        }
        counts = {
            "mixed_candidate_entry_count": _count_array(
                self.mixed_candidate_entry_count,
                shape,
                "mixed_candidate_entry_count",
            )
        }
        for name, array in numeric_vectors.items():
            object.__setattr__(self, name, array)
        for name, array in numeric_scalars.items():
            object.__setattr__(self, name, array)
        for name, array in booleans.items():
            object.__setattr__(self, name, array)
        for name, array in statuses.items():
            object.__setattr__(self, name, array)
        for name, array in counts.items():
            object.__setattr__(self, name, array)

        self._validate_terminal_contract()

        array_names = (
            tuple(numeric_vectors)
            + tuple(numeric_scalars)
            + tuple(booleans)
            + tuple(statuses)
            + tuple(counts)
        )
        for name in array_names:
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))
        object.__setattr__(self, "manifest", _freeze_manifest(self.manifest))

    @property
    def grid(self) -> InversionGrid:
        return self.plan.grid

    @property
    def codebooks(self) -> Mapping[str, Any]:
        return self.manifest["status_codebooks"]

    @property
    def nbytes(self) -> int:
        """Exact bytes owned by the dense immutable result columns."""

        arrays = (
            self.source_coordinates,
            self.achieved_targets,
            self.residual,
            self.scaled_residual,
            self.scaled_residual_norm_inf,
            self.scaled_distance_l2,
            self.auxiliary_values,
            self.auxiliary_defined,
            self.achieved_tilde_values,
            self.achieved_tilde_defined,
            self.disposition_codes,
            self.route_codes,
            self.prescreen_codes,
            self.solve_status_codes,
            self.fast_status_codes,
            self.proof_status_codes,
            self.projection_quality_codes,
            self.initial_guess_source_codes,
            self.phase_codes,
            self.phase_route_codes,
            self.phase_selection_reason_codes,
            self.mixed_candidate,
            self.exact_mixed_candidate,
            self.mixed_candidate_entry_count,
            self.mixed_verified_root_count,
            self.phase_boundary,
            self.phase_multivalued,
            self.line_mixture_fraction,
            self.line_mixture_fraction_defined,
            self.alternate_single_phase_source,
            self.alternate_single_phase_source_defined,
        )
        return int(sum(array.nbytes for array in arrays))

    def _mask(self, value: Any) -> np.ndarray:
        return _immutable_array(np.asarray(value, dtype=bool))

    @property
    def prescreen_evaluated_mask(self) -> np.ndarray:
        return self._mask(
            self.prescreen_codes != _enum_code(
                PrescreenStatus.NOT_RUN, _PRESCREEN_CODES
            )
        )

    @property
    def candidate_mask(self) -> np.ndarray:
        """Points the screen could not exclude; false alone is not a proof.

        Consult :attr:`prescreen_evaluated_mask` or
        :attr:`prescreen_excluded_mask` to distinguish an exclusion proof from
        a target for which the screen never ran.
        """

        return self._mask(
            self.prescreen_codes
            == _enum_code(PrescreenStatus.CANDIDATE, _PRESCREEN_CODES)
        )

    @property
    def prescreen_excluded_mask(self) -> np.ndarray:
        return self._mask(
            self.prescreen_codes
            == _enum_code(
                PrescreenStatus.DEFINITELY_UNREACHABLE, _PRESCREEN_CODES
            )
        )

    @property
    def solution_mask(self) -> np.ndarray:
        return self._mask(
            self.disposition_codes
            == _enum_code(PointDisposition.SOLVED, _DISPOSITION_CODES)
        )

    @property
    def projected_mask(self) -> np.ndarray:
        return self._mask(
            self.disposition_codes
            == _enum_code(PointDisposition.PROJECTED, _DISPOSITION_CODES)
        )

    @property
    def usable_state_mask(self) -> np.ndarray:
        return self._mask(self.solution_mask | self.projected_mask)

    @property
    def single_phase_mask(self) -> np.ndarray:
        return self._mask(
            self.phase_codes
            == _enum_code(PhaseLabel.SINGLE_PHASE, _PHASE_CODES)
        )

    @property
    def mixed_phase_mask(self) -> np.ndarray:
        return self._mask(
            self.phase_codes
            == _enum_code(PhaseLabel.MIXED_PHASE, _PHASE_CODES)
        )

    @property
    def phase_applicable_mask(self) -> np.ndarray:
        return self._mask(self.single_phase_mask | self.mixed_phase_mask)

    @property
    def jacobian_selected_mask(self) -> np.ndarray:
        """Solved roots selected from a verified local interpolation fold."""

        return self._mask(
            self.phase_selection_reason_codes
            == _enum_code(
                PhaseRootSelectionReason.SELECTED_FROM_JACOBIAN,
                _PHASE_SELECTION_REASON_CODES,
            )
        )

    @property
    def certified_reachable_mask(self) -> np.ndarray:
        return self._mask(
            self.proof_status_codes
            == _enum_code(ProofStatus.UNIQUE_IN_BOX, _PROOF_STATUS_CODES)
        )

    @property
    def certified_unreachable_mask(self) -> np.ndarray:
        return self._mask(
            self.prescreen_excluded_mask
            | (
                self.proof_status_codes
                == _enum_code(
                    ProofStatus.NO_ROOT_IN_DOMAIN, _PROOF_STATUS_CODES
                )
            )
        )

    @property
    def unknown_mask(self) -> np.ndarray:
        return self._mask(
            self.disposition_codes
            == _enum_code(PointDisposition.UNRESOLVED, _DISPOSITION_CODES)
        )

    @property
    def projection_failed_mask(self) -> np.ndarray:
        return self._mask(
            self.disposition_codes
            == _enum_code(
                PointDisposition.PROJECTION_FAILED, _DISPOSITION_CODES
            )
        )

    @property
    def invalid_target_mask(self) -> np.ndarray:
        return self._mask(
            self.disposition_codes
            == _enum_code(PointDisposition.INVALID_TARGET, _DISPOSITION_CODES)
        )

    @property
    def error_mask(self) -> np.ndarray:
        return self._mask(
            self.disposition_codes
            == _enum_code(PointDisposition.ERROR, _DISPOSITION_CODES)
        )

    @property
    def native_status_valid_mask(self) -> np.ndarray:
        return self._mask(self.solve_status_codes != 0)

    @property
    def native_proof_valid_mask(self) -> np.ndarray:
        return self._mask(self.proof_status_codes != 0)

    @property
    def projection_quality_valid_mask(self) -> np.ndarray:
        return self._mask(self.projection_quality_codes != 0)

    def auxiliary_defined_mask(self, name: str) -> np.ndarray:
        try:
            index = self.plan.auxiliary_names.index(name)
        except ValueError as exc:
            raise KeyError("unknown result auxiliary field: %s" % name) from exc
        return _immutable_array(self.auxiliary_defined[..., index])

    @property
    def status_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                disposition.value: int(
                    np.count_nonzero(
                        self.disposition_codes
                        == _enum_code(disposition, _DISPOSITION_CODES)
                    )
                )
                for disposition in PointDisposition
            }
        )

    def _validate_terminal_contract(self) -> None:
        disposition = self.disposition_codes
        route = self.route_codes
        prescreen = self.prescreen_codes
        solve = self.solve_status_codes
        fast = self.fast_status_codes
        proof = self.proof_status_codes
        quality = self.projection_quality_codes
        guess = self.initial_guess_source_codes
        phase = self.phase_codes
        phase_route = self.phase_route_codes
        phase_selection_reason = self.phase_selection_reason_codes

        if not _codes_are_valid(disposition, _DISPOSITION_CODES.values()):
            if np.any(disposition == 0):
                raise PipelineStateError(
                    "cannot finalize while point dispositions are unassigned"
                )
            raise PipelineStateError("result contains an unknown disposition code")
        if not _codes_are_valid(route, _ROUTE_CODES.values()):
            raise PipelineStateError("result contains an unknown or unassigned route code")
        if not _codes_are_valid(prescreen, _PRESCREEN_CODES.values()):
            raise PipelineStateError("result contains an unknown prescreen code")
        if not _codes_are_valid(solve, (0,) + tuple(_SOLVE_STATUS_CODES.values())):
            raise PipelineStateError("result contains an unknown solve-status code")
        if not _codes_are_valid(fast, (0,) + tuple(_SOLVE_STATUS_CODES.values())):
            raise PipelineStateError("result contains an unknown FAST-status code")
        if not _codes_are_valid(proof, (0,) + tuple(_PROOF_STATUS_CODES.values())):
            raise PipelineStateError("result contains an unknown proof-status code")
        if not _codes_are_valid(
            quality, (0,) + tuple(_PROJECTION_QUALITY_CODES.values())
        ):
            raise PipelineStateError("result contains an unknown projection-quality code")
        if not _codes_are_valid(guess, (0,) + tuple(_INITIAL_GUESS_CODES.values())):
            raise PipelineStateError("result contains an unknown initial-guess code")
        if not _codes_are_valid(phase, tuple(_PHASE_CODES.values())):
            raise PipelineStateError("result contains an unknown phase code")
        if not _codes_are_valid(
            phase_route, (0,) + tuple(_PHASE_ROUTE_CODES.values())
        ):
            raise PipelineStateError("result contains an unknown phase-route code")
        if not _codes_are_valid(
            phase_selection_reason,
            tuple(_PHASE_SELECTION_REASON_CODES.values()),
        ):
            raise PipelineStateError(
                "result contains an unknown phase-selection-reason code"
            )

        allowed_routes = {
            PointDisposition.SOLVED: (
                PointRoute.FAST,
                PointRoute.FALLBACK_CERTIFIED,
            ),
            PointDisposition.PROJECTED: (
                PointRoute.PROJECTED_PRESCREEN,
                PointRoute.PROJECTED_CERTIFIED_NO_ROOT,
            ),
            PointDisposition.CERTIFIED_UNREACHABLE: (
                PointRoute.CERTIFIED_UNREACHABLE_PRESCREEN,
                PointRoute.CERTIFIED_UNREACHABLE_FALLBACK,
            ),
            PointDisposition.ROOT_CERTIFIED_UNREFINED: (
                PointRoute.FALLBACK_ROOT_UNREFINED,
            ),
            PointDisposition.UNRESOLVED: (PointRoute.FALLBACK_UNRESOLVED,),
            PointDisposition.INVALID_TARGET: (PointRoute.INVALID_TRANSFORM,),
            PointDisposition.PROJECTION_FAILED: (
                PointRoute.PROJECTION_FAILED_PRESCREEN,
                PointRoute.PROJECTION_FAILED_CERTIFIED_NO_ROOT,
            ),
            PointDisposition.ERROR: (PointRoute.ERROR,),
        }
        for outcome, routes in allowed_routes.items():
            mask = disposition == _enum_code(outcome, _DISPOSITION_CODES)
            allowed = [_enum_code(item, _ROUTE_CODES) for item in routes]
            if np.any(mask & ~np.isin(route, allowed)):
                raise PipelineStateError(
                    "route is inconsistent with disposition %s" % outcome.value
                )

        candidate_routes = (
            PointRoute.FAST,
            PointRoute.FALLBACK_CERTIFIED,
            PointRoute.FALLBACK_ROOT_UNREFINED,
            PointRoute.FALLBACK_UNRESOLVED,
            PointRoute.PROJECTED_CERTIFIED_NO_ROOT,
            PointRoute.CERTIFIED_UNREACHABLE_FALLBACK,
            PointRoute.PROJECTION_FAILED_CERTIFIED_NO_ROOT,
        )
        candidate_route_mask = np.isin(
            route, [_enum_code(item, _ROUTE_CODES) for item in candidate_routes]
        )
        if np.any(
            candidate_route_mask
            & (prescreen != _enum_code(PrescreenStatus.CANDIDATE, _PRESCREEN_CODES))
        ):
            raise PipelineStateError("solver routes require a candidate pre-screen")
        prescreen_routes = (
            PointRoute.PROJECTED_PRESCREEN,
            PointRoute.CERTIFIED_UNREACHABLE_PRESCREEN,
            PointRoute.PROJECTION_FAILED_PRESCREEN,
        )
        prescreen_route_mask = np.isin(
            route, [_enum_code(item, _ROUTE_CODES) for item in prescreen_routes]
        )
        if np.any(
            prescreen_route_mask
            & (
                prescreen
                != _enum_code(
                    PrescreenStatus.DEFINITELY_UNREACHABLE, _PRESCREEN_CODES
                )
            )
        ):
            raise PipelineStateError(
                "pre-screen terminal routes require a definite exclusion"
            )
        invalid_mask = disposition == _enum_code(
            PointDisposition.INVALID_TARGET, _DISPOSITION_CODES
        )
        if np.any(invalid_mask & (prescreen != 0)):
            raise PipelineStateError("invalid targets cannot have a pre-screen result")

        route_masks = {
            item: route == _enum_code(item, _ROUTE_CODES) for item in PointRoute
        }

        def require_codes(
            mask: np.ndarray,
            *,
            solve_status: int,
            proof_status: int,
            fast_statuses: Sequence[int],
            label: str,
        ) -> None:
            if np.any(mask & (solve != solve_status)):
                raise PipelineStateError("%s has an invalid solve status" % label)
            if np.any(mask & (proof != proof_status)):
                raise PipelineStateError("%s has an invalid proof status" % label)
            if np.any(mask & ~np.isin(fast, tuple(fast_statuses))):
                raise PipelineStateError("%s has an invalid FAST status" % label)

        not_run_fast = (0,)
        failed_fast = (_enum_code(SolveStatus.FAST_PATH_FAILED, _SOLVE_STATUS_CODES),)
        require_codes(
            route_masks[PointRoute.FAST],
            solve_status=_enum_code(SolveStatus.CONVERGED_FAST, _SOLVE_STATUS_CODES),
            proof_status=_enum_code(ProofStatus.NONE, _PROOF_STATUS_CODES),
            fast_statuses=(
                _enum_code(SolveStatus.CONVERGED_FAST, _SOLVE_STATUS_CODES),
            ),
            label="FAST route",
        )
        require_codes(
            route_masks[PointRoute.FALLBACK_CERTIFIED],
            solve_status=_enum_code(
                SolveStatus.CONVERGED_FALLBACK, _SOLVE_STATUS_CODES
            ),
            proof_status=_enum_code(
                ProofStatus.UNIQUE_IN_BOX, _PROOF_STATUS_CODES
            ),
            fast_statuses=failed_fast,
            label="certified FALLBACK route",
        )
        require_codes(
            route_masks[PointRoute.FALLBACK_ROOT_UNREFINED],
            solve_status=_enum_code(
                SolveStatus.CERTIFIED_ROOT_UNREFINED, _SOLVE_STATUS_CODES
            ),
            proof_status=_enum_code(
                ProofStatus.UNIQUE_IN_BOX, _PROOF_STATUS_CODES
            ),
            fast_statuses=failed_fast,
            label="unrefined certified-root route",
        )
        require_codes(
            route_masks[PointRoute.FALLBACK_UNRESOLVED],
            solve_status=_enum_code(SolveStatus.UNRESOLVED, _SOLVE_STATUS_CODES),
            proof_status=_enum_code(
                ProofStatus.INCONCLUSIVE, _PROOF_STATUS_CODES
            ),
            fast_statuses=failed_fast,
            label="unresolved FALLBACK route",
        )
        for item in (
            PointRoute.PROJECTED_CERTIFIED_NO_ROOT,
            PointRoute.CERTIFIED_UNREACHABLE_FALLBACK,
            PointRoute.PROJECTION_FAILED_CERTIFIED_NO_ROOT,
        ):
            require_codes(
                route_masks[item],
                solve_status=_enum_code(
                    SolveStatus.CERTIFIED_NO_ROOT, _SOLVE_STATUS_CODES
                ),
                proof_status=_enum_code(
                    ProofStatus.NO_ROOT_IN_DOMAIN, _PROOF_STATUS_CODES
                ),
                fast_statuses=failed_fast,
                label=item.value,
            )
        for item in (
            PointRoute.PROJECTED_PRESCREEN,
            PointRoute.CERTIFIED_UNREACHABLE_PRESCREEN,
            PointRoute.PROJECTION_FAILED_PRESCREEN,
            PointRoute.INVALID_TRANSFORM,
        ):
            require_codes(
                route_masks[item],
                solve_status=0,
                proof_status=0,
                fast_statuses=not_run_fast,
                label=item.value,
            )

        projected = disposition == _enum_code(
            PointDisposition.PROJECTED, _DISPOSITION_CODES
        )
        if np.any(projected & (quality == 0)) or np.any(~projected & (quality != 0)):
            raise PipelineStateError(
                "projection quality must be set exactly for projected points"
            )
        solver_invoked = candidate_route_mask
        if np.any(solver_invoked & (guess == 0)):
            raise PipelineStateError("solver routes require initial-guess provenance")
        if np.any((prescreen_route_mask | invalid_mask) & (guess != 0)):
            raise PipelineStateError(
                "routes without a point solve cannot record a solver initial guess"
            )

        usable = (
            (disposition == _enum_code(PointDisposition.SOLVED, _DISPOSITION_CODES))
            | projected
        )

        solved = disposition == _enum_code(
            PointDisposition.SOLVED, _DISPOSITION_CODES
        )
        single_phase = phase == _enum_code(
            PhaseLabel.SINGLE_PHASE, _PHASE_CODES
        )
        mixed_phase = phase == _enum_code(
            PhaseLabel.MIXED_PHASE, _PHASE_CODES
        )
        not_applicable = phase == _enum_code(
            PhaseLabel.NOT_APPLICABLE, _PHASE_CODES
        )
        if np.any((single_phase | mixed_phase) & ~solved):
            raise PipelineStateError(
                "single/mixed phase labels require a solved requested state"
            )
        no_transition = phase_route == _enum_code(
            PhaseRoute.NO_TRANSITION, _PHASE_ROUTE_CODES
        )
        ordinary_phase = phase_route == _enum_code(
            PhaseRoute.ORDINARY, _PHASE_ROUTE_CODES
        )
        line_phase = phase_route == _enum_code(
            PhaseRoute.LINE_REGION_S, _PHASE_ROUTE_CODES
        )
        binodal_phase = phase_route == _enum_code(
            PhaseRoute.BINODALS_ORDINARY, _PHASE_ROUTE_CODES
        )
        if np.any(solved & no_transition & ~not_applicable):
            raise PipelineStateError(
                "no-transition roots must use phase not_applicable"
            )
        if np.any(solved & ordinary_phase & ~single_phase):
            raise PipelineStateError("ordinary transition roots must be single-phase")
        if np.any((line_phase | binodal_phase) & ~mixed_phase):
            raise PipelineStateError("mixed phase routes require a mixed-phase root")
        if np.any(mixed_phase & ~(line_phase | binodal_phase)):
            raise PipelineStateError("mixed-phase roots require LINE/BINODALS provenance")
        if np.any(self.exact_mixed_candidate & ~self.mixed_candidate):
            raise PipelineStateError(
                "an exact mixed candidate must also be a routing candidate"
            )
        if np.any(
            (self.mixed_candidate_entry_count > 0) & ~self.mixed_candidate
        ):
            raise PipelineStateError(
                "mixed candidate counts require a conservative candidate"
            )
        if np.any((self.mixed_verified_root_count > 0) != mixed_phase):
            raise PipelineStateError(
                "verified mixed-root counts must be positive exactly for mixed roots"
            )
        if np.any((self.phase_boundary | self.phase_multivalued) & ~mixed_phase):
            raise PipelineStateError(
                "phase boundary/multivalued flags require a mixed-phase root"
            )
        jacobian_selected = phase_selection_reason == _enum_code(
            PhaseRootSelectionReason.SELECTED_FROM_JACOBIAN,
            _PHASE_SELECTION_REASON_CODES,
        )
        unambiguous = phase_selection_reason == _enum_code(
            PhaseRootSelectionReason.UNAMBIGUOUS,
            _PHASE_SELECTION_REASON_CODES,
        )
        selection_not_applicable = phase_selection_reason == _enum_code(
            PhaseRootSelectionReason.NOT_APPLICABLE,
            _PHASE_SELECTION_REASON_CODES,
        )
        if np.any(jacobian_selected & (~mixed_phase | ~self.phase_multivalued)):
            raise PipelineStateError(
                "Jacobian-selected provenance requires a multivalued mixed root"
            )
        if np.any(jacobian_selected & (self.mixed_verified_root_count < 2)):
            raise PipelineStateError(
                "Jacobian-selected provenance requires multiple verified roots"
            )
        if np.any(unambiguous & ~solved):
            raise PipelineStateError(
                "unambiguous phase-selection provenance requires a solved root"
            )
        if np.any(selection_not_applicable & (single_phase | mixed_phase)):
            raise PipelineStateError(
                "solved first-order roots require an explicit selection reason"
            )
        line_fraction = self.line_mixture_fraction
        line_defined = self.line_mixture_fraction_defined
        if np.any(line_defined != (mixed_phase & line_phase)):
            raise PipelineStateError(
                "LINE fraction must be defined exactly for LINE mixed roots"
            )
        if np.any(
            line_defined
            & (
                ~np.isfinite(line_fraction)
                | (line_fraction < 0.0)
                | (line_fraction > 1.0)
            )
        ) or np.any(~line_defined & ~np.isnan(line_fraction)):
            raise PipelineStateError("LINE fraction values violate their defined mask")
        alternate_defined = self.alternate_single_phase_source_defined
        alternate = self.alternate_single_phase_source
        alternate_finite = np.all(np.isfinite(alternate), axis=-1)
        alternate_nan = np.all(np.isnan(alternate), axis=-1)
        if np.any(alternate_defined & (~mixed_phase | ~alternate_finite)):
            raise PipelineStateError(
                "alternate roots must be finite diagnostics of mixed roots"
            )
        if np.any(~alternate_defined & ~alternate_nan):
            raise PipelineStateError("undefined alternate roots must be NaN")
        if np.any(alternate_defined & ~self.phase_multivalued):
            raise PipelineStateError(
                "retained alternate roots require the multivalued flag"
            )
        if np.any(self.phase_multivalued != (alternate_defined | jacobian_selected)):
            raise PipelineStateError(
                "multivalued roots require alternate-root or Jacobian provenance"
            )

        vector_arrays = (
            self.source_coordinates,
            self.achieved_targets,
            self.residual,
            self.scaled_residual,
        )
        for array in vector_arrays:
            finite_rows = np.all(np.isfinite(array), axis=-1)
            nan_rows = np.all(np.isnan(array), axis=-1)
            if np.any(usable & ~finite_rows):
                raise PipelineStateError(
                    "usable states require finite dense numerical vectors"
                )
            if np.any(~usable & ~nan_rows):
                raise PipelineStateError(
                    "states without a usable point must retain NaN numerical vectors"
                )
        for array in (self.scaled_residual_norm_inf, self.scaled_distance_l2):
            if np.any(usable & (~np.isfinite(array) | (array < 0.0))):
                raise PipelineStateError(
                    "usable states require finite non-negative residual norms"
                )
            if np.any(~usable & ~np.isnan(array)):
                raise PipelineStateError(
                    "states without a usable point must retain NaN residual norms"
                )

        expected_auxiliary_defined = (
            usable[..., np.newaxis] & np.isfinite(self.auxiliary_values)
        )
        if not np.array_equal(self.auxiliary_defined, expected_auxiliary_defined):
            raise PipelineStateError(
                "auxiliary_defined must identify finite values at usable states"
            )
        if np.any(~self.auxiliary_defined & ~np.isnan(self.auxiliary_values)):
            raise PipelineStateError("undefined auxiliary values must be NaN")

        if self.plan.grid.space is GridSpace.TARGET:
            if self.achieved_tilde_values.shape[-1] != 0 or np.any(
                self.achieved_tilde_defined
            ):
                raise PipelineStateError(
                    "target-space results cannot store achieved tilde columns"
                )
        else:
            tilde_finite = np.all(np.isfinite(self.achieved_tilde_values), axis=-1)
            tilde_nan = np.all(np.isnan(self.achieved_tilde_values), axis=-1)
            if np.any(self.achieved_tilde_defined & (~usable | ~tilde_finite)):
                raise PipelineStateError(
                    "defined achieved tilde rows must be finite usable states"
                )
            if np.any(~self.achieved_tilde_defined & ~tilde_nan):
                raise PipelineStateError("undefined achieved tilde rows must be NaN")

        self._validate_residual_columns(usable)

    def _validate_residual_columns(self, usable: np.ndarray) -> None:
        if not np.any(usable):
            return
        flat_usable = usable.reshape(-1)
        flat_achieved = self.achieved_targets.reshape(
            self.grid.size, self.grid.dimension
        )
        flat_residual = self.residual.reshape(
            self.grid.size, self.grid.dimension
        )
        flat_scaled = self.scaled_residual.reshape(
            self.grid.size, self.grid.dimension
        )
        flat_inf = self.scaled_residual_norm_inf.reshape(-1)
        flat_l2 = self.scaled_distance_l2.reshape(-1)
        rtol = 64.0 * np.finfo(float).eps
        atol = 64.0 * np.nextafter(0.0, 1.0)
        for chunk in self.grid.iter_chunks(self.plan.config.execution.chunk_size):
            local_usable = flat_usable[chunk.start : chunk.stop]
            if not np.any(local_usable):
                continue
            if np.any(local_usable & ~chunk.transform_valid):
                raise PipelineStateError(
                    "a usable state cannot have an invalid requested target transform"
                )
            achieved = flat_achieved[chunk.start : chunk.stop][local_usable]
            expected_residual = (
                achieved - chunk.canonical_targets[local_usable]
            )
            residual = flat_residual[chunk.start : chunk.stop][local_usable]
            if not np.allclose(
                residual, expected_residual, rtol=rtol, atol=atol
            ):
                raise PipelineStateError(
                    "residual must equal achieved target minus requested target"
                )
            expected_scaled = expected_residual / self.plan.residual_scales
            scaled = flat_scaled[chunk.start : chunk.stop][local_usable]
            if not np.allclose(scaled, expected_scaled, rtol=rtol, atol=atol):
                raise PipelineStateError(
                    "scaled_residual is inconsistent with residual_scales"
                )
            expected_inf = np.max(np.abs(expected_scaled), axis=1)
            expected_l2 = np.hypot.reduce(expected_scaled, axis=1)
            actual_inf = flat_inf[chunk.start : chunk.stop][local_usable]
            actual_l2 = flat_l2[chunk.start : chunk.stop][local_usable]
            if not np.allclose(actual_inf, expected_inf, rtol=rtol, atol=atol):
                raise PipelineStateError(
                    "scaled_residual_norm_inf is inconsistent with residual"
                )
            if not np.allclose(actual_l2, expected_l2, rtol=rtol, atol=atol):
                raise PipelineStateError(
                    "scaled_distance_l2 is inconsistent with residual"
                )


_CHUNK_ARRAY_FIELD_NAMES = (
    "source_coordinates",
    "achieved_targets",
    "residual",
    "scaled_residual",
    "scaled_residual_norm_inf",
    "scaled_distance_l2",
    "auxiliary_values",
    "auxiliary_defined",
    "achieved_tilde_values",
    "achieved_tilde_defined",
    "disposition_codes",
    "route_codes",
    "prescreen_codes",
    "solve_status_codes",
    "fast_status_codes",
    "proof_status_codes",
    "projection_quality_codes",
    "initial_guess_source_codes",
    "phase_codes",
    "phase_route_codes",
    "phase_selection_reason_codes",
    "mixed_candidate",
    "exact_mixed_candidate",
    "mixed_candidate_entry_count",
    "mixed_verified_root_count",
    "phase_boundary",
    "phase_multivalued",
    "line_mixture_fraction",
    "line_mixture_fraction_defined",
    "alternate_single_phase_source",
    "alternate_single_phase_source_defined",
)


@dataclass
class _PipelineChunkResult:
    """Picklable column block returned by one serial or process worker."""

    start: int
    stop: int
    source_coordinates: np.ndarray
    achieved_targets: np.ndarray
    residual: np.ndarray
    scaled_residual: np.ndarray
    scaled_residual_norm_inf: np.ndarray
    scaled_distance_l2: np.ndarray
    auxiliary_values: np.ndarray
    auxiliary_defined: np.ndarray
    achieved_tilde_values: np.ndarray
    achieved_tilde_defined: np.ndarray
    disposition_codes: np.ndarray
    route_codes: np.ndarray
    prescreen_codes: np.ndarray
    solve_status_codes: np.ndarray
    fast_status_codes: np.ndarray
    proof_status_codes: np.ndarray
    projection_quality_codes: np.ndarray
    initial_guess_source_codes: np.ndarray
    phase_codes: np.ndarray
    phase_route_codes: np.ndarray
    phase_selection_reason_codes: np.ndarray
    mixed_candidate: np.ndarray
    exact_mixed_candidate: np.ndarray
    mixed_candidate_entry_count: np.ndarray
    mixed_verified_root_count: np.ndarray
    phase_boundary: np.ndarray
    phase_multivalued: np.ndarray
    line_mixture_fraction: np.ndarray
    line_mixture_fraction_defined: np.ndarray
    alternate_single_phase_source: np.ndarray
    alternate_single_phase_source_defined: np.ndarray
    counters: Dict[str, int]
    timing_seconds: Dict[str, float]
    error_records: Tuple[Dict[str, Any], ...] = ()
    projection_failure_records: Tuple[Dict[str, Any], ...] = ()
    worker_setup_seconds: float = 0.0
    worker_peak_rss_bytes: int = 0

    @classmethod
    def allocate(
        cls,
        start: int,
        stop: int,
        dimension: int,
        auxiliary_count: int,
        *,
        tilde: bool,
    ) -> "_PipelineChunkResult":
        count = stop - start
        vector_shape = (count, dimension)
        tilde_columns = 4 if tilde else 0
        return cls(
            start=start,
            stop=stop,
            source_coordinates=np.full(vector_shape, np.nan, dtype=float),
            achieved_targets=np.full(vector_shape, np.nan, dtype=float),
            residual=np.full(vector_shape, np.nan, dtype=float),
            scaled_residual=np.full(vector_shape, np.nan, dtype=float),
            scaled_residual_norm_inf=np.full(count, np.nan, dtype=float),
            scaled_distance_l2=np.full(count, np.nan, dtype=float),
            auxiliary_values=np.full(
                (count, auxiliary_count), np.nan, dtype=float
            ),
            auxiliary_defined=np.zeros((count, auxiliary_count), dtype=bool),
            achieved_tilde_values=np.full(
                (count, tilde_columns), np.nan, dtype=float
            ),
            achieved_tilde_defined=np.zeros(count, dtype=bool),
            disposition_codes=np.zeros(count, dtype=np.uint8),
            route_codes=np.zeros(count, dtype=np.uint8),
            prescreen_codes=np.zeros(count, dtype=np.uint8),
            solve_status_codes=np.zeros(count, dtype=np.uint8),
            fast_status_codes=np.zeros(count, dtype=np.uint8),
            proof_status_codes=np.zeros(count, dtype=np.uint8),
            projection_quality_codes=np.zeros(count, dtype=np.uint8),
            initial_guess_source_codes=np.zeros(count, dtype=np.uint8),
            phase_codes=np.full(
                count,
                _enum_code(PhaseLabel.NOT_APPLICABLE, _PHASE_CODES),
                dtype=np.uint8,
            ),
            phase_route_codes=np.zeros(count, dtype=np.uint8),
            phase_selection_reason_codes=np.full(
                count,
                _enum_code(
                    PhaseRootSelectionReason.NOT_APPLICABLE,
                    _PHASE_SELECTION_REASON_CODES,
                ),
                dtype=np.uint8,
            ),
            mixed_candidate=np.zeros(count, dtype=bool),
            exact_mixed_candidate=np.zeros(count, dtype=bool),
            mixed_candidate_entry_count=np.zeros(count, dtype=np.uint32),
            mixed_verified_root_count=np.zeros(count, dtype=np.uint8),
            phase_boundary=np.zeros(count, dtype=bool),
            phase_multivalued=np.zeros(count, dtype=bool),
            line_mixture_fraction=np.full(count, np.nan, dtype=float),
            line_mixture_fraction_defined=np.zeros(count, dtype=bool),
            alternate_single_phase_source=np.full(
                vector_shape, np.nan, dtype=float
            ),
            alternate_single_phase_source_defined=np.zeros(count, dtype=bool),
            counters={},
            timing_seconds={},
        )


@dataclass(frozen=True)
class _WorkerInitialization:
    metadata_path: str
    data_path: str
    table_fingerprint: str
    pipeline_configuration: Dict[str, Any]
    reachability_configuration: Dict[str, Any]
    first_order_contract: Dict[str, Any]
    shared_state_manifest: Optional[str] = None


@dataclass(frozen=True)
class _FastWork:
    flat_index: int
    local_index: int
    target: np.ndarray
    initial_guess: Optional[np.ndarray] = None
    initial_guess_source: Optional[InitialGuessSource] = None


@dataclass(frozen=True)
class _FallbackWork:
    fast: _FastWork
    fallback_guess: np.ndarray
    fallback_guess_source: InitialGuessSource
    fast_best_candidate: Optional[np.ndarray]


@dataclass(frozen=True)
class _ProjectionWork:
    flat_index: int
    local_index: int
    target: np.ndarray
    success_route: PointRoute
    failure_route: PointRoute
    fast_best_candidate: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _CandidateCenterSeed:
    """Deterministic best center from one complete routing-cell row."""

    point: np.ndarray
    cell_flat_index: int
    candidate_count: int
    scaled_residual_norm_inf: float
    scoring_seconds: float

    def __post_init__(self) -> None:
        point = np.asarray(self.point, dtype=float)
        if point.ndim != 1 or not np.all(np.isfinite(point)):
            raise PipelineStateError(
                "candidate-center seed point must be a finite vector"
            )
        for name in ("cell_flat_index", "candidate_count"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise PipelineStateError("%s must be an integer" % name)
            converted = int(value)
            if converted < 0 or (name == "candidate_count" and converted == 0):
                raise PipelineStateError("%s has an invalid value" % name)
            object.__setattr__(self, name, converted)
        for name in ("scaled_residual_norm_inf", "scoring_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise PipelineStateError("%s must be finite and non-negative" % name)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "point", _immutable_array(point))

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "selected_cell_flat_index": self.cell_flat_index,
            "scaled_residual_norm_inf": self.scaled_residual_norm_inf,
            "scoring_seconds": self.scoring_seconds,
            "metric": "scaled_residual_infinity_norm",
            "tie_break": "smallest_global_c_order_cell_id",
        }


@dataclass(frozen=True)
class _SeedSelection:
    """One selected source point; selection never represents retry stages."""

    point: np.ndarray
    source: InitialGuessSource
    direction: Optional[str] = None
    candidate_center: Optional[_CandidateCenterSeed] = None

    def __post_init__(self) -> None:
        point = np.asarray(self.point, dtype=float)
        if point.ndim != 1 or not np.all(np.isfinite(point)):
            raise PipelineStateError("selected seed must be a finite vector")
        if not isinstance(self.source, InitialGuessSource):
            raise PipelineStateError("selected seed source is invalid")
        if self.source is InitialGuessSource.DIRECT_NEIGHBOR:
            if not isinstance(self.direction, str) or not self.direction:
                raise PipelineStateError(
                    "direct-neighbor selection requires a direction"
                )
        elif self.direction is not None:
            raise PipelineStateError(
                "only direct-neighbor selection can carry a direction"
            )
        if self.source is InitialGuessSource.CONTAINING_CELL:
            if not isinstance(self.candidate_center, _CandidateCenterSeed):
                raise PipelineStateError(
                    "containing-cell selection requires center diagnostics"
                )
            if not np.array_equal(point, self.candidate_center.point):
                raise PipelineStateError(
                    "selected point and candidate-center diagnostics disagree"
                )
        elif self.candidate_center is not None:
            raise PipelineStateError(
                "non-cell seed cannot carry candidate-center diagnostics"
            )
        object.__setattr__(self, "point", _immutable_array(point))


class _CoarseSeedIndex:
    """Small deterministic catalog of solved reduced inverse-grid points."""

    def __init__(
        self,
        grid_shape: Sequence[int],
        flat_indices: Any,
        source_coordinates: Any,
    ) -> None:
        self._grid_shape = tuple(int(size) for size in grid_shape)
        indices = np.asarray(flat_indices, dtype=np.int64)
        sources = np.asarray(source_coordinates, dtype=float)
        dimension = len(self._grid_shape)
        if (
            indices.ndim != 1
            or sources.shape != (indices.size, dimension)
            or np.any(indices < 0)
            or np.any(indices >= math.prod(self._grid_shape))
            or np.any(~np.isfinite(sources))
        ):
            raise PipelineStateError("coarse seed catalog is invalid")
        order = np.argsort(indices, kind="stable")
        indices = indices[order]
        sources = sources[order]
        if indices.size and np.any(np.diff(indices) <= 0):
            raise PipelineStateError("coarse seed flat indices must be unique")
        if indices.size:
            multi = np.column_stack(
                np.unravel_index(indices, self._grid_shape, order="C")
            ).astype(float, copy=False)
        else:
            multi = np.empty((0, dimension), dtype=float)
        spans = np.maximum(np.asarray(self._grid_shape, dtype=float) - 1.0, 1.0)
        self._flat_indices = _immutable_array(indices, dtype=np.int64)
        self._multi_indices = _immutable_array(multi / spans)
        self._source_coordinates = _immutable_array(sources)
        self._spans = _immutable_array(spans)

    @property
    def size(self) -> int:
        return int(self._flat_indices.size)

    def nearest(self, flat_index: int) -> Optional[np.ndarray]:
        if self.size == 0:
            return None
        checked = int(flat_index)
        if checked < 0 or checked >= math.prod(self._grid_shape):
            raise PipelineStateError("fine-grid index is outside the coarse catalog")
        query = np.asarray(
            np.unravel_index(checked, self._grid_shape, order="C"),
            dtype=float,
        ) / self._spans
        distances = np.hypot.reduce(self._multi_indices - query, axis=1)
        selected = int(np.argmin(distances))
        return np.asarray(self._source_coordinates[selected], dtype=float).copy()


class EoSInversionPipeline:
    """Execute the deterministic complete-table inversion pipeline.

    Serial and process modes share the same self-contained chunk engine.  The
    executor intentionally accepts only the full validated source cell grid:
    a nontrivial active-cell mask is rejected because the current FAST solver
    is bounded by the global source rectangle and cannot yet be constrained
    to a disconnected union of cells.  Reachability, solving, and projection
    therefore cannot silently disagree about the admissible domain.
    """

    def __init__(
        self,
        table: ForwardTable,
        config: PipelineConfig,
        *,
        active_cell_mask: Optional[Any] = None,
        reachability_config: Optional[Any] = None,
    ) -> None:
        if not isinstance(table, ForwardTable):
            raise TypeError("table must be a ForwardTable")
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig")
        planning_start = time.perf_counter()
        self._table = table
        self._plan = PipelinePlan.from_table(table, config)
        self._table_fingerprint = _validated_table_fingerprint(table)

        from .reachability import ReachabilityConfig

        if reachability_config is None:
            reachability_config = ReachabilityConfig()
        if not isinstance(reachability_config, ReachabilityConfig):
            raise TypeError("reachability_config must be a ReachabilityConfig")
        if (
            reachability_config.method == "hierarchical"
            and len(table.coordinate_names) != 4
        ):
            raise PipelineConfigurationError(
                "hierarchical reachability is initially supported only for 4D tables"
            )
        self._reachability_config = reachability_config
        self._validate_active_cell_mask(active_cell_mask)
        self._planning_seconds = float(time.perf_counter() - planning_start)
        self._prepared = False
        self._target_interpolator: Optional[MultilinearInterpolator] = None
        self._auxiliary_interpolator: Optional[MultilinearInterpolator] = None
        self._reachability: Optional[Any] = None
        self._solver: Optional[EoSRootSolver] = None
        self._phase_solver: Optional[Any] = None
        self._mixed_phase_candidates: Optional[Any] = None
        self._first_order_manifest = self._first_order_contract()
        self._projector: Optional[EoSProjector] = None
        self._coarse_seeds: Optional[_CoarseSeedIndex] = None
        self._coarse_seed_report: Dict[str, Any] = {
            "enabled": False,
            "requested_points": 0,
            "transform_valid": 0,
            "reachability_candidates": 0,
            "solved": 0,
            "fast_failures": 0,
            "fast_errors": 0,
        }
        self._setup_timing_seconds: Dict[str, float] = {}
        self._solved_seed_cache: Dict[int, np.ndarray] = {}

    def _validate_active_cell_mask(self, value: Optional[Any]) -> None:
        if value is None:
            return
        expected = tuple(size - 1 for size in self._table.shape)
        mask = np.asarray(value)
        if mask.shape != expected or mask.dtype.kind != "b":
            raise PipelineConfigurationError(
                "active_cell_mask must be a boolean array with shape %r"
                % (expected,)
            )
        if not np.any(mask):
            raise PipelineConfigurationError("the active-cell domain cannot be empty")
        if not np.all(mask):
            raise PipelineConfigurationError(
                "the serial root solver cannot yet enforce a restricted "
                "active-cell union; Phase 4 rejects nontrivial masks rather "
                "than allowing FAST/FALLBACK to search excluded cells"
            )

    def _first_order_contract(self) -> Dict[str, Any]:
        """Return the deterministic transition contract shared by all workers."""

        from .phase_solver import PhaseSolverConfig
        from .reachability import BinodalSubdivisionConfig
        from .transition import RegionSConfig

        specification = self._table.first_order_transition
        coexistence = self._table.coexistence_table
        phase_config = PhaseSolverConfig()
        contract: Dict[str, Any] = {
            "configured": specification is not None,
            "representation": (
                None if specification is None else specification.representation
            ),
            "worker_reconstruction": "validated_table_snapshot",
            "table_snapshot_fingerprint": self._table_fingerprint,
            "branch_precedence": "verified_mixed_before_distinct_ordinary",
            "phase_solver": {
                item.name: getattr(phase_config, item.name)
                for item in dataclass_fields(phase_config)
            },
        }
        if specification is None:
            return contract
        if coexistence is None:
            raise PipelineStateError(
                "validated first-order metadata lack a coexistence snapshot"
            )
        muB = np.asarray(coexistence.field("muB"), dtype=float)
        contract.update(
            {
                "sidecar_fingerprint": coexistence.fingerprint,
                "muB_bounds_GeV": [float(muB[0]), float(muB[-1])],
                "no_extrapolation": True,
            }
        )
        if specification.representation == "line":
            contract.update(
                {
                    "region_s_configuration": RegionSConfig().to_dict(),
                    "line_fraction_semantics": (
                        "lambda=0 hadronic endpoint; lambda=1 qgp endpoint"
                    ),
                }
            )
        elif specification.representation == "binodals":
            subdivision = BinodalSubdivisionConfig()
            contract["binodal_subdivision"] = {
                item.name: getattr(subdivision, item.name)
                for item in dataclass_fields(subdivision)
            }
        return contract

    def plan(self) -> PipelinePlan:
        """Return the fully resolved, immutable run plan."""

        return self._plan

    @property
    def resolved_plan(self) -> PipelinePlan:
        """Property alias for callers that prefer attribute-style planning."""

        return self._plan

    @property
    def reachability_config(self) -> Any:
        return self._reachability_config

    def _uses_complete_ordinary_localization(self) -> bool:
        """Return whether the generic ordinary localization flow applies.

        Phase 8 covers ordinary dimensions 1--4.  Exhaustive reachability
        supplies complete rows in dimensions 1--3 and optional exhaustive 4D;
        accepted hierarchical 4D runs supply the same contract at depth 7.
        First-order LINE/BINODALS routing remains deliberately isolated.
        """

        return bool(
            self._table.first_order_transition is None
            and 1 <= self._plan.grid.dimension <= 4
        )

    def _generic_initial_guess(self) -> np.ndarray:
        assert self._target_interpolator is not None
        bounds = self._target_interpolator.bounds
        midpoint = bounds[:, 0] / 2.0 + bounds[:, 1] / 2.0
        if not np.all(np.isfinite(midpoint)):
            raise PipelineStateError("source-domain midpoint is not finite")
        return np.asarray(midpoint, dtype=float)

    @staticmethod
    def _source_cell_center(
        interpolator: MultilinearInterpolator, cell_flat_index: int
    ) -> np.ndarray:
        cell_shape = interpolator.cell_shape
        checked = int(cell_flat_index)
        if checked < 0 or checked >= math.prod(cell_shape):
            raise PipelineStateError("reachability witness cell is invalid")
        multi = np.unravel_index(checked, cell_shape, order="C")
        return np.asarray(
            [
                axis[index] / 2.0 + axis[index + 1] / 2.0
                for axis, index in zip(interpolator.axes, multi)
            ],
            dtype=float,
        )

    @staticmethod
    def _source_cell_centers(
        interpolator: MultilinearInterpolator, cell_flat_indices: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorize stable centers for sorted global C-order cell IDs."""

        raw = np.asarray(cell_flat_indices)
        if raw.dtype.kind not in "iu" or raw.dtype.kind == "b" or raw.ndim != 1:
            raise PipelineStateError(
                "candidate cell flat indices must be an integer vector"
            )
        cells = np.asarray(raw, dtype=np.int64)
        if cells.size == 0:
            raise PipelineStateError(
                "candidate-center selection requires at least one cell"
            )
        if np.any(cells < 0) or np.any(cells >= math.prod(interpolator.cell_shape)):
            raise PipelineStateError("candidate cell is outside the source lattice")
        if cells.size > 1 and np.any(np.diff(cells) <= 0):
            raise PipelineStateError(
                "candidate cell IDs must be unique and strictly increasing"
            )
        cell_multi = np.column_stack(
            np.unravel_index(cells, interpolator.cell_shape, order="C")
        )
        lower, upper = interpolator.cell_coordinate_bounds(cell_multi)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        widths = upper - lower
        if (
            lower.shape != (cells.size, interpolator.dimension)
            or upper.shape != lower.shape
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(widths <= 0.0)
        ):
            raise PipelineStateError(
                "candidate cells do not have finite positive coordinate widths"
            )
        centers = lower / 2.0 + upper / 2.0
        if not np.all(np.isfinite(centers)):
            raise PipelineStateError("candidate cell center is not finite")
        if np.any(centers < lower) or np.any(centers > upper):
            raise PipelineStateError("candidate cell center lies outside its cell")
        return centers, cells

    def _localized_candidate_cells(
        self,
        target: Any,
        localization: Any,
        local_target_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate proof/query identity and return one complete CSR row."""

        from .reachability import ReachabilityLocalization

        if self._target_interpolator is None or self._reachability is None:
            raise PipelineStateError(
                "candidate-center selection requires a prepared pipeline"
            )
        if not isinstance(localization, ReachabilityLocalization):
            raise PipelineStateError(
                "candidate-center selection requires ReachabilityLocalization"
            )
        localization.require_complete_for(self._reachability.provenance)
        if isinstance(local_target_index, (bool, np.bool_)) or not isinstance(
            local_target_index, (int, np.integer)
        ):
            raise PipelineStateError("local_target_index must be an integer")
        local = int(local_target_index)
        if local < 0 or local >= localization.target_count:
            raise PipelineStateError(
                "local_target_index is outside the complete localization"
            )
        checked_target = np.asarray(target, dtype=float)
        if (
            checked_target.shape != (self._plan.grid.dimension,)
            or not np.all(np.isfinite(checked_target))
        ):
            raise PipelineStateError(
                "candidate-center target must be a finite canonical vector"
            )
        stored_target = localization.result.targets.reshape(
            -1, self._plan.grid.dimension
        )[local]
        if not np.array_equal(checked_target, stored_target):
            raise PipelineStateError(
                "candidate-center target does not match its localization row"
            )
        if not bool(localization.result.candidate_mask.reshape(-1)[local]):
            raise PipelineStateError(
                "candidate-center selection requires a reachability candidate"
            )
        return checked_target, localization.cells_for(local)

    def _candidate_center_seed(
        self,
        target: Any,
        localization: Any,
        local_target_index: int,
    ) -> _CandidateCenterSeed:
        """Score all complete candidate-cell centers in one vectorized pass."""

        checked_target, candidate_cells = self._localized_candidate_cells(
            target,
            localization,
            local_target_index,
        )

        started = time.perf_counter()
        centers, cells = self._source_cell_centers(
            self._target_interpolator,
            candidate_cells,
        )
        achieved = np.asarray(
            self._target_interpolator.evaluate(centers), dtype=float
        )
        expected_shape = (cells.size, self._plan.grid.dimension)
        if achieved.shape != expected_shape or not np.all(np.isfinite(achieved)):
            raise PipelineStateError(
                "candidate-center forward evaluation is not finite"
            )
        scaled_residual = (
            achieved - checked_target[np.newaxis, :]
        ) / self._plan.residual_scales[np.newaxis, :]
        norms = np.max(np.abs(scaled_residual), axis=1)
        if norms.shape != (cells.size,) or not np.all(np.isfinite(norms)):
            raise PipelineStateError(
                "candidate-center scaled residual is not finite"
            )
        minimum = float(np.min(norms))
        # ``cells`` is strictly increasing, so the first exact metric tie is
        # the required smallest global C-order cell ID.
        selected = int(np.flatnonzero(norms == minimum)[0])
        elapsed = float(time.perf_counter() - started)
        return _CandidateCenterSeed(
            point=centers[selected],
            cell_flat_index=int(cells[selected]),
            candidate_count=int(cells.size),
            scaled_residual_norm_inf=minimum,
            scoring_seconds=elapsed,
        )

    def _select_localized_seed(
        self,
        flat_index: int,
        target: Any,
        localization: Optional[Any],
        local_target_index: int,
        *,
        use_reachability_seed: bool = True,
    ) -> _SeedSelection:
        """Select one ordinary seed; sources are alternatives, not retries."""

        if not isinstance(use_reachability_seed, (bool, np.bool_)):
            raise PipelineStateError("use_reachability_seed must be boolean")
        if self._table.first_order_transition is not None:
            raise PipelineStateError(
                "ordinary localized seeding is disabled for first-order tables"
            )
        if localization is not None:
            # A direct-neighbor override still belongs to a localized path;
            # validate its complete proof domain before selecting any seed.
            self._localized_candidate_cells(
                target,
                localization,
                local_target_index,
            )
        direct, direction = self._direct_neighbor_seed(flat_index)
        if direct is not None and self._localized_direct_seed_is_valid(direct):
            return _SeedSelection(
                point=direct,
                source=InitialGuessSource.DIRECT_NEIGHBOR,
                direction=direction,
            )
        if bool(use_reachability_seed) and localization is not None:
            center = self._candidate_center_seed(
                target,
                localization,
                local_target_index,
            )
            return _SeedSelection(
                point=center.point,
                source=InitialGuessSource.CONTAINING_CELL,
                candidate_center=center,
            )
        if self._coarse_seeds is not None:
            coarse = self._coarse_seeds.nearest(flat_index)
            if coarse is not None:
                return _SeedSelection(
                    point=coarse,
                    source=InitialGuessSource.COARSE_LEVEL,
                )
        return _SeedSelection(
            point=self._generic_initial_guess(),
            source=InitialGuessSource.DOMAIN_MIDPOINT,
        )

    def _localized_direct_seed_is_valid(self, point: Any) -> bool:
        """Reject boundary continuation seeds that can pin bounded FAST.

        A solved adjacent point on a source-domain face is valid as a root for
        its own target, but it is not a robust starting point for a different
        target: bounded least-squares can retain the active constraint and
        miss an otherwise trivial interior direction.  Candidate-cell centers
        are the deterministic alternative on those faces.  Interior direct
        neighbors remain opportunistic overrides.
        """

        if self._target_interpolator is None:
            raise PipelineStateError(
                "localized direct-seed validation requires a prepared pipeline"
            )
        checked = np.asarray(point, dtype=float)
        if not self._solver_seed_is_usable(
            self._target_interpolator, checked
        ):
            return False
        bounds = np.asarray(self._target_interpolator.bounds, dtype=float)
        return bool(
            np.all(checked > bounds[:, 0])
            and np.all(checked < bounds[:, 1])
        )

    def _solve_selected_fast(
        self, target: Any, selection: _SeedSelection
    ) -> InversionResult:
        """Invoke FAST exactly once for an already selected ordinary seed."""

        if self._solver is None:
            raise PipelineStateError("FAST selection requires a prepared solver")
        if not isinstance(selection, _SeedSelection):
            raise PipelineStateError("FAST selection has an invalid type")
        checked_target = np.asarray(target, dtype=float)
        if checked_target.shape != (self._plan.grid.dimension,) or not np.all(
            np.isfinite(checked_target)
        ):
            raise PipelineStateError("FAST target must be a finite vector")
        return self._solver.solve(
            checked_target,
            selection.point,
            strategy=SolveStrategy.FAST,
        )

    def _coarse_grid_rows(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid = self._plan.grid
        configuration = self._plan.config.coarse_seed
        per_axis = []
        for size in grid.shape:
            count = min(configuration.points_per_axis, size)
            selected = np.unique(
                np.rint(np.linspace(0, size - 1, count)).astype(np.int64)
            )
            per_axis.append(selected)
        coarse_multi = np.asarray(list(product(*per_axis)), dtype=np.int64)
        if coarse_multi.shape[0] > configuration.maximum_points:
            raise PipelineConfigurationError(
                "coarse seed grid contains %d points, exceeding maximum_points=%d"
                % (coarse_multi.shape[0], configuration.maximum_points)
            )
        flat = np.ravel_multi_index(
            tuple(coarse_multi[:, axis] for axis in range(grid.dimension)),
            grid.shape,
            order="C",
        ).astype(np.int64, copy=False)
        requested = np.column_stack(
            [
                grid.axes[axis][coarse_multi[:, axis]]
                for axis in range(grid.dimension)
            ]
        )
        if grid.space is GridSpace.TARGET:
            canonical = requested.copy()
            valid = np.ones(flat.size, dtype=bool)
        else:
            canonical, valid, _ = grid._transform_tilde(requested)
        return flat, canonical, valid

    def _build_coarse_seed_index(
        self,
        interpolator: MultilinearInterpolator,
        reachability: Any,
        solver: EoSRootSolver,
        *,
        ordinary_smooth: bool,
    ) -> Tuple[_CoarseSeedIndex, Dict[str, Any]]:
        configuration = self._plan.config.coarse_seed
        empty = _CoarseSeedIndex(
            self._plan.grid.shape,
            np.empty(0, dtype=np.int64),
            np.empty((0, self._plan.grid.dimension), dtype=float),
        )
        if not configuration.enabled or not ordinary_smooth:
            return empty, {
                "enabled": False,
                "disabled_reason": (
                    "configuration"
                    if not configuration.enabled
                    else "first_order_branch_aware_path"
                ),
                "requested_points": 0,
                "transform_valid": 0,
                "reachability_candidates": 0,
                "solved": 0,
                "fast_failures": 0,
                "fast_errors": 0,
            }
        flat, targets, valid = self._coarse_grid_rows()
        classified = reachability.classify(targets, valid_mask=valid)
        candidate = np.asarray(classified.candidate_mask, dtype=bool).reshape(-1)
        witnesses = np.asarray(
            classified.routing_witness_flat, dtype=np.int64
        ).reshape(-1)
        solved_flat = []
        solved_source = []
        fast_failures = 0
        fast_errors = 0
        for position in np.flatnonzero(candidate):
            witness = int(witnesses[position])
            if witness < 0:
                raise PipelineStateError(
                    "coarse reachability candidate lacks a routing witness"
                )
            initial = self._source_cell_center(interpolator, witness)
            try:
                result = solver.solve(
                    targets[position], initial, strategy=SolveStrategy.FAST
                )
            except Exception:
                # A coarse seed is an optional convergence aid. Failure to
                # construct one must fall through to the generic fine-grid
                # seed rather than aborting table preparation.
                fast_errors += 1
                continue
            if result.status is SolveStatus.CONVERGED_FAST:
                if result.solution is None:
                    raise PipelineStateError(
                        "coarse FAST success lacks a source solution"
                    )
                solved_flat.append(int(flat[position]))
                solved_source.append(np.asarray(result.solution, dtype=float))
            else:
                fast_failures += 1
        sources = (
            np.asarray(solved_source, dtype=float)
            if solved_source
            else np.empty((0, self._plan.grid.dimension), dtype=float)
        )
        index = _CoarseSeedIndex(
            self._plan.grid.shape,
            np.asarray(solved_flat, dtype=np.int64),
            sources,
        )
        return index, {
            "enabled": True,
            "construction": "regular_inverse_subgrid_fast_from_routing_witness_cell_center",
            "points_per_axis": configuration.points_per_axis,
            "requested_points": int(flat.size),
            "transform_valid": int(np.count_nonzero(valid)),
            "reachability_candidates": int(np.count_nonzero(candidate)),
            "solved": index.size,
            "fast_failures": int(fast_failures),
            "fast_errors": int(fast_errors),
            "selection_metric": "normalized_inverse_grid_index_l2",
            "tie_break": "smallest_c_order_coarse_flat_index",
        }

    def _direct_neighbor_seed(
        self, flat_index: int
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        grid = self._plan.grid
        multi = list(np.unravel_index(int(flat_index), grid.shape, order="C"))
        for axis in range(grid.dimension):
            for delta, suffix in ((-1, "minus"), (1, "plus")):
                neighbor = multi.copy()
                neighbor[axis] += delta
                if neighbor[axis] < 0 or neighbor[axis] >= grid.shape[axis]:
                    continue
                neighbor_flat = int(
                    np.ravel_multi_index(tuple(neighbor), grid.shape, order="C")
                )
                source = self._solved_seed_cache.get(neighbor_flat)
                if source is not None:
                    return source.copy(), "axis%d_%s" % (axis, suffix)
        return None, None

    def _select_seed(
        self, flat_index: int
    ) -> Tuple[np.ndarray, InitialGuessSource, Optional[str]]:
        # First-order solvers retain their explicit phase/branch routing and do
        # not consume ordinary continuation/coarse seeds.
        if self._table.first_order_transition is None:
            direct, direction = self._direct_neighbor_seed(flat_index)
            if direct is not None:
                return direct, InitialGuessSource.DIRECT_NEIGHBOR, direction
            if self._coarse_seeds is not None:
                coarse = self._coarse_seeds.nearest(flat_index)
                if coarse is not None:
                    return coarse, InitialGuessSource.COARSE_LEVEL, None
        return (
            self._generic_initial_guess(),
            InitialGuessSource.DOMAIN_MIDPOINT,
            None,
        )

    def _register_solved_seed(
        self,
        flat_index: int,
        source: Any,
        disposition: PointDisposition = PointDisposition.SOLVED,
    ) -> bool:
        if disposition is not PointDisposition.SOLVED:
            return False
        assert self._target_interpolator is not None
        point = np.asarray(source, dtype=float)
        if not self._solver_seed_is_usable(self._target_interpolator, point):
            raise PipelineStateError("solved continuation seed is invalid")
        self._solved_seed_cache[int(flat_index)] = point.copy()
        return True

    def _seed_work(
        self, work: _FastWork, counters: Dict[str, int]
    ) -> _FastWork:
        point, source, direction = self._select_seed(work.flat_index)
        if source is InitialGuessSource.DIRECT_NEIGHBOR:
            counters["seed_direct_neighbor"] += 1
            if direction is None:
                raise PipelineStateError("direct-neighbor seed lacks a direction")
            counters["seed_direct_neighbor_%s" % direction] += 1
        elif source is InitialGuessSource.COARSE_LEVEL:
            counters["seed_coarse_level"] += 1
        elif source is InitialGuessSource.DOMAIN_MIDPOINT:
            counters["seed_generic"] += 1
        else:
            raise PipelineStateError("production selected an unsupported seed source")
        return _FastWork(
            work.flat_index,
            work.local_index,
            work.target,
            point,
            source,
        )

    @staticmethod
    def _record_seed_selection(
        selection: _SeedSelection, counters: Dict[str, int]
    ) -> None:
        """Update the stable seed counters for one no-retry selection."""

        source = selection.source
        if source is InitialGuessSource.DIRECT_NEIGHBOR:
            counters["seed_direct_neighbor"] += 1
            if selection.direction is None:
                raise PipelineStateError(
                    "direct-neighbor seed lacks a direction"
                )
            counters[
                "seed_direct_neighbor_%s" % selection.direction
            ] += 1
        elif source is InitialGuessSource.CONTAINING_CELL:
            # The stable dense initial-guess codebook already records this
            # source.  Phase 7 intentionally does not change checkpoint
            # counter schemas; persistence versioning belongs to Phase 9.
            return
        elif source is InitialGuessSource.COARSE_LEVEL:
            counters["seed_coarse_level"] += 1
        elif source is InitialGuessSource.DOMAIN_MIDPOINT:
            counters["seed_generic"] += 1
        else:
            raise PipelineStateError(
                "localized production selected an unsupported seed source"
            )

    def _fast_retry_work(self, work: _FastWork) -> Tuple[_FastWork, ...]:
        """Return bounded alternate FAST seeds after the preferred seed fails.

        Direct continuation remains the first attempt.  Retrying the existing
        coarse seed, then the generic midpoint, prevents worker scheduling
        from changing terminal status solely because a local Newton basin was
        missed.  The retry path performs no global source-node or solved-point
        search.
        """

        if work.initial_guess is None or work.initial_guess_source is None:
            raise PipelineStateError("FAST retry requires initialized work")
        alternatives = []
        if (
            work.initial_guess_source is InitialGuessSource.DIRECT_NEIGHBOR
            and self._coarse_seeds is not None
        ):
            coarse = self._coarse_seeds.nearest(work.flat_index)
            if coarse is not None and not np.array_equal(coarse, work.initial_guess):
                alternatives.append(
                    _FastWork(
                        work.flat_index,
                        work.local_index,
                        work.target,
                        coarse,
                        InitialGuessSource.COARSE_LEVEL,
                    )
                )
        if work.initial_guess_source is not InitialGuessSource.DOMAIN_MIDPOINT:
            generic = self._generic_initial_guess()
            prior = [work.initial_guess] + [item.initial_guess for item in alternatives]
            if not any(np.array_equal(generic, point) for point in prior):
                alternatives.append(
                    _FastWork(
                        work.flat_index,
                        work.local_index,
                        work.target,
                        generic,
                        InitialGuessSource.DOMAIN_MIDPOINT,
                    )
                )
        return tuple(alternatives)

    def _prepare(self, *, shared_state_manifest: Optional[Any] = None) -> float:
        if self._prepared:
            return 0.0
        started = time.perf_counter()
        from .phase_solver import PhaseAwareRootSolver, PhaseSolverConfig
        from .reachability import (
            BinodalSubdivisionConfig,
            MixedPhaseCandidateIndex,
            ReachabilityIndex,
        )
        from .transition import (
            BinodalBand,
            TransitionLine,
            build_region_s,
            transition_model_from_table,
        )

        setup_timing: Dict[str, float] = {}
        stage_started = time.perf_counter()
        target_interpolator = MultilinearInterpolator(
            self._table, self._plan.target_names
        )
        setup_timing["target_interpolator"] = float(
            time.perf_counter() - stage_started
        )
        stage_started = time.perf_counter()
        if shared_state_manifest is None:
            reachability = ReachabilityIndex.from_plan(
                target_interpolator,
                self._plan,
                config=self._reachability_config,
            )
        else:
            from .shared_state import load_shared_reachability

            reachability = load_shared_reachability(
                shared_state_manifest,
                target_interpolator,
                self._plan,
                self._reachability_config,
            )
        setup_timing["reachability_index"] = float(
            time.perf_counter() - stage_started
        )
        stage_started = time.perf_counter()
        solver = EoSRootSolver(
            target_interpolator,
            self._plan.config.solver,
            residual_scales=self._plan.residual_scales,
        )
        setup_timing["root_solver"] = float(
            time.perf_counter() - stage_started
        )
        stage_started = time.perf_counter()
        transition_model = transition_model_from_table(self._table)
        phase_solver = None
        mixed_phase_candidates = None
        first_order_manifest: Dict[str, Any] = {
            "configured": transition_model is not None,
            "representation": None,
            "worker_reconstruction": "validated_table_snapshot",
            "table_snapshot_fingerprint": self._table_fingerprint,
        }
        if isinstance(transition_model, TransitionLine):
            linear_auxiliary_fields = tuple(
                name
                for name in self._plan.auxiliary_names
                if self._plan.config.line_auxiliary_policies.get(
                    name,
                    self._plan.config.line_auxiliary_default_policy,
                )
                is LineAuxiliaryPolicy.LINEAR
            )
            strip = build_region_s(
                self._table,
                auxiliary_fields=linear_auxiliary_fields,
            )
            mixed_phase_candidates = MixedPhaseCandidateIndex.for_line(
                strip,
                self._plan.residual_scales,
                solver.config.root_residual_tolerance,
                config=self._reachability_config,
            )
            phase_solver = PhaseAwareRootSolver(
                solver, mixed_phase_candidates, config=PhaseSolverConfig()
            )
            first_order_manifest.update(
                {
                    "representation": "line",
                    "transition": transition_model.diagnostics.to_dict(),
                    "region_s": strip.diagnostics.to_dict(),
                    "mixed_image_count": mixed_phase_candidates.images.image_count,
                    "mixed_image_bytes": mixed_phase_candidates.images.nbytes,
                    "line_fraction_semantics": (
                        "lambda=0 hadronic endpoint; lambda=1 qgp endpoint"
                    ),
                    "linear_auxiliary_endpoint_fields": list(
                        strip.auxiliary_names
                    ),
                    "no_extrapolation": True,
                }
            )
        elif isinstance(transition_model, BinodalBand):
            subdivision = BinodalSubdivisionConfig()
            mixed_phase_candidates = MixedPhaseCandidateIndex.for_binodals(
                target_interpolator,
                transition_model,
                self._plan.residual_scales,
                solver.config.root_residual_tolerance,
                subdivision=subdivision,
                config=self._reachability_config,
            )
            phase_solver = PhaseAwareRootSolver(
                solver, mixed_phase_candidates, config=PhaseSolverConfig()
            )
            first_order_manifest.update(
                {
                    "representation": "binodals",
                    "transition": transition_model.diagnostics.to_dict(),
                    "binodal_subdivision": {
                        item.name: getattr(subdivision, item.name)
                        for item in dataclass_fields(subdivision)
                    },
                    "mixed_image_count": mixed_phase_candidates.images.image_count,
                    "mixed_boundary_image_count": int(
                        np.count_nonzero(
                            mixed_phase_candidates.images.boundary_candidate_mask
                        )
                    ),
                    "mixed_image_bytes": mixed_phase_candidates.images.nbytes,
                    "no_extrapolation": True,
                }
            )
        if phase_solver is not None:
            first_order_manifest["phase_solver"] = {
                item.name: getattr(phase_solver.config, item.name)
                for item in dataclass_fields(phase_solver.config)
            }
        setup_timing["first_order_models"] = float(
            time.perf_counter() - stage_started
        )
        stage_started = time.perf_counter()
        projector = (
            EoSProjector.from_reachability(
                reachability, self._plan.config.projection
            )
            if self._plan.config.projection.enabled
            else None
        )
        auxiliary = (
            MultilinearInterpolator(self._table, self._plan.auxiliary_names)
            if self._plan.auxiliary_names
            else None
        )
        setup_timing["projection_and_auxiliary"] = float(
            time.perf_counter() - stage_started
        )
        stage_started = time.perf_counter()
        coarse_seeds, coarse_report = self._build_coarse_seed_index(
            target_interpolator,
            reachability,
            solver,
            ordinary_smooth=transition_model is None,
        )
        setup_timing["coarse_seed_precomputation"] = float(
            time.perf_counter() - stage_started
        )
        self._target_interpolator = target_interpolator
        self._reachability = reachability
        self._solver = solver
        self._phase_solver = phase_solver
        self._mixed_phase_candidates = mixed_phase_candidates
        self._projector = projector
        self._auxiliary_interpolator = auxiliary
        self._coarse_seeds = coarse_seeds
        self._coarse_seed_report = coarse_report
        self._setup_timing_seconds = setup_timing
        self._prepared = True
        return float(time.perf_counter() - started)

    def _persistent_memory_estimates(
        self, workers_used: int, *, mmap_shared: bool = False
    ) -> Dict[str, Any]:
        """Estimate dense persistent storage without claiming Python overhead."""

        dimension = self._plan.grid.dimension
        node_count = int(math.prod(self._table.shape))
        cell_shape = tuple(size - 1 for size in self._table.shape)
        cell_count = int(math.prod(cell_shape))
        raw = self._table.raw_data
        raw_bytes = 0 if raw is None else int(raw.nbytes)
        field_bytes = int(sum(value.nbytes for value in self._table.fields.values()))
        axis_bytes = int(sum(axis.nbytes for axis in self._table.axes))
        permutation_bytes = int(self._table.grid.row_permutation.nbytes)
        table_bytes = raw_bytes + field_bytes + axis_bytes + permutation_bytes

        # CellTargetEnclosures owns one int64 identifier and four target-space
        # float64 bounds for every active cell, plus two d-vectors.
        enclosure_bytes = int(
            cell_count
            * (
                np.dtype(np.int64).itemsize
                + 4 * dimension * np.dtype(np.float64).itemsize
            )
            + 2 * dimension * np.dtype(np.float64).itemsize
        )
        hierarchy = (
            None
            if self._reachability is None
            else getattr(self._reachability, "hierarchy", None)
        )
        hierarchy_owned_bytes = int(
            0 if hierarchy is None else hierarchy.owned_nbytes
        )
        if (
            hierarchy is None
            and self._reachability_config.method == "hierarchical"
            and dimension == 4
            and self._reachability_config.hierarchy_levels is not None
        ):
            # Two float64 four-component bounds per non-leaf block.  Axis
            # boundary vectors are tiny and intentionally omitted from this
            # dense-array lower bound.
            hierarchy_owned_bytes = int(
                sum(
                    2
                    * dimension
                    * np.dtype(np.float64).itemsize
                    * math.prod(
                        min(1 << (level - 1), cells)
                        for cells in cell_shape
                    )
                    for level in range(
                        1, self._reachability_config.hierarchy_levels + 1
                    )
                    if tuple(
                        min(1 << (level - 1), cells)
                        for cells in cell_shape
                    )
                    != cell_shape
                )
            )
        reachability_bytes = enclosure_bytes + hierarchy_owned_bytes
        projection_index_bytes = 0
        projection_unique_bytes = 0
        if self._plan.config.projection.enabled:
            projection_index_bytes = int(
                cell_count * np.dtype(np.bool_).itemsize
                + node_count * np.dtype(np.bool_).itemsize
                + cell_count
                * (
                    np.dtype(np.int64).itemsize
                    + 2 * dimension * np.dtype(np.float64).itemsize
                )
                + node_count
                * (
                    np.dtype(np.int64).itemsize
                    + dimension * np.dtype(np.float64).itemsize
                )
            )
            shared_cell_bounds = int(
                cell_count * 2 * dimension * np.dtype(np.float64).itemsize
            )
            projection_unique_bytes = max(
                0, projection_index_bytes - shared_cell_bounds
            )
        worker_lower_bound = (
            table_bytes + reachability_bytes + projection_unique_bytes
        )
        worker_count = _nonnegative_integer(workers_used, "workers_used")
        shared_bytes = table_bytes + reachability_bytes if mmap_shared else 0
        private_per_worker = (
            projection_unique_bytes if mmap_shared else worker_lower_bound
        )
        return {
            "interpretation": (
                "dense_persistent_array_lower_bounds; excludes Python/SciPy "
                "objects, allocator overhead, and temporary work arrays"
            ),
            "immutable_storage_backend": (
                "read_only_file_backed_shared"
                if mmap_shared
                else "private_process_arrays"
            ),
            "validated_table_snapshot_bytes": table_bytes,
            "validated_table_components": {
                "raw_input": raw_bytes,
                "canonical_fields": field_bytes,
                "coordinate_axes": axis_bytes,
                "row_permutation": permutation_bytes,
            },
            "reachability_index_bytes": reachability_bytes,
            "reachability_enclosure_bytes": enclosure_bytes,
            "reachability_hierarchy_owned_bytes": hierarchy_owned_bytes,
            "projection_index_standalone_estimate_bytes": projection_index_bytes,
            "projection_incremental_bytes_after_reachability_sharing": (
                projection_unique_bytes
            ),
            "shared_file_backed_lower_bound_bytes": shared_bytes,
            "per_worker_private_lower_bound_bytes": private_per_worker,
            "all_workers_private_lower_bound_bytes": (
                worker_count * private_per_worker
            ),
            "per_worker_lower_bound_bytes": private_per_worker,
            "all_workers_lower_bound_bytes": (
                shared_bytes + worker_count * private_per_worker
                if mmap_shared
                else worker_count * worker_lower_bound
            ),
            "mutable_result_bytes": self._plan.estimated_result_bytes,
            "result_finalization_peak_bytes": (
                self._plan.estimated_peak_result_bytes
            ),
        }

    def _initial_guess(self, target: np.ndarray) -> Tuple[np.ndarray, InitialGuessSource]:
        # Compatibility helper for the private phase-4 reference. Production
        # chunk execution uses _select_seed(flat_index), which applies direct
        # neighbor and coarse-grid precedence first.
        del target
        return self._generic_initial_guess(), InitialGuessSource.DOMAIN_MIDPOINT

    def _configured_initial_guess_source(self) -> InitialGuessSource:
        return InitialGuessSource.DOMAIN_MIDPOINT

    @staticmethod
    def _clear_point(buffer: Any, flat_index: int) -> None:
        size = int(buffer.disposition_codes.size)
        dimension = int(buffer.source_coordinates.shape[-1])
        for array in (
            buffer.source_coordinates,
            buffer.achieved_targets,
            buffer.residual,
            buffer.scaled_residual,
        ):
            array.reshape(size, dimension)[flat_index] = np.nan
        buffer.scaled_residual_norm_inf.reshape(-1)[flat_index] = np.nan
        buffer.scaled_distance_l2.reshape(-1)[flat_index] = np.nan
        if buffer.auxiliary_values.shape[-1]:
            buffer.auxiliary_values.reshape(size, -1)[flat_index] = np.nan
            buffer.auxiliary_defined.reshape(size, -1)[flat_index] = False
        if buffer.achieved_tilde_values.shape[-1]:
            buffer.achieved_tilde_values.reshape(size, 4)[flat_index] = np.nan
        buffer.achieved_tilde_defined.reshape(-1)[flat_index] = False
        buffer.line_mixture_fraction.reshape(-1)[flat_index] = np.nan
        buffer.line_mixture_fraction_defined.reshape(-1)[flat_index] = False
        buffer.alternate_single_phase_source.reshape(size, dimension)[
            flat_index
        ] = np.nan
        buffer.alternate_single_phase_source_defined.reshape(-1)[flat_index] = False
        buffer.phase_codes.reshape(-1)[flat_index] = _enum_code(
            PhaseLabel.NOT_APPLICABLE, _PHASE_CODES
        )
        buffer.phase_selection_reason_codes.reshape(-1)[flat_index] = _enum_code(
            PhaseRootSelectionReason.NOT_APPLICABLE,
            _PHASE_SELECTION_REASON_CODES,
        )
        previous_phase_route = int(
            buffer.phase_route_codes.reshape(-1)[flat_index]
        )
        if previous_phase_route != _enum_code(
            PhaseRoute.NO_TRANSITION, _PHASE_ROUTE_CODES
        ):
            buffer.phase_route_codes.reshape(-1)[flat_index] = 0
        buffer.mixed_verified_root_count.reshape(-1)[flat_index] = 0
        buffer.phase_boundary.reshape(-1)[flat_index] = False
        buffer.phase_multivalued.reshape(-1)[flat_index] = False
        for array in (
            buffer.disposition_codes,
            buffer.route_codes,
            buffer.solve_status_codes,
            buffer.fast_status_codes,
            buffer.proof_status_codes,
            buffer.projection_quality_codes,
            buffer.initial_guess_source_codes,
        ):
            array.reshape(-1)[flat_index] = 0

    @staticmethod
    def _set_point_codes(
        buffer: Any,
        flat_index: int,
        *,
        disposition: PointDisposition,
        route: PointRoute,
        solve_status: Optional[SolveStatus] = None,
        fast_status: Optional[SolveStatus] = None,
        proof_status: Optional[ProofStatus] = None,
        projection_quality: Optional[ProjectionQuality] = None,
        initial_guess_source: Optional[InitialGuessSource] = None,
    ) -> None:
        values = (
            (buffer.disposition_codes, disposition),
            (buffer.route_codes, route),
            (buffer.solve_status_codes, solve_status),
            (buffer.fast_status_codes, fast_status),
            (buffer.proof_status_codes, proof_status),
            (buffer.projection_quality_codes, projection_quality),
            (buffer.initial_guess_source_codes, initial_guess_source),
        )
        for array, value in values:
            array.reshape(-1)[flat_index] = (
                0 if value is None else pipeline_status_code(value)
            )

    def _store_usable_point(
        self,
        buffer: Any,
        flat_index: int,
        requested_target: np.ndarray,
        source: np.ndarray,
        *,
        require_root: bool,
    ) -> None:
        assert self._target_interpolator is not None
        point = np.asarray(source, dtype=float)
        if point.shape != (self._plan.grid.dimension,) or not np.all(
            np.isfinite(point)
        ):
            raise PipelineStateError("a usable source point must be finite")
        if not self._target_interpolator.contains(
            point, include_boundary_tolerance=False
        ):
            raise PipelineStateError("a usable source point lies outside the domain")
        achieved = np.asarray(
            self._target_interpolator.evaluate(point), dtype=float
        )
        self._store_usable_values(
            buffer,
            flat_index,
            requested_target,
            point,
            achieved,
            require_root=require_root,
        )

    def _store_usable_values(
        self,
        buffer: Any,
        flat_index: int,
        requested_target: np.ndarray,
        source: np.ndarray,
        achieved: np.ndarray,
        *,
        require_root: bool,
    ) -> None:
        point = np.asarray(source, dtype=float)
        achieved = np.asarray(achieved, dtype=float)
        dimension = self._plan.grid.dimension
        if (
            point.shape != (dimension,)
            or achieved.shape != (dimension,)
            or not np.all(np.isfinite(point))
        ):
            raise PipelineStateError(
                "usable source and achieved target vectors must be finite"
            )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            residual = achieved - requested_target
            scaled = residual / self._plan.residual_scales
        norm_inf = float(np.max(np.abs(scaled)))
        distance = float(np.hypot.reduce(scaled))
        if not (
            np.all(np.isfinite(achieved))
            and np.all(np.isfinite(residual))
            and np.all(np.isfinite(scaled))
            and math.isfinite(norm_inf)
            and math.isfinite(distance)
        ):
            raise PipelineStateError(
                "final forward residual is not representable as finite binary64"
            )
        if (
            require_root
            and norm_inf > self._plan.config.solver.root_residual_tolerance
        ):
            raise PipelineStateError(
                "final authoritative reevaluation does not satisfy root tolerance"
            )
        size = int(buffer.disposition_codes.size)
        buffer.source_coordinates.reshape(size, dimension)[flat_index] = point
        buffer.achieved_targets.reshape(size, dimension)[flat_index] = achieved
        buffer.residual.reshape(size, dimension)[flat_index] = residual
        buffer.scaled_residual.reshape(size, dimension)[flat_index] = scaled
        buffer.scaled_residual_norm_inf.reshape(-1)[flat_index] = norm_inf
        buffer.scaled_distance_l2.reshape(-1)[flat_index] = distance

    def _mark_ordinary_phase(self, buffer: Any, flat_index: int) -> None:
        if self._phase_solver is None:
            buffer.phase_codes.reshape(-1)[flat_index] = _enum_code(
                PhaseLabel.NOT_APPLICABLE, _PHASE_CODES
            )
            buffer.phase_route_codes.reshape(-1)[flat_index] = _enum_code(
                PhaseRoute.NO_TRANSITION, _PHASE_ROUTE_CODES
            )
            buffer.phase_selection_reason_codes.reshape(-1)[flat_index] = _enum_code(
                PhaseRootSelectionReason.NOT_APPLICABLE,
                _PHASE_SELECTION_REASON_CODES,
            )
            return
        buffer.phase_codes.reshape(-1)[flat_index] = _enum_code(
            PhaseLabel.SINGLE_PHASE, _PHASE_CODES
        )
        buffer.phase_route_codes.reshape(-1)[flat_index] = _enum_code(
            PhaseRoute.ORDINARY, _PHASE_ROUTE_CODES
        )
        buffer.phase_selection_reason_codes.reshape(-1)[flat_index] = _enum_code(
            PhaseRootSelectionReason.UNAMBIGUOUS,
            _PHASE_SELECTION_REASON_CODES,
        )

    @staticmethod
    def _record_subcell_fallback_counters(
        counters: Dict[str, int], fallback: Any
    ) -> None:
        """Aggregate private subcell diagnostics without adding statuses."""

        report = getattr(fallback, "subcell_report", None)
        if report is None:
            return
        counters["subcell_fallback_attempted"] += 1
        counters["subcell_boxes_evaluated"] += int(
            report.total_boxes_evaluated
        )
        counters["subcell_residual_boxes"] += int(
            getattr(fallback, "subcell_residual_box_count", 0)
        )
        counters["subcell_krawczyk_boxes_processed"] += int(
            getattr(fallback, "boxes_processed", 0)
        )
        if not report.complete:
            counters["subcell_guarded_incomplete"] += 1
        if report.first_exclusion_depth is not None:
            counters["subcell_certified_no_root"] += 1
            key = "subcell_certificate_depth_%d" % int(
                report.first_exclusion_depth
            )
            counters[key] += 1
        elif int(getattr(fallback, "boxes_processed", 0)) > 0:
            counters["subcell_residual_krawczyk"] += 1

    def _store_phase_result(
        self,
        buffer: Any,
        flat_index: int,
        requested_target: np.ndarray,
        result: Any,
    ) -> None:
        """Store complete phase provenance and an authoritative solved root."""

        from .phase_solver import PhaseRootResult

        if not isinstance(result, PhaseRootResult):
            raise PipelineStateError("phase solver returned an invalid result")
        route = PhaseRoute(result.route.value)
        buffer.phase_codes.reshape(-1)[flat_index] = _enum_code(
            result.phase_label, _PHASE_CODES
        )
        buffer.phase_route_codes.reshape(-1)[flat_index] = _enum_code(
            route, _PHASE_ROUTE_CODES
        )
        buffer.phase_selection_reason_codes.reshape(-1)[flat_index] = _enum_code(
            result.selection_reason, _PHASE_SELECTION_REASON_CODES
        )
        buffer.mixed_candidate.reshape(-1)[flat_index] = result.mixed_candidate
        buffer.exact_mixed_candidate.reshape(-1)[flat_index] = (
            result.exact_mixed_candidate
        )
        if result.mixed_candidate_entry_count > np.iinfo(np.uint32).max:
            raise PipelineStateError("mixed candidate count exceeds uint32 storage")
        if result.mixed_verified_root_count > np.iinfo(np.uint8).max:
            raise PipelineStateError("verified mixed-root count exceeds uint8 storage")
        buffer.mixed_candidate_entry_count.reshape(-1)[flat_index] = (
            result.mixed_candidate_entry_count
        )
        buffer.mixed_verified_root_count.reshape(-1)[flat_index] = (
            result.mixed_verified_root_count
        )
        buffer.phase_boundary.reshape(-1)[flat_index] = result.phase_boundary
        buffer.phase_multivalued.reshape(-1)[flat_index] = result.multivalued
        if result.line_mixture_fraction is not None:
            buffer.line_mixture_fraction.reshape(-1)[flat_index] = (
                result.line_mixture_fraction
            )
            buffer.line_mixture_fraction_defined.reshape(-1)[flat_index] = True
        if result.alternate_single_phase_root is not None:
            size = int(buffer.disposition_codes.size)
            dimension = self._plan.grid.dimension
            buffer.alternate_single_phase_source.reshape(size, dimension)[
                flat_index
            ] = result.alternate_single_phase_root
            buffer.alternate_single_phase_source_defined.reshape(-1)[
                flat_index
            ] = True
        if not result.solved or result.source_coordinates is None:
            return
        assert self._target_interpolator is not None
        source = np.asarray(result.source_coordinates, dtype=float)
        if not self._target_interpolator.contains(
            source, include_boundary_tolerance=False
        ):
            raise PipelineStateError("phase root lies outside the ordinary source domain")
        if route is PhaseRoute.LINE_REGION_S:
            if self._mixed_phase_candidates is None:
                raise PipelineStateError("LINE phase root has no mixed interpolator")
            parameters = np.asarray(result.parameter_coordinates, dtype=float)
            achieved = np.asarray(
                self._mixed_phase_candidates.interpolator.evaluate(parameters),
                dtype=float,
            )
        else:
            achieved = np.asarray(
                self._target_interpolator.evaluate(source), dtype=float
            )
        self._store_usable_values(
            buffer,
            flat_index,
            requested_target,
            source,
            achieved,
            require_root=True,
        )

    @staticmethod
    def _solver_seed_is_usable(
        interpolator: MultilinearInterpolator, candidate: Optional[np.ndarray]
    ) -> bool:
        if candidate is None:
            return False
        point = np.asarray(candidate, dtype=float)
        return bool(
            point.shape == (interpolator.dimension,)
            and np.all(np.isfinite(point))
            and interpolator.contains(point, include_boundary_tolerance=False)
        )

    def _mark_unexpected_error(
        self,
        buffer: Any,
        buffer_index: int,
        stage: str,
        error: Exception,
        error_records: list,
        *,
        reported_flat_index: Optional[int] = None,
    ) -> None:
        flat_index = (
            int(buffer_index)
            if reported_flat_index is None
            else int(reported_flat_index)
        )
        if self._plan.config.execution.error_policy is ErrorPolicy.FAIL_FAST:
            raise PipelineStateError(
                "%s failed at flat grid index %d: %s"
                % (stage, flat_index, error)
            ) from error
        prescreen = int(buffer.prescreen_codes.reshape(-1)[buffer_index])
        self._clear_point(buffer, buffer_index)
        buffer.prescreen_codes.reshape(-1)[buffer_index] = prescreen
        self._set_point_codes(
            buffer,
            buffer_index,
            disposition=PointDisposition.ERROR,
            route=PointRoute.ERROR,
        )
        error_records.append(
            {
                "flat_index": int(flat_index),
                "stage": stage,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
        )

    @staticmethod
    def _code_counts(
        values: np.ndarray,
        codebook: Mapping[Enum, int],
        *,
        zero_label: str,
    ) -> Dict[str, int]:
        flat = np.asarray(values).reshape(-1)
        counts = {zero_label: int(np.count_nonzero(flat == 0))}
        counts.update(
            {
                member.value: int(np.count_nonzero(flat == code))
                for member, code in codebook.items()
                if code != 0
            }
        )
        return counts

    def _status_summary(self, buffer: _PipelineResultBuffer) -> Dict[str, Any]:
        line_mask = buffer.phase_route_codes == _enum_code(
            PhaseRoute.LINE_REGION_S, _PHASE_ROUTE_CODES
        )
        binodal_mask = buffer.phase_route_codes == _enum_code(
            PhaseRoute.BINODALS_ORDINARY, _PHASE_ROUTE_CODES
        )

        def auxiliary_counts(mask: np.ndarray) -> Dict[str, Dict[str, int]]:
            rows = int(np.count_nonzero(mask))
            return {
                name: {
                    "rows": rows,
                    "defined": int(
                        np.count_nonzero(
                            buffer.auxiliary_defined[..., index][mask]
                        )
                    ),
                    "undefined": rows
                    - int(
                        np.count_nonzero(
                            buffer.auxiliary_defined[..., index][mask]
                        )
                    ),
                }
                for index, name in enumerate(self._plan.auxiliary_names)
            }

        summary = {
            "grid_points": self._plan.grid.size,
            "disposition": self._code_counts(
                buffer.disposition_codes,
                _DISPOSITION_CODES,
                zero_label="unassigned",
            ),
            "route": self._code_counts(
                buffer.route_codes, _ROUTE_CODES, zero_label="unassigned"
            ),
            "prescreen": self._code_counts(
                buffer.prescreen_codes,
                _PRESCREEN_CODES,
                zero_label="not_run",
            ),
            "solve_status": self._code_counts(
                buffer.solve_status_codes,
                _SOLVE_STATUS_CODES,
                zero_label="not_run",
            ),
            "fast_status": self._code_counts(
                buffer.fast_status_codes,
                _SOLVE_STATUS_CODES,
                zero_label="not_run",
            ),
            "proof_status": self._code_counts(
                buffer.proof_status_codes,
                _PROOF_STATUS_CODES,
                zero_label="not_run",
            ),
            "projection_quality": self._code_counts(
                buffer.projection_quality_codes,
                _PROJECTION_QUALITY_CODES,
                zero_label="not_run",
            ),
            "initial_guess_source": self._code_counts(
                buffer.initial_guess_source_codes,
                _INITIAL_GUESS_CODES,
                zero_label="not_run",
            ),
            "phase": self._code_counts(
                buffer.phase_codes,
                _PHASE_CODES,
                zero_label="unassigned",
            ),
            "phase_route": self._code_counts(
                buffer.phase_route_codes,
                _PHASE_ROUTE_CODES,
                zero_label="not_evaluated",
            ),
            "phase_selection_reason": self._code_counts(
                buffer.phase_selection_reason_codes,
                _PHASE_SELECTION_REASON_CODES,
                zero_label="not_evaluated",
            ),
            "phase_diagnostics": {
                "mixed_candidates": int(
                    np.count_nonzero(buffer.mixed_candidate)
                ),
                "exact_mixed_candidates": int(
                    np.count_nonzero(buffer.exact_mixed_candidate)
                ),
                "candidate_image_matches": int(
                    np.sum(
                        buffer.mixed_candidate_entry_count,
                        dtype=np.uint64,
                    )
                ),
                "verified_mixed_roots": int(
                    np.sum(
                        buffer.mixed_verified_root_count,
                        dtype=np.uint64,
                    )
                ),
                "boundary_roots": int(
                    np.count_nonzero(buffer.phase_boundary)
                ),
                "multivalued_roots": int(
                    np.count_nonzero(buffer.phase_multivalued)
                ),
                "jacobian_selected_roots": int(
                    np.count_nonzero(
                        buffer.phase_selection_reason_codes
                        == _enum_code(
                            PhaseRootSelectionReason.SELECTED_FROM_JACOBIAN,
                            _PHASE_SELECTION_REASON_CODES,
                        )
                    )
                ),
                "line_fraction_defined": int(
                    np.count_nonzero(buffer.line_mixture_fraction_defined)
                ),
                "alternate_single_phase_roots": int(
                    np.count_nonzero(
                        buffer.alternate_single_phase_source_defined
                    )
                ),
            },
            "auxiliary_defined": {
                name: int(np.count_nonzero(buffer.auxiliary_defined[..., index]))
                for index, name in enumerate(self._plan.auxiliary_names)
            },
            "auxiliary_phase_policy": {
                "line": {
                    "scientific_status": (
                        "linear_is_a_lever_rule_or_effective_numerical_"
                        "prescription_not_an_exact_singular_response"
                    ),
                    "lambda_semantics": (
                        "lambda=0 hadronic endpoint; lambda=1 qgp endpoint"
                    ),
                    "linear_rule": "(1-lambda)*X_H + lambda*X_Q",
                    "fields": {
                        name: self._plan.config.line_auxiliary_policies.get(
                            name,
                            self._plan.config.line_auxiliary_default_policy,
                        ).value
                        for name in self._plan.auxiliary_names
                    },
                    "counts": auxiliary_counts(line_mask),
                },
                "binodals": {
                    "policy": "ordinary_table_interpolation_at_solved_source",
                    "counts": auxiliary_counts(binodal_mask),
                },
                "undefined_encoding": "nan_with_auxiliary_defined_false",
            },
            "achieved_tilde_defined": int(
                np.count_nonzero(buffer.achieved_tilde_defined)
            ),
        }
        for name in (
            "disposition",
            "route",
            "prescreen",
            "solve_status",
            "fast_status",
            "proof_status",
            "projection_quality",
            "initial_guess_source",
            "phase",
            "phase_route",
        ):
            if sum(summary[name].values()) != self._plan.grid.size:
                raise PipelineStateError(
                    "run status summary does not cover every grid point for %s"
                    % name
                )
        if (
            summary["disposition"]["unassigned"]
            or summary["route"]["unassigned"]
            or summary["phase"]["unassigned"]
        ):
            raise PipelineStateError("run ended with unassigned terminal states")
        return summary

    def _finish_outputs(
        self,
        buffer: Any,
        error_records: list,
        *,
        flat_offset: int = 0,
    ) -> Tuple[int, int]:
        size = int(buffer.disposition_codes.size)
        usable_codes = (
            pipeline_status_code(PointDisposition.SOLVED),
            pipeline_status_code(PointDisposition.PROJECTED),
        )
        usable = np.flatnonzero(
            np.isin(buffer.disposition_codes.reshape(-1), usable_codes)
        )
        source = buffer.source_coordinates.reshape(size, self._plan.grid.dimension)
        auxiliary_count = 0
        tilde_count = 0
        transform = (
            ConstrainedTildeTransform(self._plan.grid.constraints)
            if self._plan.grid.space is GridSpace.TILDE
            else None
        )
        for flat_index in usable:
            index = int(flat_index)
            try:
                if self._auxiliary_interpolator is not None:
                    is_line = (
                        int(buffer.phase_route_codes.reshape(-1)[index])
                        == _enum_code(
                            PhaseRoute.LINE_REGION_S,
                            _PHASE_ROUTE_CODES,
                        )
                    )
                    evaluate = np.ones(
                        len(self._plan.auxiliary_names), dtype=bool
                    )
                    values = np.full(
                        len(self._plan.auxiliary_names), np.nan, dtype=float
                    )
                    if is_line:
                        evaluate = np.asarray(
                            [
                                self._plan.config.line_auxiliary_policies.get(
                                    name,
                                    self._plan.config.line_auxiliary_default_policy,
                                )
                                is LineAuxiliaryPolicy.LINEAR
                                for name in self._plan.auxiliary_names
                            ],
                            dtype=bool,
                        )
                        if np.any(evaluate):
                            if self._mixed_phase_candidates is None:
                                raise PipelineStateError(
                                    "LINE auxiliary evaluation has no RegionS model"
                                )
                            strip = self._mixed_phase_candidates.model
                            if not hasattr(strip, "evaluate_auxiliaries"):
                                raise PipelineStateError(
                                    "LINE auxiliary evaluation has an invalid model"
                                )
                            if not bool(
                                buffer.line_mixture_fraction_defined.reshape(-1)[
                                    index
                                ]
                            ):
                                raise PipelineStateError(
                                    "LINE auxiliary evaluation lacks lambda"
                                )
                            fraction = float(
                                buffer.line_mixture_fraction.reshape(-1)[index]
                            )
                            linear_values = np.asarray(
                                strip.evaluate_auxiliaries(
                                    float(source[index, 1]), fraction
                                ),
                                dtype=float,
                            )
                            linear_by_name = dict(
                                zip(strip.auxiliary_names, linear_values)
                            )
                            for field_index, name in enumerate(
                                self._plan.auxiliary_names
                            ):
                                if evaluate[field_index]:
                                    values[field_index] = linear_by_name[name]
                            auxiliary_count += 1
                    elif np.any(evaluate):
                        evaluated = np.asarray(
                            self._auxiliary_interpolator.evaluate(source[index]),
                            dtype=float,
                        )
                        values[evaluate] = evaluated[evaluate]
                        auxiliary_count += 1
                    finite = evaluate & np.isfinite(values)
                    stored = np.where(finite, values, np.nan)
                    buffer.auxiliary_values.reshape(size, -1)[index] = stored
                    buffer.auxiliary_defined.reshape(size, -1)[index] = finite
                if transform is not None:
                    achieved = buffer.achieved_targets.reshape(
                        size, self._plan.grid.dimension
                    )[index]
                    coordinates = transform.targets_to_coordinates(
                        achieved[np.newaxis, :]
                    ).all_values[0]
                    if coordinates.shape != (4,) or not np.all(
                        np.isfinite(coordinates)
                    ):
                        raise PipelineStateError(
                            "achieved tilde coordinates are not finite"
                        )
                    buffer.achieved_tilde_values.reshape(size, 4)[index] = coordinates
                    buffer.achieved_tilde_defined.reshape(-1)[index] = True
                    tilde_count += 1
            except Exception as exc:
                self._mark_unexpected_error(
                    buffer,
                    index,
                    "output_evaluation",
                    exc,
                    error_records,
                    reported_flat_index=flat_offset + index,
                )
        return auxiliary_count, tilde_count

    def _execute_chunk(self, chunk: GridChunk) -> _PipelineChunkResult:
        """Execute one self-contained chunk with worker-local numerical state."""

        if not isinstance(chunk, GridChunk):
            raise PipelineStateError("pipeline chunk has an invalid type")
        if chunk.start < 0 or chunk.stop > self._plan.grid.size:
            raise PipelineStateError("pipeline chunk lies outside the target grid")
        expected_indices = np.arange(chunk.start, chunk.stop, dtype=np.int64)
        if not np.array_equal(chunk.flat_indices, expected_indices):
            raise PipelineStateError(
                "pipeline chunk flat indices do not match its assigned range"
            )
        expected_chunk = self._plan.grid.chunk(chunk.start, chunk.stop)
        for name in (
            "requested_values",
            "canonical_targets",
            "transform_valid",
            "requested_dependent_tilde",
        ):
            if not np.array_equal(
                getattr(chunk, name), getattr(expected_chunk, name), equal_nan=True
            ):
                raise PipelineStateError(
                    "pipeline chunk %s does not match the worker grid" % name
                )
        if not self._prepared:
            self._prepare()
        assert self._target_interpolator is not None
        assert self._reachability is not None
        assert self._solver is not None
        output = _PipelineChunkResult.allocate(
            chunk.start,
            chunk.stop,
            self._plan.grid.dimension,
            len(self._plan.auxiliary_names),
            tilde=self._plan.grid.space is GridSpace.TILDE,
        )
        if self._phase_solver is None:
            output.phase_route_codes[:] = _enum_code(
                PhaseRoute.NO_TRANSITION, _PHASE_ROUTE_CODES
            )
        fast_queue = []
        phase_queue = []
        fallback_queue = []
        projection_queue = []
        error_records = []
        projection_failure_records = []
        counters = {
            "chunks": 1,
            "prescreen_evaluated": 0,
            "prescreen_excluded": 0,
            "exact_candidates": 0,
            "tolerance_shell_candidates": 0,
            "mixed_prescreen_evaluated": 0,
            "mixed_candidates": 0,
            "exact_mixed_candidates": 0,
            "invalid_targets": 0,
            "fast_attempted": 0,
            "fallback_attempted": 0,
            "phase_attempted": 0,
            "mixed_roots": 0,
            "single_phase_roots": 0,
            "phase_boundary_roots": 0,
            "phase_multivalued_roots": 0,
            "jacobian_selected_roots": 0,
            "line_fraction_defined": 0,
            "line_auxiliary_values_defined": 0,
            "line_auxiliary_values_undefined": 0,
            "projection_attempted": 0,
            "auxiliary_points_evaluated": 0,
            "tilde_points_evaluated": 0,
        }
        counters.update({name: 0 for name in _SEED_COUNTER_NAMES})
        counters.update({name: 0 for name in _SUBCELL_COUNTER_NAMES})

        stage_started = time.perf_counter()
        initial_guess_seconds = 0.0
        localization = None
        try:
            if self._uses_complete_ordinary_localization():
                hierarchy = self._reachability.hierarchy
                if (
                    self._reachability_config.method == "hierarchical"
                    and (hierarchy is None or not hierarchy.reaches_leaf)
                ):
                    raise PipelineStateError(
                        "ordinary hierarchical localization requires a hierarchy "
                        "that reaches individual source cells"
                    )
                localization = self._reachability.localize_chunk(chunk)
                localization.require_complete_for(
                    self._reachability.provenance
                )
                classified = localization.result
            else:
                classified = self._reachability.classify_chunk(chunk)
            phase_classified = (
                None
                if self._mixed_phase_candidates is None
                else self._mixed_phase_candidates.classify(
                    chunk.canonical_targets,
                    valid_mask=chunk.transform_valid,
                )
            )
        except Exception as exc:
            for local, flat in enumerate(chunk.flat_indices):
                global_index = int(flat)
                if not chunk.transform_valid[local]:
                    self._set_point_codes(
                        output,
                        local,
                        disposition=PointDisposition.INVALID_TARGET,
                        route=PointRoute.INVALID_TRANSFORM,
                    )
                    counters["invalid_targets"] += 1
                else:
                    self._mark_unexpected_error(
                        output,
                        local,
                        "reachability",
                        exc,
                        error_records,
                        reported_flat_index=global_index,
                    )
        else:
            prescreen = np.asarray(classified.prescreen_codes).reshape(-1).copy()
            mixed_candidate = np.zeros(prescreen.shape, dtype=bool)
            exact_mixed_candidate = np.zeros(prescreen.shape, dtype=bool)
            if phase_classified is not None:
                mixed_candidate = np.asarray(
                    phase_classified.candidate, dtype=bool
                ).reshape(-1)
                exact_mixed_candidate = np.asarray(
                    phase_classified.exact_candidate, dtype=bool
                ).reshape(-1)
                output.mixed_candidate[:] = mixed_candidate
                output.exact_mixed_candidate[:] = exact_mixed_candidate
                counters["mixed_prescreen_evaluated"] += int(
                    np.count_nonzero(phase_classified.evaluated)
                )
                counters["mixed_candidates"] += int(
                    np.count_nonzero(mixed_candidate)
                )
                counters["exact_mixed_candidates"] += int(
                    np.count_nonzero(exact_mixed_candidate)
                )
                prescreen[mixed_candidate] = pipeline_status_code(
                    PrescreenStatus.CANDIDATE
                )
            output.prescreen_codes[:] = prescreen
            counters["prescreen_evaluated"] += int(
                np.count_nonzero(classified.evaluated_mask)
            )
            counters["prescreen_excluded"] += int(
                np.count_nonzero(
                    prescreen
                    == pipeline_status_code(
                        PrescreenStatus.DEFINITELY_UNREACHABLE
                    )
                )
            )
            counters["exact_candidates"] += int(
                np.count_nonzero(classified.exact_candidate_mask)
            )
            counters["tolerance_shell_candidates"] += int(
                np.count_nonzero(classified.tolerance_shell_mask)
            )
            for local, flat in enumerate(chunk.flat_indices):
                global_index = int(flat)
                if not chunk.transform_valid[local]:
                    self._set_point_codes(
                        output,
                        local,
                        disposition=PointDisposition.INVALID_TARGET,
                        route=PointRoute.INVALID_TRANSFORM,
                    )
                    counters["invalid_targets"] += 1
                    continue
                target = np.asarray(
                    chunk.canonical_targets[local], dtype=float
                ).copy()
                status = int(prescreen[local])
                if status == pipeline_status_code(
                    PrescreenStatus.DEFINITELY_UNREACHABLE
                ):
                    if self._plan.config.projection.enabled:
                        projection_queue.append(
                            _ProjectionWork(
                                global_index,
                                local,
                                target,
                                PointRoute.PROJECTED_PRESCREEN,
                                PointRoute.PROJECTION_FAILED_PRESCREEN,
                            )
                        )
                    else:
                        self._set_point_codes(
                            output,
                            local,
                            disposition=PointDisposition.CERTIFIED_UNREACHABLE,
                            route=PointRoute.CERTIFIED_UNREACHABLE_PRESCREEN,
                        )
                    continue
                if status != pipeline_status_code(PrescreenStatus.CANDIDATE):
                    self._mark_unexpected_error(
                        output,
                        local,
                        "reachability",
                        PipelineStateError(
                            "valid target did not receive a terminal pre-screen code"
                        ),
                        error_records,
                        reported_flat_index=global_index,
                    )
                    continue
                work = _FastWork(global_index, local, target)
                if mixed_candidate[local]:
                    phase_queue.append(work)
                else:
                    if self._phase_solver is not None:
                        output.phase_route_codes[local] = _enum_code(
                            PhaseRoute.ORDINARY, _PHASE_ROUTE_CODES
                        )
                    fast_queue.append(work)
        prescreen_seconds = max(
            0.0,
            float(time.perf_counter() - stage_started)
            - initial_guess_seconds,
        )

        stage_started = time.perf_counter()
        phase_guess_seconds = 0.0
        for work in phase_queue:
            local = work.local_index
            guess_started = time.perf_counter()
            try:
                work = self._seed_work(work, counters)
            except Exception as exc:
                self._mark_unexpected_error(
                    output,
                    local,
                    "initial_guess",
                    exc,
                    error_records,
                    reported_flat_index=work.flat_index,
                )
                continue
            finally:
                measured = float(time.perf_counter() - guess_started)
                initial_guess_seconds += measured
                phase_guess_seconds += measured
            counters["phase_attempted"] += 1
            counters["fast_attempted"] += 1
            try:
                assert self._phase_solver is not None
                assert work.initial_guess is not None
                assert work.initial_guess_source is not None
                phase_result = self._phase_solver.solve(
                    work.target,
                    work.initial_guess,
                    strategy=SolveStrategy.AUTO,
                )
                self._store_phase_result(
                    output, local, work.target, phase_result
                )
                solved = phase_result.selected_result
                common = {
                    "initial_guess_source": work.initial_guess_source,
                }
                if solved.status is SolveStatus.CONVERGED_FAST:
                    self._set_point_codes(
                        output,
                        local,
                        disposition=PointDisposition.SOLVED,
                        route=PointRoute.FAST,
                        solve_status=SolveStatus.CONVERGED_FAST,
                        fast_status=SolveStatus.CONVERGED_FAST,
                        proof_status=ProofStatus.NONE,
                        **common,
                    )
                elif solved.status is SolveStatus.CONVERGED_FALLBACK:
                    counters["fallback_attempted"] += 1
                    self._set_point_codes(
                        output,
                        local,
                        disposition=PointDisposition.SOLVED,
                        route=PointRoute.FALLBACK_CERTIFIED,
                        solve_status=SolveStatus.CONVERGED_FALLBACK,
                        fast_status=SolveStatus.FAST_PATH_FAILED,
                        proof_status=ProofStatus.UNIQUE_IN_BOX,
                        **common,
                    )
                elif solved.status is SolveStatus.CERTIFIED_ROOT_UNREFINED:
                    counters["fallback_attempted"] += 1
                    self._set_point_codes(
                        output,
                        local,
                        disposition=PointDisposition.ROOT_CERTIFIED_UNREFINED,
                        route=PointRoute.FALLBACK_ROOT_UNREFINED,
                        solve_status=SolveStatus.CERTIFIED_ROOT_UNREFINED,
                        fast_status=SolveStatus.FAST_PATH_FAILED,
                        proof_status=ProofStatus.UNIQUE_IN_BOX,
                        **common,
                    )
                elif solved.status is SolveStatus.UNRESOLVED:
                    counters["fallback_attempted"] += 1
                    self._set_point_codes(
                        output,
                        local,
                        disposition=PointDisposition.UNRESOLVED,
                        route=PointRoute.FALLBACK_UNRESOLVED,
                        solve_status=SolveStatus.UNRESOLVED,
                        fast_status=SolveStatus.FAST_PATH_FAILED,
                        proof_status=ProofStatus.INCONCLUSIVE,
                        **common,
                    )
                elif solved.status is SolveStatus.CERTIFIED_NO_ROOT:
                    counters["fallback_attempted"] += 1
                    self._set_point_codes(
                        output,
                        local,
                        disposition=PointDisposition.CERTIFIED_UNREACHABLE,
                        route=PointRoute.CERTIFIED_UNREACHABLE_FALLBACK,
                        solve_status=SolveStatus.CERTIFIED_NO_ROOT,
                        fast_status=SolveStatus.FAST_PATH_FAILED,
                        proof_status=ProofStatus.NO_ROOT_IN_DOMAIN,
                        **common,
                    )
                    if self._plan.config.projection.enabled:
                        best = (
                            np.asarray(solved.best_candidate, dtype=float).copy()
                            if self._solver_seed_is_usable(
                                self._target_interpolator,
                                solved.best_candidate,
                            )
                            else None
                        )
                        projection_queue.append(
                            _ProjectionWork(
                                work.flat_index,
                                local,
                                work.target,
                                PointRoute.PROJECTED_CERTIFIED_NO_ROOT,
                                PointRoute.PROJECTION_FAILED_CERTIFIED_NO_ROOT,
                                best,
                            )
                        )
                else:
                    raise PipelineStateError(
                        "phase-aware AUTO returned unexpected status %s"
                        % solved.status.value
                    )
                if phase_result.phase_label is PhaseLabel.MIXED_PHASE:
                    counters["mixed_roots"] += 1
                    counters["phase_boundary_roots"] += int(
                        phase_result.phase_boundary
                    )
                    counters["phase_multivalued_roots"] += int(
                        phase_result.multivalued
                    )
                    counters["jacobian_selected_roots"] += int(
                        phase_result.selection_reason
                        is PhaseRootSelectionReason.SELECTED_FROM_JACOBIAN
                    )
                    counters["line_fraction_defined"] += int(
                        phase_result.line_mixture_fraction is not None
                    )
                elif phase_result.phase_label is PhaseLabel.SINGLE_PHASE:
                    counters["single_phase_roots"] += 1
                if solved.status in (
                    SolveStatus.CONVERGED_FAST,
                    SolveStatus.CONVERGED_FALLBACK,
                ):
                    assert phase_result.source_coordinates is not None
                    self._register_solved_seed(
                        work.flat_index, phase_result.source_coordinates
                    )
            except Exception as exc:
                self._mark_unexpected_error(
                    output,
                    local,
                    "phase_solver",
                    exc,
                    error_records,
                    reported_flat_index=work.flat_index,
                )
        phase_seconds = max(
            0.0,
            float(time.perf_counter() - stage_started) - phase_guess_seconds,
        )

        stage_started = time.perf_counter()
        fast_guess_seconds = 0.0
        for work in fast_queue:
            local = work.local_index
            selection = None
            guess_started = time.perf_counter()
            try:
                if localization is not None:
                    selection = self._select_localized_seed(
                        work.flat_index,
                        work.target,
                        localization,
                        local,
                        use_reachability_seed=(
                            self._plan.config.use_reachability_seed
                        ),
                    )
                    self._record_seed_selection(selection, counters)
                    work = _FastWork(
                        work.flat_index,
                        work.local_index,
                        work.target,
                        selection.point,
                        selection.source,
                    )
                else:
                    work = self._seed_work(work, counters)
            except Exception as exc:
                self._mark_unexpected_error(
                    output,
                    local,
                    "initial_guess",
                    exc,
                    error_records,
                    reported_flat_index=work.flat_index,
                )
                continue
            finally:
                measured = float(time.perf_counter() - guess_started)
                initial_guess_seconds += measured
                fast_guess_seconds += measured
            counters["fast_attempted"] += 1
            try:
                assert work.initial_guess is not None
                assert work.initial_guess_source is not None
                solved = (
                    self._solve_selected_fast(work.target, selection)
                    if selection is not None
                    else self._solver.solve(
                        work.target,
                        work.initial_guess,
                        strategy=SolveStrategy.FAST,
                    )
                )
                failed_attempts = [solved]
                if (
                    localization is None
                    and solved.status is SolveStatus.FAST_PATH_FAILED
                ):
                    retry_started = time.perf_counter()
                    retries = self._fast_retry_work(work)
                    measured = float(time.perf_counter() - retry_started)
                    initial_guess_seconds += measured
                    fast_guess_seconds += measured
                    for retry in retries:
                        if (
                            retry.initial_guess_source
                            is InitialGuessSource.COARSE_LEVEL
                        ):
                            counters["seed_retry_coarse_level"] += 1
                        elif (
                            retry.initial_guess_source
                            is InitialGuessSource.DOMAIN_MIDPOINT
                        ):
                            counters["seed_retry_generic"] += 1
                        else:
                            raise PipelineStateError(
                                "FAST retry selected an unsupported seed source"
                            )
                        counters["fast_attempted"] += 1
                        assert retry.initial_guess is not None
                        retry_result = self._solver.solve(
                            retry.target,
                            retry.initial_guess,
                            strategy=SolveStrategy.FAST,
                        )
                        work = retry
                        solved = retry_result
                        if solved.status is SolveStatus.CONVERGED_FAST:
                            break
                        if solved.status is not SolveStatus.FAST_PATH_FAILED:
                            raise PipelineStateError(
                                "FAST retry returned unexpected status %s"
                                % solved.status.value
                            )
                        failed_attempts.append(solved)
                if solved.status is SolveStatus.CONVERGED_FAST:
                    if (
                        solved.solution is None
                        or solved.proof_status is not ProofStatus.NONE
                    ):
                        raise PipelineStateError(
                            "FAST returned an inconsistent success"
                        )
                    self._store_usable_point(
                        output,
                        local,
                        work.target,
                        solved.solution,
                        require_root=True,
                    )
                    self._mark_ordinary_phase(output, local)
                    if self._phase_solver is not None:
                        counters["single_phase_roots"] += 1
                    self._set_point_codes(
                        output,
                        local,
                        disposition=PointDisposition.SOLVED,
                        route=PointRoute.FAST,
                        solve_status=SolveStatus.CONVERGED_FAST,
                        fast_status=SolveStatus.CONVERGED_FAST,
                        proof_status=ProofStatus.NONE,
                        initial_guess_source=work.initial_guess_source,
                    )
                    self._register_solved_seed(
                        work.flat_index, solved.solution
                    )
                elif solved.status is SolveStatus.FAST_PATH_FAILED:
                    candidate = None
                    candidate_norm = math.inf
                    for attempt in failed_attempts:
                        if not self._solver_seed_is_usable(
                            self._target_interpolator, attempt.best_candidate
                        ):
                            continue
                        norm = (
                            math.inf
                            if attempt.scaled_residual is None
                            else float(
                                np.max(np.abs(attempt.scaled_residual))
                            )
                        )
                        if candidate is None or norm < candidate_norm:
                            candidate = np.asarray(
                                attempt.best_candidate, dtype=float
                            ).copy()
                            candidate_norm = norm
                    fallback_queue.append(
                        _FallbackWork(
                            fast=work,
                            fallback_guess=(
                                candidate.copy()
                                if candidate is not None
                                else work.initial_guess.copy()
                            ),
                            fallback_guess_source=(
                                InitialGuessSource.FAST_BEST_CANDIDATE
                                if candidate is not None
                                else work.initial_guess_source
                            ),
                            fast_best_candidate=candidate,
                        )
                    )
                else:
                    raise PipelineStateError(
                        "FAST returned unexpected status %s"
                        % solved.status.value
                    )
            except Exception as exc:
                self._mark_unexpected_error(
                    output,
                    local,
                    "fast_solver",
                    exc,
                    error_records,
                    reported_flat_index=work.flat_index,
                )
        fast_seconds = max(
            0.0,
            float(time.perf_counter() - stage_started) - fast_guess_seconds,
        )

        stage_started = time.perf_counter()
        for work in fallback_queue:
            local = work.fast.local_index
            counters["fallback_attempted"] += 1
            try:
                solved = (
                    self._solver.solve_with_candidate_cells(
                        work.fast.target,
                        localization,
                        local,
                        work.fallback_guess,
                        expected_provenance=self._reachability.provenance,
                        strategy=SolveStrategy.FALLBACK,
                    )
                    if localization is not None
                    else self._solver.solve(
                        work.fast.target,
                        work.fallback_guess,
                        strategy=SolveStrategy.FALLBACK,
                    )
                )
                if solved.fallback is not None:
                    self._record_subcell_fallback_counters(
                        counters, solved.fallback
                    )
                common = {
                    "fast_status": SolveStatus.FAST_PATH_FAILED,
                    "initial_guess_source": work.fallback_guess_source,
                }
                if solved.status is SolveStatus.CONVERGED_FALLBACK:
                    if (
                        solved.solution is None
                        or solved.proof_status is not ProofStatus.UNIQUE_IN_BOX
                    ):
                        raise PipelineStateError(
                            "FALLBACK returned an inconsistent converged result"
                        )
                    self._store_usable_point(
                        output,
                        local,
                        work.fast.target,
                        solved.solution,
                        require_root=True,
                    )
                    self._mark_ordinary_phase(output, local)
                    if self._phase_solver is not None:
                        counters["single_phase_roots"] += 1
                    self._set_point_codes(
                        output,
                        local,
                        disposition=PointDisposition.SOLVED,
                        route=PointRoute.FALLBACK_CERTIFIED,
                        solve_status=SolveStatus.CONVERGED_FALLBACK,
                        proof_status=ProofStatus.UNIQUE_IN_BOX,
                        **common,
                    )
                    self._register_solved_seed(
                        work.fast.flat_index, solved.solution
                    )
                elif solved.status is SolveStatus.CERTIFIED_ROOT_UNREFINED:
                    if solved.proof_status is not ProofStatus.UNIQUE_IN_BOX:
                        raise PipelineStateError(
                            "unrefined root lacks its uniqueness certificate"
                        )
                    self._set_point_codes(
                        output,
                        local,
                        disposition=PointDisposition.ROOT_CERTIFIED_UNREFINED,
                        route=PointRoute.FALLBACK_ROOT_UNREFINED,
                        solve_status=SolveStatus.CERTIFIED_ROOT_UNREFINED,
                        proof_status=ProofStatus.UNIQUE_IN_BOX,
                        **common,
                    )
                elif solved.status is SolveStatus.UNRESOLVED:
                    if solved.proof_status is not ProofStatus.INCONCLUSIVE:
                        raise PipelineStateError(
                            "unresolved FALLBACK result has invalid proof status"
                        )
                    self._set_point_codes(
                        output,
                        local,
                        disposition=PointDisposition.UNRESOLVED,
                        route=PointRoute.FALLBACK_UNRESOLVED,
                        solve_status=SolveStatus.UNRESOLVED,
                        proof_status=ProofStatus.INCONCLUSIVE,
                        **common,
                    )
                elif solved.status is SolveStatus.CERTIFIED_NO_ROOT:
                    if solved.proof_status is not ProofStatus.NO_ROOT_IN_DOMAIN:
                        raise PipelineStateError(
                            "certified no-root result lacks its proof status"
                        )
                    self._set_point_codes(
                        output,
                        local,
                        disposition=PointDisposition.CERTIFIED_UNREACHABLE,
                        route=PointRoute.CERTIFIED_UNREACHABLE_FALLBACK,
                        solve_status=SolveStatus.CERTIFIED_NO_ROOT,
                        proof_status=ProofStatus.NO_ROOT_IN_DOMAIN,
                        **common,
                    )
                    if self._plan.config.projection.enabled:
                        projection_queue.append(
                            _ProjectionWork(
                                work.fast.flat_index,
                                local,
                                work.fast.target,
                                PointRoute.PROJECTED_CERTIFIED_NO_ROOT,
                                PointRoute.PROJECTION_FAILED_CERTIFIED_NO_ROOT,
                                work.fast_best_candidate,
                            )
                        )
                else:
                    raise PipelineStateError(
                        "FALLBACK returned unexpected status %s"
                        % solved.status.value
                    )
            except Exception as exc:
                self._mark_unexpected_error(
                    output,
                    local,
                    "fallback_solver",
                    exc,
                    error_records,
                    reported_flat_index=work.fast.flat_index,
                )
        fallback_seconds = float(time.perf_counter() - stage_started)

        stage_started = time.perf_counter()
        for work in projection_queue:
            local = work.local_index
            counters["projection_attempted"] += 1
            try:
                assert self._projector is not None
                seeds = (
                    (
                        ProjectionSeed(
                            work.fast_best_candidate,
                            ProjectionSeedSource.FAST_BEST_CANDIDATE,
                        ),
                    )
                    if work.fast_best_candidate is not None
                    else ()
                )
                projected = self._projector.project(work.target, seeds=seeds)
                self._store_usable_point(
                    output,
                    local,
                    work.target,
                    projected.source_coordinates,
                    require_root=False,
                )
                output.disposition_codes[local] = pipeline_status_code(
                    PointDisposition.PROJECTED
                )
                output.route_codes[local] = pipeline_status_code(
                    work.success_route
                )
                output.projection_quality_codes[local] = pipeline_status_code(
                    projected.quality
                )
            except ProjectionError as exc:
                if (
                    self._plan.config.execution.error_policy
                    is ErrorPolicy.FAIL_FAST
                ):
                    raise PipelineStateError(
                        "projection failed at flat grid index %d: %s"
                        % (work.flat_index, exc)
                    ) from exc
                self._clear_point(output, local)
                if work.success_route is PointRoute.PROJECTED_PRESCREEN:
                    output.prescreen_codes[local] = pipeline_status_code(
                        PrescreenStatus.DEFINITELY_UNREACHABLE
                    )
                else:
                    output.prescreen_codes[local] = pipeline_status_code(
                        PrescreenStatus.CANDIDATE
                    )
                self._set_point_codes(
                    output,
                    local,
                    disposition=PointDisposition.PROJECTION_FAILED,
                    route=work.failure_route,
                    solve_status=(
                        None
                        if work.success_route is PointRoute.PROJECTED_PRESCREEN
                        else SolveStatus.CERTIFIED_NO_ROOT
                    ),
                    fast_status=(
                        None
                        if work.success_route is PointRoute.PROJECTED_PRESCREEN
                        else SolveStatus.FAST_PATH_FAILED
                    ),
                    proof_status=(
                        None
                        if work.success_route is PointRoute.PROJECTED_PRESCREEN
                        else ProofStatus.NO_ROOT_IN_DOMAIN
                    ),
                    initial_guess_source=(
                        None
                        if work.success_route is PointRoute.PROJECTED_PRESCREEN
                        else InitialGuessSource.FAST_BEST_CANDIDATE
                        if work.fast_best_candidate is not None
                        else self._configured_initial_guess_source()
                    ),
                )
                projection_failure_records.append(
                    {
                        "flat_index": int(work.flat_index),
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            except Exception as exc:
                self._mark_unexpected_error(
                    output,
                    local,
                    "projection",
                    exc,
                    error_records,
                    reported_flat_index=work.flat_index,
                )
        projection_seconds = float(time.perf_counter() - stage_started)

        stage_started = time.perf_counter()
        auxiliary_count, tilde_count = self._finish_outputs(
            output, error_records, flat_offset=chunk.start
        )
        counters["auxiliary_points_evaluated"] = auxiliary_count
        counters["tilde_points_evaluated"] = tilde_count
        line_rows = output.phase_route_codes == _enum_code(
            PhaseRoute.LINE_REGION_S, _PHASE_ROUTE_CODES
        )
        line_slots = int(np.count_nonzero(line_rows)) * len(
            self._plan.auxiliary_names
        )
        line_defined = int(
            np.count_nonzero(output.auxiliary_defined[line_rows])
        )
        counters["line_auxiliary_values_defined"] = line_defined
        counters["line_auxiliary_values_undefined"] = line_slots - line_defined
        output_seconds = float(time.perf_counter() - stage_started)
        output.counters = counters
        output.timing_seconds = {
            "prescreen": prescreen_seconds,
            "initial_guess": initial_guess_seconds,
            "phase_solve": phase_seconds,
            "fast": fast_seconds,
            "fallback": fallback_seconds,
            "projection": projection_seconds,
            "output_evaluation": output_seconds,
        }
        output.error_records = tuple(error_records)
        output.projection_failure_records = tuple(projection_failure_records)
        output.worker_peak_rss_bytes = _process_peak_rss_bytes()
        return output

    def _run_phase4_reference(self) -> PipelineResult:
        """Retained serial oracle from Phase 4 for implementation audits."""

        if self._plan.config.execution.workers != 1:
            raise PipelineConfigurationError(
                "Phase-4 EoSInversionPipeline supports workers=1 only; "
                "process execution is reserved for Phase 5"
            )
        if self._plan.config.retain_detailed_diagnostics:
            raise PipelineConfigurationError(
                "retain_detailed_diagnostics=True is not yet supported by the "
                "compact Phase-4 result schema"
            )
        if self._table.first_order_transition is not None:
            raise PipelineConfigurationError(
                "the private Phase-4 oracle is phase-blind; use run() for "
                "first-order tables"
            )

        run_started = time.perf_counter()
        setup_seconds = self._prepare()
        assert self._target_interpolator is not None
        assert self._reachability is not None
        assert self._solver is not None
        buffer = self._plan.allocate_result_buffer()
        buffer.phase_route_codes[:] = _enum_code(
            PhaseRoute.NO_TRANSITION, _PHASE_ROUTE_CODES
        )
        fast_queue = []
        fallback_queue = []
        projection_queue = []
        error_records = []
        projection_failure_records = []
        counters = {
            "chunks": 0,
            "prescreen_evaluated": 0,
            "prescreen_excluded": 0,
            "exact_candidates": 0,
            "tolerance_shell_candidates": 0,
            "invalid_targets": 0,
            "fast_attempted": 0,
            "fallback_attempted": 0,
            "projection_attempted": 0,
            "auxiliary_points_evaluated": 0,
            "tilde_points_evaluated": 0,
        }
        counters.update({name: 0 for name in _SEED_COUNTER_NAMES})
        counters.update({name: 0 for name in _SUBCELL_COUNTER_NAMES})

        stage_started = time.perf_counter()
        for chunk in self._plan.iter_chunks():
            counters["chunks"] += 1
            try:
                classified = self._reachability.classify_chunk(chunk)
            except Exception as exc:
                for local, flat in enumerate(chunk.flat_indices):
                    index = int(flat)
                    if not chunk.transform_valid[local]:
                        self._set_point_codes(
                            buffer,
                            index,
                            disposition=PointDisposition.INVALID_TARGET,
                            route=PointRoute.INVALID_TRANSFORM,
                        )
                        counters["invalid_targets"] += 1
                    else:
                        self._mark_unexpected_error(
                            buffer, index, "reachability", exc, error_records
                        )
                continue
            prescreen = np.asarray(classified.prescreen_codes).reshape(-1)
            buffer.prescreen_codes.reshape(-1)[chunk.start : chunk.stop] = prescreen
            counters["prescreen_evaluated"] += int(
                np.count_nonzero(classified.evaluated_mask)
            )
            counters["prescreen_excluded"] += int(
                np.count_nonzero(classified.definitely_unreachable_mask)
            )
            counters["exact_candidates"] += int(
                np.count_nonzero(classified.exact_candidate_mask)
            )
            counters["tolerance_shell_candidates"] += int(
                np.count_nonzero(classified.tolerance_shell_mask)
            )

            for local, flat in enumerate(chunk.flat_indices):
                index = int(flat)
                if not chunk.transform_valid[local]:
                    self._set_point_codes(
                        buffer,
                        index,
                        disposition=PointDisposition.INVALID_TARGET,
                        route=PointRoute.INVALID_TRANSFORM,
                    )
                    counters["invalid_targets"] += 1
                    continue
                target = np.asarray(chunk.canonical_targets[local], dtype=float).copy()
                status = int(prescreen[local])
                if status == pipeline_status_code(
                    PrescreenStatus.DEFINITELY_UNREACHABLE
                ):
                    if self._plan.config.projection.enabled:
                        projection_queue.append(
                            _ProjectionWork(
                                index,
                                index,
                                target,
                                PointRoute.PROJECTED_PRESCREEN,
                                PointRoute.PROJECTION_FAILED_PRESCREEN,
                            )
                        )
                    else:
                        self._set_point_codes(
                            buffer,
                            index,
                            disposition=PointDisposition.CERTIFIED_UNREACHABLE,
                            route=PointRoute.CERTIFIED_UNREACHABLE_PRESCREEN,
                        )
                    continue
                if status != pipeline_status_code(PrescreenStatus.CANDIDATE):
                    self._mark_unexpected_error(
                        buffer,
                        index,
                        "reachability",
                        PipelineStateError(
                            "valid target did not receive a terminal pre-screen code"
                        ),
                        error_records,
                    )
                    continue
                try:
                    guess, source = self._initial_guess(target)
                    fast_queue.append(
                        _FastWork(index, index, target, guess, source)
                    )
                except Exception as exc:
                    self._mark_unexpected_error(
                        buffer, index, "initial_guess", exc, error_records
                    )
        prescreen_seconds = float(time.perf_counter() - stage_started)

        stage_started = time.perf_counter()
        for work in fast_queue:
            counters["fast_attempted"] += 1
            try:
                result = self._solver.solve(
                    work.target,
                    work.initial_guess,
                    strategy=SolveStrategy.FAST,
                )
                if result.status is SolveStatus.CONVERGED_FAST:
                    if result.solution is None or result.proof_status is not ProofStatus.NONE:
                        raise PipelineStateError("FAST returned an inconsistent success")
                    self._store_usable_point(
                        buffer,
                        work.flat_index,
                        work.target,
                        result.solution,
                        require_root=True,
                    )
                    self._set_point_codes(
                        buffer,
                        work.flat_index,
                        disposition=PointDisposition.SOLVED,
                        route=PointRoute.FAST,
                        solve_status=SolveStatus.CONVERGED_FAST,
                        fast_status=SolveStatus.CONVERGED_FAST,
                        proof_status=ProofStatus.NONE,
                        initial_guess_source=work.initial_guess_source,
                    )
                elif result.status is SolveStatus.FAST_PATH_FAILED:
                    candidate = (
                        np.asarray(result.best_candidate, dtype=float).copy()
                        if self._solver_seed_is_usable(
                            self._target_interpolator, result.best_candidate
                        )
                        else None
                    )
                    fallback_queue.append(
                        _FallbackWork(
                            fast=work,
                            fallback_guess=(
                                candidate.copy()
                                if candidate is not None
                                else work.initial_guess.copy()
                            ),
                            fallback_guess_source=(
                                InitialGuessSource.FAST_BEST_CANDIDATE
                                if candidate is not None
                                else work.initial_guess_source
                            ),
                            fast_best_candidate=candidate,
                        )
                    )
                else:
                    raise PipelineStateError(
                        "FAST returned unexpected status %s" % result.status.value
                    )
            except Exception as exc:
                self._mark_unexpected_error(
                    buffer, work.flat_index, "fast_solver", exc, error_records
                )
        fast_seconds = float(time.perf_counter() - stage_started)

        stage_started = time.perf_counter()
        for work in fallback_queue:
            index = work.fast.flat_index
            counters["fallback_attempted"] += 1
            try:
                result = self._solver.solve(
                    work.fast.target,
                    work.fallback_guess,
                    strategy=SolveStrategy.FALLBACK,
                )
                common = {
                    "fast_status": SolveStatus.FAST_PATH_FAILED,
                    "initial_guess_source": work.fallback_guess_source,
                }
                if result.status is SolveStatus.CONVERGED_FALLBACK:
                    if (
                        result.solution is None
                        or result.proof_status is not ProofStatus.UNIQUE_IN_BOX
                    ):
                        raise PipelineStateError(
                            "FALLBACK returned an inconsistent converged result"
                        )
                    self._store_usable_point(
                        buffer,
                        index,
                        work.fast.target,
                        result.solution,
                        require_root=True,
                    )
                    self._set_point_codes(
                        buffer,
                        index,
                        disposition=PointDisposition.SOLVED,
                        route=PointRoute.FALLBACK_CERTIFIED,
                        solve_status=SolveStatus.CONVERGED_FALLBACK,
                        proof_status=ProofStatus.UNIQUE_IN_BOX,
                        **common,
                    )
                elif result.status is SolveStatus.CERTIFIED_ROOT_UNREFINED:
                    if result.proof_status is not ProofStatus.UNIQUE_IN_BOX:
                        raise PipelineStateError(
                            "unrefined root lacks its uniqueness certificate"
                        )
                    self._set_point_codes(
                        buffer,
                        index,
                        disposition=PointDisposition.ROOT_CERTIFIED_UNREFINED,
                        route=PointRoute.FALLBACK_ROOT_UNREFINED,
                        solve_status=SolveStatus.CERTIFIED_ROOT_UNREFINED,
                        proof_status=ProofStatus.UNIQUE_IN_BOX,
                        **common,
                    )
                elif result.status is SolveStatus.UNRESOLVED:
                    if result.proof_status is not ProofStatus.INCONCLUSIVE:
                        raise PipelineStateError(
                            "unresolved FALLBACK result has invalid proof status"
                        )
                    self._set_point_codes(
                        buffer,
                        index,
                        disposition=PointDisposition.UNRESOLVED,
                        route=PointRoute.FALLBACK_UNRESOLVED,
                        solve_status=SolveStatus.UNRESOLVED,
                        proof_status=ProofStatus.INCONCLUSIVE,
                        **common,
                    )
                elif result.status is SolveStatus.CERTIFIED_NO_ROOT:
                    if result.proof_status is not ProofStatus.NO_ROOT_IN_DOMAIN:
                        raise PipelineStateError(
                            "certified no-root result lacks its proof status"
                        )
                    if self._plan.config.projection.enabled:
                        self._set_point_codes(
                            buffer,
                            index,
                            disposition=PointDisposition.CERTIFIED_UNREACHABLE,
                            route=PointRoute.CERTIFIED_UNREACHABLE_FALLBACK,
                            solve_status=SolveStatus.CERTIFIED_NO_ROOT,
                            proof_status=ProofStatus.NO_ROOT_IN_DOMAIN,
                            **common,
                        )
                        projection_queue.append(
                            _ProjectionWork(
                                index,
                                index,
                                work.fast.target,
                                PointRoute.PROJECTED_CERTIFIED_NO_ROOT,
                                PointRoute.PROJECTION_FAILED_CERTIFIED_NO_ROOT,
                                work.fast_best_candidate,
                            )
                        )
                    else:
                        self._set_point_codes(
                            buffer,
                            index,
                            disposition=PointDisposition.CERTIFIED_UNREACHABLE,
                            route=PointRoute.CERTIFIED_UNREACHABLE_FALLBACK,
                            solve_status=SolveStatus.CERTIFIED_NO_ROOT,
                            proof_status=ProofStatus.NO_ROOT_IN_DOMAIN,
                            **common,
                        )
                else:
                    raise PipelineStateError(
                        "FALLBACK returned unexpected status %s"
                        % result.status.value
                    )
            except Exception as exc:
                self._mark_unexpected_error(
                    buffer, index, "fallback_solver", exc, error_records
                )
        fallback_seconds = float(time.perf_counter() - stage_started)

        stage_started = time.perf_counter()
        for work in projection_queue:
            counters["projection_attempted"] += 1
            try:
                assert self._projector is not None
                seeds = (
                    (
                        ProjectionSeed(
                            work.fast_best_candidate,
                            ProjectionSeedSource.FAST_BEST_CANDIDATE,
                        ),
                    )
                    if work.fast_best_candidate is not None
                    else ()
                )
                projected = self._projector.project(work.target, seeds=seeds)
                self._store_usable_point(
                    buffer,
                    work.flat_index,
                    work.target,
                    projected.source_coordinates,
                    require_root=False,
                )
                # Preserve native FALLBACK/proof/guess columns already written
                # for the certified-no-root route.  Pre-screen projection rows
                # still contain their required NOT_RUN zero sentinels.
                buffer.disposition_codes.reshape(-1)[work.flat_index] = (
                    pipeline_status_code(PointDisposition.PROJECTED)
                )
                buffer.route_codes.reshape(-1)[work.flat_index] = pipeline_status_code(
                    work.success_route
                )
                buffer.projection_quality_codes.reshape(-1)[work.flat_index] = (
                    pipeline_status_code(projected.quality)
                )
            except ProjectionError as exc:
                if self._plan.config.execution.error_policy is ErrorPolicy.FAIL_FAST:
                    raise PipelineStateError(
                        "projection failed at flat grid index %d: %s"
                        % (work.flat_index, exc)
                    ) from exc
                self._clear_point(buffer, work.flat_index)
                # Reconstruct the no-root evidence erased by _clear_point.
                if work.success_route is PointRoute.PROJECTED_PRESCREEN:
                    buffer.prescreen_codes.reshape(-1)[work.flat_index] = (
                        pipeline_status_code(PrescreenStatus.DEFINITELY_UNREACHABLE)
                    )
                else:
                    buffer.prescreen_codes.reshape(-1)[work.flat_index] = (
                        pipeline_status_code(PrescreenStatus.CANDIDATE)
                    )
                self._set_point_codes(
                    buffer,
                    work.flat_index,
                    disposition=PointDisposition.PROJECTION_FAILED,
                    route=work.failure_route,
                    solve_status=(
                        None
                        if work.success_route is PointRoute.PROJECTED_PRESCREEN
                        else SolveStatus.CERTIFIED_NO_ROOT
                    ),
                    fast_status=(
                        None
                        if work.success_route is PointRoute.PROJECTED_PRESCREEN
                        else SolveStatus.FAST_PATH_FAILED
                    ),
                    proof_status=(
                        None
                        if work.success_route is PointRoute.PROJECTED_PRESCREEN
                        else ProofStatus.NO_ROOT_IN_DOMAIN
                    ),
                    initial_guess_source=(
                        None
                        if work.success_route is PointRoute.PROJECTED_PRESCREEN
                        else InitialGuessSource.FAST_BEST_CANDIDATE
                        if work.fast_best_candidate is not None
                        else self._configured_initial_guess_source()
                    ),
                )
                projection_failure_records.append(
                    {
                        "flat_index": int(work.flat_index),
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            except Exception as exc:
                self._mark_unexpected_error(
                    buffer, work.flat_index, "projection", exc, error_records
                )
        projection_seconds = float(time.perf_counter() - stage_started)

        stage_started = time.perf_counter()
        auxiliary_count, tilde_count = self._finish_outputs(buffer, error_records)
        counters["auxiliary_points_evaluated"] = auxiliary_count
        counters["tilde_points_evaluated"] = tilde_count
        output_seconds = float(time.perf_counter() - stage_started)

        summary = self._status_summary(buffer)
        from scipy import __version__ as scipy_version

        reachability_config = {
            item.name: getattr(self._reachability_config, item.name)
            for item in dataclass_fields(self._reachability_config)
        }
        total_seconds = float(time.perf_counter() - run_started)
        run_manifest = {
            "pipeline_run_manifest_version": PIPELINE_RUN_MANIFEST_VERSION,
            "completed": True,
            "execution": {
                "mode": "serial",
                "workers_used": 1,
                "chunk_size": self._plan.config.execution.chunk_size,
                "error_policy": self._plan.config.execution.error_policy.value,
            },
            "active_domain": {
                "kind": "full_cell_grid",
                "cell_shape": list(self._target_interpolator.cell_shape),
                "active_cell_count": int(
                    self._reachability.enclosures.active_cell_count
                ),
                "restricted": False,
            },
            "reachability": {
                "method": self._reachability_config.method,
                "implementation": (
                    "balanced_source_index_hierarchy_exact_leaf"
                    if self._reachability_config.method == "hierarchical"
                    else "batched_all_cell_reference"
                ),
                "configuration": reachability_config,
                "rigorous_exact_boxes": True,
                "routing_outward_rounded": bool(
                    self._reachability.enclosures.outward_rounded
                ),
            },
            "first_order": dict(self._first_order_manifest),
            "auxiliary_evaluation": summary["auxiliary_phase_policy"],
            "initial_guess": {
                "precedence": [
                    "direct_solved_inverse_grid_neighbor",
                    "precomputed_coarse_grid_seed",
                    "generic_configured_initial_guess",
                ],
                "direct_neighbor_priority": [
                    "%s_%s" % (name, direction)
                    for name in self._plan.grid.axis_names
                    for direction in ("minus", "plus")
                ],
                "direct_neighbor_eligibility": "terminal_solved_only",
                "coarse_configuration": {
                    item.name: getattr(self._plan.config.coarse_seed, item.name)
                    for item in dataclass_fields(self._plan.config.coarse_seed)
                },
                "coarse_precomputation": dict(self._coarse_seed_report),
                "generic_policy": self._plan.config.initial_guess.value,
                "global_source_node_lookup": False,
                "role": "convergence_efficiency_not_branch_selection",
                "multivalued_branch_selection": (
                    "unanimous_local_jacobian_orientation_only"
                ),
            },
            "projection": {
                "enabled": self._plan.config.projection.enabled,
                "method": (
                    "deterministic_local_multistart"
                    if self._plan.config.projection.enabled
                    else None
                ),
                "globally_certified": False,
            },
            "software": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "numpy": np.__version__,
                "scipy": scipy_version,
                "platform": sys.platform,
            },
            "timing_seconds": {
                "planning": self._planning_seconds,
                "resource_setup": setup_seconds,
                "prescreen": prescreen_seconds,
                "fast": fast_seconds,
                "fallback": fallback_seconds,
                "projection": projection_seconds,
                "output_evaluation": output_seconds,
                "total_run": total_seconds,
            },
            "stage_counters": counters,
            "status_counts": {
                name: count
                for name, count in summary["disposition"].items()
                if name != "unassigned"
            },
            "status_summary": summary,
            "errors": {
                "count": len(error_records),
                "records": error_records,
            },
            "projection_failures": {
                "count": len(projection_failure_records),
                "records": projection_failure_records,
            },
            "detailed_diagnostics_retained": False,
        }
        result = buffer.finalize(run_manifest)
        expected_dispositions = {
            name: count
            for name, count in summary["disposition"].items()
            if name != "unassigned"
        }
        if dict(result.status_counts) != expected_dispositions:
            raise PipelineStateError(
                "final result status counts disagree with the run manifest"
            )
        return result

    def _worker_initialization(
        self, *, shared_state_manifest: Optional[Any] = None
    ) -> _WorkerInitialization:
        data_path = self._table.data_path
        if data_path is None:
            raise PipelineStateError("validated table has no resolved data path")
        return _WorkerInitialization(
            metadata_path=str(self._table.metadata_path),
            data_path=str(data_path),
            table_fingerprint=self._table_fingerprint,
            pipeline_configuration=self._plan.config.to_dict(),
            reachability_configuration={
                item.name: getattr(self._reachability_config, item.name)
                for item in dataclass_fields(self._reachability_config)
            },
            first_order_contract=dict(self._first_order_manifest),
            shared_state_manifest=(
                None
                if shared_state_manifest is None
                else str(Path(shared_state_manifest).expanduser().resolve())
            ),
        )

    def _split_checkpoint_chunk(
        self,
        chunk: GridChunk,
        internal_batch_size: Optional[int] = None,
    ) -> Tuple[GridChunk, ...]:
        """Split one atomic checkpoint range into fixed deterministic tasks."""

        if internal_batch_size is None:
            internal_batch_size = (
                self._plan.config.execution.worker_task_size
            )
        checked_size = _positive_integer(
            internal_batch_size, "internal_batch_size"
        )
        boundaries = list(range(chunk.start, chunk.stop, checked_size))
        boundaries.append(chunk.stop)
        parts = tuple(
            self._plan.grid.chunk(start, stop)
            for start, stop in zip(boundaries[:-1], boundaries[1:])
        )
        if (
            not parts
            or parts[0].start != chunk.start
            or parts[-1].stop != chunk.stop
            or any(
                left.stop != right.start or left.stop <= left.start
                for left, right in zip(parts[:-1], parts[1:])
            )
        ):
            raise PipelineStateError("internal chunk partition is incomplete")
        return parts

    def _assemble_checkpoint_chunk(
        self,
        chunk: GridChunk,
        subresults: Sequence[_PipelineChunkResult],
    ) -> _PipelineChunkResult:
        """Assemble ordered internal tasks into one atomically saved result."""

        ordered = sorted(subresults, key=lambda item: (item.start, item.stop))
        if (
            not ordered
            or ordered[0].start != chunk.start
            or ordered[-1].stop != chunk.stop
            or any(
                left.stop != right.start
                for left, right in zip(ordered[:-1], ordered[1:])
            )
        ):
            raise PipelineStateError(
                "internal worker results do not cover the checkpoint chunk exactly"
            )
        combined = _PipelineChunkResult.allocate(
            chunk.start,
            chunk.stop,
            self._plan.grid.dimension,
            len(self._plan.auxiliary_names),
            tilde=self._plan.grid.space is GridSpace.TILDE,
        )
        counter_names = set(ordered[0].counters)
        timing_names = set(ordered[0].timing_seconds)
        combined.counters = {name: 0 for name in counter_names}
        combined.timing_seconds = {name: 0.0 for name in timing_names}
        errors = []
        projection_failures = []
        for result in ordered:
            if set(result.counters) != counter_names:
                raise PipelineStateError(
                    "internal worker results have different counter schemas"
                )
            if set(result.timing_seconds) != timing_names:
                raise PipelineStateError(
                    "internal worker results have different timing schemas"
                )
            destination_start = result.start - chunk.start
            destination_stop = result.stop - chunk.start
            expected_count = destination_stop - destination_start
            for name in _CHUNK_ARRAY_FIELD_NAMES:
                source = np.asarray(getattr(result, name))
                if source.shape[0] != expected_count:
                    raise PipelineStateError(
                        "internal worker field %s has an invalid leading size"
                        % name
                    )
                getattr(combined, name)[destination_start:destination_stop] = source
            for name, value in result.counters.items():
                combined.counters[name] += int(value)
            for name, value in result.timing_seconds.items():
                combined.timing_seconds[name] += float(value)
            errors.extend(dict(record) for record in result.error_records)
            projection_failures.extend(
                dict(record) for record in result.projection_failure_records
            )
            combined.worker_setup_seconds += float(result.worker_setup_seconds)
            combined.worker_peak_rss_bytes = max(
                combined.worker_peak_rss_bytes,
                int(result.worker_peak_rss_bytes),
            )
        # Each worker task reports one processed chunk. Persistence and the
        # public aggregation contract count the enclosing atomic range.
        if "chunks" not in combined.counters:
            raise PipelineStateError("worker counter schema lacks chunks")
        combined.counters["chunks"] = 1
        combined.error_records = tuple(errors)
        combined.projection_failure_records = tuple(projection_failures)
        return combined

    def _merge_chunk_result(
        self,
        buffer: _PipelineResultBuffer,
        chunk_result: _PipelineChunkResult,
        expected_range: Tuple[int, int],
        counters: Dict[str, int],
        worker_timings: Dict[str, float],
        performance_observations: Dict[str, Any],
        error_records: list,
        projection_failure_records: list,
    ) -> None:
        if not isinstance(chunk_result, _PipelineChunkResult):
            raise PipelineStateError("worker returned an invalid chunk result type")
        start, stop = expected_range
        if (chunk_result.start, chunk_result.stop) != (start, stop):
            raise PipelineStateError(
                "worker returned range [%d, %d) for assigned range [%d, %d)"
                % (chunk_result.start, chunk_result.stop, start, stop)
            )
        count = stop - start
        size = self._plan.grid.size
        dimension = self._plan.grid.dimension
        auxiliary_count = len(self._plan.auxiliary_names)
        tilde_columns = 4 if self._plan.grid.space is GridSpace.TILDE else 0
        expected_shapes = {
            "source_coordinates": (count, dimension),
            "achieved_targets": (count, dimension),
            "residual": (count, dimension),
            "scaled_residual": (count, dimension),
            "scaled_residual_norm_inf": (count,),
            "scaled_distance_l2": (count,),
            "auxiliary_values": (count, auxiliary_count),
            "auxiliary_defined": (count, auxiliary_count),
            "achieved_tilde_values": (count, tilde_columns),
            "achieved_tilde_defined": (count,),
            "disposition_codes": (count,),
            "route_codes": (count,),
            "prescreen_codes": (count,),
            "solve_status_codes": (count,),
            "fast_status_codes": (count,),
            "proof_status_codes": (count,),
            "projection_quality_codes": (count,),
            "initial_guess_source_codes": (count,),
            "phase_codes": (count,),
            "phase_route_codes": (count,),
            "phase_selection_reason_codes": (count,),
            "mixed_candidate": (count,),
            "exact_mixed_candidate": (count,),
            "mixed_candidate_entry_count": (count,),
            "mixed_verified_root_count": (count,),
            "phase_boundary": (count,),
            "phase_multivalued": (count,),
            "line_mixture_fraction": (count,),
            "line_mixture_fraction_defined": (count,),
            "alternate_single_phase_source": (count, dimension),
            "alternate_single_phase_source_defined": (count,),
        }
        for name, expected_shape in expected_shapes.items():
            source = np.asarray(getattr(chunk_result, name))
            if source.shape != expected_shape:
                raise PipelineStateError(
                    "worker chunk field %s has shape %r instead of %r"
                    % (name, source.shape, expected_shape)
                )
            destination = getattr(buffer, name)
            if source.dtype != destination.dtype:
                raise PipelineStateError(
                    "worker chunk field %s has dtype %s instead of %s"
                    % (name, source.dtype, destination.dtype)
                )
            destination.reshape((size,) + destination.shape[len(self._plan.grid.shape) :])[
                start:stop
            ] = source
        if set(chunk_result.counters) != set(counters):
            raise PipelineStateError("worker returned an invalid counter schema")
        if set(chunk_result.timing_seconds) != set(worker_timings):
            raise PipelineStateError("worker returned an invalid timing schema")
        for name, value in chunk_result.counters.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise PipelineStateError("worker returned an invalid counter value")
            counters[name] += int(value)
        for name, value in chunk_result.timing_seconds.items():
            converted = float(value)
            if not math.isfinite(converted) or converted < 0.0:
                raise PipelineStateError("worker returned an invalid timing value")
            worker_timings[name] += converted
            performance_observations["chunk_stage_samples"][name].append(
                converted
            )
        chunk_total = float(sum(chunk_result.timing_seconds.values()))
        performance_observations["chunk_stage_totals"].append(chunk_total)
        setup_seconds = float(chunk_result.worker_setup_seconds)
        if not math.isfinite(setup_seconds) or setup_seconds < 0.0:
            raise PipelineStateError(
                "worker returned an invalid setup duration"
            )
        if setup_seconds > 0.0:
            performance_observations["worker_setup_samples"].append(
                setup_seconds
            )
        peak_rss = chunk_result.worker_peak_rss_bytes
        if (
            isinstance(peak_rss, (bool, np.bool_))
            or not isinstance(peak_rss, (int, np.integer))
            or int(peak_rss) < 0
        ):
            raise PipelineStateError("worker returned an invalid peak RSS")
        performance_observations["peak_rss_bytes"] = max(
            int(performance_observations["peak_rss_bytes"]), int(peak_rss)
        )
        for records, destination in (
            (chunk_result.error_records, error_records),
            (
                chunk_result.projection_failure_records,
                projection_failure_records,
            ),
        ):
            for record in records:
                flat_index = int(record.get("flat_index", -1))
                if flat_index < start or flat_index >= stop:
                    raise PipelineStateError(
                        "worker returned an out-of-range diagnostic index"
                    )
                destination.append(dict(record))

    def run(
        self,
        *,
        checkpoint: Optional[Any] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PipelineResult:
        """Execute serially or with a bounded spawn-safe CPU process pool.

        ``checkpoint`` may be a :class:`~eos_trensmuther.CheckpointConfig`.  The
        parent process then validates and restores complete atomic chunks before
        scheduling only the missing flat ranges.
        """

        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")
        if self._plan.config.retain_detailed_diagnostics:
            raise PipelineConfigurationError(
                "retain_detailed_diagnostics=True is not yet supported by the "
                "compact pipeline result schema"
            )
        self._solved_seed_cache.clear()
        execution = self._plan.config.execution
        expected_chunk_count = int(
            math.ceil(self._plan.grid.size / execution.chunk_size)
        )
        run_started = time.perf_counter()
        buffer = self._plan.allocate_result_buffer()
        counters = {
            "chunks": 0,
            "prescreen_evaluated": 0,
            "prescreen_excluded": 0,
            "exact_candidates": 0,
            "tolerance_shell_candidates": 0,
            "mixed_prescreen_evaluated": 0,
            "mixed_candidates": 0,
            "exact_mixed_candidates": 0,
            "invalid_targets": 0,
            "fast_attempted": 0,
            "fallback_attempted": 0,
            "phase_attempted": 0,
            "mixed_roots": 0,
            "single_phase_roots": 0,
            "phase_boundary_roots": 0,
            "phase_multivalued_roots": 0,
            "jacobian_selected_roots": 0,
            "line_fraction_defined": 0,
            "line_auxiliary_values_defined": 0,
            "line_auxiliary_values_undefined": 0,
            "projection_attempted": 0,
            "auxiliary_points_evaluated": 0,
            "tilde_points_evaluated": 0,
        }
        counters.update({name: 0 for name in _SEED_COUNTER_NAMES})
        counters.update({name: 0 for name in _SUBCELL_COUNTER_NAMES})
        worker_timings = {
            "prescreen": 0.0,
            "initial_guess": 0.0,
            "phase_solve": 0.0,
            "fast": 0.0,
            "fallback": 0.0,
            "projection": 0.0,
            "output_evaluation": 0.0,
        }
        error_records = []
        projection_failure_records = []
        performance_observations = {
            "chunk_stage_totals": [],
            "chunk_stage_samples": {
                name: [] for name in worker_timings
            },
            "worker_setup_samples": [],
            "peak_rss_bytes": 0,
        }
        merged_ranges = set()
        computed_ranges = set()
        checkpoint_store = None
        if checkpoint is not None:
            from .checkpoint import CheckpointConfig, CheckpointStore

            if not isinstance(checkpoint, CheckpointConfig):
                raise TypeError("checkpoint must be a CheckpointConfig or None")
            checkpoint_store = CheckpointStore(
                checkpoint,
                self._plan,
                table_fingerprint=self._table_fingerprint,
                reachability_configuration={
                    item.name: getattr(self._reachability_config, item.name)
                    for item in dataclass_fields(self._reachability_config)
                },
                first_order_contract=(
                    self._first_order_manifest
                    if self._first_order_manifest["configured"]
                    else {}
                ),
            )
        resource_setup_seconds = 0.0
        immutable_state_backend = "private_validated_arrays"
        shared_state_manifest_path: Optional[str] = None
        shared_state_temporary: Optional[tempfile.TemporaryDirectory] = None
        process_schedule: Optional[DynamicScheduleReport] = None
        dispatch_started = time.perf_counter()

        def merge(
            chunk_result: _PipelineChunkResult,
            expected: Tuple[int, int],
            *,
            newly_computed: bool,
        ) -> None:
            if expected in merged_ranges:
                raise PipelineStateError("a target-grid chunk was returned twice")
            self._merge_chunk_result(
                buffer,
                chunk_result,
                expected,
                counters,
                worker_timings,
                performance_observations,
                error_records,
                projection_failure_records,
            )
            if newly_computed and checkpoint_store is not None:
                checkpoint_store.save_chunk(chunk_result)
            merged_ranges.add(expected)
            if newly_computed:
                computed_ranges.add(expected)
            if progress_callback is not None:
                completed_points = sum(stop - start for start, stop in merged_ranges)
                progress_callback(completed_points, self._plan.grid.size)

        if checkpoint_store is not None:
            for restored_chunk in checkpoint_store.load_completed_chunks():
                merge(
                    restored_chunk,
                    (restored_chunk.start, restored_chunk.stop),
                    newly_computed=False,
                )

        restored_ranges = set(merged_ranges)

        def chunks_to_execute() -> Iterator[GridChunk]:
            for chunk in self._plan.iter_chunks():
                if (chunk.start, chunk.stop) not in merged_ranges:
                    yield chunk

        missing_chunk_count = expected_chunk_count - len(merged_ranges)

        if missing_chunk_count == 0:
            execution_mode = "checkpoint_resume_only"
            effective_pending = 0
            start_method = "not_started"
        elif execution.workers == 1:
            from threadpoolctl import threadpool_limits

            with _THREAD_LIMIT_CONFIGURATION_LOCK:
                with threadpool_limits(limits=1):
                    resource_setup_seconds = self._prepare()
                    for chunk in chunks_to_execute():
                        merge(
                            self._execute_chunk(chunk),
                            (chunk.start, chunk.stop),
                            newly_computed=True,
                        )
            execution_mode = "serial"
            effective_pending = 1
            start_method = "in_process"
        else:
            available_methods = set(mp.get_all_start_methods())
            requested_method = execution.start_method.value
            if requested_method not in available_methods:
                raise PipelineConfigurationError(
                    "process start method '%s' is unavailable on this platform; "
                    "available methods are %s"
                    % (requested_method, ", ".join(sorted(available_methods)))
                )
            effective_pending = (
                execution.max_pending_chunks
                if execution.max_pending_chunks is not None
                else max(1, 2 * execution.workers)
            )
            context = mp.get_context(requested_method)
            if self._table.first_order_transition is None:
                from .shared_state import (
                    attach_table_to_shared_state,
                    create_shared_worker_state,
                )

                if checkpoint_store is not None:
                    shared_state_directory = (
                        checkpoint_store.directory.parent
                        / "shared_worker_state_v1"
                    )
                else:
                    shared_state_temporary = tempfile.TemporaryDirectory(
                        prefix="eosinverter-shared-state-"
                    )
                    shared_state_directory = (
                        Path(shared_state_temporary.name) / "state"
                    )
                shared_manifest = create_shared_worker_state(
                    self._table,
                    self._plan,
                    self._reachability_config,
                    shared_state_directory,
                )
                if checkpoint_store is not None:
                    # Persistent production stores also replace the parent's
                    # private raw/canonical copies.  Ephemeral test stores are
                    # removed with the pool and therefore must not become the
                    # long-lived backing of the caller-owned table object.
                    attach_table_to_shared_state(self._table, shared_manifest)
                shared_state_manifest_path = str(shared_manifest)
                immutable_state_backend = "numpy_npy_mmap_mode_r"
            payload = self._worker_initialization(
                shared_state_manifest=shared_state_manifest_path
            )
            executor: Optional[ProcessPoolExecutor] = None
            pool_started = time.perf_counter()
            try:
                with _worker_thread_environment():
                    executor = ProcessPoolExecutor(
                        max_workers=execution.workers,
                        mp_context=context,
                        initializer=_initialize_pipeline_worker,
                        initargs=(payload,),
                    )

                    def publish_checkpoint_chunk(
                        checkpoint_chunk: GridChunk,
                        assembled: _PipelineChunkResult,
                    ) -> None:
                        merge(
                            assembled,
                            (checkpoint_chunk.start, checkpoint_chunk.stop),
                            newly_computed=True,
                        )

                    process_schedule = execute_dynamic_tasks(
                        executor,
                        chunks_to_execute(),
                        split_chunk=self._split_checkpoint_chunk,
                        run_task=_run_pipeline_worker_chunk,
                        assemble_chunk=self._assemble_checkpoint_chunk,
                        publish_chunk=publish_checkpoint_chunk,
                        maximum_pending_tasks=effective_pending,
                        failure_type=PipelineStateError,
                    )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
                if shared_state_temporary is not None:
                    shared_state_temporary.cleanup()
            resource_setup_seconds = float(time.perf_counter() - pool_started)
            execution_mode = "process"
            start_method = requested_method

        dispatch_seconds = float(time.perf_counter() - dispatch_started)
        if (
            len(merged_ranges) != expected_chunk_count
            or counters["chunks"] != expected_chunk_count
            or sum(stop - start for start, stop in merged_ranges)
            != self._plan.grid.size
        ):
            raise PipelineStateError(
                "chunk aggregation did not cover the target grid exactly once"
            )
        error_records.sort(
            key=lambda item: (
                int(item["flat_index"]),
                str(item.get("stage", "")),
                str(item.get("exception_type", "")),
            )
        )
        projection_failure_records.sort(
            key=lambda item: (
                int(item["flat_index"]),
                str(item.get("exception_type", "")),
            )
        )
        summary = self._status_summary(buffer)
        from scipy import __version__ as scipy_version

        cell_shape = tuple(size - 1 for size in self._table.shape)
        reachability_config = {
            item.name: getattr(self._reachability_config, item.name)
            for item in dataclass_fields(self._reachability_config)
        }
        total_seconds = float(time.perf_counter() - run_started)
        workers_used = (
            0
            if execution_mode == "checkpoint_resume_only"
            else 1
            if execution_mode == "serial"
            else min(
                execution.workers,
                (
                    0
                    if process_schedule is None
                    else process_schedule.submitted_task_count
                ),
            )
        )
        setup_samples = list(
            performance_observations["worker_setup_samples"]
        )
        if execution_mode == "serial":
            setup_samples = [resource_setup_seconds]
        route_counts = summary["route"]
        disposition_counts = summary["disposition"]
        performance = {
            "sample_scope": (
                "all_result_chunks; checkpoint-restored chunks retain their "
                "original stage/setup/RSS observations"
            ),
            "computed_points_per_second_this_invocation": (
                float(sum(stop - start for start, stop in computed_ranges))
                / total_seconds
                if total_seconds > 0.0
                else 0.0
            ),
            "restored_points": int(
                sum(stop - start for start, stop in restored_ranges)
            ),
            "computed_points_this_invocation": int(
                sum(stop - start for start, stop in computed_ranges)
            ),
            "rates": {
                "prescreen_excluded": _fraction(
                    counters["prescreen_excluded"],
                    counters["prescreen_evaluated"],
                ),
                "prescreen_exact_candidate": _fraction(
                    counters["exact_candidates"],
                    counters["prescreen_evaluated"],
                ),
                "prescreen_tolerance_shell": _fraction(
                    counters["tolerance_shell_candidates"],
                    counters["prescreen_evaluated"],
                ),
                "fast_success": _fraction(
                    route_counts[PointRoute.FAST.value],
                    counters["fast_attempted"],
                ),
                "fallback_after_fast": _fraction(
                    counters["fallback_attempted"],
                    counters["fast_attempted"],
                ),
                "fallback_solved": _fraction(
                    route_counts[PointRoute.FALLBACK_CERTIFIED.value],
                    counters["fallback_attempted"],
                ),
                "projection_success": _fraction(
                    disposition_counts[PointDisposition.PROJECTED.value],
                    counters["projection_attempted"],
                ),
            },
            "chunk_processing_seconds": _duration_statistics(
                performance_observations["chunk_stage_totals"]
            ),
            "chunk_stage_seconds": {
                name: _duration_statistics(values)
                for name, values in performance_observations[
                    "chunk_stage_samples"
                ].items()
            },
            "worker_initialization_seconds": _duration_statistics(
                setup_samples
            ),
            "observed_peak_rss_bytes": {
                "parent_before_result_finalization": _process_peak_rss_bytes(),
                "maximum_reporting_worker": int(
                    performance_observations["peak_rss_bytes"]
                ),
                "measurement": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
                "normalization": (
                    "bytes_on_darwin; kibibytes_multiplied_by_1024_elsewhere"
                ),
                "caveat": (
                    "process-local lifetime peak; worker value is the maximum "
                    "reported by a completed chunk"
                ),
            },
            "memory_estimates": self._persistent_memory_estimates(
                workers_used,
                mmap_shared=immutable_state_backend
                == "numpy_npy_mmap_mode_r",
            ),
        }
        run_manifest = {
            "pipeline_run_manifest_version": PIPELINE_RUN_MANIFEST_VERSION,
            "completed": True,
            "execution": {
                "mode": execution_mode,
                "workers_requested": execution.workers,
                "workers_used_upper_bound": workers_used,
                "executor_max_workers": (
                    0
                    if execution_mode == "checkpoint_resume_only"
                    else 1
                    if execution_mode == "serial"
                    else execution.workers
                ),
                "maximum_concurrent_chunks": min(
                    1, missing_chunk_count
                )
                if process_schedule is None
                else process_schedule.maximum_active_chunks_observed,
                "maximum_concurrent_internal_batches": (
                    0
                    if process_schedule is None
                    else process_schedule.maximum_pending_tasks_observed
                ),
                "submitted_worker_tasks": (
                    0
                    if process_schedule is None
                    else process_schedule.submitted_task_count
                ),
                "completed_worker_tasks": (
                    0
                    if process_schedule is None
                    else process_schedule.completed_task_count
                ),
                "internal_batch_size": execution.worker_task_size,
                "start_method": start_method,
                "chunk_size": execution.chunk_size,
                "max_pending_chunks": effective_pending,
                "error_policy": execution.error_policy.value,
                "nested_numerical_threads_per_worker": 1,
                "aggregation": (
                    "cross_chunk_dynamic_tasks_parent_ordered_then_atomic_publish"
                    if execution_mode == "process"
                    else "parent_only_flat_range"
                ),
                "immutable_worker_state_backend": immutable_state_backend,
                "shared_state_manifest": shared_state_manifest_path,
            },
            "checkpoint": {
                "enabled": checkpoint_store is not None,
                "restored_chunks": len(restored_ranges),
                "computed_chunks_this_invocation": len(computed_ranges),
                "expected_chunks": expected_chunk_count,
                "run_fingerprint": (
                    None
                    if checkpoint_store is None
                    else checkpoint_store.run_fingerprint
                ),
                "layout_fingerprint": (
                    None
                    if checkpoint_store is None
                    else checkpoint_store.layout_fingerprint
                ),
                "error_rows_reused_without_automatic_retry": True,
            },
            "active_domain": {
                "kind": "full_cell_grid",
                "cell_shape": list(cell_shape),
                "active_cell_count": int(math.prod(cell_shape)),
                "restricted": False,
            },
            "table_snapshot": {
                "fingerprint_algorithm": "sha256",
                "fingerprint_schema": "forward_table_snapshot_v%d"
                % TABLE_SNAPSHOT_FINGERPRINT_VERSION,
                "fingerprint_purpose": (
                    "worker_reload_and_persistence_compatibility"
                ),
                "snapshot_fingerprint": self._table_fingerprint,
                "stable_persistence_identifier": True,
            },
            "reachability": {
                "method": self._reachability_config.method,
                "implementation": (
                    "balanced_source_index_hierarchy_exact_leaf"
                    if self._reachability_config.method == "hierarchical"
                    else "batched_all_cell_reference"
                ),
                "configuration": reachability_config,
                "rigorous_exact_boxes": True,
                "routing_outward_rounded": bool(
                    self._plan.config.solver.root_residual_tolerance > 0.0
                ),
                "complete_leaf_localization": {
                    "enabled": self._uses_complete_ordinary_localization(),
                    "scope": (
                        "ordinary_dimensions_1_to_4"
                        if self._uses_complete_ordinary_localization()
                        else None
                    ),
                    "backend": (
                        self._reachability_config.method
                        if self._uses_complete_ordinary_localization()
                        else None
                    ),
                    "lifetime": (
                        "worker_chunk_only"
                        if self._uses_complete_ordinary_localization()
                        else None
                    ),
                    "persisted_in_checkpoint": False,
                },
            },
            "first_order": dict(self._first_order_manifest),
            "auxiliary_evaluation": summary["auxiliary_phase_policy"],
            "initial_guess": {
                "precedence": (
                    [
                        "direct_solved_inverse_grid_neighbor",
                        "best_complete_candidate_cell_center",
                    ]
                    if self._uses_complete_ordinary_localization()
                    and self._plan.config.use_reachability_seed
                    else [
                        "direct_solved_inverse_grid_neighbor",
                        "precomputed_coarse_grid_seed",
                        "generic_configured_initial_guess",
                    ]
                ),
                "use_reachability_seed": (
                    self._plan.config.use_reachability_seed
                    if self._uses_complete_ordinary_localization()
                    else False
                ),
                "fast_attempt_policy": (
                    "exactly_one_selected_seed"
                    if self._uses_complete_ordinary_localization()
                    else "bounded_seed_retries_after_failure"
                ),
                "fallback_domain": (
                    "complete_reachability_routing_cell_set"
                    if self._uses_complete_ordinary_localization()
                    else "global_source_cell_scan"
                ),
                "direct_neighbor_priority": [
                    "%s_%s" % (name, direction)
                    for name in self._plan.grid.axis_names
                    for direction in ("minus", "plus")
                ],
                "direct_neighbor_eligibility": (
                    "terminal_solved_and_strictly_interior_source_point"
                    if self._uses_complete_ordinary_localization()
                    else "terminal_solved_only"
                ),
                "coarse_configuration": {
                    item.name: getattr(self._plan.config.coarse_seed, item.name)
                    for item in dataclass_fields(self._plan.config.coarse_seed)
                },
                "coarse_precomputation": dict(self._coarse_seed_report),
                "generic_policy": self._plan.config.initial_guess.value,
                "global_source_node_lookup": False,
                "role": "convergence_efficiency_not_branch_selection",
                "multivalued_branch_selection": (
                    "unanimous_local_jacobian_orientation_only"
                ),
            },
            "projection": {
                "enabled": self._plan.config.projection.enabled,
                "method": (
                    "deterministic_local_multistart"
                    if self._plan.config.projection.enabled
                    else None
                ),
                "globally_certified": False,
            },
            "software": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "numpy": np.__version__,
                "scipy": scipy_version,
                "platform": sys.platform,
            },
            "timing_seconds": {
                "planning": self._planning_seconds,
                "serial_resource_setup": (
                    resource_setup_seconds if execution_mode == "serial" else 0.0
                ),
                "process_pool_lifetime": (
                    resource_setup_seconds if execution_mode == "process" else 0.0
                ),
                "parent_dispatch_and_aggregation": dispatch_seconds,
                "worker_stage_sum": worker_timings,
                "serial_setup_stages": (
                    dict(self._setup_timing_seconds)
                    if execution_mode == "serial"
                    else {}
                ),
                "total_run": total_seconds,
            },
            "performance": performance,
            "stage_counters": counters,
            "status_counts": {
                name: count
                for name, count in summary["disposition"].items()
                if name != "unassigned"
            },
            "status_summary": summary,
            "errors": {
                "count": len(error_records),
                "records": error_records,
            },
            "projection_failures": {
                "count": len(projection_failure_records),
                "records": projection_failure_records,
            },
            "detailed_diagnostics_retained": False,
        }
        result = buffer.finalize(run_manifest)
        expected_dispositions = {
            name: count
            for name, count in summary["disposition"].items()
            if name != "unassigned"
        }
        if dict(result.status_counts) != expected_dispositions:
            raise PipelineStateError(
                "final result status counts disagree with the run manifest"
            )
        if checkpoint_store is not None:
            checkpoint_store.mark_complete()
        return result


def _pipeline_config_from_dict(value: Mapping[str, Any]) -> PipelineConfig:
    """Rebuild immutable configuration inside a spawned worker."""

    return PipelineConfig.from_dict(value)


@contextmanager
def _worker_thread_environment() -> Iterator[None]:
    """Temporarily make pre-import thread limits inheritable by children."""

    # os.environ and threadpoolctl both affect process-global state.  Holding
    # one lock makes concurrent pipeline calls from parent threads safe; their
    # process-pool lifetimes are intentionally serialized.
    with _THREAD_LIMIT_CONFIGURATION_LOCK:
        previous = {
            name: os.environ.get(name) for name in _WORKER_THREAD_ENVIRONMENT
        }
        try:
            for name in _WORKER_THREAD_ENVIRONMENT:
                os.environ[name] = "1"
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


_PIPELINE_WORKER: Optional[EoSInversionPipeline] = None
_PIPELINE_WORKER_THREAD_LIMITER: Optional[Any] = None
_PIPELINE_WORKER_INITIALIZATION_ERROR: Optional[str] = None
_PIPELINE_WORKER_SETUP_SECONDS = 0.0
_PIPELINE_WORKER_SETUP_REPORTED = False
_PIPELINE_WORKER_SHARED_STATE_REPORT: Optional[Dict[str, Any]] = None


def _initialize_pipeline_worker(payload: _WorkerInitialization) -> None:
    """Spawn-safe initializer that constructs one persistent numerical stack."""

    setup_started = time.perf_counter()
    global _PIPELINE_WORKER
    global _PIPELINE_WORKER_INITIALIZATION_ERROR
    global _PIPELINE_WORKER_SETUP_REPORTED
    global _PIPELINE_WORKER_SETUP_SECONDS
    global _PIPELINE_WORKER_THREAD_LIMITER
    global _PIPELINE_WORKER_SHARED_STATE_REPORT
    _PIPELINE_WORKER = None
    _PIPELINE_WORKER_INITIALIZATION_ERROR = None
    _PIPELINE_WORKER_THREAD_LIMITER = None
    _PIPELINE_WORKER_SETUP_SECONDS = 0.0
    _PIPELINE_WORKER_SETUP_REPORTED = False
    _PIPELINE_WORKER_SHARED_STATE_REPORT = None
    try:
        if not isinstance(payload, _WorkerInitialization):
            raise PipelineStateError("worker initialization payload is invalid")
        for name in _WORKER_THREAD_ENVIRONMENT:
            os.environ[name] = "1"
        from threadpoolctl import threadpool_limits

        _PIPELINE_WORKER_THREAD_LIMITER = threadpool_limits(limits=1)
        if payload.shared_state_manifest is None:
            table = ForwardTable.load_validated(
                Path(payload.metadata_path), data_path=Path(payload.data_path)
            )
        else:
            from .shared_state import load_shared_table

            table = load_shared_table(payload.shared_state_manifest)
        fingerprint = _validated_table_fingerprint(table)
        if fingerprint != payload.table_fingerprint:
            raise PipelineStateError(
                "worker table snapshot does not match the coordinator fingerprint"
            )
        configuration = _pipeline_config_from_dict(
            payload.pipeline_configuration
        )
        from .reachability import ReachabilityConfig

        reachability = ReachabilityConfig(
            **payload.reachability_configuration
        )
        pipeline = EoSInversionPipeline(
            table, configuration, reachability_config=reachability
        )
        if pipeline._table_fingerprint != payload.table_fingerprint:
            raise PipelineStateError(
                "worker pipeline changed the validated table snapshot"
            )
        if pipeline._first_order_manifest != payload.first_order_contract:
            raise PipelineStateError(
                "worker transition contract does not match the coordinator"
            )
        pipeline._prepare(
            shared_state_manifest=payload.shared_state_manifest
        )
        if payload.shared_state_manifest is not None:
            from .shared_state import shared_state_backing_report

            assert pipeline._reachability is not None
            _PIPELINE_WORKER_SHARED_STATE_REPORT = shared_state_backing_report(
                table, pipeline._reachability
            )
            if not _PIPELINE_WORKER_SHARED_STATE_REPORT[
                "all_large_arrays_mmap_backed"
            ]:
                raise PipelineStateError(
                    "spawned worker materialized a private copy of shared large arrays"
                )
    except Exception as exc:
        # An initializer that raises causes ProcessPoolExecutor to repeatedly
        # respawn workers and eventually exposes only BrokenProcessPool.  Keep
        # the process alive long enough for its first task to return the real,
        # picklable setup failure to the coordinator.
        _PIPELINE_WORKER_INITIALIZATION_ERROR = "%s: %s" % (
            type(exc).__name__,
            exc,
        )
        return
    _PIPELINE_WORKER = pipeline
    _PIPELINE_WORKER_SETUP_SECONDS = float(
        time.perf_counter() - setup_started
    )


def _run_pipeline_worker_chunk(chunk: GridChunk) -> _PipelineChunkResult:
    """Module-level task entry used by ProcessPoolExecutor."""

    if _PIPELINE_WORKER_INITIALIZATION_ERROR is not None:
        raise PipelineStateError(
            "process worker initialization failed: %s"
            % _PIPELINE_WORKER_INITIALIZATION_ERROR
        )
    if _PIPELINE_WORKER is None:
        raise PipelineStateError("process worker was not initialized")
    # Internal batch boundaries are fixed independently of worker count.
    # Scoping opportunistic continuation to the task prevents scheduling from
    # changing which cross-batch neighbor happens to be available, while
    # retaining the accepted direct-neighbor override within every batch.
    _PIPELINE_WORKER._solved_seed_cache.clear()
    result = _PIPELINE_WORKER._execute_chunk(chunk)
    global _PIPELINE_WORKER_SETUP_REPORTED
    if not _PIPELINE_WORKER_SETUP_REPORTED:
        result.worker_setup_seconds = _PIPELINE_WORKER_SETUP_SECONDS
        _PIPELINE_WORKER_SETUP_REPORTED = True
    return result


def _pipeline_worker_state_diagnostics(
    hold_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Return process/backing evidence for tests and memory benchmarks."""

    if _PIPELINE_WORKER_INITIALIZATION_ERROR is not None:
        raise PipelineStateError(
            "process worker initialization failed: %s"
            % _PIPELINE_WORKER_INITIALIZATION_ERROR
        )
    if _PIPELINE_WORKER is None:
        raise PipelineStateError("process worker was not initialized")
    delay = float(hold_seconds)
    if not math.isfinite(delay) or delay < 0.0 or delay > 60.0:
        raise PipelineStateError("diagnostic hold_seconds is invalid")
    diagnostics = {
        "pid": os.getpid(),
        "peak_rss_bytes": _process_peak_rss_bytes(),
        "memory": _process_memory_snapshot(),
        "setup_seconds": _PIPELINE_WORKER_SETUP_SECONDS,
        "shared_state": _PIPELINE_WORKER_SHARED_STATE_REPORT,
    }
    if delay:
        time.sleep(delay)
    return diagnostics


__all__ = [
    "DEFAULT_MAX_GRID_POINTS",
    "DEFAULT_MAX_RESULT_BYTES",
    "PIPELINE_RESULT_SCHEMA_VERSION",
    "PIPELINE_RUN_MANIFEST_VERSION",
    "PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION",
    "CoarseSeedConfig",
    "EoSInversionPipeline",
    "ErrorPolicy",
    "ExecutionConfig",
    "GridAxisSpec",
    "GridChunk",
    "GridSpace",
    "InitialGuessPolicy",
    "InitialGuessSource",
    "LineAuxiliaryPolicy",
    "PhaseRoute",
    "InversionGrid",
    "InversionGridSpec",
    "PipelineConfig",
    "PipelineConfigurationError",
    "PipelineError",
    "PipelinePlan",
    "PipelineResult",
    "PipelineStateError",
    "PointDisposition",
    "PointRoute",
    "PrescreenStatus",
    "ProcessStartMethod",
    "ProjectionConfig",
    "ProjectionQuality",
    "pipeline_status_code",
]
