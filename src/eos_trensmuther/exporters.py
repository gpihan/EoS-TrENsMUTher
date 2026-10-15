"""Policy-explicit downstream serializers for completed inversion results.

The lossless ``.eosinv`` artifact remains authoritative. Text and binary
consumer exports retain the requested grid ordering, use named fields, and
record projected/undefined-value policies in a JSON manifest.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._version import __version__
from .pipeline import PipelineResult


TEXT_EXPORT_SCHEMA_VERSION = 6
TEXT_EXPORT_TYPE = "EoS-TrENsMUTher.text.table"

EXTRA_INFO_COLUMN_NAMES = (
    "projected",
    "certified_unreachable",
    "mixed_phase",
    "phase_boundary",
    "phase_multivalued",
    "selected_from_jacobian",
)

# Pre-v1 hydro-code-specific serializers remain below only to read/reproduce
# historical validation evidence. They are intentionally absent from the
# package root, CLI, configuration adapter, documentation, and this module's
# declared public export list.
VHLLE_EXPORT_SCHEMA_VERSION = 4
VHLLE_EXPORT_TYPE = "EoSInverter.vHLLE.table"
# This prefix stays stable; other retained auxiliaries are appended.
VHLLE_COLUMN_NAMES = ("e", "nB", "T", "muB", "P", "s", "cs2", "chi2")

MUSIC_EXPORT_SCHEMA_VERSION = 2
MUSIC_EXPORT_TYPE = "EoSInverter.MUSIC.table"
MUSIC_LEGACY_BINARY_FIELDS = ("t", "mub", "p", "s")
PHASE_EXPORT_SCHEMA_VERSION = 1
PHASE_EXPORT_COLUMN_NAMES = (
    "phase_code",
    "phase_route_code",
    "line_mixture_fraction",
    "phase_boundary",
    "phase_multivalued",
)


class ExportError(Exception):
    """Base exception for downstream export failures."""


class ExportConfigurationError(ExportError):
    """Raised when an export policy or field mapping is invalid."""


class ExportDataError(ExportError):
    """Raised when a completed result cannot satisfy a consumer schema."""


class ExportPathError(ExportError):
    """Raised when an output path cannot be published safely."""


def _freeze_mapping(value: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType({str(key): str(item) for key, item in value.items()})


def _public_unit_name(unit: str) -> str:
    """Return the explicit unit spelling used by public text exports."""

    return "dimensionless" if unit in {"1", "dimensionless"} else unit


def _validate_format(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExportConfigurationError("%s must be non-empty" % name)
    try:
        probe = value % 1.25
    except (TypeError, ValueError) as exc:
        raise ExportConfigurationError(
            "%s must format one floating-point value" % name
        ) from exc
    if not isinstance(probe, str) or not probe:
        raise ExportConfigurationError("%s is invalid" % name)
    return value


def _validate_field_map(
    value: Any, *, required: Sequence[str], label: str
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ExportConfigurationError("%s field_map must be a mapping" % label)
    normalized = _freeze_mapping(value)
    required_set = set(required)
    if set(normalized) != required_set:
        missing = sorted(required_set - set(normalized))
        extra = sorted(set(normalized) - required_set)
        details = []
        if missing:
            details.append("missing %s" % ", ".join(missing))
        if extra:
            details.append("unexpected %s" % ", ".join(extra))
        raise ExportConfigurationError(
            "%s field_map must define %s (%s)"
            % (label, ", ".join(required), "; ".join(details))
        )
    if any(not name for name in normalized.values()):
        raise ExportConfigurationError(
            "%s field_map values must be non-empty field names" % label
        )
    if len(set(normalized.values())) != len(normalized):
        raise ExportConfigurationError(
            "%s field_map cannot map multiple columns to one field" % label
        )
    return normalized


@dataclass(frozen=True)
class PhaseExportConfig:
    """Explicit opt-in schema for phase columns in consumer outputs."""

    schema_version: int = PHASE_EXPORT_SCHEMA_VERSION
    columns: Tuple[str, ...] = PHASE_EXPORT_COLUMN_NAMES

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, (int, np.integer))
            or int(self.schema_version) != PHASE_EXPORT_SCHEMA_VERSION
        ):
            raise ExportConfigurationError(
                "phase export schema_version must be %d"
                % PHASE_EXPORT_SCHEMA_VERSION
            )
        if isinstance(self.columns, (str, bytes)) or not isinstance(
            self.columns, Sequence
        ):
            raise ExportConfigurationError(
                "phase export columns must be a sequence"
            )
        columns = tuple(self.columns)
        if not columns:
            raise ExportConfigurationError(
                "phase export columns cannot be empty; omit phase export instead"
            )
        if any(not isinstance(name, str) or not name for name in columns):
            raise ExportConfigurationError(
                "phase export column names must be non-empty strings"
            )
        if len(set(columns)) != len(columns):
            raise ExportConfigurationError(
                "phase export columns cannot contain duplicates"
            )
        unknown = sorted(set(columns) - set(PHASE_EXPORT_COLUMN_NAMES))
        if unknown:
            raise ExportConfigurationError(
                "unsupported phase export column(s): %s" % ", ".join(unknown)
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "columns", columns)


@dataclass(frozen=True)
class VHLLEExportConfig:
    """Policy for vHLLE text output with a stable legacy prefix."""

    field_map: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType(
            {"P": "P", "s": "s", "cs2": "cs2", "chi2": "chi2B"}
        )
    )
    allow_projected: bool = False
    allow_unusable: bool = False
    allow_undefined_auxiliary: bool = False
    write_header: bool = False
    write_manifest: bool = True
    float_format: str = "%.8e"
    phase: Optional[PhaseExportConfig] = None

    def __post_init__(self) -> None:
        normalized = _validate_field_map(
            self.field_map,
            required=("P", "s", "cs2", "chi2"),
            label="vHLLE",
        )
        for name in (
            "allow_projected",
            "allow_unusable",
            "allow_undefined_auxiliary",
            "write_header",
            "write_manifest",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ExportConfigurationError("%s must be boolean" % name)
        object.__setattr__(
            self, "float_format", _validate_format(self.float_format, "float_format")
        )
        object.__setattr__(self, "field_map", normalized)
        if self.phase is not None and not isinstance(
            self.phase, PhaseExportConfig
        ):
            raise ExportConfigurationError(
                "vHLLE phase must be a PhaseExportConfig or None"
            )


@dataclass(frozen=True)
class TextExportConfig:
    """Policy for the public, generic column-based text export."""

    # Deprecated compatibility input. Its value is ignored: projected states
    # are included and explicitly tagged rather than requiring an opt-in.
    include_projected: bool = True
    include_unusable: bool = False
    allow_undefined_auxiliary: bool = False
    write_header: bool = True
    write_manifest: bool = True
    float_format: str = "%.8e"
    phase: Optional[PhaseExportConfig] = None
    extra_info: bool = True
    include_tilde: bool = False

    def __post_init__(self) -> None:
        for name in (
            "include_projected",
            "include_unusable",
            "allow_undefined_auxiliary",
            "write_header",
            "write_manifest",
            "extra_info",
            "include_tilde",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ExportConfigurationError("%s must be boolean" % name)
        object.__setattr__(
            self, "float_format", _validate_format(self.float_format, "float_format")
        )
        if self.phase is not None and not isinstance(self.phase, PhaseExportConfig):
            raise ExportConfigurationError(
                "text-export phase must be a PhaseExportConfig or None"
            )


@dataclass(frozen=True)
class MUSICExportConfig:
    """Policy for legacy MUSIC float32 binaries plus the full text table."""

    field_map: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({"P": "P", "s": "s"})
    )
    allow_projected: bool = False
    allow_undefined_auxiliary: bool = False
    write_header: bool = False
    write_manifest: bool = True
    float_format: str = "%.8e"
    phase: Optional[PhaseExportConfig] = None

    def __post_init__(self) -> None:
        normalized = _validate_field_map(
            self.field_map, required=("P", "s"), label="MUSIC"
        )
        for name in (
            "allow_projected",
            "allow_undefined_auxiliary",
            "write_header",
            "write_manifest",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ExportConfigurationError("%s must be boolean" % name)
        object.__setattr__(
            self, "float_format", _validate_format(self.float_format, "float_format")
        )
        object.__setattr__(self, "field_map", normalized)
        if self.phase is not None and not isinstance(
            self.phase, PhaseExportConfig
        ):
            raise ExportConfigurationError(
                "MUSIC phase must be a PhaseExportConfig or None"
            )


@dataclass(frozen=True)
class VHLLEExportResult:
    """Paths and counts produced by one successful vHLLE export."""

    table_path: Path
    manifest_path: Optional[Path]
    row_count: int
    solved_count: int
    projected_count: int
    unusable_count: int
    undefined_auxiliary_count: int
    undefined_phase_count: int
    sha256: str
    column_names: Tuple[str, ...]
    phase_column_names: Tuple[str, ...]


@dataclass(frozen=True)
class TextExportResult:
    """Paths, layout, and counts produced by a generic text export."""

    table_path: Path
    manifest_path: Optional[Path]
    row_count: int
    solved_count: int
    projected_count: int
    unusable_count: int
    undefined_auxiliary_count: int
    undefined_phase_count: int
    sha256: str
    column_names: Tuple[str, ...]
    column_units: Tuple[str, ...]
    phase_column_names: Tuple[str, ...]
    extra_info_column_names: Tuple[str, ...]


@dataclass(frozen=True)
class MUSICExportResult:
    """Paths and counts produced by one successful MUSIC export."""

    table_path: Path
    manifest_path: Optional[Path]
    binary_paths: Mapping[str, Path]
    row_count: int
    solved_count: int
    projected_count: int
    undefined_auxiliary_count: int
    undefined_phase_count: int
    table_sha256: str
    binary_sha256: Mapping[str, str]
    column_names: Tuple[str, ...]
    phase_column_names: Tuple[str, ...]


@dataclass(frozen=True)
class _PreparedTable:
    matrix: np.ndarray
    defined: np.ndarray
    column_names: Tuple[str, ...]
    column_units: Tuple[str, ...]
    source_fields: Tuple[str, ...]
    column_roles: Tuple[str, ...]
    auxiliary_indices: Tuple[int, ...]
    solved_count: int
    projected_count: int
    unusable_count: int
    undefined_count: int
    undefined_phase_count: int


def _absolute_output_path(value: Any, label: str = "output") -> Path:
    if isinstance(value, bool):
        raise TypeError("%s path must be path-like" % label)
    try:
        path = Path(value).expanduser()
    except (TypeError, ValueError) as exc:
        raise TypeError("%s path must be path-like" % label) from exc
    path = Path(os.path.abspath(os.fspath(path)))
    if not path.name:
        raise ExportPathError("%s path cannot be a filesystem root" % label)
    if not path.parent.is_dir():
        raise ExportPathError("%s parent does not exist: %s" % (label, path.parent))
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ExportPathError("%s must be a regular file: %s" % (label, path))
    return path


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(str(path), flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EBADF):
            raise ExportPathError("could not synchronize export directory") from exc


def _temporary_path(target: Path) -> Tuple[int, Path]:
    descriptor, name = tempfile.mkstemp(
        prefix=".%s." % target.name,
        suffix=".tmp",
        dir=str(target.parent),
    )
    return descriptor, Path(name)


def _publish(temporary: Path, target: Path, *, overwrite: bool) -> None:
    if overwrite:
        if target.exists() and (target.is_symlink() or not target.is_file()):
            raise ExportPathError("overwrite target must be a regular file")
        os.replace(str(temporary), str(target))
        return
    try:
        os.link(str(temporary), str(target))
    except FileExistsError as exc:
        raise FileExistsError("export already exists: %s" % target) from exc
    except OSError as exc:
        raise ExportPathError("could not publish export atomically: %s" % exc) from exc
    temporary.unlink()


def _requested_targets(result: PipelineResult) -> np.ndarray:
    chunks = [
        chunk.canonical_targets
        for chunk in result.grid.iter_chunks(result.plan.config.execution.chunk_size)
    ]
    if not chunks:
        return np.empty((0, result.grid.dimension), dtype=float)
    return np.concatenate(chunks, axis=0)


def _requested_tilde_coordinates(
    result: PipelineResult,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Return all requested tilde coordinates in canonical row order."""

    if result.grid.space.value != "tilde":
        raise ExportDataError(
            "text export can include tilde columns only for a tilde-grid run"
        )
    chunks = tuple(
        result.grid.iter_chunks(result.plan.config.execution.chunk_size)
    )
    if not chunks:
        values = np.empty((0, 4), dtype=float)
    else:
        values = np.concatenate(
            [
                np.column_stack(
                    (chunk.requested_values, chunk.requested_dependent_tilde)
                )
                for chunk in chunks
            ],
            axis=0,
        )
    names = result.grid.axis_names + result.grid.dependent_tilde_names
    if values.shape != (result.grid.size, 4) or len(names) != 4:
        raise ExportDataError(
            "tilde export requires T_tilde and all three chemical tilde values"
        )
    if not np.all(np.isfinite(values)):
        raise ExportDataError("requested tilde export columns are not finite")
    return values, names


def _validate_result_shape(
    result: PipelineResult,
    label: str,
    *,
    supported_dimensions: Tuple[int, ...] = (2,),
    allow_unusable: bool = False,
) -> None:
    if not isinstance(result, PipelineResult):
        raise TypeError("result must be a PipelineResult")
    dimension = result.grid.dimension
    if dimension not in supported_dimensions:
        rendered = ", ".join("%dD" % value for value in supported_dimensions)
        raise ExportDataError(
            "the %s exporter supports dimensions %s" % (label, rendered)
        )
    if dimension == 2:
        if result.plan.target_names != ("e", "nB"):
            raise ExportDataError("%s targets must be ordered as e, nB" % label)
        if result.plan.source_names != ("T", "muB"):
            raise ExportDataError(
                "%s source coordinates must be ordered as T, muB" % label
            )
    unusable = ~result.usable_state_mask
    unsupported_unusable = unusable & ~result.certified_unreachable_mask
    if np.any(unsupported_unusable):
        raise ExportDataError(
            "%s cannot encode %d unresolved, invalid, failed, or error row(s)"
            % (label, int(np.count_nonzero(unsupported_unusable)))
        )
    if np.any(unusable) and not allow_unusable:
        raise ExportDataError(
            "%s requires a usable solved/projected state at every grid slot; "
            "found %d unusable row(s); explicitly allow unusable rows to "
            "encode their source and auxiliary values as NaN"
            % (label, int(np.count_nonzero(unusable)))
        )


def _auxiliary_order(
    result: PipelineResult, priority: Sequence[Tuple[str, str]], label: str
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[int, ...]]:
    available = result.plan.auxiliary_names
    output_names = []
    source_fields = []
    indices = []
    for output_name, field_name in priority:
        try:
            index = available.index(field_name)
        except ValueError as exc:
            raise ExportDataError(
                "%s column '%s' maps to auxiliary field '%s', but the result "
                "contains only: %s"
                % (label, output_name, field_name, ", ".join(available) or "<none>")
            ) from exc
        output_names.append(output_name)
        source_fields.append(field_name)
        indices.append(index)
    prioritized = set(source_fields)
    for index, name in enumerate(available):
        if name not in prioritized:
            output_names.append(name)
            source_fields.append(name)
            indices.append(index)
    return tuple(output_names), tuple(source_fields), tuple(indices)


def _phase_columns(
    result: PipelineResult, config: Optional[PhaseExportConfig]
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    if config is None:
        return (
            np.empty((result.grid.size, 0), dtype=float),
            np.empty((result.grid.size, 0), dtype=bool),
            (),
            (),
        )
    values_by_name = {
        "phase_code": np.asarray(result.phase_codes, dtype=float).reshape(-1),
        "phase_route_code": np.asarray(
            result.phase_route_codes, dtype=float
        ).reshape(-1),
        "line_mixture_fraction": np.asarray(
            result.line_mixture_fraction, dtype=float
        ).reshape(-1),
        "phase_boundary": np.asarray(
            result.phase_boundary, dtype=float
        ).reshape(-1),
        "phase_multivalued": np.asarray(
            result.phase_multivalued, dtype=float
        ).reshape(-1),
    }
    defined_by_name = {
        "phase_code": np.ones(result.grid.size, dtype=bool),
        "phase_route_code": np.ones(result.grid.size, dtype=bool),
        "line_mixture_fraction": np.asarray(
            result.line_mixture_fraction_defined, dtype=bool
        ).reshape(-1),
        "phase_boundary": np.ones(result.grid.size, dtype=bool),
        "phase_multivalued": np.ones(result.grid.size, dtype=bool),
    }
    matrix = np.column_stack([values_by_name[name] for name in config.columns])
    defined = np.column_stack(
        [defined_by_name[name] for name in config.columns]
    )
    return matrix, defined, config.columns, config.columns


def _extra_info_columns(
    result: PipelineResult,
    enabled: bool,
    excluded_names: Sequence[str] = (),
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    if not enabled:
        return np.empty((result.grid.size, 0), dtype=float), ()
    values_by_name = dict(
        zip(
            EXTRA_INFO_COLUMN_NAMES,
            (
                result.projected_mask,
                result.certified_unreachable_mask,
                result.mixed_phase_mask,
                result.phase_boundary,
                result.phase_multivalued,
                result.jacobian_selected_mask,
            ),
        )
    )
    names = tuple(
        name for name in EXTRA_INFO_COLUMN_NAMES if name not in set(excluded_names)
    )
    matrix = np.column_stack(
        [
            np.asarray(values_by_name[name], dtype=np.uint8).reshape(-1)
            for name in names
        ]
    ).astype(float, copy=False)
    return matrix, names


def _prepare_table(
    result: PipelineResult,
    *,
    priority: Sequence[Tuple[str, str]],
    allow_projected: bool,
    allow_unusable: bool = False,
    allow_undefined_auxiliary: bool,
    phase: Optional[PhaseExportConfig],
    extra_info: bool = False,
    include_tilde: bool = False,
    label: str,
    supported_dimensions: Tuple[int, ...] = (2,),
) -> _PreparedTable:
    _validate_result_shape(
        result,
        label,
        supported_dimensions=supported_dimensions,
        allow_unusable=allow_unusable,
    )
    usable = np.asarray(result.usable_state_mask, dtype=bool).reshape(-1)
    unusable_count = int(np.count_nonzero(~usable))
    projected_count = int(np.count_nonzero(result.projected_mask))
    if projected_count and not allow_projected:
        raise ExportDataError(
            "result contains %d projected row(s); set allow_projected only if the "
            "consumer should use nearest admissible thermodynamics for unreachable "
            "requested targets" % projected_count
        )
    output_names, source_fields, indices = _auxiliary_order(result, priority, label)
    auxiliary_values = result.auxiliary_values.reshape(
        result.grid.size, len(result.plan.auxiliary_names)
    )[:, indices]
    auxiliary_defined = result.auxiliary_defined.reshape(
        result.grid.size, len(result.plan.auxiliary_names)
    )[:, indices]
    undefined_count = int(np.count_nonzero(~auxiliary_defined))
    undefined_on_usable = int(
        np.count_nonzero((~auxiliary_defined) & usable[:, None])
    )
    if undefined_on_usable and not allow_undefined_auxiliary:
        raise ExportDataError(
            "%s auxiliary columns contain %d undefined value(s); explicitly "
            "allow them to write NaN" % (label, undefined_count)
        )

    phase_values, phase_defined, phase_names, phase_sources = _phase_columns(
        result, phase
    )
    extra_values, extra_names = _extra_info_columns(
        result, extra_info, excluded_names=phase_names
    )
    collisions = sorted(set(output_names) & set(phase_names))
    if collisions:
        raise ExportDataError(
            "%s phase columns collide with retained auxiliary output name(s): %s"
            % (label, ", ".join(collisions))
        )
    collisions = sorted(
        (set(output_names) | set(phase_names)) & set(extra_names)
    )
    if collisions:
        raise ExportDataError(
            "%s ExtraInfo columns collide with existing output name(s): %s"
            % (label, ", ".join(collisions))
        )
    undefined_phase_count = int(np.count_nonzero(~phase_defined))
    requested = _requested_targets(result)
    if include_tilde:
        tilde_values, tilde_names = _requested_tilde_coordinates(result)
    else:
        tilde_values = np.empty((result.grid.size, 0), dtype=float)
        tilde_names = ()
    dimension = result.grid.dimension
    sources = result.source_coordinates.reshape(result.grid.size, dimension)
    matrix = np.column_stack(
        (
            requested,
            tilde_values,
            sources,
            auxiliary_values,
            phase_values,
            extra_values,
        )
    )
    defined = np.column_stack(
        (
            np.ones((result.grid.size, dimension), dtype=bool),
            np.ones(tilde_values.shape, dtype=bool),
            np.repeat(usable[:, None], dimension, axis=1),
            auxiliary_defined,
            phase_defined,
            np.ones(extra_values.shape, dtype=bool),
        )
    )
    if np.any(defined & ~np.isfinite(matrix)):
        raise ExportDataError("%s required columns contain non-finite values" % label)

    units_by_field = dict(zip(result.plan.auxiliary_names, result.plan.auxiliary_units))
    names = (
        result.plan.target_names
        + tilde_names
        + result.plan.source_names
        + output_names
        + phase_names
        + extra_names
    )
    raw_units = (
        result.plan.target_units
        + ("GeV",) * len(tilde_names)
        + result.plan.source_units
        + tuple(units_by_field[name] for name in source_fields)
        + ("dimensionless",) * len(phase_names)
        + ("dimensionless",) * len(extra_names)
    )
    units = tuple(_public_unit_name(unit) for unit in raw_units)
    return _PreparedTable(
        matrix=matrix,
        defined=defined,
        column_names=names,
        column_units=units,
        source_fields=(
            result.plan.target_names
            + tilde_names
            + result.plan.source_names
            + source_fields
            + phase_sources
            + extra_names
        ),
        column_roles=(
            ("target",) * dimension
            + ("target_transform",) * len(tilde_names)
            + ("source",) * dimension
            + ("auxiliary",) * len(output_names)
            + ("phase",) * len(phase_names)
            + ("state",) * len(extra_names)
        ),
        auxiliary_indices=indices,
        solved_count=int(np.count_nonzero(result.solution_mask)),
        projected_count=projected_count,
        unusable_count=unusable_count,
        undefined_count=undefined_count,
        undefined_phase_count=undefined_phase_count,
    )


def _table_fingerprint(result: PipelineResult) -> Optional[str]:
    run_manifest = result.manifest.get("run", {})
    table_snapshot = (
        run_manifest.get("table_snapshot", {})
        if isinstance(run_manifest, Mapping)
        else {}
    )
    if not isinstance(table_snapshot, Mapping):
        return None
    value = table_snapshot.get("snapshot_fingerprint")
    return value if isinstance(value, str) else None


def _plain_json(value: Any) -> Any:
    """Return JSON-native containers for immutable pipeline manifest values."""

    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _source_auxiliary_evaluation(result: PipelineResult) -> Optional[Any]:
    run = result.manifest.get("run", {})
    if not isinstance(run, Mapping):
        return None
    value = run.get("auxiliary_evaluation")
    return None if value is None else _plain_json(value)


def _write_text_temporary(
    target: Path,
    prepared: _PreparedTable,
    *,
    header_lines: Sequence[str] = (),
    float_format: str,
) -> Tuple[Path, str]:
    descriptor, temporary = _temporary_path(target)
    os.close(descriptor)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for line in header_lines:
                stream.write("# %s\n" % line)
            np.savetxt(stream, prepared.matrix, fmt=float_format)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary, hashlib.sha256(temporary.read_bytes()).hexdigest()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_manifest_temporary(target: Path, document: Mapping[str, Any]) -> Path:
    descriptor, temporary = _temporary_path(target)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _column_manifest(prepared: _PreparedTable) -> list[Dict[str, str]]:
    return [
        {"name": name, "unit": unit, "role": role, "source_field": source}
        for name, unit, role, source in zip(
            prepared.column_names,
            prepared.column_units,
            prepared.column_roles,
            prepared.source_fields,
        )
    ]


def _phase_manifest(
    result: PipelineResult,
    config: Optional[PhaseExportConfig],
    prepared: _PreparedTable,
) -> Dict[str, Any]:
    if config is None:
        return {
            "enabled": False,
            "schema_version": None,
            "columns": [],
        }
    return {
        "enabled": True,
        "schema_version": config.schema_version,
        "columns": list(config.columns),
        "codebooks": {
            "phase_code": dict(result.codebooks["phase"]),
            "phase_route_code": dict(result.codebooks["phase_route"]),
        },
        "line_mixture_fraction": {
            "defined_only_for": "line_region_s mixed roots",
            "undefined_encoding": "nan",
        },
        "undefined_phase_values": prepared.undefined_phase_count,
    }


def _format_formula_number(value: float) -> str:
    numeric = 0.0 if value == 0.0 else float(value)
    return "%.15g" % numeric


def _linear_density_expression(
    coefficients: Sequence[float], names: Sequence[str]
) -> str:
    terms = []
    for coefficient, name in zip(coefficients, names):
        numeric = float(coefficient)
        if numeric == 0.0:
            continue
        magnitude = abs(numeric)
        factor = "" if magnitude == 1.0 else "%s*" % _format_formula_number(magnitude)
        term = "%s%s" % (factor, name)
        if not terms:
            terms.append(("-" if numeric < 0.0 else "") + term)
        else:
            terms.append((" - " if numeric < 0.0 else " + ") + term)
    return "".join(terms) if terms else "0"


def _tilde_transform_manifest(result: PipelineResult, included: bool) -> Dict[str, Any]:
    if result.grid.space.value != "tilde":
        return {"applicable": False, "columns_included": False}
    constraints = result.grid.constraints
    independent_density_names = constraints.independent_density_names
    reduced_formulas = {}
    reduced_coefficients = {}
    for name, coefficients in zip(
        ("muB_tilde", "muQ_tilde", "muS_tilde"),
        constraints.chemical_map,
    ):
        coefficient_map = {
            density: float(_format_formula_number(float(coefficient)))
            for density, coefficient in zip(independent_density_names, coefficients)
            if float(coefficient) != 0.0
        }
        numerator = _linear_density_expression(
            coefficients, independent_density_names
        )
        reduced_coefficients[name] = coefficient_map
        reduced_formulas[name] = "%s = (%s) / T_tilde^2" % (name, numerator)
    return {
        "applicable": True,
        "columns_included": bool(included),
        "column_semantics": "requested transformed-grid coordinates",
        "transform_id": result.grid.transform_id,
        "temperature_formula": "T_tilde = (e / (19*pi^2/12))^(1/4)",
        "density_constraints": {
            name: dict(relation)
            for name, relation in constraints.dependent_densities.items()
        },
        "reduced_chemical_coefficients": reduced_coefficients,
        "reduced_chemical_formulas": reduced_formulas,
    }


def _tilde_header_lines(result: PipelineResult, included: bool) -> Tuple[str, ...]:
    document = _tilde_transform_manifest(result, included)
    if not document["applicable"]:
        return ()
    constraints = result.grid.constraints
    relations = []
    for dependent_name, coefficients in constraints.dependent_densities.items():
        independent_names = constraints.independent_density_names
        relation_coefficients = [
            coefficients.get(name, 0.0) for name in independent_names
        ]
        relations.append(
            "%s = %s"
            % (
                dependent_name,
                _linear_density_expression(
                    relation_coefficients, independent_names
                ),
            )
        )
    lines = ["Tilde: %s" % document["temperature_formula"]]
    if relations:
        lines.append("Tilde density constraints: %s" % "; ".join(relations))
    lines.extend(
        "Tilde: %s" % document["reduced_chemical_formulas"][name]
        for name in ("muB_tilde", "muQ_tilde", "muS_tilde")
    )
    if included:
        lines.append(
            "Tilde columns are requested grid coordinates; projected rows may "
            "have different returned-state tilde values"
        )
    return tuple(lines)


def _generic_text_header(
    result: PipelineResult,
    prepared: _PreparedTable,
    config: TextExportConfig,
) -> Tuple[str, ...]:
    """Build a portable conventional-comment header for a text table."""

    columns = " ".join(
        "%s[%s]" % (name, unit)
        for name, unit in zip(prepared.column_names, prepared.column_units)
    )
    lines = [
        "EoS-TrENsMUTher version %s" % __version__,
        "Fields: %s" % columns,
    ]
    lines.extend(_tilde_header_lines(result, config.include_tilde))
    lines.append(
        "Grid shape: %s; C order; last axis changes fastest"
        % " x ".join(map(str, result.grid.shape))
    )
    return tuple(lines)


def export_text(
    result: PipelineResult,
    path: Any,
    *,
    config: Optional[TextExportConfig] = None,
    overwrite: bool = False,
    manifest_path: Optional[Any] = None,
) -> TextExportResult:
    """Write a generic self-describing column-based EoS text table."""

    if config is None:
        config = TextExportConfig()
    if not isinstance(config, TextExportConfig):
        raise TypeError("config must be a TextExportConfig")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")

    prepared = _prepare_table(
        result,
        priority=(),
        allow_projected=True,
        allow_unusable=config.include_unusable,
        allow_undefined_auxiliary=config.allow_undefined_auxiliary,
        phase=config.phase,
        extra_info=config.extra_info,
        include_tilde=config.include_tilde,
        label="text export",
        supported_dimensions=(1, 2, 3, 4),
    )
    selected_indices = np.flatnonzero(
        np.asarray(result.jacobian_selected_mask, dtype=bool).reshape(-1)
    )
    extra_info_names = tuple(
        name
        for name in EXTRA_INFO_COLUMN_NAMES
        if config.extra_info
        and (config.phase is None or name not in config.phase.columns)
    )
    if selected_indices.size and not (
        config.extra_info or config.write_header or config.write_manifest
    ):
        raise ExportDataError(
            "Jacobian-selected roots require a text header or JSON manifest "
            "so branch-selection provenance is not discarded"
        )
    target = _absolute_output_path(path, "text export")
    manifest_target = None
    if config.write_manifest:
        manifest_target = _absolute_output_path(
            manifest_path
            if manifest_path is not None
            else target.with_name(target.name + ".manifest.json"),
            "text-export manifest",
        )
    for candidate in (target, manifest_target):
        if candidate is not None and candidate.exists() and not overwrite:
            raise FileExistsError("export already exists: %s" % candidate)

    header_lines = (
        _generic_text_header(result, prepared, config)
        if config.write_header
        else ()
    )
    temporary, digest = _write_text_temporary(
        target,
        prepared,
        header_lines=header_lines,
        float_format=config.float_format,
    )
    manifest_temporary: Optional[Path] = None
    try:
        if manifest_target is not None:
            document = {
                "artifact_type": TEXT_EXPORT_TYPE,
                "schema_version": TEXT_EXPORT_SCHEMA_VERSION,
                "software_version": __version__,
                "table_file": target.name,
                "table_path": str(target),
                "table_sha256": digest,
                "row_count": result.grid.size,
                "columns": _column_manifest(prepared),
                "column_contract": (
                    "targets_tilde_sources_auxiliaries_extra_info"
                    if config.include_tilde
                    else "targets_sources_auxiliaries_extra_info"
                ),
                "dimension": result.grid.dimension,
                "ordering": result.grid.ordering,
                "grid_space": result.grid.space.value,
                "grid_shape": list(result.grid.shape),
                "target_columns": "requested canonical targets",
                "tilde_transform": _tilde_transform_manifest(
                    result, config.include_tilde
                ),
                "state_columns": (
                    "solved state, or projected state when projection produced one"
                ),
                "phase_export": _phase_manifest(result, config.phase, prepared),
                "root_selection_provenance": {
                    "selection": "SELECTED_FROM_JACOBIAN",
                    "count": int(selected_indices.size),
                    "flat_target_indices": [
                        int(index) for index in selected_indices
                    ],
                },
                "include_projected": True,
                "extra_info": {
                    "enabled": config.extra_info,
                    "columns": list(extra_info_names),
                    "encoding": "0_or_1",
                },
                "include_unusable": config.include_unusable,
                "unusable_state_encoding": "nan",
                "allow_undefined_auxiliary": config.allow_undefined_auxiliary,
                "undefined_auxiliary_encoding": "nan",
                "write_header": config.write_header,
                "counts": {
                    "solved": prepared.solved_count,
                    "projected": prepared.projected_count,
                    "unusable": prepared.unusable_count,
                    "undefined_auxiliary_values": prepared.undefined_count,
                    "undefined_phase_values": prepared.undefined_phase_count,
                },
                "source_pipeline_result_schema_version": result.manifest.get(
                    "pipeline_result_schema_version"
                ),
                "source_table_snapshot_fingerprint": _table_fingerprint(result),
                "source_auxiliary_evaluation": _source_auxiliary_evaluation(result),
            }
            manifest_temporary = _write_manifest_temporary(
                manifest_target, document
            )

        _publish(temporary, target, overwrite=overwrite)
        if manifest_target is not None and manifest_temporary is not None:
            try:
                _publish(manifest_temporary, manifest_target, overwrite=overwrite)
            except Exception:
                if not overwrite and target.exists() and not target.is_symlink():
                    target.unlink()
                raise
        _fsync_directory(target.parent)
        if manifest_target is not None and manifest_target.parent != target.parent:
            _fsync_directory(manifest_target.parent)
        return TextExportResult(
            table_path=target,
            manifest_path=manifest_target,
            row_count=result.grid.size,
            solved_count=prepared.solved_count,
            projected_count=prepared.projected_count,
            unusable_count=prepared.unusable_count,
            undefined_auxiliary_count=prepared.undefined_count,
            undefined_phase_count=prepared.undefined_phase_count,
            sha256=digest,
            column_names=prepared.column_names,
            column_units=prepared.column_units,
            phase_column_names=(
                () if config.phase is None else config.phase.columns
            ),
            extra_info_column_names=(
                extra_info_names
            ),
        )
    finally:
        for candidate in (temporary, manifest_temporary):
            if candidate is not None and candidate.exists():
                try:
                    candidate.unlink()
                except OSError:
                    pass


def export_vhlle(
    result: PipelineResult,
    path: Any,
    *,
    config: Optional[VHLLEExportConfig] = None,
    overwrite: bool = False,
    manifest_path: Optional[Any] = None,
) -> VHLLEExportResult:
    """Write vHLLE ``EoS_all.dat`` and its provenance sidecar."""

    if config is None:
        config = VHLLEExportConfig()
    if not isinstance(config, VHLLEExportConfig):
        raise TypeError("config must be a VHLLEExportConfig")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")
    dimension = result.grid.dimension
    priority = (
        tuple(
            (name, config.field_map[name])
            for name in ("P", "s", "cs2", "chi2")
        )
        if dimension == 2
        else ()
    )
    prepared = _prepare_table(
        result,
        priority=priority,
        allow_projected=config.allow_projected,
        allow_unusable=config.allow_unusable,
        allow_undefined_auxiliary=config.allow_undefined_auxiliary,
        phase=config.phase,
        label="vHLLE",
        supported_dimensions=(1, 2, 3, 4),
    )
    if dimension == 2 and prepared.column_names[:8] != VHLLE_COLUMN_NAMES:
        raise ExportDataError("vHLLE legacy column prefix is inconsistent")
    target = _absolute_output_path(path, "vHLLE output")
    manifest_target = None
    if config.write_manifest:
        manifest_target = _absolute_output_path(
            manifest_path
            if manifest_path is not None
            else target.with_name(target.name + ".manifest.json"),
            "vHLLE manifest",
        )
    for candidate in (target, manifest_target):
        if candidate is not None and candidate.exists() and not overwrite:
            raise FileExistsError("export already exists: %s" % candidate)

    temporary, digest = _write_text_temporary(
        target,
        prepared,
        header_lines=(
            (
                " ".join(
                    "%s[%s]" % (name, unit)
                    for name, unit in zip(
                        prepared.column_names, prepared.column_units
                    )
                ),
            )
            if config.write_header
            else ()
        ),
        float_format=config.float_format,
    )
    manifest_temporary: Optional[Path] = None
    try:
        if manifest_target is not None:
            document = {
                "artifact_type": VHLLE_EXPORT_TYPE,
                "schema_version": VHLLE_EXPORT_SCHEMA_VERSION,
                "table_file": target.name,
                "table_path": str(target),
                "table_sha256": digest,
                "row_count": result.grid.size,
                "columns": _column_manifest(prepared),
                "column_contract": (
                    "legacy_2d" if dimension == 2 else "dimension_generalized"
                ),
                "dimension": dimension,
                "legacy_column_prefix": (
                    list(VHLLE_COLUMN_NAMES) if dimension == 2 else []
                ),
                "ordering": result.grid.ordering,
                "grid_space": result.grid.space.value,
                "grid_shape": list(result.grid.shape),
                "target_columns": "requested canonical targets",
                "state_columns": "solved state, or projected state when allowed",
                "field_map": (
                    dict(config.field_map) if dimension == 2 else {}
                ),
                "phase_export": _phase_manifest(
                    result, config.phase, prepared
                ),
                "allow_projected": config.allow_projected,
                "allow_unusable": config.allow_unusable,
                "unusable_state_encoding": "nan",
                "allow_undefined_auxiliary": config.allow_undefined_auxiliary,
                "undefined_auxiliary_encoding": "nan",
                "write_header": config.write_header,
                "counts": {
                    "solved": prepared.solved_count,
                    "projected": prepared.projected_count,
                    "unusable": prepared.unusable_count,
                    "undefined_auxiliary_values": prepared.undefined_count,
                    "undefined_phase_values": prepared.undefined_phase_count,
                },
                "source_pipeline_result_schema_version": result.manifest.get(
                    "pipeline_result_schema_version"
                ),
                "source_table_snapshot_fingerprint": _table_fingerprint(result),
                "source_auxiliary_evaluation": _source_auxiliary_evaluation(
                    result
                ),
            }
            manifest_temporary = _write_manifest_temporary(manifest_target, document)

        _publish(temporary, target, overwrite=overwrite)
        if manifest_target is not None and manifest_temporary is not None:
            try:
                _publish(manifest_temporary, manifest_target, overwrite=overwrite)
            except Exception:
                if not overwrite and target.exists() and not target.is_symlink():
                    target.unlink()
                raise
        _fsync_directory(target.parent)
        if manifest_target is not None and manifest_target.parent != target.parent:
            _fsync_directory(manifest_target.parent)
        return VHLLEExportResult(
            table_path=target,
            manifest_path=manifest_target,
            row_count=result.grid.size,
            solved_count=prepared.solved_count,
            projected_count=prepared.projected_count,
            unusable_count=prepared.unusable_count,
            undefined_auxiliary_count=prepared.undefined_count,
            undefined_phase_count=prepared.undefined_phase_count,
            sha256=digest,
            column_names=prepared.column_names,
            phase_column_names=(
                () if config.phase is None else config.phase.columns
            ),
        )
    finally:
        for candidate in (temporary, manifest_temporary):
            if candidate is not None and candidate.exists():
                try:
                    candidate.unlink()
                except OSError:
                    pass


def _safe_binary_field_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ExportDataError(
            "MUSIC binary field name is not filename-safe: %r" % name
        )
    return name


def _music_header(result: PipelineResult) -> np.ndarray:
    values = []
    for axis in result.grid.axes:
        differences = np.diff(axis)
        spacing = float(differences[0])
        if not np.allclose(differences, spacing, rtol=1e-12, atol=1e-15):
            raise ExportDataError("MUSIC requires uniformly spaced inversion axes")
        values.extend((float(axis[0]), spacing, float(axis.size)))
    return np.asarray(values, dtype="<f4")


def _write_binary_temporary(
    target: Path, header: np.ndarray, values: np.ndarray
) -> Tuple[Path, str]:
    descriptor, temporary = _temporary_path(target)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(np.asarray(header, dtype="<f4").tobytes(order="C"))
            stream.write(np.asarray(values, dtype="<f4").tobytes(order="C"))
            stream.flush()
            os.fsync(stream.fileno())
        return temporary, hashlib.sha256(temporary.read_bytes()).hexdigest()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def export_music(
    result: PipelineResult,
    path: Any,
    *,
    config: Optional[MUSICExportConfig] = None,
    overwrite: bool = False,
    manifest_path: Optional[Any] = None,
) -> MUSICExportResult:
    """Write legacy MUSIC binaries and a full ``EoS_all.dat`` table.

    Every binary begins with float32 ``minimum, spacing, count`` values for
    each inversion axis. Traditional ``t``, ``mub``, ``p``, and ``s`` files
    remain first; each additional auxiliary gets an individual binary file.
    """

    if config is None:
        config = MUSICExportConfig()
    if not isinstance(config, MUSICExportConfig):
        raise TypeError("config must be a MUSICExportConfig")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")
    priority = tuple((name, config.field_map[name]) for name in ("P", "s"))
    prepared = _prepare_table(
        result,
        priority=priority,
        allow_projected=config.allow_projected,
        allow_undefined_auxiliary=config.allow_undefined_auxiliary,
        phase=config.phase,
        label="MUSIC",
    )
    target = _absolute_output_path(path, "MUSIC text output")
    manifest_target = None
    if config.write_manifest:
        manifest_target = _absolute_output_path(
            manifest_path
            if manifest_path is not None
            else target.with_name(target.name + ".manifest.json"),
            "MUSIC manifest",
        )

    binary_values: Dict[str, np.ndarray] = {
        "t": result.source_coordinates[..., 0].reshape(-1),
        "mub": result.source_coordinates[..., 1].reshape(-1),
    }
    auxiliary_matrix = prepared.matrix[:, 4:]
    for output_name, values in zip(prepared.column_names[4:], auxiliary_matrix.T):
        file_key = {"P": "p", "s": "s"}.get(output_name, output_name)
        binary_values[_safe_binary_field_name(file_key)] = values
    if tuple(binary_values)[:4] != MUSIC_LEGACY_BINARY_FIELDS:
        raise ExportDataError("MUSIC legacy binary field prefix is inconsistent")

    binary_targets = {
        name: target.parent / ("EoS_%s_b.dat" % name) for name in binary_values
    }
    all_targets = [target, *binary_targets.values()]
    if manifest_target is not None:
        all_targets.append(manifest_target)
    for candidate in all_targets:
        if candidate.exists() and not overwrite:
            raise FileExistsError("export already exists: %s" % candidate)
        if candidate.exists() and (candidate.is_symlink() or not candidate.is_file()):
            raise ExportPathError("export target must be a regular file: %s" % candidate)

    text_temporary, table_digest = _write_text_temporary(
        target,
        prepared,
        header_lines=(
            (
                " ".join(
                    "%s[%s]" % (name, unit)
                    for name, unit in zip(
                        prepared.column_names, prepared.column_units
                    )
                ),
            )
            if config.write_header
            else ()
        ),
        float_format=config.float_format,
    )
    temporaries: Dict[Path, Path] = {target: text_temporary}
    binary_digests: Dict[str, str] = {}
    manifest_temporary: Optional[Path] = None
    published = []
    try:
        header = _music_header(result)
        for name, values in binary_values.items():
            temporary, digest = _write_binary_temporary(
                binary_targets[name], header, values
            )
            temporaries[binary_targets[name]] = temporary
            binary_digests[name] = digest

        if manifest_target is not None:
            document = {
                "artifact_type": MUSIC_EXPORT_TYPE,
                "schema_version": MUSIC_EXPORT_SCHEMA_VERSION,
                "table_file": target.name,
                "table_path": str(target),
                "table_sha256": table_digest,
                "row_count": result.grid.size,
                "columns": _column_manifest(prepared),
                "ordering": result.grid.ordering,
                "grid_space": result.grid.space.value,
                "grid_shape": list(result.grid.shape),
                "grid_axes": [
                    {
                        "name": name,
                        "unit": unit,
                        "minimum": float(axis[0]),
                        "spacing": float(axis[1] - axis[0]),
                        "count": int(axis.size),
                    }
                    for name, unit, axis in zip(
                        result.grid.axis_names,
                        result.grid.axis_units,
                        result.grid.axes,
                    )
                ],
                "binary_dtype": "little-endian float32",
                "binary_header": "minimum, spacing, count for each grid axis",
                "binary_files": {
                    name: {
                        "file": binary_targets[name].name,
                        "sha256": binary_digests[name],
                    }
                    for name in binary_values
                },
                "legacy_binary_field_prefix": list(MUSIC_LEGACY_BINARY_FIELDS),
                "field_map": dict(config.field_map),
                "phase_export": _phase_manifest(
                    result, config.phase, prepared
                ),
                "allow_projected": config.allow_projected,
                "allow_undefined_auxiliary": config.allow_undefined_auxiliary,
                "undefined_auxiliary_encoding": "IEEE-754 NaN",
                "write_header": config.write_header,
                "counts": {
                    "solved": prepared.solved_count,
                    "projected": prepared.projected_count,
                    "undefined_auxiliary_values": prepared.undefined_count,
                    "undefined_phase_values": prepared.undefined_phase_count,
                },
                "source_pipeline_result_schema_version": result.manifest.get(
                    "pipeline_result_schema_version"
                ),
                "source_table_snapshot_fingerprint": _table_fingerprint(result),
                "source_auxiliary_evaluation": _source_auxiliary_evaluation(
                    result
                ),
            }
            manifest_temporary = _write_manifest_temporary(manifest_target, document)

        for destination, temporary in temporaries.items():
            _publish(temporary, destination, overwrite=overwrite)
            published.append(destination)
        if manifest_target is not None and manifest_temporary is not None:
            _publish(manifest_temporary, manifest_target, overwrite=overwrite)
            published.append(manifest_target)
        _fsync_directory(target.parent)
        if manifest_target is not None and manifest_target.parent != target.parent:
            _fsync_directory(manifest_target.parent)
        return MUSICExportResult(
            table_path=target,
            manifest_path=manifest_target,
            binary_paths=MappingProxyType(dict(binary_targets)),
            row_count=result.grid.size,
            solved_count=prepared.solved_count,
            projected_count=prepared.projected_count,
            undefined_auxiliary_count=prepared.undefined_count,
            undefined_phase_count=prepared.undefined_phase_count,
            table_sha256=table_digest,
            binary_sha256=MappingProxyType(binary_digests),
            column_names=prepared.column_names,
            phase_column_names=(
                () if config.phase is None else config.phase.columns
            ),
        )
    except Exception:
        if not overwrite:
            for candidate in published:
                if candidate.exists() and candidate.is_file() and not candidate.is_symlink():
                    candidate.unlink()
        raise
    finally:
        for candidate in [*temporaries.values(), manifest_temporary]:
            if candidate is not None and candidate.exists():
                try:
                    candidate.unlink()
                except OSError:
                    pass


__all__ = [
    "ExportConfigurationError",
    "ExportDataError",
    "ExportError",
    "ExportPathError",
    "PHASE_EXPORT_COLUMN_NAMES",
    "PHASE_EXPORT_SCHEMA_VERSION",
    "PhaseExportConfig",
    "TEXT_EXPORT_SCHEMA_VERSION",
    "TEXT_EXPORT_TYPE",
    "TextExportConfig",
    "TextExportResult",
    "export_text",
]
