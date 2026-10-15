"""Legacy-style Python parameter files for the package application layer.

The numerical core remains configured by its immutable typed dataclasses.  This
module is a deliberately thin adapter from a user-authored
``general_parameters`` dictionary to those types.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass, fields as dataclass_fields
from difflib import get_close_matches
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .checkpoint import CheckpointConfig
from .exporters import (
    ExportConfigurationError,
    PhaseExportConfig,
    TextExportConfig,
)
from .pipeline import (
    CoarseSeedConfig,
    DEFAULT_MAX_GRID_POINTS,
    DEFAULT_MAX_RESULT_BYTES,
    ExecutionConfig,
    GridAxisSpec,
    InversionGridSpec,
    PipelineConfig,
)
from .projection import ProjectionConfig
from .reachability import ReachabilityConfig
from .solver import SolverConfig
from .tables import ForwardTable


class ApplicationConfigError(ValueError):
    """Raised when a user parameter dictionary is incomplete or inconsistent."""


_USER_PARAMETER_NAMES = frozenset(
    {
        "EoS_metadata",
        "EoS_table",
        "OutputFolder",
        "EoSName",
        "RunSpecification",
        "Dimension",
        "Variables",
        "GridAxes",
        "AuxiliaryFields",
        "ExtraInfo",
        "Export",
        "Projection",
        "Execution",
        "Checkpoint",
        "Overwrite",
    }
)

_ADVANCED_PARAMETER_NAMES = frozenset(
    {
        "LineAuxiliaryDefaultPolicy",
        "LineAuxiliaryPolicies",
        "Solver",
        "InitialGuess",
        "UseReachabilitySeed",
        "CoarseSeed",
        "Reachability",
        "ArtifactCompression",
        "MaxGridPoints",
        "MaxResultBytes",
        "MaxPeakResultBytes",
        "ResidualScales",
    }
)

_DIAGNOSTIC_PARAMETER_NAMES = frozenset(
    {
        "PreprocessReport",
        "ResultFile",
    }
)

_COMPATIBILITY_PARAMETER_NAMES = frozenset(
    {
        "MetadataFile",
        "DataFile",
        "GridSpace",
        "AutoSetBoundaries",
        "ProjectionEnabled",
        "Number_of_cores",
        "ChunkSize",
        "WorkerTaskSize",
        "MaxPendingChunks",
        "ProcessStartMethod",
        "ErrorPolicy",
        "CheckpointEnabled",
        "CheckpointFolder",
        "Resume",
        "NT",
        "NB",
        "NQ",
        "NS",
        "Ttilde",
        "muBtilde",
        "muQtilde",
        "muStilde",
        "e",
        "nB",
        "nQ",
        "nS",
        "RunMode",
        "Accuracy",
        "MAXITER",
        "TransitionLine",
        "RegionS",
        "OutputMergedEoS",
        "NTildeT",
        "NTildemuB",
    }
)

PARAMETER_GROUPS = MappingProxyType(
    {
        "user": tuple(sorted(_USER_PARAMETER_NAMES)),
        "advanced": tuple(sorted(_ADVANCED_PARAMETER_NAMES)),
        "diagnostic": tuple(sorted(_DIAGNOSTIC_PARAMETER_NAMES)),
        "compatibility": tuple(sorted(_COMPATIBILITY_PARAMETER_NAMES)),
    }
)

_KNOWN_PARAMETERS = frozenset().union(*PARAMETER_GROUPS.values())

ORDINARY_SUBCELL_CONFIG_FIELDS = (
    "enable_subcell_fallback",
    "max_subcell_depth",
    "max_subcell_active_boxes",
)

_IGNORED_LEGACY_PARAMETERS = {
    "Accuracy",
    "MAXITER",
    "TransitionLine",
    "RegionS",
    "OutputMergedEoS",
    "NTildeT",
    "NTildemuB",
}

@dataclass(frozen=True)
class ApplicationConfig:
    """Fully resolved local run and output configuration."""

    parameter_path: Path
    metadata_path: Path
    data_path: Optional[Path]
    output_root: Path
    eos_directory: Path
    output_directory: Path
    diagnostic_directory: Path
    preprocess_report_path: Path
    result_path: Path
    text_export_path: Path
    text_export_manifest_path: Path
    pipeline: PipelineConfig
    reachability: ReachabilityConfig
    checkpoint: Optional[CheckpointConfig]
    text_export: TextExportConfig
    export_enabled: bool
    eos_name: str
    run_name: str
    specification: str
    overwrite_mode: str
    overwrite: bool
    artifact_compression: str
    ignored_legacy_parameters: Tuple[str, ...]


def _plain_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ApplicationConfigError("%s must be a dictionary" % name)
    return dict(value)


def _unknown_keys_message(
    unknown: Sequence[str], allowed: Sequence[str], label: str
) -> str:
    choices = tuple(sorted(allowed))
    details = []
    for name in sorted(unknown):
        matches = get_close_matches(name, choices, n=1, cutoff=0.72)
        detail = "Unknown %s:\n  %s" % (label, name)
        if matches:
            detail += "\n\nDid you mean:\n  %s" % matches[0]
        details.append(detail)
    return "\n\n".join(details)


def _validate_mapping_keys(
    mapping: Mapping[str, Any], allowed: Sequence[str], label: str
) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ApplicationConfigError(
            _unknown_keys_message(unknown, allowed, label)
        )


def load_parameter_file(path: Any) -> Tuple[Path, Mapping[str, Any]]:
    """Execute a Python parameter file and return ``general_parameters``.

    Python parameter files are executable code by design.  They must therefore
    be treated as trusted input, matching the legacy interface requested by the
    project.
    """

    if isinstance(path, bool):
        raise TypeError("parameter file must be path-like")
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (TypeError, ValueError, OSError) as exc:
        raise ApplicationConfigError("parameter file is unavailable: %s" % path) from exc
    if not resolved.is_file():
        raise ApplicationConfigError("parameter path is not a regular file: %s" % resolved)
    token = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    module_name = "_eos_trensmuther_parameters_%s" % token
    specification = importlib.util.spec_from_file_location(module_name, resolved)
    if specification is None or specification.loader is None:
        raise ApplicationConfigError("could not load parameter file: %s" % resolved)
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception as exc:
        raise ApplicationConfigError(
            "parameter file raised %s: %s" % (type(exc).__name__, exc)
        ) from exc
    finally:
        sys.modules.pop(module_name, None)
    if not hasattr(module, "general_parameters"):
        raise ApplicationConfigError(
            "parameter file must define a general_parameters dictionary"
        )
    mapping = _plain_mapping(module.general_parameters, "general_parameters")
    unknown = sorted(set(mapping) - _KNOWN_PARAMETERS)
    if unknown:
        raise ApplicationConfigError(
            _unknown_keys_message(unknown, tuple(_KNOWN_PARAMETERS), "parameter")
        )
    return resolved, MappingProxyType(mapping)


def _select_alias(
    values: Mapping[str, Any], first: str, second: str, *, required: bool = False
) -> Any:
    supplied = [name for name in (first, second) if name in values]
    if len(supplied) > 1:
        raise ApplicationConfigError(
            "use only one of %s and %s" % (first, second)
        )
    if not supplied:
        if required:
            raise ApplicationConfigError("missing required parameter: %s" % first)
        return None
    return values[supplied[0]]


def _input_path(value: Any, base: Path, name: str) -> Path:
    if isinstance(value, bool):
        raise ApplicationConfigError("%s must be path-like" % name)
    try:
        path = Path(value).expanduser()
    except (TypeError, ValueError) as exc:
        raise ApplicationConfigError("%s must be path-like" % name) from exc
    if not path.is_absolute():
        path = base / path
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        label = "Metadata file" if name == "EoS_metadata" else "Input file"
        raise ApplicationConfigError(
            "%s not found:\n  %s" % (label, path)
        ) from exc
    except OSError as exc:
        raise ApplicationConfigError(
            "%s could not be opened:\n  %s\n\n%s" % (name, path, exc)
        ) from exc


def _output_root(value: Any, base: Path, override: Optional[Any]) -> Path:
    selected = override if override is not None else value
    if isinstance(selected, bool):
        raise ApplicationConfigError("OutputFolder must be path-like")
    try:
        path = Path(selected).expanduser()
    except (TypeError, ValueError) as exc:
        raise ApplicationConfigError("OutputFolder must be path-like") from exc
    if not path.is_absolute():
        path = (Path.cwd() if override is not None else base) / path
    path = Path(os.path.abspath(os.fspath(path)))
    if not path.name:
        raise ApplicationConfigError("OutputFolder cannot be a filesystem root")
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise ApplicationConfigError("OutputFolder must be a directory: %s" % path)
    return path


def _safe_run_component(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ApplicationConfigError("%s must be a non-empty string" % name)
    if not value[0].isalnum() or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in value
    ):
        raise ApplicationConfigError(
            "%s must start with a letter/number and contain only letters, "
            "numbers, '_', '-', or '.'" % name
        )
    if value in {".", ".."}:
        raise ApplicationConfigError("%s is not a valid folder name" % name)
    return value


def _run_specification(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or not value:
        raise ApplicationConfigError("run specification must be a non-empty string")
    if not value.startswith("_"):
        raise ApplicationConfigError(
            "run specification must start with '_', for example '_test2mub500'"
        )
    if any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in value
    ) or value in {"_..", "_."}:
        raise ApplicationConfigError(
            "run specification may contain only letters, numbers, '_', '-', or '.'"
        )
    return value


def _overwrite_mode(value: Any) -> str:
    # Booleans remain accepted for compatibility with existing parameter files.
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "default"
    if not isinstance(value, str):
        raise ApplicationConfigError("Overwrite must be 'yes', 'no', or 'default'")
    normalized = value.strip().lower()
    if normalized not in {"yes", "no", "default"}:
        raise ApplicationConfigError("Overwrite must be 'yes', 'no', or 'default'")
    return normalized


def _resolve_run_directory(
    output_root: Path,
    eos_name: str,
    specification: str,
    overwrite_mode: str,
    operation: str,
) -> Tuple[Path, Path, str]:
    eos_directory = output_root / eos_name
    creating = operation in {"run", "preprocess", "invert", "slurm-plan"}
    requires_new_directory = operation in {"run", "preprocess", "slurm-plan"}

    if overwrite_mode == "default" and not specification:
        if not creating:
            raise ApplicationConfigError(
                "Overwrite='default' creates numbered runs. Provide --spec '_N' "
                "to select an existing run for postprocessing."
            )
        existing_count = 0
        if eos_directory.exists():
            if eos_directory.is_symlink() or not eos_directory.is_dir():
                raise ApplicationConfigError(
                    "EoS output path must be a directory: %s" % eos_directory
                )
            existing_count = sum(
                1
                for child in eos_directory.iterdir()
                if child.is_dir() and not child.is_symlink()
            )
        number = existing_count + 1
        while (eos_directory / ("%s_%d" % (eos_name, number))).exists():
            number += 1
        run_name = "%s_%d" % (eos_name, number)
    else:
        run_name = eos_name + specification

    run_directory = eos_directory / run_name
    if run_directory.exists() and (
        run_directory.is_symlink() or not run_directory.is_dir()
    ):
        raise ApplicationConfigError(
            "run output path must be a directory: %s" % run_directory
        )
    if creating and run_directory.exists() and overwrite_mode != "yes":
        if overwrite_mode == "no" and not specification:
            raise ApplicationConfigError(
                "run '%s' already exists and Overwrite='no'. Re-run with a "
                "specification such as --spec '_test2mub500'; no files were changed."
                % run_name
            )
        if requires_new_directory:
            raise ApplicationConfigError(
                "run '%s' already exists and will not be overwritten; choose another "
                "--spec value" % run_name
            )
    if not creating and not run_directory.is_dir():
        raise ApplicationConfigError(
            "run '%s' does not exist under %s" % (run_name, eos_directory)
        )
    return eos_directory, run_directory, run_name


def _member_path(directory: Path, value: Any, name: str, default: str) -> Path:
    selected = default if value is None else value
    if isinstance(selected, bool):
        raise ApplicationConfigError("%s must be a relative path" % name)
    try:
        relative = Path(selected)
    except (TypeError, ValueError) as exc:
        raise ApplicationConfigError("%s must be a relative path" % name) from exc
    if relative.is_absolute() or not relative.name or ".." in relative.parts:
        raise ApplicationConfigError(
            "%s must stay inside OutputFolder" % name
        )
    return directory / relative


def _bool(values: Mapping[str, Any], name: str, default: bool) -> bool:
    value = values.get(name, default)
    if not isinstance(value, bool):
        raise ApplicationConfigError("%s must be boolean" % name)
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ApplicationConfigError(
            "Invalid value for %s:\n  received: %r\n  expected: positive integer"
            % (name, value)
        )
    converted = int(value)
    if converted < 1:
        raise ApplicationConfigError(
            "Invalid value for %s:\n  received: %r\n  expected: positive integer"
            % (name, value)
        )
    return converted


def _axis_spec(value: Any, name: str) -> GridAxisSpec:
    if isinstance(value, Mapping):
        unknown = set(value) - {"count", "lower", "upper"}
        if unknown or "count" not in value:
            raise ApplicationConfigError(
                "GridAxes.%s must contain count and optional lower/upper" % name
            )
        return GridAxisSpec(
            count=value["count"],
            lower=value.get("lower"),
            upper=value.get("upper"),
        )
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return GridAxisSpec(int(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = tuple(value)
        if len(parts) != 3:
            raise ApplicationConfigError(
                "GridAxes.%s must be COUNT or [LOWER, UPPER, COUNT]" % name
            )
        return GridAxisSpec(count=parts[2], lower=parts[0], upper=parts[1])
    raise ApplicationConfigError(
        "GridAxes.%s must be COUNT, a mapping, or [LOWER, UPPER, COUNT]" % name
    )


def _legacy_axes(
    values: Mapping[str, Any], table: ForwardTable, space: str
) -> Mapping[str, GridAxisSpec]:
    envelope = table.find_boundaries(space=space)
    automatic = _bool(values, "AutoSetBoundaries", True)
    tilde_aliases = {
        "T_tilde": ("Ttilde", "NT"),
        "muB_tilde": ("muBtilde", "NB"),
        "muQ_tilde": ("muQtilde", "NQ"),
        "muS_tilde": ("muStilde", "NS"),
    }
    axes: Dict[str, GridAxisSpec] = {}
    for name in envelope.names:
        if space == "tilde":
            value_name, count_name = tilde_aliases[name]
            legacy = values.get(value_name)
            count = values.get(count_name)
            if count is None and isinstance(legacy, Sequence) and not isinstance(
                legacy, (str, bytes)
            ) and len(legacy) == 3:
                count = legacy[2]
            if count is None:
                raise ApplicationConfigError(
                    "GridAxes is missing; provide %s or %s" % (count_name, value_name)
                )
            if automatic:
                axes[name] = GridAxisSpec(count=count)
            else:
                if not isinstance(legacy, Sequence) or isinstance(legacy, (str, bytes)):
                    raise ApplicationConfigError(
                        "%s must be [LOWER, UPPER, COUNT]" % value_name
                    )
                axes[name] = _axis_spec(legacy, name)
        else:
            legacy = values.get(name)
            if not isinstance(legacy, Sequence) or isinstance(legacy, (str, bytes)):
                raise ApplicationConfigError(
                    "GridAxes is missing; provide %s=[LOWER, UPPER, COUNT]" % name
                )
            parts = tuple(legacy)
            if len(parts) != 3:
                raise ApplicationConfigError(
                    "%s must be [LOWER, UPPER, COUNT]" % name
                )
            axes[name] = (
                GridAxisSpec(count=parts[2])
                if automatic
                else _axis_spec(parts, name)
            )
    return MappingProxyType(axes)


def _grid_axes(
    values: Mapping[str, Any], table: ForwardTable, space: str
) -> Mapping[str, GridAxisSpec]:
    envelope = table.find_boundaries(space=space)
    supplied = values.get("GridAxes")
    if supplied is None:
        return _legacy_axes(values, table, space)
    if isinstance(supplied, (int, np.integer)) and not isinstance(
        supplied, (bool, np.bool_)
    ):
        return MappingProxyType(
            {name: GridAxisSpec(count=int(supplied)) for name in envelope.names}
        )
    mapping = _plain_mapping(supplied, "GridAxes")
    if space == "tilde" and set(mapping) == set(table.coordinate_names):
        mapping = {
            tilde_name: mapping[source_name]
            for source_name, tilde_name in zip(
                table.coordinate_names, envelope.names
            )
        }
    if set(mapping) != set(envelope.names):
        raise ApplicationConfigError(
            "GridAxes must define exactly: %s; for Variables='tilde', the "
            "ordinary coordinate names %s are also accepted"
            % (
                ", ".join(envelope.names),
                ", ".join(table.coordinate_names),
            )
        )
    return MappingProxyType(
        {name: _axis_spec(mapping[name], name) for name in envelope.names}
    )


def _construct(kind: Any, values: Any, name: str) -> Any:
    mapping = _plain_mapping({} if values is None else values, name)
    allowed = tuple(field.name for field in dataclass_fields(kind) if field.init)
    _validate_mapping_keys(mapping, allowed, name)
    try:
        return kind(**mapping)
    except (TypeError, ValueError) as exc:
        raise ApplicationConfigError("invalid %s: %s" % (name, exc)) from exc


def _phase_export(value: Any, name: str) -> Optional[PhaseExportConfig]:
    if value is None:
        return None
    mapping = _plain_mapping(value, name)
    allowed = tuple(
        field.name for field in dataclass_fields(PhaseExportConfig) if field.init
    )
    _validate_mapping_keys(mapping, allowed, name)
    try:
        return PhaseExportConfig(**mapping)
    except (TypeError, ExportConfigurationError) as exc:
        raise ApplicationConfigError("invalid %s: %s" % (name, exc)) from exc


def _text_export(
    value: Any,
    *,
    extra_info: bool,
) -> Tuple[bool, str, TextExportConfig]:
    mapping = _plain_mapping({} if value is None else value, "Export")
    allowed = {
        "enabled",
        "filename",
        "include_projected",
        "include_unusable",
        "allow_undefined_auxiliary",
        "write_header",
        "write_manifest",
        "float_format",
        "include_tilde",
        "phase",
    }
    _validate_mapping_keys(mapping, tuple(allowed), "Export")
    enabled = mapping.pop("enabled", True)
    filename = mapping.pop("filename", "eos_table.dat")
    legacy_include_projected = mapping.pop("include_projected", None)
    if not isinstance(enabled, bool):
        raise ApplicationConfigError("Export.enabled must be boolean")
    if not isinstance(filename, str) or not filename:
        raise ApplicationConfigError("Export.filename must be a non-empty string")
    if legacy_include_projected is not None and not isinstance(
        legacy_include_projected, bool
    ):
        raise ApplicationConfigError(
            "Export.include_projected must be boolean when used as a legacy setting"
        )
    # Projected states are now always exported and tagged. The old nested
    # switch remains accepted so existing parameter files continue to load.
    mapping["include_projected"] = True
    mapping["extra_info"] = extra_info
    phase = _phase_export(mapping.pop("phase", None), "Export.phase")
    try:
        config = TextExportConfig(phase=phase, **mapping)
    except (TypeError, ExportConfigurationError) as exc:
        raise ApplicationConfigError("invalid Export: %s" % exc) from exc
    return enabled, filename, config


def _execution(values: Mapping[str, Any]) -> ExecutionConfig:
    nested = _plain_mapping(values.get("Execution", {}), "Execution")
    aliases = {
        "workers": ("Number_of_cores", 1),
        "chunk_size": ("ChunkSize", 256),
        "worker_task_size": ("WorkerTaskSize", 8),
        "max_pending_chunks": ("MaxPendingChunks", None),
        "start_method": ("ProcessStartMethod", "spawn"),
        "error_policy": ("ErrorPolicy", "fail_fast"),
    }
    for target, (source, default) in aliases.items():
        if target not in nested:
            nested[target] = values.get(source, default)
    for target, (source, _) in aliases.items():
        if target in {"workers", "chunk_size", "worker_task_size"}:
            label = "Execution.%s" % target
            if source in values and target not in values.get("Execution", {}):
                label = source
            nested[target] = _positive_integer(nested[target], label)
    if nested["max_pending_chunks"] is not None:
        nested["max_pending_chunks"] = _positive_integer(
            nested["max_pending_chunks"], "Execution.max_pending_chunks"
        )
    return _construct(ExecutionConfig, nested, "Execution")


def _solver_config(value: Any) -> SolverConfig:
    mapping = _plain_mapping({} if value is None else value, "Solver")
    unrecognized_subcell = sorted(
        name
        for name in mapping
        if "subcell" in name and name not in ORDINARY_SUBCELL_CONFIG_FIELDS
    )
    if unrecognized_subcell:
        raise ApplicationConfigError(
            "%s; the supported ordinary-subcell keys are: %s"
            % (
                _unknown_keys_message(
                    unrecognized_subcell,
                    ORDINARY_SUBCELL_CONFIG_FIELDS,
                    "Solver",
                ),
                ", ".join(ORDINARY_SUBCELL_CONFIG_FIELDS),
            )
        )
    return _construct(SolverConfig, mapping, "Solver")


def _checkpoint(
    values: Mapping[str, Any], output_directory: Path
) -> Optional[CheckpointConfig]:
    raw = values.get("Checkpoint")
    if raw is None:
        enabled = _bool(values, "CheckpointEnabled", True)
        directory_value = values.get("CheckpointFolder", "checkpoint")
        resume = _bool(values, "Resume", True)
    elif isinstance(raw, bool):
        enabled = raw
        directory_value = values.get("CheckpointFolder", "checkpoint")
        resume = _bool(values, "Resume", True)
    else:
        mapping = _plain_mapping(raw, "Checkpoint")
        unknown = set(mapping) - {"enabled", "directory", "resume"}
        if unknown:
            raise ApplicationConfigError(
                _unknown_keys_message(
                    sorted(unknown), ("enabled", "directory", "resume"), "Checkpoint"
                )
            )
        enabled = mapping.get("enabled", True)
        resume = mapping.get("resume", True)
        directory_value = mapping.get("directory", "checkpoint")
        if not isinstance(enabled, bool) or not isinstance(resume, bool):
            raise ApplicationConfigError("Checkpoint enabled/resume must be boolean")
    if not enabled:
        return None
    directory = _member_path(
        output_directory, directory_value, "Checkpoint.directory", "checkpoint"
    )
    return CheckpointConfig(directory=directory, resume=resume)


def load_application_config(
    parameter_file: Any,
    *,
    output_directory: Optional[Any] = None,
    overwrite: Optional[bool] = None,
    specification: Optional[str] = None,
    operation: str = "run",
) -> Tuple[ApplicationConfig, ForwardTable]:
    """Load, validate, and resolve a complete local application run."""

    parameter_path, values = load_parameter_file(parameter_file)
    base = parameter_path.parent
    metadata_value = _select_alias(
        values, "EoS_metadata", "MetadataFile", required=True
    )
    metadata_path = _input_path(metadata_value, base, "EoS_metadata")
    data_value = _select_alias(values, "EoS_table", "DataFile")
    data_path = (
        None if data_value is None else _input_path(data_value, base, "EoS_table")
    )

    # Resolve collisions before reading/validating the potentially large table.
    # In Overwrite='no' mode this makes batch jobs fail immediately and tells
    # the user which --spec option is required.
    output_root = _output_root(
        values.get("OutputFolder", "Output"), base, output_directory
    )
    eos_name = _safe_run_component(
        values.get("EoSName", metadata_path.parent.name), "EoSName"
    )
    parameter_specification = values.get("RunSpecification")
    if specification is not None and parameter_specification is not None:
        raise ApplicationConfigError(
            "set the run specification with either --spec or RunSpecification, not both"
        )
    run_specification = _run_specification(
        specification if specification is not None else parameter_specification
    )
    overwrite_mode = _overwrite_mode(values.get("Overwrite"))
    if overwrite is not None:
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite override must be boolean")
        if overwrite:
            overwrite_mode = "yes"
    eos_directory, output, run_name = _resolve_run_directory(
        output_root,
        eos_name,
        run_specification,
        overwrite_mode,
        operation,
    )
    overwrite_value = overwrite_mode == "yes"
    diagnostic_directory = output / "diagnostic"

    table = ForwardTable.load_validated(metadata_path, data_path=data_path)

    if "Dimension" in values:
        dimension = _positive_integer(values["Dimension"], "Dimension")
        assert table.metadata is not None
        if dimension != table.metadata.dimension:
            raise ApplicationConfigError(
                "Dimension=%d disagrees with metadata dimension=%d"
                % (dimension, table.metadata.dimension)
            )
    run_mode = values.get("RunMode", 0)
    if run_mode != 0:
        raise ApplicationConfigError(
            "the quick interface currently supports RunMode=0 only; scheduler "
            "adapters are a separate next step"
        )

    variables = values.get("Variables")
    if variables is not None and "GridSpace" in values:
        raise ApplicationConfigError(
            "use Variables or the legacy GridSpace setting, not both"
        )
    if variables is None:
        space = values.get("GridSpace", "target")
    elif not isinstance(variables, str):
        raise ApplicationConfigError(
            "Variables must be 'tilde' when transformed coordinates are requested"
        )
    elif variables.strip().lower() == "tilde":
        space = "tilde"
    elif variables.strip().lower() in {"regular", "target"}:
        space = "target"
    else:
        raise ApplicationConfigError(
            "Variables must be 'tilde' for transformed variables; omit it for "
            "regular (e, nB, ...) variables"
        )
    if not isinstance(space, str) or space not in {"target", "tilde"}:
        raise ApplicationConfigError(
            "legacy GridSpace must be 'target' or 'tilde'"
        )
    axes = _grid_axes(values, table, space)
    scales = values.get("ResidualScales")
    if scales == "automatic":
        scales = None
    if scales is not None and not isinstance(scales, Mapping):
        raise ApplicationConfigError(
            "ResidualScales must be omitted, 'automatic', or explicitly map "
            "every target field to a positive scale"
        )

    export_enabled, export_filename, text_export = _text_export(
        values.get("Export"),
        extra_info=_bool(values, "ExtraInfo", True),
    )
    requested_auxiliary = values.get("AuxiliaryFields")
    if requested_auxiliary is None:
        # None delegates to PipelinePlan, which retains every metadata
        # auxiliary field marked output=true.
        auxiliary_fields = None
    else:
        if isinstance(requested_auxiliary, (str, bytes)) or not isinstance(
            requested_auxiliary, Sequence
        ):
            raise ApplicationConfigError("AuxiliaryFields must be a list of names")
        auxiliary_fields = tuple(requested_auxiliary)

    solver = _solver_config(values.get("Solver"))
    projection_values = _plain_mapping(values.get("Projection", {}), "Projection")
    if "ProjectionEnabled" in values:
        if "enabled" in projection_values:
            raise ApplicationConfigError(
                "set projection enabled in either Projection or ProjectionEnabled"
            )
        projection_values["enabled"] = values["ProjectionEnabled"]
    projection = _construct(ProjectionConfig, projection_values, "Projection")
    execution = _execution(values)
    coarse_seed = _construct(
        CoarseSeedConfig, values.get("CoarseSeed"), "CoarseSeed"
    )
    reachability = _construct(
        ReachabilityConfig, values.get("Reachability"), "Reachability"
    )

    max_result_bytes = values.get("MaxResultBytes", DEFAULT_MAX_RESULT_BYTES)
    max_peak_result_bytes = values.get("MaxPeakResultBytes")
    grid = InversionGridSpec(
        space=space,
        axes=axes,
        max_points=values.get("MaxGridPoints", DEFAULT_MAX_GRID_POINTS),
        max_result_bytes=max_result_bytes,
        max_peak_result_bytes=max_peak_result_bytes,
    )
    try:
        pipeline = PipelineConfig(
            grid=grid,
            residual_scales=scales,
            solver=solver,
            projection=projection,
            execution=execution,
            coarse_seed=coarse_seed,
            initial_guess=values.get("InitialGuess", "domain_midpoint"),
            use_reachability_seed=_bool(
                values, "UseReachabilitySeed", True
            ),
            auxiliary_fields=auxiliary_fields,
            line_auxiliary_default_policy=values.get(
                "LineAuxiliaryDefaultPolicy",
                "linear",
            ),
            line_auxiliary_policies=values.get(
                "LineAuxiliaryPolicies", {}
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ApplicationConfigError("invalid pipeline configuration: %s" % exc) from exc

    compression = values.get("ArtifactCompression", "none")
    if compression not in {"none", "deflate"}:
        raise ApplicationConfigError("ArtifactCompression must be 'none' or 'deflate'")
    preprocess_path = _member_path(
        diagnostic_directory,
        values.get("PreprocessReport"),
        "PreprocessReport",
        "preprocess.json",
    )
    result_path = _member_path(
        output, values.get("ResultFile"), "ResultFile", "inverse.eosinv"
    )
    text_export_path = _member_path(
        output, export_filename, "Export.filename", "eos_table.dat"
    )
    text_export_manifest_path = diagnostic_directory / "text_export_manifest.json"
    checkpoint = _checkpoint(values, diagnostic_directory)
    ignored = tuple(sorted(set(values) & _IGNORED_LEGACY_PARAMETERS))
    return (
        ApplicationConfig(
            parameter_path=parameter_path,
            metadata_path=metadata_path,
            data_path=data_path,
            output_root=output_root,
            eos_directory=eos_directory,
            output_directory=output,
            diagnostic_directory=diagnostic_directory,
            preprocess_report_path=preprocess_path,
            result_path=result_path,
            text_export_path=text_export_path,
            text_export_manifest_path=text_export_manifest_path,
            pipeline=pipeline,
            reachability=reachability,
            checkpoint=checkpoint,
            text_export=text_export,
            export_enabled=export_enabled,
            eos_name=eos_name,
            run_name=run_name,
            specification=run_specification,
            overwrite_mode=overwrite_mode,
            overwrite=overwrite_value,
            artifact_compression=compression,
            ignored_legacy_parameters=ignored,
        ),
        table,
    )


__all__ = [
    "ApplicationConfig",
    "ApplicationConfigError",
    "ORDINARY_SUBCELL_CONFIG_FIELDS",
    "PARAMETER_GROUPS",
    "load_application_config",
    "load_parameter_file",
]
