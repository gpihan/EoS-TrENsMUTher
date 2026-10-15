"""First-axis Slurm-array planning, shard execution, and final assembly.

The scheduler layer owns no numerical algorithm.  Every shard is a contiguous
global flat range of the ordinary :class:`~eos_trensmuther.pipeline.InversionGrid`
and is evaluated by the same pipeline chunk engine used by local runs.  Shard
artifacts are partial and deliberately use ``.eosshard`` rather than posing as
complete ``.eosinv`` results; assembly is the only operation that constructs a
normal complete :class:`~eos_trensmuther.pipeline.PipelineResult`.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import shlex
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields as dataclass_fields, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .checkpoint import (
    PIPELINE_CHECKPOINT_SCHEMA_VERSION,
    CheckpointConfig,
    CheckpointStore,
)
from ._scheduler import execute_dynamic_tasks
from .outputs import (
    PersistenceFormatError,
    PersistenceIntegrityError,
    PersistencePathError,
    _array_sha256,
    _canonical_array,
    _canonical_json,
    _fsync_directory,
    _fsync_file,
    _read_json_manifest,
    _read_npy_member,
    _result_dtype,
    _safe_member_name,
    _write_archive,
    save_pipeline_result,
)
from .pipeline import (
    PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
    PIPELINE_RUN_MANIFEST_VERSION,
    EoSInversionPipeline,
    ExecutionConfig,
    PipelineConfig,
    PipelineConfigurationError,
    PipelinePlan,
    PipelineResult,
    PipelineStateError,
    PointDisposition,
    _PipelineChunkResult,
    _initialize_pipeline_worker,
    _run_pipeline_worker_chunk,
    _worker_thread_environment,
    pipeline_status_code,
)
from .reachability import ReachabilityConfig
from .tables import ForwardTable, TABLE_SNAPSHOT_FINGERPRINT_VERSION


ARRAY_PLAN_SCHEMA_VERSION = 3
# Frozen legacy executable-state identifiers retained for strict validation of
# existing plans/shards. They are not public command or package names.
ARRAY_PLAN_TYPE = "eos_inverter.first_axis_array_plan"
SHARD_SCHEMA_VERSION = 3
SHARD_TYPE = "eos_inverter.first_axis_shard"
_PLAN_INTEGRITY_SCHEMA = "first_axis_array_plan_v3"
_SHARD_INTEGRITY_SCHEMA = "first_axis_shard_v3"
_SHARD_MANIFEST = "manifest.json"
_MAX_MANIFEST_BYTES = 32 * 1024**2
_SHARD_SUFFIX = ".eosshard"

_SHARD_ARRAY_NAMES = (
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


class ExecutorError(PipelineStateError):
    """Base error for array planning, shard execution, and assembly."""


class ArrayPlanError(ExecutorError):
    """Raised when an array plan is invalid or incompatible."""


class ShardError(ExecutorError):
    """Raised when a partial shard is invalid or incompatible."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _digest(label: str, value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(("EoSInverter:%s\0" % label).encode("ascii"))
    digest.update(_canonical_json(_plain(value)))
    return digest.hexdigest()


def _strict_json(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PersistencePathError("array plan must be a regular file: %s" % path)

    def reject_pairs(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: %s" % key)
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError("non-finite JSON constant: %s" % value)

    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=reject_pairs,
                parse_constant=reject_constant,
            )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ArrayPlanError("invalid array plan JSON: %s" % exc) from exc
    if not isinstance(value, dict):
        raise ArrayPlanError("array plan must be a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PersistencePathError("array-plan directory is unsafe: %s" % path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError("array plan already exists: %s" % path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(_plain(value)))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            if path.exists() and (path.is_symlink() or not path.is_file()):
                raise PersistencePathError(
                    "array-plan overwrite target is unsafe: %s" % path
                )
            os.replace(str(temporary), str(path))
        else:
            os.link(str(temporary), str(path))
            temporary.unlink()
        _fsync_directory(path.parent)
        return path
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_directive(value: Optional[Any], name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ArrayPlanError("%s must be a non-empty string or None" % name)
    normalized = value.strip()
    if any(character in normalized for character in "\r\n\0"):
        raise ArrayPlanError("%s cannot contain control-line characters" % name)
    return normalized


@dataclass(frozen=True)
class SlurmArrayConfig:
    """Scheduler-only options used to render, but never submit, a script."""

    cpus_per_task: int = 1
    account: Optional[str] = None
    partition: Optional[str] = None
    wall_time: str = "01:00:00"
    memory: Optional[str] = None
    job_name: str = "eos-trensmuther"
    working_directory: Path = Path(".")
    log_directory: Path = Path("slurm-logs")
    activation_commands: Tuple[str, ...] = ()
    python_executable: str = sys.executable
    entrypoint: Path = Path("EoS-TrENsMUTher.py")

    def __post_init__(self) -> None:
        if (
            isinstance(self.cpus_per_task, (bool, np.bool_))
            or not isinstance(self.cpus_per_task, (int, np.integer))
            or int(self.cpus_per_task) < 1
        ):
            raise ArrayPlanError("cpus_per_task must be a positive integer")
        object.__setattr__(self, "cpus_per_task", int(self.cpus_per_task))
        for name in ("account", "partition", "memory"):
            object.__setattr__(self, name, _safe_directive(getattr(self, name), name))
        for name in ("wall_time", "job_name", "python_executable"):
            object.__setattr__(
                self, name, _safe_directive(getattr(self, name), name)
            )
        commands = tuple(self.activation_commands)
        if not all(isinstance(command, str) and command.strip() for command in commands):
            raise ArrayPlanError("activation_commands must contain non-empty strings")
        if any(any(character in command for character in "\r\n\0") for command in commands):
            raise ArrayPlanError("activation commands must each occupy one line")
        object.__setattr__(self, "activation_commands", commands)
        for name in ("working_directory", "log_directory", "entrypoint"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ArrayPlanError("%s must be path-like" % name)
            object.__setattr__(self, name, Path(value).expanduser().resolve())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpus_per_task": self.cpus_per_task,
            "account": self.account,
            "partition": self.partition,
            "wall_time": self.wall_time,
            "memory": self.memory,
            "job_name": self.job_name,
            "working_directory": str(self.working_directory),
            "log_directory": str(self.log_directory),
            "activation_commands": list(self.activation_commands),
            "python_executable": self.python_executable,
            "entrypoint": str(self.entrypoint),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SlurmArrayConfig":
        if not isinstance(value, Mapping):
            raise ArrayPlanError("slurm configuration must be an object")
        expected = {
            "cpus_per_task",
            "account",
            "partition",
            "wall_time",
            "memory",
            "job_name",
            "working_directory",
            "log_directory",
            "activation_commands",
            "python_executable",
            "entrypoint",
        }
        if set(value) != expected:
            raise ArrayPlanError("slurm configuration has unexpected fields")
        return cls(**dict(value))


@dataclass(frozen=True)
class FirstAxisShard:
    index: int
    axis_value: float
    start: int
    stop: int
    slice_shape: Tuple[int, ...]

    @property
    def size(self) -> int:
        return self.stop - self.start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "axis_value": self.axis_value,
            "start": self.start,
            "stop": self.stop,
            "slice_shape": list(self.slice_shape),
        }


@dataclass(frozen=True)
class FirstAxisArrayPlan:
    """Immutable first-axis ownership contract for one full inverse grid."""

    metadata_path: Path
    data_path: Path
    table_fingerprint: str
    table_name: str
    pipeline_configuration: Mapping[str, Any]
    reachability_configuration: Mapping[str, Any]
    numerical_routing_contract: Mapping[str, Any]
    axis_names: Tuple[str, ...]
    axis_units: Tuple[str, ...]
    axes: Tuple[np.ndarray, ...]
    shape: Tuple[int, ...]
    ordering: str
    full_plan_fingerprint: str
    output_directory: Path
    final_artifact: Path
    shard_directory: Path
    checkpoint_directory: Path
    compression: str
    slurm: SlurmArrayConfig
    fingerprint: str = ""

    def __post_init__(self) -> None:
        paths = (
            "metadata_path",
            "data_path",
            "output_directory",
            "final_artifact",
            "shard_directory",
            "checkpoint_directory",
        )
        for name in paths:
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ArrayPlanError("%s must be path-like" % name)
            object.__setattr__(self, name, Path(value).expanduser().resolve())
        names = tuple(self.axis_names)
        units = tuple(self.axis_units)
        shape = tuple(int(value) for value in self.shape)
        axes = tuple(np.asarray(axis, dtype=float) for axis in self.axes)
        if not names or len(names) != len(units) or len(names) != len(axes):
            raise ArrayPlanError("array-plan grid metadata does not align")
        if shape != tuple(int(axis.size) for axis in axes) or any(
            value < 2 for value in shape
        ):
            raise ArrayPlanError("array-plan grid shape is invalid")
        immutable_axes = []
        for axis in axes:
            if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
                raise ArrayPlanError("array-plan axes must be strictly increasing")
            copied = np.frombuffer(np.ascontiguousarray(axis).tobytes(), dtype=float)
            immutable_axes.append(copied.reshape(axis.shape))
        if self.ordering != "C_last_axis_fastest":
            raise ArrayPlanError("array plan requires C-order grid ownership")
        if self.compression not in {"none", "deflate"}:
            raise ArrayPlanError("compression must be 'none' or 'deflate'")
        if not isinstance(self.slurm, SlurmArrayConfig):
            raise ArrayPlanError("slurm configuration is invalid")
        for name in ("table_fingerprint", "table_name", "full_plan_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ArrayPlanError("%s must be non-empty" % name)
        object.__setattr__(self, "axis_names", names)
        object.__setattr__(self, "axis_units", units)
        object.__setattr__(self, "axes", tuple(immutable_axes))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(
            self, "pipeline_configuration", MappingProxyType(dict(self.pipeline_configuration))
        )
        object.__setattr__(
            self,
            "reachability_configuration",
            MappingProxyType(dict(self.reachability_configuration)),
        )
        if not isinstance(self.numerical_routing_contract, Mapping):
            raise ArrayPlanError(
                "array-plan numerical routing contract must be an object"
            )
        routing_contract = dict(self.numerical_routing_contract)
        if (
            routing_contract.get("version")
            != PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION
        ):
            raise ArrayPlanError(
                "array plan has an unsupported numerical routing contract"
            )
        object.__setattr__(
            self,
            "numerical_routing_contract",
            MappingProxyType(routing_contract),
        )
        expected = self._computed_fingerprint()
        if self.fingerprint and self.fingerprint != expected:
            raise ArrayPlanError("array-plan fingerprint does not match its contents")
        object.__setattr__(self, "fingerprint", expected)

    @property
    def slice_size(self) -> int:
        return int(math.prod(self.shape[1:]))

    @property
    def shards(self) -> Tuple[FirstAxisShard, ...]:
        return tuple(
            FirstAxisShard(
                index=index,
                axis_value=float(value),
                start=index * self.slice_size,
                stop=(index + 1) * self.slice_size,
                slice_shape=(1,) + self.shape[1:],
            )
            for index, value in enumerate(self.axes[0])
        )

    def shard(self, index: Any) -> FirstAxisShard:
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise ArrayPlanError("shard index must be an integer")
        converted = int(index)
        if converted < 0 or converted >= self.shape[0]:
            raise ArrayPlanError(
                "shard index %d lies outside [0, %d)" % (converted, self.shape[0])
            )
        return self.shards[converted]

    def shard_path(self, index: int) -> Path:
        return self.shard_directory / ("shard_%06d%s" % (index, _SHARD_SUFFIX))

    def shard_checkpoint(self, index: int) -> Path:
        return self.checkpoint_directory / ("shard_%06d" % index)

    def _content(self) -> Dict[str, Any]:
        return {
            "artifact_type": ARRAY_PLAN_TYPE,
            "schema_version": ARRAY_PLAN_SCHEMA_VERSION,
            "table": {
                "metadata_path": str(self.metadata_path),
                "data_path": str(self.data_path),
                "snapshot_fingerprint": self.table_fingerprint,
                "name": self.table_name,
            },
            "pipeline_configuration": _plain(self.pipeline_configuration),
            "reachability_configuration": _plain(self.reachability_configuration),
            "numerical_routing_contract": _plain(
                self.numerical_routing_contract
            ),
            "grid": {
                "axis_names": list(self.axis_names),
                "axis_units": list(self.axis_units),
                "axes": [axis.tolist() for axis in self.axes],
                "shape": list(self.shape),
                "ordering": self.ordering,
                "axis_0_ownership": "one_complete_fixed_axis_0_slice_per_task",
                "slice_size": self.slice_size,
            },
            "full_plan_fingerprint": self.full_plan_fingerprint,
            "paths": {
                "output_directory": str(self.output_directory),
                "final_artifact": str(self.final_artifact),
                "shard_directory": str(self.shard_directory),
                "checkpoint_directory": str(self.checkpoint_directory),
            },
            "compression": self.compression,
            "slurm": self.slurm.to_dict(),
            "shards": [shard.to_dict() for shard in self.shards],
        }

    def _computed_fingerprint(self) -> str:
        return _digest("FirstAxisArrayPlan:v2", self._content())

    def to_document(self) -> Dict[str, Any]:
        content = self._content()
        content["fingerprint"] = self.fingerprint
        content["integrity"] = {
            "algorithm": "sha256",
            "schema": _PLAN_INTEGRITY_SCHEMA,
            "digest": _digest("FirstAxisArrayPlanDocument:v2", content),
        }
        return content


def _plan_fingerprint(
    plan: PipelinePlan,
    table_fingerprint: str,
    reachability: Mapping[str, Any],
) -> str:
    return _digest(
        "ResolvedSlurmPipelinePlan:v2",
        {
            "table_fingerprint": table_fingerprint,
            "plan": plan.manifest(),
            "reachability": reachability,
        },
    )


def _reachability_dict(config: ReachabilityConfig) -> Dict[str, Any]:
    return {
        descriptor.name: getattr(config, descriptor.name)
        for descriptor in dataclass_fields(config)
    }


def create_array_plan(
    table: ForwardTable,
    pipeline_configuration: PipelineConfig,
    *,
    reachability_configuration: Optional[ReachabilityConfig] = None,
    output_directory: Any,
    final_artifact: Any,
    slurm: Optional[SlurmArrayConfig] = None,
    compression: str = "none",
) -> FirstAxisArrayPlan:
    """Resolve one immutable first-axis plan without submitting any job."""

    table.require_validated()
    if slurm is None:
        slurm = SlurmArrayConfig()
    if reachability_configuration is None:
        reachability_configuration = ReachabilityConfig()
    if not isinstance(reachability_configuration, ReachabilityConfig):
        raise TypeError("reachability_configuration must be ReachabilityConfig")
    execution = replace(
        pipeline_configuration.execution, workers=slurm.cpus_per_task
    )
    configured = replace(pipeline_configuration, execution=execution)
    pipeline = EoSInversionPipeline(
        table, configured, reachability_config=reachability_configuration
    )
    resolved = pipeline.resolved_plan
    output = Path(output_directory).expanduser().resolve()
    data_path = table.data_path
    if data_path is None or table.metadata is None:
        raise ArrayPlanError("validated table has no resolved data path")
    reachability = _reachability_dict(reachability_configuration)
    return FirstAxisArrayPlan(
        metadata_path=table.metadata_path,
        data_path=data_path,
        table_fingerprint=table.snapshot_fingerprint(),
        table_name=table.metadata.name,
        pipeline_configuration=configured.to_dict(),
        reachability_configuration=reachability,
        numerical_routing_contract=resolved.numerical_routing_contract(),
        axis_names=resolved.grid.axis_names,
        axis_units=resolved.grid.axis_units,
        axes=resolved.grid.axes,
        shape=resolved.grid.shape,
        ordering=resolved.grid.ordering,
        full_plan_fingerprint=_plan_fingerprint(
            resolved, table.snapshot_fingerprint(), reachability
        ),
        output_directory=output,
        final_artifact=final_artifact,
        shard_directory=output / "shards",
        checkpoint_directory=output / "checkpoint",
        compression=compression,
        slurm=slurm,
    )


def save_array_plan(
    plan: FirstAxisArrayPlan, path: Any, *, overwrite: bool = False
) -> Path:
    if not isinstance(plan, FirstAxisArrayPlan):
        raise TypeError("plan must be FirstAxisArrayPlan")
    return _atomic_json(Path(path), plan.to_document(), overwrite=overwrite)


def load_array_plan(path: Any) -> FirstAxisArrayPlan:
    source = Path(path).expanduser().resolve()
    document = _strict_json(source)
    if (
        document.get("artifact_type") != ARRAY_PLAN_TYPE
        or document.get("schema_version") != ARRAY_PLAN_SCHEMA_VERSION
    ):
        raise ArrayPlanError(
            "unsupported array-plan schema; plans from an older numerical "
            "routing contract cannot be executed"
        )
    required = {
        "artifact_type",
        "schema_version",
        "table",
        "pipeline_configuration",
        "reachability_configuration",
        "numerical_routing_contract",
        "grid",
        "full_plan_fingerprint",
        "paths",
        "compression",
        "slurm",
        "shards",
        "fingerprint",
        "integrity",
    }
    if set(document) != required:
        raise ArrayPlanError("array plan has unexpected fields")
    integrity = document["integrity"]
    if not isinstance(integrity, dict) or set(integrity) != {
        "algorithm",
        "schema",
        "digest",
    }:
        raise ArrayPlanError("array-plan integrity descriptor is invalid")
    without_integrity = dict(document)
    without_integrity.pop("integrity")
    if (
        integrity["algorithm"] != "sha256"
        or integrity["schema"] != _PLAN_INTEGRITY_SCHEMA
        or integrity["digest"]
        != _digest("FirstAxisArrayPlanDocument:v2", without_integrity)
    ):
        raise ArrayPlanError("array-plan integrity digest does not match")
    routing = document.get("numerical_routing_contract")
    routing_version = (
        routing.get("version") if isinstance(routing, dict) else None
    )
    if routing_version != PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION:
        qualifier = (
            "older"
            if isinstance(routing_version, int)
            and routing_version < PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION
            else "unsupported"
        )
        raise ArrayPlanError(
            "array plan uses an %s numerical routing contract and cannot be "
            "executed" % qualifier
        )
    table = document["table"]
    grid = document["grid"]
    paths = document["paths"]
    if not all(isinstance(value, dict) for value in (table, grid, paths)):
        raise ArrayPlanError("array plan has malformed sections")
    if set(table) != {
        "metadata_path",
        "data_path",
        "snapshot_fingerprint",
        "name",
    }:
        raise ArrayPlanError("array-plan table section has unexpected fields")
    if set(grid) != {
        "axis_names",
        "axis_units",
        "axes",
        "shape",
        "ordering",
        "axis_0_ownership",
        "slice_size",
    }:
        raise ArrayPlanError("array-plan grid section has unexpected fields")
    if set(paths) != {
        "output_directory",
        "final_artifact",
        "shard_directory",
        "checkpoint_directory",
    }:
        raise ArrayPlanError("array-plan path section has unexpected fields")
    try:
        PipelineConfig.from_dict(document["pipeline_configuration"])
        ReachabilityConfig(**document["reachability_configuration"])
    except (TypeError, ValueError, PipelineStateError) as exc:
        raise ArrayPlanError("array plan has invalid numerical configuration") from exc
    plan = FirstAxisArrayPlan(
        metadata_path=table["metadata_path"],
        data_path=table["data_path"],
        table_fingerprint=table["snapshot_fingerprint"],
        table_name=table["name"],
        pipeline_configuration=document["pipeline_configuration"],
        reachability_configuration=document["reachability_configuration"],
        numerical_routing_contract=document["numerical_routing_contract"],
        axis_names=tuple(grid["axis_names"]),
        axis_units=tuple(grid["axis_units"]),
        axes=tuple(np.asarray(axis, dtype=float) for axis in grid["axes"]),
        shape=tuple(grid["shape"]),
        ordering=grid["ordering"],
        full_plan_fingerprint=document["full_plan_fingerprint"],
        output_directory=paths["output_directory"],
        final_artifact=paths["final_artifact"],
        shard_directory=paths["shard_directory"],
        checkpoint_directory=paths["checkpoint_directory"],
        compression=document["compression"],
        slurm=SlurmArrayConfig.from_dict(document["slurm"]),
        fingerprint=document["fingerprint"],
    )
    expected_shards = [shard.to_dict() for shard in plan.shards]
    if document["shards"] != expected_shards:
        raise ArrayPlanError("array-plan shard ownership is inconsistent")
    if grid.get("slice_size") != plan.slice_size:
        raise ArrayPlanError("array-plan slice size is inconsistent")
    if grid.get("axis_0_ownership") != "one_complete_fixed_axis_0_slice_per_task":
        raise ArrayPlanError("array-plan ownership semantics are unsupported")
    return plan


def _validate_runtime_plan(
    plan: FirstAxisArrayPlan,
) -> Tuple[ForwardTable, PipelineConfig, ReachabilityConfig, EoSInversionPipeline]:
    table = ForwardTable.load_validated(plan.metadata_path, data_path=plan.data_path)
    if table.snapshot_fingerprint() != plan.table_fingerprint:
        raise ArrayPlanError("source table fingerprint changed after array planning")
    configuration = PipelineConfig.from_dict(plan.pipeline_configuration)
    try:
        reachability = ReachabilityConfig(**dict(plan.reachability_configuration))
    except (TypeError, ValueError) as exc:
        raise ArrayPlanError("invalid persisted reachability configuration") from exc
    pipeline = EoSInversionPipeline(
        table, configuration, reachability_config=reachability
    )
    resolved = pipeline.resolved_plan
    if dict(plan.numerical_routing_contract) != resolved.numerical_routing_contract():
        raise ArrayPlanError(
            "numerical routing contract changed after array planning"
        )
    if (
        resolved.grid.axis_names != plan.axis_names
        or resolved.grid.axis_units != plan.axis_units
        or resolved.grid.shape != plan.shape
        or any(
            not np.array_equal(left, right)
            for left, right in zip(resolved.grid.axes, plan.axes)
        )
    ):
        raise ArrayPlanError("resolved inverse grid changed after array planning")
    fingerprint = _plan_fingerprint(
        resolved, table.snapshot_fingerprint(), plan.reachability_configuration
    )
    if fingerprint != plan.full_plan_fingerprint:
        raise ArrayPlanError("resolved pipeline plan changed after array planning")
    return table, configuration, reachability, pipeline


def _slurm_environment() -> Dict[str, Optional[str]]:
    return {
        name: os.environ.get(name)
        for name in (
            "SLURM_JOB_ID",
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_CPUS_PER_TASK",
            "SLURM_SUBMIT_DIR",
        )
    }


def _aggregate_chunks(
    chunks: Sequence[_PipelineChunkResult], shard: FirstAxisShard
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    ordered = tuple(sorted(chunks, key=lambda item: item.start))
    expected_start = shard.start
    for chunk in ordered:
        if chunk.start != expected_start or chunk.stop <= chunk.start:
            raise ShardError("shard chunks do not provide exact contiguous coverage")
        expected_start = chunk.stop
    if expected_start != shard.stop:
        raise ShardError("shard chunks do not cover the owned range")
    arrays = {
        name: np.ascontiguousarray(
            np.concatenate([np.asarray(getattr(chunk, name)) for chunk in ordered], axis=0)
        )
        for name in _SHARD_ARRAY_NAMES
    }
    counters: Dict[str, int] = {}
    timings: Dict[str, float] = {}
    errors = []
    projection_failures = []
    for chunk in ordered:
        for name, value in chunk.counters.items():
            counters[name] = counters.get(name, 0) + int(value)
        for name, value in chunk.timing_seconds.items():
            timings[name] = timings.get(name, 0.0) + float(value)
        errors.extend(dict(item) for item in chunk.error_records)
        projection_failures.extend(
            dict(item) for item in chunk.projection_failure_records
        )
    errors.sort(key=lambda item: int(item["flat_index"]))
    projection_failures.sort(key=lambda item: int(item["flat_index"]))
    diagnostics = {
        "chunk_count": len(ordered),
        "stage_counters": counters,
        "worker_stage_seconds": timings,
        "errors": {"count": len(errors), "records": errors},
        "projection_failures": {
            "count": len(projection_failures),
            "records": projection_failures,
        },
        "worker_setup_seconds": [
            float(chunk.worker_setup_seconds) for chunk in ordered
        ],
        "worker_peak_rss_bytes": max(
            (int(chunk.worker_peak_rss_bytes) for chunk in ordered), default=0
        ),
    }
    return arrays, diagnostics


def _shard_digest(document: Mapping[str, Any]) -> str:
    return _digest("FirstAxisShardDocument:v2", document)


def _write_shard(
    path: Path,
    plan: FirstAxisArrayPlan,
    shard: FirstAxisShard,
    arrays: Mapping[str, np.ndarray],
    execution: Mapping[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = {
        name: {
            "member": "arrays/%s.npy" % name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
            "sha256": _array_sha256(array),
        }
        for name, array in arrays.items()
    }
    document: Dict[str, Any] = {
        "artifact_type": SHARD_TYPE,
        "schema_version": SHARD_SCHEMA_VERSION,
        "array_plan_fingerprint": plan.fingerprint,
        "full_plan_fingerprint": plan.full_plan_fingerprint,
        "table_fingerprint": plan.table_fingerprint,
        "numerical_routing_contract": _plain(
            plan.numerical_routing_contract
        ),
        "shard": shard.to_dict(),
        "execution": _plain(execution),
        "arrays": registry,
    }
    document["integrity"] = {
        "algorithm": "sha256",
        "schema": _SHARD_INTEGRITY_SCHEMA,
        "digest": _shard_digest(document),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_archive(temporary, document, arrays, plan.compression)
        _fsync_file(temporary)
        _load_shard(temporary, plan, shard)
        try:
            os.link(str(temporary), str(path))
        except FileExistsError:
            existing_arrays, _ = _load_shard(path, plan, shard)
            for name in _SHARD_ARRAY_NAMES:
                if not np.array_equal(
                    existing_arrays[name], arrays[name], equal_nan=True
                ):
                    raise ShardError(
                        "a concurrently published shard has different arrays"
                    )
            return path
        temporary.unlink()
        _fsync_directory(path.parent)
        return path
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_shard(
    path: Path,
    plan: FirstAxisArrayPlan,
    shard: FirstAxisShard,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise PersistencePathError("shard must be a regular file: %s" % path)
    configuration = PipelineConfig.from_dict(plan.pipeline_configuration)
    auxiliary = configuration.auxiliary_fields
    if auxiliary is None:
        metadata = ForwardTable(plan.metadata_path).load_metadata()
        auxiliary_count = len(
            tuple(
                field
                for field in metadata.auxiliary_names
                if metadata.fields[field].output
            )
        )
    else:
        auxiliary_count = len(auxiliary)
    tilde_columns = 4 if configuration.grid.space.value == "tilde" else 0
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)):
                raise PersistenceFormatError("shard contains duplicate ZIP members")
            if any(
                item.is_dir()
                or not _safe_member_name(item.filename)
                or item.flag_bits & 0x1
                or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                for item in infos
            ):
                raise PersistenceFormatError("shard contains an unsafe ZIP member")
            by_name = {item.filename: item for item in infos}
            if _SHARD_MANIFEST not in by_name:
                raise PersistenceFormatError("shard has no manifest")
            if by_name[_SHARD_MANIFEST].file_size > _MAX_MANIFEST_BYTES:
                raise PersistenceFormatError("shard manifest is unreasonably large")
            document = _read_json_manifest(archive, by_name[_SHARD_MANIFEST])
            if (
                document.get("artifact_type") != SHARD_TYPE
                or document.get("schema_version") != SHARD_SCHEMA_VERSION
            ):
                raise ShardError(
                    "unsupported shard schema; shards from an older numerical "
                    "routing contract cannot be assembled"
                )
            required = {
                "artifact_type",
                "schema_version",
                "array_plan_fingerprint",
                "full_plan_fingerprint",
                "table_fingerprint",
                "numerical_routing_contract",
                "shard",
                "execution",
                "arrays",
                "integrity",
            }
            if set(document) != required:
                raise PersistenceFormatError("shard manifest has unexpected fields")
            integrity = document["integrity"]
            without_integrity = dict(document)
            without_integrity.pop("integrity")
            if (
                not isinstance(integrity, dict)
                or integrity.get("algorithm") != "sha256"
                or integrity.get("schema") != _SHARD_INTEGRITY_SCHEMA
                or integrity.get("digest") != _shard_digest(without_integrity)
            ):
                raise PersistenceIntegrityError("shard manifest digest does not match")
            stored_routing = document.get("numerical_routing_contract")
            stored_routing_version = (
                stored_routing.get("version")
                if isinstance(stored_routing, dict)
                else None
            )
            if (
                stored_routing_version
                != PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION
            ):
                qualifier = (
                    "older"
                    if isinstance(stored_routing_version, int)
                    and stored_routing_version
                    < PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION
                    else "unsupported"
                )
                raise ShardError(
                    "shard uses an %s numerical routing contract and cannot "
                    "be assembled" % qualifier
                )
            if (
                document["array_plan_fingerprint"] != plan.fingerprint
                or document["full_plan_fingerprint"] != plan.full_plan_fingerprint
                or document["table_fingerprint"] != plan.table_fingerprint
                or document["numerical_routing_contract"]
                != _plain(plan.numerical_routing_contract)
                or document["shard"] != shard.to_dict()
            ):
                raise PersistenceIntegrityError("shard is incompatible with array plan")
            registry = document["arrays"]
            if not isinstance(registry, dict) or set(registry) != set(
                _SHARD_ARRAY_NAMES
            ):
                raise PersistenceFormatError("shard array registry is invalid")
            arrays: Dict[str, np.ndarray] = {}
            expected_members = {_SHARD_MANIFEST}
            for name in _SHARD_ARRAY_NAMES:
                descriptor = registry[name]
                if not isinstance(descriptor, dict) or set(descriptor) != {
                    "member",
                    "dtype",
                    "shape",
                    "nbytes",
                    "sha256",
                }:
                    raise PersistenceFormatError("shard array descriptor is invalid")
                dtype = _result_dtype(name)
                trailing = ()
                if name in {
                    "source_coordinates",
                    "achieved_targets",
                    "residual",
                    "scaled_residual",
                    "alternate_single_phase_source",
                }:
                    trailing = (len(plan.shape),)
                elif name in {"auxiliary_values", "auxiliary_defined"}:
                    trailing = (auxiliary_count,)
                elif name == "achieved_tilde_values":
                    trailing = (tilde_columns,)
                shape = (shard.size,) + trailing
                member = "arrays/%s.npy" % name
                nbytes = int(math.prod(shape)) * dtype.itemsize
                if (
                    descriptor["member"] != member
                    or descriptor["dtype"] != dtype.str
                    or descriptor["shape"] != list(shape)
                    or descriptor["nbytes"] != nbytes
                    or member not in by_name
                ):
                    raise PersistenceIntegrityError(
                        "shard array declaration disagrees with ownership layout"
                    )
                expected_members.add(member)
                arrays[name] = _read_npy_member(
                    archive,
                    by_name[member],
                    expected_dtype=dtype,
                    expected_shape=shape,
                    expected_nbytes=nbytes,
                    expected_sha256=descriptor["sha256"],
                )
            if set(by_name) != expected_members:
                raise PersistenceFormatError("shard contains unexpected ZIP members")
    except (PersistenceFormatError, PersistenceIntegrityError):
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PersistenceFormatError("invalid shard artifact: %s" % exc) from exc
    if not isinstance(document["execution"], dict):
        raise PersistenceFormatError("shard execution diagnostics must be an object")
    expected_execution = {
        "chunk_count",
        "stage_counters",
        "worker_stage_seconds",
        "errors",
        "projection_failures",
        "worker_setup_seconds",
        "worker_peak_rss_bytes",
        "completed",
        "execution_mode",
        "workers",
        "chunk_size",
        "restored_chunks",
        "computed_chunks_this_invocation",
        "checkpoint_directory",
        "elapsed_seconds_this_invocation",
        "slurm_environment",
    }
    if set(document["execution"]) != expected_execution:
        raise PersistenceFormatError(
            "shard execution diagnostics have unexpected fields"
        )
    if document["execution"].get("completed") is not True:
        raise PersistenceIntegrityError("shard execution is not complete")
    for section_name in ("errors", "projection_failures"):
        section = document["execution"].get(section_name)
        if not isinstance(section, dict) or set(section) != {"count", "records"}:
            raise PersistenceFormatError(
                "shard %s diagnostics are invalid" % section_name
            )
        records = section["records"]
        if (
            isinstance(section["count"], bool)
            or not isinstance(section["count"], int)
            or not isinstance(records, list)
            or section["count"] != len(records)
        ):
            raise PersistenceIntegrityError(
                "shard %s count and records disagree" % section_name
            )
        for record in records:
            if not isinstance(record, dict):
                raise PersistenceFormatError(
                    "shard diagnostic records must be objects"
                )
            index = record.get("flat_index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < shard.start
                or index >= shard.stop
            ):
                raise PersistenceIntegrityError(
                    "shard diagnostic index lies outside owned range"
                )
    return arrays, dict(document["execution"])


def _execute_missing_chunks(
    pipeline: EoSInversionPipeline,
    ranges: Sequence[Tuple[int, int]],
    checkpoint: CheckpointStore,
) -> Tuple[_PipelineChunkResult, ...]:
    if not ranges:
        return ()
    execution = pipeline.resolved_plan.config.execution
    completed = []
    if execution.workers == 1:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            pipeline._prepare()
            for start, stop in ranges:
                result = pipeline._execute_chunk(
                    pipeline.resolved_plan.grid.chunk(start, stop)
                )
                checkpoint.save_chunk(result)
                completed.append(result)
        return tuple(completed)

    requested = execution.start_method.value
    available = set(mp.get_all_start_methods())
    if requested not in available:
        raise PipelineConfigurationError(
            "process start method '%s' is unavailable" % requested
        )
    context = mp.get_context(requested)
    maximum_pending = (
        execution.max_pending_chunks
        if execution.max_pending_chunks is not None
        else max(1, 2 * execution.workers)
    )
    chunks = (
        pipeline.resolved_plan.grid.chunk(start, stop) for start, stop in ranges
    )
    shared_state_manifest = None
    if pipeline._table.first_order_transition is None:
        from .shared_state import create_shared_worker_state

        shared_state_directory = checkpoint.directory.parent / (
            checkpoint.directory.name + "_shared_worker_state_v1"
        )
        shared_state_manifest = create_shared_worker_state(
            pipeline._table,
            pipeline.resolved_plan,
            pipeline._reachability_config,
            shared_state_directory,
        )
    with _worker_thread_environment():
        with ProcessPoolExecutor(
            max_workers=execution.workers,
            mp_context=context,
            initializer=_initialize_pipeline_worker,
            initargs=(
                pipeline._worker_initialization(
                    shared_state_manifest=shared_state_manifest
                ),
            ),
        ) as executor:

            def publish_chunk(
                _chunk: Any, result: _PipelineChunkResult
            ) -> None:
                checkpoint.save_chunk(result)
                completed.append(result)

            execute_dynamic_tasks(
                executor,
                chunks,
                split_chunk=pipeline._split_checkpoint_chunk,
                run_task=_run_pipeline_worker_chunk,
                assemble_chunk=pipeline._assemble_checkpoint_chunk,
                publish_chunk=publish_chunk,
                maximum_pending_tasks=maximum_pending,
                failure_type=ShardError,
            )
    return tuple(sorted(completed, key=lambda item: (item.start, item.stop)))


def execute_array_shard(
    plan_or_path: Any,
    shard_index: Optional[Any] = None,
) -> Path:
    """Execute or safely resume one array task and publish one shard."""

    plan = (
        plan_or_path
        if isinstance(plan_or_path, FirstAxisArrayPlan)
        else load_array_plan(plan_or_path)
    )
    if shard_index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ArrayPlanError(
                "shard index is required outside a Slurm array task"
            )
        try:
            shard_index = int(raw)
        except ValueError as exc:
            raise ArrayPlanError("SLURM_ARRAY_TASK_ID is not an integer") from exc
    shard = plan.shard(shard_index)
    output_path = plan.shard_path(shard.index)
    if output_path.exists():
        _load_shard(output_path, plan, shard)
        return output_path
    _, _, _, pipeline = _validate_runtime_plan(plan)
    plan.shard_directory.mkdir(parents=True, exist_ok=True)
    plan.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointStore(
        CheckpointConfig(plan.shard_checkpoint(shard.index), resume=True),
        pipeline.resolved_plan,
        table_fingerprint=plan.table_fingerprint,
        reachability_configuration=plan.reachability_configuration,
        first_order_contract=(
            pipeline._first_order_manifest
            if pipeline._first_order_manifest["configured"]
            else {}
        ),
        owned_range=(shard.start, shard.stop),
    )
    started = time.perf_counter()
    restored = list(checkpoint.load_completed_chunks())
    restored_ranges = {(chunk.start, chunk.stop) for chunk in restored}
    missing = [item for item in checkpoint.ranges if item not in restored_ranges]
    computed = _execute_missing_chunks(pipeline, missing, checkpoint)
    chunks = tuple(restored) + tuple(computed)
    checkpoint.mark_complete()
    arrays, diagnostics = _aggregate_chunks(chunks, shard)
    diagnostics.update(
        {
            "completed": True,
            "execution_mode": (
                "serial" if pipeline.resolved_plan.config.execution.workers == 1 else "process"
            ),
            "workers": pipeline.resolved_plan.config.execution.workers,
            "chunk_size": pipeline.resolved_plan.config.execution.chunk_size,
            "restored_chunks": len(restored),
            "computed_chunks_this_invocation": len(computed),
            "checkpoint_directory": str(plan.shard_checkpoint(shard.index)),
            "elapsed_seconds_this_invocation": float(time.perf_counter() - started),
            "slurm_environment": _slurm_environment(),
        }
    )
    return _write_shard(output_path, plan, shard, arrays, diagnostics)


def _copy_shard_arrays(
    buffer: Any,
    plan: FirstAxisArrayPlan,
    shard: FirstAxisShard,
    arrays: Mapping[str, np.ndarray],
) -> None:
    full_size = int(math.prod(plan.shape))
    for name in _SHARD_ARRAY_NAMES:
        source = np.asarray(arrays[name])
        destination = getattr(buffer, name)
        trailing = destination.shape[len(plan.shape) :]
        flattened = destination.reshape((full_size,) + trailing)
        if source.shape != (shard.size,) + trailing:
            raise ShardError("shard array %s has an incompatible shape" % name)
        flattened[shard.start : shard.stop] = source


def assemble_array_shards(
    plan_or_path: Any,
    *,
    output_path: Optional[Any] = None,
    overwrite: bool = False,
) -> PipelineResult:
    """Validate exact shard coverage and publish a normal full result."""

    plan = (
        plan_or_path
        if isinstance(plan_or_path, FirstAxisArrayPlan)
        else load_array_plan(plan_or_path)
    )
    _, _, _, pipeline = _validate_runtime_plan(plan)
    if not plan.shard_directory.is_dir() or plan.shard_directory.is_symlink():
        raise ShardError("shard directory is missing or unsafe")
    expected_names = {plan.shard_path(shard.index).name for shard in plan.shards}
    actual_names = {
        path.name
        for path in plan.shard_directory.iterdir()
        if path.name.endswith(_SHARD_SUFFIX)
    }
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        details = []
        if missing:
            details.append("missing: %s" % ", ".join(missing))
        if extra:
            details.append("unexpected: %s" % ", ".join(extra))
        raise ShardError("shard coverage is incomplete (%s)" % "; ".join(details))
    buffer = pipeline.resolved_plan.allocate_result_buffer()
    shard_execution = []
    all_errors = []
    all_projection_failures = []
    stage_counters: Dict[str, int] = {}
    stage_seconds: Dict[str, float] = {}
    total_task_seconds = 0.0
    for shard in plan.shards:
        arrays, execution = _load_shard(plan.shard_path(shard.index), plan, shard)
        _copy_shard_arrays(buffer, plan, shard, arrays)
        shard_execution.append(
            {
                "index": shard.index,
                "axis_value": shard.axis_value,
                "start": shard.start,
                "stop": shard.stop,
                "elapsed_seconds_this_invocation": execution.get(
                    "elapsed_seconds_this_invocation", 0.0
                ),
                "slurm_environment": execution.get("slurm_environment", {}),
            }
        )
        total_task_seconds += float(execution.get("elapsed_seconds_this_invocation", 0.0))
        for name, value in execution.get("stage_counters", {}).items():
            stage_counters[name] = stage_counters.get(name, 0) + int(value)
        for name, value in execution.get("worker_stage_seconds", {}).items():
            stage_seconds[name] = stage_seconds.get(name, 0.0) + float(value)
        all_errors.extend(execution.get("errors", {}).get("records", []))
        all_projection_failures.extend(
            execution.get("projection_failures", {}).get("records", [])
        )
    all_errors.sort(key=lambda item: int(item["flat_index"]))
    all_projection_failures.sort(key=lambda item: int(item["flat_index"]))
    summary = pipeline._status_summary(buffer)
    status_counts = {
        disposition.value: int(
            np.count_nonzero(
                buffer.disposition_codes == pipeline_status_code(disposition)
            )
        )
        for disposition in PointDisposition
    }
    maximum_task_seconds = max(
        (
            float(item["elapsed_seconds_this_invocation"])
            for item in shard_execution
        ),
        default=0.0,
    )
    cell_shape = tuple(size - 1 for size in pipeline._table.shape)
    complete_ordinary = pipeline._uses_complete_ordinary_localization()
    reachability_method = plan.reachability_configuration.get(
        "method", "exhaustive"
    )
    run_manifest = {
        "pipeline_run_manifest_version": PIPELINE_RUN_MANIFEST_VERSION,
        "completed": True,
        "execution": {
            "mode": "slurm_array_assembled",
            "workers_per_task": plan.slurm.cpus_per_task,
            "array_size": plan.shape[0],
            "axis_0_name": plan.axis_names[0],
            "axis_0_ownership": "one_complete_fixed_axis_0_slice_per_task",
            "aggregation": "validated_shards_in_C_order",
        },
        "slurm_array": {
            "array_plan_schema_version": ARRAY_PLAN_SCHEMA_VERSION,
            "shard_schema_version": SHARD_SCHEMA_VERSION,
            "array_plan_fingerprint": plan.fingerprint,
            "full_plan_fingerprint": plan.full_plan_fingerprint,
            "task_provenance": shard_execution,
            "submission_performed_by_library": False,
        },
        "checkpoint": {
            "enabled": True,
            "ownership": "exclusive_directory_per_axis_0_shard",
            "checkpoint_schema_version": PIPELINE_CHECKPOINT_SCHEMA_VERSION,
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
            "snapshot_fingerprint": plan.table_fingerprint,
            "stable_persistence_identifier": True,
        },
        "reachability": {
            "method": reachability_method,
            "implementation": (
                "balanced_source_index_hierarchy_exact_leaf"
                if reachability_method == "hierarchical"
                else "batched_all_cell_reference"
            ),
            "configuration": _plain(plan.reachability_configuration),
            "rigorous_exact_boxes": True,
            "routing_outward_rounded": bool(
                pipeline.resolved_plan.config.solver.root_residual_tolerance
                > 0.0
            ),
            "complete_leaf_localization": {
                "enabled": complete_ordinary,
                "scope": (
                    "ordinary_dimensions_1_to_4"
                    if complete_ordinary
                    else None
                ),
                "backend": reachability_method if complete_ordinary else None,
                "lifetime": (
                    "worker_chunk_only" if complete_ordinary else None
                ),
                "persisted_in_checkpoint": False,
            },
        },
        "first_order": dict(pipeline._first_order_manifest),
        "auxiliary_evaluation": summary["auxiliary_phase_policy"],
        "projection": {
            "enabled": pipeline.resolved_plan.config.projection.enabled,
            "method": (
                "deterministic_local_multistart"
                if pipeline.resolved_plan.config.projection.enabled
                else None
            ),
            "globally_certified": False,
        },
        "software": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "platform": sys.platform,
        },
        "timing_seconds": {
            "sum_task_invocations": total_task_seconds,
            "maximum_task_invocation": maximum_task_seconds,
            "worker_stage_sum": stage_seconds,
        },
        "performance": {
            "sample_scope": "assembled_first_axis_shards",
            "computed_points_per_second_this_invocation": (
                pipeline.resolved_plan.grid.size / maximum_task_seconds
                if maximum_task_seconds > 0.0
                else 0.0
            ),
            "restored_points": 0,
            "computed_points_this_invocation": pipeline.resolved_plan.grid.size,
            "rates": {},
            "array_task_seconds_sum": total_task_seconds,
            "array_task_seconds_maximum": maximum_task_seconds,
        },
        "stage_counters": stage_counters,
        "status_counts": status_counts,
        "status_summary": summary,
        "errors": {"count": len(all_errors), "records": all_errors},
        "projection_failures": {
            "count": len(all_projection_failures),
            "records": all_projection_failures,
        },
        "detailed_diagnostics_retained": False,
    }
    result = buffer.finalize(run_manifest)
    destination = plan.final_artifact if output_path is None else Path(output_path)
    save_pipeline_result(
        result,
        destination,
        overwrite=overwrite,
        compression=plan.compression,
    )
    return result


def render_slurm_script(
    plan_or_path: Any,
    script_path: Any,
    *,
    plan_path: Optional[Any] = None,
    overwrite: bool = False,
) -> Path:
    """Render an inspectable ``sbatch`` script; this function never submits."""

    if isinstance(plan_or_path, FirstAxisArrayPlan):
        plan = plan_or_path
        if plan_path is None:
            raise ArrayPlanError("plan_path is required when rendering an in-memory plan")
        persisted_plan = Path(plan_path).expanduser().resolve()
    else:
        persisted_plan = Path(plan_or_path).expanduser().resolve()
        plan = load_array_plan(persisted_plan)
    target = Path(script_path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError("Slurm script already exists: %s" % target)
    directives = [
        "#SBATCH --job-name=%s" % plan.slurm.job_name,
        "#SBATCH --array=0-%d" % (plan.shape[0] - 1),
        "#SBATCH --time=%s" % plan.slurm.wall_time,
        "#SBATCH --cpus-per-task=%d" % plan.slurm.cpus_per_task,
        "#SBATCH --output=%s"
        % shlex.quote(str(plan.slurm.log_directory / "%x_%A_%a.out")),
        "#SBATCH --error=%s"
        % shlex.quote(str(plan.slurm.log_directory / "%x_%A_%a.err")),
    ]
    if plan.slurm.account is not None:
        directives.append("#SBATCH --account=%s" % plan.slurm.account)
    if plan.slurm.partition is not None:
        directives.append("#SBATCH --partition=%s" % plan.slurm.partition)
    if plan.slurm.memory is not None:
        directives.append("#SBATCH --mem=%s" % plan.slurm.memory)
    command = " ".join(
        (
            shlex.quote(plan.slurm.python_executable),
            shlex.quote(str(plan.slurm.entrypoint)),
            "slurm-task",
            "--plan",
            shlex.quote(str(persisted_plan)),
            '--shard-index "${SLURM_ARRAY_TASK_ID}"',
        )
    )
    lines = [
        "#!/bin/bash",
        *directives,
        "",
        "set -euo pipefail",
        "mkdir -p %s" % shlex.quote(str(plan.slurm.log_directory)),
        "cd %s" % shlex.quote(str(plan.slurm.working_directory)),
        *plan.slurm.activation_commands,
        command,
        "",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name, suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o755)
        if overwrite:
            os.replace(str(temporary), str(target))
        else:
            os.link(str(temporary), str(target))
            temporary.unlink()
        _fsync_directory(target.parent)
        return target
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "ARRAY_PLAN_SCHEMA_VERSION",
    "ARRAY_PLAN_TYPE",
    "SHARD_SCHEMA_VERSION",
    "SHARD_TYPE",
    "ArrayPlanError",
    "ExecutorError",
    "FirstAxisArrayPlan",
    "FirstAxisShard",
    "ShardError",
    "SlurmArrayConfig",
    "assemble_array_shards",
    "create_array_plan",
    "execute_array_shard",
    "load_array_plan",
    "render_slurm_script",
    "save_array_plan",
]
