"""Atomic chunk checkpoints for restartable table-wide inversion runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .outputs import (
    PersistenceError,
    PersistenceFormatError,
    PersistenceIntegrityError,
    PersistencePathError,
    PersistenceVersionError,
    _array_sha256,
    _canonical_array,
    _canonical_json,
    _fsync_directory,
    _fsync_file,
    _plain_json,
    _read_json_manifest,
    _read_npy_member,
    _reject_duplicate_keys,
    _reject_json_constant,
    _result_dtype,
    _safe_member_name,
    _write_archive,
)
from .pipeline import (
    PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
    PIPELINE_RESULT_SCHEMA_VERSION,
    PIPELINE_RUN_MANIFEST_VERSION,
    PipelinePlan,
    _PipelineChunkResult,
)


PIPELINE_CHECKPOINT_SCHEMA_VERSION = 4
# Frozen legacy on-disk identifiers; changing them would silently invalidate
# compatible checkpoints rather than merely renaming the public application.
PIPELINE_CHECKPOINT_TYPE = "eos_inverter.pipeline_checkpoint"
PIPELINE_CHECKPOINT_CHUNK_TYPE = "eos_inverter.pipeline_checkpoint_chunk"
_CHECKPOINT_MANIFEST = "checkpoint.json"
_CHUNK_MANIFEST = "manifest.json"
_CHUNK_SUFFIX = ".eoschunk"
_MAX_CHECKPOINT_JSON_BYTES = 16 * 1024**2
_CHUNK_ARRAY_NAMES = (
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


@dataclass(frozen=True)
class CheckpointConfig:
    """Restart policy for a directory of independently atomic chunk files.

    A new run may initialize a missing or empty directory.  When ``resume`` is
    true, a compatible existing checkpoint is reused.  No checkpoint is ever
    silently replaced or mixed with another run.
    """

    directory: Path
    resume: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.directory, bool):
            raise TypeError("checkpoint directory must be path-like")
        try:
            directory = Path(self.directory).expanduser()
        except (TypeError, ValueError) as exc:
            raise TypeError("checkpoint directory must be path-like") from exc
        if not directory.name:
            raise ValueError("checkpoint directory must not be a filesystem root")
        if not isinstance(self.resume, bool):
            raise TypeError("checkpoint resume must be boolean")
        object.__setattr__(self, "directory", directory)


def _digest(kind: str, value: Mapping[str, Any]) -> str:
    encoded = _canonical_json(value)
    digest = hashlib.sha256()
    digest.update(("EoSInverter:%s:v1\0" % kind).encode("ascii"))
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


def _with_integrity(kind: str, value: Mapping[str, Any]) -> Dict[str, Any]:
    output = dict(value)
    output["integrity"] = {
        "algorithm": "sha256",
        "schema": "%s_v1" % kind,
        "digest": _digest(kind, value),
    }
    return output


def _verify_integrity(kind: str, value: Mapping[str, Any]) -> Dict[str, Any]:
    document = dict(value)
    integrity = document.pop("integrity", None)
    if not isinstance(integrity, dict) or set(integrity) != {
        "algorithm",
        "schema",
        "digest",
    }:
        raise PersistenceFormatError("checkpoint integrity descriptor is invalid")
    if (
        integrity["algorithm"] != "sha256"
        or integrity["schema"] != "%s_v1" % kind
    ):
        raise PersistenceFormatError("checkpoint integrity schema is unsupported")
    actual = integrity["digest"]
    if (
        not isinstance(actual, str)
        or len(actual) != 64
        or any(character not in "0123456789abcdef" for character in actual)
    ):
        raise PersistenceFormatError("checkpoint digest is not SHA-256 hex")
    if _digest(kind, document) != actual:
        raise PersistenceIntegrityError("checkpoint descriptor digest does not match")
    return document


def _read_strict_json(path: Path) -> Dict[str, Any]:
    try:
        if path.stat().st_size > _MAX_CHECKPOINT_JSON_BYTES:
            raise PersistenceFormatError("checkpoint manifest is unreasonably large")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except PersistenceError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PersistenceFormatError("invalid checkpoint manifest: %s" % exc) from exc
    if not isinstance(value, dict):
        raise PersistenceFormatError("checkpoint manifest must be an object")
    return value


def _atomic_json(path: Path, document: Mapping[str, Any], *, replace: bool) -> None:
    payload = _canonical_json(document)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(str(temporary), str(path))
        else:
            try:
                os.link(str(temporary), str(path))
            except FileExistsError as exc:
                raise FileExistsError("checkpoint manifest already exists") from exc
            temporary.unlink()
        published = True
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                if not published:
                    pass


def _execution_independent_configuration(plan: PipelinePlan) -> Dict[str, Any]:
    configuration = plan.config.to_dict()
    execution = dict(configuration["execution"])
    execution.pop("workers", None)
    execution.pop("start_method", None)
    execution.pop("max_pending_chunks", None)
    execution.pop("chunk_size", None)
    execution.pop("worker_task_size", None)
    configuration["execution"] = execution
    grid = dict(configuration["grid"])
    grid.pop("max_points", None)
    grid.pop("max_result_bytes", None)
    grid.pop("max_peak_result_bytes", None)
    configuration["grid"] = grid
    return configuration


def _run_contract(
    plan: PipelinePlan,
    table_fingerprint: str,
    reachability_configuration: Mapping[str, Any],
    first_order_contract: Mapping[str, Any],
    owned_range: Tuple[int, int],
) -> Dict[str, Any]:
    return {
        "pipeline_result_schema_version": PIPELINE_RESULT_SCHEMA_VERSION,
        "pipeline_run_manifest_version": PIPELINE_RUN_MANIFEST_VERSION,
        "numerical_routing_contract": plan.numerical_routing_contract(),
        "table_snapshot_fingerprint": table_fingerprint,
        "configuration": _execution_independent_configuration(plan),
        "resolved_grid": {
            "space": plan.grid.space.value,
            "axis_names": list(plan.grid.axis_names),
            "axis_sha256": [_array_sha256(axis) for axis in plan.grid.axes],
            "shape": list(plan.grid.shape),
            "ordering": plan.grid.ordering,
            "transform_id": plan.grid.transform_id,
            "constraints": plan.grid.constraints.identifier,
        },
        "source_names": list(plan.source_names),
        "target_names": list(plan.target_names),
        "auxiliary_names": list(plan.auxiliary_names),
        "residual_scales": plan.residual_scales.tolist(),
        "reachability_configuration": _plain_json(
            reachability_configuration, "reachability_configuration"
        ),
        "first_order_contract": _plain_json(
            first_order_contract, "first_order_contract"
        ),
        "active_domain": "full_cell_grid",
        "owned_target_range": list(owned_range),
    }


def _expected_ranges(
    plan: PipelinePlan, owned_range: Tuple[int, int]
) -> Tuple[Tuple[int, int], ...]:
    chunk_size = plan.config.execution.chunk_size
    owned_start, owned_stop = owned_range
    return tuple(
        (start, min(owned_stop, start + chunk_size))
        for start in range(owned_start, owned_stop, chunk_size)
    )


def _chunk_shapes(plan: PipelinePlan, count: int) -> Dict[str, Tuple[int, ...]]:
    dimension = plan.grid.dimension
    auxiliary = len(plan.auxiliary_names)
    tilde_columns = 4 if plan.grid.space.value == "tilde" else 0
    vector = (count, dimension)
    return {
        "source_coordinates": vector,
        "achieved_targets": vector,
        "residual": vector,
        "scaled_residual": vector,
        "scaled_residual_norm_inf": (count,),
        "scaled_distance_l2": (count,),
        "auxiliary_values": (count, auxiliary),
        "auxiliary_defined": (count, auxiliary),
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
        "alternate_single_phase_source": vector,
        "alternate_single_phase_source_defined": (count,),
    }


class CheckpointStore:
    """Single-writer store for validated, independently atomic chunk results."""

    def __init__(
        self,
        config: CheckpointConfig,
        plan: PipelinePlan,
        *,
        table_fingerprint: str,
        reachability_configuration: Mapping[str, Any],
        first_order_contract: Optional[Mapping[str, Any]] = None,
        owned_range: Optional[Tuple[int, int]] = None,
    ) -> None:
        if not isinstance(config, CheckpointConfig):
            raise TypeError("config must be a CheckpointConfig")
        if not isinstance(plan, PipelinePlan):
            raise TypeError("plan must be a PipelinePlan")
        absolute = Path(os.path.abspath(os.fspath(config.directory)))
        if absolute.exists() and absolute.is_symlink():
            raise PersistencePathError("checkpoint directory cannot be a symbolic link")
        try:
            absolute.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PersistencePathError(
                "could not create checkpoint directory: %s" % exc
            ) from exc
        if not absolute.is_dir():
            raise PersistencePathError("checkpoint path is not a directory")
        self.directory = absolute.resolve(strict=True)
        self.chunks_directory = self.directory / "chunks"
        if self.chunks_directory.exists() and self.chunks_directory.is_symlink():
            raise PersistencePathError("checkpoint chunks path cannot be a symlink")
        self.chunks_directory.mkdir(exist_ok=True)
        if not self.chunks_directory.is_dir():
            raise PersistencePathError("checkpoint chunks path is not a directory")
        self.plan = plan
        if owned_range is None:
            owned_range = (0, plan.grid.size)
        if (
            not isinstance(owned_range, tuple)
            or len(owned_range) != 2
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in owned_range
            )
        ):
            raise TypeError("owned_range must be a pair of integer flat indices")
        owned_range = (int(owned_range[0]), int(owned_range[1]))
        if not (0 <= owned_range[0] < owned_range[1] <= plan.grid.size):
            raise ValueError(
                "owned_range must satisfy 0 <= start < stop <= grid size"
            )
        self.owned_range = owned_range
        self.ranges = _expected_ranges(plan, owned_range)
        run_contract = _run_contract(
            plan,
            table_fingerprint,
            reachability_configuration,
            {} if first_order_contract is None else first_order_contract,
            owned_range,
        )
        self.run_fingerprint = _digest("PipelineRunContract", run_contract)
        layout = {
            "run_fingerprint": self.run_fingerprint,
            "grid_size": plan.grid.size,
            "owned_range": list(owned_range),
            "chunk_size": plan.config.execution.chunk_size,
            "ranges": [list(value) for value in self.ranges],
        }
        self.layout_fingerprint = _digest("PipelineCheckpointLayout", layout)
        self._base_document = {
            "artifact_type": PIPELINE_CHECKPOINT_TYPE,
            "checkpoint_schema_version": PIPELINE_CHECKPOINT_SCHEMA_VERSION,
            "run_fingerprint": self.run_fingerprint,
            "layout_fingerprint": self.layout_fingerprint,
            "run_contract": run_contract,
            "layout": layout,
            "completed": False,
            "completed_chunk_count": 0,
        }
        self.manifest_path = self.directory / _CHECKPOINT_MANIFEST
        self._open(config.resume)

    def _open(self, resume: bool) -> None:
        if self.manifest_path.exists():
            if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
                raise PersistencePathError("checkpoint manifest must be a regular file")
            if not resume:
                raise FileExistsError(
                    "checkpoint already exists and resume=False: %s"
                    % self.directory
                )
            stored = _verify_integrity(
                "PipelineCheckpoint", _read_strict_json(self.manifest_path)
            )
            stored_contract = stored.get("run_contract")
            stored_routing = (
                stored_contract.get("numerical_routing_contract")
                if isinstance(stored_contract, dict)
                else None
            )
            stored_routing_version = (
                stored_routing.get("version")
                if isinstance(stored_routing, dict)
                else None
            )
            if (
                not isinstance(stored_contract, dict)
                or stored_contract.get("pipeline_run_manifest_version")
                != PIPELINE_RUN_MANIFEST_VERSION
                or stored_routing_version
                != PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION
            ):
                raise PersistenceVersionError(
                    "checkpoint uses an older numerical routing contract and "
                    "cannot be resumed; preserve it as read-only evidence and "
                    "start a fresh run"
                )
            expected = dict(self._base_document)
            for key in (
                "artifact_type",
                "checkpoint_schema_version",
                "run_fingerprint",
                "layout_fingerprint",
                "run_contract",
                "layout",
            ):
                if stored.get(key) != expected[key]:
                    raise PersistenceIntegrityError(
                        "checkpoint is incompatible at field '%s'" % key
                    )
            if not isinstance(stored.get("completed"), bool):
                raise PersistenceFormatError("checkpoint completion flag is invalid")
            count = stored.get("completed_chunk_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise PersistenceFormatError("checkpoint chunk count is invalid")
            self._document = stored
            return
        unexpected = [
            path
            for path in self.directory.iterdir()
            if path.name != "chunks"
            and not (
                path.name.startswith(".%s." % _CHECKPOINT_MANIFEST)
                and path.name.endswith(".tmp")
            )
        ]
        chunk_entries = [
            path
            for path in self.chunks_directory.iterdir()
            if not (path.name.startswith(".") and path.name.endswith(".tmp"))
        ]
        if unexpected or chunk_entries:
            raise PersistencePathError(
                "checkpoint directory is non-empty but has no checkpoint.json"
            )
        self._document = dict(self._base_document)
        _atomic_json(
            self.manifest_path,
            _with_integrity("PipelineCheckpoint", self._document),
            replace=False,
        )

    @property
    def completed(self) -> bool:
        return bool(self._document["completed"])

    def _chunk_path(self, start: int, stop: int) -> Path:
        return self.chunks_directory / (
            "%012d-%012d%s" % (start, stop, _CHUNK_SUFFIX)
        )

    def _chunk_document(
        self, chunk: _PipelineChunkResult, arrays: Mapping[str, np.ndarray]
    ) -> Dict[str, Any]:
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
        value = {
            "artifact_type": PIPELINE_CHECKPOINT_CHUNK_TYPE,
            "checkpoint_schema_version": PIPELINE_CHECKPOINT_SCHEMA_VERSION,
            "run_fingerprint": self.run_fingerprint,
            "layout_fingerprint": self.layout_fingerprint,
            "start": int(chunk.start),
            "stop": int(chunk.stop),
            "counters": _plain_json(chunk.counters, "chunk.counters"),
            "timing_seconds": _plain_json(
                chunk.timing_seconds, "chunk.timing_seconds"
            ),
            "error_records": _plain_json(
                chunk.error_records, "chunk.error_records"
            ),
            "projection_failure_records": _plain_json(
                chunk.projection_failure_records,
                "chunk.projection_failure_records",
            ),
            "worker_setup_seconds": float(chunk.worker_setup_seconds),
            "worker_peak_rss_bytes": int(chunk.worker_peak_rss_bytes),
            "arrays": registry,
        }
        return _with_integrity("PipelineCheckpointChunk", value)

    def save_chunk(self, chunk: _PipelineChunkResult) -> Path:
        if not isinstance(chunk, _PipelineChunkResult):
            raise TypeError("chunk must be a pipeline chunk result")
        expected = (int(chunk.start), int(chunk.stop))
        if expected not in self.ranges:
            raise PersistenceFormatError("chunk range is outside checkpoint layout")
        path = self._chunk_path(*expected)
        if path.exists():
            raise FileExistsError("checkpoint chunk already exists: %s" % path)
        shapes = _chunk_shapes(self.plan, chunk.stop - chunk.start)
        arrays = {
            name: _canonical_array(
                getattr(chunk, name), _result_dtype(name), shapes[name]
            )
            for name in _CHUNK_ARRAY_NAMES
        }
        document = self._chunk_document(chunk, arrays)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".%s." % path.name,
            suffix=".tmp",
            dir=str(self.chunks_directory),
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        published = False
        try:
            _write_archive(temporary, document, arrays, "none")
            _fsync_file(temporary)
            self._load_chunk_path(temporary, expected)
            try:
                os.link(str(temporary), str(path))
            except FileExistsError as exc:
                raise FileExistsError(
                    "checkpoint chunk already exists: %s" % path
                ) from exc
            temporary.unlink()
            published = True
            _fsync_directory(self.chunks_directory)
            return path
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    if not published:
                        pass

    def _load_chunk_path(
        self, path: Path, expected_range: Tuple[int, int]
    ) -> _PipelineChunkResult:
        count = expected_range[1] - expected_range[0]
        shapes = _chunk_shapes(self.plan, count)
        try:
            with zipfile.ZipFile(path, mode="r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(set(names)):
                    raise PersistenceFormatError(
                        "checkpoint chunk contains duplicate members"
                    )
                by_name = {info.filename: info for info in infos}
                if any(
                    info.is_dir()
                    or not _safe_member_name(info.filename)
                    or info.flag_bits & 0x1
                    or info.compress_type
                    not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    for info in infos
                ):
                    raise PersistenceFormatError(
                        "checkpoint chunk contains an unsafe member"
                    )
                if _CHUNK_MANIFEST not in by_name:
                    raise PersistenceFormatError(
                        "checkpoint chunk has no manifest"
                    )
                if by_name[_CHUNK_MANIFEST].file_size > _MAX_CHECKPOINT_JSON_BYTES:
                    raise PersistenceFormatError(
                        "checkpoint chunk manifest is unreasonably large"
                    )
                expected_payload_limit = (
                    _MAX_CHECKPOINT_JSON_BYTES
                    + sum(
                        int(math.prod(shape)) * _result_dtype(name).itemsize
                        + 65536
                        for name, shape in shapes.items()
                    )
                )
                if sum(info.file_size for info in infos) > expected_payload_limit:
                    raise PersistenceFormatError(
                        "checkpoint chunk exceeds its storage contract"
                    )
                document = _verify_integrity(
                    "PipelineCheckpointChunk",
                    _read_json_manifest(archive, by_name[_CHUNK_MANIFEST]),
                )
                required = {
                    "artifact_type",
                    "checkpoint_schema_version",
                    "run_fingerprint",
                    "layout_fingerprint",
                    "start",
                    "stop",
                    "counters",
                    "timing_seconds",
                    "error_records",
                    "projection_failure_records",
                    "worker_setup_seconds",
                    "worker_peak_rss_bytes",
                    "arrays",
                }
                if set(document) != required:
                    raise PersistenceFormatError(
                        "checkpoint chunk descriptor has unexpected fields"
                    )
                if (
                    document["artifact_type"] != PIPELINE_CHECKPOINT_CHUNK_TYPE
                    or document["checkpoint_schema_version"]
                    != PIPELINE_CHECKPOINT_SCHEMA_VERSION
                    or document["run_fingerprint"] != self.run_fingerprint
                    or document["layout_fingerprint"] != self.layout_fingerprint
                    or (document["start"], document["stop"]) != expected_range
                ):
                    raise PersistenceIntegrityError(
                        "checkpoint chunk is incompatible with this run"
                    )
                registry = document["arrays"]
                if not isinstance(registry, dict) or set(registry) != set(
                    _CHUNK_ARRAY_NAMES
                ):
                    raise PersistenceFormatError(
                        "checkpoint chunk array registry is invalid"
                    )
                expected_members = {_CHUNK_MANIFEST}
                arrays: Dict[str, np.ndarray] = {}
                for name in _CHUNK_ARRAY_NAMES:
                    descriptor = registry[name]
                    if not isinstance(descriptor, dict) or set(descriptor) != {
                        "member",
                        "dtype",
                        "shape",
                        "nbytes",
                        "sha256",
                    }:
                        raise PersistenceFormatError(
                            "checkpoint array descriptor is invalid"
                        )
                    dtype = _result_dtype(name)
                    shape = shapes[name]
                    member = "arrays/%s.npy" % name
                    nbytes = int(math.prod(shape)) * dtype.itemsize
                    if (
                        descriptor["member"] != member
                        or descriptor["dtype"] != dtype.str
                        or descriptor["shape"] != list(shape)
                        or descriptor["nbytes"] != nbytes
                    ):
                        raise PersistenceIntegrityError(
                            "checkpoint array declaration disagrees with layout"
                        )
                    digest = descriptor["sha256"]
                    if not isinstance(digest, str):
                        raise PersistenceFormatError(
                            "checkpoint array digest is invalid"
                        )
                    if member not in by_name:
                        raise PersistenceFormatError(
                            "checkpoint array member is missing"
                        )
                    expected_members.add(member)
                    arrays[name] = _read_npy_member(
                        archive,
                        by_name[member],
                        expected_dtype=dtype,
                        expected_shape=shape,
                        expected_nbytes=nbytes,
                        expected_sha256=digest,
                    )
                    if dtype.kind == "f":
                        arrays[name] = np.ascontiguousarray(
                            arrays[name], dtype=float
                        )
                if set(by_name) != expected_members:
                    raise PersistenceFormatError(
                        "checkpoint chunk contains unexpected members"
                    )
        except PersistenceError:
            raise
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            raise PersistenceFormatError(
                "invalid checkpoint chunk: %s" % exc
            ) from exc

        mappings = ("counters", "timing_seconds")
        for name in mappings:
            if not isinstance(document[name], dict):
                raise PersistenceFormatError("checkpoint %s must be an object" % name)
        records = ("error_records", "projection_failure_records")
        for name in records:
            if not isinstance(document[name], list) or not all(
                isinstance(item, dict) for item in document[name]
            ):
                raise PersistenceFormatError("checkpoint %s must be an array" % name)
        return _PipelineChunkResult(
            start=expected_range[0],
            stop=expected_range[1],
            counters=dict(document["counters"]),
            timing_seconds=dict(document["timing_seconds"]),
            error_records=tuple(dict(item) for item in document["error_records"]),
            projection_failure_records=tuple(
                dict(item) for item in document["projection_failure_records"]
            ),
            worker_setup_seconds=document["worker_setup_seconds"],
            worker_peak_rss_bytes=document["worker_peak_rss_bytes"],
            **arrays,
        )

    def load_completed_chunks(self) -> Tuple[_PipelineChunkResult, ...]:
        expected_names = {
            self._chunk_path(start, stop).name for start, stop in self.ranges
        }
        actual_names = {
            path.name
            for path in self.chunks_directory.iterdir()
            if path.name.endswith(_CHUNK_SUFFIX)
        }
        unexpected = actual_names - expected_names
        if unexpected:
            raise PersistenceFormatError(
                "checkpoint has unexpected chunk file(s): %s"
                % ", ".join(sorted(unexpected))
            )
        chunks = []
        for expected in self.ranges:
            path = self._chunk_path(*expected)
            if path.exists():
                if path.is_symlink() or not path.is_file():
                    raise PersistencePathError(
                        "checkpoint chunk must be a regular file: %s" % path
                    )
                chunks.append(self._load_chunk_path(path, expected))
        if self.completed and len(chunks) != len(self.ranges):
            raise PersistenceIntegrityError(
                "checkpoint is marked complete but chunks are missing"
            )
        return tuple(chunks)

    def mark_complete(self) -> None:
        present = sum(
            self._chunk_path(start, stop).is_file()
            for start, stop in self.ranges
        )
        if present != len(self.ranges):
            raise PersistenceIntegrityError(
                "cannot complete a checkpoint with missing chunks"
            )
        self._document = dict(self._base_document)
        self._document["completed"] = True
        self._document["completed_chunk_count"] = present
        _atomic_json(
            self.manifest_path,
            _with_integrity("PipelineCheckpoint", self._document),
            replace=True,
        )


__all__ = [
    "CheckpointConfig",
    "CheckpointStore",
    "PIPELINE_CHECKPOINT_SCHEMA_VERSION",
    "PIPELINE_CHECKPOINT_TYPE",
]
