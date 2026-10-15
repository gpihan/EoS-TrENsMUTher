"""Versioned, self-contained persistence for completed pipeline results.

The artifact is a ZIP container with one strict JSON descriptor and one NPY
member per grid axis/result column.  Loading never follows paths recorded in
the run manifest and never enables NumPy pickle support.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import math
import os
import tempfile
import zipfile
from dataclasses import fields as dataclass_fields
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .pipeline import (
    DEFAULT_MAX_RESULT_BYTES,
    PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
    PIPELINE_RESULT_SCHEMA_VERSION,
    PIPELINE_RUN_MANIFEST_VERSION,
    GridSpace,
    InversionGrid,
    PipelineConfig,
    PipelineError,
    PipelinePlan,
    PipelineResult,
)
from .transforms import ChargeConstraintSpec


PIPELINE_ARTIFACT_SCHEMA_VERSION = 3
"""Current schema of the completed-result archive container."""

# Frozen legacy on-disk identifier. Changing it would make accepted v1
# artifacts unreadable; it is not a current package or product name.
PIPELINE_ARTIFACT_TYPE = "eos_inverter.pipeline_result"
DEFAULT_MAX_MANIFEST_BYTES = 64 * 1024**2
DEFAULT_MAX_ARTIFACT_BYTES = DEFAULT_MAX_RESULT_BYTES + 64 * 1024**2
DEFAULT_MAX_ARTIFACT_UNCOMPRESSED_BYTES = (
    DEFAULT_MAX_RESULT_BYTES + 64 * 1024**2
)

_MANIFEST_MEMBER = "manifest.json"
_INTEGRITY_SCHEMA = "pipeline_result_artifact_v2"
_READABLE_PIPELINE_RUN_MANIFEST_VERSIONS = frozenset(
    {8, 9, 10, PIPELINE_RUN_MANIFEST_VERSION}
)
_LEGACY_NUMERICAL_ROUTING_CONTRACT_VERSION = 1
_PRE_JACOBIAN_NUMERICAL_ROUTING_CONTRACT_VERSION = 2
_RESULT_ARRAY_NAMES = (
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
_LEGACY_RESULT_ARRAY_NAMES = tuple(
    name for name in _RESULT_ARRAY_NAMES if name != "phase_selection_reason_codes"
)
_FLOAT_RESULT_NAMES = frozenset(
    {
        "source_coordinates",
        "achieved_targets",
        "residual",
        "scaled_residual",
        "scaled_residual_norm_inf",
        "scaled_distance_l2",
        "auxiliary_values",
        "achieved_tilde_values",
        "line_mixture_fraction",
        "alternate_single_phase_source",
    }
)
_BOOLEAN_RESULT_NAMES = frozenset(
    {
        "auxiliary_defined",
        "achieved_tilde_defined",
        "mixed_candidate",
        "exact_mixed_candidate",
        "phase_boundary",
        "phase_multivalued",
        "line_mixture_fraction_defined",
        "alternate_single_phase_source_defined",
    }
)
_UINT32_RESULT_NAMES = frozenset({"mixed_candidate_entry_count"})


class PersistenceError(PipelineError):
    """Base exception for completed-result persistence failures."""


class PersistencePathError(PersistenceError):
    """Raised when an artifact path is unsafe or unusable."""


class PersistenceFormatError(PersistenceError):
    """Raised when an artifact does not satisfy its declared schema."""


class PersistenceIntegrityError(PersistenceError):
    """Raised when persisted bytes do not match their integrity metadata."""


class PersistenceVersionError(PersistenceFormatError):
    """Raised when an artifact uses an unsupported schema version."""


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


def _positive_limit(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("%s must be a positive integer" % name)
    converted = int(value)
    if converted <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return converted


def _plain_json(value: Any, path: str = "value") -> Any:
    if isinstance(value, Mapping):
        output: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PersistenceFormatError("%s keys must be strings" % path)
            output[key] = _plain_json(item, "%s.%s" % (path, key))
        return output
    if isinstance(value, (tuple, list)):
        return [_plain_json(item, "%s[]" % path) for item in value]
    if isinstance(value, np.generic):
        return _plain_json(value.item(), path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PersistenceFormatError(
                "%s contains a non-finite JSON number" % path
            )
        return value
    raise PersistenceFormatError(
        "%s contains unsupported type %s" % (path, type(value).__name__)
    )


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PersistenceFormatError(
            "artifact descriptor is not canonical JSON: %s" % exc
        ) from exc


def _expect_object(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise PersistenceFormatError("%s must be a JSON object" % path)
    return value


def _expect_keys(
    value: Mapping[str, Any],
    required: Sequence[str],
    path: str,
    *,
    optional: Sequence[str] = (),
) -> None:
    required_set = set(required)
    actual = set(value)
    missing = sorted(required_set - actual)
    extra = sorted(actual - required_set - set(optional))
    if missing or extra:
        details = []
        if missing:
            details.append("missing %s" % ", ".join(missing))
        if extra:
            details.append("unexpected %s" % ", ".join(extra))
        raise PersistenceFormatError("%s has %s" % (path, "; ".join(details)))


def _expect_string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise PersistenceFormatError("%s must be %s" % (path, qualifier))
    return value


def _expect_integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PersistenceFormatError(
            "%s must be an integer at least %d" % (path, minimum)
        )
    return int(value)


def _expect_finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PersistenceFormatError("%s must be a finite number" % path)
    converted = float(value)
    if not math.isfinite(converted):
        raise PersistenceFormatError("%s must be a finite number" % path)
    return converted


def _string_tuple(
    value: Any, path: str, *, allow_duplicates: bool = False
) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise PersistenceFormatError("%s must be an array" % path)
    result = tuple(
        _expect_string(item, "%s[]" % path) for item in value
    )
    if not allow_duplicates and len(set(result)) != len(result):
        raise PersistenceFormatError("%s cannot contain duplicates" % path)
    return result


def _float_vector(value: Any, length: int, path: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise PersistenceFormatError(
            "%s must contain exactly %d values" % (path, length)
        )
    return np.asarray(
        [_expect_finite_float(item, "%s[]" % path) for item in value],
        dtype=float,
    )


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    if contiguous.nbytes:
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _artifact_digest(document_without_integrity: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"EoSInverter:PipelineResultArtifact:v2\0")
    encoded = _canonical_json(document_without_integrity)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


def _result_shapes(result: PipelineResult) -> Dict[str, Tuple[int, ...]]:
    shape = result.grid.shape
    dimension = result.grid.dimension
    auxiliary_count = len(result.plan.auxiliary_names)
    tilde_columns = 4 if result.grid.space is GridSpace.TILDE else 0
    vector = shape + (dimension,)
    return {
        "source_coordinates": vector,
        "achieved_targets": vector,
        "residual": vector,
        "scaled_residual": vector,
        "scaled_residual_norm_inf": shape,
        "scaled_distance_l2": shape,
        "auxiliary_values": shape + (auxiliary_count,),
        "auxiliary_defined": shape + (auxiliary_count,),
        "achieved_tilde_values": shape + (tilde_columns,),
        "achieved_tilde_defined": shape,
        "disposition_codes": shape,
        "route_codes": shape,
        "prescreen_codes": shape,
        "solve_status_codes": shape,
        "fast_status_codes": shape,
        "proof_status_codes": shape,
        "projection_quality_codes": shape,
        "initial_guess_source_codes": shape,
        "phase_codes": shape,
        "phase_route_codes": shape,
        "phase_selection_reason_codes": shape,
        "mixed_candidate": shape,
        "exact_mixed_candidate": shape,
        "mixed_candidate_entry_count": shape,
        "mixed_verified_root_count": shape,
        "phase_boundary": shape,
        "phase_multivalued": shape,
        "line_mixture_fraction": shape,
        "line_mixture_fraction_defined": shape,
        "alternate_single_phase_source": vector,
        "alternate_single_phase_source_defined": shape,
    }


def _result_dtype(name: str) -> np.dtype:
    if name in _FLOAT_RESULT_NAMES:
        return np.dtype("<f8")
    if name in _BOOLEAN_RESULT_NAMES:
        return np.dtype(np.bool_)
    if name in _UINT32_RESULT_NAMES:
        return np.dtype("<u4")
    return np.dtype(np.uint8)


def _canonical_array(value: Any, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise PersistenceFormatError(
            "array shape %r does not match expected %r" % (array.shape, shape)
        )
    if dtype == np.dtype("<f8"):
        if array.dtype.kind != "f" or array.dtype.itemsize != 8:
            raise PersistenceFormatError("result floating arrays must use float64")
    elif array.dtype != dtype:
        raise PersistenceFormatError(
            "array dtype %s does not match expected %s" % (array.dtype, dtype)
        )
    return np.ascontiguousarray(array, dtype=dtype)


def _constraint_descriptor(value: ChargeConstraintSpec) -> Dict[str, Any]:
    return {
        "dimension": value.dimension,
        "dependent_densities": {
            name: dict(relation)
            for name, relation in value.dependent_densities.items()
        },
        "source": value.source,
        "absolute_tolerance": value.absolute_tolerance,
        "relative_tolerance": value.relative_tolerance,
        "model": value.model,
    }


def _plan_estimates(plan: PipelinePlan) -> Dict[str, int]:
    return {
        descriptor.name: int(getattr(plan, descriptor.name))
        for descriptor in dataclass_fields(PipelinePlan)
        if descriptor.name.startswith("estimated_")
    }


def _plan_descriptor(plan: PipelinePlan, axis_keys: Sequence[str]) -> Dict[str, Any]:
    grid = plan.grid
    return {
        "configuration": plan.config.to_dict(),
        "table": {
            "name": plan.table_name,
            "revision": plan.table_revision,
            "metadata_path": plan.metadata_path,
            "data_path": plan.data_path,
        },
        "source": {
            "names": list(plan.source_names),
            "units": list(plan.source_units),
        },
        "target": {
            "names": list(plan.target_names),
            "units": list(plan.target_units),
        },
        "auxiliary": {
            "names": list(plan.auxiliary_names),
            "units": list(plan.auxiliary_units),
        },
        "residual_scales": plan.residual_scales.tolist(),
        "storage_estimates": _plan_estimates(plan),
        "constraints": _constraint_descriptor(grid.constraints),
        "grid": {
            "space": grid.space.value,
            "axis_names": list(grid.axis_names),
            "axis_units": list(grid.axis_units),
            "axis_arrays": list(axis_keys),
            "shape": list(grid.shape),
            "size": grid.size,
            "canonical_target_names": list(grid.canonical_target_names),
            "canonical_target_units": list(grid.canonical_target_units),
            "sampling_envelope_lower": grid.sampling_envelope_lower.tolist(),
            "sampling_envelope_upper": grid.sampling_envelope_upper.tolist(),
            "coverage": grid.coverage,
            "transform_id": grid.transform_id,
            "dependent_tilde_names": list(grid.dependent_tilde_names),
            "dependent_tilde_relations": {
                name: dict(relation)
                for name, relation in grid.dependent_tilde_relations.items()
            },
            "ordering": grid.ordering,
        },
    }


def _completed_manifest(result: PipelineResult) -> Dict[str, Any]:
    manifest = _plain_json(result.manifest, "manifest")
    if not isinstance(manifest, dict):
        raise PersistenceFormatError("result manifest must be a JSON object")
    if manifest.get("pipeline_result_schema_version") != PIPELINE_RESULT_SCHEMA_VERSION:
        raise PersistenceVersionError(
            "result manifest schema is not supported by this writer"
        )
    run = manifest.get("run")
    if not isinstance(run, dict) or run.get("completed") is not True:
        raise PersistenceFormatError(
            "only terminal pipeline results with run.completed=true can be saved"
        )
    if run.get("pipeline_run_manifest_version") != PIPELINE_RUN_MANIFEST_VERSION:
        raise PersistenceVersionError(
            "run manifest schema is not supported by this writer"
        )
    return manifest


def _build_document(
    result: PipelineResult,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    run_manifest = _completed_manifest(result)
    arrays: Dict[str, np.ndarray] = {}
    registry: Dict[str, Any] = {}
    axis_keys = []
    for index, axis in enumerate(result.grid.axes):
        key = "grid_axis_%03d" % index
        axis_keys.append(key)
        arrays[key] = _canonical_array(axis, np.dtype("<f8"), (axis.size,))
    shapes = _result_shapes(result)
    for name in _RESULT_ARRAY_NAMES:
        key = "result_" + name
        arrays[key] = _canonical_array(
            getattr(result, name), _result_dtype(name), shapes[name]
        )
    for key, array in arrays.items():
        registry[key] = {
            "member": "arrays/%s.npy" % key,
            "kind": "grid_axis" if key.startswith("grid_axis_") else "result",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
            "sha256": _array_sha256(array),
        }
    document: Dict[str, Any] = {
        "artifact_type": PIPELINE_ARTIFACT_TYPE,
        "artifact_schema_version": PIPELINE_ARTIFACT_SCHEMA_VERSION,
        "pipeline_result_schema_version": PIPELINE_RESULT_SCHEMA_VERSION,
        "pipeline_run_manifest_version": PIPELINE_RUN_MANIFEST_VERSION,
        "plan": _plan_descriptor(result.plan, axis_keys),
        "run_manifest": run_manifest,
        "arrays": registry,
    }
    document["integrity"] = {
        "algorithm": "sha256",
        "schema": _INTEGRITY_SCHEMA,
        "digest": _artifact_digest(document),
    }
    return document, arrays


def _lexical_absolute_path(value: Any, operation: str) -> Path:
    try:
        supplied = Path(value).expanduser()
    except (TypeError, ValueError) as exc:
        raise PersistencePathError(
            "%s path is invalid: %s" % (operation, exc)
        ) from exc
    absolute = Path(os.path.abspath(os.fspath(supplied)))
    if not absolute.name:
        raise PersistencePathError("%s path must name a file" % operation)
    return absolute


def _resolved_save_path(value: Any) -> Path:
    absolute = _lexical_absolute_path(value, "output")
    try:
        parent = absolute.parent.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise PersistencePathError(
            "output parent directory is unavailable: %s" % exc
        ) from exc
    if not parent.is_dir():
        raise PersistencePathError("output parent is not a directory: %s" % parent)
    target = parent / absolute.name
    if target.is_symlink():
        raise PersistencePathError("output path cannot be a symbolic link")
    return target


def _write_archive(
    path: Path,
    document: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    compression: str,
) -> None:
    if compression == "none":
        method = zipfile.ZIP_STORED
        options: Dict[str, Any] = {}
    elif compression == "deflate":
        method = zipfile.ZIP_DEFLATED
        options = {"compresslevel": 6}
    else:
        raise ValueError("compression must be 'none' or 'deflate'")
    manifest_bytes = _canonical_json(document)
    try:
        with zipfile.ZipFile(
            path,
            mode="w",
            compression=method,
            allowZip64=True,
            **options,
        ) as archive:
            archive.writestr(_MANIFEST_MEMBER, manifest_bytes)
            registry = document["arrays"]
            for key in sorted(arrays):
                member = registry[key]["member"]
                with archive.open(member, mode="w", force_zip64=True) as stream:
                    np.lib.format.write_array(
                        stream,
                        arrays[key],
                        version=(1, 0),
                        allow_pickle=False,
                    )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise PersistenceError("could not write result artifact: %s" % exc) from exc


def _fsync_file(path: Path) -> None:
    try:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    except OSError as exc:
        raise PersistenceError("could not synchronize result artifact: %s" % exc) from exc


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
            raise PersistenceError(
                "could not synchronize output directory: %s" % exc
            ) from exc


def save_pipeline_result(
    result: PipelineResult,
    path: Any,
    *,
    overwrite: bool = False,
    compression: str = "none",
) -> Path:
    """Atomically save one completed :class:`PipelineResult`.

    The temporary artifact is written, synchronized, and fully reloaded before
    it is published in the destination directory.  Existing outputs are never
    replaced unless ``overwrite=True`` is explicit.
    """

    if not isinstance(result, PipelineResult):
        raise TypeError("result must be a PipelineResult")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")
    if compression not in {"none", "deflate"}:
        raise ValueError("compression must be 'none' or 'deflate'")
    target = _resolved_save_path(path)
    if target.exists():
        if target.is_dir():
            raise PersistencePathError("output path is a directory: %s" % target)
        if not overwrite:
            raise FileExistsError("result artifact already exists: %s" % target)

    document, arrays = _build_document(result)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name,
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    published = False
    try:
        _write_archive(temporary, document, arrays, compression)
        _fsync_file(temporary)
        archive_size = max(DEFAULT_MAX_ARTIFACT_BYTES, temporary.stat().st_size)
        uncompressed_size = max(
            DEFAULT_MAX_ARTIFACT_UNCOMPRESSED_BYTES,
            len(_canonical_json(document))
            + sum(array.nbytes + 4096 for array in arrays.values()),
        )
        result_size = max(
            DEFAULT_MAX_RESULT_BYTES,
            int(result.plan.estimated_result_bytes),
        )
        load_pipeline_result(
            temporary,
            max_archive_bytes=archive_size,
            max_uncompressed_bytes=uncompressed_size,
            max_manifest_bytes=max(
                DEFAULT_MAX_MANIFEST_BYTES, len(_canonical_json(document))
            ),
            max_result_bytes=result_size,
        )
        if overwrite:
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise PersistencePathError(
                    "overwrite target must be a regular file"
                )
            os.replace(str(temporary), str(target))
            published = True
        else:
            try:
                os.link(str(temporary), str(target))
            except FileExistsError as exc:
                raise FileExistsError(
                    "result artifact already exists: %s" % target
                ) from exc
            except OSError as exc:
                raise PersistencePathError(
                    "could not atomically publish result artifact without "
                    "overwriting: %s" % exc
                ) from exc
            published = True
            temporary.unlink()
        _fsync_directory(target.parent)
        return target
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                if not published:
                    pass


def _safe_member_name(name: str) -> bool:
    if not name or "\\" in name:
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _read_json_manifest(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> Dict[str, Any]:
    try:
        raw = archive.read(info)
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, _DuplicateJSONKey) as exc:
        raise PersistenceFormatError("invalid artifact manifest: %s" % exc) from exc
    return _expect_object(value, "manifest")


def _shape_tuple(value: Any, path: str) -> Tuple[int, ...]:
    if not isinstance(value, list):
        raise PersistenceFormatError("%s must be an array" % path)
    if len(value) > 8:
        raise PersistenceFormatError("%s has too many dimensions" % path)
    return tuple(
        _expect_integer(item, "%s[]" % path, minimum=0) for item in value
    )


def _checked_nbytes(shape: Tuple[int, ...], dtype: np.dtype, path: str) -> int:
    count = 1
    for size in shape:
        count *= size
    nbytes = count * dtype.itemsize
    if nbytes < 0:
        raise PersistenceFormatError("%s has an invalid byte count" % path)
    return nbytes


def _read_npy_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    expected_dtype: np.dtype,
    expected_shape: Tuple[int, ...],
    expected_nbytes: int,
    expected_sha256: str,
) -> np.ndarray:
    try:
        with archive.open(info, mode="r") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    stream, max_header_size=65536
                )
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    stream, max_header_size=65536
                )
            else:
                raise PersistenceFormatError(
                    "unsupported NPY member version %r" % (version,)
                )
            dtype = np.dtype(dtype)
            if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
                raise PersistenceFormatError(
                    "object and structured NPY dtypes are forbidden"
                )
            if fortran_order:
                raise PersistenceFormatError("NPY members must use C ordering")
            if tuple(shape) != expected_shape or dtype.str != expected_dtype.str:
                raise PersistenceFormatError(
                    "NPY header does not match the declared shape/dtype"
                )
            if _checked_nbytes(tuple(shape), dtype, "NPY member") != expected_nbytes:
                raise PersistenceFormatError(
                    "NPY header does not match the declared byte count"
                )
            data = bytearray(expected_nbytes)
            view = memoryview(data)
            offset = 0
            while offset < expected_nbytes:
                count = stream.readinto(view[offset:])
                if not count:
                    raise PersistenceIntegrityError("NPY member is truncated")
                offset += int(count)
            if stream.read(1):
                raise PersistenceFormatError("NPY member contains trailing bytes")
    except PersistenceError:
        raise
    except (EOFError, OSError, ValueError) as exc:
        raise PersistenceFormatError("invalid NPY member: %s" % exc) from exc
    digest = hashlib.sha256(view).hexdigest()
    if digest != expected_sha256:
        raise PersistenceIntegrityError("array SHA-256 digest does not match")
    return np.frombuffer(data, dtype=dtype).reshape(expected_shape, order="C")


def _parse_constraint(value: Any) -> ChargeConstraintSpec:
    document = _expect_object(value, "plan.constraints")
    required = (
        "dimension",
        "dependent_densities",
        "source",
        "absolute_tolerance",
        "relative_tolerance",
        "model",
    )
    _expect_keys(document, required, "plan.constraints")
    dependent = _expect_object(
        document["dependent_densities"], "plan.constraints.dependent_densities"
    )
    relations: Dict[str, Dict[str, float]] = {}
    for name, relation_value in dependent.items():
        _expect_string(name, "dependent density name")
        relation = _expect_object(
            relation_value,
            "plan.constraints.dependent_densities.%s" % name,
        )
        relations[name] = {
            _expect_string(source, "density relation source"): _expect_finite_float(
                coefficient, "density relation coefficient"
            )
            for source, coefficient in relation.items()
        }
    try:
        return ChargeConstraintSpec(
            dimension=_expect_integer(
                document["dimension"], "plan.constraints.dimension", minimum=1
            ),
            dependent_densities=relations,
            source=_expect_string(document["source"], "plan.constraints.source"),
            absolute_tolerance=_expect_finite_float(
                document["absolute_tolerance"],
                "plan.constraints.absolute_tolerance",
            ),
            relative_tolerance=_expect_finite_float(
                document["relative_tolerance"],
                "plan.constraints.relative_tolerance",
            ),
            model=_expect_string(document["model"], "plan.constraints.model"),
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceFormatError("invalid charge constraints: %s" % exc) from exc


def _parse_relations(value: Any) -> Dict[str, Dict[str, float]]:
    document = _expect_object(value, "plan.grid.dependent_tilde_relations")
    output: Dict[str, Dict[str, float]] = {}
    for name, relation_value in document.items():
        name = _expect_string(name, "dependent tilde relation name")
        relation = _expect_object(
            relation_value, "dependent tilde relation %s" % name
        )
        output[name] = {
            _expect_string(source, "tilde relation source"): _expect_finite_float(
                coefficient, "tilde relation coefficient"
            )
            for source, coefficient in relation.items()
        }
    return output


def _expected_storage_estimates(value: Any) -> Dict[str, int]:
    document = _expect_object(value, "plan.storage_estimates")
    expected_names = {
        descriptor.name
        for descriptor in dataclass_fields(PipelinePlan)
        if descriptor.name.startswith("estimated_")
    }
    _expect_keys(document, tuple(expected_names), "plan.storage_estimates")
    return {
        name: _expect_integer(document[name], "plan.storage_estimates.%s" % name)
        for name in expected_names
    }


def _reconstruct_plan(
    value: Any,
    arrays: Mapping[str, np.ndarray],
    run_manifest_version: int,
) -> PipelinePlan:
    document = _expect_object(value, "plan")
    required = (
        "configuration",
        "table",
        "source",
        "target",
        "auxiliary",
        "residual_scales",
        "storage_estimates",
        "constraints",
        "grid",
    )
    _expect_keys(document, required, "plan")
    table = _expect_object(document["table"], "plan.table")
    _expect_keys(
        table, ("name", "revision", "metadata_path", "data_path"), "plan.table"
    )
    source = _expect_object(document["source"], "plan.source")
    target = _expect_object(document["target"], "plan.target")
    auxiliary = _expect_object(document["auxiliary"], "plan.auxiliary")
    for name, section in (
        ("source", source),
        ("target", target),
        ("auxiliary", auxiliary),
    ):
        _expect_keys(section, ("names", "units"), "plan.%s" % name)
    source_names = _string_tuple(source["names"], "plan.source.names")
    source_units = _string_tuple(
        source["units"], "plan.source.units", allow_duplicates=True
    )
    target_names = _string_tuple(target["names"], "plan.target.names")
    target_units = _string_tuple(
        target["units"], "plan.target.units", allow_duplicates=True
    )
    auxiliary_names = _string_tuple(
        auxiliary["names"], "plan.auxiliary.names"
    )
    auxiliary_units = _string_tuple(
        auxiliary["units"],
        "plan.auxiliary.units",
        allow_duplicates=True,
    )
    if len(source_names) != len(source_units):
        raise PersistenceFormatError("source names and units do not align")
    if len(target_names) != len(target_units):
        raise PersistenceFormatError("target names and units do not align")
    if len(auxiliary_names) != len(auxiliary_units):
        raise PersistenceFormatError("auxiliary names and units do not align")

    grid_value = _expect_object(document["grid"], "plan.grid")
    grid_keys = (
        "space",
        "axis_names",
        "axis_units",
        "axis_arrays",
        "shape",
        "size",
        "canonical_target_names",
        "canonical_target_units",
        "sampling_envelope_lower",
        "sampling_envelope_upper",
        "coverage",
        "transform_id",
        "dependent_tilde_names",
        "dependent_tilde_relations",
        "ordering",
    )
    _expect_keys(grid_value, grid_keys, "plan.grid")
    axis_names = _string_tuple(grid_value["axis_names"], "plan.grid.axis_names")
    axis_units = _string_tuple(
        grid_value["axis_units"],
        "plan.grid.axis_units",
        allow_duplicates=True,
    )
    axis_keys = _string_tuple(grid_value["axis_arrays"], "plan.grid.axis_arrays")
    shape = _shape_tuple(grid_value["shape"], "plan.grid.shape")
    dimension = len(axis_names)
    if dimension not in range(1, 5):
        raise PersistenceFormatError("persisted grid dimension must be 1 through 4")
    if len(axis_units) != dimension or len(axis_keys) != dimension:
        raise PersistenceFormatError("persisted grid axis metadata does not align")
    if len(shape) != dimension or any(size < 2 for size in shape):
        raise PersistenceFormatError("persisted grid shape is invalid")
    size = math.prod(shape)
    if _expect_integer(grid_value["size"], "plan.grid.size") != size:
        raise PersistenceFormatError("persisted grid size disagrees with its shape")
    axes = []
    for index, (key, count) in enumerate(zip(axis_keys, shape)):
        if key not in arrays:
            raise PersistenceFormatError("missing persisted grid axis %d" % index)
        axis = arrays[key]
        if axis.shape != (count,) or axis.dtype.str != np.dtype("<f8").str:
            raise PersistenceFormatError("persisted grid axis has invalid storage")
        axes.append(axis)
    canonical_names = _string_tuple(
        grid_value["canonical_target_names"],
        "plan.grid.canonical_target_names",
    )
    canonical_units = _string_tuple(
        grid_value["canonical_target_units"],
        "plan.grid.canonical_target_units",
        allow_duplicates=True,
    )
    dependent_names = _string_tuple(
        grid_value["dependent_tilde_names"],
        "plan.grid.dependent_tilde_names",
    )
    transform_id = grid_value["transform_id"]
    if transform_id is not None:
        transform_id = _expect_string(transform_id, "plan.grid.transform_id")
    constraints = _parse_constraint(document["constraints"])
    try:
        grid = InversionGrid(
            space=_expect_string(grid_value["space"], "plan.grid.space"),
            axis_names=axis_names,
            axis_units=axis_units,
            axes=tuple(axes),
            canonical_target_names=canonical_names,
            canonical_target_units=canonical_units,
            constraints=constraints,
            sampling_envelope_lower=_float_vector(
                grid_value["sampling_envelope_lower"],
                dimension,
                "plan.grid.sampling_envelope_lower",
            ),
            sampling_envelope_upper=_float_vector(
                grid_value["sampling_envelope_upper"],
                dimension,
                "plan.grid.sampling_envelope_upper",
            ),
            coverage=_expect_string(grid_value["coverage"], "plan.grid.coverage"),
            transform_id=transform_id,
            dependent_tilde_names=dependent_names,
            dependent_tilde_relations=_parse_relations(
                grid_value["dependent_tilde_relations"]
            ),
            ordering=_expect_string(grid_value["ordering"], "plan.grid.ordering"),
        )
    except (PipelineError, TypeError, ValueError) as exc:
        raise PersistenceFormatError("invalid persisted inversion grid: %s" % exc) from exc
    configuration = copy.deepcopy(
        _expect_object(document["configuration"], "plan.configuration")
    )
    if run_manifest_version == 8:
        # Version 8 predates the explicit numerical-routing field. It remains
        # readable as completed diagnostic evidence, never executable state.
        configuration["numerical_routing_contract_version"] = (
            PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION
        )
    elif run_manifest_version == 9:
        if (
            configuration.get("numerical_routing_contract_version")
            != _LEGACY_NUMERICAL_ROUTING_CONTRACT_VERSION
        ):
            raise PersistenceVersionError(
                "run-manifest version 9 does not carry numerical-routing "
                "contract 1"
            )
        # Reconstruct the immutable in-memory plan with the current parser.
        # Manifest validation below restores the legacy version in the
        # expected serialized representation before exact comparison.
        configuration["numerical_routing_contract_version"] = (
            PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION
        )
    elif run_manifest_version == 10:
        if (
            configuration.get("numerical_routing_contract_version")
            != _PRE_JACOBIAN_NUMERICAL_ROUTING_CONTRACT_VERSION
        ):
            raise PersistenceVersionError(
                "run-manifest version 10 does not carry numerical-routing "
                "contract 2"
            )
        configuration["numerical_routing_contract_version"] = (
            PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION
        )
    try:
        config = PipelineConfig.from_dict(configuration)
    except (PipelineError, TypeError, ValueError) as exc:
        raise PersistenceFormatError(
            "invalid persisted pipeline configuration: %s" % exc
        ) from exc
    estimates = _expected_storage_estimates(document["storage_estimates"])
    plan_arguments: Dict[str, Any] = {
        "config": config,
        "grid": grid,
        "source_names": source_names,
        "source_units": source_units,
        "target_names": target_names,
        "target_units": target_units,
        "auxiliary_names": auxiliary_names,
        "auxiliary_units": auxiliary_units,
        "residual_scales": _float_vector(
            document["residual_scales"], dimension, "plan.residual_scales"
        ),
        "table_name": _expect_string(table["name"], "plan.table.name"),
        "table_revision": table["revision"],
        "metadata_path": _expect_string(
            table["metadata_path"], "plan.table.metadata_path", allow_empty=True
        ),
        "data_path": _expect_string(
            table["data_path"], "plan.table.data_path", allow_empty=True
        ),
    }
    if table["revision"] is not None:
        plan_arguments["table_revision"] = _expect_string(
            table["revision"], "plan.table.revision"
        )
    plan_arguments.update(estimates)
    try:
        plan = PipelinePlan(**plan_arguments)
    except (PipelineError, TypeError, ValueError) as exc:
        raise PersistenceFormatError("invalid persisted pipeline plan: %s" % exc) from exc
    _validate_plan_consistency(plan)
    return plan


def _validate_plan_consistency(plan: PipelinePlan) -> None:
    grid = plan.grid
    specification = plan.config.grid
    if specification.space is not grid.space:
        raise PersistenceFormatError("plan configuration and resolved grid spaces differ")
    if set(specification.axes) != set(grid.axis_names):
        raise PersistenceFormatError("plan configuration and resolved grid axes differ")
    for name, axis in zip(grid.axis_names, grid.axes):
        axis_spec = specification.axes[name]
        if axis_spec.count != axis.size:
            raise PersistenceFormatError("grid axis count differs from configuration")
        if axis_spec.lower is not None and float(axis[0]) != axis_spec.lower:
            raise PersistenceFormatError("grid lower bound differs from configuration")
        if axis_spec.upper is not None and float(axis[-1]) != axis_spec.upper:
            raise PersistenceFormatError("grid upper bound differs from configuration")
    if grid.size > specification.max_points:
        raise PersistenceFormatError("resolved grid exceeds configured max_points")
    if tuple(plan.target_names) != tuple(grid.canonical_target_names):
        raise PersistenceFormatError("plan target names disagree with the grid")
    if tuple(plan.target_units) != tuple(grid.canonical_target_units):
        raise PersistenceFormatError("plan target units disagree with the grid")
    expected_scales = np.asarray(
        [plan.config.residual_scales[name] for name in plan.target_names],
        dtype=float,
    )
    if not np.array_equal(plan.residual_scales, expected_scales):
        raise PersistenceFormatError("plan residual scales disagree with configuration")


def _validate_document_header(document: Mapping[str, Any]) -> Tuple[int, int]:
    required = (
        "artifact_type",
        "artifact_schema_version",
        "pipeline_result_schema_version",
        "pipeline_run_manifest_version",
        "plan",
        "run_manifest",
        "arrays",
        "integrity",
    )
    _expect_keys(document, required, "manifest")
    if document["artifact_type"] != PIPELINE_ARTIFACT_TYPE:
        raise PersistenceFormatError("artifact type is not a pipeline result")
    artifact_version = document["artifact_schema_version"]
    result_version = document["pipeline_result_schema_version"]
    for name, value in (
        ("artifact_schema_version", artifact_version),
        ("pipeline_result_schema_version", result_version),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise PersistenceFormatError("%s must be an integer" % name)
    if (artifact_version, result_version) not in {
        (2, 3),
        (PIPELINE_ARTIFACT_SCHEMA_VERSION, PIPELINE_RESULT_SCHEMA_VERSION),
    }:
        raise PersistenceVersionError(
            "unsupported artifact/result schema pair (%r, %r)"
            % (artifact_version, result_version)
        )
    run_version = document["pipeline_run_manifest_version"]
    if isinstance(run_version, bool) or not isinstance(run_version, int):
        raise PersistenceFormatError(
            "pipeline_run_manifest_version must be an integer"
        )
    if run_version not in _READABLE_PIPELINE_RUN_MANIFEST_VERSIONS:
        raise PersistenceVersionError(
            "unsupported pipeline_run_manifest_version=%r (readable values are %s)"
            % (
                run_version,
                ", ".join(
                    str(value)
                    for value in sorted(
                        _READABLE_PIPELINE_RUN_MANIFEST_VERSIONS
                    )
                ),
            )
        )
    integrity = _expect_object(document["integrity"], "manifest.integrity")
    _expect_keys(
        integrity, ("algorithm", "schema", "digest"), "manifest.integrity"
    )
    if integrity["algorithm"] != "sha256" or integrity["schema"] != _INTEGRITY_SCHEMA:
        raise PersistenceVersionError("unsupported artifact integrity schema")
    digest = _expect_string(integrity["digest"], "manifest.integrity.digest")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise PersistenceFormatError("artifact integrity digest is not SHA-256 hex")
    without_integrity = dict(document)
    without_integrity.pop("integrity")
    if _artifact_digest(without_integrity) != digest:
        raise PersistenceIntegrityError("artifact descriptor digest does not match")
    return run_version, result_version


def _expected_array_contract(
    plan_value: Mapping[str, Any], result_schema_version: int
) -> Dict[str, Tuple[str, Tuple[int, ...], str]]:
    grid = _expect_object(plan_value.get("grid"), "plan.grid")
    axis_keys = _string_tuple(grid.get("axis_arrays"), "plan.grid.axis_arrays")
    shape = _shape_tuple(grid.get("shape"), "plan.grid.shape")
    axis_names = _string_tuple(grid.get("axis_names"), "plan.grid.axis_names")
    if len(axis_keys) != len(shape) or len(axis_names) != len(shape):
        raise PersistenceFormatError("plan grid axis arrays do not match its shape")
    auxiliary = _expect_object(plan_value.get("auxiliary"), "plan.auxiliary")
    auxiliary_names = _string_tuple(
        auxiliary.get("names"), "plan.auxiliary.names"
    )
    space = grid.get("space")
    if space not in (GridSpace.TARGET.value, GridSpace.TILDE.value):
        raise PersistenceFormatError("plan grid space is invalid")
    dimension = len(shape)
    tilde_columns = 4 if space == GridSpace.TILDE.value else 0
    contract: Dict[str, Tuple[str, Tuple[int, ...], str]] = {}
    for key, count in zip(axis_keys, shape):
        contract[key] = (np.dtype("<f8").str, (count,), "grid_axis")
    result_shapes = {
        "source_coordinates": shape + (dimension,),
        "achieved_targets": shape + (dimension,),
        "residual": shape + (dimension,),
        "scaled_residual": shape + (dimension,),
        "scaled_residual_norm_inf": shape,
        "scaled_distance_l2": shape,
        "auxiliary_values": shape + (len(auxiliary_names),),
        "auxiliary_defined": shape + (len(auxiliary_names),),
        "achieved_tilde_values": shape + (tilde_columns,),
        "achieved_tilde_defined": shape,
        "disposition_codes": shape,
        "route_codes": shape,
        "prescreen_codes": shape,
        "solve_status_codes": shape,
        "fast_status_codes": shape,
        "proof_status_codes": shape,
        "projection_quality_codes": shape,
        "initial_guess_source_codes": shape,
        "phase_codes": shape,
        "phase_route_codes": shape,
        "phase_selection_reason_codes": shape,
        "mixed_candidate": shape,
        "exact_mixed_candidate": shape,
        "mixed_candidate_entry_count": shape,
        "mixed_verified_root_count": shape,
        "phase_boundary": shape,
        "phase_multivalued": shape,
        "line_mixture_fraction": shape,
        "line_mixture_fraction_defined": shape,
        "alternate_single_phase_source": shape + (dimension,),
        "alternate_single_phase_source_defined": shape,
    }
    result_names = (
        _RESULT_ARRAY_NAMES
        if result_schema_version == PIPELINE_RESULT_SCHEMA_VERSION
        else _LEGACY_RESULT_ARRAY_NAMES
    )
    for name in result_names:
        result_shape = result_shapes[name]
        contract["result_" + name] = (
            _result_dtype(name).str,
            result_shape,
            "result",
        )
    if len(contract) != len(axis_keys) + len(result_names):
        raise PersistenceFormatError("persisted array logical names collide")
    return contract


def _parse_registry(
    value: Any,
    contract: Mapping[str, Tuple[str, Tuple[int, ...], str]],
    *,
    max_result_bytes: int,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    registry = _expect_object(value, "manifest.arrays")
    if set(registry) != set(contract):
        missing = sorted(set(contract) - set(registry))
        extra = sorted(set(registry) - set(contract))
        raise PersistenceFormatError(
            "array registry mismatch; missing=%s, unexpected=%s"
            % (missing, extra)
        )
    result_nbytes = 0
    parsed: Dict[str, Dict[str, Any]] = {}
    for key, (dtype_string, shape, kind) in contract.items():
        descriptor = _expect_object(registry[key], "manifest.arrays.%s" % key)
        _expect_keys(
            descriptor,
            ("member", "kind", "dtype", "shape", "nbytes", "sha256"),
            "manifest.arrays.%s" % key,
        )
        member = _expect_string(
            descriptor["member"], "manifest.arrays.%s.member" % key
        )
        if member != "arrays/%s.npy" % key:
            raise PersistenceFormatError("array member name is not canonical")
        if descriptor["kind"] != kind or descriptor["dtype"] != dtype_string:
            raise PersistenceFormatError("array kind or dtype does not match schema")
        declared_shape = _shape_tuple(
            descriptor["shape"], "manifest.arrays.%s.shape" % key
        )
        if declared_shape != shape:
            raise PersistenceFormatError("array shape does not match schema")
        dtype = np.dtype(dtype_string)
        nbytes = _expect_integer(
            descriptor["nbytes"], "manifest.arrays.%s.nbytes" % key
        )
        if nbytes != _checked_nbytes(shape, dtype, "manifest.arrays.%s" % key):
            raise PersistenceFormatError("array byte count does not match schema")
        digest = _expect_string(
            descriptor["sha256"], "manifest.arrays.%s.sha256" % key
        )
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise PersistenceFormatError("array digest is not SHA-256 hex")
        if kind == "result":
            result_nbytes += nbytes
        parsed[key] = {
            "member": member,
            "dtype": dtype,
            "shape": shape,
            "nbytes": nbytes,
            "sha256": digest,
        }
    if result_nbytes > max_result_bytes:
        raise PersistenceFormatError(
            "persisted result uses %d bytes, exceeding max_result_bytes=%d"
            % (result_nbytes, max_result_bytes)
        )
    return parsed, result_nbytes


def _validate_manifest_against_plan(
    run_manifest: Any,
    plan: PipelinePlan,
    run_manifest_version: int,
    result_schema_version: int,
) -> Dict[str, Any]:
    manifest = _expect_object(run_manifest, "run_manifest")
    expected = _plain_json(plan.manifest(), "expected_manifest")
    if run_manifest_version == 8:
        expected.pop("numerical_routing_contract", None)
        configuration = expected.get("configuration")
        stored_configuration = manifest.get("configuration")
        if isinstance(configuration, dict):
            configuration.pop("numerical_routing_contract_version", None)
            if (
                isinstance(stored_configuration, dict)
                and "use_reachability_seed" not in stored_configuration
            ):
                configuration.pop("use_reachability_seed", None)
    elif run_manifest_version == 9:
        routing = expected.get("numerical_routing_contract")
        if isinstance(routing, dict):
            routing["version"] = _LEGACY_NUMERICAL_ROUTING_CONTRACT_VERSION
        configuration = expected.get("configuration")
        if isinstance(configuration, dict):
            configuration["numerical_routing_contract_version"] = (
                _LEGACY_NUMERICAL_ROUTING_CONTRACT_VERSION
            )
    elif run_manifest_version == 10:
        routing = expected.get("numerical_routing_contract")
        if isinstance(routing, dict):
            routing["version"] = (
                _PRE_JACOBIAN_NUMERICAL_ROUTING_CONTRACT_VERSION
            )
        configuration = expected.get("configuration")
        if isinstance(configuration, dict):
            configuration["numerical_routing_contract_version"] = (
                _PRE_JACOBIAN_NUMERICAL_ROUTING_CONTRACT_VERSION
            )
    if result_schema_version == 3:
        expected["pipeline_result_schema_version"] = result_schema_version
        codebooks = expected.get("status_codebooks")
        if isinstance(codebooks, dict):
            codebooks.pop("phase_selection_reason", None)
    # SolverConfig gained then-disabled-by-default subcell options while the
    # current run-manifest version was already in use. PipelineConfig.from_dict
    # reconstructs a v9 artifact with all three fields absent as that exact
    # historical route, irrespective of the newer enabled default. Normalize
    # only that complete legacy absence: partially present or non-historical
    # contracts remain strict.
    configuration = expected.get("configuration")
    stored_configuration = manifest.get("configuration")
    if (
        run_manifest_version in (8, 9)
        and isinstance(configuration, dict)
        and isinstance(stored_configuration, dict)
    ):
        solver = configuration.get("solver")
        stored_solver = stored_configuration.get("solver")
        subcell_defaults = {
            "enable_subcell_fallback": False,
            "max_subcell_depth": 7,
            "max_subcell_active_boxes": 1_000_000,
        }
        if (
            isinstance(solver, dict)
            and isinstance(stored_solver, dict)
            and not any(name in stored_solver for name in subcell_defaults)
            and all(solver.get(name) == value for name, value in subcell_defaults.items())
        ):
            for name in subcell_defaults:
                solver.pop(name, None)
        execution = configuration.get("execution")
        stored_execution = stored_configuration.get("execution")
        if (
            isinstance(execution, dict)
            and isinstance(stored_execution, dict)
            and "worker_task_size" not in stored_execution
            and execution.get("worker_task_size") == 256
        ):
            # Phase-7 separated process task size from atomic checkpoint chunk
            # size without changing the v9 artifact schema. Complete absence
            # denotes the historical fixed 256-point internal task size.
            execution.pop("worker_task_size", None)
    if set(manifest) != set(expected) | {"run"}:
        raise PersistenceFormatError(
            "run manifest does not contain exactly the plan manifest plus run data"
        )
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise PersistenceFormatError(
                "run manifest field '%s' disagrees with the persisted plan" % key
            )
    run = _expect_object(manifest.get("run"), "run_manifest.run")
    if run.get("completed") is not True:
        raise PersistenceFormatError("persisted run is not complete")
    if run.get("pipeline_run_manifest_version") != run_manifest_version:
        raise PersistenceVersionError(
            "persisted run-manifest version disagrees with the artifact header"
        )
    return manifest


def _validate_result_manifest(result: PipelineResult) -> None:
    run = result.manifest["run"]
    status_counts = run.get("status_counts")
    if not isinstance(status_counts, Mapping) or dict(status_counts) != dict(
        result.status_counts
    ):
        raise PersistenceIntegrityError(
            "run status counts disagree with persisted result columns"
        )
    for name in ("errors", "projection_failures"):
        section = run.get(name)
        if not isinstance(section, Mapping):
            raise PersistenceFormatError("run.%s must be an object" % name)
        count = section.get("count")
        records = section.get("records")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or not isinstance(records, tuple)
            or count != len(records)
        ):
            raise PersistenceIntegrityError(
                "run.%s count and records disagree" % name
            )
        for record in records:
            if not isinstance(record, Mapping):
                raise PersistenceFormatError("run.%s records must be objects" % name)
            index = record.get("flat_index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= result.grid.size
            ):
                raise PersistenceIntegrityError(
                    "run.%s contains an out-of-range flat index" % name
                )


def load_pipeline_result(
    path: Any,
    *,
    max_archive_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_uncompressed_bytes: int = DEFAULT_MAX_ARTIFACT_UNCOMPRESSED_BYTES,
    max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
) -> PipelineResult:
    """Load and exhaustively validate a completed-result artifact.

    Recorded EoS metadata/data paths are provenance only.  This function is
    self-contained and does not access either source file.
    """

    archive_limit = _positive_limit(max_archive_bytes, "max_archive_bytes")
    uncompressed_limit = _positive_limit(
        max_uncompressed_bytes, "max_uncompressed_bytes"
    )
    manifest_limit = _positive_limit(max_manifest_bytes, "max_manifest_bytes")
    result_limit = _positive_limit(max_result_bytes, "max_result_bytes")
    source = _lexical_absolute_path(path, "input")
    try:
        stat = source.stat()
    except OSError as exc:
        raise PersistencePathError("result artifact is unavailable: %s" % exc) from exc
    if not source.is_file():
        raise PersistencePathError("result artifact is not a regular file: %s" % source)
    if stat.st_size > archive_limit:
        raise PersistenceFormatError(
            "artifact uses %d bytes, exceeding max_archive_bytes=%d"
            % (stat.st_size, archive_limit)
        )

    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            if archive.comment:
                raise PersistenceFormatError("artifact ZIP comments are forbidden")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise PersistenceFormatError("artifact contains duplicate ZIP members")
            total_uncompressed = 0
            total_compressed = 0
            by_name: Dict[str, zipfile.ZipInfo] = {}
            for info in infos:
                if (
                    not _safe_member_name(info.filename)
                    or info.is_dir()
                    or info.flag_bits & 0x1
                ):
                    raise PersistenceFormatError(
                        "artifact contains an unsafe ZIP member"
                    )
                if info.compress_type not in (
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                ):
                    raise PersistenceFormatError(
                        "artifact uses an unsupported ZIP compression method"
                    )
                total_uncompressed += int(info.file_size)
                total_compressed += int(info.compress_size)
                if total_uncompressed > uncompressed_limit:
                    raise PersistenceFormatError(
                        "artifact exceeds max_uncompressed_bytes=%d"
                        % uncompressed_limit
                    )
                if total_compressed > archive_limit:
                    raise PersistenceFormatError(
                        "artifact exceeds max_archive_bytes=%d" % archive_limit
                    )
                by_name[info.filename] = info
            if _MANIFEST_MEMBER not in by_name:
                raise PersistenceFormatError("artifact has no manifest.json")
            manifest_info = by_name[_MANIFEST_MEMBER]
            if manifest_info.file_size > manifest_limit:
                raise PersistenceFormatError(
                    "artifact manifest exceeds max_manifest_bytes=%d"
                    % manifest_limit
                )
            document = _read_json_manifest(archive, manifest_info)
            run_manifest_version, result_schema_version = (
                _validate_document_header(document)
            )
            plan_value = _expect_object(document["plan"], "plan")
            contract = _expected_array_contract(
                plan_value, result_schema_version
            )
            registry, result_nbytes = _parse_registry(
                document["arrays"], contract, max_result_bytes=result_limit
            )
            expected_members = {_MANIFEST_MEMBER} | {
                descriptor["member"] for descriptor in registry.values()
            }
            if set(by_name) != expected_members:
                raise PersistenceFormatError(
                    "artifact ZIP members do not exactly match the array registry"
                )
            arrays: Dict[str, np.ndarray] = {}
            for key in sorted(registry):
                descriptor = registry[key]
                arrays[key] = _read_npy_member(
                    archive,
                    by_name[descriptor["member"]],
                    expected_dtype=descriptor["dtype"],
                    expected_shape=descriptor["shape"],
                    expected_nbytes=descriptor["nbytes"],
                    expected_sha256=descriptor["sha256"],
                )
    except PersistenceError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PersistenceFormatError("invalid result artifact: %s" % exc) from exc

    plan = _reconstruct_plan(
        document["plan"], arrays, run_manifest_version
    )
    if int(plan.estimated_result_bytes) != result_nbytes:
        raise PersistenceIntegrityError(
            "persisted array bytes disagree with the pipeline plan"
        )
    manifest = _validate_manifest_against_plan(
        document["run_manifest"],
        plan,
        run_manifest_version,
        result_schema_version,
    )
    stored_result_names = (
        _RESULT_ARRAY_NAMES
        if result_schema_version == PIPELINE_RESULT_SCHEMA_VERSION
        else _LEGACY_RESULT_ARRAY_NAMES
    )
    arguments = {
        name: arrays["result_" + name] for name in stored_result_names
    }
    if result_schema_version == 3:
        from .phase_solver import PhaseRootSelectionReason
        from .pipeline import (
            _PHASE_CODES,
            _PHASE_SELECTION_REASON_CODES,
            _enum_code,
        )
        from .transition import PhaseLabel

        phase_codes = np.asarray(arguments["phase_codes"])
        reasons = np.full(
            phase_codes.shape,
            _enum_code(
                PhaseRootSelectionReason.NOT_APPLICABLE,
                _PHASE_SELECTION_REASON_CODES,
            ),
            dtype=np.uint8,
        )
        first_order_solved = np.isin(
            phase_codes,
            np.asarray(
                [
                    _enum_code(PhaseLabel.SINGLE_PHASE, _PHASE_CODES),
                    _enum_code(PhaseLabel.MIXED_PHASE, _PHASE_CODES),
                ],
                dtype=np.uint8,
            ),
        )
        reasons[first_order_solved] = _enum_code(
            PhaseRootSelectionReason.UNAMBIGUOUS,
            _PHASE_SELECTION_REASON_CODES,
        )
        arguments["phase_selection_reason_codes"] = reasons
    try:
        result = PipelineResult(plan=plan, manifest=manifest, **arguments)
    except (PipelineError, TypeError, ValueError) as exc:
        raise PersistenceIntegrityError(
            "persisted result violates the pipeline contract: %s" % exc
        ) from exc
    _validate_result_manifest(result)
    return result


__all__ = [
    "DEFAULT_MAX_ARTIFACT_BYTES",
    "DEFAULT_MAX_ARTIFACT_UNCOMPRESSED_BYTES",
    "DEFAULT_MAX_MANIFEST_BYTES",
    "PIPELINE_ARTIFACT_SCHEMA_VERSION",
    "PIPELINE_ARTIFACT_TYPE",
    "PersistenceError",
    "PersistenceFormatError",
    "PersistenceIntegrityError",
    "PersistencePathError",
    "PersistenceVersionError",
    "load_pipeline_result",
    "save_pipeline_result",
]
