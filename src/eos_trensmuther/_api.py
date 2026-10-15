"""Frozen v1 top-level API declaration.

The package root retains additional historical exports for v1 compatibility.
Only the names declared here are the supported public surface for new code.
"""

from __future__ import annotations

from types import MappingProxyType


PUBLIC_API_VERSION = 7

_GROUPS = {
    "contract": (
        "__version__",
        "PUBLIC_API_VERSION",
        "PUBLIC_API_GROUPS",
        "PUBLIC_API_NAMES",
        "V1_CONTRACT_VERSIONS",
    ),
    "tables": (
        "ForwardTable",
        "BoundaryEnvelope",
        "ValidationIssue",
        "ValidationReport",
        "EoSError",
        "EoSValidationError",
    ),
    "configuration": (
        "ApplicationConfig",
        "ApplicationConfigError",
        "GridAxisSpec",
        "InversionGridSpec",
        "PipelineConfig",
        "ExecutionConfig",
        "ProjectionConfig",
        "SolverConfig",
        "ReachabilityConfig",
        "CheckpointConfig",
        "CoarseSeedConfig",
        "ErrorPolicy",
        "ProcessStartMethod",
        "PARAMETER_GROUPS",
        "ORDINARY_SUBCELL_CONFIG_FIELDS",
        "load_application_config",
        "load_parameter_file",
    ),
    "pipeline": (
        "EoSInversionPipeline",
        "PipelineResult",
        "PipelineError",
        "PipelineConfigurationError",
    ),
    "status": (
        "PointDisposition",
        "PointRoute",
        "PrescreenStatus",
        "SolveStatus",
        "ProofStatus",
        "ProjectionQuality",
        "PhaseLabel",
        "PhaseRoute",
        "PhaseRootSelectionReason",
        "InitialGuessSource",
        "pipeline_status_code",
    ),
    "persistence": (
        "save_pipeline_result",
        "load_pipeline_result",
        "PersistenceError",
    ),
    "export": (
        "TextExportConfig",
        "TextExportResult",
        "ExportError",
        "export_text",
    ),
    "advanced_inversion": (
        "MultilinearInterpolator",
        "EoSRootSolver",
        "InversionResult",
    ),
    "slurm": (
        "SlurmArrayConfig",
        "ExecutorError",
        "create_array_plan",
        "save_array_plan",
        "load_array_plan",
        "render_slurm_script",
        "execute_array_shard",
        "assemble_array_shards",
    ),
}

PUBLIC_API_GROUPS = MappingProxyType(
    {name: tuple(values) for name, values in _GROUPS.items()}
)
PUBLIC_API_NAMES = tuple(
    name for values in PUBLIC_API_GROUPS.values() for name in values
)

if len(PUBLIC_API_NAMES) != len(set(PUBLIC_API_NAMES)):
    raise RuntimeError("the frozen public API contains duplicate names")


__all__ = ["PUBLIC_API_GROUPS", "PUBLIC_API_NAMES", "PUBLIC_API_VERSION"]
