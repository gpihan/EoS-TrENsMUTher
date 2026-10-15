"""Read-only registry of the persisted and routing contracts frozen for v1."""

from __future__ import annotations

from types import MappingProxyType

from .checkpoint import PIPELINE_CHECKPOINT_SCHEMA_VERSION
from .executors import ARRAY_PLAN_SCHEMA_VERSION, SHARD_SCHEMA_VERSION
from .exporters import (
    PHASE_EXPORT_SCHEMA_VERSION,
    TEXT_EXPORT_SCHEMA_VERSION,
)
from .outputs import PIPELINE_ARTIFACT_SCHEMA_VERSION
from .pipeline import (
    PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
    PIPELINE_RESULT_SCHEMA_VERSION,
    PIPELINE_RUN_MANIFEST_VERSION,
)
from .reachability import REACHABILITY_ROUTING_CONTRACT_VERSION
from .shared_state import SHARED_WORKER_STATE_SCHEMA_VERSION
from .tables import METADATA_SCHEMA_VERSION, TABLE_SNAPSHOT_FINGERPRINT_VERSION


V1_CONTRACT_VERSIONS = MappingProxyType(
    {
        "public_api": 7,
        "metadata_schema": METADATA_SCHEMA_VERSION,
        "table_snapshot_fingerprint": TABLE_SNAPSHOT_FINGERPRINT_VERSION,
        "pipeline_result_schema": PIPELINE_RESULT_SCHEMA_VERSION,
        "pipeline_run_manifest": PIPELINE_RUN_MANIFEST_VERSION,
        "numerical_routing": PIPELINE_NUMERICAL_ROUTING_CONTRACT_VERSION,
        "checkpoint_schema": PIPELINE_CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema": PIPELINE_ARTIFACT_SCHEMA_VERSION,
        "reachability_routing": REACHABILITY_ROUTING_CONTRACT_VERSION,
        "shared_worker_state_schema": SHARED_WORKER_STATE_SCHEMA_VERSION,
        "array_plan_schema": ARRAY_PLAN_SCHEMA_VERSION,
        "array_shard_schema": SHARD_SCHEMA_VERSION,
        "text_export_schema": TEXT_EXPORT_SCHEMA_VERSION,
        "phase_export_schema": PHASE_EXPORT_SCHEMA_VERSION,
    }
)


__all__ = ["V1_CONTRACT_VERSIONS"]
