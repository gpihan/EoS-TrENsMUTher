"""Structured, threshold-free performance reports for inversion runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from .pipeline import PipelineResult, PipelineStateError


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PipelineStateError("%s must be a non-negative number" % name)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PipelineStateError(
            "%s must be a non-negative number" % name
        ) from exc
    if not math.isfinite(converted) or converted < 0.0:
        raise PipelineStateError("%s must be a non-negative number" % name)
    return converted


@dataclass(frozen=True)
class PipelinePerformanceSummary:
    """Portable performance and routing summary extracted from one result.

    The object deliberately reports observations rather than asserting timing
    thresholds.  Cross-machine comparisons should retain the software and
    execution fields emitted by :meth:`to_dict`.
    """

    table_name: str
    table_revision: Any
    grid_space: str
    grid_shape: tuple
    grid_points: int
    execution: Mapping[str, Any]
    software: Mapping[str, Any]
    timing_seconds: Mapping[str, Any]
    stage_counters: Mapping[str, Any]
    status_counts: Mapping[str, Any]
    performance: Mapping[str, Any]

    @classmethod
    def from_result(cls, result: PipelineResult) -> "PipelinePerformanceSummary":
        if not isinstance(result, PipelineResult):
            raise TypeError("result must be a PipelineResult")
        run = result.manifest.get("run")
        if not isinstance(run, Mapping) or run.get("completed") is not True:
            raise PipelineStateError("performance requires a completed result")
        required = (
            "execution",
            "software",
            "timing_seconds",
            "stage_counters",
            "status_counts",
            "performance",
        )
        if any(not isinstance(run.get(name), Mapping) for name in required):
            raise PipelineStateError("run manifest lacks Phase-6 performance data")
        performance = _plain(run["performance"])
        throughput = _finite_nonnegative(
            performance.get("computed_points_per_second_this_invocation"),
            "computed throughput",
        )
        computed = performance.get("computed_points_this_invocation")
        restored = performance.get("restored_points")
        if (
            isinstance(computed, bool)
            or not isinstance(computed, int)
            or computed < 0
            or isinstance(restored, bool)
            or not isinstance(restored, int)
            or restored < 0
            or computed + restored != result.grid.size
        ):
            raise PipelineStateError(
                "performance point accounting does not cover the grid"
            )
        rates = performance.get("rates")
        if not isinstance(rates, dict):
            raise PipelineStateError("performance rates are missing")
        for name, value in rates.items():
            rate = _finite_nonnegative(value, "performance rate %s" % name)
            if rate > 1.0:
                raise PipelineStateError("performance rates cannot exceed one")
        performance["computed_points_per_second_this_invocation"] = throughput
        return cls(
            table_name=result.plan.table_name,
            table_revision=result.plan.table_revision,
            grid_space=result.grid.space.value,
            grid_shape=tuple(result.grid.shape),
            grid_points=result.grid.size,
            execution=_plain(run["execution"]),
            software=_plain(run["software"]),
            timing_seconds=_plain(run["timing_seconds"]),
            stage_counters=_plain(run["stage_counters"]),
            status_counts=_plain(run["status_counts"]),
            performance=performance,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": {
                "name": self.table_name,
                "revision": self.table_revision,
            },
            "grid": {
                "space": self.grid_space,
                "shape": list(self.grid_shape),
                "points": self.grid_points,
            },
            "execution": _plain(self.execution),
            "software": _plain(self.software),
            "timing_seconds": _plain(self.timing_seconds),
            "stage_counters": _plain(self.stage_counters),
            "status_counts": _plain(self.status_counts),
            "performance": _plain(self.performance),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )


def summarize_pipeline_performance(
    result: PipelineResult,
) -> PipelinePerformanceSummary:
    """Return a validated structured performance summary for ``result``."""

    return PipelinePerformanceSummary.from_result(result)


__all__ = [
    "PipelinePerformanceSummary",
    "summarize_pipeline_performance",
]
