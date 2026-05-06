"""Aggregate metrics container (T047)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

from app.metrics.compatibility import RuntimeCompatibilityStatus
from app.metrics.export_status import ExportStatus
from app.metrics.latency import LatencyHistogram
from app.metrics.logical_error_rate import LogicalErrorRate
from app.metrics.residual_density import ResidualSyndromeDensity
from app.metrics.throughput import Throughput

__all__ = ["RunMetrics", "build_run_metrics"]


class RunMetrics(BaseModel):
    schema_version: Literal["1"] = "1"
    run_id: str
    logical_error_rate: LogicalErrorRate
    residual_syndrome_density: ResidualSyndromeDensity
    latency: LatencyHistogram
    throughput: Throughput
    exports: list[ExportStatus]
    compatibility: list[RuntimeCompatibilityStatus]

    @field_validator("run_id")
    @classmethod
    def _non_empty_run_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("run_id must be a non-empty string")
        return v

    @field_validator("compatibility")
    @classmethod
    def _at_least_one_compat(
        cls, v: list[RuntimeCompatibilityStatus]
    ) -> list[RuntimeCompatibilityStatus]:
        if len(v) < 1:
            raise ValueError("compatibility must have >= 1 entry")
        return v


def build_run_metrics(
    run_id: str,
    logical_error_rate: LogicalErrorRate,
    residual_syndrome_density: ResidualSyndromeDensity,
    latency: LatencyHistogram,
    throughput: Throughput,
    exports: list[ExportStatus],
    compatibility: list[RuntimeCompatibilityStatus],
) -> RunMetrics:
    return RunMetrics(
        schema_version="1",
        run_id=run_id,
        logical_error_rate=logical_error_rate,
        residual_syndrome_density=residual_syndrome_density,
        latency=latency,
        throughput=throughput,
        exports=exports,
        compatibility=compatibility,
    )
