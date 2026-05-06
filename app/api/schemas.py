"""Request/response Pydantic schemas for the API (T093)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class ProblemResponse(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    run_id: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    db_backend: str
    version: str


class SeedResponse(BaseModel):
    master_seed: int
    worker_seeds: list[int]


class IngestResponse(BaseModel):
    source: str
    ok: bool
    summary: dict[str, Any] = {}


class BenchmarkRunRequest(BaseModel):
    distances: list[int]
    rounds: list[int]
    bases: list[str] = ["X"]
    p_errors: list[float] = [1e-3]
    backends: list[str] = ["pymatching_baseline"]
    num_shots: int = 1024
    master_seed: int = 42


class RunSummary(BaseModel):
    run_id: str
    status: str
    backend: str
    started_at: str | None = None
    finished_at: str | None = None


class MetricsSummary(BaseModel):
    run_id: str
    ler: float
    ci_low: float
    ci_high: float


class ArtefactSummary(BaseModel):
    id: int
    run_id: str
    path: str
    sha256: str
    type: str


class ExportOnnxRequest(BaseModel):
    run_id: str
    output_path: str


class ReportRequest(BaseModel):
    run_id: str
    include_pdf: bool = False


class EvidenceSummary(BaseModel):
    run_id: str
    manifests: list[str] = []
    reports: list[str] = []
    artefacts: list[str] = []
