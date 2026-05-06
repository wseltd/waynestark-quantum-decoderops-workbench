"""FastAPI application factory (T091)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routers import (
    artifacts,
    benchmark,
    evidence,
    export_onnx,
    health,
    ingest_dem,
    ingest_syndrome,
    metrics,
    profiles,
    reports,
    runs,
    seed,
)
from app.config.settings import Settings, get_settings
from app.core.errors import DecoderOpsError

__all__ = ["create_app"]

_LOG = logging.getLogger(__name__)


def _problem(
    status: int, title: str, detail: str | None = None, run_id: str | None = None
) -> dict[str, Any]:
    return {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "run_id": run_id,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="DecoderOps Workbench",
        version=__version__,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health first so probes keep working even if downstream errors.
    app.include_router(health.router)
    app.include_router(seed.router)
    app.include_router(ingest_dem.router)
    app.include_router(ingest_syndrome.router)
    app.include_router(benchmark.router)
    app.include_router(runs.router)
    app.include_router(metrics.router)
    app.include_router(artifacts.router)
    app.include_router(export_onnx.router)
    app.include_router(reports.router)
    app.include_router(evidence.router)
    app.include_router(profiles.router)

    @app.exception_handler(DecoderOpsError)
    async def _decoderops_handler(
        _req: Request, exc: DecoderOpsError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_problem(400, exc.reason_code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        _req: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_problem(422, "validation_error", str(exc.errors())),
        )

    @app.exception_handler(Exception)
    async def _catch_all(_req: Request, exc: Exception) -> JSONResponse:
        _LOG.exception("unhandled server error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=_problem(500, "internal_server_error", "see server logs"),
        )

    @app.on_event("startup")
    async def _on_startup() -> None:
        kind = (
            "duckdb"
            if settings.database_url.startswith("duckdb")
            else "postgresql"
            if settings.database_url.startswith("postgresql")
            else "sqlite/other"
        )
        _LOG.info("decoderops starting: db_backend=%s", kind)
        # Bootstrap schema on DuckDB / SQLite so the service is usable
        # from a clean checkout. Postgres uses Alembic migrations.
        if kind != "postgresql":
            try:
                from app.db.schema_init import bootstrap_schema
                from app.db.session import get_engine

                bootstrap_schema(get_engine())
                _LOG.info("schema bootstrapped")
            except Exception as e:  # noqa: BLE001
                _LOG.warning("schema bootstrap skipped: %s", e)

    return app
