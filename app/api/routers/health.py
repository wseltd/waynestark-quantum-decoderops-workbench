"""GET /health — liveness + backend info (T094)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.api.deps import get_settings_dep
from app.api.schemas import HealthResponse
from app.config.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health(settings: Settings = Depends(get_settings_dep)) -> HealthResponse:
    backend = "duckdb" if settings.database_url.startswith("duckdb") else (
        "postgresql"
        if settings.database_url.startswith("postgresql")
        else "sqlite"
    )
    return HealthResponse(status="ok", db_backend=backend, version=__version__)
