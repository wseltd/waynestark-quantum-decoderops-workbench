"""Shared pytest fixtures (T191)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.capability_report import CapabilityReport
from app.db.base import Base


@pytest.fixture(scope="session")
def tmp_artefact_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("artefacts")


@pytest.fixture()
def seeded_settings(
    tmp_artefact_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    from app.config.settings import reset_settings_cache, get_settings

    monkeypatch.setenv("DECODEROPS_DB_URL", "sqlite:///:memory:")
    monkeypatch.setenv(
        "DECODEROPS_ARTEFACT_DIR", str(tmp_artefact_dir)
    )
    reset_settings_cache()
    return get_settings()


@pytest.fixture()
def duckdb_session() -> Iterator[Session]:
    # DuckDB dialect may not be installed in all test envs; use sqlite
    # in-memory as the cross-env fallback that still exercises ORM wiring.
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = Session(engine)
    try:
        yield s
        s.rollback()
    finally:
        s.close()


@pytest.fixture()
def capability_stub() -> Callable[..., CapabilityReport]:
    def _factory(
        *,
        available: bool = True,
        reason: str = "stub ready",
        required: list[str] | None = None,
        detected_versions: dict[str, str] | None = None,
    ) -> CapabilityReport:
        req = required or ["stub"]
        det = detected_versions or {"stub": "0"}
        if available:
            return CapabilityReport.ready(
                reason=reason, required=req, detected_versions=det
            )
        return CapabilityReport.unavailable(
            reason=reason,
            required=req,
            category="not_installed",
        )

    return _factory


@pytest.fixture()
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> datetime:
    frozen = datetime(2026, 4, 21, 0, 0, 0, tzinfo=timezone.utc)

    try:
        import app.core.fingerprint as fp_mod

        monkeypatch.setattr(
            fp_mod, "_now_utc_iso", lambda: "2026-04-21T00:00:00Z", raising=False
        )
    except Exception:
        pass
    return frozen
