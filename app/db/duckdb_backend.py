"""DuckDB-specific engine construction (T062)."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL
from sqlalchemy.pool import NullPool

__all__ = ["build_engine", "is_duckdb_url"]


def is_duckdb_url(url: URL) -> bool:
    scheme = (url.drivername or "").lower()
    return scheme == "duckdb" or scheme.startswith("duckdb+")


def build_engine(url: URL) -> Engine:
    db = (url.database or "").strip()
    if db and db != ":memory:":
        Path(db).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        url,
        poolclass=NullPool,
        connect_args={"read_only": False},
    )
