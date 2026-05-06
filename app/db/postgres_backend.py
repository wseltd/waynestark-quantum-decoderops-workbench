"""PostgreSQL-specific engine construction (T063)."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL
from sqlalchemy.pool import QueuePool

__all__ = ["build_engine", "is_postgres_url"]


def is_postgres_url(url: URL) -> bool:
    scheme = (url.drivername or "").lower()
    return scheme == "postgresql" or scheme.startswith("postgresql+")


def build_engine(url: URL) -> Engine:
    return create_engine(
        url,
        poolclass=QueuePool,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
