"""Engine + session factory with DuckDB/PostgreSQL dispatch (T061)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import get_settings
from app.db import duckdb_backend, postgres_backend

__all__ = ["get_engine", "get_sessionmaker", "session_scope"]


_ENGINE_CACHE: dict[str, Engine] = {}
_SESSION_CACHE: dict[str, sessionmaker[Session]] = {}


def _resolve_url(url: str | None) -> str:
    return url or get_settings().database_url


def get_engine(url: str | None = None) -> Engine:
    resolved = _resolve_url(url)
    if resolved in _ENGINE_CACHE:
        return _ENGINE_CACHE[resolved]
    parsed = make_url(resolved)
    if duckdb_backend.is_duckdb_url(parsed):
        engine = duckdb_backend.build_engine(parsed)
    elif postgres_backend.is_postgres_url(parsed):
        engine = postgres_backend.build_engine(parsed)
    elif parsed.drivername.startswith("sqlite"):
        # Useful for tests; a session fallback without duckdb/postgres.
        from sqlalchemy import create_engine

        engine = create_engine(parsed)
    else:
        raise ValueError(
            f"unsupported DB URL scheme {parsed.drivername!r}; "
            "expected duckdb, postgresql, or sqlite"
        )
    _ENGINE_CACHE[resolved] = engine
    return engine


def get_sessionmaker(url: str | None = None) -> sessionmaker[Session]:
    resolved = _resolve_url(url)
    if resolved in _SESSION_CACHE:
        return _SESSION_CACHE[resolved]
    sm = sessionmaker(bind=get_engine(resolved), expire_on_commit=False)
    _SESSION_CACHE[resolved] = sm
    return sm


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    sm = get_sessionmaker(url)
    session = sm()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise RuntimeError(f"session_scope rolled back: {e}") from e
    finally:
        session.close()
