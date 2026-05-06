"""FastAPI dependency providers (T092)."""

from __future__ import annotations

from typing import Iterator

from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.db.session import get_sessionmaker

__all__ = ["get_db_session", "get_settings_dep"]


def get_settings_dep() -> Settings:
    return get_settings()


def get_db_session() -> Iterator[Session]:
    sm = get_sessionmaker()
    session = sm()
    try:
        yield session
    finally:
        session.close()
