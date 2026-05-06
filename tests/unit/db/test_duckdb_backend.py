"""Tests for app.db.duckdb_backend (T062)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.engine.url import make_url
from sqlalchemy.pool import NullPool

from app.db.duckdb_backend import build_engine, is_duckdb_url


def _has_duckdb_driver() -> bool:
    try:
        import duckdb_engine  # noqa: F401
        return True
    except ImportError:
        return False


def test_build_engine_with_memory_url_returns_engine() -> None:
    if not _has_duckdb_driver():
        pytest.skip("duckdb_engine dialect not installed")
    u = make_url("duckdb:///:memory:")
    e = build_engine(u)
    conn = e.connect()
    try:
        assert conn is not None
    finally:
        conn.close()


def test_build_engine_with_file_url_creates_parent_directory(tmp_path: Path) -> None:
    if not _has_duckdb_driver():
        # Directory creation works regardless of dialect load; we can still
        # assert it happens by invoking build_engine — but build_engine will
        # attempt create_engine which loads the dialect. Skip cleanly.
        pytest.skip("duckdb_engine dialect not installed")
    db = tmp_path / "deep" / "nested" / "d.duckdb"
    u = make_url(f"duckdb:///{db}")
    build_engine(u)
    assert db.parent.exists()


def test_build_engine_uses_nullpool() -> None:
    if not _has_duckdb_driver():
        pytest.skip("duckdb_engine dialect not installed")
    u = make_url("duckdb:///:memory:")
    e = build_engine(u)
    assert isinstance(e.pool, NullPool)


def test_is_duckdb_url_true_for_duckdb_scheme() -> None:
    assert is_duckdb_url(make_url("duckdb:///:memory:"))


def test_is_duckdb_url_false_for_postgres_scheme() -> None:
    assert not is_duckdb_url(make_url("postgresql://u:p@h/d"))
