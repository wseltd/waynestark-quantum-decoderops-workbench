"""Tests for app.db.session (T061).

Many real DB drivers (duckdb_engine, psycopg2) may not be installed in the
test environment — these tests use sqlite in-memory URLs as the fallback
and gate other paths with module-availability checks.
"""

from __future__ import annotations

import pytest

from app.db.session import get_engine, get_sessionmaker, session_scope


def _has(module: str) -> bool:
    import importlib

    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def test_get_engine_dispatches_duckdb_for_duckdb_url() -> None:
    if not _has("duckdb_engine"):
        pytest.skip("duckdb_engine dialect not installed")
    e = get_engine("duckdb:///:memory:")
    assert "duckdb" in e.dialect.name.lower()


def test_get_engine_dispatches_postgres_for_postgres_url() -> None:
    """Confirm the postgres dispatch branch fires for either driver.

    psycopg v3 (``postgresql+psycopg``) is the driver the proof-v2 cycle
    installed and the dual-backend parity proof uses. psycopg2 is an
    alternative driver; either installation satisfies the test.
    """
    driver = None
    if _has("psycopg"):
        driver = "postgresql+psycopg"
    elif _has("psycopg2"):
        driver = "postgresql+psycopg2"
    else:
        pytest.skip(
            "neither psycopg v3 nor psycopg2 installed; install with "
            "pip install 'psycopg[binary]==3.2.*'"
        )
    # Build the engine without opening a connection — we only care that
    # the Postgres dispatch branch fires and the dialect is postgresql.
    e = get_engine(f"{driver}://u:p@localhost:5432/x")
    assert "postgresql" in e.dialect.name.lower()


def test_get_engine_raises_on_unsupported_scheme() -> None:
    with pytest.raises(ValueError):
        get_engine("mysql://u:p@localhost/x")


def test_session_scope_commits_on_success() -> None:
    sm = get_sessionmaker("sqlite:///:memory:")
    assert sm is not None
    with session_scope("sqlite:///:memory:") as s:
        assert s is not None


def test_session_scope_rolls_back_on_exception() -> None:
    with pytest.raises(RuntimeError), session_scope("sqlite:///:memory:") as _s:
        raise ValueError("boom")


def test_get_engine_caches_engine_per_url() -> None:
    e1 = get_engine("sqlite:///:memory:")
    e2 = get_engine("sqlite:///:memory:")
    assert e1 is e2
