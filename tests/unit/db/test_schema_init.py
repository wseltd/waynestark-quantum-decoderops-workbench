"""Tests for app.db.schema_init (T072)."""

from __future__ import annotations

from sqlalchemy import create_engine, inspect

from app.db.schema_init import bootstrap_schema


def test_bootstrap_schema_creates_all_tables() -> None:
    e = create_engine("sqlite:///:memory:")
    bootstrap_schema(e)
    tables = set(inspect(e).get_table_names())
    assert {"runs", "metrics", "artefacts", "reports", "fingerprints"} <= tables
