"""Tests for alembic.ini + env.py (T069-T071)."""

from __future__ import annotations

import configparser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_alembic_ini_structure() -> None:
    p = REPO_ROOT / "alembic.ini"
    assert p.exists()
    c = configparser.ConfigParser()
    c.read(p)
    assert c["alembic"]["script_location"] == "alembic"
    assert c["alembic"]["sqlalchemy.url"].startswith("driver://")
    assert "%(rev)s" in c["alembic"]["file_template"]
    for s in (
        "loggers",
        "handlers",
        "formatters",
        "logger_root",
        "logger_sqlalchemy",
        "logger_alembic",
    ):
        assert s in c


def test_env_py_has_required_markers() -> None:
    p = REPO_ROOT / "alembic" / "env.py"
    text = p.read_text()
    assert "target_metadata = Base.metadata" in text
    assert "from app.db.base import Base" in text
    assert "compare_type=True" in text
    assert "compare_server_default=True" in text
    assert "def run_migrations_offline" in text
    assert "def run_migrations_online" in text


def test_initial_migration_exists() -> None:
    versions = REPO_ROOT / "alembic" / "versions"
    files = list(versions.glob("*_initial_slug.py"))
    assert files
