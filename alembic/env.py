"""Alembic environment — Postgres-only, URL from Settings (T070)."""

from __future__ import annotations

import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config.settings import Settings
from app.db.base import Base

# Make sure all ORM models are imported so their metadata is registered.
import app.models.artefact  # noqa: F401
import app.models.fingerprint  # noqa: F401
import app.models.metrics  # noqa: F401
import app.models.report  # noqa: F401
import app.models.run  # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

_log = logging.getLogger(__name__)


def _resolve_url() -> str:
    try:
        url = Settings().database_url
    except Exception as e:
        raise RuntimeError("failed to load Settings.database_url") from e
    scheme = url.split(":", 1)[0]
    if scheme not in ("postgresql", "postgresql+psycopg", "postgresql+psycopg2"):
        raise RuntimeError(
            "Alembic is Postgres-only in decoderops; "
            f"refusing to run against scheme %s ({url})" % scheme
        )
    return url


_url = _resolve_url()
config.set_main_option("sqlalchemy.url", _url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    _log.info("running migrations offline: url=%s", url)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section) or {},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as conn:
        _log.info("running migrations online: dialect=%s", conn.dialect.name)
        context.configure(
            connection=conn,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
