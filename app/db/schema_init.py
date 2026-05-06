"""DuckDB schema bootstrap via Base.metadata.create_all (T072).

DuckDB does not use Alembic in this product; the schema is applied directly
from the ORM Base. PostgreSQL uses Alembic migrations instead.
"""

from __future__ import annotations

from sqlalchemy.engine import Engine

from app.db.base import Base

# Import every ORM model so its Table is registered with Base.metadata
import app.models.artefact  # noqa: F401
import app.models.fingerprint  # noqa: F401
import app.models.metrics  # noqa: F401
import app.models.report  # noqa: F401
import app.models.run  # noqa: F401

__all__ = ["bootstrap_schema"]


def bootstrap_schema(engine: Engine) -> None:
    """Create all tables on the given engine."""
    Base.metadata.create_all(engine)
