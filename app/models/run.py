"""Run ORM model (T055)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Index,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

__all__ = ["Run"]


_ALLOWED_BACKENDS = frozenset(
    {
        "pymatching_baseline",
        "ising_fast",
        "ising_accurate",
        "onnx_validation",
        "tensorrt_optional",
    }
)
_ALLOWED_STATUSES = frozenset(
    {"pending", "running", "succeeded", "failed", "cancelled"}
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    backend: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    git_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    pip_freeze_digest: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    rng_master_seed: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )
    db_backend: Mapped[str] = mapped_column(String(16), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "backend IN ("
            + ", ".join(f"'{b}'" for b in sorted(_ALLOWED_BACKENDS))
            + ")",
            name="allowed_backends",
        ),
        CheckConstraint(
            "status IN ("
            + ", ".join(f"'{s}'" for s in sorted(_ALLOWED_STATUSES))
            + ")",
            name="allowed_statuses",
        ),
        Index("ix_runs_started_at", "started_at"),
        Index("ix_runs_backend_status", "backend", "status"),
    )

    @classmethod
    def allowed_backends(cls) -> frozenset[str]:
        return _ALLOWED_BACKENDS

    @classmethod
    def allowed_statuses(cls) -> frozenset[str]:
        return _ALLOWED_STATUSES
