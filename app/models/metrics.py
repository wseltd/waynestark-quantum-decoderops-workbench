"""Metrics ORM model (T056)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    Sequence,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.run import Run

__all__ = ["Metrics"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Metrics(Base):
    __tablename__ = "metrics"

    metrics_id: Mapped[int] = mapped_column(
        Integer, Sequence("metrics_id_seq"), primary_key=True
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.run_id"),
        nullable=False,
        index=True,
    )
    ler: Mapped[float] = mapped_column(Double, nullable=False)
    ci_low: Mapped[float] = mapped_column(Double, nullable=False)
    ci_high: Mapped[float] = mapped_column(Double, nullable=False)
    p50: Mapped[float] = mapped_column(Double, nullable=False)
    p95: Mapped[float] = mapped_column(Double, nullable=False)
    p99: Mapped[float] = mapped_column(Double, nullable=False)
    throughput: Mapped[float] = mapped_column(Double, nullable=False)
    residual_density: Mapped[float] = mapped_column(Double, nullable=False)
    shots: Mapped[int] = mapped_column(BigInteger, nullable=False)
    rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    code_distance: Mapped[int] = mapped_column(Integer, nullable=False)
    basis: Mapped[str] = mapped_column(String(8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )

    run = relationship("Run", back_populates="metrics")

    __table_args__ = (
        CheckConstraint("ci_low <= ler", name="ci_low_le_ler"),
        CheckConstraint("ler <= ci_high", name="ler_le_ci_high"),
        CheckConstraint(
            "residual_density >= 0 AND residual_density <= 1",
            name="residual_density_unit_interval",
        ),
        CheckConstraint("p50 <= p95", name="p50_le_p95"),
        CheckConstraint("p95 <= p99", name="p95_le_p99"),
        CheckConstraint("throughput >= 0", name="throughput_nonneg"),
        CheckConstraint("shots > 0", name="shots_positive"),
        CheckConstraint("rounds > 0", name="rounds_positive"),
        CheckConstraint("code_distance > 0", name="code_distance_positive"),
        CheckConstraint("basis IN ('X', 'Z')", name="basis_x_or_z"),
    )


# Back-populate the Run side.
Run.metrics = relationship(  # type: ignore[attr-defined]
    "Metrics", back_populates="run", cascade="all, delete-orphan"
)
