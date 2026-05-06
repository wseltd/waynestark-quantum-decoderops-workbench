"""Report ORM model (T058)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Sequence,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

__all__ = ["Report"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(
        Integer, Sequence("reports_id_seq"), primary_key=True
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.run_id"),
        nullable=False,
        index=True,
    )
    type: Mapped[str] = mapped_column(String, nullable=False)
    format: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )

    run = relationship("Run", backref="reports")

    __table_args__ = (
        UniqueConstraint(
            "run_id", "type", "format", name="uq_reports_run_type_format"
        ),
        CheckConstraint(
            "type IN ('engineering_benchmark','decoder_comparison',"
            "'deployment_readiness','artefact_manifest','risk_caveat',"
            "'json_results_bundle')",
            name="allowed_report_types",
        ),
        CheckConstraint(
            "format IN ('markdown','html','pdf','json')",
            name="allowed_report_formats",
        ),
    )
