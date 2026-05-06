"""Artefact ORM model (T057)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Sequence,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

__all__ = ["Artefact"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Artefact(Base):
    __tablename__ = "artefacts"

    id: Mapped[int] = mapped_column(
        Integer, Sequence("artefacts_id_seq"), primary_key=True
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.run_id"),
        nullable=False,
        index=True,
    )
    path: Mapped[str] = mapped_column(String, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String, nullable=False)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )

    run = relationship("Run", backref="artefacts")

    __table_args__ = (
        CheckConstraint(
            "length(sha256) = 64", name="sha256_length_64"
        ),
        CheckConstraint(
            "type IN ('onnx','tensorrt_engine','cudaq_bin','parquet',"
            "'report_bundle','tarball','log','manifest')",
            name="allowed_artefact_types",
        ),
    )
