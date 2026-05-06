"""Fingerprint ORM model (T059)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    Sequence,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

__all__ = ["Fingerprint"]


class Fingerprint(Base):
    __tablename__ = "fingerprints"

    id: Mapped[int] = mapped_column(
        Integer, Sequence("fingerprints_id_seq"), primary_key=True
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.run_id"),
        nullable=False,
        unique=True,
        index=True,
    )
    git_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    pip_freeze_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    rng_master_seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    cpu_model: Mapped[str] = mapped_column(String, nullable=False)
    cpu_count: Mapped[int] = mapped_column(Integer, nullable=False)
    gpu_model: Mapped[str | None] = mapped_column(String, nullable=True)
    gpu_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    gpu_driver_version: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    os_name: Mapped[str] = mapped_column(String, nullable=False)
    os_kernel: Mapped[str] = mapped_column(String, nullable=False)
    python_version: Mapped[str] = mapped_column(String, nullable=False)
    cuda_runtime_version: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    run = relationship("Run", backref="fingerprint")
