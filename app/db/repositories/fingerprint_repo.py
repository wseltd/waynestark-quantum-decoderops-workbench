"""FingerprintRepository — CRUD for Fingerprint (T068)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.fingerprint import Fingerprint

__all__ = ["FingerprintRepository"]


class FingerprintRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, fingerprint: Any) -> Fingerprint:
        obj = (
            fingerprint
            if isinstance(fingerprint, Fingerprint)
            else Fingerprint(**dict(fingerprint))
        )
        self.session.add(obj)
        self.session.flush()
        return obj

    def get_by_id(self, fingerprint_id: int) -> Fingerprint | None:
        return self.session.get(Fingerprint, fingerprint_id)

    def get_by_run_id(self, run_id: str) -> Fingerprint | None:
        stmt = select(Fingerprint).where(Fingerprint.run_id == run_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def find_matching(
        self,
        git_sha: str,
        pip_freeze_digest: str,
        config_sha256: str,
        rng_master_seed: int,
    ) -> list[Fingerprint]:
        stmt = (
            select(Fingerprint)
            .where(
                Fingerprint.git_sha == git_sha,
                Fingerprint.pip_freeze_digest == pip_freeze_digest,
                Fingerprint.config_sha256 == config_sha256,
                Fingerprint.rng_master_seed == rng_master_seed,
            )
            .order_by(Fingerprint.timestamp_utc.desc())
        )
        return list(self.session.execute(stmt).scalars())
