"""RunsRepository — CRUD for Run (T064)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.run import Run

__all__ = ["RunNotFoundError", "RunsRepository"]


class RunNotFoundError(LookupError):
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"run not found: {run_id!r}")


class RunsRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, run: Run) -> Run:
        self.session.add(run)
        self.session.flush()
        return run

    def get(self, run_id: str) -> Run | None:
        return self.session.get(Run, run_id)

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at_desc",
    ) -> list[Run]:
        stmt = select(Run)
        if order_by == "created_at_desc":
            stmt = stmt.order_by(Run.started_at.desc())
        else:
            stmt = stmt.order_by(Run.started_at.asc())
        stmt = stmt.limit(limit).offset(offset)
        return list(self.session.execute(stmt).scalars())

    def update_status(self, run_id: str, status: str) -> Run:
        r = self.session.get(Run, run_id)
        if r is None:
            raise RunNotFoundError(run_id) from None
        r.status = status
        self.session.flush()
        return r

    def delete(self, run_id: str) -> bool:
        r = self.session.get(Run, run_id)
        if r is None:
            raise RunNotFoundError(run_id) from None
        self.session.delete(r)
        self.session.flush()
        return True
