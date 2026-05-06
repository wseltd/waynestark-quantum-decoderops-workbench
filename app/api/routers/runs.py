"""/runs endpoints (T099)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.api.schemas import RunSummary
from app.db.repositories.runs_repo import RunsRepository

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("", response_model=list[RunSummary])
def list_runs(
    session: Session = Depends(get_db_session),
    limit: int = 100,
    offset: int = 0,
) -> list[RunSummary]:
    repo = RunsRepository(session)
    rows = repo.list(limit=limit, offset=offset)
    return [
        RunSummary(
            run_id=r.run_id,
            status=r.status,
            backend=r.backend,
            started_at=r.started_at.isoformat() if r.started_at else None,
            finished_at=r.finished_at.isoformat() if r.finished_at else None,
        )
        for r in rows
    ]


@router.get("/{run_id}", response_model=RunSummary)
def get_run(
    run_id: str, session: Session = Depends(get_db_session)
) -> RunSummary:
    repo = RunsRepository(session)
    r = repo.get(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return RunSummary(
        run_id=r.run_id,
        status=r.status,
        backend=r.backend,
        started_at=r.started_at.isoformat() if r.started_at else None,
        finished_at=r.finished_at.isoformat() if r.finished_at else None,
    )
