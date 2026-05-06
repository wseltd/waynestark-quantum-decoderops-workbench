"""Evidence endpoints (T104).

- GET /evidence/latest     — evidence bundle for the most recent run.
- GET /evidence/{run_id}   — evidence bundle for a specific run (alias).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.api.schemas import EvidenceSummary
from app.db.repositories.runs_repo import RunsRepository

router = APIRouter(tags=["evidence"])


@router.get("/evidence/latest", response_model=EvidenceSummary)
def get_evidence_latest(
    session: Session = Depends(get_db_session),
) -> EvidenceSummary:
    """Return the evidence bundle summary for the most recent run.

    'Most recent' is ordered by Run.started_at DESC. An empty DB
    returns a 404 so callers can branch.
    """
    repo = RunsRepository(session)
    rows = repo.list(limit=1, offset=0, order_by="created_at_desc")
    if not rows:
        raise HTTPException(
            status_code=404, detail="no runs recorded yet"
        )
    return EvidenceSummary(run_id=rows[0].run_id)


@router.get("/evidence/{run_id}", response_model=EvidenceSummary)
def get_evidence(run_id: str) -> EvidenceSummary:
    return EvidenceSummary(run_id=run_id)
