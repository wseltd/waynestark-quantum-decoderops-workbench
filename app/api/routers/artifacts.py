"""Artifacts endpoints (T101).

- GET /artifacts/{run_id}  — list artefact rows persisted for a run.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.api.schemas import ArtefactSummary
from app.db.repositories.artefacts_repo import ArtefactsRepository

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.get("/{run_id}", response_model=list[ArtefactSummary])
def list_artefacts_for_run(
    run_id: str, session: Session = Depends(get_db_session)
) -> list[ArtefactSummary]:
    repo = ArtefactsRepository(session)
    rows = repo.get_by_run_id(run_id)
    return [
        ArtefactSummary(
            id=a.id, run_id=a.run_id, path=a.path, sha256=a.sha256, type=a.type
        )
        for a in rows
    ]
