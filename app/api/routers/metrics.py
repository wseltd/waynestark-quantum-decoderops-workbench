"""/metrics endpoints (T100)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.api.schemas import MetricsSummary
from app.db.repositories.metrics_repo import MetricsRepository

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/{run_id}", response_model=list[MetricsSummary])
def get_metrics(
    run_id: str, session: Session = Depends(get_db_session)
) -> list[MetricsSummary]:
    repo = MetricsRepository(session)
    ms = repo.get_by_run_id(run_id)
    return [
        MetricsSummary(
            run_id=m.run_id, ler=m.ler, ci_low=m.ci_low, ci_high=m.ci_high
        )
        for m in ms
    ]
