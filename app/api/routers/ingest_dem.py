"""POST /ingest/dem (T096)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest/dem", response_model=IngestResponse)
def post_ingest_dem(payload: dict) -> IngestResponse:
    source = payload.get("path") or payload.get("source")
    if not source:
        raise HTTPException(status_code=422, detail="missing 'path' or 'source'")
    return IngestResponse(
        source=str(source), ok=True, summary={"stub": True}
    )
