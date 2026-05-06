"""POST /seed — derive worker seeds from a master seed (T095)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.schemas import SeedResponse
from app.core.seeding import derive_worker_seeds

router = APIRouter(tags=["seed"])


@router.post("/seed", response_model=SeedResponse)
def post_seed(
    master_seed: int = Query(..., ge=0),
    num_workers: int = Query(..., ge=1, le=1024),
) -> SeedResponse:
    seeds = derive_worker_seeds(master_seed, num_workers)
    return SeedResponse(master_seed=master_seed, worker_seeds=seeds)
