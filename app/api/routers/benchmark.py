"""Benchmark endpoints (T098).

- POST /benchmark/run      — expand a SweepSpec and return the RunConfigs.
- GET  /benchmark/{run_id} — fetch a previously-expanded run by id.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.api.schemas import BenchmarkRunRequest, RunSummary
from app.benchmarking.orchestrator import expand_sweep
from app.benchmarking.sweep import NoiseSpec, SweepSpec
from app.db.repositories.runs_repo import RunsRepository

router = APIRouter(tags=["benchmark"])


@router.post("/benchmark/run")
def post_benchmark_run(req: BenchmarkRunRequest) -> dict:
    spec = SweepSpec(
        distances=list(req.distances),
        rounds=list(req.rounds),
        basis=list(req.bases),
        noise=[
            NoiseSpec(p_error=p, model="simple_depolarizing") for p in req.p_errors
        ],
        backends=list(req.backends),
        model_variants=["none"],
        export_modes=["none"],
        num_shots=req.num_shots,
        master_seed=req.master_seed,
    )
    configs = list(expand_sweep(spec))
    return {
        "sweep_id": spec.canonical_hash(),
        "num_runs": len(configs),
        "run_ids": [c.run_id for c in configs],
    }


@router.get("/benchmark/{run_id}", response_model=RunSummary)
def get_benchmark_run(
    run_id: str, session: Session = Depends(get_db_session)
) -> RunSummary:
    """Fetch a benchmark run by id from the DB."""
    repo = RunsRepository(session)
    r = repo.get(run_id)
    if r is None:
        raise HTTPException(
            status_code=404, detail=f"benchmark run not found: {run_id}"
        )
    return RunSummary(
        run_id=r.run_id,
        status=r.status,
        backend=r.backend,
        started_at=r.started_at.isoformat() if r.started_at else None,
        finished_at=r.finished_at.isoformat() if r.finished_at else None,
    )
