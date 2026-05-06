"""Reports endpoints (T103).

- POST /reports/generate    — render the 5-type x 4-format matrix for a run.
- GET  /reports/{run_id}    — list rendered report rows persisted for a run.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.api.schemas import ReportRequest
from app.db.repositories.reports_repo import ReportsRepository
from app.reports.context import build_context
from app.reports.pipeline import render_all

router = APIRouter(tags=["reports"])


@router.post("/reports/generate")
def post_reports_generate(req: ReportRequest) -> dict:
    # Stub context when the DB has no real run — enough to exercise the
    # rendering pipeline end-to-end.
    context = build_context(
        run={"run_id": req.run_id},
        metrics=[],
        artefacts=[],
        host={
            "cpu_model": "x86",
            "cpu_count": 1,
            "gpu_model": "",
            "gpu_count": 0,
            "driver_version": "",
            "cuda_runtime_version": "",
            "os_kernel": "",
            "python_version": "",
        },
        decoders=[],
        sweep_axes={
            "code_distance": [],
            "rounds": [],
            "basis": [],
            "noise_params": [],
            "model_variant": [],
            "export_mode": [],
        },
        shots_total=0,
        reproducibility_fingerprint_sha256="",
    )
    out = Path(mkdtemp(prefix=f"reports-{req.run_id}-"))
    rendered = render_all(
        context=context, output_dir=out, include_pdf=req.include_pdf
    )
    return {
        "run_id": req.run_id,
        "output_dir": str(out),
        "reports": [
            {
                "type": r.type,
                "format": r.format,
                "path": str(r.path),
                "sha256": r.sha256,
            }
            for r in rendered
        ],
    }


@router.get("/reports/{run_id}")
def get_reports_for_run(
    run_id: str, session: Session = Depends(get_db_session)
) -> list[dict]:
    """List report rows persisted in the DB for a run."""
    repo = ReportsRepository(session)
    rows = repo.get_by_run_id(run_id)
    return [
        {
            "id": r.id,
            "run_id": r.run_id,
            "type": r.type,
            "format": r.format,
            "path": r.path,
            "sha256": r.sha256,
        }
        for r in rows
    ]
