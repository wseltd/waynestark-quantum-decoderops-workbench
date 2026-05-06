"""Profiles endpoints.

- GET  /profiles                    — list available profiles
- GET  /profiles/{profile_id}       — full ProfileSpec (model_dump)
- POST /profiles/{profile_id}/run   — execute the profile and emit a
                                      decision + rendered report bundle
- GET  /decisions/{run_id}          — fetch a persisted decision bundle
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.profiles.registry import ProfileNotFoundError, get_profile, iter_profiles
from app.profiles.runner import run_profile
from app.reports.pipeline import render_decision_report

router = APIRouter(tags=["profiles"])


class ProfileRunRequest(BaseModel):
    num_shots: int = 512
    master_seed: int = 20260422
    bases: list[str] | None = None
    include_pdf: bool = False


@router.get("/profiles")
def list_profiles() -> list[dict[str, Any]]:
    return [
        {
            "profile_id": p.profile_id,
            "name": p.name,
            "architecture": p.architecture,
            "caution_label": p.caution_label,
            "num_decoder_paths": len(p.decoder_paths),
            "num_export_checks": len(p.export_checks),
            "has_runtime_budget": p.runtime_budget is not None,
        }
        for p in iter_profiles()
    ]


@router.get("/profiles/{profile_id}")
def get_profile_spec(profile_id: str) -> dict[str, Any]:
    try:
        p = get_profile(profile_id)
    except ProfileNotFoundError:
        raise HTTPException(status_code=404, detail=f"profile not found: {profile_id}")
    return p.model_dump(mode="json")


_LAST_DECISION_BUNDLES: dict[str, dict[str, Any]] = {}


@router.post("/profiles/{profile_id}/run")
def post_profile_run(profile_id: str, req: ProfileRunRequest) -> dict[str, Any]:
    try:
        profile = get_profile(profile_id)
    except ProfileNotFoundError:
        raise HTTPException(status_code=404, detail=f"profile not found: {profile_id}")

    out = Path(tempfile.mkdtemp(prefix=f"profile-{profile_id}-"))
    result = run_profile(
        profile,
        num_shots=req.num_shots,
        master_seed=req.master_seed,
        output_dir=out,
        bases=tuple(req.bases) if req.bases else None,
    )
    bundle = result.to_dict()

    # Decision context for the report templates.
    decision_ctx = {
        "profile": profile.model_dump(mode="json"),
        "decision": bundle["decision"],
        "provenance": bundle["provenance"],
        "run": {
            "master_seed": req.master_seed,
            "num_shots": req.num_shots,
            "effective_bases": bundle["provenance"]["effective_bases"],
        },
    }
    rendered = render_decision_report(
        decision_context=decision_ctx,
        output_dir=out / "report",
        include_pdf=req.include_pdf,
    )
    bundle["rendered_reports"] = [
        {"type": r.type, "format": r.format, "path": str(r.path), "sha256": r.sha256}
        for r in rendered
    ]

    # Persist a simple decision-bundle index by profile_sha256.
    run_id = bundle["provenance"]["profile_sha256"][:16]
    _LAST_DECISION_BUNDLES[run_id] = bundle
    bundle["decision_run_id"] = run_id
    return bundle


@router.get("/decisions/{run_id}")
def get_decision(run_id: str) -> dict[str, Any]:
    if run_id not in _LAST_DECISION_BUNDLES:
        raise HTTPException(
            status_code=404,
            detail=f"decision bundle not found for run_id={run_id}",
        )
    return _LAST_DECISION_BUNDLES[run_id]
