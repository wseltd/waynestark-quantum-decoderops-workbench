"""POST /export/onnx — stub (T102)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import ExportOnnxRequest

router = APIRouter(tags=["export"])


@router.post("/export/onnx")
def post_export_onnx(req: ExportOnnxRequest) -> dict:
    return {
        "run_id": req.run_id,
        "output_path": req.output_path,
        "accepted": True,
    }
