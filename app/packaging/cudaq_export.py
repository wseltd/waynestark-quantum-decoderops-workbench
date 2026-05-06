"""CUDA-Q QEC export artefact packager (T051)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from app.packaging.sha256_stamp import stamp_directory

__all__ = ["CudaqExportManifest", "package_cudaq_export"]


class CudaqExportManifest(BaseModel):
    run_id: str
    source_command: list[str]
    files: dict[str, str]
    bin_files: list[str]
    onnx_files: list[str]
    log_file: Optional[str] = None
    created_at_utc: str


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def package_cudaq_export(
    *,
    export_dir: Path,
    destination: Path,
    run_id: str,
    source_command: list[str],
    subprocess_log: Path | None = None,
) -> CudaqExportManifest:
    export_dir = Path(export_dir)
    destination = Path(destination)
    if not export_dir.exists() or not export_dir.is_dir():
        raise FileNotFoundError(f"export_dir missing: {export_dir}")
    src_items = [p for p in export_dir.rglob("*") if p.is_file()]
    if not src_items:
        raise FileNotFoundError(
            f"export_dir {export_dir} has no files"
        )

    dest_sub = destination / "cudaq_qec"
    dest_sub.mkdir(parents=True, exist_ok=True)

    bin_files: list[str] = []
    onnx_files: list[str] = []
    for p in sorted(src_items, key=lambda q: q.as_posix()):
        rel = p.relative_to(export_dir).as_posix()
        target = dest_sub / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target, follow_symlinks=False)
        if rel.endswith(".bin"):
            bin_files.append(rel)
        elif rel.endswith(".onnx"):
            onnx_files.append(rel)

    if not bin_files and not onnx_files:
        raise FileNotFoundError(
            f"export_dir {export_dir} produced no .bin or .onnx files"
        )

    log_rel: str | None = None
    if subprocess_log is not None and Path(subprocess_log).exists():
        log_target = dest_sub / Path(subprocess_log).name
        shutil.copy2(subprocess_log, log_target, follow_symlinks=False)
        log_rel = log_target.name

    manifest_path = dest_sub / "manifest.json"
    files_map = stamp_directory(dest_sub, manifest_path=manifest_path)

    result = CudaqExportManifest(
        run_id=run_id,
        source_command=list(source_command),
        files=files_map,
        bin_files=bin_files,
        onnx_files=onnx_files,
        log_file=log_rel,
        created_at_utc=_now_utc_iso(),
    )
    # Overwrite manifest.json with the richer structured manifest.
    payload = json.dumps(
        result.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ) + "\n"
    manifest_path.write_text(payload, encoding="utf-8")
    return result
