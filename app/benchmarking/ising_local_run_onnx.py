"""ONNX export wrapper around vendor local_run.sh (T039)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.benchmarking.ising_subprocess import (
    IsingLocalRunRunner,
    IsingSubprocessError,
)

__all__ = [
    "OnnxExportRecord",
    "_sha256_file",
    "run_onnx_export",
]

_SUPPORTED_WORKFLOWS = (1, 2)
_SUPPORTED_QUANT = ("int8", "fp8")


@dataclass(frozen=True)
class OnnxExportRecord:
    onnx_workflow: int
    quant_format: str
    model_variant: str
    returncode: int
    stdout_path: Path
    stderr_path: Path
    duration_seconds: float
    exported_onnx_paths: list[Path]
    export_success: bool
    sha256_by_path: dict[str, str]
    started_at_utc: str
    finished_at_utc: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _discover_onnx(vendor_root: Path, work_dir: Path, window_start: float) -> list[Path]:
    found: list[Path] = []
    # Vendor's local_run.sh writes ONNX artefacts to `vendor_root` itself
    # (the repo root of the clone, per its `cd "${REPO_ROOT}"` in the
    # script). Also scan `work_dir` and `vendor_root/code` for completeness.
    roots = [work_dir, vendor_root, vendor_root / "code"]
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.onnx"):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime >= window_start - 1.0:  # 1s grace for clock skew
                found.append(p)
    # Stable order: sort by path string
    return sorted(set(found), key=lambda p: p.as_posix())


def run_onnx_export(
    vendor_root: Path,
    work_dir: Path,
    *,
    onnx_workflow: Literal[1, 2],
    quant_format: Literal["int8", "fp8"],
    model_variant: Literal["fast", "accurate"],
    cuda_visible_devices: str | None = None,
    timeout_seconds: int = 1800,
) -> OnnxExportRecord:
    """Invoke local_run.sh with ONNX_WORKFLOW + QUANT_FORMAT and hash outputs."""
    if onnx_workflow not in _SUPPORTED_WORKFLOWS:
        raise ValueError(
            f"onnx_workflow must be one of {_SUPPORTED_WORKFLOWS}; "
            f"got {onnx_workflow!r}"
        )
    if quant_format not in _SUPPORTED_QUANT:
        raise ValueError(
            f"quant_format must be one of {_SUPPORTED_QUANT}; "
            f"got {quant_format!r}"
        )
    if model_variant not in ("fast", "accurate"):
        raise ValueError(
            f"model_variant must be 'fast' or 'accurate'; got {model_variant!r}"
        )

    env: dict[str, str] = {
        "WORKFLOW": "inference",
        "ONNX_WORKFLOW": str(onnx_workflow),
        "QUANT_FORMAT": quant_format,
    }
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    vendor_root = Path(vendor_root)
    work_dir = Path(work_dir)

    runner = IsingLocalRunRunner(
        vendor_root, work_dir, timeout_seconds=timeout_seconds
    )
    started = _now_utc_iso()
    import time as _t

    window_start = _t.time()
    returncode = -1
    duration = 0.0
    stdout_path = work_dir / "stdout.log"
    stderr_path = work_dir / "stderr.log"
    try:
        result = runner.run(env)
        returncode = result.returncode
        stdout_path = result.stdout_path
        stderr_path = result.stderr_path
        duration = result.duration_seconds
    except IsingSubprocessError:
        # Non-zero exit — still build a record with export_success=False.
        if stdout_path.exists() is False:
            stdout_path = work_dir / "stdout.log"
        if stderr_path.exists() is False:
            stderr_path = work_dir / "stderr.log"
    finished = _now_utc_iso()

    produced = _discover_onnx(vendor_root, work_dir, window_start)
    sha_map: dict[str, str] = {
        str(p): _sha256_file(p) for p in produced
    }
    export_success = (returncode == 0) and (len(produced) > 0)

    return OnnxExportRecord(
        onnx_workflow=onnx_workflow,
        quant_format=quant_format,
        model_variant=model_variant,
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        duration_seconds=duration,
        exported_onnx_paths=produced,
        export_success=export_success,
        sha256_by_path=sha_map,
        started_at_utc=started,
        finished_at_utc=finished,
    )
