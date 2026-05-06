"""WORKFLOW=inference wrapper around vendor local_run.sh (T038)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.benchmarking.ising_subprocess import (
    IsingLocalRunRunner,
    IsingSubprocessError,
)

__all__ = ["InferenceRunRecord", "run_inference", "_parse_inference_stdout"]


@dataclass(frozen=True)
class InferenceRunRecord:
    workflow: str
    model_variant: str
    returncode: int
    stdout_path: Path
    stderr_path: Path
    duration_seconds: float
    parsed_summary: dict[str, Any]
    vendor_git_sha: str | None
    started_at_utc: str
    finished_at_utc: str


_KV_RE = re.compile(r"^(?P<k>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?P<v>.+)\s*$")
_SUPPORTED_KEYS = ("device", "checkpoint_path", "num_shots", "wallclock_seconds")


def _parse_inference_stdout(text: str) -> dict[str, Any]:
    """Extract known keys from inference stdout; never invent missing keys."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        m = _KV_RE.match(line.strip())
        if not m:
            continue
        key = m.group("k")
        if key in _SUPPORTED_KEYS:
            out[key] = m.group("v").strip()
    return out


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _vendor_git_sha(vendor_root: Path) -> str | None:
    head = vendor_root / ".git" / "HEAD"
    if not head.exists():
        return None
    try:
        content = head.read_text().strip()
        if content.startswith("ref: "):
            ref = content[5:]
            ref_path = vendor_root / ".git" / ref
            if ref_path.exists():
                return ref_path.read_text().strip()
            return None
        return content
    except OSError:
        return None


def run_inference(
    vendor_root: Path,
    work_dir: Path,
    *,
    model_variant: Literal["fast", "accurate"],
    cuda_visible_devices: str | None = None,
    timeout_seconds: int = 1800,
) -> InferenceRunRecord:
    """Run local_run.sh with WORKFLOW=inference and capture a structured record."""
    if model_variant not in ("fast", "accurate"):
        raise ValueError(
            f"model_variant must be 'fast' or 'accurate'; got {model_variant!r}"
        )

    env: dict[str, str] = {"WORKFLOW": "inference"}
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # Accurate = vendor model_id 4 (RF=13); Fast = model_id 1 (RF=9, default).
    # config_public.yaml defaults to model_id=1 so we only override for Accurate.
    if model_variant == "accurate":
        env["EXTRA_PARAMS"] = "model_id=4"

    runner = IsingLocalRunRunner(
        vendor_root, work_dir, timeout_seconds=timeout_seconds
    )
    started = _now_utc_iso()
    try:
        result = runner.run(env)
        returncode = result.returncode
        stdout_path = result.stdout_path
        stderr_path = result.stderr_path
        duration = result.duration_seconds
    except IsingSubprocessError:
        # Still return a record: capture the partial outputs that do exist.
        finished = _now_utc_iso()
        raise
    finished = _now_utc_iso()

    parsed: dict[str, Any] = {}
    if stdout_path.exists():
        parsed = _parse_inference_stdout(stdout_path.read_text())

    return InferenceRunRecord(
        workflow="inference",
        model_variant=model_variant,
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        duration_seconds=duration,
        parsed_summary=parsed,
        vendor_git_sha=_vendor_git_sha(Path(vendor_root)),
        started_at_utc=started,
        finished_at_utc=finished,
    )
