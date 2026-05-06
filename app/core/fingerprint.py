"""Reproducibility fingerprint — one per run (T011).

Collects deterministic-ish environment facts for the run manifest.
Wall-clock is the only nondeterministic field and is injectable for tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess  # noqa: S404 — fixed argv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from app.core.seeding import SeedPlan

__all__ = [
    "ReproducibilityFingerprint",
    "build_fingerprint",
    "fingerprint_to_json",
]

_LOG = logging.getLogger(__name__)
_DEFAULT_ENV_REPORT = Path(".decoderops/environment_report.json")


class ReproducibilityFingerprint(BaseModel):
    model_config = ConfigDict(frozen=True)

    git_sha: str
    git_dirty: bool
    pip_freeze_digest: str
    config_hash: str
    master_seed: int
    worker_seeds: tuple[int, ...]
    cpu_model: str
    cpu_count: int
    gpu_models: tuple[str, ...]
    gpu_count: int
    nvidia_driver_version: str | None
    os_name: str
    os_kernel: str
    python_version: str
    cuda_runtime_version: str | None
    timestamp_utc: str


def _git_sha_and_dirty() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(  # noqa: S603
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(  # noqa: S603
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return sha, bool(status.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        _LOG.warning("git unavailable; recording sha=unknown dirty=False")
        return "unknown", False


def _pip_freeze_digest() -> str:
    try:
        out = subprocess.check_output(  # noqa: S603
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        out = ""
    lines = sorted(line.strip() for line in out.splitlines() if line.strip())
    blob = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _read_env_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def build_fingerprint(
    *,
    config_hash: str,
    seed_plan: SeedPlan,
    now: Callable[[], datetime] | None = None,
    environment_report_path: Path | None = None,
) -> ReproducibilityFingerprint:
    now_fn = now if now is not None else lambda: datetime.now(timezone.utc)
    timestamp = now_fn().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    git_sha, git_dirty = _git_sha_and_dirty()
    pip_digest = _pip_freeze_digest()

    env_path = environment_report_path or _DEFAULT_ENV_REPORT
    env = _read_env_report(Path(env_path))
    gpu_models_list = env.get("gpu_models") or []
    gpu_count = int(env.get("gpu_count") or len(gpu_models_list))
    nvidia_driver = env.get("nvidia_driver_version")
    cuda_runtime = env.get("cuda_runtime_version")

    cpu_model = platform.processor() or platform.machine() or "unknown"
    cpu_count = os.cpu_count() or 1

    return ReproducibilityFingerprint(
        git_sha=git_sha,
        git_dirty=git_dirty,
        pip_freeze_digest=pip_digest,
        config_hash=config_hash,
        master_seed=seed_plan.master_seed,
        worker_seeds=tuple(seed_plan.worker_seeds),
        cpu_model=cpu_model,
        cpu_count=cpu_count,
        gpu_models=tuple(gpu_models_list),
        gpu_count=gpu_count,
        nvidia_driver_version=nvidia_driver if nvidia_driver else None,
        os_name=platform.system() or "unknown",
        os_kernel=platform.release() or "unknown",
        python_version=sys.version.split()[0],
        cuda_runtime_version=cuda_runtime if cuda_runtime else None,
        timestamp_utc=timestamp,
    )


def fingerprint_to_json(fp: ReproducibilityFingerprint) -> str:
    data = fp.model_dump(mode="json")
    return json.dumps(data, sort_keys=True, separators=(",", ":"))
