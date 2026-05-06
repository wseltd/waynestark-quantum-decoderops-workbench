"""Controlled subprocess wrapper for vendor local_run.sh (T037)."""

from __future__ import annotations

import logging
import os
import subprocess  # noqa: S404 — fixed argv, no shell=True
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ALLOWED_ENV_KEYS",
    "IsingLocalRunResult",
    "IsingLocalRunRunner",
    "IsingSubprocessError",
]


ALLOWED_ENV_KEYS: frozenset[str] = frozenset(
    {
        "WORKFLOW",
        "ONNX_WORKFLOW",
        "EXTRA_PARAMS",
        "QUANT_FORMAT",
        "DISTANCE",
        "N_ROUNDS",
        "NUM_SAMPLES",
        "BASIS",
        "P_ERROR",
        "SIMPLE_NOISE",
        "CUDA_VISIBLE_DEVICES",
    }
)

_LOG = logging.getLogger(__name__)


class IsingSubprocessError(RuntimeError):
    """Raised when local_run.sh exits non-zero or fails to start."""


@dataclass(frozen=True)
class IsingLocalRunResult:
    returncode: int
    stdout_path: Path
    stderr_path: Path
    duration_seconds: float
    env_snapshot: dict[str, str]
    command: list[str]


class IsingLocalRunRunner:
    """Invoke ``vendor_root/code/scripts/local_run.sh`` with capture + timeout."""

    def __init__(
        self,
        vendor_root: Path,
        work_dir: Path,
        timeout_seconds: int = 1800,
    ) -> None:
        self.vendor_root = Path(vendor_root)
        self.work_dir = Path(work_dir)
        self.timeout_seconds = int(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive; got {timeout_seconds}"
            )

    def _build_env(self, env_overlay: dict[str, str]) -> dict[str, str]:
        unknown = set(env_overlay) - ALLOWED_ENV_KEYS
        if unknown:
            raise ValueError(
                "unknown env keys for Ising local_run: "
                f"{sorted(unknown)}; allowed={sorted(ALLOWED_ENV_KEYS)}"
            )
        merged = os.environ.copy()
        for k, v in env_overlay.items():
            merged[k] = str(v)
        return merged

    def run(
        self,
        env: dict[str, str],
        extra_args: list[str] | None = None,
    ) -> IsingLocalRunResult:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = self.work_dir / "stdout.log"
        stderr_path = self.work_dir / "stderr.log"
        script = self.vendor_root / "code/scripts/local_run.sh"
        command: list[str] = ["bash", str(script)]
        if extra_args:
            command.extend(str(a) for a in extra_args)
        merged_env = self._build_env(env)

        t0 = time.perf_counter()
        _LOG.info("ising local_run start: cmd=%s cwd=%s", command, self.vendor_root / "code")
        try:
            with open(stdout_path, "wb") as so, open(stderr_path, "wb") as se:
                proc = subprocess.Popen(  # noqa: S603 — fixed argv
                    command,
                    cwd=str(self.vendor_root / "code"),
                    env=merged_env,
                    stdout=so,
                    stderr=se,
                    shell=False,
                )
                try:
                    returncode = proc.wait(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired as te:
                    proc.kill()
                    proc.wait(timeout=5)
                    raise IsingSubprocessError(
                        f"local_run timed out after {self.timeout_seconds}s"
                    ) from te
        except FileNotFoundError as e:
            raise IsingSubprocessError(
                f"local_run.sh not found at {script}"
            ) from e
        duration = time.perf_counter() - t0
        _LOG.info("ising local_run end: returncode=%s duration=%.2fs", returncode, duration)

        if returncode != 0:
            raise IsingSubprocessError(
                f"local_run.sh exited with code {returncode}; "
                f"stderr: {stderr_path}"
            )

        return IsingLocalRunResult(
            returncode=returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            duration_seconds=duration,
            env_snapshot=dict(env),
            command=list(command),
        )
