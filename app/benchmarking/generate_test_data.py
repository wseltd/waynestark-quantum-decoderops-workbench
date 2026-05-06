"""CUDA-Q QEC test-data generator subprocess wrapper (T040)."""

from __future__ import annotations

import logging
import os
import subprocess  # noqa: S404 — fixed argv
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.benchmarking.ising_local_run_onnx import _sha256_file

__all__ = [
    "GenerateTestDataError",
    "GenerateTestDataResult",
    "generate_test_data",
]

_LOG = logging.getLogger(__name__)


class GenerateTestDataError(RuntimeError):
    """Raised when the vendor generator subprocess fails."""


@dataclass(frozen=True)
class GenerateTestDataResult:
    command: list[str]
    returncode: int
    stdout_path: Path
    stderr_path: Path
    duration_seconds: float
    produced_files: list[Path]
    sha256_by_path: dict[str, str]
    parameters: dict[str, Any]
    started_at_utc: str
    finished_at_utc: str


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate(
    *,
    distance: int,
    n_rounds: int,
    num_samples: int,
    basis: str,
    p_error: float,
) -> None:
    if not isinstance(distance, int) or distance < 3 or distance % 2 == 0:
        raise ValueError(
            f"distance must be an odd int >= 3; got {distance!r}"
        )
    if not isinstance(n_rounds, int) or n_rounds < 1:
        raise ValueError(f"n_rounds must be int >= 1; got {n_rounds!r}")
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError(
            f"num_samples must be int >= 1; got {num_samples!r}"
        )
    if basis not in ("X", "Z"):
        raise ValueError(f"basis must be 'X' or 'Z'; got {basis!r}")
    if not (0.0 < p_error < 0.5):
        raise ValueError(
            f"p_error must satisfy 0 < p < 0.5; got {p_error!r}"
        )


def generate_test_data(
    vendor_root: Path,
    output_dir: Path,
    *,
    distance: int,
    n_rounds: int,
    num_samples: int,
    basis: Literal["X", "Z"],
    p_error: float,
    simple_noise: bool = False,
    python_executable: str | None = None,
    timeout_seconds: int = 1800,
) -> GenerateTestDataResult:
    """Call vendor generate_test_data.py as a subprocess and hash produced files."""
    _validate(
        distance=distance,
        n_rounds=n_rounds,
        num_samples=num_samples,
        basis=basis,
        p_error=p_error,
    )
    vendor_root = Path(vendor_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    py = python_executable or sys.executable
    generator = vendor_root / "code/export/generate_test_data.py"
    command: list[str] = [
        py,
        str(generator),
        "--distance",
        str(distance),
        "--n-rounds",
        str(n_rounds),
        "--num-samples",
        str(num_samples),
        "--basis",
        basis,
        "--p-error",
        repr(p_error),
    ]
    if simple_noise:
        command.append("--simple-noise")

    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    started = _now_utc_iso()
    t0 = time.perf_counter()
    window_start = time.time()
    try:
        with open(stdout_path, "wb") as so, open(stderr_path, "wb") as se:
            proc = subprocess.Popen(  # noqa: S603 — fixed argv
                command,
                cwd=str(vendor_root / "code"),
                stdout=so,
                stderr=se,
                shell=False,
            )
            try:
                returncode = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as te:
                proc.kill()
                proc.wait(timeout=5)
                raise GenerateTestDataError(
                    f"generate_test_data.py timed out after {timeout_seconds}s"
                ) from te
    except FileNotFoundError as e:
        raise GenerateTestDataError(
            f"vendor generator not found at {generator}"
        ) from e
    duration = time.perf_counter() - t0
    finished = _now_utc_iso()

    if returncode != 0:
        raise GenerateTestDataError(
            f"generate_test_data.py exited with code {returncode}; "
            f"stderr: {stderr_path}"
        )

    produced: list[Path] = []
    for root in (output_dir, vendor_root / "code"):
        if not root.exists():
            continue
        for pattern in ("*.bin", "*.onnx"):
            for p in root.rglob(pattern):
                try:
                    if p.stat().st_mtime >= window_start - 1.0:
                        produced.append(p)
                except OSError:
                    continue
    produced = sorted(set(produced), key=lambda p: p.as_posix())
    sha_map = {str(p): _sha256_file(p) for p in produced}

    parameters: dict[str, Any] = {
        "distance": distance,
        "n_rounds": n_rounds,
        "num_samples": num_samples,
        "basis": basis,
        "p_error": p_error,
        "simple_noise": simple_noise,
    }

    return GenerateTestDataResult(
        command=list(command),
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        duration_seconds=duration,
        produced_files=produced,
        sha256_by_path=sha_map,
        parameters=parameters,
        started_at_utc=started,
        finished_at_utc=finished,
    )
