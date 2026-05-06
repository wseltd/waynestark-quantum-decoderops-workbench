"""Tests for app.benchmarking.ising_subprocess (T037)."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from app.benchmarking.ising_subprocess import (
    IsingLocalRunResult,
    IsingLocalRunRunner,
    IsingSubprocessError,
)


def _make_fake_vendor(
    tmp_path: Path, *, exit_code: int = 0, stdout: str = "ok", stderr: str = ""
) -> Path:
    vendor = tmp_path / "vendor"
    script_dir = vendor / "code" / "scripts"
    script_dir.mkdir(parents=True)
    script = script_dir / "local_run.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f"echo {stdout!r}\n"
        f"echo {stderr!r} >&2\n"
        f"exit {exit_code}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return vendor


def test_runner_rejects_unknown_env_key(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    runner = IsingLocalRunRunner(vendor, tmp_path / "work", timeout_seconds=10)
    with pytest.raises(ValueError) as exc:
        runner.run({"NOT_ALLOWED": "1"})
    assert "NOT_ALLOWED" in str(exc.value)


def test_runner_constructs_command_with_cwd_and_timeout(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    runner = IsingLocalRunRunner(vendor, tmp_path / "work", timeout_seconds=5)
    result = runner.run({"WORKFLOW": "inference"})
    assert result.command[0] == "bash"
    assert "local_run.sh" in result.command[1]
    assert result.returncode == 0


def test_runner_raises_ising_subprocess_error_on_nonzero_exit(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path, exit_code=3)
    runner = IsingLocalRunRunner(vendor, tmp_path / "work", timeout_seconds=10)
    with pytest.raises(IsingSubprocessError) as exc:
        runner.run({"WORKFLOW": "inference"})
    assert "3" in str(exc.value)


def test_runner_writes_stdout_and_stderr_to_work_dir(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(
        tmp_path, stdout="hello-out", stderr="hello-err"
    )
    work = tmp_path / "work"
    runner = IsingLocalRunRunner(vendor, work, timeout_seconds=10)
    result = runner.run({"WORKFLOW": "inference"})
    assert result.stdout_path.read_text().strip() == "hello-out"
    assert result.stderr_path.read_text().strip() == "hello-err"
    assert result.stdout_path.parent == work


def test_runner_returns_result_with_duration_and_env_snapshot(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path)
    runner = IsingLocalRunRunner(vendor, tmp_path / "work", timeout_seconds=10)
    result = runner.run({"WORKFLOW": "inference", "DISTANCE": "5"})
    assert isinstance(result, IsingLocalRunResult)
    assert result.duration_seconds >= 0.0
    assert result.env_snapshot == {"WORKFLOW": "inference", "DISTANCE": "5"}


def test_runner_rejects_non_positive_timeout(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    with pytest.raises(ValueError):
        IsingLocalRunRunner(vendor, tmp_path / "w", timeout_seconds=0)
