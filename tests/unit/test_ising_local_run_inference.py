"""Tests for app.benchmarking.ising_local_run_inference (T038)."""

from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest

from app.benchmarking.ising_local_run_inference import (
    InferenceRunRecord,
    _parse_inference_stdout,
    run_inference,
)


def _make_fake_vendor(
    tmp_path: Path,
    *,
    exit_code: int = 0,
    stdout: str = "device=cuda:0\ncheckpoint_path=/opt/model.pt\nnum_shots=1024\nwallclock_seconds=1.23",
    stderr: str = "",
) -> Path:
    vendor = tmp_path / "vendor"
    script_dir = vendor / "code" / "scripts"
    script_dir.mkdir(parents=True)
    script = script_dir / "local_run.sh"
    # Emit env info so tests can assert we passed WORKFLOW/CUDA_VISIBLE_DEVICES.
    script.write_text(
        "#!/usr/bin/env bash\n"
        "echo WORKFLOW_ENV=${WORKFLOW}\n"
        "echo CUDA_VISIBLE_DEVICES_ENV=${CUDA_VISIBLE_DEVICES}\n"
        f"cat <<'EOF'\n{stdout}\nEOF\n"
        f"echo {stderr!r} >&2\n"
        f"exit {exit_code}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return vendor


def test_parse_inference_stdout_extracts_known_keys() -> None:
    text = (
        "device=cuda:0\n"
        "checkpoint_path=/x.pt\n"
        "num_shots=1024\n"
        "wallclock_seconds=1.5\n"
    )
    r = _parse_inference_stdout(text)
    assert r["device"] == "cuda:0"
    assert r["checkpoint_path"] == "/x.pt"
    assert r["num_shots"] == "1024"
    assert r["wallclock_seconds"] == "1.5"


def test_parse_inference_stdout_omits_missing_keys() -> None:
    r = _parse_inference_stdout("device=cuda:0\n")
    assert "num_shots" not in r
    assert "checkpoint_path" not in r


def test_parse_inference_stdout_ignores_unrelated_lines() -> None:
    text = "random text no equals\n-----\nfoo=bar\ndevice=cpu\n"
    r = _parse_inference_stdout(text)
    assert r == {"device": "cpu"}


def test_run_inference_sets_workflow_env_to_inference(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    rec = run_inference(
        vendor, tmp_path / "work", model_variant="fast", timeout_seconds=10
    )
    stdout = rec.stdout_path.read_text()
    assert "WORKFLOW_ENV=inference" in stdout


def test_run_inference_passes_cuda_visible_devices_when_provided(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path)
    rec = run_inference(
        vendor,
        tmp_path / "work",
        model_variant="fast",
        cuda_visible_devices="0",
        timeout_seconds=10,
    )
    stdout = rec.stdout_path.read_text()
    assert "CUDA_VISIBLE_DEVICES_ENV=0" in stdout


def test_run_inference_returns_record_with_iso_utc_timestamps(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path)
    rec = run_inference(
        vendor, tmp_path / "work", model_variant="fast", timeout_seconds=10
    )
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    assert iso_re.match(rec.started_at_utc)
    assert iso_re.match(rec.finished_at_utc)
    assert isinstance(rec, InferenceRunRecord)


def test_run_inference_maps_model_variant_fast_and_accurate(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path)
    fast = run_inference(
        vendor, tmp_path / "w1", model_variant="fast", timeout_seconds=10
    )
    assert fast.model_variant == "fast"
    accurate = run_inference(
        vendor, tmp_path / "w2", model_variant="accurate", timeout_seconds=10
    )
    assert accurate.model_variant == "accurate"


def test_run_inference_rejects_invalid_variant(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    with pytest.raises(ValueError):
        run_inference(
            vendor,
            tmp_path / "w",
            model_variant="medium",  # type: ignore[arg-type]
            timeout_seconds=10,
        )
