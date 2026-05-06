"""Tests for app.benchmarking.ising_local_run_onnx (T039)."""

from __future__ import annotations

import hashlib
import re
import stat
import tempfile
from pathlib import Path

import pytest

from app.benchmarking.ising_local_run_onnx import (
    OnnxExportRecord,
    _sha256_file,
    run_onnx_export,
)


def _make_fake_vendor(
    tmp_path: Path,
    *,
    exit_code: int = 0,
    produce_onnx: bool = True,
    onnx_content: bytes = b"FAKE_ONNX_BYTES_001",
) -> Path:
    vendor = tmp_path / "vendor"
    script_dir = vendor / "code" / "scripts"
    script_dir.mkdir(parents=True)
    out_dir = vendor / "code" / "out"
    out_dir.mkdir(parents=True)
    script = script_dir / "local_run.sh"
    onnx_out = out_dir / "exported.onnx"
    # Write an .onnx file only when produce_onnx is True.
    body = "#!/usr/bin/env bash\n"
    body += "echo ONNX_WORKFLOW=${ONNX_WORKFLOW}\n"
    body += "echo QUANT_FORMAT=${QUANT_FORMAT}\n"
    if produce_onnx:
        # Use python to avoid shell-quoting surprises with binary content.
        body += (
            f"python3 -c \"from pathlib import Path; "
            f"Path(r'{onnx_out.as_posix()}').write_bytes({onnx_content!r})\"\n"
        )
    body += f"exit {exit_code}\n"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return vendor


def test_sha256_file_matches_hashlib_for_known_bytes(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert _sha256_file(p) == hashlib.sha256(b"abc").hexdigest()


def test_discover_includes_vendor_root_when_vendor_writes_there(
    tmp_path: Path,
) -> None:
    """Regression: real local_run.sh writes *.onnx to the vendor root
    itself (after `cd "${REPO_ROOT}"`). The discovery scan must include
    vendor_root, not just work_dir + vendor_root/code."""
    vendor = tmp_path / "vendor"
    script_dir = vendor / "code" / "scripts"
    script_dir.mkdir(parents=True)
    script = script_dir / "local_run.sh"
    # The fake vendor writes an .onnx to vendor_root itself (matches real
    # layout: vendor/Ising-Decoding/predecoder_memory_d7_T7_X.onnx)
    body = "#!/usr/bin/env bash\n"
    body += "echo ONNX_WORKFLOW=${ONNX_WORKFLOW}\n"
    body += "echo QUANT_FORMAT=${QUANT_FORMAT}\n"
    onnx_at_root = vendor / "predecoder_memory_d7_T7_X.onnx"
    body += (
        f"python3 -c \"from pathlib import Path; "
        f"Path(r'{onnx_at_root.as_posix()}').write_bytes(b'REAL_ONNX_BYTES')\"\n"
    )
    body += "exit 0\n"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    rec = run_onnx_export(
        vendor,
        tmp_path / "work",
        onnx_workflow=1,
        quant_format="int8",
        model_variant="fast",
    )
    assert len(rec.exported_onnx_paths) >= 1, (
        f"discovery missed vendor-root onnx; scanned paths: {rec.sha256_by_path}"
    )
    assert any(
        str(p).endswith("predecoder_memory_d7_T7_X.onnx")
        for p in rec.exported_onnx_paths
    )
    assert rec.export_success is True


def test_run_onnx_export_rejects_invalid_quant_format(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    with pytest.raises(ValueError):
        run_onnx_export(
            vendor,
            tmp_path / "w",
            onnx_workflow=1,
            quant_format="fp16",  # type: ignore[arg-type]
            model_variant="fast",
        )


def test_run_onnx_export_rejects_invalid_onnx_workflow_value(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path)
    with pytest.raises(ValueError):
        run_onnx_export(
            vendor,
            tmp_path / "w",
            onnx_workflow=3,  # type: ignore[arg-type]
            quant_format="int8",
            model_variant="fast",
        )


def test_run_onnx_export_sets_env_for_workflow_1_int8(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    rec = run_onnx_export(
        vendor,
        tmp_path / "w",
        onnx_workflow=1,
        quant_format="int8",
        model_variant="fast",
    )
    stdout = rec.stdout_path.read_text()
    assert "ONNX_WORKFLOW=1" in stdout
    assert "QUANT_FORMAT=int8" in stdout


def test_run_onnx_export_sets_env_for_workflow_2_fp8(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    rec = run_onnx_export(
        vendor,
        tmp_path / "w",
        onnx_workflow=2,
        quant_format="fp8",
        model_variant="accurate",
    )
    stdout = rec.stdout_path.read_text()
    assert "ONNX_WORKFLOW=2" in stdout
    assert "QUANT_FORMAT=fp8" in stdout


def test_run_onnx_export_discovers_and_hashes_emitted_onnx_files(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path, onnx_content=b"hello-onnx")
    rec = run_onnx_export(
        vendor,
        tmp_path / "w",
        onnx_workflow=1,
        quant_format="int8",
        model_variant="fast",
    )
    assert len(rec.exported_onnx_paths) >= 1
    assert rec.export_success is True
    expected = hashlib.sha256(b"hello-onnx").hexdigest()
    assert any(h == expected for h in rec.sha256_by_path.values())


def test_run_onnx_export_sets_export_success_false_when_no_onnx_emitted(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path, produce_onnx=False)
    rec = run_onnx_export(
        vendor,
        tmp_path / "w",
        onnx_workflow=1,
        quant_format="int8",
        model_variant="fast",
    )
    assert rec.export_success is False


def test_run_onnx_export_sets_export_success_false_on_nonzero_returncode(
    tmp_path: Path,
) -> None:
    # Non-zero returncode from the subprocess is captured into the record;
    # export_success is False even if .onnx files happen to exist.
    vendor = _make_fake_vendor(tmp_path, exit_code=1, produce_onnx=True)
    rec = run_onnx_export(
        vendor,
        tmp_path / "w",
        onnx_workflow=1,
        quant_format="int8",
        model_variant="fast",
    )
    assert rec.export_success is False
    assert rec.returncode != 0


def test_run_onnx_export_record_contains_iso_utc_timestamps(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path)
    rec = run_onnx_export(
        vendor,
        tmp_path / "w",
        onnx_workflow=1,
        quant_format="int8",
        model_variant="fast",
    )
    iso = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    assert iso.match(rec.started_at_utc)
    assert iso.match(rec.finished_at_utc)
    assert isinstance(rec, OnnxExportRecord)
