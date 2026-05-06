"""Tests for app.packaging.cudaq_export (T051)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.packaging.cudaq_export import (
    CudaqExportManifest,
    package_cudaq_export,
)


def _mk_export(tmp_path: Path, with_txt: bool = True) -> Path:
    d = tmp_path / "export"
    d.mkdir()
    (d / "data.bin").write_bytes(b"\x00\x01")
    (d / "model.onnx").write_bytes(b"onnx-bytes")
    if with_txt:
        (d / "README.txt").write_text("notes")
    return d


def test_package_cudaq_export_classifies_bin_and_onnx_files(
    tmp_path: Path,
) -> None:
    src = _mk_export(tmp_path)
    dst = tmp_path / "pkg"
    m = package_cudaq_export(
        export_dir=src,
        destination=dst,
        run_id="r-1",
        source_command=["python", "g.py"],
    )
    assert "data.bin" in m.bin_files
    assert "model.onnx" in m.onnx_files


def test_package_cudaq_export_copies_files_preserving_structure(
    tmp_path: Path,
) -> None:
    src = tmp_path / "e"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "x.bin").write_bytes(b"X")
    dst = tmp_path / "p"
    package_cudaq_export(
        export_dir=src, destination=dst, run_id="r", source_command=[]
    )
    assert (dst / "cudaq_qec" / "sub" / "x.bin").exists()


def test_package_cudaq_export_writes_deterministic_manifest_json(
    tmp_path: Path,
) -> None:
    src = _mk_export(tmp_path)
    dst = tmp_path / "pkg"
    m = package_cudaq_export(
        export_dir=src, destination=dst, run_id="r", source_command=[]
    )
    loaded = json.loads((dst / "cudaq_qec" / "manifest.json").read_text())
    assert loaded["run_id"] == "r"
    assert isinstance(m, CudaqExportManifest)


def test_package_cudaq_export_records_source_command(tmp_path: Path) -> None:
    src = _mk_export(tmp_path)
    dst = tmp_path / "pkg"
    cmd = ["python", "gen.py", "--distance", "3"]
    m = package_cudaq_export(
        export_dir=src, destination=dst, run_id="r", source_command=cmd
    )
    assert m.source_command == cmd


def test_package_cudaq_export_raises_when_export_dir_empty(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        package_cudaq_export(
            export_dir=empty,
            destination=tmp_path / "pkg",
            run_id="r",
            source_command=[],
        )


def test_package_cudaq_export_raises_when_no_bin_or_onnx_outputs(
    tmp_path: Path,
) -> None:
    src = tmp_path / "nobinonnx"
    src.mkdir()
    (src / "README.txt").write_text("just docs")
    with pytest.raises(FileNotFoundError):
        package_cudaq_export(
            export_dir=src,
            destination=tmp_path / "pkg",
            run_id="r",
            source_command=[],
        )


def test_package_cudaq_export_includes_optional_subprocess_log(
    tmp_path: Path,
) -> None:
    src = _mk_export(tmp_path)
    log = tmp_path / "run.log"
    log.write_text("log content")
    m = package_cudaq_export(
        export_dir=src,
        destination=tmp_path / "pkg",
        run_id="r",
        source_command=[],
        subprocess_log=log,
    )
    assert m.log_file == "run.log"


def test_package_cudaq_export_manifest_file_hashes_match_stamp_directory(
    tmp_path: Path,
) -> None:
    import hashlib

    src = _mk_export(tmp_path, with_txt=False)
    dst = tmp_path / "pkg"
    m = package_cudaq_export(
        export_dir=src, destination=dst, run_id="r", source_command=[]
    )
    assert m.files["data.bin"] == hashlib.sha256(b"\x00\x01").hexdigest()
    assert m.files["model.onnx"] == hashlib.sha256(b"onnx-bytes").hexdigest()
