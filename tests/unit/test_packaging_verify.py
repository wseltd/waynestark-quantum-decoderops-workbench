"""Tests for app.packaging.verify (T054)."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from app.packaging.verify import VerifyResult, main, verify_tarball


def _write_tarball(
    path: Path,
    files: dict[str, bytes],
    manifest: dict,
) -> Path:
    with tarfile.open(path, "w") as t:
        for name, data in sorted(files.items()):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
        mf = json.dumps(manifest).encode()
        mti = tarfile.TarInfo("manifest.json")
        mti.size = len(mf)
        t.addfile(mti, io.BytesIO(mf))
    return path


def test_verify_tarball_ok_for_matching_manifest(tmp_path: Path) -> None:
    data = b"hello"
    sha = hashlib.sha256(data).hexdigest()
    mf = {
        "schema_version": "1",
        "entries": [{"path": "p.bin", "sha256": sha, "size": len(data)}],
    }
    tp = _write_tarball(tmp_path / "a.tar", {"p.bin": data}, mf)
    r = verify_tarball(tp)
    assert r.ok
    assert r.checked_count == 1
    assert not r.missing
    assert not r.mismatched


def test_verify_tarball_flags_sha256_mismatch(tmp_path: Path) -> None:
    data = b"hello"
    wrong = "0" * 64
    mf = {
        "schema_version": "1",
        "entries": [{"path": "p.bin", "sha256": wrong, "size": 5}],
    }
    tp = _write_tarball(tmp_path / "a.tar", {"p.bin": data}, mf)
    r = verify_tarball(tp)
    assert not r.ok
    assert r.mismatched


def test_verify_tarball_flags_missing_members_listed_in_manifest(
    tmp_path: Path,
) -> None:
    mf = {
        "schema_version": "1",
        "entries": [
            {"path": "gone.bin", "sha256": "0" * 64, "size": 0},
        ],
    }
    tp = _write_tarball(tmp_path / "a.tar", {}, mf)
    r = verify_tarball(tp)
    assert "gone.bin" in r.missing


def test_verify_tarball_flags_extra_members_not_in_manifest(
    tmp_path: Path,
) -> None:
    mf = {"schema_version": "1", "entries": []}
    tp = _write_tarball(tmp_path / "a.tar", {"extra.bin": b"x"}, mf)
    r = verify_tarball(tp)
    assert "extra.bin" in r.extra


def test_verify_tarball_streams_without_extraction(tmp_path: Path) -> None:
    data = b"Z" * 4096
    sha = hashlib.sha256(data).hexdigest()
    mf = {
        "schema_version": "1",
        "entries": [{"path": "p.bin", "sha256": sha, "size": 4096}],
    }
    tp = _write_tarball(tmp_path / "a.tar", {"p.bin": data}, mf)
    before = set(p.name for p in tmp_path.iterdir())
    verify_tarball(tp)
    after = set(p.name for p in tmp_path.iterdir())
    assert before == after  # no new extracted files


def test_verify_cli_exits_zero_on_ok(tmp_path: Path) -> None:
    data = b"hi"
    sha = hashlib.sha256(data).hexdigest()
    mf = {"schema_version": "1", "entries": [{"path": "p.bin", "sha256": sha, "size": 2}]}
    tp = _write_tarball(tmp_path / "a.tar", {"p.bin": data}, mf)
    rc = main(["--tarball", str(tp)])
    assert rc == 0


def test_verify_cli_exits_nonzero_on_mismatch(tmp_path: Path) -> None:
    data = b"hi"
    mf = {"schema_version": "1", "entries": [{"path": "p.bin", "sha256": "0" * 64, "size": 2}]}
    tp = _write_tarball(tmp_path / "a.tar", {"p.bin": data}, mf)
    rc = main(["--tarball", str(tp)])
    assert rc != 0
