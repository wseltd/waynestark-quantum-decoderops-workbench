"""Tests for app.packaging.tarball (T050)."""

from __future__ import annotations

import gzip
import hashlib
import tarfile
import time
from pathlib import Path

import pytest

from app.packaging.tarball import FIXED_MTIME, build_tarball


def _mk_source(tmp_path: Path) -> Path:
    s = tmp_path / "src"
    s.mkdir()
    (s / "a.bin").write_bytes(b"AAA")
    (s / "sub").mkdir()
    (s / "sub" / "b.bin").write_bytes(b"BBBB")
    return s


def test_build_tarball_is_byte_reproducible_across_runs(tmp_path: Path) -> None:
    s = _mk_source(tmp_path)
    o1 = tmp_path / "o1"
    o2 = tmp_path / "o2"
    p1 = build_tarball(s, output_dir=o1, manifest={"run_id": "r1"})
    time.sleep(0.01)
    p2 = build_tarball(s, output_dir=o2, manifest={"run_id": "r1"})
    assert p1.name == p2.name
    assert hashlib.sha256(p1.read_bytes()).hexdigest() == hashlib.sha256(
        p2.read_bytes()
    ).hexdigest()


def test_build_tarball_uses_content_addressed_name(tmp_path: Path) -> None:
    s = _mk_source(tmp_path)
    p = build_tarball(s, output_dir=tmp_path / "o", manifest={"run_id": "r1"})
    assert p.name.startswith("decoderops-")
    assert p.name.endswith(".tar.gz")


def test_build_tarball_tarinfo_has_fixed_mtime_and_zero_uid_gid(
    tmp_path: Path,
) -> None:
    s = _mk_source(tmp_path)
    p = build_tarball(s, output_dir=tmp_path / "o")
    with tarfile.open(p, "r:gz") as t:
        for m in t.getmembers():
            assert m.mtime == FIXED_MTIME
            assert m.uid == 0
            assert m.gid == 0
            assert m.uname == ""


def test_build_tarball_gzip_header_mtime_is_zero(tmp_path: Path) -> None:
    s = _mk_source(tmp_path)
    p = build_tarball(s, output_dir=tmp_path / "o")
    # gzip header mtime is bytes 4..8 little-endian uint32.
    raw = p.read_bytes()
    mtime = int.from_bytes(raw[4:8], "little")
    assert mtime == 0


def test_build_tarball_entries_are_sorted(tmp_path: Path) -> None:
    s = _mk_source(tmp_path)
    p = build_tarball(s, output_dir=tmp_path / "o")
    with tarfile.open(p, "r:gz") as t:
        names = [m.name for m in t.getmembers() if not m.name.endswith("/")]
    names_without_manifest = [n for n in names if n != "manifest.json"]
    assert names_without_manifest == sorted(names_without_manifest)


def test_build_tarball_includes_manifest_json_at_root(tmp_path: Path) -> None:
    s = _mk_source(tmp_path)
    p = build_tarball(s, output_dir=tmp_path / "o")
    with tarfile.open(p, "r:gz") as t:
        names = [m.name for m in t.getmembers()]
    assert "manifest.json" in names


def test_build_tarball_preserves_caller_manifest_under_run_key(
    tmp_path: Path,
) -> None:
    import json

    s = _mk_source(tmp_path)
    p = build_tarball(
        s, output_dir=tmp_path / "o", manifest={"run_id": "abc"}
    )
    with tarfile.open(p, "r:gz") as t:
        mf = t.extractfile("manifest.json").read()
    data = json.loads(mf)
    assert data["run"]["run_id"] == "abc"
    assert "files" in data


def test_build_tarball_raises_when_source_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_tarball(tmp_path / "nope", output_dir=tmp_path / "o")
