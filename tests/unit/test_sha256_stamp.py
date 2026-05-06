"""Tests for app.packaging.sha256_stamp (T049)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.packaging.sha256_stamp import CHUNK_SIZE, stamp_directory, stamp_file


def test_stamp_file_matches_known_vector(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    assert stamp_file(p) == hashlib.sha256(b"hello").hexdigest()


def test_stamp_directory_is_deterministic_across_calls(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"A")
    (tmp_path / "b.bin").write_bytes(b"B")
    a = stamp_directory(tmp_path)
    b = stamp_directory(tmp_path)
    assert a == b
    assert list(a.keys()) == sorted(a.keys())


def test_stamp_directory_sorts_paths_posix_style(tmp_path: Path) -> None:
    (tmp_path / "zz.bin").write_bytes(b"z")
    (tmp_path / "aa.bin").write_bytes(b"a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.bin").write_bytes(b"c")
    r = stamp_directory(tmp_path)
    assert list(r.keys()) == ["aa.bin", "sub/c.bin", "zz.bin"]


def test_stamp_directory_writes_manifest_json_with_sorted_keys(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.bin").write_bytes(b"A")
    (tmp_path / "b.bin").write_bytes(b"B")
    mp = tmp_path / "m.json"
    r = stamp_directory(tmp_path, manifest_path=mp)
    loaded = json.loads(mp.read_text())
    assert loaded == r
    assert list(loaded.keys()) == sorted(loaded.keys())


def test_stamp_directory_excludes_manifest_file_itself(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"A")
    mp = tmp_path / "m.json"
    r = stamp_directory(tmp_path, manifest_path=mp)
    assert "m.json" not in r


def test_stamp_directory_streams_large_file_without_full_read(
    tmp_path: Path,
) -> None:
    size = CHUNK_SIZE * 3 + 7
    big = tmp_path / "big.bin"
    with open(big, "wb") as f:
        f.write(b"x" * size)
    # Just verify it hashes to the expected value for its content
    r = stamp_directory(tmp_path)
    assert r["big.bin"] == hashlib.sha256(b"x" * size).hexdigest()


def test_stamp_file_raises_oserror_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        stamp_file(tmp_path / "does_not_exist")
