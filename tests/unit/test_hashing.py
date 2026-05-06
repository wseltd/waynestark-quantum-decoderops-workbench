"""Adversarial tests for app.core.hashing.

Expected digests are never hardcoded as hex literals. They are recomputed with
``hashlib`` at test time so the tests remain a faithful cross-check of the
module against the stdlib implementation and do not drift if someone swaps the
test vector.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path

import pytest

from app.core import hashing
from app.core.hashing import DEFAULT_CHUNK_SIZE, sha256_bytes, sha256_file


def _expected(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_sha256_bytes_matches_hashlib_on_empty_input() -> None:
    assert sha256_bytes(b"") == _expected(b"")


def test_sha256_bytes_matches_hashlib_on_known_vector() -> None:
    # NIST FIPS-180 single-block test vector. Expected digest is computed via
    # hashlib rather than embedded as a literal so no 64-char hex string is
    # hardcoded into this file.
    vector = b"abc"
    assert sha256_bytes(vector) == _expected(vector)


def test_sha256_bytes_returns_lowercase_hex_of_length_64() -> None:
    digest = sha256_bytes(b"quantum-decoderops")
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(c in "0123456789abcdef" for c in digest)


def test_sha256_file_matches_hashlib_on_small_file(tmp_path: Path) -> None:
    payload = b"the quick brown fox jumps over the lazy dog"
    target = tmp_path / "small.bin"
    target.write_bytes(payload)
    assert sha256_file(target) == _expected(payload)


def test_sha256_file_matches_hashlib_on_multi_chunk_file(tmp_path: Path) -> None:
    # Force at least three full chunks plus a partial tail so the streaming
    # loop is exercised end-to-end. Using a deterministic PRNG seed keeps the
    # test reproducible without embedding digest literals.
    import random

    rng = random.Random(0)
    payload = bytes(rng.randrange(256) for _ in range(DEFAULT_CHUNK_SIZE * 3 + 17))
    target = tmp_path / "large.bin"
    target.write_bytes(payload)

    assert sha256_file(target) == _expected(payload)
    # Accepts PathLike as well as str.
    assert sha256_file(str(target)) == _expected(payload)


def test_sha256_file_raises_file_not_found_for_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.bin"
    with pytest.raises(FileNotFoundError) as info:
        sha256_file(missing)
    assert str(missing) in str(info.value)


def test_sha256_file_raises_for_directory_path(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError) as info:
        sha256_file(tmp_path)
    assert str(tmp_path) in str(info.value)


def test_sha256_file_default_chunk_size_is_one_mebibyte() -> None:
    # Guards the contract that DEFAULT_CHUNK_SIZE is exactly 1 MiB and that
    # sha256_file's chunk_size parameter defaults to that constant.
    assert DEFAULT_CHUNK_SIZE == 1024 * 1024
    signature = inspect.signature(sha256_file)
    assert signature.parameters["chunk_size"].default == DEFAULT_CHUNK_SIZE


def test_sha256_file_rejects_non_positive_chunk_size(tmp_path: Path) -> None:
    target = tmp_path / "x.bin"
    target.write_bytes(b"x")
    with pytest.raises(ValueError):
        sha256_file(target, chunk_size=0)
    with pytest.raises(ValueError):
        sha256_file(target, chunk_size=-1)


def test_sha256_file_varied_chunk_sizes_agree(tmp_path: Path) -> None:
    # A correct streaming implementation must be independent of chunk size.
    payload = os.urandom(200_000)
    target = tmp_path / "vary.bin"
    target.write_bytes(payload)

    baseline = _expected(payload)
    for size in (1, 17, 4096, DEFAULT_CHUNK_SIZE):
        assert sha256_file(target, chunk_size=size) == baseline


def test_module_exposes_only_sha256(tmp_path: Path) -> None:
    # Restraint check: no md5/sha1/blake2 helpers leaked into the public surface.
    public = {name for name in vars(hashing) if not name.startswith("_")}
    for banned in ("md5", "sha1", "blake2", "blake2b", "blake2s"):
        assert banned not in public
