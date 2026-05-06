"""End-to-end tarball build + verify (T181)."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from app.packaging.tarball import build_tarball
from app.packaging.verify import verify_tarball


def _mk_source(tmp_path: Path) -> Path:
    s = tmp_path / "src"
    s.mkdir()
    (s / "a.bin").write_bytes(b"A" * 16)
    (s / "sub").mkdir()
    (s / "sub" / "b.bin").write_bytes(b"B" * 32)
    return s


def test_build_tarball_produces_content_addressed_name(tmp_path: Path) -> None:
    p = build_tarball(
        _mk_source(tmp_path),
        output_dir=tmp_path / "out",
        manifest={"run_id": "r"},
    )
    assert p.name.startswith("decoderops-") and p.name.endswith(".tar.gz")


def test_tarball_filename_contains_sha256_prefix(tmp_path: Path) -> None:
    p = build_tarball(_mk_source(tmp_path), output_dir=tmp_path / "o")
    # name shape: decoderops-<16-hex>.tar.gz
    stem = p.name[len("decoderops-") : -len(".tar.gz")]
    assert len(stem) == 16 and all(c in "0123456789abcdef" for c in stem)


def test_tarball_is_byte_reproducible_for_same_inputs(tmp_path: Path) -> None:
    s = _mk_source(tmp_path)
    p1 = build_tarball(s, output_dir=tmp_path / "o1", manifest={"r": "x"})
    p2 = build_tarball(s, output_dir=tmp_path / "o2", manifest={"r": "x"})
    assert p1.read_bytes() == p2.read_bytes()


def test_verify_tarball_offline_succeeds_for_valid_tarball(tmp_path: Path) -> None:
    s = _mk_source(tmp_path)
    p = build_tarball(s, output_dir=tmp_path / "o", manifest={"run_id": "r"})
    result = verify_tarball(p)
    assert result.ok
    assert result.checked_count >= 2


def test_verify_tarball_offline_detects_tampered_manifest(tmp_path: Path) -> None:
    s = _mk_source(tmp_path)
    p = build_tarball(s, output_dir=tmp_path / "o", manifest={"run_id": "r"})
    # Rewrite the tarball with the file contents tampered post-build.
    tampered = tmp_path / "tampered.tar"
    with tarfile.open(p, "r:gz") as src, tarfile.open(tampered, "w") as dst:
        for m in src.getmembers():
            if m.isfile() and m.name.endswith(".bin"):
                # swap bytes
                data = b"X" * m.size
                m2 = tarfile.TarInfo(m.name)
                m2.size = len(data)
                dst.addfile(m2, __import__("io").BytesIO(data))
            else:
                f = src.extractfile(m)
                if f is None:
                    dst.addfile(m)
                else:
                    dst.addfile(m, f)
    result = verify_tarball(tampered)
    assert not result.ok
    assert result.mismatched
