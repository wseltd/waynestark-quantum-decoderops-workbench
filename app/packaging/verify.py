"""Offline tarball verifier (T054)."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from app.packaging.manifest import MANIFEST_FILENAME

__all__ = ["VerifyResult", "main", "verify_tarball"]


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    checked_count: int
    missing: list[str]
    mismatched: list[tuple[str, str, str]]
    extra: list[str]
    manifest_schema_version: str


def _hash_stream(stream: io.IOBase, chunk: int = 64 * 1024) -> str:
    h = hashlib.sha256()
    while True:
        data = stream.read(chunk)
        if not data:
            break
        h.update(data)
    return h.hexdigest()


def _extract_manifest(tf: tarfile.TarFile) -> dict:
    member = tf.getmember(MANIFEST_FILENAME)
    f = tf.extractfile(member)
    if f is None:
        raise ValueError("manifest entry is not a regular file")
    return json.loads(f.read().decode("utf-8"))


def _manifest_entries(manifest: dict) -> dict[str, str]:
    """Map artefact path -> expected sha256.

    Supports three shapes:
        * {'entries': [{'path':..., 'sha256':...}, ...]}  (T054 inline spec)
        * {'artefacts': [...]}  (T048 Manifest output)
        * {'files': {path: sha256}, 'run': {...}}  (T050 tarball output)
    """
    if "entries" in manifest and isinstance(manifest["entries"], list):
        return {e["path"]: e["sha256"] for e in manifest["entries"]}
    if "artefacts" in manifest and isinstance(manifest["artefacts"], list):
        return {a["path"]: a["sha256"] for a in manifest["artefacts"]}
    if "files" in manifest and isinstance(manifest["files"], dict):
        return dict(manifest["files"])
    raise ValueError("manifest has no entries/artefacts/files mapping")


def verify_tarball(tarball_path: Path) -> VerifyResult:
    tarball_path = Path(tarball_path)
    missing: list[str] = []
    mismatched: list[tuple[str, str, str]] = []
    extra: list[str] = []
    checked = 0
    with tarfile.open(tarball_path, "r:*") as tf:
        manifest = _extract_manifest(tf)
        schema_version = str(manifest.get("schema_version", "unknown"))
        expected = _manifest_entries(manifest)
        actual_members = {
            m.name: m for m in tf.getmembers() if m.isfile()
        }
        for path, expected_sha in expected.items():
            m = actual_members.get(path)
            if m is None:
                missing.append(path)
                continue
            f = tf.extractfile(m)
            if f is None:
                missing.append(path)
                continue
            actual_sha = _hash_stream(f)
            if actual_sha != expected_sha:
                mismatched.append((path, expected_sha, actual_sha))
            checked += 1
        for name in actual_members:
            if name == MANIFEST_FILENAME:
                continue
            if name not in expected:
                extra.append(name)
    ok = not missing and not mismatched
    return VerifyResult(
        ok=ok,
        checked_count=checked,
        missing=sorted(missing),
        mismatched=sorted(mismatched),
        extra=sorted(extra),
        manifest_schema_version=schema_version,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="verify a decoderops tarball")
    parser.add_argument("--tarball", required=True, type=Path)
    args = parser.parse_args(argv)
    result = verify_tarball(args.tarball)
    summary = {
        "ok": result.ok,
        "checked_count": result.checked_count,
        "missing": result.missing,
        "mismatched": [list(t) for t in result.mismatched],
        "extra": result.extra,
        "manifest_schema_version": result.manifest_schema_version,
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n"
    )
    if result.ok:
        return 0
    if result.mismatched:
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
