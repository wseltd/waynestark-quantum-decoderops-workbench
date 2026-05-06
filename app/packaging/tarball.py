"""Deterministic tarball builder — byte-reproducible artefact bundles (T050)."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from app.packaging.sha256_stamp import stamp_directory

__all__ = ["FIXED_MTIME", "build_tarball"]

FIXED_MTIME: int = 1704067200  # 2024-01-01 UTC


def _reset_tarinfo(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    ti.mtime = FIXED_MTIME
    ti.uid = 0
    ti.gid = 0
    ti.uname = ""
    ti.gname = ""
    if ti.isdir():
        ti.mode = 0o755
    else:
        ti.mode = 0o644
    return ti


def build_tarball(
    source_dir: Path,
    *,
    output_dir: Path,
    manifest: dict | None = None,
    name_prefix: str = "decoderops",
) -> Path:
    source_dir = Path(source_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"source_dir missing: {source_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_map = stamp_directory(source_dir)
    full_manifest: dict[str, Any] = {"files": files_map}
    if manifest is not None:
        full_manifest["run"] = dict(manifest)

    manifest_bytes = json.dumps(
        full_manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    tarball_name = f"{name_prefix}-{digest[:16]}.tar.gz"
    tarball_path = output_dir / tarball_name

    # Write raw tar into memory, then gzip with mtime=0 to a final file.
    raw = io.BytesIO()
    with tarfile.open(
        fileobj=raw, mode="w", format=tarfile.PAX_FORMAT
    ) as tf:
        entries = sorted(source_dir.rglob("*"), key=lambda p: p.as_posix())
        for p in entries:
            rel = p.relative_to(source_dir).as_posix()
            if p.is_dir():
                ti = tarfile.TarInfo(name=rel + "/")
                ti.type = tarfile.DIRTYPE
                _reset_tarinfo(ti)
                tf.addfile(ti)
            elif p.is_file():
                data = p.read_bytes()
                ti = tarfile.TarInfo(name=rel)
                ti.size = len(data)
                _reset_tarinfo(ti)
                tf.addfile(ti, io.BytesIO(data))
        # Manifest last at root
        mti = tarfile.TarInfo(name="manifest.json")
        mti.size = len(manifest_bytes)
        _reset_tarinfo(mti)
        tf.addfile(mti, io.BytesIO(manifest_bytes))

    raw_bytes = raw.getvalue()
    with open(tarball_path, "wb") as out_f:
        with gzip.GzipFile(
            filename="", mtime=0, fileobj=out_f, mode="wb"
        ) as gz:
            gz.write(raw_bytes)
    return tarball_path
