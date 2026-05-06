"""SHA256 stamping of artefact directories (T049)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

__all__ = ["CHUNK_SIZE", "stamp_directory", "stamp_file"]

CHUNK_SIZE: int = 1 << 20
_LOG = logging.getLogger(__name__)


def stamp_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                h.update(chunk)
    except OSError as e:
        raise OSError(f"failed to stamp {path}: {e}") from e
    return h.hexdigest()


def _iter_files(root: Path, follow_symlinks: bool) -> Iterable[Path]:
    for p in sorted(root.rglob("*"), key=lambda q: q.as_posix()):
        if p.is_dir():
            continue
        if p.is_symlink() and not follow_symlinks:
            continue
        yield p


def stamp_directory(
    root: Path,
    *,
    manifest_path: Path | None = None,
    follow_symlinks: bool = False,
) -> dict[str, str]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"root does not exist: {root}")
    mapping: dict[str, str] = {}
    manifest_abs = Path(manifest_path).resolve() if manifest_path else None
    for p in _iter_files(root, follow_symlinks=follow_symlinks):
        if manifest_abs is not None and p.resolve() == manifest_abs:
            continue
        rel = p.relative_to(root).as_posix()
        mapping[rel] = stamp_file(p)
    # Sort mapping deterministically
    sorted_map = dict(sorted(mapping.items(), key=lambda kv: kv[0]))
    _LOG.info("stamped %s files under %s", len(sorted_map), root)
    if manifest_path is not None:
        mp = Path(manifest_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(sorted_map, sort_keys=True, separators=(",", ":"))
        mp.write_text(payload + "\n", encoding="utf-8")
    return sorted_map
