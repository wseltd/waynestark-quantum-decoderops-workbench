"""Tests for app.packaging.manifest (T048)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.core.fingerprint import build_fingerprint
from app.core.seeding import SeedPlan, derive_worker_seeds
from app.packaging.manifest import (
    ArtefactEntry,
    Manifest,
    build_manifest,
    load_manifest,
    write_manifest,
)


def _fingerprint_dict() -> dict:
    return {
        "git_sha": "abc",
        "pip_freeze_digest": "d" * 64,
        "config_hash": "e" * 64,
        "rng_master_seed": 42,
        "python_version": "3.12.13",
        "os": "Linux 6.8",
        "cpu_model": "x86_64",
        "cpu_count": 4,
        "gpu_model": "",
        "gpu_count": 0,
        "gpu_driver_version": "",
        "cuda_runtime_version": "",
        "timestamp_utc": "2026-01-01T00:00:00Z",
    }


def _entries() -> list[ArtefactEntry]:
    return [
        ArtefactEntry(
            path="data/a.onnx",
            sha256="a" * 64,
            size_bytes=10,
            kind="onnx",
        ),
    ]


def test_build_manifest_happy_path() -> None:
    m = build_manifest(
        "r1", _fingerprint_dict(), _entries(), "e" * 64
    )
    assert m.run_id == "r1"


def test_manifest_schema_version_pinned_to_1() -> None:
    m = build_manifest("r1", _fingerprint_dict(), _entries(), "e" * 64)
    assert m.schema_version == "1"


def test_write_manifest_is_byte_deterministic_for_same_input(tmp_path: Path) -> None:
    m = build_manifest(
        "r1", _fingerprint_dict(), _entries(), "e" * 64,
        created_at_utc="2026-01-01T00:00:00Z",
    )
    p1 = Path(write_manifest(m, tmp_path / "a.json"))
    p2 = Path(write_manifest(m, tmp_path / "b.json"))
    assert p1.read_bytes() == p2.read_bytes()


def test_write_manifest_uses_sorted_keys_and_compact_separators(
    tmp_path: Path,
) -> None:
    m = build_manifest(
        "r1", _fingerprint_dict(), _entries(), "e" * 64,
        created_at_utc="2026-01-01T00:00:00Z",
    )
    p = Path(write_manifest(m, tmp_path / "m.json"))
    txt = p.read_text().rstrip("\n")
    # No ": " or ", " → compact
    assert ": " not in txt
    assert ", " not in txt
    decoded = json.loads(txt)
    re_encoded = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    assert re_encoded == txt


def test_load_manifest_round_trips_written_manifest(tmp_path: Path) -> None:
    m = build_manifest(
        "r1", _fingerprint_dict(), _entries(), "e" * 64,
        created_at_utc="2026-01-01T00:00:00Z",
    )
    p = Path(write_manifest(m, tmp_path / "m.json"))
    m2 = load_manifest(p)
    assert m2 == m


def test_manifest_rejects_invalid_sha256() -> None:
    with pytest.raises(ValueError):
        ArtefactEntry(path="a", sha256="nope", size_bytes=0, kind="other")


def test_manifest_rejects_absolute_artefact_path() -> None:
    with pytest.raises(ValueError):
        ArtefactEntry(path="/abs", sha256="a" * 64, size_bytes=0, kind="other")


def test_manifest_rejects_parent_traversal_in_artefact_path() -> None:
    with pytest.raises(ValueError):
        ArtefactEntry(
            path="../escape", sha256="a" * 64, size_bytes=0, kind="other"
        )


def test_manifest_rejects_empty_run_id() -> None:
    with pytest.raises(ValueError):
        build_manifest("", _fingerprint_dict(), _entries(), "e" * 64)


def test_manifest_rejects_unknown_artefact_kind() -> None:
    with pytest.raises(ValueError):
        ArtefactEntry(
            path="a", sha256="a" * 64, size_bytes=0, kind="bogus"  # type: ignore
        )


def test_build_manifest_defaults_created_at_utc_to_iso_z_format() -> None:
    m = build_manifest("r1", _fingerprint_dict(), _entries(), "e" * 64)
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", m.created_at_utc)


def test_manifest_accepts_reproducibility_fingerprint_from_T011(
    tmp_path: Path,
) -> None:
    sp = SeedPlan(master_seed=1, worker_seeds=tuple(derive_worker_seeds(1, 2)))
    fp = build_fingerprint(
        config_hash="c" * 64,
        seed_plan=sp,
        now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
        environment_report_path=tmp_path / "nope.json",
    )
    m = build_manifest("r1", fp, _entries(), "c" * 64)
    assert isinstance(m, Manifest)


def test_manifest_includes_all_required_fingerprint_keys() -> None:
    m = build_manifest("r1", _fingerprint_dict(), _entries(), "e" * 64)
    for key in (
        "git_sha",
        "pip_freeze_digest",
        "config_hash",
        "rng_master_seed",
        "python_version",
        "os",
        "cpu_model",
        "cpu_count",
        "gpu_model",
        "gpu_count",
        "gpu_driver_version",
        "cuda_runtime_version",
        "timestamp_utc",
    ):
        assert key in m.fingerprint
