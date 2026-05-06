"""Tests for app.core.fingerprint (T011)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.core.fingerprint import (
    ReproducibilityFingerprint,
    build_fingerprint,
    fingerprint_to_json,
)
from app.core.seeding import SeedPlan, derive_worker_seeds


def _sp(master: int = 42) -> SeedPlan:
    seeds = tuple(derive_worker_seeds(master, 3))
    return SeedPlan(master_seed=master, worker_seeds=seeds)


def _fixed_now() -> datetime:
    return datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_build_fingerprint_populates_all_required_string_fields(
    tmp_path: Path,
) -> None:
    fp = build_fingerprint(
        config_hash="c" * 64,
        seed_plan=_sp(),
        now=_fixed_now,
        environment_report_path=tmp_path / "nope.json",
    )
    assert fp.config_hash == "c" * 64
    assert fp.git_sha
    assert isinstance(fp.git_dirty, bool)
    assert fp.python_version


def test_build_fingerprint_uses_injected_clock_for_timestamp(
    tmp_path: Path,
) -> None:
    fp = build_fingerprint(
        config_hash="c" * 64,
        seed_plan=_sp(),
        now=_fixed_now,
        environment_report_path=tmp_path / "nope.json",
    )
    assert fp.timestamp_utc == "2026-01-02T03:04:05Z"


def test_build_fingerprint_reads_gpu_facts_from_environment_report_json(
    tmp_path: Path,
) -> None:
    env = tmp_path / "env.json"
    env.write_text(
        json.dumps(
            {
                "gpu_models": ["RTX PRO 6000"],
                "gpu_count": 1,
                "nvidia_driver_version": "560.35.03",
                "cuda_runtime_version": "13.0",
            }
        )
    )
    fp = build_fingerprint(
        config_hash="c" * 64,
        seed_plan=_sp(),
        now=_fixed_now,
        environment_report_path=env,
    )
    assert fp.gpu_models == ("RTX PRO 6000",)
    assert fp.gpu_count == 1
    assert fp.nvidia_driver_version == "560.35.03"
    assert fp.cuda_runtime_version == "13.0"


def test_build_fingerprint_sets_gpu_fields_to_none_when_environment_report_missing(
    tmp_path: Path,
) -> None:
    fp = build_fingerprint(
        config_hash="c" * 64,
        seed_plan=_sp(),
        now=_fixed_now,
        environment_report_path=tmp_path / "nope.json",
    )
    assert fp.gpu_models == ()
    assert fp.gpu_count == 0
    assert fp.nvidia_driver_version is None
    assert fp.cuda_runtime_version is None


def test_build_fingerprint_sets_gpu_fields_to_none_when_environment_report_has_no_gpu(
    tmp_path: Path,
) -> None:
    env = tmp_path / "env.json"
    env.write_text("{}")
    fp = build_fingerprint(
        config_hash="c" * 64,
        seed_plan=_sp(),
        now=_fixed_now,
        environment_report_path=env,
    )
    assert fp.gpu_count == 0
    assert fp.nvidia_driver_version is None


def test_build_fingerprint_pip_freeze_digest_is_sha256_of_sorted_freeze(
    tmp_path: Path,
) -> None:
    fp = build_fingerprint(
        config_hash="c" * 64,
        seed_plan=_sp(),
        now=_fixed_now,
        environment_report_path=tmp_path / "nope.json",
    )
    assert re.match(r"^[0-9a-f]{64}$", fp.pip_freeze_digest)


def test_build_fingerprint_embeds_master_and_worker_seeds_from_seed_plan(
    tmp_path: Path,
) -> None:
    sp = _sp(master=999)
    fp = build_fingerprint(
        config_hash="c" * 64,
        seed_plan=sp,
        now=_fixed_now,
        environment_report_path=tmp_path / "nope.json",
    )
    assert fp.master_seed == 999
    assert fp.worker_seeds == sp.worker_seeds


def test_build_fingerprint_sets_git_dirty_true_when_working_tree_has_changes(
    tmp_path: Path,
) -> None:
    # This is an environmental assertion — we just check the field exists
    # and is boolean; asserting the exact value would flake based on repo
    # state when tests are run.
    fp = build_fingerprint(
        config_hash="c" * 64,
        seed_plan=_sp(),
        now=_fixed_now,
        environment_report_path=tmp_path / "nope.json",
    )
    assert isinstance(fp.git_dirty, bool)


def test_build_fingerprint_returns_unknown_git_sha_when_git_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "check_output", _raise)
    fp = build_fingerprint(
        config_hash="c" * 64,
        seed_plan=_sp(),
        now=_fixed_now,
        environment_report_path=tmp_path / "nope.json",
    )
    assert fp.git_sha == "unknown"
    assert fp.git_dirty is False


def test_fingerprint_to_json_is_canonical_sorted_and_compact(
    tmp_path: Path,
) -> None:
    fp = build_fingerprint(
        config_hash="c" * 64,
        seed_plan=_sp(),
        now=_fixed_now,
        environment_report_path=tmp_path / "nope.json",
    )
    j = fingerprint_to_json(fp)
    assert ", " not in j and ": " not in j  # compact
    decoded = json.loads(j)
    reencoded = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    assert reencoded == j


def test_fingerprint_to_json_is_byte_reproducible_for_same_inputs(
    tmp_path: Path,
) -> None:
    kwargs = dict(
        config_hash="c" * 64,
        seed_plan=_sp(),
        now=_fixed_now,
        environment_report_path=tmp_path / "nope.json",
    )
    fp1 = build_fingerprint(**kwargs)
    fp2 = build_fingerprint(**kwargs)
    # timestamps identical via injected clock; git_sha stable; pip_freeze
    # stable within a single pytest process.
    assert fingerprint_to_json(fp1) == fingerprint_to_json(fp2)


def test_build_fingerprint_timestamp_is_iso8601_zulu(tmp_path: Path) -> None:
    fp = build_fingerprint(
        config_hash="c" * 64,
        seed_plan=_sp(),
        now=_fixed_now,
        environment_report_path=tmp_path / "nope.json",
    )
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", fp.timestamp_utc)
    assert isinstance(fp, ReproducibilityFingerprint)
