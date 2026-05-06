"""Unit tests for the benchmark orchestrator (T033)."""

from __future__ import annotations

import pytest

from app.benchmarking.orchestrator import (
    MAX_SWEEP_SIZE,
    RunConfig,
    SweepTooLargeError,
    compute_run_id,
    expand_sweep,
)
from app.benchmarking.sweep import SweepSpec


def _spec(**overrides) -> SweepSpec:
    defaults = dict(
        distances=[3, 5],
        rounds=[1],
        noise=[{"p_error": 0.001, "model": "simple_depolarizing"}],
        basis=["Z"],
        backends=["pymatching_baseline"],
        model_variants=["none"],
        export_modes=["none"],
        master_seed=42,
        num_shots=1000,
    )
    defaults.update(overrides)
    return SweepSpec(**defaults)  # type: ignore[arg-type]


def test_expand_sweep_cartesian_product() -> None:
    s = _spec(
        distances=[3, 5],
        rounds=[1, 3],
        basis=["X", "Z"],
    )
    cfgs = list(expand_sweep(s))
    assert len(cfgs) == 2 * 2 * 2  # 8
    assert all(isinstance(c, RunConfig) for c in cfgs)


def test_expand_sweep_deterministic_order() -> None:
    s = _spec()
    a = [c.run_id for c in expand_sweep(s)]
    b = [c.run_id for c in expand_sweep(s)]
    c_ids = [c.run_id for c in expand_sweep(SweepSpec(**s.model_dump()))]
    assert a == b == c_ids


def test_expand_sweep_run_id_is_stable_sha256_prefix() -> None:
    s = _spec()
    cfgs = list(expand_sweep(s))
    for c in cfgs:
        assert len(c.run_id) == 16
        assert all(ch in "0123456789abcdef" for ch in c.run_id)
    assert len({c.run_id for c in cfgs}) == len(cfgs)


def test_expand_sweep_worker_seed_slots_are_contiguous_indices() -> None:
    s = _spec()
    cfgs = list(expand_sweep(s))
    assert [c.worker_seed_slot for c in cfgs] == list(range(len(cfgs)))


def test_expand_sweep_sweep_id_matches_spec_canonical_hash() -> None:
    s = _spec()
    expected = s.canonical_hash()
    for c in expand_sweep(s):
        assert c.sweep_id == expected


def test_expand_sweep_too_large_raises() -> None:
    s = _spec(
        distances=[d for d in range(3, 3 + 2 * 51, 2)],
        rounds=list(range(1, 202)),
    )
    with pytest.raises(SweepTooLargeError) as excinfo:
        list(expand_sweep(s))
    assert excinfo.value.limit == MAX_SWEEP_SIZE
    assert excinfo.value.size > MAX_SWEEP_SIZE


def test_compute_run_id_excludes_run_id_field() -> None:
    cfg = {"run_id": "should-be-ignored", "distance": 3, "rounds": 1}
    a = compute_run_id(cfg)
    cfg["run_id"] = "another-different-string"
    b = compute_run_id(cfg)
    assert a == b


def test_run_config_is_frozen() -> None:
    s = _spec()
    cfg = next(iter(expand_sweep(s)))
    with pytest.raises(Exception):
        cfg.distance = 999  # type: ignore[misc]
