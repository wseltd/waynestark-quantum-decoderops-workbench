"""Tests for app.benchmarking.parallel (T035)."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from app.benchmarking.orchestrator import RunConfig
from app.benchmarking.parallel import (
    MAX_WORKERS_CAP,
    _resolve_max_workers,
    _worker_entry,
    derive_worker_seeds,
    run_parallel,
)
from app.benchmarking.runner import RunResult
from app.core.seeding import derive_worker_seed


def _cfg(slot: int, run_id: str = "") -> RunConfig:
    return RunConfig(
        run_id=run_id or f"{slot:016x}",
        sweep_id="s" * 16,
        distance=3,
        rounds=3,
        noise={"p_error": 0.001, "model": "depolarize"},
        basis="X",
        backend="fake_backend",
        model_variant="baseline",
        export_mode="native",
        worker_seed_slot=slot,
        master_seed=42,
        num_shots=16,
    )


def test_derive_worker_seeds_is_deterministic_for_same_master_seed() -> None:
    cfgs = [_cfg(i) for i in range(5)]
    a = derive_worker_seeds(100, cfgs)
    b = derive_worker_seeds(100, cfgs)
    assert a == b
    assert a == [derive_worker_seed(100, i) for i in range(5)]


def test_derive_worker_seeds_differs_for_different_master_seeds() -> None:
    cfgs = [_cfg(i) for i in range(5)]
    assert derive_worker_seeds(1, cfgs) != derive_worker_seeds(2, cfgs)


def test_run_parallel_seed_derivation_matches_seedsequence_contract() -> None:
    cfgs = [_cfg(i) for i in range(3)]
    seeds = derive_worker_seeds(777, cfgs)
    for cfg, s in zip(cfgs, seeds, strict=True):
        assert s == derive_worker_seed(777, cfg.worker_seed_slot)


def test_run_parallel_respects_max_workers_cap() -> None:
    assert _resolve_max_workers(1000, 500) == min(MAX_WORKERS_CAP, 500)
    assert _resolve_max_workers(None, 4) == min((mp.cpu_count() or 1), 4)
    assert _resolve_max_workers(2, 4) == 2
    with pytest.raises(ValueError):
        _resolve_max_workers(0, 4)


def test_run_parallel_uses_spawn_context() -> None:
    # Confirm the module wires multiprocessing.get_context('spawn')
    # (we cannot introspect an already-closed pool; check via empty-list
    # short-circuit which doesn't spawn at all, AND via the source).
    import app.benchmarking.parallel as pmod

    src = open(pmod.__file__).read()
    assert 'get_context("spawn")' in src or "get_context('spawn')" in src


def test_run_parallel_preserves_submission_order() -> None:
    cfgs = [_cfg(i) for i in range(4)]
    results = run_parallel(
        cfgs,
        decoder_factory_spec=(
            "tests.unit._parallel_fake_factory",
            "make_decoder",
        ),
        master_seed=99,
        max_workers=2,
        num_detectors=8,
        batch_size=8,
    )
    assert [r.config.worker_seed_slot for r in results] == [0, 1, 2, 3]
    assert all(r.ok for r in results)


def test_run_parallel_derives_distinct_worker_seeds() -> None:
    cfgs = [_cfg(i) for i in range(3)]
    seeds = derive_worker_seeds(12345, cfgs)
    assert len(set(seeds)) == 3


def test_run_parallel_handles_worker_exception_as_runresult_error() -> None:
    cfgs = [_cfg(0, run_id="raise_in_factory_0")]
    results = run_parallel(
        cfgs,
        decoder_factory_spec=(
            "tests.unit._parallel_fake_factory",
            "make_raising_factory",
        ),
        master_seed=1,
        max_workers=1,
        num_detectors=8,
        batch_size=8,
    )
    # When the factory raises, run_single captures it as a decoder_factory
    # failed error (not a worker_crashed) — both are acceptable error states.
    assert results[0].error is not None


def test_run_parallel_empty_returns_empty_list() -> None:
    assert run_parallel(
        [],
        decoder_factory_spec=("tests.unit._parallel_fake_factory", "make_decoder"),
        master_seed=1,
    ) == []


def test_worker_entry_returns_runresult_on_import_failure() -> None:
    cfg = _cfg(0)
    result = _worker_entry(
        (cfg, ("no.such.module.exists", "nope"), 0, 8, 8)
    )
    assert isinstance(result, RunResult)
    assert result.error is not None
    assert "worker_crashed" in result.error
