"""Behavioural tests for app.core.seeding.

The risk surface here is small but sharp: if seed derivation is not
deterministic, byte-stable, or input-validated, the entire
reproducibility fingerprint silently lies. These tests focus effort on
the determinism and validation edges; the dataclass holder gets one
boring round-trip test because there is nothing else to break.
"""

from __future__ import annotations

import hashlib

import pytest

from app.core.seeding import (
    SeedPlan,
    derive_worker_seed,
    derive_worker_seeds,
)


def test_derive_worker_seed_is_deterministic_across_calls() -> None:
    first = derive_worker_seed(42, 0)
    second = derive_worker_seed(42, 0)
    assert first == second


def test_derive_worker_seed_differs_by_worker_index() -> None:
    a = derive_worker_seed(42, 0)
    b = derive_worker_seed(42, 1)
    assert a != b


def test_derive_worker_seed_differs_by_master_seed() -> None:
    a = derive_worker_seed(42, 0)
    b = derive_worker_seed(43, 0)
    assert a != b


def test_derive_worker_seeds_returns_unique_values_for_distinct_indices() -> None:
    seeds = derive_worker_seeds(12345, 8)
    assert len(set(seeds)) == len(seeds)


def test_derive_worker_seeds_length_matches_num_workers() -> None:
    assert derive_worker_seeds(7, 0) == []
    assert len(derive_worker_seeds(7, 5)) == 5


def test_derive_worker_seed_rejects_negative_master_seed() -> None:
    with pytest.raises(ValueError, match="master_seed"):
        derive_worker_seed(-1, 0)


def test_derive_worker_seed_rejects_negative_worker_index() -> None:
    with pytest.raises(ValueError, match="worker_index"):
        derive_worker_seed(0, -1)


def test_derive_worker_seed_fits_in_63_bits() -> None:
    # Sweep a handful of values; if the mask were missing, even one of
    # these would routinely exceed 2**63.
    for master in (0, 1, 2**63, 2**64 - 1):
        for worker in (0, 1, 2**31, 2**32 - 1):
            value = derive_worker_seed(master, worker)
            assert 0 <= value < (1 << 63)


def test_seed_plan_holds_master_and_worker_seeds() -> None:
    seeds = tuple(derive_worker_seeds(99, 3))
    plan = SeedPlan(master_seed=99, worker_seeds=seeds)
    assert plan.master_seed == 99
    assert plan.worker_seeds == seeds


def test_derive_worker_seed_matches_documented_byte_layout() -> None:
    # Recompute the digest by hand to pin the wire format. If anyone
    # reorders bytes, swaps endianness, or changes field widths, this
    # test fails immediately — that change would silently invalidate
    # every historical reproducibility fingerprint.
    master, worker = 0xDEADBEEFCAFEBABE, 0x01020304
    payload = master.to_bytes(8, "big") + worker.to_bytes(4, "big")
    expected_head = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    expected = expected_head & ((1 << 63) - 1)
    assert derive_worker_seed(master, worker) == expected


def test_derive_worker_seed_rejects_master_seed_above_uint64() -> None:
    with pytest.raises(ValueError, match="master_seed"):
        derive_worker_seed(1 << 64, 0)


def test_derive_worker_seed_rejects_worker_index_above_uint32() -> None:
    with pytest.raises(ValueError, match="worker_index"):
        derive_worker_seed(0, 1 << 32)


def test_derive_worker_seed_rejects_bool_inputs() -> None:
    # bool is an int subclass; without an explicit reject, True/False
    # would silently behave as 1/0 and corrupt the fingerprint provenance.
    with pytest.raises(ValueError, match="master_seed"):
        derive_worker_seed(True, 0)
    with pytest.raises(ValueError, match="worker_index"):
        derive_worker_seed(0, False)


def test_derive_worker_seeds_rejects_negative_num_workers() -> None:
    with pytest.raises(ValueError, match="num_workers"):
        derive_worker_seeds(0, -1)


def test_derive_worker_seed_boundary_values_accepted() -> None:
    # The largest legal master_seed and worker_index must not raise and
    # must still fit in 63 bits.
    value = derive_worker_seed((1 << 64) - 1, (1 << 32) - 1)
    assert 0 <= value < (1 << 63)
