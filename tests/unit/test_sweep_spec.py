"""Unit tests for SweepSpec + expand() + canonical_hash() (T032)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.benchmarking.sweep import NoiseSpec, SweepPoint, SweepSpec


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


def test_sweep_spec_validates_odd_distances() -> None:
    with pytest.raises(ValidationError):
        _spec(distances=[4, 6])
    with pytest.raises(ValidationError):
        _spec(distances=[1])


def test_sweep_spec_validates_rounds_ge_1() -> None:
    with pytest.raises(ValidationError):
        _spec(rounds=[0])


def test_sweep_spec_validates_p_error_range() -> None:
    with pytest.raises(ValidationError):
        NoiseSpec(p_error=0.5, model="simple_depolarizing")
    with pytest.raises(ValidationError):
        NoiseSpec(p_error=0.0, model="simple_depolarizing")


def test_sweep_spec_rejects_unknown_backend() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _spec(backends=["not_a_real_backend"])
    msg = str(excinfo.value)
    assert "not_a_real_backend" in msg


def test_sweep_spec_is_frozen() -> None:
    s = _spec()
    with pytest.raises(ValidationError):
        s.master_seed = 99  # type: ignore[misc]


def test_expand_yields_cartesian_product_size() -> None:
    s = _spec(
        distances=[3, 5],
        rounds=[1, 3],
        noise=[
            {"p_error": 0.001, "model": "simple_depolarizing"},
            {"p_error": 0.003, "model": "circuit_level"},
        ],
        basis=["X", "Z"],
        backends=["pymatching_baseline"],
        model_variants=["none"],
        export_modes=["none"],
    )
    pts = list(s.expand())
    assert len(pts) == 2 * 2 * 2 * 2  # 16
    assert all(isinstance(p, SweepPoint) for p in pts)


def test_expand_order_is_deterministic_across_invocations() -> None:
    s = _spec()
    a = [p.point_seed for p in s.expand()]
    b = [p.point_seed for p in s.expand()]
    c = [p.point_seed for p in SweepSpec(**s.model_dump()).expand()]
    assert a == b == c


def test_expand_point_seeds_differ_between_points() -> None:
    s = _spec()
    seeds = [p.point_seed for p in s.expand()]
    assert len(seeds) == len(set(seeds))  # all unique for this shape


def test_expand_point_seeds_change_when_master_seed_changes() -> None:
    s1 = _spec(master_seed=42)
    s2 = _spec(master_seed=43)
    seeds1 = [p.point_seed for p in s1.expand()]
    seeds2 = [p.point_seed for p in s2.expand()]
    assert seeds1 != seeds2


def test_canonical_hash_is_stable_and_sensitive() -> None:
    s1 = _spec()
    s2 = _spec()
    assert s1.canonical_hash() == s2.canonical_hash()
    # Any field change must change the hash.
    s3 = _spec(master_seed=43)
    assert s1.canonical_hash() != s3.canonical_hash()
    # Hash is a valid SHA256 hex digest.
    assert len(s1.canonical_hash()) == 64
    assert all(c in "0123456789abcdef" for c in s1.canonical_hash())


def test_canonical_hash_format_is_sha256_of_sorted_json() -> None:
    s = _spec()
    expected = __import__("hashlib").sha256(
        json.dumps(s.model_dump(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert s.canonical_hash() == expected


def test_expand_sorted_order_matches_declared_axis_priority() -> None:
    s = _spec(
        distances=[5, 3],  # reversed
        rounds=[3, 1],
        noise=[{"p_error": 0.001, "model": "simple_depolarizing"}],
        basis=["Z", "X"],  # reversed
        backends=["pymatching_baseline"],
        model_variants=["none"],
        export_modes=["none"],
    )
    pts = list(s.expand())
    # After sort, distance grows monotonically first.
    distances = [p.distance for p in pts]
    assert distances == sorted(distances)


def test_schema_version_is_pinned_to_one() -> None:
    with pytest.raises(ValidationError):
        _spec(schema_version="2")  # type: ignore[arg-type]
