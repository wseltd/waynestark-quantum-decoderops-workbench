"""Tests for app.metrics.logical_error_rate (T041)."""

from __future__ import annotations

import pytest

from app.metrics.logical_error_rate import (
    DEFAULT_SEED,
    LERResult,
    bootstrap_ci,
    compute_logical_error_rate,
)


def test_ler_basic_rate_matches_fraction() -> None:
    r = compute_logical_error_rate(50, 1000, seed=42)
    assert r.logical_error_rate == 0.05
    assert isinstance(r, LERResult)


def test_ler_bootstrap_ci_reproducible_with_seed() -> None:
    r1 = compute_logical_error_rate(50, 1000, seed=42)
    r2 = compute_logical_error_rate(50, 1000, seed=42)
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high


def test_ler_ci_contains_point_estimate() -> None:
    r = compute_logical_error_rate(50, 1000, seed=42)
    assert 0.0 <= r.ci_low <= r.logical_error_rate <= r.ci_high <= 1.0


def test_ler_zero_errors_returns_zero_rate_and_zero_low_ci() -> None:
    r = compute_logical_error_rate(0, 1000, seed=1)
    assert r.logical_error_rate == 0.0
    assert r.ci_low == 0.0


def test_ler_all_errors_returns_rate_one_and_high_ci_one() -> None:
    r = compute_logical_error_rate(100, 100, seed=1)
    assert r.logical_error_rate == 1.0
    assert r.ci_high == 1.0


def test_ler_zero_shots_raises_value_error() -> None:
    with pytest.raises(ValueError):
        compute_logical_error_rate(0, 0, seed=1)


def test_ler_confidence_0_99_wider_than_0_90() -> None:
    r90 = compute_logical_error_rate(50, 1000, confidence=0.90, seed=7)
    r99 = compute_logical_error_rate(50, 1000, confidence=0.99, seed=7)
    assert (r99.ci_high - r99.ci_low) >= (r90.ci_high - r90.ci_low)


def test_ler_result_is_frozen_pydantic_model() -> None:
    r = compute_logical_error_rate(5, 100, seed=1)
    with pytest.raises(Exception):
        r.logical_error_rate = 0.99  # type: ignore[misc]


def test_ler_default_seed_recorded_in_result() -> None:
    r = compute_logical_error_rate(10, 100)
    assert r.seed == DEFAULT_SEED


def test_bootstrap_ci_accepts_int_seed() -> None:
    lo, hi = bootstrap_ci(10, 100, 0.95, 500, 42)
    assert 0.0 <= lo <= hi <= 1.0
