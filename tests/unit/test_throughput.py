"""Tests for app.metrics.throughput (T044)."""

from __future__ import annotations

import pytest

from app.metrics.throughput import (
    NS_PER_SECOND,
    ThroughputResult,
    compute_throughput,
)


def test_throughput_basic_shots_and_rounds_per_second() -> None:
    r = compute_throughput(
        num_shots=1000, num_rounds_per_shot=10, total_elapsed_ns=2_000_000_000
    )
    assert r.shots_per_second == 500.0
    assert r.rounds_per_second == 5000.0


def test_throughput_total_rounds_equals_shots_times_rounds_per_shot() -> None:
    r = compute_throughput(
        num_shots=5, num_rounds_per_shot=7, total_elapsed_ns=1_000_000_000
    )
    assert r.total_rounds == 35


def test_throughput_ns_to_seconds_conversion() -> None:
    r = compute_throughput(1, 1, 2_500_000_000)
    assert r.total_elapsed_seconds == 2.5


def test_throughput_zero_shots_raises_value_error() -> None:
    with pytest.raises(ValueError):
        compute_throughput(0, 10, 1_000_000_000)


def test_throughput_zero_rounds_per_shot_raises_value_error() -> None:
    with pytest.raises(ValueError):
        compute_throughput(10, 0, 1_000_000_000)


def test_throughput_zero_elapsed_raises_value_error() -> None:
    with pytest.raises(ValueError):
        compute_throughput(10, 1, 0)


def test_throughput_result_is_frozen_pydantic_model() -> None:
    r = compute_throughput(1, 1, 1_000_000_000)
    assert isinstance(r, ThroughputResult)
    with pytest.raises(Exception):
        r.shots_per_second = 0.0  # type: ignore[misc]


def test_ns_per_second_constant_is_one_billion() -> None:
    assert NS_PER_SECOND == 1_000_000_000
