"""Tests for app.metrics.latency (T043)."""

from __future__ import annotations

import numpy as np
import pytest

from app.metrics.latency import (
    LatencyHistogram,
    LatencyStats,
    compute_histogram,
    compute_latency_stats,
)


def test_latency_p50_p95_p99_on_linear_range() -> None:
    lat = np.arange(1, 101, dtype=np.int64)
    s = compute_latency_stats(lat, scope="per_shot")
    assert s.p50_ns == 50
    assert s.p95_ns == 95
    assert s.p99_ns == 99
    assert s.min_ns == 1
    assert s.max_ns == 100


def test_latency_identical_values_all_percentiles_equal() -> None:
    lat = np.full(50, 1000, dtype=np.int64)
    s = compute_latency_stats(lat, scope="per_shot")
    assert s.p50_ns == s.p95_ns == s.p99_ns == 1000


def test_latency_per_round_scope_recorded() -> None:
    lat = np.arange(1, 11, dtype=np.int64)
    s = compute_latency_stats(lat, scope="per_round")
    assert s.scope == "per_round"


def test_latency_per_shot_scope_recorded() -> None:
    lat = np.arange(1, 11, dtype=np.int64)
    s = compute_latency_stats(lat, scope="per_shot")
    assert s.scope == "per_shot"


def test_latency_empty_array_raises_value_error() -> None:
    with pytest.raises(ValueError):
        compute_latency_stats(
            np.array([], dtype=np.int64), scope="per_shot"
        )


def test_latency_float_dtype_raises_type_error() -> None:
    with pytest.raises(TypeError):
        compute_latency_stats(
            np.array([1.0, 2.0, 3.0]), scope="per_shot"
        )


def test_latency_reproducible_byte_identical_percentiles() -> None:
    lat = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int64)
    a = compute_latency_stats(lat, scope="per_shot")
    b = compute_latency_stats(lat, scope="per_shot")
    assert a.p50_ns == b.p50_ns
    assert a.p95_ns == b.p95_ns
    assert a.p99_ns == b.p99_ns


def test_histogram_bin_edges_length_num_bins_plus_one() -> None:
    lat = np.arange(1, 101, dtype=np.int64)
    h = compute_histogram(lat, num_bins=10)
    assert h.num_bins == 10
    assert len(h.bin_edges_ns) == 11
    assert len(h.bin_counts) == 10


def test_histogram_counts_sum_equals_input_length() -> None:
    lat = np.arange(1, 101, dtype=np.int64)
    h = compute_histogram(lat, num_bins=10)
    assert sum(h.bin_counts) == 100
    assert isinstance(h, LatencyHistogram)


def test_histogram_invalid_scope_literal_raises_value_error() -> None:
    lat = np.arange(1, 11, dtype=np.int64)
    with pytest.raises(ValueError):
        compute_latency_stats(lat, scope="per_batch")  # type: ignore[arg-type]
