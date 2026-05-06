"""Latency percentiles + histogram from int64 ns arrays (T043)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

__all__ = [
    "DEFAULT_NUM_BINS",
    "LatencyHistogram",
    "LatencyStats",
    "P50",
    "P95",
    "P99",
    "compute_histogram",
    "compute_latency_stats",
]


P50: int = 50
P95: int = 95
P99: int = 99
DEFAULT_NUM_BINS: int = 50

_ALLOWED_SCOPES = ("per_shot", "per_round")


class LatencyStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope: str
    count: int
    p50_ns: int
    p95_ns: int
    p99_ns: int
    min_ns: int
    max_ns: int
    mean_ns: float
    stddev_ns: float


class LatencyHistogram(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_bins: int
    bin_edges_ns: list[int]
    bin_counts: list[int]
    total: int


def _check_int64(arr: np.ndarray, name: str) -> None:
    if arr.dtype != np.int64:
        raise TypeError(f"{name} must be int64; got dtype={arr.dtype}")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")


def compute_latency_stats(
    latencies_ns: np.ndarray,
    scope: Literal["per_shot", "per_round"],
) -> LatencyStats:
    if scope not in _ALLOWED_SCOPES:
        raise ValueError(
            f"scope must be one of {_ALLOWED_SCOPES}; got {scope!r}"
        )
    _check_int64(latencies_ns, "latencies_ns")
    p50 = int(np.percentile(latencies_ns, P50, method="linear"))
    p95 = int(np.percentile(latencies_ns, P95, method="linear"))
    p99 = int(np.percentile(latencies_ns, P99, method="linear"))
    return LatencyStats(
        scope=scope,
        count=int(latencies_ns.size),
        p50_ns=p50,
        p95_ns=p95,
        p99_ns=p99,
        min_ns=int(latencies_ns.min()),
        max_ns=int(latencies_ns.max()),
        mean_ns=float(latencies_ns.mean()),
        stddev_ns=float(latencies_ns.std(ddof=0)),
    )


def compute_histogram(
    latencies_ns: np.ndarray, num_bins: int = DEFAULT_NUM_BINS
) -> LatencyHistogram:
    _check_int64(latencies_ns, "latencies_ns")
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1; got {num_bins}")
    lo = int(latencies_ns.min())
    hi = int(latencies_ns.max())
    if lo == hi:
        # Single value — use a minimal [lo, lo+1] range; all counts in bin 0.
        hi = lo + 1
    edges = np.linspace(lo, hi, num_bins + 1)
    counts, _ = np.histogram(latencies_ns, bins=edges)
    # Last edge inclusive: np.histogram handles this by default via the
    # right-edge inclusion rule of the last bin.
    return LatencyHistogram(
        num_bins=num_bins,
        bin_edges_ns=[int(round(e)) for e in edges],
        bin_counts=[int(c) for c in counts],
        total=int(counts.sum()),
    )
