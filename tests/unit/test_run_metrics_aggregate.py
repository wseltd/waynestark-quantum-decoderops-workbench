"""Tests for app.metrics.aggregate (T047)."""

from __future__ import annotations

import numpy as np
import pytest

from app.metrics.aggregate import RunMetrics, build_run_metrics
from app.metrics.compatibility import build_compatibility_status
from app.metrics.export_status import build_export_status
from app.metrics.latency import compute_histogram
from app.metrics.logical_error_rate import compute_logical_error_rate
from app.metrics.residual_syndrome import compute_residual_syndrome_density
from app.metrics.throughput import compute_throughput


def _good_latency():
    return compute_histogram(
        np.arange(1, 101, dtype=np.int64), num_bins=10
    )


def _good_kit():
    ler = compute_logical_error_rate(50, 1000, seed=1)
    pre = np.ones((10, 5), dtype=np.uint8)
    post = np.zeros((10, 5), dtype=np.uint8)
    res = compute_residual_syndrome_density(pre, post)
    latency = _good_latency()
    thr = compute_throughput(1000, 1, 1_000_000_000)
    exports = [
        build_export_status("onnx", True, True, artefact_path="/a.onnx")
    ]
    compat = [
        build_compatibility_status(
            "pymatching_baseline", "ready", "ready", "none"
        )
    ]
    return ler, res, latency, thr, exports, compat


def test_build_run_metrics_happy_path() -> None:
    ler, res, latency, thr, exports, compat = _good_kit()
    m = build_run_metrics(
        "abc123", ler, res, latency, thr, exports, compat
    )
    assert isinstance(m, RunMetrics)
    assert m.schema_version == "1"


def test_run_metrics_schema_version_is_pinned_to_1() -> None:
    ler, res, latency, thr, exports, compat = _good_kit()
    m = build_run_metrics("r", ler, res, latency, thr, exports, compat)
    assert m.schema_version == "1"


def test_run_metrics_rejects_empty_run_id() -> None:
    ler, res, latency, thr, exports, compat = _good_kit()
    with pytest.raises(ValueError):
        build_run_metrics("", ler, res, latency, thr, exports, compat)


def test_run_metrics_rejects_empty_compatibility_list() -> None:
    ler, res, latency, thr, exports, _ = _good_kit()
    with pytest.raises(ValueError):
        build_run_metrics("r", ler, res, latency, thr, exports, [])


def test_run_metrics_rejects_wrong_logical_error_rate_type() -> None:
    _, res, latency, thr, exports, compat = _good_kit()
    with pytest.raises(Exception):
        build_run_metrics(
            "r",
            "not a model",  # type: ignore[arg-type]
            res,
            latency,
            thr,
            exports,
            compat,
        )


def test_run_metrics_rejects_wrong_latency_type() -> None:
    ler, res, _, thr, exports, compat = _good_kit()
    with pytest.raises(Exception):
        build_run_metrics(
            "r",
            ler,
            res,
            "not a latency",  # type: ignore[arg-type]
            thr,
            exports,
            compat,
        )


def test_run_metrics_accepts_zero_exports() -> None:
    ler, res, latency, thr, _, compat = _good_kit()
    m = build_run_metrics("r", ler, res, latency, thr, [], compat)
    assert m.exports == []


def test_run_metrics_round_trips_through_model_dump_json() -> None:
    ler, res, latency, thr, exports, compat = _good_kit()
    m = build_run_metrics("r", ler, res, latency, thr, exports, compat)
    j = m.model_dump_json()
    m2 = RunMetrics.model_validate_json(j)
    assert m2 == m


def test_run_metrics_preserves_all_nested_fields_through_dump_and_load() -> None:
    ler, res, latency, thr, exports, compat = _good_kit()
    m = build_run_metrics("r", ler, res, latency, thr, exports, compat)
    d = m.model_dump()
    m2 = RunMetrics.model_validate(d)
    assert m2.logical_error_rate == m.logical_error_rate
    assert m2.residual_syndrome_density == m.residual_syndrome_density
