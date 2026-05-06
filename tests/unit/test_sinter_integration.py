"""Tests for app.benchmarking.sinter_integration (T036)."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from app.benchmarking.sinter_integration import (
    SUPPORTED_DECODERS,
    SinterDecoderUnsupported,
    SinterLERResult,
    run_sinter_ler,
    wilson_ci,
)


def test_wilson_ci_matches_reference_values() -> None:
    # Reference: 50/100 @ z=1.96 → roughly (0.404, 0.596) per standard tables.
    lo, hi = wilson_ci(50, 100, z=1.96)
    assert abs(lo - 0.4038) < 0.005
    assert abs(hi - 0.5962) < 0.005


def test_wilson_ci_zero_errors_lower_bound_is_zero() -> None:
    lo, hi = wilson_ci(0, 1000)
    assert lo == 0.0
    assert 0.0 < hi < 0.01


def test_wilson_ci_all_errors_upper_bound_is_one() -> None:
    lo, hi = wilson_ci(100, 100)
    assert hi == 1.0
    assert 0.9 < lo < 1.0


def test_wilson_ci_is_narrower_than_point_estimate_range() -> None:
    # 5/100: point estimate 0.05; Wilson CI must be narrower than [0, 1].
    lo, hi = wilson_ci(5, 100)
    assert hi - lo < 1.0
    assert lo < 0.05 < hi


def test_wilson_ci_symmetric_at_half() -> None:
    lo, hi = wilson_ci(50, 100)
    # Centre is shifted slightly by Wilson but symmetric around 0.5
    # within numerical tolerance.
    mid = (lo + hi) / 2
    assert abs(mid - 0.5) < 0.02


def test_wilson_ci_shots_zero_returns_unit_interval() -> None:
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_rejects_negative_args() -> None:
    with pytest.raises(ValueError):
        wilson_ci(-1, 10)
    with pytest.raises(ValueError):
        wilson_ci(1, -5)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def test_run_sinter_ler_rejects_ising_fast_with_precise_message() -> None:
    with pytest.raises(SinterDecoderUnsupported) as exc:
        run_sinter_ler(circuit=None, decoder="ising_fast", shots=10)
    assert "ising_fast" in str(exc.value)
    assert exc.value.decoder == "ising_fast"


def test_run_sinter_ler_rejects_tensorrt_optional() -> None:
    with pytest.raises(SinterDecoderUnsupported):
        run_sinter_ler(circuit=None, decoder="tensorrt_optional", shots=10)


def test_run_sinter_ler_rejects_ising_accurate() -> None:
    with pytest.raises(SinterDecoderUnsupported):
        run_sinter_ler(circuit=None, decoder="ising_accurate", shots=10)


def test_run_sinter_ler_aggregates_task_stats() -> None:
    class _FakeStat:
        def __init__(self, shots: int, errors: int, seconds: float = 0.1) -> None:
            self.shots = shots
            self.errors = errors
            self.discards = 0
            self.seconds = seconds

    fake_stats = [_FakeStat(shots=600, errors=12), _FakeStat(shots=400, errors=8)]

    with patch("sinter.collect", return_value=fake_stats), patch(
        "sinter.Task", return_value=object()
    ):
        r = run_sinter_ler(circuit=object(), decoder="pymatching", shots=1000)
    assert r.shots == 1000
    assert r.errors == 20
    assert math.isclose(r.ler, 0.020, abs_tol=1e-9)
    assert r.ci_method == "wilson_95"
    assert len(r.raw_task_stats) == 2


def test_sinter_ler_result_fields_present() -> None:
    r = SinterLERResult(
        decoder="pymatching",
        shots=100,
        errors=5,
        ler=0.05,
        ci_low=0.01,
        ci_high=0.11,
        ci_method="wilson_95",
        seconds=0.1,
        raw_task_stats=[],
    )
    for fld in (
        "decoder",
        "shots",
        "errors",
        "ler",
        "ci_low",
        "ci_high",
        "ci_method",
        "seconds",
        "raw_task_stats",
    ):
        assert hasattr(r, fld)


def test_supported_decoders_contract() -> None:
    assert "pymatching" in SUPPORTED_DECODERS
