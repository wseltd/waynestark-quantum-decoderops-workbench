"""Tests for app.metrics.compatibility (T046)."""

from __future__ import annotations

import pytest

from app.metrics.compatibility import (
    RuntimeCompatibilityStatus,
    build_compatibility_status,
)


def test_build_status_ready_pymatching() -> None:
    s = build_compatibility_status(
        "pymatching_baseline", "ready", "pymatching 2.x importable", "none"
    )
    assert s.status == "ready"
    assert s.category == "none"


def test_build_status_unavailable_tensorrt_machine() -> None:
    s = build_compatibility_status(
        "tensorrt_optional",
        "unavailable",
        "no NVIDIA GPU detected",
        "machine",
        "install a CUDA 13-capable GPU",
    )
    assert s.status == "unavailable"
    assert s.required_action is not None


def test_build_status_degraded_cudaq_software() -> None:
    s = build_compatibility_status(
        "cudaq", "degraded", "cudaq partial import", "software"
    )
    assert s.status == "degraded"


def test_ready_rejects_non_none_category() -> None:
    with pytest.raises(ValueError):
        build_compatibility_status(
            "tensorrt_optional", "ready", "ok", "machine"
        )


def test_unavailable_rejects_none_category() -> None:
    with pytest.raises(ValueError):
        build_compatibility_status(
            "tensorrt_optional", "unavailable", "gpu not here", "none"
        )


def test_unavailable_rejects_empty_reason() -> None:
    with pytest.raises(ValueError):
        build_compatibility_status(
            "tensorrt_optional", "unavailable", "", "software"
        )


def test_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        build_compatibility_status("some_backend", "ready", "ok", "none")


def test_rejects_unknown_status() -> None:
    with pytest.raises(ValueError):
        build_compatibility_status(
            "pymatching_baseline", "maybe", "ok", "none"
        )


def test_round_trips_through_model_dump() -> None:
    s = build_compatibility_status(
        "pymatching_baseline", "ready", "ready", "none"
    )
    d = s.model_dump()
    s2 = RuntimeCompatibilityStatus.model_validate(d)
    assert s2 == s
