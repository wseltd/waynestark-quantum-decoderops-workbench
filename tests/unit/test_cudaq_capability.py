"""Unit tests for CudaqCapabilityAdapter (T028)."""

from __future__ import annotations

from unittest import mock

import pytest

from app.decoders import cudaq_capability as target
from app.decoders.cudaq_capability import (
    CUDAQ_AVAILABLE,
    CudaqCapabilityAdapter,
    CudaqTargetInfo,
)


def test_cudaq_capability_module_exposes_availability_flag() -> None:
    assert isinstance(CUDAQ_AVAILABLE, bool)
    assert hasattr(target, "CUDAQ_VERSION")
    assert hasattr(target, "CUDAQ_IMPORT_ERROR")


def test_cudaq_capability_reports_missing_cudaq() -> None:
    with (
        mock.patch.object(target, "CUDAQ_AVAILABLE", False),
        mock.patch.object(target, "CUDAQ_IMPORT_ERROR",
                          "ImportError: No module named 'cudaq'"),
    ):
        rep = CudaqCapabilityAdapter().available()
    assert rep.is_available is False
    assert rep.blocker_category == "not_installed"
    assert "cudaq" in rep.reason.lower()
    # enumerate_targets must return [] rather than raising on missing
    # cudaq — consumers iterate over the list unconditionally.
    with mock.patch.object(target, "CUDAQ_AVAILABLE", False):
        assert CudaqCapabilityAdapter().enumerate_targets() == []


def test_cudaq_capability_enumerate_targets_returns_list() -> None:
    # With real cudaq installed in this venv, enumerate_targets returns
    # a list of CudaqTargetInfo records.
    if not CUDAQ_AVAILABLE:
        pytest.skip("cudaq not installed in this environment")
    targets = CudaqCapabilityAdapter().enumerate_targets()
    assert isinstance(targets, list)
    assert all(isinstance(t, CudaqTargetInfo) for t in targets)
    # Stable sorted ordering for artefact reproducibility.
    assert [t.name for t in targets] == sorted(t.name for t in targets)


def test_cudaq_capability_per_target_failure_isolated() -> None:
    # Force one target to fail set_target and verify the probe captures
    # that failure on its record without breaking the rest.
    if not CUDAQ_AVAILABLE:
        pytest.skip("cudaq not installed")
    import cudaq

    real_set_target = cudaq.set_target
    real_targets = cudaq.get_targets()
    # Pick a target name to fail; prefer one that's likely already
    # problematic in CI if any.
    break_name = real_targets[0].name

    def fake_set_target(name: str, *args, **kwargs):
        if name == break_name:
            raise RuntimeError(f"simulated failure for {name}")
        return real_set_target(name, *args, **kwargs)

    with mock.patch.object(cudaq, "set_target", side_effect=fake_set_target):
        results = CudaqCapabilityAdapter().enumerate_targets()

    by_name = {r.name: r for r in results}
    assert by_name[break_name].available is False
    assert "simulated failure" in (by_name[break_name].unavailable_reason or "")
    # Other targets must still be reported
    if len(real_targets) > 1:
        other = [t.name for t in real_targets if t.name != break_name][0]
        assert other in by_name


def test_cudaq_capability_restores_global_target_after_probe() -> None:
    if not CUDAQ_AVAILABLE:
        pytest.skip("cudaq not installed")
    import cudaq

    before = cudaq.get_target()
    _ = CudaqCapabilityAdapter().enumerate_targets()
    after = cudaq.get_target()
    # Identity or name match — cudaq.Target equality semantics vary, so
    # compare by name.
    assert getattr(before, "name", str(before)) == getattr(after, "name", str(after))
