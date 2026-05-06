"""Unit tests for CuQuantumCapability (T030)."""

from __future__ import annotations

from unittest import mock

from app.decoders import cuquantum_capability as target
from app.decoders.cuquantum_capability import CuQuantumCapability


def test_cuquantum_capability_reports_reason_when_unavailable() -> None:
    with (
        mock.patch.object(target, "CUQUANTUM_AVAILABLE", False),
        mock.patch.object(target, "CUQUANTUM_IMPORT_ERROR",
                          "ImportError: No module named 'cuquantum'"),
    ):
        rep = CuQuantumCapability().available()
    assert rep.is_available is False
    assert rep.blocker_category == "not_installed"
    assert "cuquantum" in rep.reason.lower()


def test_cuquantum_capability_report_has_required_fields() -> None:
    rep = CuQuantumCapability().available()
    assert hasattr(rep, "is_available")
    assert isinstance(rep.reason, str)
    assert len(rep.reason) > 0
    assert hasattr(rep, "blocker_category")


def test_cuquantum_capability_distinguishes_missing_package_from_missing_cuda() -> None:
    # (1) Package missing — blocker_category = not_installed
    with (
        mock.patch.object(target, "CUQUANTUM_AVAILABLE", False),
        mock.patch.object(target, "CUQUANTUM_IMPORT_ERROR",
                          "ImportError: cuquantum"),
    ):
        rep = CuQuantumCapability().available()
    assert rep.blocker_category == "not_installed"

    # (2) Package present but CUDA missing — force probe to return an
    # error mentioning cuda/device -> blocker_category = machine
    fake_details = {"error": "CUDARuntimeError: cuda device not available"}
    with (
        mock.patch.object(target, "CUQUANTUM_AVAILABLE", True),
        mock.patch.object(target, "CUQUANTUM_VERSION", "26.3.1"),
        mock.patch.object(CuQuantumCapability, "probe", return_value=fake_details),
    ):
        rep = CuQuantumCapability().available()
    assert rep.is_available is False
    assert rep.blocker_category == "machine"
    assert "cuda" in rep.reason.lower()
