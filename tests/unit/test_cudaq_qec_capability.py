"""Unit tests for CudaqQecCapability (T029)."""

from __future__ import annotations

from unittest import mock

from app.decoders import cudaq_qec_capability as target
from app.decoders.cudaq_qec_capability import CudaqQecCapability


def test_cudaq_qec_capability_reports_reason_when_unavailable() -> None:
    with (
        mock.patch.object(target, "CUDAQ_QEC_AVAILABLE", False),
        mock.patch.object(target, "CUDAQ_QEC_IMPORT_ERROR",
                          "ImportError: No module named 'cudaq_qec'"),
    ):
        rep = CudaqQecCapability().available()
    assert rep.is_available is False
    assert rep.reason  # non-empty
    assert "cudaq" in rep.reason.lower()


def test_cudaq_qec_capability_report_has_required_fields() -> None:
    # The contract: the returned CapabilityReport exposes
    # is_available, reason (str, non-empty), and we reuse the richer
    # core schema (blocker_category, required, detected_versions).
    rep = CudaqQecCapability().available()
    assert hasattr(rep, "is_available")
    assert hasattr(rep, "reason")
    assert isinstance(rep.reason, str)
    assert len(rep.reason) > 0
    assert hasattr(rep, "blocker_category")


def test_cudaq_qec_capability_distinguishes_import_error_from_runtime_error() -> None:
    # Force the probe to raise RuntimeError while keeping the package
    # imported. available() must classify this as runtime, not
    # not_installed.
    cap = CudaqQecCapability()
    with mock.patch.object(target, "CUDAQ_QEC_AVAILABLE", True), \
         mock.patch.object(cap, "probe",
                           side_effect=RuntimeError("simulated runtime blocker")):
        rep = cap.available()
    assert rep.is_available is False
    assert rep.blocker_category == "runtime"
    assert "runtime" in rep.reason.lower()

    # AttributeError (API drift) classifies as version_mismatch.
    with mock.patch.object(target, "CUDAQ_QEC_AVAILABLE", True), \
         mock.patch.object(cap, "probe",
                           side_effect=AttributeError("cudaq_qec.list_codes gone")):
        rep = cap.available()
    assert rep.is_available is False
    assert rep.blocker_category == "version_mismatch"
