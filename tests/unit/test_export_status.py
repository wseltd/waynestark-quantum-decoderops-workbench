"""Tests for app.metrics.export_status (T045)."""

from __future__ import annotations

import pytest

from app.metrics.export_status import ExportStatus, build_export_status


def test_build_export_status_onnx_success() -> None:
    s = build_export_status(
        "onnx",
        True,
        True,
        artefact_path="/tmp/x.onnx",
        duration_seconds=1.2,
        tool_version="1.24.4",
    )
    assert isinstance(s, ExportStatus)
    assert s.succeeded is True
    assert s.artefact_path.endswith(".onnx")


def test_build_export_status_tensorrt_failure_has_reason() -> None:
    s = build_export_status(
        "tensorrt",
        True,
        False,
        error_message="cuda not available",
    )
    assert not s.succeeded
    assert s.error_message == "cuda not available"


def test_build_export_status_rejects_success_without_artefact_path() -> None:
    with pytest.raises(ValueError):
        build_export_status("onnx", True, True)


def test_build_export_status_rejects_success_with_error_message() -> None:
    with pytest.raises(ValueError):
        build_export_status(
            "onnx", True, True, artefact_path="/a", error_message="no"
        )


def test_build_export_status_rejects_failure_without_error_message() -> None:
    with pytest.raises(ValueError):
        build_export_status("onnx", True, False)


def test_export_status_round_trips_through_model_dump() -> None:
    s = build_export_status(
        "onnx", True, True, artefact_path="/p.onnx", tool_version="1"
    )
    d = s.model_dump()
    s2 = ExportStatus.model_validate(d)
    assert s2 == s


def test_export_status_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        build_export_status("safetensors", True, True, artefact_path="/a")
