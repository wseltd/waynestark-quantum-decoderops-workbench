"""Unit tests for OnnxValidationDecoder (T026)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from app.decoders import onnx_validation as target
from app.decoders.onnx_validation import (
    ONNXRUNTIME_AVAILABLE,
    OnnxValidationDecoder,
)


def test_onnx_validation_module_exposes_availability_flag() -> None:
    # Contract: downstream capability aggregation can read the flag at
    # import time without invoking available().
    assert isinstance(ONNXRUNTIME_AVAILABLE, bool)
    assert hasattr(target, "ONNXRUNTIME_IMPORT_ERROR")


def test_onnx_validation_available_reports_missing_onnxruntime(
    tmp_path: Path,
) -> None:
    # Force the module-level flag to False to simulate missing install
    # without removing the wheel from the venv. Restore after.
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake-onnx-bytes")
    d = OnnxValidationDecoder(model_path=model)
    with mock.patch.object(target, "ONNXRUNTIME_AVAILABLE", False), \
         mock.patch.object(target, "ONNXRUNTIME_IMPORT_ERROR",
                           "ImportError: No module named 'onnxruntime'"):
        rep = d.available()
    assert rep.is_available is False
    assert rep.blocker_category == "not_installed"
    assert "onnxruntime" in rep.reason


def test_onnx_validation_available_reports_missing_model_file(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "no-model-here.onnx"
    d = OnnxValidationDecoder(model_path=missing)
    rep = d.available()
    assert rep.is_available is False
    assert str(missing) in rep.reason
    assert rep.blocker_category == "software"


def test_onnx_validation_available_reports_unregistered_provider(
    tmp_path: Path,
) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    d = OnnxValidationDecoder(
        model_path=model, providers=["FakeTotallyInventedEP"]
    )
    with mock.patch("onnxruntime.get_available_providers",
                    return_value=["CPUExecutionProvider"]):
        rep = d.available()
    assert rep.is_available is False
    # No requested provider matches registered — precise reason naming
    # both sets in the message.
    assert "FakeTotallyInventedEP" in rep.reason
    assert "CPUExecutionProvider" in rep.reason


def test_onnx_validation_metadata_records_active_providers(
    tmp_path: Path,
) -> None:
    # Build a tiny real ONNX model (1-op Identity) via the onnx python
    # API and actually load it through onnxruntime so active_providers
    # is the real registered list, not a mock.
    if not ONNXRUNTIME_AVAILABLE:
        pytest.skip("onnxruntime not available in this env")

    import onnx
    from onnx import TensorProto, helper

    model = tmp_path / "identity.onnx"
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "identity_graph", [inp], [out])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 7
    onnx.save(m, str(model))

    d = OnnxValidationDecoder(
        model_path=model, providers=["CPUExecutionProvider"]
    )
    d.warmup()
    md = d.metadata()
    assert md.backend_name == "onnx_validation"
    assert md.model_path is not None
    assert md.model_sha256 is not None
    # active_providers is a superset of what we requested when ORT
    # injects the default CPU provider automatically; we only check
    # that something meaningful was recorded.
    assert d._active_providers  # non-empty
    assert md.supports_batching is True
    assert md.schema_version == "1"
