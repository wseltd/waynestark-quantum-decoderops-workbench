"""ONNX export path smoke (T182)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.packaging.onnx_export import OnnxInputSignature, export_decoder_to_onnx


class _BlobDecoder:
    def __init__(self, blob: bytes) -> None:
        self._blob = blob

    def as_onnx_bytes(self) -> bytes:
        return self._blob


def test_export_roundtrip_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "m.onnx"
    res = export_decoder_to_onnx(
        _BlobDecoder(b"minimal-onnx-bytes"),
        output_path=out,
        input_signature=OnnxInputSignature(name="x", shape=(1, 4), dtype="float32"),
        validate=False,
    )
    assert out.exists()
    assert res.file_size_bytes > 0


def test_export_optional_onnxruntime_round_trip(tmp_path: Path) -> None:
    """If onnxruntime + a real ONNX model are available, load and run it."""
    try:
        import onnx
        import onnxruntime as ort
        from onnx import helper, TensorProto
    except ImportError:
        pytest.skip("onnx/onnxruntime not available")

    # Build an in-memory identity model
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "id", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx_bytes = model.SerializeToString()

    out = tmp_path / "id.onnx"
    export_decoder_to_onnx(
        _BlobDecoder(onnx_bytes),
        output_path=out,
        input_signature=OnnxInputSignature(name="x", shape=(1, 4), dtype="float32"),
        validate=True,
    )
    session = ort.InferenceSession(
        str(out), providers=["CPUExecutionProvider"]
    )
    import numpy as np

    inp = np.ones((1, 4), dtype=np.float32)
    result = session.run(None, {"x": inp})
    assert result[0].shape == (1, 4)
