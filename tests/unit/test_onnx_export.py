"""Tests for app.packaging.onnx_export (T052)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.packaging.onnx_export import (
    DEFAULT_OPSET,
    OnnxExportResult,
    OnnxInputSignature,
    export_decoder_to_onnx,
)


class _PreOnnxDecoder:
    def __init__(self, blob: bytes) -> None:
        self._blob = blob

    def as_onnx_bytes(self) -> bytes:
        return self._blob


class _NoPathsDecoder:
    pass


def _sig() -> OnnxInputSignature:
    return OnnxInputSignature(name="x", shape=(1, 3, 9, 9), dtype="float32")


def test_export_decoder_to_onnx_writes_file_at_output_path(
    tmp_path: Path,
) -> None:
    out = tmp_path / "m.onnx"
    d = _PreOnnxDecoder(b"FAKEONNX")
    res = export_decoder_to_onnx(
        d, output_path=out, input_signature=_sig(), validate=False
    )
    assert out.exists()
    assert res.file_size_bytes == len(b"FAKEONNX")


def test_export_decoder_to_onnx_returns_sha256_matching_stamp_file(
    tmp_path: Path,
) -> None:
    out = tmp_path / "m.onnx"
    d = _PreOnnxDecoder(b"ABC")
    res = export_decoder_to_onnx(
        d, output_path=out, input_signature=_sig(), validate=False
    )
    assert res.sha256 == hashlib.sha256(b"ABC").hexdigest()


def test_export_decoder_to_onnx_writes_sidecar_sha256_file(
    tmp_path: Path,
) -> None:
    out = tmp_path / "m.onnx"
    d = _PreOnnxDecoder(b"ABC")
    res = export_decoder_to_onnx(
        d, output_path=out, input_signature=_sig(), validate=False
    )
    sha_side = out.with_suffix(".onnx.sha256")
    assert sha_side.exists()
    assert sha_side.read_text().strip() == res.sha256


def test_export_decoder_to_onnx_writes_sidecar_manifest_json(
    tmp_path: Path,
) -> None:
    out = tmp_path / "m.onnx"
    d = _PreOnnxDecoder(b"ABC")
    res = export_decoder_to_onnx(
        d, output_path=out, input_signature=_sig(), validate=False
    )
    manifest_side = out.with_suffix(".onnx.manifest.json")
    assert manifest_side.exists()
    loaded = json.loads(manifest_side.read_text())
    assert loaded["sha256"] == res.sha256


def test_export_decoder_to_onnx_records_opset_and_producer(
    tmp_path: Path,
) -> None:
    out = tmp_path / "m.onnx"
    d = _PreOnnxDecoder(b"ABC")
    res = export_decoder_to_onnx(
        d, output_path=out, input_signature=_sig(), validate=False
    )
    assert res.opset == DEFAULT_OPSET
    assert res.producer == "decoderops.onnx_export"


def test_export_decoder_to_onnx_validates_with_onnx_checker_when_validate_true(
    tmp_path: Path,
) -> None:
    # With a garbage blob, onnx.checker will fail; we still produce a record
    # with validated=False and notes explaining why.
    out = tmp_path / "m.onnx"
    d = _PreOnnxDecoder(b"not really onnx")
    res = export_decoder_to_onnx(
        d, output_path=out, input_signature=_sig(), validate=True
    )
    assert res.validated is False
    assert res.validation_notes


def test_export_decoder_to_onnx_skips_checker_when_validate_false(
    tmp_path: Path,
) -> None:
    out = tmp_path / "m.onnx"
    d = _PreOnnxDecoder(b"x")
    res = export_decoder_to_onnx(
        d, output_path=out, input_signature=_sig(), validate=False
    )
    assert res.validated is False
    assert res.validation_notes is None


def test_export_decoder_to_onnx_supports_torch_module_path(
    tmp_path: Path,
) -> None:
    # This requires torch; guard so the test still passes in torch-less envs.
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")

    class _M(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2

    class _ModuleDecoder:
        def torch_module(self) -> torch.nn.Module:
            return _M()

    sig = OnnxInputSignature(name="x", shape=(1, 4), dtype="float32")
    res = export_decoder_to_onnx(
        _ModuleDecoder(),
        output_path=tmp_path / "m.onnx",
        input_signature=sig,
        validate=False,
    )
    assert res.file_size_bytes > 0


def test_export_decoder_to_onnx_supports_preonnx_bytes_path(
    tmp_path: Path,
) -> None:
    out = tmp_path / "m.onnx"
    d = _PreOnnxDecoder(b"ABCD")
    res = export_decoder_to_onnx(
        d, output_path=out, input_signature=_sig(), validate=False
    )
    assert res.file_size_bytes == 4


def test_export_decoder_to_onnx_raises_when_decoder_exposes_neither_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        export_decoder_to_onnx(
            _NoPathsDecoder(),
            output_path=tmp_path / "m.onnx",
            input_signature=_sig(),
        )


def test_export_decoder_to_onnx_raises_runtimeerror_with_dependency_name_when_torch_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.packaging.onnx_export as mod

    def _raise() -> object:
        raise RuntimeError("torch is required for torch-module ONNX export")

    monkeypatch.setattr(mod, "_import_torch", _raise)

    class _ModuleOnly:
        def torch_module(self):
            return object()

    with pytest.raises(RuntimeError) as exc:
        export_decoder_to_onnx(
            _ModuleOnly(),
            output_path=tmp_path / "m.onnx",
            input_signature=_sig(),
        )
    assert "torch" in str(exc.value)
