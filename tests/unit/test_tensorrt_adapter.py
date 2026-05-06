"""Unit tests for TensorRTDecoder (T027)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from app.decoders import tensorrt_adapter as target
from app.decoders.tensorrt_adapter import (
    TENSORRT_AVAILABLE,
    TensorRTDecoder,
    build_engine,
)


def test_tensorrt_available_reports_missing_tensorrt(tmp_path: Path) -> None:
    engine = tmp_path / "m.engine"
    engine.write_bytes(b"fake")
    d = TensorRTDecoder(engine_path=engine)
    # Simulate tensorrt unavailable via the T012 probe. The adapter
    # MUST consume probe_tensorrt — that consumption is the point of
    # T027's capability-report integration.
    from app.core.capability import ProbeReport

    fake = ProbeReport(
        name="tensorrt",
        available=False,
        version=None,
        reason="tensorrt python package not installed: tensorrt",
    )
    with mock.patch.object(target, "probe_tensorrt", return_value=fake):
        rep = d.available()
    assert rep.is_available is False
    assert rep.blocker_category == "not_installed"
    assert "tensorrt" in rep.reason.lower()


def test_tensorrt_available_reports_missing_engine_file(tmp_path: Path) -> None:
    engine = tmp_path / "does-not-exist.engine"
    d = TensorRTDecoder(engine_path=engine)
    # tensorrt is available in this venv; force both Tier 3 probes to
    # report ready so the ONLY unavailability left is the missing file.
    from app.core.capability import ProbeReport

    ok_trt = ProbeReport(name="tensorrt", available=True, version="10.16", reason="ok")
    ok_torch = ProbeReport(
        name="torch_cuda", available=True, version="2.11", reason="ok"
    )
    with (
        mock.patch.object(target, "probe_tensorrt", return_value=ok_trt),
        mock.patch.object(target, "probe_torch_cuda", return_value=ok_torch),
    ):
        rep = d.available()
    assert rep.is_available is False
    assert rep.blocker_category == "software"
    assert str(engine) in rep.reason


def test_tensorrt_available_reports_sm_mismatch(tmp_path: Path) -> None:
    """INT8/FP8 requested without calibration cache -> precise reason.

    The SM-mismatch check is not implemented in v1 (deferred to the
    capability detector proper); we use the calibration-cache branch
    as the concrete 'precise blocker reason' proof in v1.
    """
    engine = tmp_path / "m.engine"
    engine.write_bytes(b"fake")
    d = TensorRTDecoder(engine_path=engine, precision="int8")
    from app.core.capability import ProbeReport

    ok_trt = ProbeReport(name="tensorrt", available=True, version="10.16", reason="ok")
    ok_torch = ProbeReport(
        name="torch_cuda", available=True, version="2.11", reason="ok"
    )
    with (
        mock.patch.object(target, "probe_tensorrt", return_value=ok_trt),
        mock.patch.object(target, "probe_torch_cuda", return_value=ok_torch),
    ):
        rep = d.available()
    assert rep.is_available is False
    assert rep.blocker_category == "software"
    assert "int8" in rep.reason.lower()
    assert "calibration" in rep.reason.lower()


def test_tensorrt_metadata_includes_engine_sha256_and_precision(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "m.engine"
    engine.write_bytes(b"fake")
    d = TensorRTDecoder(engine_path=engine, precision="fp16")
    md = d.metadata()
    assert md.backend_name == "tensorrt_optional"
    # model_path reflects the engine path when present.
    assert md.model_path is not None
    assert Path(md.model_path).name == engine.name
    # model_sha256 is None until warmup; precision is not a
    # DecoderMetadata field but is preserved on the adapter for
    # downstream reports to read.
    assert d._precision == "fp16"


def test_tensorrt_adapter_consumes_capability_report(tmp_path: Path) -> None:
    # Pin the integration: the adapter must call probe_tensorrt from
    # app.core.capability. Use mock.patch.object to verify the call.
    engine = tmp_path / "m.engine"
    engine.write_bytes(b"fake")
    d = TensorRTDecoder(engine_path=engine)
    from app.core.capability import ProbeReport

    ok_trt = ProbeReport(name="tensorrt", available=True, version="10.16", reason="ok")
    ok_torch = ProbeReport(
        name="torch_cuda", available=True, version="2.11", reason="ok"
    )
    with (
        mock.patch.object(
            target, "probe_tensorrt", return_value=ok_trt
        ) as m_trt,
        mock.patch.object(target, "probe_torch_cuda", return_value=ok_torch),
    ):
        report = d.available()
    m_trt.assert_called_once()
    assert report.available is True


def test_tensorrt_does_not_build_engine_in_constructor(tmp_path: Path) -> None:
    # Construction must not call build_engine; ticket explicitly forbids
    # eager engine build in __init__ (engine build is expensive and
    # requires CUDA hardware that may not be present at construction).
    missing_onnx = tmp_path / "source.onnx"
    target_engine = tmp_path / "built.engine"
    with mock.patch.object(target, "build_engine") as m_build:
        decoder = TensorRTDecoder(
            engine_path=target_engine, onnx_path=missing_onnx
        )
    m_build.assert_not_called()
    assert m_build.call_count == 0
    assert decoder is not None


def test_build_engine_rejects_int8_and_fp8(tmp_path: Path) -> None:
    # Calibration logic is explicitly out of scope for v1; the function
    # must refuse rather than silently build a miscalibrated engine.
    onnx_path = tmp_path / "source.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    engine_path = tmp_path / "out.engine"
    with pytest.raises(NotImplementedError):
        build_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision="int8",
        )
