"""Tests for app.packaging.tensorrt_engine_meta (T053)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.packaging.tensorrt_engine_meta import (
    skip_reason,
    write_tensorrt_engine_meta,
)


def test_write_tensorrt_engine_meta_emits_sha256_and_versions(
    tmp_path: Path,
) -> None:
    e = tmp_path / "x.engine"
    e.write_bytes(b"abc")
    out = write_tensorrt_engine_meta(
        e,
        {
            "trt_version": "10.16.1.11",
            "cuda_version": "13.0",
            "precision_mode": "fp16",
            "input_shapes": {"x": [1, 9, 9]},
            "output_shapes": {"y": [1, 2]},
            "builder_flags": ["FP16"],
            "source_onnx_sha256": "0" * 64,
            "build_timestamp_utc": "2026-01-01T00:00:00Z",
        },
        tmp_path,
    )
    data = json.loads(out.read_text())
    assert data["engine_sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert data["trt_version"] == "10.16.1.11"
    assert data["cuda_version"] == "13.0"


def test_skip_reason_returns_precise_string_when_adapter_unavailable() -> None:
    assert skip_reason({"available": False, "reason": "no trt"}) == "no trt"
    # empty/missing reason
    s = skip_reason({"available": False})
    assert s and isinstance(s, str)


def test_write_tensorrt_engine_meta_is_deterministic_for_same_inputs(
    tmp_path: Path,
) -> None:
    e = tmp_path / "a.engine"
    e.write_bytes(b"hello world")
    md = {
        "trt_version": "10",
        "cuda_version": "13",
        "precision_mode": "fp16",
        "input_shapes": {"x": [1]},
        "output_shapes": {"y": [1]},
        "builder_flags": [],
        "source_onnx_sha256": "0" * 64,
        "build_timestamp_utc": "2026-01-01T00:00:00Z",
    }
    a = write_tensorrt_engine_meta(e, md, tmp_path / "a")
    b = write_tensorrt_engine_meta(e, md, tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()


def test_write_tensorrt_engine_meta_accepts_custom_output_dir(
    tmp_path: Path,
) -> None:
    e = tmp_path / "a.engine"
    e.write_bytes(b"eng")
    out_dir = tmp_path / "deep" / "nested"
    out = write_tensorrt_engine_meta(e, {}, out_dir)
    assert out.parent == out_dir
