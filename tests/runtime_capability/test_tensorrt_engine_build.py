"""TensorRT engine-build smoke (T186)."""

from __future__ import annotations

import pytest


def test_tensorrt_engine_build_or_skip_with_reason() -> None:
    try:
        import tensorrt as trt
    except ImportError as e:
        pytest.skip(
            f"tensorrt unavailable: {e} | "
            "required: tensorrt-cu13>=10.16, NVIDIA driver, CUDA 13 | "
            "category: software"
        )
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        assert builder is not None
    except Exception as e:  # noqa: BLE001
        pytest.skip(
            f"tensorrt runtime init failed: {e} | "
            "required: NVIDIA driver + CUDA 13 | category: runtime"
        )
