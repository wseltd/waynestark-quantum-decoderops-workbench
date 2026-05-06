"""onnxruntime provider smoke (T190)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "provider", ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
)
def test_ort_provider_available_or_skip(provider: str) -> None:
    try:
        import onnxruntime as ort
    except ImportError as e:
        pytest.skip(
            f"onnxruntime unavailable: {e} | required: onnxruntime-gpu | "
            "category: software"
        )
    available = set(ort.get_available_providers())
    if provider not in available:
        pytest.skip(
            f"ort {provider} unavailable: not in providers={sorted(available)} | "
            "required: onnxruntime-gpu + CUDA/TensorRT | category: machine"
        )
    assert provider in available
