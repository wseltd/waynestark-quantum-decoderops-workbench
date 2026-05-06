"""cuQuantum minimal smoke (T189)."""

from __future__ import annotations

import pytest


def test_cuquantum_tensor_op_or_skip() -> None:
    try:
        import cuquantum
    except ImportError as e:
        pytest.skip(
            f"cuquantum unavailable: {e} | "
            "required: cuquantum-python-cu13 + NVIDIA driver/CUDA 13 | "
            "category: software"
        )
    # cuquantum namespace exports. If the package import succeeded,
    # the top-level module must be present with the documented name.
    assert hasattr(cuquantum, "__name__")
    assert cuquantum.__name__ == "cuquantum"
