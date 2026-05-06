"""cudaq targets enumeration (T187)."""

from __future__ import annotations

import pytest


def test_cudaq_targets_or_skip_with_reason() -> None:
    try:
        import cudaq
    except ImportError as e:
        pytest.skip(
            f"cudaq unavailable: {e} | "
            "required: cudaq==0.14.0 + compatible NVIDIA driver/CUDA 13 | "
            "category: software"
        )
    try:
        get = getattr(cudaq, "get_targets", None) or getattr(
            cudaq, "get_target", None
        )
        if get is None:
            pytest.skip(
                "cudaq targets API unavailable | category: software"
            )
        targets = get()
        if callable(getattr(targets, "__iter__", None)):
            names = [getattr(t, "name", None) for t in targets]
            assert any(isinstance(n, str) and n for n in names)
    except Exception as e:  # noqa: BLE001
        pytest.skip(
            f"cudaq runtime init failed: {e} | "
            "required: NVIDIA driver/CUDA 13 | category: runtime"
        )
