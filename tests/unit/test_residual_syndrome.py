"""Tests for app.metrics.residual_syndrome (T042)."""

from __future__ import annotations

import numpy as np
import pytest

from app.metrics.residual_syndrome import (
    ResidualSyndromeResult,
    compute_activation_rate,
    compute_residual_syndrome_density,
)


def test_residual_density_equals_post_activation_rate() -> None:
    pre = np.ones((100, 20), dtype=np.uint8)
    post = np.zeros((100, 20), dtype=np.uint8)
    post[:10] = 1
    r = compute_residual_syndrome_density(pre, post)
    assert r.residual_density == r.post_activation_rate


def test_activation_rate_all_zeros_is_zero() -> None:
    arr = np.zeros((10, 5), dtype=np.uint8)
    assert compute_activation_rate(arr) == 0.0


def test_activation_rate_all_ones_is_one() -> None:
    arr = np.ones((10, 5), dtype=np.uint8)
    assert compute_activation_rate(arr) == 1.0


def test_reduction_ratio_when_pre_zero_is_zero() -> None:
    pre = np.zeros((5, 3), dtype=np.uint8)
    post = np.zeros((5, 3), dtype=np.uint8)
    r = compute_residual_syndrome_density(pre, post)
    assert r.reduction_ratio == 0.0


def test_shape_mismatch_raises_value_error() -> None:
    pre = np.zeros((5, 3), dtype=np.uint8)
    post = np.zeros((4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_residual_syndrome_density(pre, post)


def test_float_dtype_raises_type_error() -> None:
    pre = np.zeros((5, 3), dtype=np.float32)
    post = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        compute_residual_syndrome_density(pre, post)


def test_bool_dtype_accepted() -> None:
    pre = np.ones((5, 3), dtype=bool)
    post = np.zeros((5, 3), dtype=bool)
    r = compute_residual_syndrome_density(pre, post)
    assert r.post_activation_rate == 0.0


def test_uint8_values_outside_0_1_raises_value_error() -> None:
    pre = np.zeros((5, 3), dtype=np.uint8)
    post = np.zeros((5, 3), dtype=np.uint8)
    post[0, 0] = 2
    with pytest.raises(ValueError):
        compute_residual_syndrome_density(pre, post)


def test_result_is_frozen_pydantic_model() -> None:
    pre = np.ones((5, 3), dtype=np.uint8)
    post = np.zeros((5, 3), dtype=np.uint8)
    r = compute_residual_syndrome_density(pre, post)
    assert isinstance(r, ResidualSyndromeResult)
    with pytest.raises(Exception):
        r.residual_density = 0.5  # type: ignore[misc]
