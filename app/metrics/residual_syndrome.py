"""Residual syndrome density = post-predecoder activation rate (T042).

Definition (verbatim): residual syndrome density = post-predecoder activation
rate = mean of post_syndromes across all (shot, detector) cells.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

__all__ = [
    "MAX_VALIDATION_SAMPLES",
    "ResidualSyndromeDensity",
    "ResidualSyndromeResult",
    "compute_activation_rate",
    "compute_residual_syndrome_density",
]


MAX_VALIDATION_SAMPLES: int = 4096


class ResidualSyndromeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_shots: int
    num_detectors: int
    pre_activation_rate: float
    post_activation_rate: float
    residual_density: float
    reduction_ratio: float


ResidualSyndromeDensity = ResidualSyndromeResult


def _validate_binary(arr: np.ndarray, name: str) -> None:
    if arr.dtype == np.bool_:
        return
    if arr.dtype == np.uint8:
        # Sample up to MAX_VALIDATION_SAMPLES cells for 0/1 check.
        flat = arr.reshape(-1)
        if flat.size > MAX_VALIDATION_SAMPLES:
            idx = np.linspace(
                0, flat.size - 1, MAX_VALIDATION_SAMPLES, dtype=np.int64
            )
            sample = flat[idx]
        else:
            sample = flat
        if not np.all(np.isin(sample, np.array([0, 1], dtype=np.uint8))):
            raise ValueError(
                f"{name} uint8 values must be 0 or 1 only"
            )
        return
    raise TypeError(
        f"{name} must be bool or uint8 ndarray; got dtype={arr.dtype}"
    )


def compute_activation_rate(syndromes: np.ndarray) -> float:
    """Mean of a 2-D bool/uint8 array of syndrome bits."""
    if syndromes.ndim != 2:
        raise ValueError(f"expected 2-D array; got shape {syndromes.shape}")
    _validate_binary(syndromes, "syndromes")
    if syndromes.size == 0:
        return 0.0
    return float(syndromes.astype(np.float64).mean())


def compute_residual_syndrome_density(
    pre_syndromes: np.ndarray,
    post_syndromes: np.ndarray,
) -> ResidualSyndromeResult:
    if pre_syndromes.shape != post_syndromes.shape:
        raise ValueError(
            "pre/post syndrome shapes differ: "
            f"{pre_syndromes.shape} vs {post_syndromes.shape}"
        )
    if pre_syndromes.ndim != 2:
        raise ValueError(
            f"expected 2-D arrays; got shape {pre_syndromes.shape}"
        )
    _validate_binary(pre_syndromes, "pre_syndromes")
    _validate_binary(post_syndromes, "post_syndromes")

    num_shots, num_detectors = pre_syndromes.shape
    pre_rate = compute_activation_rate(pre_syndromes)
    post_rate = compute_activation_rate(post_syndromes)
    reduction = 0.0 if pre_rate == 0.0 else (1.0 - post_rate / pre_rate)
    return ResidualSyndromeResult(
        num_shots=int(num_shots),
        num_detectors=int(num_detectors),
        pre_activation_rate=pre_rate,
        post_activation_rate=post_rate,
        residual_density=post_rate,
        reduction_ratio=reduction,
    )
