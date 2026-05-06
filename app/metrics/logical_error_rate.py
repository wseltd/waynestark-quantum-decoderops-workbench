"""Logical error rate with bootstrap CIs (T041)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

__all__ = [
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_SEED",
    "LERResult",
    "LogicalErrorRate",
    "bootstrap_ci",
    "compute_logical_error_rate",
]

DEFAULT_CONFIDENCE: float = 0.95
DEFAULT_BOOTSTRAP: int = 1000
DEFAULT_SEED: int = 0xDEC0DE


class LERResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_error_rate: float
    num_errors: int
    num_shots: int
    confidence: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    seed: int


# Alias for T047 aggregate schema naming.
LogicalErrorRate = LERResult


def bootstrap_ci(
    num_errors: int,
    num_shots: int,
    confidence: float,
    n_bootstrap: int,
    seed: int | None,
) -> tuple[float, float]:
    if num_shots <= 0:
        raise ValueError(f"num_shots must be > 0; got {num_shots}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0,1); got {confidence}")
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be > 0; got {n_bootstrap}")
    used_seed = seed if seed is not None else DEFAULT_SEED
    rng = np.random.default_rng(used_seed)
    p = num_errors / num_shots
    # Resample bernoulli draws, compute sample rate
    draws = rng.binomial(n=num_shots, p=p, size=n_bootstrap)
    rates = draws / num_shots
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(rates, 100 * alpha))
    hi = float(np.percentile(rates, 100 * (1 - alpha)))
    if num_errors == 0:
        lo = 0.0
    if num_errors == num_shots:
        hi = 1.0
    return (max(0.0, lo), min(1.0, hi))


def compute_logical_error_rate(
    num_errors: int,
    num_shots: int,
    confidence: float = DEFAULT_CONFIDENCE,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int | None = None,
) -> LERResult:
    if num_shots <= 0:
        raise ValueError(f"num_shots must be > 0; got {num_shots}")
    if num_errors < 0 or num_errors > num_shots:
        raise ValueError(
            f"num_errors must satisfy 0 <= errors <= shots; "
            f"got errors={num_errors} shots={num_shots}"
        )
    used_seed = seed if seed is not None else DEFAULT_SEED
    rate = num_errors / num_shots
    lo, hi = bootstrap_ci(
        num_errors, num_shots, confidence, n_bootstrap, used_seed
    )
    return LERResult(
        logical_error_rate=rate,
        num_errors=num_errors,
        num_shots=num_shots,
        confidence=confidence,
        ci_low=lo,
        ci_high=hi,
        n_bootstrap=n_bootstrap,
        seed=used_seed,
    )
