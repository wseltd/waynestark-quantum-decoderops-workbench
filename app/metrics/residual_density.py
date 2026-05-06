"""Re-export shim for T047 aggregate's expected module path."""

from app.metrics.residual_syndrome import (
    MAX_VALIDATION_SAMPLES,
    ResidualSyndromeDensity,
    ResidualSyndromeResult,
    compute_activation_rate,
    compute_residual_syndrome_density,
)

__all__ = [
    "MAX_VALIDATION_SAMPLES",
    "ResidualSyndromeDensity",
    "ResidualSyndromeResult",
    "compute_activation_rate",
    "compute_residual_syndrome_density",
]
