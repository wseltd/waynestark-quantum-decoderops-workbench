"""Sinter integration — Monte-Carlo LER with Wilson-score CIs (T036).

Wraps Sinter Monte Carlo sampling for a single
(stim.Circuit, decoder_name, shots, max_errors) tuple and returns a
deterministic :class:`SinterLERResult` with Wilson-score 95 % CIs.

Only ``pymatching`` is supported for v1 — the NVIDIA Ising backends and
the TensorRT adapter are measured via the main runner path (T034), not
Sinter.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SUPPORTED_DECODERS",
    "SinterDecoderUnsupported",
    "SinterLERResult",
    "run_sinter_ler",
    "wilson_ci",
]


SUPPORTED_DECODERS: frozenset[str] = frozenset({"pymatching"})


class SinterDecoderUnsupported(ValueError):
    """Raised when a caller asks for a decoder Sinter cannot dispatch.

    The error carries the offending decoder name and the set of
    supported decoders so the caller can redirect to T034's main runner.
    """

    def __init__(self, decoder: str) -> None:
        self.decoder = decoder
        super().__init__(
            f"Sinter integration supports only {sorted(SUPPORTED_DECODERS)}; "
            f"got {decoder!r}. "
            "Non-Sinter decoders must be measured via app.benchmarking.runner "
            "(T034)."
        )


@dataclass(frozen=True)
class SinterLERResult:
    """Aggregated Sinter MC result for one (circuit, decoder) pair."""

    decoder: str
    shots: int
    errors: int
    ler: float
    ci_low: float
    ci_high: float
    ci_method: str
    seconds: float
    raw_task_stats: list[dict[str, Any]] = field(default_factory=list)


def wilson_ci(errors: int, shots: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-score confidence interval for a binomial proportion.

    Returns ``(lo, hi)`` both in ``[0, 1]``. Handles the ``shots == 0``
    edge case by returning ``(0.0, 1.0)``; handles the all-zero and
    all-one edges deterministically.

    See: Wilson, Edwin B. (1927). "Probable inference, the law of
    succession, and statistical inference". JASA 22 (158): 209-212.
    """
    if errors < 0:
        raise ValueError(f"errors must be non-negative; got {errors}")
    if shots < 0:
        raise ValueError(f"shots must be non-negative; got {shots}")
    if errors > shots:
        raise ValueError(
            f"errors ({errors}) cannot exceed shots ({shots})"
        )
    if shots == 0:
        return (0.0, 1.0)

    n = float(shots)
    p = errors / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    halfwidth = (
        z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    ) / denom
    lo = max(0.0, centre - halfwidth)
    hi = min(1.0, centre + halfwidth)
    if errors == 0:
        lo = 0.0
    if errors == shots:
        hi = 1.0
    return (lo, hi)


def _reject_unsupported(decoder: str) -> None:
    if decoder not in SUPPORTED_DECODERS:
        raise SinterDecoderUnsupported(decoder)


def run_sinter_ler(
    circuit: Any,
    decoder: str,
    shots: int,
    max_errors: int | None = None,
    max_workers: int = 1,
) -> SinterLERResult:
    """Run Sinter MC sampling and return a :class:`SinterLERResult`.

    Args:
        circuit: ``stim.Circuit`` — the QEC circuit to sample.
        decoder: Must be ``"pymatching"`` for v1.
        shots: Maximum shots to sample (``max_shots`` in Sinter terms).
        max_errors: Optional early stop once this many errors are seen.
        max_workers: Passed through to ``sinter.collect``.

    Raises:
        SinterDecoderUnsupported: If the decoder is not in
            :data:`SUPPORTED_DECODERS`.
    """
    _reject_unsupported(decoder)

    import sinter  # local import; only needed on the active path

    task = sinter.Task(circuit=circuit, decoder=decoder)
    t0 = time.perf_counter()
    stats = sinter.collect(
        num_workers=max_workers,
        tasks=[task],
        max_shots=shots,
        max_errors=max_errors,
        print_progress=False,
    )
    seconds = time.perf_counter() - t0

    total_shots = sum(int(s.shots) for s in stats)
    total_errors = sum(int(s.errors) for s in stats)
    ler = (total_errors / total_shots) if total_shots else 0.0
    lo, hi = wilson_ci(total_errors, total_shots)

    raw_stats: list[dict[str, Any]] = []
    for s in stats:
        raw_stats.append(
            {
                "shots": int(s.shots),
                "errors": int(s.errors),
                "discards": int(getattr(s, "discards", 0) or 0),
                "seconds": float(getattr(s, "seconds", 0.0) or 0.0),
            }
        )

    return SinterLERResult(
        decoder=decoder,
        shots=total_shots,
        errors=total_errors,
        ler=ler,
        ci_low=lo,
        ci_high=hi,
        ci_method="wilson_95",
        seconds=seconds,
        raw_task_stats=raw_stats,
    )
