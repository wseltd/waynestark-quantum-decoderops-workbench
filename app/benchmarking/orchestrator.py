"""Benchmark orchestrator — SweepSpec → list[RunConfig] (T033).

Produces a deterministic list of :class:`RunConfig` objects from a
:class:`app.benchmarking.sweep.SweepSpec`. Cartesian product over the
declared axes; stable ordering so run_ids are reproducible across hosts
and process invocations.

``run_id = sha256(canonical_json(config_without_run_id_and_timing))[:16]``

Maximum sweep size is capped at :data:`MAX_SWEEP_SIZE` to protect
against accidental explosive sweeps (2D grid scans with too many
axes).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Final, Iterator

from app.benchmarking.sweep import SweepSpec

__all__ = [
    "MAX_SWEEP_SIZE",
    "RunConfig",
    "SweepTooLargeError",
    "compute_run_id",
    "expand_sweep",
]


MAX_SWEEP_SIZE: Final[int] = 10_000


class SweepTooLargeError(ValueError):
    """Sweep expansion would exceed :data:`MAX_SWEEP_SIZE`."""

    def __init__(self, *, size: int, limit: int) -> None:
        self.size = size
        self.limit = limit
        super().__init__(
            f"sweep would expand to {size} RunConfigs; exceeds limit of {limit}. "
            "Reduce axis sizes or raise MAX_SWEEP_SIZE explicitly after review."
        )


@dataclass(frozen=True)
class RunConfig:
    """Concrete benchmark configuration for a single execution.

    Every field contributes to ``run_id`` EXCEPT ``run_id`` itself and
    ``sweep_id`` (which already encodes the full SweepSpec).
    """

    run_id: str
    sweep_id: str
    distance: int
    rounds: int
    noise: dict
    basis: str
    backend: str
    model_variant: str
    export_mode: str
    worker_seed_slot: int
    master_seed: int
    num_shots: int


def compute_run_id(config_dict: dict) -> str:
    """Compute the 16-char prefix of SHA256(canonical_json(config))."""
    # Exclude fields that can't be deterministic at hash time.
    stripped = {k: v for k, v in config_dict.items() if k != "run_id"}
    payload = json.dumps(stripped, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def expand_sweep(spec: SweepSpec) -> Iterator[RunConfig]:
    """Yield RunConfigs for every SweepPoint in deterministic order.

    Uses :meth:`SweepSpec.expand` which itself is deterministic; layers
    the sweep-level ``sweep_id`` (SweepSpec canonical hash) and a
    stable ``worker_seed_slot`` (index into the expanded sequence) on
    top.

    Raises:
        SweepTooLargeError: When the product exceeds MAX_SWEEP_SIZE.
    """
    sweep_id = spec.canonical_hash()
    # Materialise once so we can size-check and index deterministically.
    points = list(spec.expand())
    if len(points) > MAX_SWEEP_SIZE:
        raise SweepTooLargeError(size=len(points), limit=MAX_SWEEP_SIZE)

    for slot, p in enumerate(points):
        payload = {
            "sweep_id": sweep_id,
            "distance": p.distance,
            "rounds": p.rounds,
            "noise": {"p_error": p.p_error, "model": p.noise_model},
            "basis": p.basis,
            "backend": p.backend,
            "model_variant": p.model_variant,
            "export_mode": p.export_mode,
            "worker_seed_slot": slot,
            "master_seed": spec.master_seed,
            "num_shots": spec.num_shots,
        }
        run_id = compute_run_id(payload)
        yield RunConfig(
            run_id=run_id,
            sweep_id=sweep_id,
            distance=p.distance,
            rounds=p.rounds,
            noise={"p_error": p.p_error, "model": p.noise_model},
            basis=p.basis,
            backend=p.backend,
            model_variant=p.model_variant,
            export_mode=p.export_mode,
            worker_seed_slot=slot,
            master_seed=spec.master_seed,
            num_shots=spec.num_shots,
        )
