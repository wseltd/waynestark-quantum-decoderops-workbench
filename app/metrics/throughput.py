"""Shots/sec + rounds/sec throughput (T044)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["NS_PER_SECOND", "Throughput", "ThroughputResult", "compute_throughput"]


NS_PER_SECOND: int = 1_000_000_000


class ThroughputResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_shots: int
    num_rounds_per_shot: int
    total_rounds: int
    total_elapsed_ns: int
    total_elapsed_seconds: float
    shots_per_second: float
    rounds_per_second: float


Throughput = ThroughputResult


def compute_throughput(
    num_shots: int,
    num_rounds_per_shot: int,
    total_elapsed_ns: int,
) -> ThroughputResult:
    if num_shots <= 0:
        raise ValueError(f"num_shots must be > 0; got {num_shots}")
    if num_rounds_per_shot <= 0:
        raise ValueError(
            f"num_rounds_per_shot must be > 0; got {num_rounds_per_shot}"
        )
    if total_elapsed_ns <= 0:
        raise ValueError(
            f"total_elapsed_ns must be > 0; got {total_elapsed_ns}"
        )
    total_rounds = num_shots * num_rounds_per_shot
    seconds = total_elapsed_ns / NS_PER_SECOND
    return ThroughputResult(
        num_shots=num_shots,
        num_rounds_per_shot=num_rounds_per_shot,
        total_rounds=total_rounds,
        total_elapsed_ns=total_elapsed_ns,
        total_elapsed_seconds=seconds,
        shots_per_second=num_shots / seconds,
        rounds_per_second=total_rounds / seconds,
    )
