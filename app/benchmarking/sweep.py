"""SweepSpec — declarative parameter sweep specification (T032).

Defines the parameter axes of a reproducible decoder benchmark. No
execution happens here; this module is pure declarative metadata +
deterministic expansion.

Deterministic contract:
    Two SweepSpec instances with identical field values produce
    identical ``expand()`` sequences, including identical ``point_seed``
    values derived from ``master_seed`` via
    ``SHA256(master_seed_bytes || canonical_point_repr)`` truncated to
    64 bits.

Out of scope: wildcards/ranges (explicit lists only), decoder
execution, DB I/O, file reads.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.decoders.registry import BACKEND_NAMES

__all__ = [
    "NoiseSpec",
    "SweepPoint",
    "SweepSpec",
]


NoiseModelLit = Literal["simple_depolarizing", "circuit_level", "si1000"]
BasisLit = Literal["X", "Z"]
ModelVariantLit = Literal["fast", "accurate", "none"]
ExportModeLit = Literal[
    "none", "onnx_workflow_1", "onnx_workflow_2", "cudaq_qec_bin"
]


class NoiseSpec(BaseModel):
    """Noise specification for a single sweep axis.

    Attributes:
        p_error: Physical error rate per operation, 0 < p < 0.5.
        model: Noise model name. ``simple_depolarizing`` is the Stim
            circuit-generator default; ``circuit_level`` matches the
            surface-code memory circuit convention; ``si1000`` matches
            NVIDIA Ising-Decoding's training distribution.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    p_error: float = Field(..., gt=0.0, lt=0.5)
    model: NoiseModelLit


@dataclass(frozen=True, order=True)
class SweepPoint:
    """One concrete coordinate in the Cartesian product of a SweepSpec.

    Ordering is the same as the field declaration order so
    ``sorted([SweepPoint])`` yields the deterministic expansion order
    the orchestrator relies on.

    Attributes:
        distance: Code distance (odd, >= 3).
        rounds: Number of stabiliser rounds (>= 1).
        noise_model: Noise model name.
        p_error: Physical error rate.
        basis: 'X' or 'Z'.
        backend: Registered decoder backend name.
        model_variant: 'fast' | 'accurate' | 'none'.
        export_mode: One of the declared export modes.
        point_seed: Per-point seed derived from the sweep's master_seed
            via SHA256(master || canonical_point_repr) → lower 64 bits.
    """

    distance: int
    rounds: int
    p_error: float
    noise_model: str
    basis: str
    backend: str
    model_variant: str
    export_mode: str
    point_seed: int


class SweepSpec(BaseModel):
    """Declarative parameter sweep."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    distances: list[int] = Field(..., min_length=1)
    rounds: list[int] = Field(..., min_length=1)
    noise: list[NoiseSpec] = Field(..., min_length=1)
    basis: list[BasisLit] = Field(..., min_length=1)
    backends: list[str] = Field(..., min_length=1)
    model_variants: list[ModelVariantLit] = Field(..., min_length=1)
    export_modes: list[ExportModeLit] = Field(..., min_length=1)
    master_seed: int = Field(..., ge=0)
    num_shots: int = Field(..., ge=1)
    schema_version: Literal["1"] = "1"

    @field_validator("distances")
    @classmethod
    def _distances_odd_ge3(cls, v: list[int]) -> list[int]:
        for d in v:
            if d < 3:
                raise ValueError(f"distance {d} < 3; surface code requires d>=3")
            if d % 2 == 0:
                raise ValueError(
                    f"distance {d} is even; rotated surface code requires odd d"
                )
        return v

    @field_validator("rounds")
    @classmethod
    def _rounds_ge1(cls, v: list[int]) -> list[int]:
        for r in v:
            if r < 1:
                raise ValueError(f"rounds {r} < 1")
        return v

    @field_validator("backends")
    @classmethod
    def _backends_are_registered(cls, v: list[str]) -> list[str]:
        unknown = [name for name in v if name not in BACKEND_NAMES]
        if unknown:
            raise ValueError(
                f"unknown backend(s) {unknown}; must be one of {sorted(BACKEND_NAMES)}"
            )
        return v

    # -- expansion --------------------------------------------------------

    def expand(self) -> Iterator[SweepPoint]:
        """Yield SweepPoints in deterministic, stable, sorted order.

        Order: (distance, rounds, p_error, noise_model, basis, backend,
        model_variant, export_mode). Two SweepSpecs with identical
        fields produce identical sequences.
        """
        master_bytes = str(self.master_seed).encode("utf-8")
        points: list[SweepPoint] = []
        for distance in self.distances:
            for rounds in self.rounds:
                for noise in self.noise:
                    for basis in self.basis:
                        for backend in self.backends:
                            for variant in self.model_variants:
                                for export in self.export_modes:
                                    repr_tuple = (
                                        distance,
                                        rounds,
                                        noise.p_error,
                                        noise.model,
                                        basis,
                                        backend,
                                        variant,
                                        export,
                                    )
                                    canonical = json.dumps(
                                        repr_tuple, sort_keys=True
                                    ).encode("utf-8")
                                    digest = hashlib.sha256(
                                        master_bytes + b"|" + canonical
                                    ).digest()
                                    point_seed = int.from_bytes(
                                        digest[:8], "big", signed=False
                                    )
                                    points.append(
                                        SweepPoint(
                                            distance=distance,
                                            rounds=rounds,
                                            p_error=noise.p_error,
                                            noise_model=noise.model,
                                            basis=basis,
                                            backend=backend,
                                            model_variant=variant,
                                            export_mode=export,
                                            point_seed=point_seed,
                                        )
                                    )
        points.sort()
        yield from points

    # -- canonical hash ---------------------------------------------------

    def canonical_hash(self) -> str:
        """SHA256 hex of the canonical-JSON field dump.

        Stable across process invocations and Python versions because
        ``model_dump`` -> ``json.dumps(sort_keys=True)`` is pinned.
        """
        payload = json.dumps(self.model_dump(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
