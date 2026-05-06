"""Public proxy decision-profile schema.

A ``ProfileSpec`` is a first-class, typed, hashable description of a
serious public-proxy benchmark scenario. It encodes:

- what variable ranges to sweep (pinned, grounded in the research pack)
- which decoder paths to compare
- which export/runtime checks to perform
- which public source URLs back the choices
- what the profile CAN conclude from public data alone
- what the profile CANNOT conclude without customer-private inputs

Profiles are deterministic data. No profile field accepts a free-form
LLM narrative. The decision-summary layer (``app.profiles.decision``)
derives conclusions from measured outputs, not from this spec.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "ArchitectureStyle",
    "CustomerBoundary",
    "DecoderPath",
    "ExportCheck",
    "ProfileSpec",
    "ProvenanceSource",
    "RuntimeBudget",
]

# Architecture families the research pack supports with *grounded* public
# sources. Extending this list requires a corresponding primary source.
ArchitectureStyle = Literal[
    "superconducting",
    "trapped_ion",
    "silicon_spin",
    "photonic",
    "generic",
]

# Every export/runtime check supported by the workbench today. Each
# maps to an existing app.benchmarking / app.packaging entry point.
ExportCheck = Literal[
    "none",
    "onnx_export_workflow_1",  # vendor local_run.sh ONNX_WORKFLOW=1
    "onnx_export_workflow_2_int8",  # ONNX + TRT engine (int8)
    "onnx_export_workflow_2_fp8",  # ONNX + TRT engine (fp8)
    "onnx_validation",  # onnx.checker.check_model
    "tensorrt_engine_build",  # direct trt engine build via our wrapper
    "cudaq_qec_test_data",  # vendor generate_test_data.py → .bin bundle
    "tarball_offline_verify",  # packaging content-addressed tarball + verify
]

# A decoder path is a concrete comparison element of the profile.
# ``label`` is the human-readable name shown in decision reports.
# ``backend`` is the Decoder Protocol backend name used by the runner.
# ``requires`` lists capabilities that must be present at report time
# for this path to count (e.g., tensorrt for TRT variants).


class DecoderPath(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(..., min_length=1, max_length=80)
    backend: Literal[
        "pymatching_baseline",
        "pymatching_correlated",
        "ising_fast",
        "ising_accurate",
        "onnx_validation",
        "tensorrt_optional",
        "no_op",
    ]
    requires: tuple[str, ...] = ()


class RuntimeBudget(BaseModel):
    """Public-source-backed runtime envelope for a profile.

    ``latency_us_target`` is the headline decoder-latency budget (not
    ingest, not transport). ``latency_us_hard_cap`` is the envelope
    beyond which a path is considered non-viable. ``cycle_time_us``
    is the syndrome cycle time when the public source gives one
    (Willow: 1.1 μs).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    latency_us_target: float | None = Field(
        default=None, gt=0, description="Per-round decoder latency goal (μs)."
    )
    latency_us_hard_cap: float | None = Field(
        default=None, gt=0, description="Upper bound before path is non-viable."
    )
    cycle_time_us: float | None = Field(default=None, gt=0)
    source_notes: str = ""

    @model_validator(mode="after")
    def _ordering(self) -> RuntimeBudget:
        t = self.latency_us_target
        c = self.latency_us_hard_cap
        if t is not None and c is not None and t > c:
            raise ValueError(f"latency_us_target ({t}) must be <= latency_us_hard_cap ({c})")
        return self


class ProvenanceSource(BaseModel):
    """A primary-source reference for a parameter choice or claim.

    ``url`` must be an official project / paper / vendor URL. The
    field is validated to reject placeholders like ``example.com``.
    ``cites`` names the exact parameter(s) this source backs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(..., min_length=1)
    url: str = Field(..., min_length=10)
    cites: tuple[str, ...] = Field(..., min_length=1)
    note: str = ""

    @field_validator("url")
    @classmethod
    def _real_url(cls, v: str) -> str:
        if not re.match(r"^https?://", v):
            raise ValueError(f"url must be absolute http(s): got {v!r}")
        banned = ("example.com", "localhost", "127.0.0.1", "placeholder")
        if any(b in v for b in banned):
            raise ValueError(f"url looks like a placeholder: {v!r}")
        return v


class CustomerBoundary(BaseModel):
    """Explicit statement of what the profile can / cannot conclude.

    Both lists are required and non-empty. This is the core honesty
    contract of the profile system: reports render these verbatim so
    the product cannot overclaim.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    public_proxy_can_conclude: tuple[str, ...] = Field(..., min_length=1)
    requires_customer_private_inputs: tuple[str, ...] = Field(..., min_length=1)


class ProfileSpec(BaseModel):
    """A public proxy decision profile — first-class typed object.

    All ranges are pinned at profile-definition time. The runner may
    accept a narrow ``overrides`` dict restricted to approved keys
    (e.g. ``num_shots``, ``master_seed``, ``basis``), but may not
    add code distances or p_error values the profile does not list.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1"] = "1"
    profile_id: str = Field(..., pattern=r"^[a-z][a-z0-9_]{2,63}$")
    name: str = Field(..., min_length=3, max_length=120)
    description: str = Field(..., min_length=20)
    architecture: ArchitectureStyle

    # Declared intended use / limitations — rendered verbatim in reports.
    intended_use: str = Field(..., min_length=20)
    limitations: str = Field(..., min_length=20)

    # Pinned variable ranges. Empty lists are rejected.
    distances: tuple[int, ...] = Field(..., min_length=1)
    rounds_by_distance: dict[int, tuple[int, ...]] = Field(..., min_length=1)
    bases: tuple[Literal["X", "Z"], ...] = Field(..., min_length=1)
    p_errors: tuple[float, ...] = Field(..., min_length=1)
    noise_model_id: Literal["simple_depolarizing", "circuit_level", "si1000"] = (
        "simple_depolarizing"
    )

    # Decoder paths to compare — always >= 2 so there's a real comparison.
    decoder_paths: tuple[DecoderPath, ...] = Field(..., min_length=2)

    # Exports / runtime checks (may be empty for a classical-only profile).
    export_checks: tuple[ExportCheck, ...] = ()

    # Runtime envelope, if the architecture has a public budget.
    runtime_budget: RuntimeBudget | None = None

    # Customer-boundary + provenance (MANDATORY).
    boundary: CustomerBoundary
    provenance: tuple[ProvenanceSource, ...] = Field(..., min_length=2)

    # Caution tag — e.g. "proxy-only, not deployment-grade". Required
    # for architectures where the public deployment signal is weak.
    caution_label: str = ""

    # Narrow set of approved runtime overrides the runner accepts.
    allowed_overrides: tuple[str, ...] = (
        "num_shots",
        "master_seed",
        "bases",
    )

    @field_validator("distances")
    @classmethod
    def _distances_odd_ge3(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        for d in v:
            if d < 3 or d % 2 == 0:
                raise ValueError(f"distance {d} invalid — surface code requires odd d>=3")
        return v

    @field_validator("p_errors")
    @classmethod
    def _p_errors_in_range(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        for p in v:
            if not (0 < p < 0.5):
                raise ValueError(f"p_error {p} must be in (0, 0.5)")
        return v

    @model_validator(mode="after")
    def _rounds_align_to_distances(self) -> ProfileSpec:
        missing = [d for d in self.distances if d not in self.rounds_by_distance]
        if missing:
            raise ValueError(f"rounds_by_distance missing entries for distances {missing}")
        extra = [d for d in self.rounds_by_distance if d not in self.distances]
        if extra:
            raise ValueError(f"rounds_by_distance has entries for unlisted distances {extra}")
        for d, rounds in self.rounds_by_distance.items():
            if not rounds:
                raise ValueError(f"rounds for distance {d} is empty")
            for r in rounds:
                if r < 1:
                    raise ValueError(f"rounds {r} for distance {d} must be >= 1")
        return self

    def expand_points(self) -> list[dict[str, Any]]:
        """Deterministic Cartesian expansion of the pinned ranges.

        Returns a list of dicts, each a concrete (d, rounds, basis,
        p_error) tuple plus the backends to compare at that point.
        Order is stable and sorted — two runs of the same profile
        produce identical expansions.
        """
        out: list[dict[str, Any]] = []
        for d in sorted(self.distances):
            for r in sorted(self.rounds_by_distance[d]):
                for basis in sorted(self.bases):
                    for p in sorted(self.p_errors):
                        out.append(
                            {
                                "distance": d,
                                "rounds": r,
                                "basis": basis,
                                "p_error": p,
                                "noise_model_id": self.noise_model_id,
                                "backends": tuple(dp.backend for dp in self.decoder_paths),
                            }
                        )
        return out

    def matches_override(self, key: str, value: Any) -> bool:
        """Gate callers' override attempts against the allow-list."""
        return key in self.allowed_overrides
