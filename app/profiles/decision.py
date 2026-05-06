"""Decision-summary generation from MEASURED profile outputs.

Pure function layer. The decision summary is derived deterministically
from:
    - the ProfileSpec (pinned ranges, runtime budget, boundary)
    - per-decoder measured metrics (LER, latency percentiles,
      throughput, residual density, export status)

The summary is NOT generated narrative. It is structured JSON with
explicit rule-based recommendations. Rendering layers add prose *around*
the structured decision; they do not invent conclusions.

Selection rule (documented contract):

    1. Filter out decoder paths whose export_checks failed when the
       profile declares export_checks — those are non-deployable.
    2. Filter out decoder paths whose measured per-round latency
       exceeds runtime_budget.latency_us_hard_cap — non-viable.
    3. Among survivors, prefer the path with the lowest point-estimate
       LER. Ties resolved by higher throughput, then by lower p50
       latency.
    4. If NO path survives the hard cap, recommend NONE and say so.
    5. Record the Pareto front (LER vs p50 latency) as dominated-vs-
       dominator evidence for the report layer to render.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.profiles.schema import ProfileSpec

__all__ = [
    "DecisionOutcome",
    "DecoderMeasurement",
    "generate_decision",
]


@dataclass(frozen=True)
class DecoderMeasurement:
    """One measured row for a decoder path in a profile run."""

    backend: str
    label: str
    logical_error_rate: float
    ler_ci_low: float
    ler_ci_high: float
    latency_p50_per_round_us: float | None
    latency_p95_per_round_us: float | None
    latency_p99_per_round_us: float | None
    throughput_shots_per_s: float
    residual_syndrome_density: float | None
    export_results: dict[str, bool]  # export_check_name -> success
    export_notes: dict[str, str] = field(default_factory=dict)
    unavailable_reason: str | None = None  # e.g., capability unavailable


@dataclass(frozen=True)
class DecisionOutcome:
    """Structured, rule-based decision output."""

    profile_id: str
    recommended_backend: str | None
    recommendation_label: str | None
    recommendation_reason: str
    dominant_tradeoffs: tuple[str, ...]
    blockers: tuple[str, ...]
    pareto_dominated: tuple[str, ...]  # backends strictly dominated by recommendation
    runtime_budget_violations: dict[str, str]
    export_failures: dict[str, tuple[str, ...]]
    unavailable_paths: dict[str, str]
    # Customer-boundary block (rendered verbatim):
    public_proxy_can_conclude: tuple[str, ...]
    requires_customer_private_inputs: tuple[str, ...]
    # Raw measurements echoed back for the report layer:
    measurements: tuple[dict[str, Any], ...]


def _is_dominated(a: DecoderMeasurement, b: DecoderMeasurement) -> bool:
    """Return True if b strictly dominates a on (LER↓, p50_latency↓)."""
    a_lat = a.latency_p50_per_round_us or math.inf
    b_lat = b.latency_p50_per_round_us or math.inf
    better_or_eq = b.logical_error_rate <= a.logical_error_rate and b_lat <= a_lat
    strictly_better = b.logical_error_rate < a.logical_error_rate or b_lat < a_lat
    return better_or_eq and strictly_better


def generate_decision(
    profile: ProfileSpec,
    measurements: list[DecoderMeasurement],
) -> DecisionOutcome:
    """Run the documented selection rule and emit a DecisionOutcome."""
    budget = profile.runtime_budget

    unavailable_paths: dict[str, str] = {
        m.backend: m.unavailable_reason for m in measurements if m.unavailable_reason
    }
    available = [m for m in measurements if m.unavailable_reason is None]

    # 1. Profile-level required export checks: if any path failed ALL
    # required checks, mark as export-failed.
    export_failures: dict[str, tuple[str, ...]] = {}
    required_exports = profile.export_checks
    for m in available:
        failed = tuple(
            e for e in required_exports if e in m.export_results and not m.export_results[e]
        )
        if failed:
            export_failures[m.backend] = failed

    # 2. Runtime budget hard cap (per-round p50 latency).
    runtime_violations: dict[str, str] = {}
    if budget is not None and budget.latency_us_hard_cap is not None:
        cap = budget.latency_us_hard_cap
        for m in available:
            lat = m.latency_p50_per_round_us
            if lat is not None and lat > cap:
                runtime_violations[m.backend] = (
                    f"p50 per-round latency {lat:.2f} μs exceeds hard cap {cap:.2f} μs"
                )

    # Survivors: not export-failed, not runtime-violating.
    survivors = [
        m
        for m in available
        if m.backend not in export_failures and m.backend not in runtime_violations
    ]

    blockers: list[str] = []
    if runtime_violations:
        blockers.append(f"{len(runtime_violations)} path(s) exceeded runtime hard cap.")
    if export_failures:
        blockers.append(f"{len(export_failures)} path(s) failed required export checks.")
    if unavailable_paths:
        blockers.append(f"{len(unavailable_paths)} path(s) unavailable on this environment.")

    # 3-4. Pick the lowest-LER survivor; tie-break on throughput then p50.
    recommended: DecoderMeasurement | None = None
    if survivors:
        recommended = min(
            survivors,
            key=lambda m: (
                m.logical_error_rate,
                -m.throughput_shots_per_s,
                m.latency_p50_per_round_us or math.inf,
            ),
        )

    # 5. Pareto-dominated backends (for report context).
    dominated: list[str] = []
    if recommended is not None:
        for m in available:
            if m.backend == recommended.backend:
                continue
            if _is_dominated(m, recommended):
                dominated.append(m.backend)

    # Trade-off callouts — structured, not narrative.
    tradeoffs: list[str] = []
    if len(available) >= 2 and recommended is not None:
        other = [m for m in available if m.backend != recommended.backend]
        other_min_ler = min((m.logical_error_rate for m in other), default=math.inf)
        ler_gap = other_min_ler - recommended.logical_error_rate
        tradeoffs.append(
            f"Recommended LER {recommended.logical_error_rate:.3e} vs next-"
            f"best {other_min_ler:.3e} (Δ={ler_gap:+.3e})."
        )
        if budget is not None and recommended.latency_p50_per_round_us is not None:
            headroom = (budget.latency_us_target or math.inf) - recommended.latency_p50_per_round_us
            tradeoffs.append(
                f"p50 latency {recommended.latency_p50_per_round_us:.2f} μs "
                f"vs target {budget.latency_us_target}; headroom={headroom:+.2f} μs."
            )
        if recommended.residual_syndrome_density is not None:
            tradeoffs.append(
                f"Residual syndrome density after recommended path: "
                f"{recommended.residual_syndrome_density:.3e}."
            )
        if required_exports and recommended.export_results:
            passed = sum(1 for e in required_exports if recommended.export_results.get(e))
            tradeoffs.append(f"Required exports passed: {passed}/{len(required_exports)}.")

    reason_lines: list[str] = []
    if recommended is None:
        reason_lines.append(
            "No decoder path survived the profile's export and runtime "
            "gates. Recommendation suppressed — see blockers."
        )
    else:
        reason_lines.append(
            f"Selected {recommended.label!r} (backend={recommended.backend}) "
            f"because it has the lowest LER among paths that passed the "
            f"profile's export and runtime gates."
        )
        if dominated:
            reason_lines.append(f"Strictly dominates on (LER, p50 latency): {sorted(dominated)}.")
        if unavailable_paths:
            reason_lines.append(f"Paths skipped due to environment: {sorted(unavailable_paths)}.")

    return DecisionOutcome(
        profile_id=profile.profile_id,
        recommended_backend=recommended.backend if recommended else None,
        recommendation_label=recommended.label if recommended else None,
        recommendation_reason=" ".join(reason_lines),
        dominant_tradeoffs=tuple(tradeoffs),
        blockers=tuple(blockers),
        pareto_dominated=tuple(sorted(dominated)),
        runtime_budget_violations=dict(sorted(runtime_violations.items())),
        export_failures={k: v for k, v in sorted(export_failures.items())},
        unavailable_paths=dict(sorted(unavailable_paths.items())),
        public_proxy_can_conclude=profile.boundary.public_proxy_can_conclude,
        requires_customer_private_inputs=(profile.boundary.requires_customer_private_inputs),
        measurements=tuple(
            {
                "backend": m.backend,
                "label": m.label,
                "logical_error_rate": m.logical_error_rate,
                "ler_ci_low": m.ler_ci_low,
                "ler_ci_high": m.ler_ci_high,
                "latency_p50_per_round_us": m.latency_p50_per_round_us,
                "latency_p95_per_round_us": m.latency_p95_per_round_us,
                "latency_p99_per_round_us": m.latency_p99_per_round_us,
                "throughput_shots_per_s": m.throughput_shots_per_s,
                "residual_syndrome_density": m.residual_syndrome_density,
                "export_results": dict(m.export_results),
                "unavailable_reason": m.unavailable_reason,
            }
            for m in measurements
        ),
    )
