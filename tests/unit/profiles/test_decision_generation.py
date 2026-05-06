"""Unit tests — decision-summary generator (pure function)."""

from __future__ import annotations

from app.profiles.decision import DecoderMeasurement, generate_decision
from app.profiles.registry import get_profile


def _mkm(
    backend: str, ler: float, p50: float | None = 10.0, thr: float = 1e5, **kw
) -> DecoderMeasurement:
    return DecoderMeasurement(
        backend=backend,
        label=kw.get("label", backend),
        logical_error_rate=ler,
        ler_ci_low=max(0.0, ler * 0.8),
        ler_ci_high=min(1.0, ler * 1.2),
        latency_p50_per_round_us=p50,
        latency_p95_per_round_us=(p50 * 1.5) if p50 else None,
        latency_p99_per_round_us=(p50 * 2.0) if p50 else None,
        throughput_shots_per_s=thr,
        residual_syndrome_density=kw.get("residual", 0.01),
        export_results=kw.get("exports", {}),
        unavailable_reason=kw.get("unavailable"),
    )


def test_recommends_lowest_ler_among_survivors() -> None:
    profile = get_profile("generic_surface_code_readiness")
    ms = [
        _mkm("pymatching_baseline", ler=1e-2, p50=2.0),
        _mkm("pymatching_correlated", ler=5e-3, p50=3.5),
    ]
    d = generate_decision(profile, ms)
    assert d.recommended_backend == "pymatching_correlated"
    assert "pymatching_baseline" in d.pareto_dominated or not d.pareto_dominated


def test_filters_out_path_exceeding_hard_cap() -> None:
    # Superconducting profile has latency_us_hard_cap=63.
    profile = get_profile("superconducting_latency_aware")
    ms = [
        _mkm("pymatching_baseline", ler=1e-2, p50=5.0),
        _mkm("ising_fast", ler=1e-3, p50=100.0),  # over cap
        _mkm("ising_accurate", ler=1e-3, p50=150.0),  # over cap
        _mkm("pymatching_correlated", ler=8e-3, p50=10.0),
    ]
    d = generate_decision(profile, ms)
    # Paths over the cap must be in runtime_budget_violations.
    assert "ising_fast" in d.runtime_budget_violations
    assert "ising_accurate" in d.runtime_budget_violations
    # The recommendation must come from survivors only.
    assert d.recommended_backend in ("pymatching_baseline", "pymatching_correlated")


def test_no_survivor_yields_null_recommendation() -> None:
    profile = get_profile("superconducting_latency_aware")
    ms = [
        _mkm("pymatching_baseline", ler=1e-2, p50=100.0),
        _mkm("pymatching_correlated", ler=5e-3, p50=100.0),
        _mkm("ising_fast", ler=1e-3, p50=100.0),
        _mkm("ising_accurate", ler=1e-3, p50=100.0),
    ]
    d = generate_decision(profile, ms)
    assert d.recommended_backend is None
    assert d.blockers  # at least one blocker explains why


def test_unavailable_path_is_not_candidate_but_recorded() -> None:
    profile = get_profile("generic_surface_code_readiness")
    ms = [
        _mkm("pymatching_baseline", ler=1e-2, p50=2.0),
        _mkm("pymatching_correlated", ler=5e-3, p50=3.5, unavailable="torch missing"),
    ]
    d = generate_decision(profile, ms)
    assert d.recommended_backend == "pymatching_baseline"
    assert d.unavailable_paths == {"pymatching_correlated": "torch missing"}


def test_customer_boundary_is_verbatim_from_profile() -> None:
    profile = get_profile("generic_surface_code_readiness")
    ms = [
        _mkm("pymatching_baseline", ler=1e-2, p50=2.0),
        _mkm("pymatching_correlated", ler=5e-3, p50=3.5),
    ]
    d = generate_decision(profile, ms)
    assert d.public_proxy_can_conclude == profile.boundary.public_proxy_can_conclude
    assert d.requires_customer_private_inputs == profile.boundary.requires_customer_private_inputs


def test_export_failure_blocks_decoder_path() -> None:
    profile = get_profile("ai_predecoder_export_runtime")
    # Every path fails onnx_export_workflow_1 — should all be marked failed.
    ms = [
        _mkm(
            "no_op",
            ler=0.2,
            p50=0.1,
            exports={c: False for c in profile.export_checks},
        ),
        _mkm(
            "pymatching_baseline",
            ler=0.1,
            p50=0.5,
            exports={c: False for c in profile.export_checks},
        ),
        _mkm(
            "ising_fast",
            ler=0.01,
            p50=1.0,
            exports={c: True for c in profile.export_checks},
        ),
        _mkm(
            "ising_accurate",
            ler=0.008,
            p50=2.0,
            exports={c: True for c in profile.export_checks},
        ),
    ]
    d = generate_decision(profile, ms)
    # Expect Ising paths survive; no_op + pymatching fail on export.
    assert d.recommended_backend in ("ising_fast", "ising_accurate")
    assert "no_op" in d.export_failures
    assert "pymatching_baseline" in d.export_failures


def test_decision_is_deterministic_on_same_measurements() -> None:
    profile = get_profile("generic_surface_code_readiness")
    ms = [
        _mkm("pymatching_baseline", ler=1e-2, p50=2.0),
        _mkm("pymatching_correlated", ler=5e-3, p50=3.5),
    ]
    d1 = generate_decision(profile, list(ms))
    d2 = generate_decision(profile, list(ms))
    assert d1.recommended_backend == d2.recommended_backend
    assert d1.dominant_tradeoffs == d2.dominant_tradeoffs
    assert d1.pareto_dominated == d2.pareto_dominated
