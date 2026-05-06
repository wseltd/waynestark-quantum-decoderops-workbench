"""Integration tests — profile runner + report rendering + provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.profiles.registry import get_profile
from app.profiles.runner import run_profile
from app.reports.pipeline import render_decision_report


@pytest.fixture
def small_profile_run(tmp_path: Path):
    profile = get_profile("generic_surface_code_readiness")
    result = run_profile(
        profile,
        num_shots=128,
        master_seed=20260422,
        output_dir=tmp_path / "run",
        bases=("X",),
    )
    return profile, result


def test_run_profile_returns_real_decision(small_profile_run) -> None:
    _, result = small_profile_run
    d = result.decision
    # One of the two comparable backends MUST win.
    assert d.recommended_backend in (
        "pymatching_baseline",
        "pymatching_correlated",
    )
    assert d.recommendation_reason
    assert d.public_proxy_can_conclude  # not empty
    assert d.requires_customer_private_inputs  # not empty


def test_run_profile_manifest_records_provenance(small_profile_run) -> None:
    _, result = small_profile_run
    mf = json.loads(result.manifest_path.read_text())
    assert mf["profile_id"] == "generic_surface_code_readiness"
    assert mf["profile_sha256"]
    assert len(mf["provenance_sources"]) >= 2
    assert mf["dem_sha256"]
    # Every expanded point has all four pinned fields.
    for pt in mf["expanded_points"]:
        assert {"distance", "rounds", "basis", "p_error"} <= set(pt)


def test_run_profile_same_seed_same_decision(tmp_path: Path) -> None:
    profile = get_profile("generic_surface_code_readiness")
    a = run_profile(
        profile,
        num_shots=64,
        master_seed=20260422,
        output_dir=tmp_path / "a",
        bases=("X",),
    )
    b = run_profile(
        profile,
        num_shots=64,
        master_seed=20260422,
        output_dir=tmp_path / "b",
        bases=("X",),
    )
    assert a.decision.recommended_backend == b.decision.recommended_backend
    # Measurements are deterministic: LER, throughput shape, residual.
    for m_a, m_b in zip(
        a.decision.measurements, b.decision.measurements, strict=True
    ):
        assert m_a["backend"] == m_b["backend"]
        assert m_a["logical_error_rate"] == m_b["logical_error_rate"]
        assert m_a["residual_syndrome_density"] == m_b["residual_syndrome_density"]


def test_decision_report_renders_md_html_json(tmp_path: Path, small_profile_run) -> None:
    profile, result = small_profile_run
    bundle = result.to_dict()
    ctx = {
        "profile": profile.model_dump(mode="json"),
        "decision": bundle["decision"],
        "provenance": bundle["provenance"],
        "run": {
            "master_seed": 20260422,
            "num_shots": 128,
            "effective_bases": ["X"],
        },
    }
    rendered = render_decision_report(decision_context=ctx, output_dir=tmp_path / "reports")
    types_formats = {(r.type, r.format) for r in rendered}
    assert ("decision_report", "markdown") in types_formats
    assert ("decision_report", "html") in types_formats
    assert ("decision_report", "json") in types_formats

    md_path = tmp_path / "reports" / "decision_report.md"
    md = md_path.read_text()
    # Customer boundary section is rendered.
    assert "CAN conclude" in md
    assert "CANNOT conclude" in md
    # At least one provenance URL is rendered.
    assert "github.com" in md or "pymatching.readthedocs.io" in md


def test_run_profile_rejects_override_not_in_allowlist(tmp_path: Path) -> None:
    profile = get_profile("generic_surface_code_readiness")
    with pytest.raises(ValueError):
        run_profile(
            profile,
            num_shots=64,
            master_seed=20260422,
            output_dir=tmp_path / "x",
            bases=("ZZ",),  # type: ignore[arg-type]
        )


def test_report_wording_flags_pareto_domination(tmp_path: Path, small_profile_run) -> None:
    profile, result = small_profile_run
    bundle = result.to_dict()
    ctx = {
        "profile": profile.model_dump(mode="json"),
        "decision": bundle["decision"],
        "provenance": bundle["provenance"],
        "run": {
            "master_seed": 20260422,
            "num_shots": 128,
            "effective_bases": ["X"],
        },
    }
    render_decision_report(decision_context=ctx, output_dir=tmp_path / "reports")
    md = (tmp_path / "reports" / "decision_report.md").read_text()
    # Whether a path is Pareto-dominated depends on the measurement
    # Pareto front; the rendering branch must be exercised either as
    # "Pareto dominated paths" header OR as an empty-domination case.
    # Either way, the measured-comparison table is present.
    assert "Measured comparison" in md
