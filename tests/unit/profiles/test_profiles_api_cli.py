"""Tests — profiles over HTTP API and Typer CLI."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from app.api.app import create_app
from app.cli.main import app as cli_app


@pytest.fixture(autouse=True)
def _sqlite_db(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the DB session to a file-backed sqlite for API tests."""
    import app.db.session as session_mod
    from app.config.settings import reset_settings_cache
    from app.db.schema_init import bootstrap_schema

    db_file = tmp_path_factory.mktemp("db") / "api.sqlite"
    url = f"sqlite:///{db_file}"
    session_mod._ENGINE_CACHE.clear()
    session_mod._SESSION_CACHE.clear()
    monkeypatch.setenv("DECODEROPS_DB_URL", url)
    reset_settings_cache()
    engine = session_mod.get_engine(url)
    bootstrap_schema(engine)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_list_profiles_returns_registry(client: TestClient) -> None:
    r = client.get("/profiles")
    assert r.status_code == 200
    ids = {p["profile_id"] for p in r.json()}
    assert {
        "generic_surface_code_readiness",
        "superconducting_latency_aware",
        "ai_predecoder_export_runtime",
    }.issubset(ids)


def test_get_profile_returns_full_spec(client: TestClient) -> None:
    r = client.get("/profiles/generic_surface_code_readiness")
    assert r.status_code == 200
    body = r.json()
    assert body["profile_id"] == "generic_surface_code_readiness"
    assert len(body["decoder_paths"]) >= 2
    assert body["boundary"]["public_proxy_can_conclude"]
    assert body["boundary"]["requires_customer_private_inputs"]


def test_get_profile_404_for_unknown(client: TestClient) -> None:
    r = client.get("/profiles/not_a_profile")
    assert r.status_code == 404


def test_post_run_profile_end_to_end(client: TestClient) -> None:
    r = client.post(
        "/profiles/generic_surface_code_readiness/run",
        json={
            "num_shots": 64,
            "master_seed": 20260422,
            "bases": ["X"],
            "include_pdf": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["profile_id"] == "generic_surface_code_readiness"
    assert body["decision"]["recommended_backend"] in (
        "pymatching_baseline",
        "pymatching_correlated",
    )
    assert body["decision"]["public_proxy_can_conclude"]
    assert body["decision"]["requires_customer_private_inputs"]
    assert body["rendered_reports"], "rendered_reports must be non-empty"
    assert body["decision_run_id"]
    # Round-trip via /decisions/{run_id}
    run_id = body["decision_run_id"]
    r2 = client.get(f"/decisions/{run_id}")
    assert r2.status_code == 200
    assert r2.json()["decision"]["recommended_backend"] == body["decision"]["recommended_backend"]


# --- CLI ---


def test_cli_list_profiles_registered() -> None:
    names = {c.name for c in cli_app.registered_commands}
    assert "list-profiles" in names
    assert "show-profile" in names
    assert "run-profile" in names


def test_cli_list_profiles_hits_api() -> None:
    runner = CliRunner()
    fake_response = [
        {
            "profile_id": "x",
            "name": "X",
            "architecture": "generic",
            "caution_label": "",
            "num_decoder_paths": 2,
            "num_export_checks": 0,
            "has_runtime_budget": False,
        }
    ]
    with patch("app.cli.commands.list_profiles.DecoderOpsClient") as mock_client:
        mock_client.return_value.request.return_value = fake_response
        result = runner.invoke(cli_app, ["list-profiles"])
    assert result.exit_code == 0
    assert '"profile_id": "x"' in result.stdout


def test_cli_run_profile_prints_decision_summary() -> None:
    runner = CliRunner()
    fake_response = {
        "profile_id": "generic_surface_code_readiness",
        "decision_run_id": "abc123",
        "decision": {
            "recommended_backend": "pymatching_correlated",
            "recommendation_label": "PyMatching correlated MWPM",
            "recommendation_reason": "because",
            "blockers": [],
            "pareto_dominated": ["pymatching_baseline"],
        },
        "manifest_path": "/tmp/x/profile_manifest.json",
        "rendered_reports": [],
    }
    with patch("app.cli.commands.run_profile.DecoderOpsClient") as mock_client:
        mock_client.return_value.request.return_value = fake_response
        result = runner.invoke(
            cli_app,
            ["run-profile", "generic_surface_code_readiness", "--num-shots", "64"],
        )
    assert result.exit_code == 0
    assert "pymatching_correlated" in result.stdout
    assert "abc123" in result.stdout
