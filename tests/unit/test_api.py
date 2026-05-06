"""Consolidated API tests (T091-T104)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app


@pytest.fixture(autouse=True)
def _sqlite_db(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route the DB session to a per-test file-backed SQLite with schema."""
    import app.db.session as session_mod
    from app.db.schema_init import bootstrap_schema
    from app.config.settings import reset_settings_cache

    db_file = tmp_path_factory.mktemp("db") / "test.sqlite"
    url = f"sqlite:///{db_file}"
    session_mod._ENGINE_CACHE.clear()
    session_mod._SESSION_CACHE.clear()
    monkeypatch.setenv("DECODEROPS_DB_URL", url)
    reset_settings_cache()
    engine = session_mod.get_engine(url)
    bootstrap_schema(engine)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def test_health_returns_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "db_backend" in body


def test_seed_derives_worker_seeds(client: TestClient) -> None:
    r = client.post("/seed?master_seed=42&num_workers=3")
    assert r.status_code == 200
    body = r.json()
    assert body["master_seed"] == 42
    assert len(body["worker_seeds"]) == 3


def test_ingest_dem_accepts_path(client: TestClient) -> None:
    r = client.post("/ingest/dem", json={"path": "/tmp/a.dem"})
    assert r.status_code == 200


def test_ingest_dem_rejects_missing_path(client: TestClient) -> None:
    r = client.post("/ingest/dem", json={})
    assert r.status_code == 422


def test_ingest_syndrome_accepts_path(client: TestClient) -> None:
    r = client.post("/ingest/syndrome", json={"path": "/tmp/a.npz"})
    assert r.status_code == 200


def test_benchmark_run_expands_sweep(client: TestClient) -> None:
    r = client.post(
        "/benchmark/run",
        json={
            "distances": [3],
            "rounds": [3],
            "bases": ["X"],
            "p_errors": [1e-3],
            "backends": ["pymatching_baseline"],
            "num_shots": 16,
            "master_seed": 42,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "sweep_id" in body
    assert body["num_runs"] >= 1


def test_benchmark_get_404_for_missing(client: TestClient) -> None:
    r = client.get("/benchmark/does-not-exist")
    assert r.status_code == 404


def test_runs_list_returns_empty(client: TestClient) -> None:
    r = client.get("/runs")
    assert r.status_code == 200


def test_runs_get_404_for_missing(client: TestClient) -> None:
    r = client.get("/runs/nope-nope")
    assert r.status_code == 404


def test_metrics_returns_empty_for_unknown_run(client: TestClient) -> None:
    r = client.get("/metrics/unknown")
    assert r.status_code == 200
    assert r.json() == []


def test_artifacts_empty_for_unknown_run(client: TestClient) -> None:
    r = client.get("/artifacts/unknown")
    assert r.status_code == 200
    assert r.json() == []


def test_export_onnx_accepts_request(client: TestClient) -> None:
    r = client.post(
        "/export/onnx", json={"run_id": "r1", "output_path": "/tmp/x.onnx"}
    )
    assert r.status_code == 200


def test_reports_generate_renders_all_five_types(client: TestClient) -> None:
    r = client.post(
        "/reports/generate",
        json={"run_id": "r1", "include_pdf": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["reports"]) == 5 * 3  # md/html/json × 5 types


def test_reports_get_empty_for_unknown_run(client: TestClient) -> None:
    r = client.get("/reports/unknown")
    assert r.status_code == 200
    assert r.json() == []


def test_evidence_latest_404_when_db_empty(client: TestClient) -> None:
    r = client.get("/evidence/latest")
    assert r.status_code == 404


def test_evidence_by_run_id_returns_stub(client: TestClient) -> None:
    r = client.get("/evidence/r1")
    assert r.status_code == 200
    assert r.json()["run_id"] == "r1"


def test_openapi_exposes_all_13_required_endpoints(client: TestClient) -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    required_13 = {
        "/health",
        "/seed",
        "/ingest/dem",
        "/ingest/syndrome",
        "/benchmark/run",
        "/benchmark/{run_id}",
        "/runs",
        "/metrics/{run_id}",
        "/artifacts/{run_id}",
        "/export/onnx",
        "/reports/generate",
        "/reports/{run_id}",
        "/evidence/latest",
    }
    missing = required_13 - set(paths.keys())
    assert not missing, f"required endpoints missing: {missing}"
