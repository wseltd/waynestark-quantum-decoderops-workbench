"""Consolidated repository tests (T064-T068)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.repositories.artefacts_repo import ArtefactsRepository
from app.db.repositories.fingerprint_repo import FingerprintRepository
from app.db.repositories.metrics_repo import MetricsRepository
from app.db.repositories.reports_repo import ReportsRepository
from app.db.repositories.runs_repo import RunNotFoundError, RunsRepository
from app.models.artefact import Artefact
from app.models.fingerprint import Fingerprint
from app.models.metrics import Metrics
from app.models.report import Report
from app.models.run import Run


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _fk_on(conn, _):  # noqa: ANN001
        conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    s = Session(engine)
    try:
        yield s
    finally:
        s.close()


def _mk_run(s: Session, rid: str = "r1") -> Run:
    r = Run(
        run_id=rid,
        config_hash="0" * 64,
        backend="pymatching_baseline",
        status="succeeded",
        db_backend="duckdb",
    )
    s.add(r)
    s.flush()
    return r


# ------------------ T064 RunsRepository ------------------


def test_runs_create_persists_and_returns(session: Session) -> None:
    repo = RunsRepository(session)
    r = Run(
        run_id="x1",
        config_hash="0" * 64,
        backend="pymatching_baseline",
        status="pending",
        db_backend="duckdb",
    )
    out = repo.create(r)
    assert out.run_id == "x1"
    assert repo.get("x1") is not None


def test_runs_get_returns_none_when_missing(session: Session) -> None:
    assert RunsRepository(session).get("nope") is None


def test_runs_list_order_and_pagination(session: Session) -> None:
    repo = RunsRepository(session)
    for rid in ["a", "b", "c"]:
        repo.create(
            Run(
                run_id=rid,
                config_hash="0" * 64,
                backend="pymatching_baseline",
                status="pending",
                db_backend="duckdb",
            )
        )
    rows = repo.list(limit=2, offset=0, order_by="created_at_desc")
    assert len(rows) == 2


def test_runs_update_status_and_missing(session: Session) -> None:
    repo = RunsRepository(session)
    _mk_run(session, "u1")
    repo.update_status("u1", "failed")
    assert repo.get("u1").status == "failed"
    with pytest.raises(RunNotFoundError):
        repo.update_status("nope", "failed")


def test_runs_delete_and_missing(session: Session) -> None:
    repo = RunsRepository(session)
    _mk_run(session, "d1")
    assert repo.delete("d1") is True
    with pytest.raises(RunNotFoundError):
        repo.delete("d1")


# ------------------ T065 MetricsRepository ------------------


def _good_metric(rid: str) -> Metrics:
    return Metrics(
        run_id=rid,
        ler=1e-3,
        ci_low=5e-4,
        ci_high=2e-3,
        p50=1.0,
        p95=2.0,
        p99=3.0,
        throughput=100.0,
        residual_density=0.1,
        shots=1000,
        rounds=3,
        code_distance=5,
        basis="X",
    )


def test_metrics_create_persists_and_returns(session: Session) -> None:
    _mk_run(session, "m1")
    repo = MetricsRepository(session)
    obj = repo.create(_good_metric("m1"))
    assert obj.metrics_id is not None


def test_metrics_get_by_run_id(session: Session) -> None:
    _mk_run(session, "m2")
    repo = MetricsRepository(session)
    repo.create(_good_metric("m2"))
    assert len(repo.get_by_run_id("m2")) == 1


def test_metrics_get_by_id_none(session: Session) -> None:
    assert MetricsRepository(session).get_by_id(99999) is None


def test_metrics_bulk_insert(session: Session) -> None:
    _mk_run(session, "b1")
    repo = MetricsRepository(session)
    ms = [_good_metric("b1") for _ in range(3)]
    out = repo.bulk_insert(ms)
    assert len(out) == 3


# ------------------ T066 ArtefactsRepository ------------------


def test_artefacts_create_and_lookup(session: Session) -> None:
    _mk_run(session, "a1")
    repo = ArtefactsRepository(session)
    a = repo.create(
        Artefact(
            run_id="a1",
            path="x.onnx",
            sha256="a" * 64,
            type="onnx",
            size=10,
        )
    )
    assert repo.get_by_id(a.id) is not None
    assert repo.get_by_sha256("a" * 64) is not None
    assert repo.get_by_sha256("b" * 64) is None
    assert len(repo.get_by_run_id("a1")) == 1
    assert len(repo.list_by_kind("a1", "onnx")) == 1


def test_artefacts_mark_verified(session: Session) -> None:
    _mk_run(session, "a2")
    repo = ArtefactsRepository(session)
    a = repo.create(
        Artefact(
            run_id="a2",
            path="p",
            sha256="c" * 64,
            type="onnx",
            size=1,
        )
    )
    result = repo.mark_verified(a.id, datetime.now(timezone.utc))
    assert result.id == a.id


# ------------------ T067 ReportsRepository ------------------


def test_reports_create_and_lookups(session: Session) -> None:
    _mk_run(session, "rp1")
    repo = ReportsRepository(session)
    repo.create(
        Report(
            run_id="rp1",
            type="engineering_benchmark",
            format="markdown",
            path="r.md",
            sha256="d" * 64,
        )
    )
    assert len(repo.get_by_run_id("rp1")) == 1
    assert repo.get_latest_for_run("rp1", "engineering_benchmark") is not None
    assert len(repo.list_by_type("rp1", "engineering_benchmark")) == 1
    assert len(repo.list_by_format("rp1", "markdown")) == 1
    assert repo.get_latest_for_run("rp1", "deployment_readiness") is None


# ------------------ T068 FingerprintRepository ------------------


def test_fingerprint_create_and_get_by_run_id(session: Session) -> None:
    _mk_run(session, "f1")
    repo = FingerprintRepository(session)
    f = Fingerprint(
        run_id="f1",
        git_sha="abc",
        pip_freeze_digest="d" * 64,
        config_sha256="e" * 64,
        rng_master_seed=42,
        cpu_model="x86_64",
        cpu_count=4,
        gpu_model=None,
        gpu_count=0,
        gpu_driver_version=None,
        os_name="Linux",
        os_kernel="6.8",
        python_version="3.12.13",
        cuda_runtime_version=None,
        timestamp_utc=datetime.now(timezone.utc),
    )
    repo.create(f)
    assert repo.get_by_run_id("f1") is not None
    assert repo.get_by_id(f.id) is not None


def test_fingerprint_find_matching(session: Session) -> None:
    _mk_run(session, "f2")
    repo = FingerprintRepository(session)
    repo.create(
        Fingerprint(
            run_id="f2",
            git_sha="abc",
            pip_freeze_digest="d" * 64,
            config_sha256="e" * 64,
            rng_master_seed=42,
            cpu_model="x",
            cpu_count=1,
            os_name="Linux",
            os_kernel="6",
            python_version="3",
            timestamp_utc=datetime.now(timezone.utc),
        )
    )
    matches = repo.find_matching("abc", "d" * 64, "e" * 64, 42)
    assert len(matches) == 1
    # Mismatch on any identity field → empty
    assert repo.find_matching("abc", "d" * 64, "e" * 64, 43) == []
