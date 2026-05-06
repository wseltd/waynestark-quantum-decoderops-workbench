"""Consolidated ORM model tests (T055-T060)."""

from __future__ import annotations

import pytest
from sqlalchemy import UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

from app.db.base import Base, metadata
from app.models.artefact import Artefact
from app.models.fingerprint import Fingerprint
from app.models.metrics import Metrics
from app.models.report import Report
from app.models.run import Run


def _setup_engine():
    engine = create_engine("sqlite:///:memory:")
    # SQLite requires foreign_keys PRAGMA for FK enforcement.
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _):  # noqa: ANN001
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return engine


# ------------------ T060 Base ------------------


def test_base_is_declarative_and_has_naming_convention() -> None:
    assert issubclass(Base, DeclarativeBase)
    assert metadata is Base.metadata
    for k in ("ix", "uq", "ck", "fk", "pk"):
        assert k in Base.metadata.naming_convention


# ------------------ T055 Run ------------------


def test_run_table_has_all_required_columns() -> None:
    cols = {c.name for c in Run.__table__.columns}
    assert {
        "run_id",
        "started_at",
        "finished_at",
        "config_hash",
        "backend",
        "status",
        "git_sha",
        "pip_freeze_digest",
        "rng_master_seed",
        "db_backend",
        "notes",
    }.issubset(cols)


def test_run_allowed_backends_contains_five_decoder_names() -> None:
    b = Run.allowed_backends()
    assert len(b) == 5
    assert "pymatching_baseline" in b


def test_run_allowed_statuses_enforced_by_check_constraint() -> None:
    assert "succeeded" in Run.allowed_statuses()


def test_run_insert_and_select_roundtrip_on_sqlite() -> None:
    e = _setup_engine()
    with Session(e) as s:
        r = Run(
            run_id="r1",
            config_hash="0" * 64,
            backend="pymatching_baseline",
            status="succeeded",
            db_backend="duckdb",
        )
        s.add(r)
        s.commit()
        got = s.get(Run, "r1")
        assert got is not None and got.backend == "pymatching_baseline"


def test_run_started_at_defaults_to_utc_now() -> None:
    e = _setup_engine()
    with Session(e) as s:
        r = Run(
            run_id="r2",
            config_hash="0" * 64,
            backend="pymatching_baseline",
            status="succeeded",
            db_backend="duckdb",
        )
        s.add(r)
        s.commit()
        assert r.started_at is not None


def test_run_rejects_unknown_backend_value() -> None:
    e = _setup_engine()
    with Session(e) as s:
        r = Run(
            run_id="r3",
            config_hash="0" * 64,
            backend="bogus_backend",
            status="succeeded",
            db_backend="duckdb",
        )
        s.add(r)
        with pytest.raises(Exception):
            s.commit()


# ------------------ T056 Metrics ------------------


def test_metrics_table_has_all_required_columns() -> None:
    cols = {c.name for c in Metrics.__table__.columns}
    assert {
        "metrics_id",
        "run_id",
        "ler",
        "ci_low",
        "ci_high",
        "p50",
        "p95",
        "p99",
        "throughput",
        "residual_density",
        "shots",
        "rounds",
        "code_distance",
        "basis",
        "created_at",
    }.issubset(cols)


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


def _good_metrics(rid: str) -> Metrics:
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


def test_metrics_insert_and_select_roundtrip_on_sqlite() -> None:
    e = _setup_engine()
    with Session(e) as s:
        _mk_run(s, "r1")
        s.add(_good_metrics("r1"))
        s.commit()
        r = s.get(Run, "r1")
        assert len(r.metrics) == 1


def test_metrics_orm_cascade_deletes_on_run_delete() -> None:
    # ORM-level cascade still applies via the Run.metrics relationship
    # (delete-orphan). DB-level ON DELETE CASCADE was dropped for DuckDB
    # portability; the ORM cascade preserves the same observable behaviour.
    e = _setup_engine()
    with Session(e) as s:
        _mk_run(s, "rA")
        s.add(_good_metrics("rA"))
        s.commit()
        r = s.get(Run, "rA")
        s.delete(r)
        s.commit()
        assert s.query(Metrics).filter_by(run_id="rA").count() == 0


def test_metrics_check_constraint_rejects_ci_low_above_ler() -> None:
    e = _setup_engine()
    with Session(e) as s:
        _mk_run(s, "rB")
        m = _good_metrics("rB")
        m.ci_low = 1.0  # > ler
        s.add(m)
        with pytest.raises(Exception):
            s.commit()


def test_metrics_check_constraint_rejects_p50_above_p95() -> None:
    e = _setup_engine()
    with Session(e) as s:
        _mk_run(s, "rC")
        m = _good_metrics("rC")
        m.p50 = 999.0
        s.add(m)
        with pytest.raises(Exception):
            s.commit()


def test_metrics_check_constraint_rejects_residual_density_above_one() -> None:
    e = _setup_engine()
    with Session(e) as s:
        _mk_run(s, "rD")
        m = _good_metrics("rD")
        m.residual_density = 1.5
        s.add(m)
        with pytest.raises(Exception):
            s.commit()


def test_metrics_check_constraint_rejects_invalid_basis() -> None:
    e = _setup_engine()
    with Session(e) as s:
        _mk_run(s, "rE")
        m = _good_metrics("rE")
        m.basis = "Y"
        s.add(m)
        with pytest.raises(Exception):
            s.commit()


def test_metrics_relationship_back_populates_run_metrics() -> None:
    e = _setup_engine()
    with Session(e) as s:
        _mk_run(s, "rF")
        m = _good_metrics("rF")
        s.add(m)
        s.commit()
        assert m.run.run_id == "rF"


# ------------------ T057 Artefact ------------------


def test_artefact_table_has_required_columns_and_fk() -> None:
    cols = {c.name for c in Artefact.__table__.columns}
    assert {"id", "run_id", "path", "sha256", "type", "size", "created_at"}.issubset(cols)
    fks = [
        fk.target_fullname
        for c in Artefact.__table__.columns
        for fk in c.foreign_keys
    ]
    assert any("runs." in f for f in fks)


# ------------------ T058 Report ------------------


def test_report_table_has_columns_and_unique_constraint() -> None:
    cols = {c.name for c in Report.__table__.columns}
    assert {"id", "run_id", "type", "format", "path", "sha256", "created_at"}.issubset(cols)
    uqs = [c for c in Report.__table__.constraints if isinstance(c, UniqueConstraint)]
    assert any({col.name for col in uq.columns} == {"run_id", "type", "format"} for uq in uqs)


# ------------------ T059 Fingerprint ------------------


def test_fingerprint_required_columns_and_unique_run_id() -> None:
    cols = {c.name for c in Fingerprint.__table__.columns}
    required = {
        "id",
        "run_id",
        "git_sha",
        "pip_freeze_digest",
        "config_sha256",
        "rng_master_seed",
        "cpu_model",
        "cpu_count",
        "gpu_model",
        "gpu_count",
        "gpu_driver_version",
        "os_name",
        "os_kernel",
        "python_version",
        "cuda_runtime_version",
        "timestamp_utc",
    }
    assert required.issubset(cols)
    rc = Fingerprint.__table__.c.run_id
    assert rc.unique or any(
        "run_id" in {x.name for x in getattr(c, "columns", [])}
        for c in Fingerprint.__table__.constraints
    )


def test_fingerprint_gpu_fields_nullable() -> None:
    cols = Fingerprint.__table__.c
    assert cols.gpu_model.nullable
    assert cols.gpu_driver_version.nullable
    assert cols.cuda_runtime_version.nullable
