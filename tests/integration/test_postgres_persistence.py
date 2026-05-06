"""Real PostgreSQL persistence via testcontainers.

Every test runs an actual ``postgres:16-alpine`` container driven by
``testcontainers.postgres.PostgresContainer(driver="psycopg")`` — no
pytest.skip smokes. The module is gated cleanly on docker + testcontainers
availability; when both are present it exercises real schema creation,
real repository round-trips, and real DB-side constraint enforcement.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def _docker_reachable() -> bool:
    try:
        import docker  # noqa: F401
    except ImportError:
        return False
    try:
        import docker as _d

        _d.from_env().ping()
        return True
    except Exception:
        return False


def _testcontainers_importable() -> bool:
    try:
        from testcontainers.postgres import PostgresContainer  # noqa: F401

        return True
    except ImportError:
        return False


_pg_unavailable = not (_testcontainers_importable() and _docker_reachable())

pytestmark = pytest.mark.skipif(
    _pg_unavailable,
    reason=(
        "real Postgres proof requires docker + testcontainers[postgres]; "
        "install: pip install 'testcontainers[postgres]==4.*' and ensure "
        "the local docker daemon is running"
    ),
)


@pytest.fixture(scope="module")
def pg_container():
    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer("postgres:16-alpine", driver="psycopg")
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="module")
def pg_engine(pg_container):
    # Register every ORM Table with Base.metadata before create_all —
    # side-effect imports, same pattern as app.db.schema_init.
    import app.models.artefact  # noqa: F401
    import app.models.fingerprint  # noqa: F401
    import app.models.metrics  # noqa: F401
    import app.models.report  # noqa: F401
    import app.models.run  # noqa: F401
    from app.db.base import Base

    engine = create_engine(pg_container.get_connection_url())
    Base.metadata.create_all(engine)
    return engine


def test_schema_creates_cleanly_on_postgres_16(pg_engine) -> None:
    from sqlalchemy import inspect

    tables = set(inspect(pg_engine).get_table_names())
    assert {"runs", "metrics", "artefacts", "reports", "fingerprints"} <= tables


def test_run_row_roundtrip_matches_duckdb_schema(pg_engine) -> None:
    from app.db.repositories.runs_repo import RunsRepository
    from app.models.run import Run

    with Session(pg_engine) as s:
        repo = RunsRepository(s)
        repo.create(
            Run(
                run_id="pg-run-1",
                config_hash="0" * 64,
                backend="pymatching_baseline",
                status="succeeded",
                db_backend="postgresql",
            )
        )
        s.commit()
        r = repo.get("pg-run-1")
        assert r is not None
        assert r.backend == "pymatching_baseline"
        assert r.db_backend == "postgresql"


def test_metrics_row_roundtrip_fields_match(pg_engine) -> None:
    from app.db.repositories.metrics_repo import MetricsRepository
    from app.db.repositories.runs_repo import RunsRepository
    from app.models.metrics import Metrics
    from app.models.run import Run

    with Session(pg_engine) as s:
        RunsRepository(s).create(
            Run(
                run_id="pg-metrics-1",
                config_hash="0" * 64,
                backend="pymatching_baseline",
                status="succeeded",
                db_backend="postgresql",
            )
        )
        repo = MetricsRepository(s)
        repo.create(
            Metrics(
                run_id="pg-metrics-1",
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
        )
        s.commit()
        rows = repo.get_by_run_id("pg-metrics-1")
        assert len(rows) == 1
        # DOUBLE PRECISION → no 32-bit precision drift.
        assert rows[0].ler == 1e-3


def test_artefact_row_roundtrip_sha256_preserved(pg_engine) -> None:
    from app.db.repositories.artefacts_repo import ArtefactsRepository
    from app.db.repositories.runs_repo import RunsRepository
    from app.models.artefact import Artefact
    from app.models.run import Run

    with Session(pg_engine) as s:
        RunsRepository(s).create(
            Run(
                run_id="pg-artef-1",
                config_hash="0" * 64,
                backend="pymatching_baseline",
                status="succeeded",
                db_backend="postgresql",
            )
        )
        repo = ArtefactsRepository(s)
        sha = "a" * 64
        repo.create(
            Artefact(
                run_id="pg-artef-1",
                path="x.onnx",
                sha256=sha,
                type="onnx",
                size=1234,
            )
        )
        s.commit()
        found = repo.get_by_sha256(sha)
        assert found is not None and found.path == "x.onnx"


def test_fingerprint_unique_run_id_enforced_by_postgres(pg_engine) -> None:
    from app.db.repositories.fingerprint_repo import FingerprintRepository
    from app.db.repositories.runs_repo import RunsRepository
    from app.models.fingerprint import Fingerprint
    from app.models.run import Run

    with Session(pg_engine) as s:
        RunsRepository(s).create(
            Run(
                run_id="pg-fp-1",
                config_hash="0" * 64,
                backend="pymatching_baseline",
                status="succeeded",
                db_backend="postgresql",
            )
        )
        repo = FingerprintRepository(s)
        repo.create(
            Fingerprint(
                run_id="pg-fp-1",
                git_sha="abc",
                pip_freeze_digest="d" * 64,
                config_sha256="e" * 64,
                rng_master_seed=42,
                cpu_model="x86_64",
                cpu_count=4,
                gpu_model="RTX",
                gpu_count=1,
                gpu_driver_version="580",
                os_name="Linux",
                os_kernel="6.8",
                python_version="3.12.13",
                cuda_runtime_version="13.0",
                timestamp_utc=datetime.now(UTC),
            )
        )
        s.commit()

    # Fresh session — second insert on same run_id must fail the
    # UNIQUE(run_id) constraint from T059.
    with Session(pg_engine) as s2:
        s2.add(
            Fingerprint(
                run_id="pg-fp-1",
                git_sha="xyz",
                pip_freeze_digest="f" * 64,
                config_sha256="0" * 64,
                rng_master_seed=99,
                cpu_model="x",
                cpu_count=1,
                os_name="Linux",
                os_kernel="6",
                python_version="3",
                timestamp_utc=datetime.now(UTC),
            )
        )
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            s2.commit()
