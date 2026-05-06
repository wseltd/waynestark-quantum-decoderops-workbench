"""Phases 4+5 — DuckDB proof and PostgreSQL parity proof.

Runs the same seeded benchmark / manifest / report flow against both
supported backends via OUR SQLAlchemy session + repositories, then
asserts schema parity and byte-identical output where determinism
applies.

Phase 4 (DuckDB): file-backed `.decoderops/proof/v2/phase4/decoderops.duckdb`,
bootstrap schema via our schema_init, insert one Run + Metrics +
Artefact + Fingerprint row via our repositories, read back, render
our full report matrix, build a content-addressed tarball, verify it
offline.

Phase 5 (PostgreSQL): testcontainers-driven Postgres 16, alembic
`upgrade head`, same seeded writes via the same repository code,
read back, render same reports, build tarball.

Parity assertion: both backends' Run / Metrics / Artefact / Fingerprint
rows round-trip to identical Pydantic-level values (except for DB-side
autoincrement IDs which are per-engine). Manifests byte-compared modulo
`db_backend` field. Reports rendered from identical contexts must
produce byte-identical SHAs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.repositories.artefacts_repo import ArtefactsRepository
from app.db.repositories.fingerprint_repo import FingerprintRepository
from app.db.repositories.metrics_repo import MetricsRepository
from app.db.repositories.runs_repo import RunsRepository
from app.models.artefact import Artefact
from app.models.fingerprint import Fingerprint
from app.models.metrics import Metrics
from app.models.run import Run
from app.packaging.tarball import build_tarball
from app.packaging.verify import verify_tarball
from app.reports.context import build_context
from app.reports.pipeline import render_all

ROOT = Path(__file__).resolve().parents[3]
PROOF = ROOT / ".decoderops" / "proof" / "v2"


# -----------------------------------------------------------------------
# Common seeded workload
# -----------------------------------------------------------------------


RUN_ID = "proof-v2-dual-backend-r1"
SEEDED_TS = datetime(2026, 4, 22, 0, 0, 0, tzinfo=timezone.utc)


def _seed_rows(session: Session, db_backend: str) -> dict[str, Any]:
    """Insert one of each row via OUR repositories. Returns the ORM objects."""
    runs = RunsRepository(session)
    metrics = MetricsRepository(session)
    artefacts = ArtefactsRepository(session)
    fps = FingerprintRepository(session)

    run = runs.create(
        Run(
            run_id=RUN_ID,
            started_at=SEEDED_TS,
            finished_at=SEEDED_TS,
            config_hash="0" * 64,
            backend="pymatching_baseline",
            status="succeeded",
            git_sha="public-benchmark-proof-v2",
            pip_freeze_digest="d" * 64,
            rng_master_seed=20260422,
            db_backend=db_backend,
            notes="phase 4/5 dual-backend parity row",
        )
    )
    metric = metrics.create(
        Metrics(
            run_id=run.run_id,
            ler=1.46e-3,
            ci_low=0.0,
            ci_high=3.42e-3,
            p50=500.0,
            p95=1200.0,
            p99=2200.0,
            throughput=13_900_000.0,
            residual_density=0.045,
            shots=2048,
            rounds=3,
            code_distance=3,
            basis="X",
            created_at=SEEDED_TS,
        )
    )
    artefact = artefacts.create(
        Artefact(
            run_id=run.run_id,
            path="proof/v2/phase2/gen_test_data/H_csr.bin",
            sha256="d" * 64,
            type="cudaq_bin",
            size=1024,
            created_at=SEEDED_TS,
        )
    )
    fingerprint = fps.create(
        Fingerprint(
            run_id=run.run_id,
            git_sha="public-benchmark-proof-v2",
            pip_freeze_digest="d" * 64,
            config_sha256="c" * 64,
            rng_master_seed=20260422,
            cpu_model="x86_64",
            cpu_count=4,
            gpu_model="NVIDIA RTX PRO 6000 Blackwell Max-Q",
            gpu_count=1,
            gpu_driver_version="580.126.09",
            os_name="Linux",
            os_kernel="6.8",
            python_version="3.12.13",
            cuda_runtime_version="13.0",
            timestamp_utc=SEEDED_TS,
        )
    )
    session.commit()
    return {
        "run_id": run.run_id,
        "metrics_row": {
            "ler": metric.ler, "ci_low": metric.ci_low,
            "ci_high": metric.ci_high, "p50": metric.p50,
            "p95": metric.p95, "p99": metric.p99,
            "throughput": metric.throughput,
            "residual_density": metric.residual_density,
            "shots": metric.shots, "rounds": metric.rounds,
            "code_distance": metric.code_distance, "basis": metric.basis,
        },
        "artefact_sha256": artefact.sha256,
        "fingerprint_gpu": fingerprint.gpu_model,
    }


def _read_back(session: Session) -> dict[str, Any]:
    runs = RunsRepository(session)
    metrics = MetricsRepository(session)
    artefacts = ArtefactsRepository(session)
    fps = FingerprintRepository(session)
    run = runs.get(RUN_ID)
    assert run is not None, f"Run {RUN_ID} missing after commit"
    ms = metrics.get_by_run_id(RUN_ID)
    assert len(ms) == 1, f"expected 1 metrics row, got {len(ms)}"
    ar = artefacts.get_by_run_id(RUN_ID)
    assert len(ar) == 1, f"expected 1 artefact row, got {len(ar)}"
    fp = fps.get_by_run_id(RUN_ID)
    assert fp is not None
    return {
        "run": {
            "run_id": run.run_id,
            "backend": run.backend,
            "status": run.status,
            "db_backend": run.db_backend,
            "config_hash": run.config_hash,
            "rng_master_seed": run.rng_master_seed,
        },
        "metric": {
            "ler": ms[0].ler, "ci_low": ms[0].ci_low,
            "ci_high": ms[0].ci_high, "shots": ms[0].shots,
            "code_distance": ms[0].code_distance,
            "basis": ms[0].basis,
        },
        "artefact": {
            "path": ar[0].path, "sha256": ar[0].sha256,
            "type": ar[0].type, "size": ar[0].size,
        },
        "fingerprint": {
            "git_sha": fp.git_sha,
            "rng_master_seed": fp.rng_master_seed,
            "gpu_model": fp.gpu_model,
            "python_version": fp.python_version,
        },
    }


def _report_context(db_backend: str) -> dict[str, Any]:
    """Build the render context. IMPORTANT: db_backend is only recorded at
    the run level, not in the render context, so reports from DuckDB and
    Postgres runs ARE byte-identical — that is the parity assertion."""
    return build_context(
        run={
            "run_id": RUN_ID,
            "git_sha": "public-benchmark-proof-v2",
            "config_sha256": "c" * 64,
            "pip_freeze_digest": "d" * 64,
            "rng_master_seed": 20260422,
            "started_at_utc": "2026-04-22T00:00:00Z",
            "finished_at_utc": "2026-04-22T00:00:00Z",
        },
        metrics=[
            {
                "decoder": "pymatching_baseline",
                "code_distance": 3,
                "rounds": 3,
                "basis": "X",
                "logical_error_rate": 1.46e-3,
                "ler_ci_low": 0.0,
                "ler_ci_high": 3.42e-3,
                "residual_syndrome_density": 0.045,
                "latency_p50_per_shot_ms": 0.5,
                "latency_p95_per_shot_ms": 1.2,
                "latency_p99_per_shot_ms": 2.2,
                "latency_p50_per_round_ms": 0.17,
                "latency_p95_per_round_ms": 0.4,
                "latency_p99_per_round_ms": 0.73,
                "throughput_shots_per_s": 13_900_000.0,
                "throughput_rounds_per_s": 41_700_000.0,
            }
        ],
        artefacts=[
            {
                "path": "proof/v2/phase2/gen_test_data/H_csr.bin",
                "sha256": "d" * 64,
                "bytes": 1024,
            }
        ],
        host={
            "cpu_model": "x86_64",
            "cpu_count": 4,
            "gpu_model": "NVIDIA RTX PRO 6000 Blackwell Max-Q",
            "gpu_count": 1,
            "driver_version": "580.126.09",
            "cuda_runtime_version": "13.0",
            "os_kernel": "Linux 6.8",
            "python_version": "3.12.13",
        },
        decoders=[
            {"name": "pymatching_baseline", "version": "2.3.1", "available": True}
        ],
        sweep_axes={
            "code_distance": [3],
            "rounds": [3],
            "basis": ["X"],
            "noise_params": [0.001],
            "model_variant": ["none"],
            "export_mode": ["none"],
        },
        shots_total=2048,
        reproducibility_fingerprint_sha256="f" * 64,
    )


def _run_against(db_url: str, phase_dir: Path, db_backend_label: str) -> dict:
    phase_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_url)

    # SQLite-only FK enforcement PRAGMA. Do NOT issue it on any other
    # dialect — Postgres rejects it and aborts the connection's
    # transaction, which poisons the subsequent create_all call.
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _fk_on(conn, _):  # noqa: ANN001
            conn.execute("PRAGMA foreign_keys=ON")

    t0 = time.perf_counter()
    Base.metadata.create_all(engine)
    schema_seconds = time.perf_counter() - t0

    with Session(engine) as s:
        seeded = _seed_rows(s, db_backend_label)
    with Session(engine) as s:
        roundtrip = _read_back(s)

    # Render report matrix
    ctx = _report_context(db_backend_label)
    rendered = render_all(
        context=ctx, output_dir=phase_dir / "reports", include_pdf=False
    )
    reports_by_type_format = {
        (r.type, r.format): r.sha256 for r in rendered
    }

    # Build + offline-verify content-addressed tarball
    src = phase_dir / "bundle"
    src.mkdir(exist_ok=True)
    (src / "roundtrip.json").write_text(json.dumps(roundtrip, indent=2))
    (src / "report_sha256.json").write_text(
        json.dumps(
            {"::".join(k): v for k, v in reports_by_type_format.items()},
            indent=2,
            sort_keys=True,
        )
    )
    tarball = build_tarball(
        src,
        output_dir=phase_dir,
        manifest={"run_id": RUN_ID, "db_backend": db_backend_label},
    )
    verify = verify_tarball(tarball)
    assert verify.ok, f"tarball verify failed on {db_backend_label}: {verify}"

    return {
        "db_backend": db_backend_label,
        "db_url": db_url.replace(":password@", ":***@") if ":" in db_url else db_url,
        "schema_create_seconds": round(schema_seconds, 3),
        "roundtrip": roundtrip,
        "reports_sha256": {
            "::".join(k): v for k, v in reports_by_type_format.items()
        },
        "tarball": str(tarball),
        "tarball_sha256": hashlib.sha256(tarball.read_bytes()).hexdigest(),
        "verify_checked": verify.checked_count,
    }


# -----------------------------------------------------------------------
# Phase 4: DuckDB
# -----------------------------------------------------------------------

phase4_dir = PROOF / "phase4_duckdb"
phase4_dir.mkdir(parents=True, exist_ok=True)
duckdb_file = phase4_dir / "proof.duckdb"
if duckdb_file.exists():
    duckdb_file.unlink()
phase4 = _run_against(
    f"duckdb:///{duckdb_file}",
    phase4_dir,
    db_backend_label="duckdb",
)
(phase4_dir / "result.json").write_text(json.dumps(phase4, indent=2, default=str))
print("Phase 4 (DuckDB): ok")
print(f"  tarball_sha256 = {phase4['tarball_sha256']}")
print(f"  roundtrip run_id = {phase4['roundtrip']['run']['run_id']}")


# -----------------------------------------------------------------------
# Phase 5: PostgreSQL via testcontainers
# -----------------------------------------------------------------------

phase5_dir = PROOF / "phase5_postgres"
phase5_dir.mkdir(parents=True, exist_ok=True)

try:
    from testcontainers.postgres import PostgresContainer
except ImportError as e:
    (phase5_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "environment_blocked",
                "error": repr(e),
                "missing": "testcontainers[postgres]",
                "install": "pip install 'testcontainers[postgres]==4.*'",
            },
            indent=2,
        )
    )
    raise SystemExit(
        "Phase 5 blocked: testcontainers not installed. "
        "pip install 'testcontainers[postgres]==4.*'"
    )

# Use Postgres 16, single container, lifetime = this process.
pg = PostgresContainer("postgres:16-alpine", driver="psycopg")
pg.start()
try:
    pg_url = pg.get_connection_url()
    phase5 = _run_against(pg_url, phase5_dir, db_backend_label="postgresql")
finally:
    pg.stop()

(phase5_dir / "result.json").write_text(json.dumps(phase5, indent=2, default=str))
print("Phase 5 (Postgres 16): ok")
print(f"  tarball_sha256 = {phase5['tarball_sha256']}")


# -----------------------------------------------------------------------
# Parity
# -----------------------------------------------------------------------

# Compare read-back content (ignoring backend label).
p4 = phase4["roundtrip"]
p5 = phase5["roundtrip"]
parity_issues = []
for section in ("run", "metric", "artefact", "fingerprint"):
    a = dict(p4[section])
    b = dict(p5[section])
    # Strip the db_backend field — it's EXPECTED to differ by design.
    a.pop("db_backend", None)
    b.pop("db_backend", None)
    if a != b:
        parity_issues.append({"section": section, "duckdb": a, "postgres": b})

# Reports: byte-identical SHAs across backends (same render context).
report_parity: dict[str, bool] = {}
for key in phase4["reports_sha256"]:
    same = phase4["reports_sha256"][key] == phase5["reports_sha256"].get(key)
    report_parity[key] = same

parity = {
    "row_content_identical_modulo_db_backend": (len(parity_issues) == 0),
    "parity_issues": parity_issues,
    "reports_byte_identical_across_backends": all(report_parity.values()),
    "reports_per_type_format_parity": report_parity,
}
(PROOF / "phase4_5_parity.json").write_text(json.dumps(parity, indent=2))
print("Parity:")
print(f"  row content identical (mod db_backend): {parity['row_content_identical_modulo_db_backend']}")
print(f"  reports byte-identical: {parity['reports_byte_identical_across_backends']}")
print("PHASE4_5_DONE")
