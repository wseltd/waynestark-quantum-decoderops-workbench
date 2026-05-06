"""Initial schema (T071).

Revision ID: 0001_initial
Revises:
Create Date: 2026-01-01 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(36), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("backend", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("git_sha", sa.String(40), nullable=True),
        sa.Column("pip_freeze_digest", sa.String(64), nullable=True),
        sa.Column("rng_master_seed", sa.BigInteger, nullable=True),
        sa.Column("db_backend", sa.String(16), nullable=False),
        sa.Column("notes", sa.Text, nullable=True),
    )
    op.create_index("ix_runs_started_at", "runs", ["started_at"])
    op.create_index(
        "ix_runs_backend_status", "runs", ["backend", "status"]
    )

    op.create_table(
        "metrics",
        sa.Column("metrics_id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("runs.run_id"),
            nullable=False,
            index=True,
        ),
        sa.Column("ler", sa.Double, nullable=False),
        sa.Column("ci_low", sa.Double, nullable=False),
        sa.Column("ci_high", sa.Double, nullable=False),
        sa.Column("p50", sa.Double, nullable=False),
        sa.Column("p95", sa.Double, nullable=False),
        sa.Column("p99", sa.Double, nullable=False),
        sa.Column("throughput", sa.Double, nullable=False),
        sa.Column("residual_density", sa.Double, nullable=False),
        sa.Column("shots", sa.BigInteger, nullable=False),
        sa.Column("rounds", sa.Integer, nullable=False),
        sa.Column("code_distance", sa.Integer, nullable=False),
        sa.Column("basis", sa.String(8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "artefacts",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("runs.run_id"),
            nullable=False,
            index=True,
        ),
        sa.Column("path", sa.String, nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False, index=True),
        sa.Column("type", sa.String, nullable=False),
        sa.Column("size", sa.BigInteger, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "reports",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("runs.run_id"),
            nullable=False,
            index=True,
        ),
        sa.Column("type", sa.String, nullable=False),
        sa.Column("format", sa.String, nullable=False),
        sa.Column("path", sa.String, nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "run_id", "type", "format", name="uq_reports_run_type_format"
        ),
    )

    op.create_table(
        "fingerprints",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("runs.run_id"),
            nullable=False,
            unique=True,
            index=True,
        ),
        sa.Column("git_sha", sa.String(40), nullable=False),
        sa.Column("pip_freeze_digest", sa.String(64), nullable=False),
        sa.Column("config_sha256", sa.String(64), nullable=False),
        sa.Column("rng_master_seed", sa.BigInteger, nullable=False),
        sa.Column("cpu_model", sa.String, nullable=False),
        sa.Column("cpu_count", sa.Integer, nullable=False),
        sa.Column("gpu_model", sa.String, nullable=True),
        sa.Column("gpu_count", sa.Integer, nullable=False),
        sa.Column("gpu_driver_version", sa.String, nullable=True),
        sa.Column("os_name", sa.String, nullable=False),
        sa.Column("os_kernel", sa.String, nullable=False),
        sa.Column("python_version", sa.String, nullable=False),
        sa.Column("cuda_runtime_version", sa.String, nullable=True),
        sa.Column("timestamp_utc", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("fingerprints")
    op.drop_table("reports")
    op.drop_table("artefacts")
    op.drop_table("metrics")
    op.drop_index("ix_runs_backend_status", table_name="runs")
    op.drop_index("ix_runs_started_at", table_name="runs")
    op.drop_table("runs")
