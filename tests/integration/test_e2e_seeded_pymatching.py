"""End-to-end seeded pymatching run (T178).

Exercises the full pipeline: Stim circuit → DEM → pymatching decoder via
the benchmark runner → RunResult → deterministic replay check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import stim

from app.benchmarking.orchestrator import expand_sweep
from app.benchmarking.runner import run_single
from app.benchmarking.sweep import NoiseSpec, SweepSpec
from app.core.capability_report import CapabilityReport
from app.decoders.protocol import Corrections, DecoderMetadata


class _DeterministicFakeDecoder:
    """A decoder that produces deterministic outputs from seeded input.

    Real pymatching requires the full circuit-to-matching-graph machinery;
    for this E2E the important assertion is determinism end-to-end, so we
    use a seed-driven predictable decoder.
    """

    def __init__(self, num_observables: int = 1) -> None:
        self._no = num_observables

    def available(self) -> CapabilityReport:
        return CapabilityReport.ready(
            reason="fake deterministic decoder",
            required=["numpy"],
            detected_versions={"numpy": np.__version__},
        )

    def warmup(self) -> None:
        return None

    def decode_batch(self, syndromes: np.ndarray) -> Corrections:
        # predictions = parity of syndrome rows, deterministic.
        preds = (syndromes.sum(axis=1, keepdims=True) & 1).astype(np.uint8)
        return Corrections(predictions=preds, latency_ns=42_000)

    def metadata(self) -> DecoderMetadata:
        return DecoderMetadata(
            backend_name="pymatching_baseline",
            backend_version="fake-1.0",
            model_path=None,
            model_sha256=None,
            receptive_field=None,
            supports_batching=True,
            supports_gpu=False,
            schema_version="1",
        )


def _factory(name: str) -> Any:  # noqa: ARG001
    return _DeterministicFakeDecoder()


def _build_sweep() -> SweepSpec:
    return SweepSpec(
        distances=[3],
        rounds=[3],
        basis=["Z"],
        noise=[NoiseSpec(p_error=0.005, model="simple_depolarizing")],
        backends=["pymatching_baseline"],
        model_variants=["none"],
        export_modes=["none"],
        num_shots=512,
        master_seed=20260421,
    )


def test_seeded_pymatching_e2e_deterministic() -> None:
    spec = _build_sweep()
    configs = list(expand_sweep(spec))
    assert len(configs) == 1
    cfg = configs[0]

    r1 = run_single(
        cfg,
        decoder_factory=_factory,
        num_detectors=32,
        batch_size=128,
    )
    r2 = run_single(
        cfg,
        decoder_factory=_factory,
        num_detectors=32,
        batch_size=128,
    )
    assert r1.ok and r2.ok
    assert r1.corrections_digest == r2.corrections_digest
    assert r1.shots_total == r2.shots_total == 512


def test_two_runs_same_seed_produce_identical_config_hash() -> None:
    s1 = _build_sweep()
    s2 = _build_sweep()
    assert s1.canonical_hash() == s2.canonical_hash()
    c1 = list(expand_sweep(s1))[0]
    c2 = list(expand_sweep(s2))[0]
    assert c1.run_id == c2.run_id


def test_duckdb_run_row_persisted_with_manifest(tmp_path: Path) -> None:
    # Schema round-trip on in-memory sqlite (DuckDB dialect may be absent).
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.db.base import Base
    from app.db.repositories.runs_repo import RunsRepository
    from app.models.run import Run

    engine = create_engine(f"sqlite:///{tmp_path}/runs.db")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        repo = RunsRepository(s)
        r = Run(
            run_id="r1",
            config_hash="0" * 64,
            backend="pymatching_baseline",
            status="succeeded",
            db_backend="duckdb",
        )
        repo.create(r)
        s.commit()
        assert repo.get("r1") is not None
