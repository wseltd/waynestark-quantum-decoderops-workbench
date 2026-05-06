"""Byte-reproducibility for the report pipeline (T180)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.reports.context import build_context
from app.reports.pipeline import REPORT_TYPES, render_all


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fixed_context() -> dict:
    return build_context(
        run={
            "run_id": "fixed-1",
            "git_sha": "0" * 40,
            "config_sha256": "c" * 64,
            "pip_freeze_digest": "d" * 64,
            "rng_master_seed": 42,
            "started_at_utc": "2026-01-01T00:00:00Z",
            "finished_at_utc": "2026-01-01T00:00:05Z",
        },
        metrics=[],
        artefacts=[],
        host={
            "cpu_model": "x86",
            "cpu_count": 1,
            "gpu_model": "",
            "gpu_count": 0,
            "driver_version": "",
            "cuda_runtime_version": "",
            "os_kernel": "6.8",
            "python_version": "3.12",
        },
        decoders=[],
        sweep_axes={
            "code_distance": [],
            "rounds": [],
            "basis": [],
            "noise_params": [],
            "model_variant": [],
            "export_mode": [],
        },
        shots_total=0,
        reproducibility_fingerprint_sha256="f" * 64,
    )


def _render_into(tmp_path: Path, sub: str) -> list:
    out_dir = tmp_path / sub
    return render_all(
        context=_fixed_context(), output_dir=out_dir, include_pdf=False
    )


def test_report_matrix_byte_reproducible(tmp_path: Path) -> None:
    a = _render_into(tmp_path, "a")
    b = _render_into(tmp_path, "b")
    for ra, rb in zip(a, b, strict=True):
        assert ra.type == rb.type
        assert ra.format == rb.format
        assert _sha(ra.path) == _sha(rb.path)


def test_engineering_benchmark_byte_reproducible(tmp_path: Path) -> None:
    a = _render_into(tmp_path, "a")
    md = [r for r in a if r.type == "engineering_benchmark" and r.format == "markdown"][0]
    b = _render_into(tmp_path, "b")
    md_b = [r for r in b if r.type == "engineering_benchmark" and r.format == "markdown"][0]
    assert _sha(md.path) == _sha(md_b.path)


def test_report_matrix_covers_all_types() -> None:
    assert set(REPORT_TYPES) == {
        "engineering_benchmark",
        "decoder_comparison",
        "deployment_readiness",
        "artefact_manifest",
        "risk_caveat",
    }
