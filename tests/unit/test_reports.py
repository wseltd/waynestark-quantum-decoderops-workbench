"""Consolidated tests for the reports layer (T073-T090)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.metrics.compatibility import build_compatibility_status
from app.reports.compatibility_matrix import build_compatibility_matrix
from app.reports.context import build_context
from app.reports.html_renderer import render_html
from app.reports.json_renderer import render_json
from app.reports.markdown_renderer import render_markdown
from app.reports.pipeline import REPORT_TYPES, RenderedReport, render_all
from app.reports.risk_register import build_risk_register


def _sample_context() -> dict:
    return build_context(
        run={
            "run_id": "r-123",
            "git_sha": "abc",
            "config_sha256": "c" * 64,
            "pip_freeze_digest": "d" * 64,
            "rng_master_seed": 42,
            "started_at_utc": "2026-01-01T00:00:00Z",
            "finished_at_utc": "2026-01-01T00:00:05Z",
        },
        metrics=[
            {
                "decoder": "pymatching",
                "code_distance": 3,
                "rounds": 3,
                "basis": "X",
                "logical_error_rate": 1e-3,
                "ler_ci_low": 5e-4,
                "ler_ci_high": 2e-3,
                "residual_syndrome_density": 0.1,
                "latency_p50_per_shot_ms": 0.5,
                "latency_p95_per_shot_ms": 0.9,
                "latency_p99_per_shot_ms": 1.5,
                "latency_p50_per_round_ms": 0.2,
                "latency_p95_per_round_ms": 0.4,
                "latency_p99_per_round_ms": 0.8,
                "throughput_shots_per_s": 1000.0,
                "throughput_rounds_per_s": 3000.0,
            }
        ],
        artefacts=[{"path": "a.onnx", "sha256": "a" * 64, "bytes": 10}],
        host={
            "cpu_model": "x86",
            "cpu_count": 4,
            "gpu_model": "RTX",
            "gpu_count": 1,
            "driver_version": "560",
            "cuda_runtime_version": "13",
            "os_kernel": "6.8",
            "python_version": "3.12",
        },
        decoders=[
            {
                "name": "pymatching_baseline",
                "version": "2.0",
                "available": True,
                "unavailable_reason": "",
            }
        ],
        sweep_axes={
            "code_distance": [3, 5],
            "rounds": [3, 5],
            "basis": ["X"],
            "noise_params": [{"p": 1e-3}],
            "model_variant": ["baseline"],
            "export_mode": ["native"],
        },
        shots_total=1024,
        reproducibility_fingerprint_sha256="f" * 64,
    )


def test_markdown_renderer_engineering_benchmark() -> None:
    out = render_markdown("engineering_benchmark", _sample_context())
    assert "Engineering Benchmark Report" in out
    assert "r-123" in out
    assert "pymatching" in out


def test_html_renderer_engineering_benchmark() -> None:
    out = render_html("engineering_benchmark", _sample_context())
    assert "<html" in out
    assert "r-123" in out


def test_json_renderer_is_canonical() -> None:
    out = render_json({"b": 2, "a": 1})
    assert out == '{"a":1,"b":2}'


def test_render_all_produces_three_formats_per_type(tmp_path: Path) -> None:
    rendered = render_all(
        context=_sample_context(), output_dir=tmp_path, include_pdf=False
    )
    types = {(r.type, r.format) for r in rendered}
    assert len(types) == 3 * len(REPORT_TYPES)
    for r in rendered:
        assert Path(r.path).exists()
        assert len(r.sha256) == 64
        assert isinstance(r, RenderedReport)


def test_render_all_is_byte_reproducible(tmp_path: Path) -> None:
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    ctx = _sample_context()
    r1 = render_all(context=ctx, output_dir=d1)
    r2 = render_all(context=ctx, output_dir=d2)
    assert [r.sha256 for r in r1] == [r.sha256 for r in r2]


def test_compatibility_matrix_sorted() -> None:
    statuses = [
        build_compatibility_status(
            "tensorrt_optional", "unavailable", "no gpu", "machine"
        ),
        build_compatibility_status(
            "pymatching_baseline", "ready", "ok", "none"
        ),
    ]
    m = build_compatibility_matrix(statuses)
    assert list(m.keys()) == sorted(m.keys())


def test_risk_register_excludes_ready() -> None:
    statuses = [
        build_compatibility_status(
            "pymatching_baseline", "ready", "ok", "none"
        ),
        build_compatibility_status(
            "tensorrt_optional", "unavailable", "no gpu", "machine"
        ),
    ]
    rows = build_risk_register(statuses)
    assert len(rows) == 1
    assert rows[0]["backend"] == "tensorrt_optional"
    assert rows[0]["severity"] == "blocker"


def test_template_files_exist() -> None:
    templates_dir = Path("app/reports/templates")
    for t in (
        "engineering_benchmark",
        "decoder_comparison",
        "deployment_readiness",
        "artefact_manifest",
        "risk_caveat",
    ):
        assert (templates_dir / f"{t}.md.j2").exists()
        assert (templates_dir / f"{t}.html.j2").exists()
