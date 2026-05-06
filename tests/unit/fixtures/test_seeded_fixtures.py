"""Tests for fixtures/ content (T114-T118)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import stim
import torch

REPO = Path(__file__).resolve().parents[3]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_surface_d3_r3_loads_and_sidecar_matches() -> None:
    p = REPO / "fixtures" / "seeded_circuits" / "surface_d3_r3.stim"
    assert p.exists()
    c = stim.Circuit.from_file(str(p))
    assert c.num_detectors > 0 and c.num_observables == 1
    meta = json.loads(p.with_suffix(".meta.json").read_text())
    assert meta["distance"] == 3 and meta["rounds"] == 3
    assert meta["sha256"] == _sha(p)


def test_surface_d5_r5_loads_and_sidecar_matches() -> None:
    p = REPO / "fixtures" / "seeded_circuits" / "surface_d5_r5.stim"
    assert p.exists()
    c = stim.Circuit.from_file(str(p))
    assert c.num_detectors > 0
    meta = json.loads(p.with_suffix(".meta.json").read_text())
    assert meta["distance"] == 5 and meta["rounds"] == 5
    assert meta["sha256"] == _sha(p)


def test_surface_d5_differs_from_d3() -> None:
    p3 = REPO / "fixtures" / "seeded_circuits" / "surface_d3_r3.stim"
    p5 = REPO / "fixtures" / "seeded_circuits" / "surface_d5_r5.stim"
    assert _sha(p3) != _sha(p5)


def test_dem_d3_r3_loads_and_sidecar_matches() -> None:
    p = REPO / "fixtures" / "seeded_dems" / "surface_d3_r3.dem"
    assert p.exists()
    dem = stim.DetectorErrorModel.from_file(str(p))
    assert dem.num_detectors > 0 and dem.num_observables == 1
    meta = json.loads(p.with_suffix(".meta.json").read_text())
    circ = REPO / "fixtures" / "seeded_circuits" / "surface_d3_r3.stim"
    assert meta["source_circuit_sha256"] == _sha(circ)
    assert meta["dem_sha256"] == _sha(p)
    assert meta["decompose_errors"] is True


def test_dem_generator_rejects_mismatched_source(tmp_path: Path) -> None:
    from fixtures.seeded_dems._generate import generate_dem_from_circuit

    # Copy the real circuit but mangle the sidecar so SHA mismatches.
    circ_src = REPO / "fixtures" / "seeded_circuits" / "surface_d3_r3.stim"
    circ_tmp = tmp_path / "circuit.stim"
    circ_tmp.write_bytes(circ_src.read_bytes())
    meta_tmp = circ_tmp.with_suffix(".meta.json")
    meta_tmp.write_text(
        json.dumps({"distance": 3, "rounds": 3, "sha256": "0" * 64})
    )
    with pytest.raises(ValueError):
        generate_dem_from_circuit(circ_tmp, tmp_path / "dem.dem")


def test_expected_metrics_fixture_shape() -> None:
    p = REPO / "fixtures" / "expected_metrics" / "surface_d3_r3_pymatching.json"
    d = json.loads(p.read_text())
    assert d["decoder"] == "pymatching_baseline"
    assert d["distance"] == 3 and d["rounds"] == 3
    assert d["master_seed"] == 42
    assert "point_estimate" in d["logical_error_rate"]
    assert d["logical_error_rate"]["ci_method"] == "bootstrap"
    assert {"p50", "p95", "p99", "unit"} <= set(d["latency_ns"].keys())
    assert "ler_rel_tolerance" in d


def test_fake_ising_stub_loads_and_is_labelled() -> None:
    p = REPO / "fixtures" / "fake_models" / "tiny_ising_fast_stub.pt"
    assert p.exists() and p.stat().st_size < 10240
    d = torch.load(p, map_location="cpu", weights_only=False)
    assert d["_FAKE_STUB"] is True
    assert "stub" in d["not_real"].lower()
    assert d["receptive_field"] == 9
    readme = (p.parent / "README.md").read_text().lower()
    assert "stub" in readme
    assert "vendor/ising-decoding" in readme
