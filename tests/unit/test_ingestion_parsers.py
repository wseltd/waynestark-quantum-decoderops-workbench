"""Consolidated ingestion-parser tests (T015-T019)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import stim

from app.ingestion.dem_parser import DEMParseError, parse_dem_file
from app.ingestion.sinter_logs import (
    InvalidSinterLogError,
    parse_sinter_shot_log,
)
from app.ingestion.stim_circuit import (
    StimCircuitParseError,
    parse_stim_circuit,
)
from app.ingestion.syndrome_parser import (
    InvalidSyndromeInputError,
    parse_syndrome_file,
)


_FIXED_NOW = datetime(2026, 4, 21, tzinfo=timezone.utc)


def _sample_circuit() -> str:
    c = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    return str(c)


def test_parse_stim_circuit_happy_path() -> None:
    src = _sample_circuit()
    n = parse_stim_circuit(
        src,
        ingester_version="0.1.0",
        now_utc_fn=lambda: _FIXED_NOW,
    )
    assert n.provenance.source_kind == "stim_circuit"
    assert n.dem is not None and n.dem.num_detectors >= 1
    assert n.circuit_stim_source == src
    assert n.syndrome is None


def test_parse_stim_circuit_rejects_invalid() -> None:
    with pytest.raises(StimCircuitParseError):
        parse_stim_circuit(
            "NOT A VALID STIM CIRCUIT",
            ingester_version="0.1.0",
            now_utc_fn=lambda: _FIXED_NOW,
        )


def test_parse_dem_file_happy_path(tmp_path: Path) -> None:
    c = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = c.detector_error_model(decompose_errors=True)
    dem_path = tmp_path / "x.dem"
    dem.to_file(str(dem_path))
    n = parse_dem_file(
        dem_path,
        ingester_version="0.1.0",
        now_utc_fn=lambda: _FIXED_NOW,
    )
    assert n.provenance.source_kind == "stim_dem_file"
    assert (
        n.provenance.source_sha256
        == hashlib.sha256(dem_path.read_bytes()).hexdigest()
    )


def test_parse_dem_file_missing(tmp_path: Path) -> None:
    with pytest.raises(DEMParseError):
        parse_dem_file(tmp_path / "nope.dem", ingester_version="0.1.0")


def test_parse_syndrome_npy(tmp_path: Path) -> None:
    arr = np.zeros((4, 6), dtype=np.uint8)
    arr[0, 1] = 1
    p = tmp_path / "s.npy"
    np.save(p, arr)
    r = parse_syndrome_file(p)
    assert r.input_source == "syndrome_npy"
    assert r.syndromes.shape == (4, 6)
    assert r.syndromes.dtype == np.uint8


def test_parse_syndrome_bin_round_trip(tmp_path: Path) -> None:
    arr = np.zeros((4, 6), dtype=np.uint8)
    p = tmp_path / "s.bin"
    p.write_bytes(arr.tobytes())
    r = parse_syndrome_file(p, shape_hint=(4, 6))
    assert r.input_source == "syndrome_bin"
    assert r.syndromes.shape == (4, 6)


def test_parse_syndrome_bin_requires_shape(tmp_path: Path) -> None:
    p = tmp_path / "s.bin"
    p.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(InvalidSyndromeInputError):
        parse_syndrome_file(p)


def test_parse_sinter_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "stats.jsonl"
    p.write_text(
        '{"decoder":"pymatching","shots":100,"errors":3,"discards":0,"seconds":1.2}\n'
        '{"decoder":"pymatching","shots":200,"errors":6,"discards":0,"seconds":2.4}\n'
    )
    r = parse_sinter_shot_log(p)
    assert r["shots_total"] == 300
    assert r["errors_total"] == 9
    assert r["provenance"]["format_detected"] == "jsonl"


def test_parse_sinter_empty_raises(tmp_path: Path) -> None:
    p = tmp_path / "e.jsonl"
    p.write_text("")
    with pytest.raises(InvalidSinterLogError):
        parse_sinter_shot_log(p)


def test_parse_sinter_negative_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"shots":-1,"errors":0,"discards":0,"seconds":0}\n')
    with pytest.raises(InvalidSinterLogError):
        parse_sinter_shot_log(p)


def test_parse_customer_dem_bundle(tmp_path: Path) -> None:
    from app.ingestion.customer_dem import parse_customer_dem_bundle

    c = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = c.detector_error_model(decompose_errors=True)
    dem_path = tmp_path / "a.dem"
    dem.to_file(str(dem_path))
    r = parse_customer_dem_bundle(dem_path, customer_label="acme")
    assert r["customer_label"] == "acme"
    assert r["count"] == 1
