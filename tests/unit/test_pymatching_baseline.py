"""Unit tests for the PyMatching MWPM baseline decoder (T023).

The test matrix covers the full Decoder Protocol contract plus the
error paths that the benchmark runner relies on to fail loudly rather
than silently produce bogus corrections.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from app.core.capability_report import CapabilityReport
from app.decoders.pymatching_baseline import DecoderInputError, PyMatchingBaseline
from app.decoders.protocol import Corrections, Decoder, DecoderMetadata


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def d3_r3_dem() -> stim.DetectorErrorModel:
    """Deterministic distance-3, rounds-3 surface code DEM.

    Small enough to be fast, non-trivial enough to exercise shape
    validation against both the detector and observable axes.
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.001,
    )
    return circuit.detector_error_model(decompose_errors=True)


@pytest.fixture()
def decoder(d3_r3_dem: stim.DetectorErrorModel) -> PyMatchingBaseline:
    return PyMatchingBaseline(dem=d3_r3_dem)


# --------------------------------------------------------------------------- #
# available()
# --------------------------------------------------------------------------- #


def test_available_returns_ready_when_pymatching_installed(
    decoder: PyMatchingBaseline,
) -> None:
    report = decoder.available()
    assert isinstance(report, CapabilityReport)
    assert report.is_available is True
    assert report.available is True  # both attribute spellings resolve
    assert report.blocker_category == "none"
    assert "pymatching" in report.detected_versions


# --------------------------------------------------------------------------- #
# metadata()
# --------------------------------------------------------------------------- #


def test_metadata_reports_backend_name_and_version(
    decoder: PyMatchingBaseline,
) -> None:
    md = decoder.metadata()
    assert isinstance(md, DecoderMetadata)
    assert md.backend_name == "pymatching_baseline"
    assert md.backend_version  # non-empty
    assert md.schema_version == "1"


def test_metadata_supports_batching_true_supports_gpu_false(
    decoder: PyMatchingBaseline,
) -> None:
    md = decoder.metadata()
    assert md.supports_batching is True
    assert md.supports_gpu is False
    # Reference backend has no learned model on disk.
    assert md.model_path is None
    assert md.model_sha256 is None
    assert md.receptive_field is None


# --------------------------------------------------------------------------- #
# warmup() idempotency
# --------------------------------------------------------------------------- #


def test_warmup_is_idempotent(decoder: PyMatchingBaseline) -> None:
    decoder.warmup()
    matching_first = decoder._matching
    decoder.warmup()
    matching_second = decoder._matching
    # Re-warm must NOT rebuild the graph: object identity is stable.
    assert matching_first is matching_second


# --------------------------------------------------------------------------- #
# decode_batch() happy path + validation errors
# --------------------------------------------------------------------------- #


def test_decode_batch_returns_corrections_with_correct_shape(
    decoder: PyMatchingBaseline, d3_r3_dem: stim.DetectorErrorModel
) -> None:
    batch = 4
    syndromes = np.zeros((batch, d3_r3_dem.num_detectors), dtype=np.uint8)
    result = decoder.decode_batch(syndromes)
    assert isinstance(result, Corrections)
    assert result.predictions.shape == (batch, d3_r3_dem.num_observables)
    assert result.predictions.dtype == np.uint8


def test_decode_batch_latency_ns_is_nonnegative_integer(
    decoder: PyMatchingBaseline, d3_r3_dem: stim.DetectorErrorModel
) -> None:
    syndromes = np.zeros((2, d3_r3_dem.num_detectors), dtype=np.uint8)
    result = decoder.decode_batch(syndromes)
    assert isinstance(result.latency_ns, int)
    assert result.latency_ns >= 0


def test_decode_batch_rejects_wrong_dtype(
    decoder: PyMatchingBaseline, d3_r3_dem: stim.DetectorErrorModel
) -> None:
    bad = np.zeros((2, d3_r3_dem.num_detectors), dtype=np.int32)
    with pytest.raises(DecoderInputError) as excinfo:
        decoder.decode_batch(bad)
    # DecoderInputError carries structured expected/actual for logging.
    assert "uint8" in str(excinfo.value)


def test_decode_batch_rejects_wrong_detector_count(
    decoder: PyMatchingBaseline, d3_r3_dem: stim.DetectorErrorModel
) -> None:
    wrong_detectors = d3_r3_dem.num_detectors + 3
    bad = np.zeros((2, wrong_detectors), dtype=np.uint8)
    with pytest.raises(DecoderInputError) as excinfo:
        decoder.decode_batch(bad)
    assert excinfo.value.expected == d3_r3_dem.num_detectors
    assert excinfo.value.actual == wrong_detectors


def test_decode_batch_rejects_wrong_ndim(decoder: PyMatchingBaseline) -> None:
    bad = np.zeros(7, dtype=np.uint8)  # 1D
    with pytest.raises(DecoderInputError):
        decoder.decode_batch(bad)


def test_decode_batch_before_warmup_raises_or_autowarms(
    d3_r3_dem: stim.DetectorErrorModel,
) -> None:
    # We chose the autowarm contract. Verify: calling decode_batch
    # without explicit warmup completes successfully and flips _warmed.
    d = PyMatchingBaseline(dem=d3_r3_dem)
    assert d._warmed is False
    syndromes = np.zeros((1, d3_r3_dem.num_detectors), dtype=np.uint8)
    d.decode_batch(syndromes)
    assert d._warmed is True


# --------------------------------------------------------------------------- #
# Correctness sanity: zero syndrome → zero correction
# --------------------------------------------------------------------------- #


def test_distance3_rounds3_zero_syndrome_yields_zero_correction(
    decoder: PyMatchingBaseline, d3_r3_dem: stim.DetectorErrorModel
) -> None:
    # With no detection events, MWPM must predict the identity
    # correction (all zeros). Any non-zero output indicates a
    # miscompiled matching graph or shape confusion.
    syndromes = np.zeros((8, d3_r3_dem.num_detectors), dtype=np.uint8)
    result = decoder.decode_batch(syndromes)
    assert result.predictions.sum() == 0


# --------------------------------------------------------------------------- #
# dem_path constructor resolves a .dem file on disk
# --------------------------------------------------------------------------- #


def test_dem_path_constructor_resolves_file(
    tmp_path, d3_r3_dem: stim.DetectorErrorModel
) -> None:
    dem_file = tmp_path / "test.dem"
    with dem_file.open("w") as fh:
        fh.write(str(d3_r3_dem))
    d = PyMatchingBaseline(dem_path=dem_file)
    syndromes = np.zeros((1, d3_r3_dem.num_detectors), dtype=np.uint8)
    result = d.decode_batch(syndromes)
    assert result.predictions.shape == (1, d3_r3_dem.num_observables)


# --------------------------------------------------------------------------- #
# Structural conformance to Decoder Protocol
# --------------------------------------------------------------------------- #


def test_structural_type_check_against_decoder_protocol(
    decoder: PyMatchingBaseline,
) -> None:
    # Runtime-checkable Protocol: PyMatchingBaseline must conform
    # without any inheritance. This test is the single source of truth
    # for plugin-registration guards in the benchmark runner.
    assert isinstance(decoder, Decoder)


# --------------------------------------------------------------------------- #
# Constructor validation
# --------------------------------------------------------------------------- #


def test_constructor_rejects_both_dem_and_path(
    d3_r3_dem: stim.DetectorErrorModel, tmp_path
) -> None:
    dem_file = tmp_path / "x.dem"
    dem_file.write_text(str(d3_r3_dem))
    with pytest.raises(ValueError):
        PyMatchingBaseline(dem=d3_r3_dem, dem_path=dem_file)


def test_constructor_rejects_neither() -> None:
    with pytest.raises(ValueError):
        PyMatchingBaseline()
