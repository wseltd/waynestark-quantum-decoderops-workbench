"""Unit tests for the decoder registry (T031)."""

from __future__ import annotations

from pathlib import Path

import pytest
import stim

from app.decoders.protocol import Decoder
from app.decoders.registry import (
    BACKEND_NAMES,
    DecoderAvailability,
    DecoderConfig,
    UnknownDecoderError,
    available_decoders,
    get_decoder,
    list_decoders,
)


@pytest.fixture(scope="module")
def d3_r3_dem() -> stim.DetectorErrorModel:
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.001,
    )
    return c.detector_error_model(decompose_errors=True)


def test_registry_lists_exactly_five_backends() -> None:
    assert set(list_decoders()) == {
        "pymatching_baseline",
        "ising_fast",
        "ising_accurate",
        "onnx_validation",
        "tensorrt_optional",
    }
    assert len(BACKEND_NAMES) == 5
    # Order is stable.
    assert list(list_decoders()) == list(BACKEND_NAMES)


def test_registry_get_decoder_returns_protocol_compliant_instance(
    d3_r3_dem: stim.DetectorErrorModel,
) -> None:
    d = get_decoder(
        "pymatching_baseline", config=DecoderConfig(dem=d3_r3_dem)
    )
    # Structural Decoder conformance — the registry must never return
    # anything that fails isinstance(Decoder).
    assert isinstance(d, Decoder)


def test_registry_unknown_name_raises_with_valid_names_listed() -> None:
    with pytest.raises(UnknownDecoderError) as excinfo:
        get_decoder("not_a_real_backend")
    msg = str(excinfo.value)
    # Every valid name must appear in the message.
    for name in BACKEND_NAMES:
        assert name in msg


def test_registry_available_decoders_returns_report_per_backend() -> None:
    # No configs supplied: each backend reports its own availability,
    # including backends that can't construct (they report the missing
    # config as a blocker rather than raising).
    results = available_decoders()
    assert len(results) == len(BACKEND_NAMES)
    names = [r.name for r in results]
    assert set(names) == set(BACKEND_NAMES)
    for r in results:
        assert isinstance(r, DecoderAvailability)
        assert r.report.reason  # non-empty reason in both ready and unavailable


def test_registry_does_not_silently_substitute_unavailable_backend(
    tmp_path: Path,
) -> None:
    # Request tensorrt_optional but provide a bogus engine path. The
    # registry must return a TensorRTDecoder (not a pymatching
    # substitute), whose available() reports the missing file.
    fake_engine = tmp_path / "does-not-exist.engine"
    d = get_decoder(
        "tensorrt_optional",
        config=DecoderConfig(engine_path=fake_engine),
    )
    # Metadata confirms backend identity — no silent substitution.
    assert d.metadata().backend_name == "tensorrt_optional"


def test_registry_get_decoder_returns_fresh_instance_per_call(
    d3_r3_dem: stim.DetectorErrorModel,
) -> None:
    # Successive calls must yield DIFFERENT instances — no hidden
    # module-level cache. Shared state would make benchmark runs
    # with different configs observe each other's warmup state.
    cfg = DecoderConfig(dem=d3_r3_dem)
    a = get_decoder("pymatching_baseline", config=cfg)
    b = get_decoder("pymatching_baseline", config=cfg)
    assert a is not b
