"""Unit tests for the Decoder Protocol contract (T022).

These tests pin the *contract* — every decoder backend (pymatching
baseline, ising fast/accurate, onnx validation, tensorrt adapter) must
pass these structural checks or it is not a conforming Decoder. The
tests deliberately avoid importing any concrete backend: the contract
has to hold without torch, pymatching, or any heavyweight dependency
loaded.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from app.decoders.protocol import (
    CapabilityReport,
    Corrections,
    Decoder,
    DecoderMetadata,
)


# --------------------------------------------------------------------------- #
# Decoder protocol: structural / runtime-checkable
# --------------------------------------------------------------------------- #


class _ConformingDecoder:
    """Minimal stand-in used to exercise structural conformance.

    Implements all four Decoder methods with stub bodies — no real
    library imports — so the tests cover the Protocol contract purely.
    """

    def available(self) -> CapabilityReport:
        return CapabilityReport.ready(
            reason="stub ready",
            required=["stub-stdlib"],
            detected_versions={},
        )

    def warmup(self) -> None:
        return None

    def decode_batch(self, syndromes: np.ndarray) -> Corrections:
        batch = int(syndromes.shape[0])
        preds = np.zeros((batch, 1), dtype=np.uint8)
        return Corrections(predictions=preds, latency_ns=0)

    def metadata(self) -> DecoderMetadata:
        return DecoderMetadata(
            backend_name="stub",
            backend_version="0.0",
            model_path=None,
            model_sha256=None,
            receptive_field=None,
            supports_batching=True,
            supports_gpu=False,
            schema_version="1",
        )


def test_decoder_is_runtime_checkable_protocol() -> None:
    # The contract leans on @runtime_checkable so the benchmark runner
    # can guard plugin registration with isinstance(obj, Decoder).
    # A class with all four methods must pass; missing any method must fail.
    assert isinstance(_ConformingDecoder(), Decoder)

    class MissingMetadata:
        def available(self) -> CapabilityReport: ...  # pragma: no cover - stub
        def warmup(self) -> None: ...  # pragma: no cover - stub
        def decode_batch(self, s: np.ndarray) -> Corrections: ...  # pragma: no cover - stub

    assert not isinstance(MissingMetadata(), Decoder)


# --------------------------------------------------------------------------- #
# DecoderMetadata
# --------------------------------------------------------------------------- #


def _sample_metadata(**overrides: object) -> DecoderMetadata:
    defaults = dict(
        backend_name="pymatching_baseline",
        backend_version="2.3.1",
        model_path=None,
        model_sha256=None,
        receptive_field=None,
        supports_batching=True,
        supports_gpu=False,
        schema_version="1",
    )
    defaults.update(overrides)
    return DecoderMetadata(**defaults)  # type: ignore[arg-type]


def test_decoder_metadata_is_frozen() -> None:
    # Manifests embed this object verbatim. Once constructed, it must
    # not be mutable; otherwise two reads of the same run manifest in
    # memory could see different values.
    m = _sample_metadata()
    with pytest.raises(ValidationError):
        m.backend_name = "mutated"  # type: ignore[misc]


def test_decoder_metadata_round_trips_json() -> None:
    # The run manifest writer serialises metadata with model_dump_json
    # and readers reconstruct with model_validate_json. The round trip
    # must be lossless for every field, including the None-valued
    # optional model_* fields that pymatching exercises.
    original = _sample_metadata(
        backend_name="ising_fast",
        backend_version="0.1.0",
        model_path="vendor/Ising-Decoding/models/Ising-Decoder-SurfaceCode-1-Fast.pt",
        model_sha256="08ce4b1e09e3f396" + "0" * 48,
        receptive_field=9,
        supports_gpu=True,
    )
    encoded = original.model_dump_json()
    decoded = DecoderMetadata.model_validate_json(encoded)
    assert decoded == original
    # Ensure the JSON form is a plain object, not wrapped in any
    # non-standard envelope — keeps on-disk manifests trivially greppable.
    payload = json.loads(encoded)
    assert isinstance(payload, dict)
    assert payload["schema_version"] == "1"


def test_decoder_metadata_schema_version_is_one() -> None:
    # schema_version is a Literal['1']. Anything else must be rejected
    # at validation time; downstream readers rely on the pin to decide
    # whether a migration is needed.
    with pytest.raises(ValidationError):
        _sample_metadata(schema_version="2")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# CapabilityReport (re-exported from app.core.capability_report)
# --------------------------------------------------------------------------- #


def test_capability_report_reason_required_when_unavailable() -> None:
    # "Capability unavailable with empty reason" silently drops useful
    # signal out of the Risk Register. The schema has reason with
    # min_length=1; an empty string must raise at construction.
    with pytest.raises(ValidationError):
        CapabilityReport.unavailable(reason="", required=["torch"], category="software")

    # And the factory itself forbids category='none' for unavailable —
    # it's reserved for the ready path.
    with pytest.raises(ValueError):
        CapabilityReport.unavailable(
            reason="something", required=["x"], category="none"
        )


def test_capability_report_category_is_validated_literal() -> None:
    # The Literal contract must reject typo'd category values at
    # construction; silent acceptance of 'not-installed' would break
    # Risk Register joins later.
    with pytest.raises(ValidationError):
        CapabilityReport(
            available=False,
            reason="torch missing",
            required=["torch"],
            blocker_category="not-installed",  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# Corrections
# --------------------------------------------------------------------------- #


def test_corrections_shape_matches_batch_and_observables() -> None:
    preds = np.zeros((7, 3), dtype=np.uint8)
    c = Corrections(predictions=preds, latency_ns=1234)
    assert c.batch_size == 7
    assert c.num_observables == 3
    assert c.latency_ns == 1234

    # Shape / dtype / latency validation — each guarded individually so
    # the error messages are actionable at the call site.
    with pytest.raises(ValueError):
        Corrections(predictions=np.zeros(5, dtype=np.uint8), latency_ns=0)  # 1D
    with pytest.raises(TypeError):
        Corrections(predictions=np.zeros((2, 2), dtype=np.int32), latency_ns=0)
    with pytest.raises(ValueError):
        Corrections(predictions=preds, latency_ns=-1)
    with pytest.raises(TypeError):
        Corrections(predictions=[[0, 1]], latency_ns=0)  # type: ignore[arg-type]
