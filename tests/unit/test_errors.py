"""Unit tests for app.core.errors.

These tests protect the public contract used by downstream consumers
(manifest writers, API boundary mappers, report generators). They
deliberately exercise the risk surface — reason-code distinctness, cause
chaining via ``raise ... from e``, and serialisation when ``details`` is
absent — not just the happy path.
"""

from __future__ import annotations

import json

import pytest

from app.core.errors import (
    CapabilityError,
    DecoderError,
    DecoderOpsError,
    IngestionError,
    PackagingError,
)


SUBCLASSES = (IngestionError, DecoderError, PackagingError, CapabilityError)


def test_root_has_default_reason_code() -> None:
    """DecoderOpsError exposes a dotted default_reason_code as a class attribute."""
    assert DecoderOpsError.default_reason_code == "decoderops.error"
    err = DecoderOpsError("boom")
    assert err.reason_code == "decoderops.error"
    assert err.message == "boom"


def test_subclasses_inherit_from_root() -> None:
    """Every domain subclass is a DecoderOpsError and therefore an Exception."""
    for cls in SUBCLASSES:
        assert issubclass(cls, DecoderOpsError)
        assert issubclass(cls, Exception)
        instance = cls("msg")
        assert isinstance(instance, DecoderOpsError)


def test_to_dict_serialises_reason_code_message_and_details() -> None:
    """to_dict returns the deterministic four-key contract and stays JSON-safe."""
    err = CapabilityError(
        "tensorrt missing",
        details={"runtime": "tensorrt", "blocker_kind": "software"},
    )
    payload = err.to_dict()

    assert payload == {
        "type": "CapabilityError",
        "reason_code": "capability.missing_runtime",
        "message": "tensorrt missing",
        "details": {"runtime": "tensorrt", "blocker_kind": "software"},
    }
    # Key order matters for byte-reproducible report output.
    assert list(payload.keys()) == ["type", "reason_code", "message", "details"]
    # Must round-trip through json so manifests are deterministic.
    assert json.loads(json.dumps(payload)) == payload


def test_subclass_default_reason_codes_are_distinct() -> None:
    """Each subclass carries a unique dotted reason_code in its own namespace.

    The compatibility matrix and Risk Register switch on reason_code prefixes,
    so collisions would silently merge unrelated failures into one row.
    """
    codes = {cls.default_reason_code for cls in SUBCLASSES}
    assert len(codes) == len(SUBCLASSES), f"duplicate default_reason_code: {codes}"

    expected_prefixes = {
        IngestionError: "ingestion.",
        DecoderError: "decoder.",
        PackagingError: "packaging.",
        CapabilityError: "capability.",
    }
    for cls, prefix in expected_prefixes.items():
        assert cls.default_reason_code.startswith(prefix), (
            f"{cls.__name__}.default_reason_code={cls.default_reason_code!r} "
            f"must start with {prefix!r}"
        )


def test_raise_from_preserves_cause() -> None:
    """``raise DecoderOpsError(...) from e`` keeps the original cause on __cause__."""
    original = ValueError("bad syndrome shape")
    try:
        try:
            raise original
        except ValueError as e:
            raise IngestionError("could not parse syndrome", details={"source": "syn.bin"}) from e
    except IngestionError as caught:
        assert caught.__cause__ is original
        assert caught.__suppress_context__ is True
        assert caught.details == {"source": "syn.bin"}
    else:
        pytest.fail("IngestionError was not raised")


def test_to_dict_handles_missing_details() -> None:
    """When details is not supplied the serialised payload carries details=None.

    Downstream consumers rely on the key being present (not absent) so JSON
    schemas for the manifest stay stable across runs with and without context.
    """
    err = PackagingError("sha mismatch on tarball")
    payload = err.to_dict()

    assert err.details is None
    assert payload["details"] is None
    assert set(payload.keys()) == {"type", "reason_code", "message", "details"}
    assert payload["type"] == "PackagingError"
    assert payload["reason_code"] == "packaging.sha_mismatch"


def test_custom_reason_code_overrides_default() -> None:
    """A caller-supplied reason_code replaces the class default without subclassing.

    This is the escape hatch that lets ingestion express sub-cases (e.g.
    ``ingestion.unsupported_format``) without fattening the class hierarchy.
    """
    err = IngestionError(
        "unsupported format",
        reason_code="ingestion.unsupported_format",
        details={"mime": "application/x-foo"},
    )
    assert err.reason_code == "ingestion.unsupported_format"
    assert err.to_dict()["reason_code"] == "ingestion.unsupported_format"


def test_repr_includes_class_name_message_reason_code_and_details() -> None:
    """__repr__ surfaces all structured context so tracebacks stay debuggable.

    Without this, a bare ``repr(err)`` would drop ``reason_code``/``details``
    and make log triage harder than it needs to be.
    """
    err = DecoderError(
        "warmup failed",
        details={"backend": "ising_fast", "stage": "warmup"},
    )
    text = repr(err)
    assert text.startswith("DecoderError(")
    assert "message='warmup failed'" in text
    assert "reason_code='decoder.unavailable'" in text
    assert "'backend': 'ising_fast'" in text
