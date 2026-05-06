"""Tests for the Ising-Decoding bundle error type.

The parser itself is implemented in a sibling ticket; this file pins
the contract of the exception class so downstream code (and the parser's
own tests) can rely on identity, inheritance, and message-passing
behaviour without re-validating those properties everywhere.
"""

from __future__ import annotations

import copy

import pytest

from app.ingestion.ising_bundle_errors import InvalidIsingBundleError


def test_invalid_ising_bundle_error_is_exception_subclass():
    # The parser is expected to raise this from various failure paths;
    # callers must be able to catch it via ``except Exception``.
    assert issubclass(InvalidIsingBundleError, Exception)


def test_invalid_ising_bundle_error_carries_message_through_args():
    # Exception messages are part of the contract — the parser will
    # embed absolute paths and offending field names in them, and tests
    # downstream assert on substrings. Verify args/str round-trip.
    message = "missing checkpoint at /vendor/Ising-Decoding/models/missing.pt"
    err = InvalidIsingBundleError(message)

    assert err.args == (message,)
    assert str(err) == message


def test_invalid_ising_bundle_error_can_be_raised_and_caught_specifically():
    # Down-stream code uses ``except InvalidIsingBundleError`` so that
    # generic system errors do not get swallowed by bundle handling.
    with pytest.raises(InvalidIsingBundleError) as exc_info:
        raise InvalidIsingBundleError("variant 'turbo' is not supported")

    assert "turbo" in str(exc_info.value)


def test_invalid_ising_bundle_error_supports_exception_chaining():
    # The parser maps low-level IO/format errors to InvalidIsingBundleError
    # using ``raise ... from e``. Confirm the chain is preserved so that
    # tracebacks remain useful in production logs.
    original = FileNotFoundError("ising_assets.json missing")
    try:
        try:
            raise original
        except FileNotFoundError as e:
            raise InvalidIsingBundleError("assets manifest unreadable") from e
    except InvalidIsingBundleError as wrapped:
        assert wrapped.__cause__ is original
        assert isinstance(wrapped.__cause__, FileNotFoundError)


def test_invalid_ising_bundle_error_is_deepcopyable():
    # Exceptions cross worker boundaries (pytest-xdist, multiprocessing)
    # via copy semantics, not pickle, in our test stack. Verify deepcopy
    # preserves the message so error reporting stays intact.
    err = InvalidIsingBundleError("sha256 mismatch for Ising-Decoder-SurfaceCode-1-Fast.pt")
    duplicate = copy.deepcopy(err)

    assert isinstance(duplicate, InvalidIsingBundleError)
    assert str(duplicate) == str(err)


def test_invalid_ising_bundle_error_has_descriptive_docstring():
    # The docstring is the only place that enumerates the parser's
    # failure modes for users reading help() output. If it disappears,
    # the exception becomes opaque.
    doc = InvalidIsingBundleError.__doc__ or ""

    assert "variant" in doc
    assert "receptive_field" in doc
    assert "SHA256" in doc
    assert "checkpoint" in doc
    assert "suffix" in doc


def test_module_exports_only_the_error_class():
    # The ticket explicitly forbids any other classes or functions in
    # this module. Guard against future drift by asserting the public
    # surface stays minimal.
    from app.ingestion import ising_bundle_errors

    public = [name for name in vars(ising_bundle_errors) if not name.startswith("_")]
    # ``annotations`` is added by ``from __future__ import annotations``;
    # filter it so the assertion targets actual exports.
    public = [name for name in public if name != "annotations"]

    assert public == ["InvalidIsingBundleError"]
