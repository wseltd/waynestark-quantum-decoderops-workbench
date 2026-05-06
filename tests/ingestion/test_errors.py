"""Tests for the ingestion exception hierarchy.

These tests exercise the observable contract of :mod:`app.ingestion.errors`:

* fields survive ``__init__`` unchanged,
* ``str`` / ``repr`` include every field so tracebacks and structured logs
  stay debuggable,
* the subclass relationship holds so callers can ``except IngestionError``
  and catch both concrete types.

The test effort is weighted toward the ``__str__`` / ``__repr__`` contract
and the subclass invariants: those are the properties other ingestion
modules rely on and the ones most likely to regress silently.
Constructor shape is covered once per class — it is trivial and a larger
suite would be performative rather than protective.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.errors import IngestionError, UnsupportedInputError


def test_ingestion_error_stores_source_and_detail_verbatim() -> None:
    err = IngestionError(source="fixtures/bundle.json", detail="missing variant")
    assert err.source == "fixtures/bundle.json"
    assert err.detail == "missing variant"


def test_ingestion_error_str_contains_every_field(tmp_path: Path) -> None:
    source = str(tmp_path / "input.bin")
    err = IngestionError(source=source, detail="short header")
    text = str(err)
    assert source in text
    assert "short header" in text


def test_ingestion_error_repr_contains_every_field(tmp_path: Path) -> None:
    source = str(tmp_path / "input.bin")
    err = IngestionError(source=source, detail="short header")
    text = repr(err)
    assert source in text
    assert "short header" in text


def test_ingestion_error_is_an_exception() -> None:
    err = IngestionError(source="x", detail="y")
    assert isinstance(err, Exception)


def test_ingestion_error_can_be_raised_and_caught() -> None:
    with pytest.raises(IngestionError) as excinfo:
        raise IngestionError(source="fixtures/a.dem", detail="parse failure")
    assert excinfo.value.source == "fixtures/a.dem"
    assert excinfo.value.detail == "parse failure"


def test_unsupported_input_error_stores_all_four_fields() -> None:
    err = UnsupportedInputError(
        source="payload.bin",
        extension=".bin",
        sniffed_signature="unknown",
        detail="no parser matched",
    )
    assert err.source == "payload.bin"
    assert err.extension == ".bin"
    assert err.sniffed_signature == "unknown"
    assert err.detail == "no parser matched"


def test_unsupported_input_error_is_an_ingestion_error() -> None:
    """Callers must be able to ``except IngestionError`` and catch this."""
    err = UnsupportedInputError(
        source="x",
        extension=".x",
        sniffed_signature="unknown",
        detail="y",
    )
    assert isinstance(err, IngestionError)
    assert isinstance(err, Exception)


def test_unsupported_input_error_caught_by_base_class(tmp_path: Path) -> None:
    source = str(tmp_path / "payload.bin")
    with pytest.raises(IngestionError) as excinfo:
        raise UnsupportedInputError(
            source=source,
            extension=".bin",
            sniffed_signature="unknown",
            detail="no parser matched",
        )
    assert isinstance(excinfo.value, UnsupportedInputError)


def test_unsupported_input_error_str_contains_every_field(tmp_path: Path) -> None:
    source = str(tmp_path / "payload.bin")
    err = UnsupportedInputError(
        source=source,
        extension=".bin",
        sniffed_signature="npy-maybe",
        detail="sniffed signature does not match extension",
    )
    text = str(err)
    assert source in text
    assert ".bin" in text
    assert "npy-maybe" in text
    assert "sniffed signature does not match extension" in text


def test_unsupported_input_error_repr_contains_every_field(tmp_path: Path) -> None:
    source = str(tmp_path / "payload.bin")
    err = UnsupportedInputError(
        source=source,
        extension=".bin",
        sniffed_signature="npy-maybe",
        detail="sniffed signature does not match extension",
    )
    text = repr(err)
    assert source in text
    assert ".bin" in text
    assert "npy-maybe" in text
    assert "sniffed signature does not match extension" in text


def test_unsupported_input_error_allows_empty_extension_and_signature() -> None:
    """In-memory byte dispatch has no extension and may skip sniffing."""
    err = UnsupportedInputError(
        source="<bytes>",
        extension="",
        sniffed_signature="",
        detail="hint not recognised",
    )
    assert err.extension == ""
    assert err.sniffed_signature == ""
    assert "hint not recognised" in str(err)


def test_ingestion_error_preserves_unicode_in_fields() -> None:
    """Customer filenames are not guaranteed to be ASCII."""
    err = IngestionError(source="données.dem", detail="mauvais entête — version inconnue")
    text = str(err)
    assert "données.dem" in text
    assert "mauvais entête" in text


def test_unsupported_input_error_preserves_unicode_in_fields() -> None:
    err = UnsupportedInputError(
        source="données.bin",
        extension=".bin",
        sniffed_signature="inconnu",
        detail="aucun parseur compatible",
    )
    text = str(err)
    assert "données.bin" in text
    assert "inconnu" in text
    assert "aucun parseur compatible" in text


def test_ingestion_error_handles_empty_strings() -> None:
    """Empty strings are legal (e.g. unknown source) and must not crash formatting."""
    err = IngestionError(source="", detail="")
    text = str(err)
    assert "IngestionError" in text
