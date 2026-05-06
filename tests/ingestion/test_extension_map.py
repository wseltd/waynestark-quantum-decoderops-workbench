"""Tests for :mod:`app.ingestion.extension_map`.

The lookup is small, so the tests focus on the normalisation boundary —
leading-dot handling, case, and malformed inputs — rather than on each
supported extension in isolation. The goal is to catch silent regressions
where a refactor accepts inputs the dispatcher should reject (``" .stim"``,
``"..stim"``) or rejects inputs it should accept (``"STIM"``).
"""

from __future__ import annotations

import pytest

from app.ingestion.extension_map import SUPPORTED_EXTENSIONS, extension_to_parser_key


@pytest.mark.parametrize(
    ("ext", "expected"),
    [
        (".stim", "stim"),
        (".dem", "dem"),
        (".npy", "npy"),
        (".bin", "bin"),
        (".jsonl", "jsonl"),
        (".json", "json"),
    ],
)
def test_dotted_lowercase_extension_returns_parser_key(ext: str, expected: str) -> None:
    assert extension_to_parser_key(ext) == expected


@pytest.mark.parametrize(
    ("ext", "expected"),
    [
        ("stim", "stim"),
        ("dem", "dem"),
        ("npy", "npy"),
        ("bin", "bin"),
        ("jsonl", "jsonl"),
        ("json", "json"),
    ],
)
def test_bare_lowercase_extension_is_normalised(ext: str, expected: str) -> None:
    assert extension_to_parser_key(ext) == expected


@pytest.mark.parametrize(
    ("ext", "expected"),
    [
        (".STIM", "stim"),
        (".Dem", "dem"),
        (".NpY", "npy"),
        ("JSON", "json"),
        ("JsOnL", "jsonl"),
    ],
)
def test_extension_is_case_insensitive(ext: str, expected: str) -> None:
    assert extension_to_parser_key(ext) == expected


@pytest.mark.parametrize(
    "ext",
    [".txt", ".parquet", ".csv", ".yaml", ".pt", ".onnx"],
)
def test_unsupported_extension_returns_none(ext: str) -> None:
    assert extension_to_parser_key(ext) is None


def test_empty_string_returns_none() -> None:
    assert extension_to_parser_key("") is None


def test_bare_dot_returns_none() -> None:
    # ``"."`` becomes ``"."`` after normalisation and is not a key in the
    # map. Guards against a future implementation that strips the dot and
    # then looks up the empty string.
    assert extension_to_parser_key(".") is None


def test_doubled_leading_dot_is_not_accepted() -> None:
    # Only a single leading dot is synthesised. ``"..stim"`` is malformed
    # and must not be silently coerced to ``".stim"``.
    assert extension_to_parser_key("..stim") is None


@pytest.mark.parametrize("ext", [" .stim", ".stim ", "\t.dem", ".json\n"])
def test_surrounding_whitespace_is_not_trimmed(ext: str) -> None:
    # Trimming is the caller's responsibility. Accepting these inputs
    # would hide upstream bugs where an extension is pulled from a
    # partially-parsed filename.
    assert extension_to_parser_key(ext) is None


def test_extension_with_embedded_path_separator_is_not_accepted() -> None:
    # Defensive: a caller that passes ``"dir/.stim"`` as an extension has
    # a bug. We should not paper over it by matching on a suffix.
    assert extension_to_parser_key("dir/.stim") is None


def test_supported_extensions_is_frozenset() -> None:
    assert isinstance(SUPPORTED_EXTENSIONS, frozenset)


def test_supported_extensions_is_exactly_the_documented_six() -> None:
    expected = {".stim", ".dem", ".npy", ".bin", ".jsonl", ".json"}
    assert set(SUPPORTED_EXTENSIONS) == expected


def test_every_supported_extension_resolves_to_a_parser_key() -> None:
    # If the map and the frozenset ever drift, this fails loudly instead
    # of silently returning ``None`` from a lookup the caller expected to
    # succeed.
    for ext in SUPPORTED_EXTENSIONS:
        key = extension_to_parser_key(ext)
        assert key is not None, ext
        assert isinstance(key, str)
