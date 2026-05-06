"""Tests for :func:`app.ingestion.signature_sniff.sniff_signature`.

Effort is weighted toward priority ordering and the 512-byte bound:
those are the invariants other ingestion modules will rely on and the
ones most likely to regress silently. Happy-path recognition for each
tag is covered once; the ordering / edge-case battery carries the rest.
"""

from __future__ import annotations

from app.ingestion.signature_sniff import (
    MAX_SNIFF_BYTES,
    NPY_MAGIC,
    sniff_signature,
)


def test_empty_bytes_returns_unknown() -> None:
    assert sniff_signature(b"") == "unknown"


def test_npy_magic_exact_prefix_returns_npy() -> None:
    assert sniff_signature(NPY_MAGIC + b"\x01\x00v\x00{}") == "npy"


def test_npy_magic_only_recognised_at_offset_zero() -> None:
    # A payload that happens to contain the NumPy magic mid-stream is
    # not a .npy file. This guards against over-eager substring matches.
    assert sniff_signature(b"   " + NPY_MAGIC) != "npy"


def test_stim_detector_keyword_returns_stim() -> None:
    assert sniff_signature(b"R 0 1\nDETECTOR rec[-1]\n") == "stim"


def test_stim_qubit_coords_keyword_returns_stim() -> None:
    assert sniff_signature(b"QUBIT_COORDS(0, 0) 0\nH 0\n") == "stim"


def test_stim_observable_include_keyword_returns_stim() -> None:
    assert sniff_signature(b"M 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n") == "stim"


def test_dem_error_marker_returns_dem() -> None:
    assert sniff_signature(b"error(0.01) D0 D1\n") == "dem"


def test_dem_detector_marker_returns_dem() -> None:
    assert sniff_signature(b"detector(0, 0, 0) D0\n") == "dem"


def test_json_object_returns_json() -> None:
    assert sniff_signature(b'{"shots": 1024, "run": "abc"}') == "json"


def test_json_array_returns_json() -> None:
    assert sniff_signature(b"[1, 2, 3]") == "json"


def test_json_with_leading_whitespace_returns_json() -> None:
    assert sniff_signature(b"   \n\t{\"k\": 1}") == "json"


def test_jsonl_first_line_object_returns_json() -> None:
    assert sniff_signature(b'{"decoder":"pymatching"}\n{"decoder":"ising"}\n') == "json"


def test_plain_text_returns_unknown() -> None:
    assert sniff_signature(b"hello world, not a known format\n") == "unknown"


def test_only_whitespace_returns_unknown() -> None:
    # Whitespace-only buffers have no first non-whitespace byte, so the
    # JSON rule cannot fire and we fall through to ``unknown``.
    assert sniff_signature(b"   \n\t\r\n   ") == "unknown"


def test_stim_takes_priority_over_dem() -> None:
    # A Stim circuit annotated with a lowercase "error(" comment must
    # not be reclassified as DEM. Stim markers are checked first.
    payload = b"H 0\nDETECTOR rec[-1]  # error(0.01) comment\n"
    assert sniff_signature(payload) == "stim"


def test_dem_takes_priority_over_json_when_both_markers_present() -> None:
    # DEM text that also contains a '{' later on must not be misread
    # as JSON: the JSON rule only fires on the first non-whitespace byte.
    payload = b"error(0.05) D0 ^ D1 {some trailing noise\n"
    assert sniff_signature(payload) == "dem"


def test_npy_wins_over_accidental_stim_substring_at_offset_zero() -> None:
    # The NumPy check runs first, so a .npy file whose later bytes
    # happen to spell "DETECTOR" is still classified as npy.
    payload = NPY_MAGIC + b"\x01\x00DETECTOR inside npy body"
    assert sniff_signature(payload) == "npy"


def test_stim_marker_just_inside_limit_is_detected() -> None:
    # Padding ends exactly where the marker starts; the last byte of
    # the marker sits at index 511 — still inside the window.
    marker = b"DETECTOR"
    pad_len = MAX_SNIFF_BYTES - len(marker)
    payload = b"X" * pad_len + marker + b" rec[-1]\n"
    assert sniff_signature(payload) == "stim"


def test_stim_marker_past_limit_is_not_detected() -> None:
    # Shifting the marker one byte past the 512-byte window must hide
    # it from the sniffer. This pins the documented contract.
    marker = b"DETECTOR"
    payload = b"X" * MAX_SNIFF_BYTES + marker + b" rec[-1]\n"
    assert sniff_signature(payload) == "unknown"


def test_dem_marker_past_limit_is_not_detected() -> None:
    payload = b"#" * MAX_SNIFF_BYTES + b"error(0.01) D0\n"
    assert sniff_signature(payload) == "unknown"


def test_json_leading_byte_past_whitespace_past_limit_is_not_detected() -> None:
    # 512 bytes of whitespace push the '{' out of the sniff window.
    payload = b" " * MAX_SNIFF_BYTES + b"{}"
    assert sniff_signature(payload) == "unknown"


def test_large_buffer_does_not_match_beyond_window() -> None:
    # A Stim circuit with its first keyword at byte 600 must not be
    # classified as Stim — the scan stops at 512 bytes regardless of
    # input length.
    payload = b"noise " * 200 + b"DETECTOR rec[-1]\n"
    assert len(payload) > MAX_SNIFF_BYTES
    assert sniff_signature(payload) == "unknown"


def test_null_bytes_in_text_still_classify_by_marker() -> None:
    # Corrupted text with embedded NULs should still match a keyword
    # when one is present — the sniffer is a substring check, not a
    # text validator.
    assert sniff_signature(b"\x00\x00QUBIT_COORDS(0, 0) 0\n") == "stim"


def test_returns_are_exact_literal_tags() -> None:
    # Guard against accidental renames: the five return values are
    # part of the module's public contract.
    assert sniff_signature(b"") == "unknown"
    assert sniff_signature(NPY_MAGIC) == "npy"
    assert sniff_signature(b"DETECTOR\n") == "stim"
    assert sniff_signature(b"error(0.1) D0\n") == "dem"
    assert sniff_signature(b"[]") == "json"
