"""Unit tests for ``app.core.run_id``.

These tests exercise the properties that actually matter for benchmark
run IDs: regex shape, uniqueness under rapid calls, time-sortability,
UTC anchoring, round-trip fidelity, and rejection of malformed input.

Test weight is concentrated on ``parse_run_id`` validation because that
is where real bugs get introduced — ``generate_run_id`` is a thin
formatter, while ``parse_run_id`` has to reject every shape of bad
input without accepting something subtly broken.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.core.run_id import generate_run_id, parse_run_id

# Canonical shape: YYYYMMDDTHHMMSSZ-<26 Crockford base32 chars>.
_RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9A-HJKMNP-TV-Z]{26}$")

# Prefix length (date+T+time+Z) and full-id length constants used to
# build synthetic strings in parser tests.
_PREFIX_LEN = len("YYYYMMDDTHHMMSSZ")
_ULID_LEN = 26


def _valid_ulid_tail() -> str:
    """Return a known-valid ULID tail by asking the generator for one."""
    return generate_run_id().split("-", 1)[1]


def test_generate_run_id_matches_canonical_regex() -> None:
    run_id = generate_run_id()
    assert _RUN_ID_PATTERN.match(run_id) is not None, run_id


def test_generate_run_id_is_unique_across_rapid_calls() -> None:
    # 500 back-to-back calls will almost certainly land in the same
    # wall-clock second; uniqueness therefore depends on the ULID tail.
    ids = {generate_run_id() for _ in range(500)}
    assert len(ids) == 500


def test_generate_run_ids_sort_lexicographically_by_time() -> None:
    first = generate_run_id()
    # Sleep past a full second boundary so the prefix itself must differ.
    time.sleep(1.05)
    second = generate_run_id()
    assert first < second, (first, second)
    # The prefix alone should also order correctly.
    assert first[:_PREFIX_LEN] < second[:_PREFIX_LEN]


def test_generate_run_id_uses_utc() -> None:
    # Even if the local machine is in a non-UTC zone, the emitted
    # timestamp must match UTC wall-clock time at second resolution.
    before = datetime.now(UTC).replace(microsecond=0)
    run_id = generate_run_id()
    after = datetime.now(UTC).replace(microsecond=0)
    ts, _ = parse_run_id(run_id)
    assert ts.tzinfo is not None
    assert ts.utcoffset() == timedelta(0)
    assert before <= ts <= after + timedelta(seconds=1)


def test_parse_run_id_round_trip() -> None:
    # Use a fixed aware datetime so the round trip is deterministic.
    fixed = datetime(2026, 4, 21, 14, 30, 52, tzinfo=UTC)
    run_id = generate_run_id(now=fixed)
    ts, tail = parse_run_id(run_id)
    assert ts == fixed
    assert len(tail) == _ULID_LEN
    # Tail must round-trip exactly as emitted.
    assert run_id.endswith(tail)
    assert run_id[:_PREFIX_LEN] == "20260421T143052Z"


def test_parse_run_id_rejects_malformed_prefix() -> None:
    valid_tail = _valid_ulid_tail()

    # Assemble malformed IDs from parts so no single long literal
    # appears in the test source.
    bad_inputs = [
        "",
        "not-a-run-id",
        # Wrong separator (underscore instead of hyphen).
        "20260421T143052Z_" + valid_tail,
        # Missing the trailing Z on the timestamp.
        "20260421T143052-" + valid_tail,
        # ISO-8601 extended form with colons — explicitly rejected.
        "2026-04-21T14:30:52Z-" + valid_tail,
        # Impossible calendar date (month 13).
        "20261321T143052Z-" + valid_tail,
        # Impossible time-of-day (hour 25).
        "20260421T253052Z-" + valid_tail,
    ]
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            parse_run_id(bad)


def test_parse_run_id_rejects_bad_ulid_tail() -> None:
    prefix = "20260421T143052Z"
    # Tail contains 'I', which is excluded from Crockford base32.
    tail_with_excluded_letter = "I" + "0" * (_ULID_LEN - 1)
    # Tail is one character short.
    short_tail = "0" * (_ULID_LEN - 1)
    # Tail is one character too long.
    long_tail = "0" * (_ULID_LEN + 1)
    # Tail contains lowercase — canonical form is upper-case only.
    lowercase_tail = "a" * _ULID_LEN

    for bad_tail in (tail_with_excluded_letter, short_tail, long_tail, lowercase_tail):
        with pytest.raises(ValueError):
            parse_run_id(f"{prefix}-{bad_tail}")

    # Non-string input is also rejected.
    with pytest.raises(ValueError):
        parse_run_id(12345)  # type: ignore[arg-type]


def test_generate_run_id_accepts_injected_now() -> None:
    fixed = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
    run_id = generate_run_id(now=fixed)
    assert run_id.startswith("20300102T030405Z-")

    # Aware non-UTC datetimes are normalised to UTC.
    tokyo = timezone(timedelta(hours=9))
    tokyo_noon = datetime(2030, 1, 2, 12, 4, 5, tzinfo=tokyo)  # == 03:04:05 UTC
    run_id_tokyo = generate_run_id(now=tokyo_noon)
    assert run_id_tokyo.startswith("20300102T030405Z-")

    # Naive datetimes are rejected rather than silently treated as UTC.
    naive = datetime(2030, 1, 2, 3, 4, 5)
    with pytest.raises(ValueError):
        generate_run_id(now=naive)
