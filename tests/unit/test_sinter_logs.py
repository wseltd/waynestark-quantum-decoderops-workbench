"""Regression-proof tests for app.ingestion.sinter_logs.

Guards the whitespace-padded-header bug found while routing real
`sinter combine` output through the product (public benchmark proof
phase 2). Every test in this file would fail on the pre-fix parser.

Fixture source: `.decoderops/proof/phase2/stats_combined.csv` —
a verbatim capture of `sinter combine` output from stim-generated
rotated surface-code memory circuits at d∈{3,5,7}, basis∈{X,Z},
p∈{0.001,0.003,0.005,0.01}. Copied verbatim (bytes preserved) into
tests/unit/fixtures/sinter_real_padded.csv.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from app.ingestion.sinter_logs import (
    InvalidSinterLogError,
    parse_sinter_shot_log,
)

REAL_PADDED = (
    Path(__file__).parent / "fixtures" / "sinter_real_padded.csv"
)


# --- real padded fixture (would have broken the old parser) ------------------


def test_parse_real_sinter_combine_csv_aggregates_shots_total_nonzero() -> None:
    """The pre-fix parser returned shots_total=0 on this exact CSV."""
    r = parse_sinter_shot_log(REAL_PADDED)
    assert r["shots_total"] > 0
    # Real CSV: 24 sweep cells, >300k shots when the full sweep is committed.
    assert r["shots_total"] >= 100_000, (
        f"shots_total looks implausibly low: {r['shots_total']}"
    )
    assert r["errors_total"] > 0
    assert r["provenance"]["format_detected"] == "csv"
    assert r["provenance"]["line_count"] >= 10


def test_parse_csv_normalises_whitespace_padded_headers(tmp_path: Path) -> None:
    p = tmp_path / "padded.csv"
    p.write_text(
        "     shots,    errors,  discards, seconds,decoder,strong_id,json_metadata\n"
        "       100,         5,         0,   0.010,pymatching,abc,\"{\"\"d\"\":3}\"\n"
    )
    r = parse_sinter_shot_log(p)
    assert r["shots_total"] == 100
    assert r["errors_total"] == 5
    # Task dict keys are stripped of the leading whitespace.
    t = r["tasks"][0]
    assert "shots" in t
    assert " shots" not in t  # proves the old buggy shape doesn't leak


def test_parse_csv_rejects_missing_shots_column(tmp_path: Path) -> None:
    p = tmp_path / "no_shots.csv"
    p.write_text("errors,seconds\n3,0.1\n")
    with pytest.raises(InvalidSinterLogError) as exc:
        parse_sinter_shot_log(p)
    assert "'shots'" in str(exc.value)


def test_parse_csv_rejects_missing_errors_column(tmp_path: Path) -> None:
    p = tmp_path / "no_errors.csv"
    p.write_text("shots,seconds\n100,0.1\n")
    with pytest.raises(InvalidSinterLogError) as exc:
        parse_sinter_shot_log(p)
    assert "'errors'" in str(exc.value)


def test_parse_csv_rejects_nonnumeric_shots(tmp_path: Path) -> None:
    p = tmp_path / "nan.csv"
    p.write_text("shots,errors,seconds\nNOPE,5,0.1\n")
    with pytest.raises(InvalidSinterLogError) as exc:
        parse_sinter_shot_log(p)
    assert "non-integer" in str(exc.value) or "shots" in str(exc.value)


def test_parse_csv_rejects_errors_greater_than_shots(tmp_path: Path) -> None:
    p = tmp_path / "invariant.csv"
    p.write_text("shots,errors,seconds\n10,20,0.1\n")
    with pytest.raises(InvalidSinterLogError) as exc:
        parse_sinter_shot_log(p)
    assert "exceed" in str(exc.value)


def test_parse_csv_rejects_header_only_empty_data(tmp_path: Path) -> None:
    p = tmp_path / "header_only.csv"
    p.write_text("shots,errors,seconds\n")
    # Zero rows is OK — aggregation yields 0/0/0. The file was not empty.
    r = parse_sinter_shot_log(p)
    assert r["shots_total"] == 0
    assert r["provenance"]["line_count"] == 0


def test_parse_csv_preserves_column_order_after_normalisation(tmp_path: Path) -> None:
    # sinter emits columns in a documented order; the parser must not
    # reorder them or drop the custom_counts tail column.
    p = tmp_path / "ordered.csv"
    p.write_text(
        "     shots,    errors,  discards, seconds,decoder,strong_id,json_metadata,custom_counts\n"
        "       100,         5,         0,   0.010,pymatching,sid,\"{\"\"d\"\":3}\",\n"
    )
    r = parse_sinter_shot_log(p)
    t = r["tasks"][0]
    # All expected keys present after normalisation.
    for k in ("shots", "errors", "discards", "seconds", "decoder",
              "strong_id", "json_metadata", "custom_counts"):
        assert k in t, f"lost column {k!r} after header normalisation"


def test_parse_csv_embedded_json_metadata_decodes(tmp_path: Path) -> None:
    p = tmp_path / "jm.csv"
    p.write_text(
        'shots,errors,seconds,json_metadata\n'
        '100,5,0.01,"{""d"":3,""p"":0.005}"\n'
    )
    r = parse_sinter_shot_log(p)
    jm = r["tasks"][0]["json_metadata"]
    assert jm == {"d": 3, "p": 0.005}


def test_shots_total_matches_sum_of_row_shots(tmp_path: Path) -> None:
    """Explicit invariant: aggregated total == sum of per-row shots."""
    r = parse_sinter_shot_log(REAL_PADDED)
    summed = sum(int(t["shots"]) for t in r["tasks"])
    assert summed == r["shots_total"]
    errors_summed = sum(int(t["errors"]) for t in r["tasks"])
    assert errors_summed == r["errors_total"]


def test_empty_file_still_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(InvalidSinterLogError) as exc:
        parse_sinter_shot_log(p)
    assert "empty" in str(exc.value).lower()
