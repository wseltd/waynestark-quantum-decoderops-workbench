"""Tests for the ingestion envelope schema.

These tests focus on the cross-field invariant between
provenance.source_kind and the populated payload field -- that is the
contract every downstream parser (T015 stim_circuit, T016 dem_parser,
later Sinter / customer DEM parsers) must respect, and the only way to
catch a parser that silently sets the wrong combination.

Coverage is intentionally uneven: the model_validator and the
ULID/SHA256 boundary validators get many adversarial cases; the simple
field bounds get one each.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.ingestion.schema import (
    SCHEMA_VERSION,
    DEMRef,
    NormalisedInput,
    Provenance,
    SourceKind,
    Syndrome,
)

VALID_ULID = "01HZX5M8K4Q9W2N7R3T6Y8B0CF"
VALID_SHA256 = "a" * 64
VALID_CIRCUIT = "R 0\nM 0"
VALID_DATA_REF = "syndromes/run_001.bin"
VALID_DEM_PATH = "dems/surface_d3.dem"
VALID_SOURCE_PATH = "/var/decoderops/inputs/circuit_d3.stim"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _provenance(
    kind: SourceKind = "stim_circuit",
    source_path: str | None = VALID_SOURCE_PATH,
) -> Provenance:
    return Provenance(
        source_kind=kind,
        source_path=source_path,
        source_sha256=VALID_SHA256,
        ingested_at_utc=_utc_now(),
        ingester_version="0.1.0",
    )


def _dem_ref() -> DEMRef:
    return DEMRef(
        dem_path=VALID_DEM_PATH,
        num_detectors=24,
        num_observables=1,
        num_error_mechanisms=120,
        hyperedge_max_degree=4,
    )


def _syndrome() -> Syndrome:
    return Syndrome(
        shots=1024,
        detectors_per_shot=24,
        basis="Z",
        rounds=3,
        dtype="uint8",
        data_ref=VALID_DATA_REF,
    )


# ---------------------------------------------------------------------------
# SCHEMA_VERSION constant
# ---------------------------------------------------------------------------


def test_schema_version_constant_is_1() -> None:
    assert SCHEMA_VERSION == "1"
    # Constructing with a different literal must fail at the type-validation
    # boundary -- that is how readers detect schema drift. model_validate
    # accepts a dict so we can submit an invalid value at runtime without
    # tripping mypy's static Literal check.
    with pytest.raises(ValidationError):
        NormalisedInput.model_validate(
            {
                "schema_version": "2",
                "input_id": VALID_ULID,
                "provenance": _provenance(),
                "dem": None,
                "syndrome": None,
                "circuit_stim_source": VALID_CIRCUIT,
            }
        )


# ---------------------------------------------------------------------------
# Provenance boundary validation
# ---------------------------------------------------------------------------


def test_provenance_rejects_non_utc_datetime() -> None:
    naive = datetime(2026, 4, 21, 12, 0, 0)
    with pytest.raises(ValidationError) as exc_info:
        Provenance(
            source_kind="stim_circuit",
            source_path=VALID_SOURCE_PATH,
            source_sha256=VALID_SHA256,
            ingested_at_utc=naive,
            ingester_version="0.1.0",
        )
    assert "UTC" in str(exc_info.value)

    east_of_utc = datetime(2026, 4, 21, 12, 0, 0, tzinfo=timezone(timedelta(hours=5)))
    with pytest.raises(ValidationError):
        Provenance(
            source_kind="stim_circuit",
            source_path=VALID_SOURCE_PATH,
            source_sha256=VALID_SHA256,
            ingested_at_utc=east_of_utc,
            ingester_version="0.1.0",
        )


def test_provenance_rejects_invalid_sha256() -> None:
    # Wrong length
    with pytest.raises(ValidationError):
        Provenance(
            source_kind="stim_circuit",
            source_path=VALID_SOURCE_PATH,
            source_sha256="a" * 63,
            ingested_at_utc=_utc_now(),
            ingester_version="0.1.0",
        )
    # Uppercase rejected -- provenance hashes must be byte-comparable.
    with pytest.raises(ValidationError):
        Provenance(
            source_kind="stim_circuit",
            source_path=VALID_SOURCE_PATH,
            source_sha256="A" * 64,
            ingested_at_utc=_utc_now(),
            ingester_version="0.1.0",
        )
    # Non-hex characters
    with pytest.raises(ValidationError):
        Provenance(
            source_kind="stim_circuit",
            source_path=VALID_SOURCE_PATH,
            source_sha256="g" * 64,
            ingested_at_utc=_utc_now(),
            ingester_version="0.1.0",
        )


def test_provenance_accepts_explicit_source_path() -> None:
    # Exercises the source_path write site so it is not effectively dead.
    prov = _provenance(source_path=VALID_SOURCE_PATH)
    assert prov.source_path == VALID_SOURCE_PATH


def test_provenance_accepts_null_source_path_for_in_memory_inputs() -> None:
    prov = _provenance(source_path=None)
    assert prov.source_path is None


# ---------------------------------------------------------------------------
# input_id (ULID) validation
# ---------------------------------------------------------------------------


def test_input_id_rejects_wrong_length() -> None:
    too_short = VALID_ULID[:-1]
    too_long = VALID_ULID + "0"
    for bad in (too_short, too_long, "", "0"):
        with pytest.raises(ValidationError):
            NormalisedInput(
                schema_version="1",
                input_id=bad,
                provenance=_provenance(),
                dem=None,
                syndrome=None,
                circuit_stim_source=VALID_CIRCUIT,
            )


def test_input_id_rejects_invalid_alphabet() -> None:
    # Crockford base32 deliberately excludes I, L, O, U. Lowercase is also
    # rejected so input ids compare byte-for-byte across systems.
    for forbidden in ("I", "L", "O", "U"):
        bad = forbidden + VALID_ULID[1:]
        with pytest.raises(ValidationError):
            NormalisedInput(
                schema_version="1",
                input_id=bad,
                provenance=_provenance(),
                dem=None,
                syndrome=None,
                circuit_stim_source=VALID_CIRCUIT,
            )
    lowercase = VALID_ULID.lower()
    with pytest.raises(ValidationError):
        NormalisedInput(
            schema_version="1",
            input_id=lowercase,
            provenance=_provenance(),
            dem=None,
            syndrome=None,
            circuit_stim_source=VALID_CIRCUIT,
        )


# ---------------------------------------------------------------------------
# Cross-field model_validator -- the central invariant
# ---------------------------------------------------------------------------


def test_normalised_input_requires_at_least_one_payload() -> None:
    # customer_dem is one of the kinds with no specific required field,
    # so it lets us isolate the "all three None" rejection path.
    with pytest.raises(ValidationError) as exc_info:
        NormalisedInput(
            schema_version="1",
            input_id=VALID_ULID,
            provenance=_provenance(kind="customer_dem"),
            dem=None,
            syndrome=None,
            circuit_stim_source=None,
        )
    assert "at least one" in str(exc_info.value)


def test_source_kind_stim_circuit_requires_circuit_source() -> None:
    # Providing a DEM but declaring stim_circuit must be rejected -- the
    # downstream parsers index off source_kind, so a mismatched payload
    # would silently route to the wrong handler.
    with pytest.raises(ValidationError) as exc_info:
        NormalisedInput(
            schema_version="1",
            input_id=VALID_ULID,
            provenance=_provenance(kind="stim_circuit"),
            dem=_dem_ref(),
            syndrome=None,
            circuit_stim_source=None,
        )
    assert "stim_circuit" in str(exc_info.value)

    # Happy path -- source_kind matches payload.
    ok = NormalisedInput(
        schema_version="1",
        input_id=VALID_ULID,
        provenance=_provenance(kind="stim_circuit"),
        dem=None,
        syndrome=None,
        circuit_stim_source=VALID_CIRCUIT,
    )
    assert ok.circuit_stim_source == VALID_CIRCUIT


def test_source_kind_stim_dem_file_requires_dem_ref() -> None:
    with pytest.raises(ValidationError) as exc_info:
        NormalisedInput(
            schema_version="1",
            input_id=VALID_ULID,
            provenance=_provenance(kind="stim_dem_file"),
            dem=None,
            syndrome=_syndrome(),
            circuit_stim_source=None,
        )
    assert "stim_dem_file" in str(exc_info.value)

    dem = _dem_ref()
    ok = NormalisedInput(
        schema_version="1",
        input_id=VALID_ULID,
        provenance=_provenance(kind="stim_dem_file"),
        dem=dem,
        syndrome=None,
        circuit_stim_source=None,
    )
    assert ok.dem == dem


def test_source_kind_syndrome_array_requires_syndrome() -> None:
    with pytest.raises(ValidationError) as exc_info:
        NormalisedInput(
            schema_version="1",
            input_id=VALID_ULID,
            provenance=_provenance(kind="syndrome_array"),
            dem=_dem_ref(),
            syndrome=None,
            circuit_stim_source=None,
        )
    assert "syndrome_array" in str(exc_info.value)

    syn = _syndrome()
    ok = NormalisedInput(
        schema_version="1",
        input_id=VALID_ULID,
        provenance=_provenance(kind="syndrome_array"),
        dem=None,
        syndrome=syn,
        circuit_stim_source=None,
    )
    assert ok.syndrome == syn


# ---------------------------------------------------------------------------
# Frozen + extra='forbid'
# ---------------------------------------------------------------------------


def test_models_are_frozen_and_forbid_extra() -> None:
    prov = _provenance()
    # frozen=True: attribute assignment on a constructed model must fail
    # at runtime. The attribute name is held in a variable so this is a
    # genuine dynamic setattr (not an obfuscated `prov.x = y`, which
    # mypy would correctly reject under a typed model).
    frozen_field = "ingester_version"
    with pytest.raises(ValidationError):
        setattr(prov, frozen_field, "9.9.9")

    # extra='forbid': unknown fields must fail at construction. We use
    # model_validate(dict) instead of **kwargs because the latter forces
    # mypy to expand each key against the typed signature, which it
    # rightly rejects -- the runtime rejection by Pydantic is what this
    # test protects.
    with pytest.raises(ValidationError):
        Provenance.model_validate(
            {
                "source_kind": "stim_circuit",
                "source_path": VALID_SOURCE_PATH,
                "source_sha256": VALID_SHA256,
                "ingested_at_utc": _utc_now(),
                "ingester_version": "0.1.0",
                "unexpected_field": "boom",
            }
        )

    with pytest.raises(ValidationError):
        Syndrome.model_validate(
            {
                "shots": 1,
                "detectors_per_shot": 1,
                "basis": "X",
                "rounds": 1,
                "dtype": "bool",
                "data_ref": VALID_DATA_REF,
                "extra": "boom",
            }
        )

    with pytest.raises(ValidationError):
        DEMRef.model_validate(
            {
                "dem_path": VALID_DEM_PATH,
                "num_detectors": 1,
                "num_observables": 1,
                "num_error_mechanisms": 1,
                "hyperedge_max_degree": 1,
                "extra": "boom",
            }
        )

    n = NormalisedInput(
        schema_version="1",
        input_id=VALID_ULID,
        provenance=prov,
        dem=None,
        syndrome=None,
        circuit_stim_source=VALID_CIRCUIT,
    )
    notes_field = "notes"
    with pytest.raises(ValidationError):
        setattr(n, notes_field, "mutated")


# ---------------------------------------------------------------------------
# Numeric bounds on Syndrome and DEMRef
# ---------------------------------------------------------------------------


def test_syndrome_shots_and_rounds_must_be_positive() -> None:
    for shots in (0, -1):
        with pytest.raises(ValidationError):
            Syndrome(
                shots=shots,
                detectors_per_shot=1,
                basis="Z",
                rounds=1,
                dtype="bool",
                data_ref=VALID_DATA_REF,
            )
    for rounds in (0, -5):
        with pytest.raises(ValidationError):
            Syndrome(
                shots=1,
                detectors_per_shot=1,
                basis="Z",
                rounds=rounds,
                dtype="bool",
                data_ref=VALID_DATA_REF,
            )


def test_dem_ref_numeric_fields_must_be_positive() -> None:
    bad_kwargs_sets: list[dict[str, int]] = [
        {"num_detectors": 0},
        {"num_observables": 0},
        {"num_error_mechanisms": 0},
        {"hyperedge_max_degree": 0},
    ]
    for override in bad_kwargs_sets:
        kwargs: dict[str, int] = {
            "num_detectors": 1,
            "num_observables": 1,
            "num_error_mechanisms": 1,
            "hyperedge_max_degree": 1,
            **override,
        }
        with pytest.raises(ValidationError):
            DEMRef(dem_path=VALID_DEM_PATH, **kwargs)
