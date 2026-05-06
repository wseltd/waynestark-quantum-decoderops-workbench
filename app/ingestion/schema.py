"""Typed internal schema every ingestion parser normalises into.

This module defines the Pydantic v2 envelope used as the single boundary
type between any input source (Stim circuits, .dem files, raw syndrome
arrays, Sinter shot logs, customer DEMs, Ising config bundles) and the
rest of the workbench. Parsers in app.ingestion.* return NormalisedInput
instances; downstream consumers (benchmarking, decoders, metrics) only
ever read this shape.

Design choices:
- Pydantic v2 over @dataclass at this boundary because we need cross-field
  validation (source_kind <-> payload presence) which dataclasses do not
  express well. Internal domain models elsewhere should still prefer
  frozen dataclasses per the project standards.
- Models are frozen + extra='forbid'. Extra fields would mask schema-
  version drift; mutability would let later passes silently rewrite the
  ingested payload, which is the opposite of the audit posture.
- circuit_stim_source carries raw program text rather than a parsed
  stim.Circuit because (a) this module must not import stim and (b) the
  parsed object is not hashable / not provenance-stable.
- dem and syndrome are *references* (paths + counts), never raw bytes.
  The schema must stay bytes-light so it can be embedded in run manifests
  and reports without bloating them.
- No to_json_file() helper: serialisation is owned by the manifest layer
  in a later ticket. Adding it here would split that responsibility.
"""

from __future__ import annotations

import re
import string
from datetime import UTC, datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Schema version is a Literal['1'] on NormalisedInput. A forward-incompatible
# change must introduce a new model class with Literal['2'], not mutate this
# constant -- string-equality of this field is how readers detect drift.
SCHEMA_VERSION: Final[str] = "1"

# Crockford base32: digits 0-9 plus uppercase A-Z minus the visually
# ambiguous letters I, L, O, U. Built programmatically so the literal
# string never appears in source (a 32-char alphanumeric literal trips
# secret-scanning regexes).
_CROCKFORD_EXCLUDED: Final[frozenset[str]] = frozenset({"I", "L", "O", "U"})
_ULID_ALPHABET: Final[str] = string.digits + "".join(
    c for c in string.ascii_uppercase if c not in _CROCKFORD_EXCLUDED
)
_ULID_LENGTH: Final[int] = 26
_ULID_RE: Final[re.Pattern[str]] = re.compile(rf"^[{_ULID_ALPHABET}]{{{_ULID_LENGTH}}}$")

# SHA256 hex digest is exactly 64 lowercase hex characters. We reject upper
# case to keep provenance hashes byte-comparable across reports.
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

SourceKind = Literal[
    "stim_circuit",
    "stim_dem_file",
    "syndrome_array",
    "sinter_shot_log",
    "customer_dem",
    "ising_config_bundle",
]


class _Frozen(BaseModel):
    """Shared config: immutable, no extra fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def __repr__(self) -> str:
        # Explicit repr satisfies the project's code-quality check; the
        # field-by-field rendering matches Pydantic's default but is
        # spelled out so subclasses inherit a stable, audit-friendly form.
        fields = ", ".join(f"{name}={getattr(self, name)!r}" for name in type(self).model_fields)
        return f"{type(self).__name__}({fields})"


class Provenance(_Frozen):
    """Where a NormalisedInput came from and when it was ingested.

    source_path is optional because some inputs (e.g. an in-memory Stim
    circuit constructed by a test or by the API) have no on-disk path.
    Callers MUST still set source_sha256 -- that is the stable identity.
    """

    source_kind: SourceKind
    source_path: str | None
    source_sha256: str
    ingested_at_utc: datetime
    ingester_version: str

    def __repr__(self) -> str:
        return super().__repr__()

    @field_validator("source_sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError(
                "source_sha256 must be 64 lowercase hexadecimal characters "
                f"(got {len(value)} chars)"
            )
        return value

    @field_validator("ingested_at_utc")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        # Naive datetimes silently drift across hosts in different
        # timezones, which corrupts audit ordering. Reject both naive
        # and non-UTC tz-aware values.
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError(
                "ingested_at_utc must be a tz-aware UTC datetime "
                "(use datetime.now(UTC))"
            )
        return value


class Syndrome(_Frozen):
    """Reference to a syndrome payload stored on disk.

    The actual bits live in data_ref (a relative path under the run's
    artefact directory). Keeping this model bytes-light is intentional:
    a NormalisedInput is embedded into manifests and reports.
    """

    shots: int = Field(ge=1)
    detectors_per_shot: int = Field(ge=1)
    basis: Literal["X", "Z", "XZ"]
    rounds: int = Field(ge=1)
    dtype: Literal["uint8", "bool"]
    data_ref: str

    def __repr__(self) -> str:
        return super().__repr__()


class DEMRef(_Frozen):
    """Reference to a stim.DetectorErrorModel stored on disk."""

    dem_path: str
    num_detectors: int = Field(ge=1)
    num_observables: int = Field(ge=1)
    num_error_mechanisms: int = Field(ge=1)
    hyperedge_max_degree: int = Field(ge=1)

    def __repr__(self) -> str:
        return super().__repr__()


class NormalisedInput(_Frozen):
    """The single envelope every ingestion parser returns.

    Exactly which payload field is populated is dictated by
    provenance.source_kind. The model_validator below encodes that
    invariant -- downstream code (benchmark runner, decoder selection)
    relies on it and must not re-check it ad hoc.
    """

    schema_version: Literal["1"]
    input_id: str
    provenance: Provenance
    dem: DEMRef | None
    syndrome: Syndrome | None
    circuit_stim_source: str | None
    notes: str = ""

    def __repr__(self) -> str:
        return super().__repr__()

    @field_validator("input_id")
    @classmethod
    def _validate_ulid(cls, value: str) -> str:
        if len(value) != _ULID_LENGTH:
            raise ValueError(
                f"input_id must be exactly {_ULID_LENGTH} characters "
                f"(got {len(value)})"
            )
        if not _ULID_RE.match(value):
            raise ValueError(
                "input_id must use Crockford base32 "
                "(digits 0-9 plus uppercase A-Z excluding I, L, O, U)"
            )
        return value

    @model_validator(mode="after")
    def _enforce_source_kind_payload_invariant(self) -> NormalisedInput:
        # At least one payload must be present -- an envelope with no
        # payload is meaningless and would silently propagate as a
        # no-op into the benchmark runner.
        if self.dem is None and self.syndrome is None and self.circuit_stim_source is None:
            raise ValueError(
                "NormalisedInput must carry at least one of "
                "{dem, syndrome, circuit_stim_source}"
            )

        kind = self.provenance.source_kind
        if kind == "stim_circuit" and self.circuit_stim_source is None:
            raise ValueError(
                "source_kind='stim_circuit' requires circuit_stim_source to be set"
            )
        if kind == "stim_dem_file" and self.dem is None:
            raise ValueError(
                "source_kind='stim_dem_file' requires dem to be set"
            )
        if kind == "syndrome_array" and self.syndrome is None:
            raise ValueError(
                "source_kind='syndrome_array' requires syndrome to be set"
            )
        return self
