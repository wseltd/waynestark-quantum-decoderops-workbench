"""Stim DEM file parser (T016)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import stim

from app.ingestion.schema import DEMRef, NormalisedInput, Provenance
from app.ingestion.stim_circuit import _count_error_mechanisms_and_max_degree

__all__ = ["DEMParseError", "parse_dem_file"]

_ULID_FIXED = "01HZX5M8K4Q9W2N7R3T6Y8B0CF"


class DEMParseError(ValueError):
    pass


def _stream_sha256(path: Path, chunk: int = 64 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_dem_file(
    path: Path | str,
    *,
    ingester_version: str,
    now_utc_fn: Callable[[], datetime] | None = None,
    ulid_fn: Callable[[], str] | None = None,
) -> NormalisedInput:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as e:
        raise DEMParseError(f"DEM file not found: {path}") from e
    try:
        dem = stim.DetectorErrorModel.from_file(str(resolved))
    except Exception as e:
        raise DEMParseError(f"malformed DEM at {resolved}: {e}") from e
    n_mech, max_deg = _count_error_mechanisms_and_max_degree(dem)
    if n_mech == 0:
        raise DEMParseError(f"DEM at {resolved} has zero error instructions")
    sha = _stream_sha256(resolved)
    now = now_utc_fn() if now_utc_fn else datetime.now(timezone.utc)
    ulid = ulid_fn() if ulid_fn else _ULID_FIXED
    return NormalisedInput(
        schema_version="1",
        input_id=ulid,
        provenance=Provenance(
            source_kind="stim_dem_file",
            source_path=str(resolved),
            source_sha256=sha,
            ingested_at_utc=now,
            ingester_version=ingester_version,
        ),
        dem=DEMRef(
            dem_path=str(resolved),
            num_detectors=int(dem.num_detectors),
            num_observables=max(1, int(dem.num_observables)),
            num_error_mechanisms=n_mech,
            hyperedge_max_degree=max(1, max_deg),
        ),
        syndrome=None,
        circuit_stim_source=None,
    )
