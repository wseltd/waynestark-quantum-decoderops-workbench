"""Stim circuit parser (T015)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Callable

import stim

from app.ingestion.schema import DEMRef, NormalisedInput, Provenance

__all__ = ["StimCircuitParseError", "parse_stim_circuit"]

_ULID_FIXED = "01HZX5M8K4Q9W2N7R3T6Y8B0CF"


class StimCircuitParseError(ValueError):
    pass


def _count_error_mechanisms_and_max_degree(
    model: stim.DetectorErrorModel,
) -> tuple[int, int]:
    n = 0
    max_deg = 0
    flat = model.flattened()
    for instr in flat:
        if getattr(instr, "type", None) == "error":
            n += 1
            deg = sum(
                1
                for t in instr.targets_copy()
                if t.is_relative_detector_id() or t.is_logical_observable_id()
            )
            if deg > max_deg:
                max_deg = deg
    return n, max_deg


def parse_stim_circuit(
    circuit_source: str,
    *,
    source_path: str | None = None,
    ingester_version: str,
    now_utc_fn: Callable[[], datetime] | None = None,
    ulid_fn: Callable[[], str] | None = None,
) -> NormalisedInput:
    if not isinstance(circuit_source, str):
        raise StimCircuitParseError(
            f"circuit_source must be str; got {type(circuit_source).__name__}"
        )
    try:
        circuit = stim.Circuit(circuit_source)
    except Exception as e:
        raise StimCircuitParseError(f"invalid Stim circuit: {e}") from e
    try:
        dem = circuit.detector_error_model(decompose_errors=True)
    except Exception as e:
        raise StimCircuitParseError(f"DEM derivation failed: {e}") from e
    n_mech, max_deg = _count_error_mechanisms_and_max_degree(dem)
    if n_mech == 0:
        raise StimCircuitParseError(
            "circuit produced zero error mechanisms; cannot normalise"
        )
    source_sha = hashlib.sha256(circuit_source.encode("utf-8")).hexdigest()
    now = now_utc_fn() if now_utc_fn else datetime.now(timezone.utc)
    ulid = ulid_fn() if ulid_fn else _ULID_FIXED
    return NormalisedInput(
        schema_version="1",
        input_id=ulid,
        provenance=Provenance(
            source_kind="stim_circuit",
            source_path=source_path,
            source_sha256=source_sha,
            ingested_at_utc=now,
            ingester_version=ingester_version,
        ),
        dem=DEMRef(
            dem_path="",
            num_detectors=int(dem.num_detectors),
            num_observables=max(1, int(dem.num_observables)),
            num_error_mechanisms=n_mech,
            hyperedge_max_degree=max(1, max_deg),
        ),
        syndrome=None,
        circuit_stim_source=circuit_source,
    )
