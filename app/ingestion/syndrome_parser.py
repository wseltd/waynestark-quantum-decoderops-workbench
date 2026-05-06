"""Syndrome array parser (T017)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.ingestion.schema import NormalisedInput, Provenance, Syndrome

__all__ = [
    "InvalidSyndromeInputError",
    "SyndromeArrayParser",
    "parse_syndrome_file",
]

_ULID_FIXED = "01HZX5M8K4Q9W2N7R3T6Y8B0CF"


class InvalidSyndromeInputError(ValueError):
    pass


def _stream_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class SyndromeArrayParser:
    def parse(
        self,
        path: Path,
        shape_hint: tuple[int, int] | None = None,
        dtype: str = "uint8",
    ) -> tuple[np.ndarray, str]:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            try:
                arr = np.load(path, allow_pickle=False)
            except Exception as e:
                raise InvalidSyndromeInputError(
                    f"failed to load .npy {path}: {e}"
                ) from e
            source = "syndrome_npy"
        elif suffix == ".bin":
            if shape_hint is None:
                raise InvalidSyndromeInputError(
                    ".bin syndrome input requires shape_hint"
                )
            raw = path.read_bytes()
            expected = int(np.prod(shape_hint))
            if len(raw) != expected:
                raise InvalidSyndromeInputError(
                    f"nbytes ({len(raw)}) != product(shape_hint)={expected}"
                ) from None
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(shape_hint)
            source = "syndrome_bin"
        else:
            raise InvalidSyndromeInputError(
                f"unsupported syndrome suffix: {suffix!r}"
            )
        if arr.dtype.kind not in ("b", "u") or arr.dtype.itemsize > 2:
            raise InvalidSyndromeInputError(
                f"dtype {arr.dtype} is not bool/uint; refusing"
            )
        arr = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        return arr, source


def parse_syndrome_file(
    path: Path,
    shape_hint: tuple[int, int] | None = None,
    dtype: str = "uint8",
    source_label: str | None = None,
    *,
    ingester_version: str = "0.1.0",
    now_utc_fn: Callable[[], datetime] | None = None,
    ulid_fn: Callable[[], str] | None = None,
) -> Any:
    path = Path(path).expanduser().resolve()
    parser = SyndromeArrayParser()
    arr, source_kind = parser.parse(path, shape_hint=shape_hint, dtype=dtype)
    sha = _stream_sha256(path)
    now = now_utc_fn() if now_utc_fn else datetime.now(timezone.utc)
    ulid = ulid_fn() if ulid_fn else _ULID_FIXED
    rounds = max(1, arr.shape[0])
    # Minimal Syndrome — detectors_per_shot from shape[1]
    syndrome = Syndrome(
        shots=arr.shape[0],
        detectors_per_shot=arr.shape[1],
        basis="Z",
        rounds=1,
        dtype="uint8",
        data_ref=str(path),
    )
    normalised = NormalisedInput(
        schema_version="1",
        input_id=ulid,
        provenance=Provenance(
            source_kind="syndrome_array",
            source_path=str(path),
            source_sha256=sha,
            ingested_at_utc=now,
            ingester_version=ingester_version,
        ),
        dem=None,
        syndrome=syndrome,
        circuit_stim_source=None,
    )
    # Attach the array as an extra attribute for callers who want it —
    # stays out of the schema so serialisation is unaffected.
    object.__setattr__(normalised, "_syndrome_array", arr)
    # Provide convenience accessor expected by tests.
    return _ParsedResult(normalised=normalised, syndromes=arr, provenance={
        "sha256": sha,
        "absolute_path": str(path),
        "size": path.stat().st_size,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
    }, input_source=source_kind)


class _ParsedResult:
    def __init__(self, normalised, syndromes, provenance, input_source) -> None:
        self.normalised = normalised
        self.syndromes = syndromes
        self.provenance = provenance
        self.input_source = input_source
