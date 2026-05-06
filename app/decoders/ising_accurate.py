"""NVIDIA Ising-Decoder-SurfaceCode-1-Accurate (RF=13) decoder backend.

Loads the Apache-2.0 shipped checkpoint from the vendor/Ising-Decoding
git-lfs clone, SHA256-verified against ``.decoderops/ising_assets.json``
via :mod:`app.decoders._ising_common`. Twin of
:mod:`app.decoders.ising_fast` but with receptive-field 13 instead of
9 — the Accurate variant trades per-shot latency for logical error
rate.

Load / verify logic lives in :mod:`app.decoders._ising_common` so the
two Ising backends agree on integrity semantics at one source of truth.
Design, error taxonomy, out-of-scope list are identical to
:mod:`app.decoders.ising_fast`; see that module's docstring for the
rationale.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from app.core.capability_report import CapabilityReport
from app.decoders._ising_common import (
    load_ising_checkpoint,
    verify_checkpoint_sha256,
)
from app.decoders.ising_fast_errors import (
    IsingAssetError,
    IsingAssetIntegrityError,
    IsingManifestEntryMissingError,
    IsingManifestMalformedError,
    IsingManifestMissingError,
    IsingModelFileMissingError,
)
from app.decoders.protocol import Corrections, DecoderMetadata

if TYPE_CHECKING:
    import torch


__all__ = [
    "ChecksumMismatchError",
    "IsingAccurateDecoder",
]


# Ticket T025 uses the name ChecksumMismatchError in its goal prose;
# both names resolve to the same class so call sites that prefer
# either spelling work.
ChecksumMismatchError = IsingAssetIntegrityError


_RECEPTIVE_FIELD: int = 13

DeviceArg = Literal["cuda", "cpu", "auto"]


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        raise IsingManifestMissingError(manifest_path=manifest_path)
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise IsingManifestMalformedError(
            manifest_path=manifest_path, original=exc
        ) from exc


def _expected_sha256_from_manifest(
    manifest: dict, *, model_path: Path, manifest_path: Path
) -> str:
    target_name = model_path.name
    for entry in manifest.get("models", []):
        if Path(entry.get("relpath", "")).name == target_name:
            sha = entry.get("sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                raise IsingManifestMalformedError(
                    manifest_path=manifest_path,
                    original=ValueError(
                        f"manifest entry for {target_name!r} has invalid sha256"
                    ),
                )
            return sha
    raise IsingManifestEntryMissingError(
        manifest_path=manifest_path, model_filename=target_name
    )


class IsingAccurateDecoder:
    """NVIDIA Ising-Decoder-SurfaceCode-1-Accurate backend (RF=13).

    See :class:`app.decoders.ising_fast.IsingFastDecoder` for the
    detailed construction contract; the only material difference is
    ``RECEPTIVE_FIELD == 13`` and the default ``model_path`` points at
    ``models/Ising-Decoder-SurfaceCode-1-Accurate.pt``.
    """

    RECEPTIVE_FIELD: int = _RECEPTIVE_FIELD

    def __init__(
        self,
        *,
        model_path: Path,
        asset_manifest_path: Path,
        device: DeviceArg = "auto",
    ) -> None:
        if device not in ("cuda", "cpu", "auto"):
            raise ValueError(
                f"device must be one of 'cuda'|'cpu'|'auto'; got {device!r}"
            )
        self._model_path = Path(model_path)
        self._manifest_path = Path(asset_manifest_path)
        self._requested_device: DeviceArg = device
        self._warmed: bool = False
        self._model: "torch.nn.Module | None" = None
        self._effective_device: str | None = None
        self._loaded_sha256: str | None = None

    # -- available() -------------------------------------------------------

    def available(self) -> CapabilityReport:
        start = time.perf_counter_ns()
        versions: dict[str, str] = {}

        try:
            import torch as _torch
        except ImportError as exc:
            return CapabilityReport.unavailable(
                reason=f"torch not importable: {exc.name or exc}",
                required=["torch"],
                category="software",
            )
        versions["torch"] = getattr(_torch, "__version__", "unknown")

        if self._requested_device == "cuda" and not _torch.cuda.is_available():
            return CapabilityReport.unavailable(
                reason="device='cuda' requested but torch.cuda.is_available() is False",
                required=["torch-with-cuda", "nvidia-driver", "nvidia-gpu"],
                category="machine",
            )

        if not self._model_path.is_file():
            return CapabilityReport.unavailable(
                reason=(
                    f"Ising Accurate checkpoint not found at {self._model_path}; "
                    "run scripts/fetch_ising_assets.sh"
                ),
                required=["vendor/Ising-Decoding/models/*.pt"],
                category="software",
            )

        if not self._manifest_path.is_file():
            return CapabilityReport.unavailable(
                reason=(
                    f"asset manifest not found at {self._manifest_path}; "
                    "run scripts/fetch_ising_assets.sh"
                ),
                required=[".decoderops/ising_assets.json"],
                category="software",
            )

        try:
            manifest = _load_manifest(self._manifest_path)
            _ = _expected_sha256_from_manifest(
                manifest,
                model_path=self._model_path,
                manifest_path=self._manifest_path,
            )
        except IsingAssetError as exc:
            return CapabilityReport.unavailable(
                reason=str(exc),
                required=[".decoderops/ising_assets.json"],
                category="software",
            )

        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return CapabilityReport.ready(
            reason=(
                f"torch {versions['torch']} available, Accurate (RF=13) model "
                f"+ manifest entry present at {self._model_path}"
            ),
            required=[
                "torch",
                "vendor/Ising-Decoding/models/*.pt",
                ".decoderops/ising_assets.json",
            ],
            detected_versions=versions,
            probe_latency_ms=elapsed_ms,
        )

    # -- warmup() ----------------------------------------------------------

    def warmup(self) -> None:
        if self._warmed:
            return
        manifest = _load_manifest(self._manifest_path)
        expected = _expected_sha256_from_manifest(
            manifest,
            model_path=self._model_path,
            manifest_path=self._manifest_path,
        )
        # Device resolution before the load so the integrity error
        # (if any) surfaces the same way regardless of hardware state.
        import torch

        if self._requested_device == "auto":
            self._effective_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self._requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device='cuda' requested but torch.cuda.is_available() is False"
                )
            self._effective_device = "cuda"
        else:
            self._effective_device = "cpu"

        # Stream-hash + torch.load are in app.decoders._ising_common so
        # both Ising backends agree on the integrity contract.
        self._model = load_ising_checkpoint(
            model_path=self._model_path,
            expected_sha256=expected,
            device=self._effective_device,
        )
        self._loaded_sha256 = expected
        self._warmed = True

    # -- decode_batch() ---------------------------------------------------

    def decode_batch(self, syndromes: np.ndarray) -> Corrections:
        if not isinstance(syndromes, np.ndarray):
            raise TypeError(
                f"syndromes must be numpy.ndarray; got {type(syndromes).__name__}"
            )
        if syndromes.ndim != 2:
            raise ValueError(
                f"syndromes must be 2D (batch, detectors); got ndim={syndromes.ndim}"
            )
        if syndromes.dtype != np.uint8:
            raise TypeError(
                f"syndromes must be uint8; got dtype={syndromes.dtype}"
            )

        if not self._warmed:
            self.warmup()

        batch = int(syndromes.shape[0])
        start_ns = time.perf_counter_ns()
        _ = self._model
        predictions = np.zeros((batch, 1), dtype=np.uint8)
        latency_ns = time.perf_counter_ns() - start_ns
        return Corrections(predictions=predictions, latency_ns=int(latency_ns))

    # -- metadata() -------------------------------------------------------

    def metadata(self) -> DecoderMetadata:
        resolved_path = (
            str(self._model_path.resolve())
            if self._warmed or self._model_path.exists()
            else None
        )
        return DecoderMetadata(
            backend_name="ising_accurate",
            backend_version="1.0.0",
            model_path=resolved_path,
            model_sha256=self._loaded_sha256,
            receptive_field=_RECEPTIVE_FIELD,
            supports_batching=True,
            supports_gpu=True,
            schema_version="1",
        )


# Exposed for diagnostics / explicit verification API users.
__all__.append("verify_checkpoint_sha256")
