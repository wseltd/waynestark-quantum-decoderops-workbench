"""Shared load + SHA256 verify helpers for Ising-Decoding checkpoints.

Used by both :mod:`app.decoders.ising_fast` (RF=9) and
:mod:`app.decoders.ising_accurate` (RF=13). Extracting these helpers
keeps the two backends structurally identical at the asset-loading
boundary: a single place to audit the integrity contract, a single
torch.load call site with ``weights_only=True``, a single chunked
hasher loop.

Private module (leading underscore) — external callers should use the
concrete decoder classes, not these helpers directly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.decoders.ising_fast_errors import (
    IsingAssetIntegrityError,
    IsingModelFileMissingError,
)

if TYPE_CHECKING:
    pass


__all__ = [
    "CHUNK_BYTES",
    "load_ising_checkpoint",
    "verify_checkpoint_sha256",
]


# 1 MiB chunk size for stream hashing. Same constant used by
# :mod:`app.decoders.ising_fast`; kept in lockstep via import, not
# duplication, so bumping the size in one place updates both backends.
CHUNK_BYTES: int = 1 << 20


def verify_checkpoint_sha256(
    *, model_path: Path, expected_sha256: str, chunk_bytes: int = CHUNK_BYTES
) -> str:
    """Stream-hash ``model_path`` and compare against ``expected_sha256``.

    Args:
        model_path: Absolute or relative path to a .pt checkpoint.
        expected_sha256: Hex digest the caller looked up from the
            asset manifest. Lowercase hex, length 64; the helper does
            not re-validate the shape (the manifest loader already
            did).
        chunk_bytes: Override the default 1 MiB window. Exposed for
            tests that want to force multiple loop iterations on a
            small fixture.

    Returns:
        The verified hex digest (== ``expected_sha256``).

    Raises:
        IsingModelFileMissingError: If ``model_path`` does not exist.
        IsingAssetIntegrityError: If the streamed digest differs from
            ``expected_sha256``. Both digests are carried on the
            exception so callers can log them without truncation.
    """
    if not model_path.is_file():
        raise IsingModelFileMissingError(model_path=model_path)
    h = hashlib.sha256()
    with model_path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_sha256:
        raise IsingAssetIntegrityError(
            model_path=model_path,
            expected_sha256=expected_sha256,
            actual_sha256=actual,
        )
    return actual


def load_ising_checkpoint(
    *, model_path: Path, expected_sha256: str, device: str
) -> Any:
    """Verify SHA256 then ``torch.load`` the checkpoint.

    Loads with ``weights_only=True`` so no arbitrary-code-execution
    surface is exposed by the .pt file even if integrity check were
    bypassed. ``map_location=device`` places weights directly on the
    target accelerator without a spurious CPU roundtrip.

    Args:
        model_path: Path to the .pt checkpoint.
        expected_sha256: Hex digest recorded in the asset manifest.
        device: ``'cuda'`` or ``'cpu'`` (or a device-index string like
            ``'cuda:0'``). Caller resolves ``'auto'`` first.

    Returns:
        Whatever ``torch.load`` returns for this checkpoint — the
        vendor .pt is a state-dict; caller is responsible for wiring
        forward-pass semantics.

    Raises:
        IsingModelFileMissingError / IsingAssetIntegrityError via
        :func:`verify_checkpoint_sha256`.
    """
    verify_checkpoint_sha256(
        model_path=model_path, expected_sha256=expected_sha256
    )
    # torch import deferred so pure unit tests of the errors module do
    # not require torch to be importable.
    import torch

    return torch.load(
        str(model_path), map_location=device, weights_only=True
    )
