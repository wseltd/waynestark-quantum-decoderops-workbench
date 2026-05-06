"""Real Ising-Fast checkpoint smoke (T183)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

_CHECKPOINT = Path(
    os.environ.get(
        "DECODEROPS_ISING_FAST_CHECKPOINT",
        "vendor/Ising-Decoding/models/Ising-Decoder-SurfaceCode-1-Fast.pt",
    )
)
_ASSETS = Path(".decoderops/ising_assets.json")


def _skip_if_missing() -> None:
    if not _CHECKPOINT.exists():
        pytest.skip(
            f"Ising Fast checkpoint missing at {_CHECKPOINT}; "
            "run scripts/fetch_ising_assets.sh or set DECODEROPS_ISING_FAST_CHECKPOINT"
        )
    if not _ASSETS.exists():
        pytest.skip(f"ising_assets.json missing at {_ASSETS}")


def test_checkpoint_sha256_matches_sidecar() -> None:
    _skip_if_missing()
    assets = json.loads(_ASSETS.read_text())
    expected = None
    for entry in assets.get("models", []):
        if "Fast" in entry.get("relpath", ""):
            expected = entry.get("sha256")
            break
    if not expected:
        pytest.skip("no Fast checkpoint SHA256 in ising_assets.json")
    actual = hashlib.sha256(_CHECKPOINT.read_bytes()).hexdigest()
    assert actual == expected


def test_ising_fast_decoder_module_imports() -> None:
    _skip_if_missing()
    try:
        from app.decoders import ising_fast
    except ImportError as e:
        pytest.skip(f"ising_fast module missing: {e}")
    assert hasattr(ising_fast, "IsingFastDecoder")
