"""Unit tests for IsingAccurateDecoder (T025).

AC-covering tests: available-reports-missing-checkpoint,
receptive-field-is-13, checksum-mismatch-raises, metadata-contains-
sha256-and-path. Plus a structural conformance check against the
Decoder Protocol to ensure the backend plugs into the benchmark
runner without special-casing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from app.decoders.ising_accurate import (
    ChecksumMismatchError,
    IsingAccurateDecoder,
)
from app.decoders.ising_fast_errors import IsingAssetIntegrityError
from app.decoders.protocol import Decoder


def _tiny_pt(path: Path) -> str:
    torch.save({"dummy": torch.zeros(2)}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, *, model_name: str, sha256: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "decoderops.ising_assets.v1",
                "status": "ok",
                "models": [
                    {"relpath": f"models/{model_name}", "sha256": sha256}
                ],
            }
        )
    )


def test_ising_accurate_available_reports_missing_checkpoint(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist.pt"
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, model_name=missing.name, sha256="a" * 64)
    d = IsingAccurateDecoder(
        model_path=missing, asset_manifest_path=manifest, device="cpu"
    )
    rep = d.available()
    assert rep.is_available is False
    assert rep.blocker_category == "software"
    assert str(missing) in rep.reason


def test_ising_accurate_receptive_field_is_13(tmp_path: Path) -> None:
    model = tmp_path / "Ising-Decoder-SurfaceCode-1-Accurate.pt"
    _tiny_pt(model)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, model_name=model.name, sha256="a" * 64)
    d = IsingAccurateDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    assert IsingAccurateDecoder.RECEPTIVE_FIELD == 13
    assert d.metadata().receptive_field == 13


def test_ising_accurate_checksum_mismatch_raises(tmp_path: Path) -> None:
    model = tmp_path / "m.pt"
    _tiny_pt(model)
    manifest = tmp_path / "manifest.json"
    # Record an obviously-wrong digest so warmup's integrity check must fail.
    _write_manifest(manifest, model_name=model.name, sha256="f" * 64)
    d = IsingAccurateDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    # ChecksumMismatchError is the ticket's named spelling; it aliases
    # IsingAssetIntegrityError so both except clauses work.
    with pytest.raises(IsingAssetIntegrityError):
        d.warmup()
    assert ChecksumMismatchError is IsingAssetIntegrityError


def test_ising_accurate_metadata_contains_sha256_and_path(
    tmp_path: Path,
) -> None:
    model = tmp_path / "Ising-Decoder-SurfaceCode-1-Accurate.pt"
    sha = _tiny_pt(model)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, model_name=model.name, sha256=sha)
    d = IsingAccurateDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    d.warmup()
    md = d.metadata()
    assert md.backend_name == "ising_accurate"
    assert md.model_sha256 == sha
    assert md.model_path is not None
    assert Path(md.model_path).name == model.name
    assert md.supports_batching is True
    assert md.supports_gpu is True
    assert md.schema_version == "1"


def test_ising_accurate_conforms_to_decoder_protocol(tmp_path: Path) -> None:
    model = tmp_path / "m.pt"
    _tiny_pt(model)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, model_name=model.name, sha256="a" * 64)
    d = IsingAccurateDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    assert isinstance(d, Decoder)
