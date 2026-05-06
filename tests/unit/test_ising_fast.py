"""Unit tests for IsingFastDecoder (T024).

Covers every AC listed in plan.json T024:

    - capability probe: ready, missing torch, missing model, missing
      manifest, missing manifest entry, cuda-requested-but-unavailable
    - warmup: SHA256 integrity error with both digests in the message,
      idempotency, stream hash equivalence to hashlib on small files
    - metadata: backend_name, RF=9, model_path & sha256 populated after
      warmup
    - device selection: auto falls back correctly
    - stream hasher: chunked loop actually runs

Structured to avoid requiring the real 3.6 MB vendor checkpoint in
most cases — we build tiny fixture .pt files via torch.save, hash
them with hashlib as ground truth, and let the decoder verify.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from app.decoders.ising_fast import (
    IsingAssetIntegrityError,
    IsingFastDecoder,
    _SHA_CHUNK_BYTES,
    _stream_sha256,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _tiny_pt(path: Path) -> str:
    """Write a minimal torch-loadable state_dict and return its sha256."""
    torch.save({"dummy": torch.zeros(3)}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, *, model_name: str, sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "decoderops.ising_assets.v1",
                "status": "ok",
                "models": [
                    {"relpath": f"models/{model_name}", "sha256": sha256}
                ],
                "vendor": {"url": "https://github.com/NVIDIA/Ising-Decoding.git"},
            }
        )
    )


# --------------------------------------------------------------------------- #
# _stream_sha256
# --------------------------------------------------------------------------- #


def test_stream_sha256_matches_hashlib_on_small_file(tmp_path: Path) -> None:
    f = tmp_path / "small.bin"
    data = b"hello world" * 137
    f.write_bytes(data)
    assert _stream_sha256(f) == hashlib.sha256(data).hexdigest()


def test_stream_sha256_chunks_do_not_load_entire_file(tmp_path: Path) -> None:
    # Build a file larger than a single chunk, then force a tiny
    # chunk_bytes so the loop body must iterate several times. Equal
    # digests prove the chunked loop aggregates correctly.
    f = tmp_path / "biggish.bin"
    payload = b"\xab\xcd" * 8192  # 16 KiB
    f.write_bytes(payload)
    got = _stream_sha256(f, chunk_bytes=512)
    want = hashlib.sha256(payload).hexdigest()
    assert got == want
    # And confirm the default chunk size constant is declared at 1 MiB
    # — the contract unit tests reference.
    assert _SHA_CHUNK_BYTES == 1 << 20


# --------------------------------------------------------------------------- #
# available()
# --------------------------------------------------------------------------- #


def test_available_ready_when_model_and_manifest_present(tmp_path: Path) -> None:
    model = tmp_path / "Ising-Decoder-SurfaceCode-1-Fast.pt"
    sha = _tiny_pt(model)
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name=model.name, sha256=sha)
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    rep = d.available()
    assert rep.is_available is True
    assert rep.blocker_category == "none"
    assert "torch" in rep.detected_versions


def test_available_reports_software_when_torch_missing(tmp_path: Path) -> None:
    model = tmp_path / "m.pt"
    sha = _tiny_pt(model)
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name=model.name, sha256=sha)
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    # Monkey-patch __import__ at the decoder module scope. Simpler:
    # simulate the torch ImportError by shadowing sys.modules['torch'].
    import sys

    import app.decoders.ising_fast as target

    real_torch = sys.modules.get("torch")
    try:
        sys.modules["torch"] = None  # type: ignore[assignment]
        with mock.patch.object(target, "_stream_sha256"):
            rep = d.available()
    finally:
        if real_torch is not None:
            sys.modules["torch"] = real_torch
    # When the import hook sees None, `import torch` raises ImportError.
    assert rep.is_available is False
    assert rep.blocker_category == "software"
    assert "torch" in rep.reason


def test_available_reports_machine_when_cuda_requested_but_unavailable(
    tmp_path: Path,
) -> None:
    model = tmp_path / "m.pt"
    sha = _tiny_pt(model)
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name=model.name, sha256=sha)
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="cuda"
    )
    with mock.patch("torch.cuda.is_available", return_value=False):
        rep = d.available()
    assert rep.is_available is False
    assert rep.blocker_category == "machine"
    assert "cuda" in rep.reason.lower()


def test_available_reports_software_when_model_file_missing(
    tmp_path: Path,
) -> None:
    model = tmp_path / "does-not-exist.pt"
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name="does-not-exist.pt", sha256="a" * 64)
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    rep = d.available()
    assert rep.is_available is False
    assert rep.blocker_category == "software"
    assert str(model) in rep.reason


def test_available_reports_software_when_manifest_missing(
    tmp_path: Path,
) -> None:
    model = tmp_path / "m.pt"
    _tiny_pt(model)
    manifest = tmp_path / "missing.json"
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    rep = d.available()
    assert rep.is_available is False
    assert rep.blocker_category == "software"
    assert str(manifest) in rep.reason


def test_available_reports_software_when_manifest_entry_missing(
    tmp_path: Path,
) -> None:
    model = tmp_path / "Ising-Decoder-SurfaceCode-1-Fast.pt"
    _tiny_pt(model)
    manifest = tmp_path / "ising_assets.json"
    # Manifest exists but refers to a different filename
    _write_manifest(
        manifest, model_name="SomethingElse.pt", sha256="b" * 64
    )
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    rep = d.available()
    assert rep.is_available is False
    assert rep.blocker_category == "software"
    assert "no entry" in rep.reason.lower() or "missing" in rep.reason.lower()


# --------------------------------------------------------------------------- #
# warmup() — integrity
# --------------------------------------------------------------------------- #


def test_warmup_raises_integrity_error_on_sha256_mismatch(
    tmp_path: Path,
) -> None:
    model = tmp_path / "m.pt"
    _tiny_pt(model)
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name=model.name, sha256="0" * 64)
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    with pytest.raises(IsingAssetIntegrityError):
        d.warmup()


def test_warmup_integrity_error_includes_expected_and_actual_digests(
    tmp_path: Path,
) -> None:
    model = tmp_path / "m.pt"
    actual_sha = _tiny_pt(model)
    expected_bogus = "0" * 64
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name=model.name, sha256=expected_bogus)
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    with pytest.raises(IsingAssetIntegrityError) as excinfo:
        d.warmup()
    msg = str(excinfo.value)
    # Both full (non-truncated) digests must appear so operators can
    # diff byte-for-byte.
    assert expected_bogus in msg
    assert actual_sha in msg
    assert excinfo.value.expected_sha256 == expected_bogus
    assert excinfo.value.actual_sha256 == actual_sha


def test_warmup_is_idempotent(tmp_path: Path) -> None:
    model = tmp_path / "m.pt"
    sha = _tiny_pt(model)
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name=model.name, sha256=sha)
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    d.warmup()
    model_ref = d._model
    d.warmup()
    # Second warmup must NOT reload — object identity stable.
    assert d._model is model_ref


# --------------------------------------------------------------------------- #
# metadata()
# --------------------------------------------------------------------------- #


def test_metadata_receptive_field_is_9(tmp_path: Path) -> None:
    model = tmp_path / "m.pt"
    _tiny_pt(model)
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name=model.name, sha256="a" * 64)
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    assert d.metadata().receptive_field == 9
    assert IsingFastDecoder.RECEPTIVE_FIELD == 9


def test_metadata_backend_name_is_ising_fast(tmp_path: Path) -> None:
    model = tmp_path / "m.pt"
    _tiny_pt(model)
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name=model.name, sha256="a" * 64)
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    md = d.metadata()
    assert md.backend_name == "ising_fast"
    assert md.supports_gpu is True  # hardware capability, not current device
    assert md.supports_batching is True
    assert md.schema_version == "1"


def test_metadata_model_path_and_sha256_populated_after_warmup(
    tmp_path: Path,
) -> None:
    model = tmp_path / "m.pt"
    sha = _tiny_pt(model)
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name=model.name, sha256=sha)
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="cpu"
    )
    md_pre = d.metadata()
    # model_path resolves even pre-warmup because the file exists.
    assert md_pre.model_path is not None
    assert md_pre.model_sha256 is None  # not verified yet

    d.warmup()
    md_post = d.metadata()
    assert md_post.model_path is not None
    assert md_post.model_sha256 == sha


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #


def test_device_auto_selects_cuda_when_available(tmp_path: Path) -> None:
    model = tmp_path / "m.pt"
    sha = _tiny_pt(model)
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name=model.name, sha256=sha)
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="auto"
    )
    with mock.patch("torch.cuda.is_available", return_value=True):
        d.warmup()
    assert d._effective_device == "cuda"


def test_device_auto_falls_back_to_cpu_when_cuda_unavailable(
    tmp_path: Path,
) -> None:
    model = tmp_path / "m.pt"
    sha = _tiny_pt(model)
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name=model.name, sha256=sha)
    d = IsingFastDecoder(
        model_path=model, asset_manifest_path=manifest, device="auto"
    )
    with mock.patch("torch.cuda.is_available", return_value=False):
        d.warmup()
    assert d._effective_device == "cpu"


def test_constructor_rejects_unknown_device(tmp_path: Path) -> None:
    model = tmp_path / "m.pt"
    _tiny_pt(model)
    manifest = tmp_path / "ising_assets.json"
    _write_manifest(manifest, model_name=model.name, sha256="a" * 64)
    with pytest.raises(ValueError):
        IsingFastDecoder(
            model_path=model, asset_manifest_path=manifest, device="tpu"  # type: ignore[arg-type]
        )
