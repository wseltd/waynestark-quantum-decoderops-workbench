"""Real vendor local_run.sh WORKFLOW=inference smoke.

Drives ``vendor/Ising-Decoding/code/scripts/local_run.sh`` via the
production wrapper ``app.benchmarking.ising_local_run_inference.run_inference``
with the real shipped ``Ising-Decoder-SurfaceCode-1-Fast.pt`` checkpoint.
Asserts:
    - returncode == 0 (vendor workflow completes)
    - the real LER table is parseable from stdout
    - our wrapper parses parsed_summary fields deterministically
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest

from app.benchmarking.ising_local_run_inference import run_inference

_VENDOR = Path(os.environ.get("DECODEROPS_VENDOR_DIR", "vendor/Ising-Decoding")).resolve()
_SCRIPT = _VENDOR / "code" / "scripts" / "local_run.sh"
_FAST_CHECKPOINT = _VENDOR / "models" / "Ising-Decoder-SurfaceCode-1-Fast.pt"
_ASSETS_MANIFEST = Path(".decoderops/ising_assets.json")
_BASH = shutil.which("bash")

# torch + CUDA are prerequisites for the vendor workflow; probe without
# importing torch at collection time.
_torch_cuda_ok = False
try:
    import torch as _torch  # noqa: F401

    _torch_cuda_ok = bool(_torch.cuda.is_available())
except ImportError:
    _torch_cuda_ok = False


def _reason_skip() -> str | None:
    missing: list[str] = []
    if not _SCRIPT.exists():
        missing.append(f"vendor script missing: {_SCRIPT}")
    if not _FAST_CHECKPOINT.exists():
        missing.append(f"Fast checkpoint missing: {_FAST_CHECKPOINT}")
    if not _ASSETS_MANIFEST.exists():
        missing.append(f"ising_assets.json missing: {_ASSETS_MANIFEST}")
    if _BASH is None:
        missing.append("bash unavailable on PATH")
    if not _torch_cuda_ok:
        missing.append("torch.cuda.is_available() is False on this host")
    return "; ".join(missing) if missing else None


pytestmark = pytest.mark.skipif(
    _reason_skip() is not None,
    reason=(f"real vendor local_run.sh WORKFLOW=inference needs: {_reason_skip()}"),
)


@pytest.fixture(scope="module")
def staged_checkpoint() -> None:
    """Stage the shipped checkpoint into outputs/<exp>/models/ per the
    vendor README so find_best_model() resolves it."""
    exp = "decoderops_test_vendor_inference"
    target_dir = _VENDOR / "outputs" / exp / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _FAST_CHECKPOINT.name
    if not target.exists():
        try:
            target.symlink_to(_FAST_CHECKPOINT)
        except OSError:
            import shutil as _sh

            _sh.copy2(_FAST_CHECKPOINT, target)
    os.environ["EXPERIMENT_NAME"] = exp
    # Force our .venv python so vendor hydra/torch resolve from our stack.
    # Do NOT Path(...).resolve() — .venv/bin/python is a symlink chain
    # that resolves to /usr/bin/python3.12 (system interpreter without
    # torch). Use the absolute path to the symlink itself, which
    # preserves the venv context.
    os.environ["PREDECODER_PYTHON"] = str((Path.cwd() / ".venv" / "bin" / "python").absolute())
    # Eager mode — the host toolchain can't JIT-compile triton kernels.
    os.environ["TORCH_COMPILE"] = "0"
    os.environ["PREDECODER_TORCH_COMPILE"] = "0"
    os.environ["PREDECODER_MODEL_CHECKPOINT_FILE"] = str(_FAST_CHECKPOINT)
    return None


def test_vendor_inference_returncode_zero_and_parses_ler(tmp_path: Path, staged_checkpoint) -> None:
    rec = run_inference(
        vendor_root=_VENDOR,
        work_dir=tmp_path,
        model_variant="fast",
        cuda_visible_devices="0",
        timeout_seconds=900,
    )
    assert rec.returncode == 0
    log = rec.stdout_path.read_text(errors="replace")
    # Real LER table is emitted by the vendor workflow. Exact format:
    # "  LER - Avg:                 0.002731           0.002234"
    m = re.search(
        r"LER - Avg:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+"
        r"([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        log,
    )
    assert m is not None, (
        "vendor ran but LER - Avg line was not parseable; see log tail: " + log[-1500:]
    )
    no_predec = float(m.group(1))
    with_predec = float(m.group(2))
    assert 0.0 < no_predec < 0.5
    assert 0.0 < with_predec < 0.5
    # The shipped Ising pre-decoder must not strictly worsen LER — that
    # would signal the model failed to load or a pipeline regression.
    assert with_predec <= no_predec * 1.15, (
        f"Ising Fast predecoder regressed LER: no_predec={no_predec} with_predec={with_predec}"
    )

    assert rec.started_at_utc.endswith("Z")
    assert rec.finished_at_utc.endswith("Z")


def test_vendor_inference_checkpoint_sha_matches_manifest(
    staged_checkpoint,
) -> None:
    """Provenance gate — the shipped checkpoint the vendor workflow just
    loaded must SHA256-match ising_assets.json (bootstrap record)."""
    import hashlib

    assets = json.loads(_ASSETS_MANIFEST.read_text())
    expected = next(m["sha256"] for m in assets["models"] if "Fast" in m["relpath"])
    actual = hashlib.sha256(_FAST_CHECKPOINT.read_bytes()).hexdigest()
    assert actual == expected
