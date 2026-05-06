"""Real vendor generate_test_data.py smoke.

Drives ``vendor/Ising-Decoding/code/export/generate_test_data.py`` via
the production wrapper ``app.benchmarking.generate_test_data`` with
the smallest public parameter set the vendor supports. Asserts the
real .bin bundle is emitted, each file is non-empty, and every path
SHA256-matches the wrapper's own stamp (content-addressed).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pytest

from app.benchmarking.generate_test_data import generate_test_data

_VENDOR = Path(os.environ.get("DECODEROPS_VENDOR_DIR", "vendor/Ising-Decoding")).resolve()
_GENERATOR = _VENDOR / "code" / "export" / "generate_test_data.py"


pytestmark = pytest.mark.skipif(
    not _GENERATOR.exists(),
    reason=(
        "vendor/Ising-Decoding/code/export/generate_test_data.py not "
        "present; run scripts/fetch_ising_assets.sh or set "
        "DECODEROPS_VENDOR_DIR to a cloned vendor checkout"
    ),
)


def test_vendor_generate_test_data_emits_real_bin_bundle(tmp_path: Path) -> None:
    rec = generate_test_data(
        vendor_root=_VENDOR,
        output_dir=tmp_path,
        distance=3,
        n_rounds=3,
        num_samples=128,
        basis="X",
        p_error=0.003,
        simple_noise=True,
        timeout_seconds=300,
    )
    assert rec.returncode == 0

    emitted = {p.name for p in rec.produced_files}
    for required in (
        "detectors.bin",
        "observables.bin",
        "H_csr.bin",
        "O_csr.bin",
        "priors.bin",
    ):
        assert required in emitted, f"vendor did not emit {required!r}"
    for p in rec.produced_files:
        assert p.stat().st_size > 0, f"{p} is empty"

    for path, stamped in rec.sha256_by_path.items():
        on_disk = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert on_disk == stamped, f"SHA drift for {path}"

    assert "--distance" in rec.command
    assert "3" in rec.command
    assert "--basis" in rec.command and "X" in rec.command


def test_vendor_generate_test_data_timestamps_are_iso_utc_z(
    tmp_path: Path,
) -> None:
    rec = generate_test_data(
        vendor_root=_VENDOR,
        output_dir=tmp_path,
        distance=3,
        n_rounds=3,
        num_samples=64,
        basis="X",
        p_error=0.003,
        simple_noise=True,
        timeout_seconds=300,
    )
    iso = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    assert iso.match(rec.started_at_utc)
    assert iso.match(rec.finished_at_utc)
