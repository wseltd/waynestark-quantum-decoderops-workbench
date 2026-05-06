"""Tests for app.benchmarking.generate_test_data (T040)."""

from __future__ import annotations

import hashlib
import re
import stat
import sys
from pathlib import Path

import pytest

from app.benchmarking.generate_test_data import (
    GenerateTestDataError,
    GenerateTestDataResult,
    generate_test_data,
)


def _make_fake_vendor(
    tmp_path: Path,
    *,
    exit_code: int = 0,
    bin_content: bytes = b"BIN_BYTES_0",
    onnx_content: bytes = b"ONNX_BYTES_0",
    produce_files: bool = True,
) -> Path:
    vendor = tmp_path / "vendor"
    (vendor / "code" / "export").mkdir(parents=True)
    generator = vendor / "code" / "export" / "generate_test_data.py"
    body = [
        "import sys, json, os",
        "from pathlib import Path",
        "print('args=' + json.dumps(sys.argv[1:]))",
        "print('cwd=' + os.getcwd())",
    ]
    if produce_files:
        body.append(
            f"Path('out.bin').write_bytes({bin_content!r})"
        )
        body.append(
            f"Path('out.onnx').write_bytes({onnx_content!r})"
        )
    body.append(f"sys.exit({exit_code})")
    generator.write_text("\n".join(body) + "\n")
    return vendor


def test_generate_test_data_rejects_even_distance(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    with pytest.raises(ValueError):
        generate_test_data(
            vendor,
            tmp_path / "out",
            distance=4,
            n_rounds=1,
            num_samples=1,
            basis="X",
            p_error=0.01,
        )


def test_generate_test_data_rejects_distance_below_three(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    with pytest.raises(ValueError):
        generate_test_data(
            vendor,
            tmp_path / "out",
            distance=1,
            n_rounds=1,
            num_samples=1,
            basis="X",
            p_error=0.01,
        )


def test_generate_test_data_rejects_non_positive_n_rounds(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    with pytest.raises(ValueError):
        generate_test_data(
            vendor,
            tmp_path / "out",
            distance=3,
            n_rounds=0,
            num_samples=1,
            basis="X",
            p_error=0.01,
        )


def test_generate_test_data_rejects_non_positive_num_samples(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path)
    with pytest.raises(ValueError):
        generate_test_data(
            vendor,
            tmp_path / "out",
            distance=3,
            n_rounds=1,
            num_samples=0,
            basis="X",
            p_error=0.01,
        )


def test_generate_test_data_rejects_p_error_out_of_range(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    with pytest.raises(ValueError):
        generate_test_data(
            vendor,
            tmp_path / "out",
            distance=3,
            n_rounds=1,
            num_samples=1,
            basis="X",
            p_error=0.0,
        )
    with pytest.raises(ValueError):
        generate_test_data(
            vendor,
            tmp_path / "out",
            distance=3,
            n_rounds=1,
            num_samples=1,
            basis="X",
            p_error=0.7,
        )


def test_generate_test_data_rejects_invalid_basis(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    with pytest.raises(ValueError):
        generate_test_data(
            vendor,
            tmp_path / "out",
            distance=3,
            n_rounds=1,
            num_samples=1,
            basis="Y",  # type: ignore[arg-type]
            p_error=0.01,
        )


def test_generate_test_data_builds_command_with_expected_flags(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path)
    rec = generate_test_data(
        vendor,
        tmp_path / "out",
        distance=3,
        n_rounds=2,
        num_samples=5,
        basis="X",
        p_error=0.01,
    )
    cmd = rec.command
    assert "--distance" in cmd and "3" in cmd
    assert "--n-rounds" in cmd and "2" in cmd
    assert "--num-samples" in cmd and "5" in cmd
    assert "--basis" in cmd and "X" in cmd
    assert "--p-error" in cmd


def test_generate_test_data_appends_simple_noise_flag_only_when_true(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path)
    rec = generate_test_data(
        vendor,
        tmp_path / "out",
        distance=3,
        n_rounds=1,
        num_samples=1,
        basis="X",
        p_error=0.01,
        simple_noise=True,
    )
    assert "--simple-noise" in rec.command


def test_generate_test_data_omits_simple_noise_flag_when_false(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path)
    rec = generate_test_data(
        vendor,
        tmp_path / "out",
        distance=3,
        n_rounds=1,
        num_samples=1,
        basis="X",
        p_error=0.01,
        simple_noise=False,
    )
    assert "--simple-noise" not in rec.command


def test_generate_test_data_sets_cwd_to_vendor_code_dir(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    rec = generate_test_data(
        vendor,
        tmp_path / "out",
        distance=3,
        n_rounds=1,
        num_samples=1,
        basis="X",
        p_error=0.01,
    )
    stdout = rec.stdout_path.read_text()
    assert "cwd=" in stdout
    assert (vendor / "code").as_posix() in stdout


def test_generate_test_data_stamps_sha256_for_produced_bin_and_onnx_files(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(
        tmp_path, bin_content=b"BINARY", onnx_content=b"ONNXMODEL"
    )
    rec = generate_test_data(
        vendor,
        tmp_path / "out",
        distance=3,
        n_rounds=1,
        num_samples=1,
        basis="X",
        p_error=0.01,
    )
    shas = set(rec.sha256_by_path.values())
    assert hashlib.sha256(b"BINARY").hexdigest() in shas
    assert hashlib.sha256(b"ONNXMODEL").hexdigest() in shas


def test_generate_test_data_raises_generate_test_data_error_on_nonzero_exit(
    tmp_path: Path,
) -> None:
    vendor = _make_fake_vendor(tmp_path, exit_code=2, produce_files=False)
    with pytest.raises(GenerateTestDataError):
        generate_test_data(
            vendor,
            tmp_path / "out",
            distance=3,
            n_rounds=1,
            num_samples=1,
            basis="X",
            p_error=0.01,
        )


def test_generate_test_data_records_parameters_in_result(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    rec = generate_test_data(
        vendor,
        tmp_path / "out",
        distance=5,
        n_rounds=3,
        num_samples=7,
        basis="Z",
        p_error=0.015,
        simple_noise=True,
    )
    assert rec.parameters == {
        "distance": 5,
        "n_rounds": 3,
        "num_samples": 7,
        "basis": "Z",
        "p_error": 0.015,
        "simple_noise": True,
    }


def test_generate_test_data_records_iso_utc_timestamps(tmp_path: Path) -> None:
    vendor = _make_fake_vendor(tmp_path)
    rec = generate_test_data(
        vendor,
        tmp_path / "out",
        distance=3,
        n_rounds=1,
        num_samples=1,
        basis="X",
        p_error=0.01,
    )
    iso = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    assert iso.match(rec.started_at_utc)
    assert iso.match(rec.finished_at_utc)
    assert isinstance(rec, GenerateTestDataResult)
