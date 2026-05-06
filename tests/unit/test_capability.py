"""Unit tests for the unified Tier 3 capability detector (T012).

The test matrix pins the contract downstream reports rely on:

    * no Tier 3 module is imported at module top-level
    * every probe returns a CapabilityReport with a precise reason
    * detect_all walks every known capability name and never raises

Probes are exercised with monkeypatched imports so failures for
missing modules are simulated without uninstalling anything from the
venv.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from app.core import capability
from app.core.capability import (
    CAPABILITY_NAMES,
    ProbeReport as CapabilityReport,
    detect_all,
    load_environment_report,
    probe_cudaq,
    probe_cudaq_qec,
    probe_cupy,
    probe_cuquantum,
    probe_modelopt,
    probe_onnxruntime_cuda,
    probe_onnxruntime_tensorrt,
    probe_tensorrt,
    probe_torch_cuda,
)


# --------------------------------------------------------------------------- #
# load_environment_report
# --------------------------------------------------------------------------- #


def test_load_environment_report_returns_empty_dict_when_file_missing(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist.json"
    assert load_environment_report(missing) == {}


def test_load_environment_report_parses_existing_json(tmp_path: Path) -> None:
    report = tmp_path / "env.json"
    report.write_text(json.dumps({"gpu": {"devices": [{"idx": 0}]}}))
    loaded = load_environment_report(report)
    assert loaded == {"gpu": {"devices": [{"idx": 0}]}}


# --------------------------------------------------------------------------- #
# torch_cuda
# --------------------------------------------------------------------------- #


def test_probe_torch_cuda_reports_unavailable_with_precise_reason_when_import_fails() -> None:
    # Simulate ImportError by removing torch from sys.modules and
    # installing a sentinel that raises on __import__.
    saved = sys.modules.pop("torch", None)
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        rep = probe_torch_cuda({})
    finally:
        if saved is not None:
            sys.modules["torch"] = saved
        else:
            sys.modules.pop("torch", None)
    assert rep.available is False
    assert rep.name == "torch_cuda"
    assert "torch" in (rep.reason or "").lower()
    assert rep.reason  # non-empty


def test_probe_torch_cuda_reports_unavailable_when_env_reports_zero_devices() -> None:
    # torch imports fine in this venv, but the env dict reports 0
    # devices AND we force torch.cuda.is_available to False. The reason
    # string must name the gate that failed.
    with mock.patch("torch.cuda.is_available", return_value=False):
        rep = probe_torch_cuda({})
    assert rep.available is False
    assert rep.reason  # non-empty
    assert "cuda" in rep.reason.lower()


# --------------------------------------------------------------------------- #
# onnxruntime probes
# --------------------------------------------------------------------------- #


def test_probe_onnxruntime_cuda_reports_unavailable_when_provider_not_registered() -> None:
    with mock.patch(
        "onnxruntime.get_available_providers",
        return_value=["CPUExecutionProvider"],
    ):
        rep = probe_onnxruntime_cuda({})
    assert rep.available is False
    assert "CUDAExecutionProvider" in (rep.reason or "")


def test_probe_onnxruntime_tensorrt_reports_unavailable_when_provider_not_registered() -> None:
    with mock.patch(
        "onnxruntime.get_available_providers",
        return_value=["CPUExecutionProvider"],
    ):
        rep = probe_onnxruntime_tensorrt({})
    assert rep.available is False
    assert "TensorrtExecutionProvider" in (rep.reason or "")


# --------------------------------------------------------------------------- #
# tier 3 import-fail probes (generic shape — use helper for brevity)
# --------------------------------------------------------------------------- #


def _force_import_error(modname: str):
    """Context-like helper that removes a module so import raises."""

    class _Restore:
        def __enter__(self):
            self._saved = sys.modules.pop(modname, None)
            # Drop any submodules too (e.g. torch.cuda) so dotted probes
            # also hit the failure path.
            for key in list(sys.modules):
                if key == modname or key.startswith(modname + "."):
                    sys.modules.pop(key, None)
            sys.modules[modname] = None  # type: ignore[assignment]
            return self

        def __exit__(self, *exc):
            sys.modules.pop(modname, None)
            if self._saved is not None:
                sys.modules[modname] = self._saved

    return _Restore()


def test_probe_tensorrt_reports_unavailable_with_precise_reason_when_import_fails() -> None:
    with _force_import_error("tensorrt"):
        rep = probe_tensorrt({})
    assert rep.available is False
    assert "tensorrt" in (rep.reason or "").lower()


def test_probe_cudaq_reports_unavailable_with_precise_reason_when_import_fails() -> None:
    with _force_import_error("cudaq"):
        rep = probe_cudaq({})
    assert rep.available is False
    assert "cudaq" in (rep.reason or "").lower()


def test_probe_cudaq_qec_reports_unavailable_with_precise_reason_when_import_fails() -> None:
    # Have to block BOTH cudaq_qec AND cudaq.qec for the fallback path.
    with _force_import_error("cudaq_qec"), _force_import_error("cudaq"):
        rep = probe_cudaq_qec({})
    assert rep.available is False
    assert "cudaq" in (rep.reason or "").lower()


def test_probe_cuquantum_reports_unavailable_with_precise_reason_when_import_fails() -> None:
    with _force_import_error("cuquantum"):
        rep = probe_cuquantum({})
    assert rep.available is False
    assert "cuquantum" in (rep.reason or "").lower()


def test_probe_cupy_reports_unavailable_with_precise_reason_when_import_fails() -> None:
    with _force_import_error("cupy"):
        rep = probe_cupy({})
    assert rep.available is False
    assert "cupy" in (rep.reason or "").lower()


def test_probe_modelopt_reports_unavailable_with_precise_reason_when_import_fails() -> None:
    with _force_import_error("modelopt"):
        rep = probe_modelopt({})
    assert rep.available is False
    assert "modelopt" in (rep.reason or "").lower()


# --------------------------------------------------------------------------- #
# detect_all
# --------------------------------------------------------------------------- #


def test_detect_all_returns_report_for_every_capability_name(tmp_path: Path) -> None:
    results = detect_all(env_report_path=tmp_path / "missing.json")
    assert set(results.keys()) == set(CAPABILITY_NAMES)
    assert list(results.keys()) == list(CAPABILITY_NAMES)


def test_detect_all_never_raises_when_no_tier3_modules_installed(
    tmp_path: Path,
) -> None:
    # Simulate a host with none of the Tier 3 modules installed by
    # blocking all of them at once. detect_all must still complete.
    ctxs = [
        _force_import_error(m)
        for m in (
            "tensorrt",
            "cudaq",
            "cudaq_qec",
            "cuquantum",
            "cupy",
            "modelopt",
        )
    ]
    for c in ctxs:
        c.__enter__()
    try:
        results = detect_all(env_report_path=tmp_path / "missing.json")
    finally:
        for c in ctxs:
            c.__exit__(None, None, None)
    assert set(results.keys()) == set(CAPABILITY_NAMES)


def test_capability_report_with_available_false_has_non_empty_reason() -> None:
    # Global contract: every unavailable probe must carry a non-empty
    # reason string. If this fails, the compat matrix will render
    # "capability unavailable" rows with no actionable signal.
    ctxs = [
        _force_import_error(m)
        for m in (
            "tensorrt",
            "cudaq",
            "cudaq_qec",
            "cuquantum",
            "cupy",
            "modelopt",
        )
    ]
    for c in ctxs:
        c.__enter__()
    try:
        results = detect_all(env_report_path=Path("/tmp/does-not-exist.json"))
    finally:
        for c in ctxs:
            c.__exit__(None, None, None)
    for name, rep in results.items():
        if not rep.available:
            assert rep.reason, f"{name} is unavailable with empty reason"
            assert rep.reason.strip() != "capability unavailable", (
                f"{name} has generic reason; must be precise"
            )


# --------------------------------------------------------------------------- #
# Invariant: no Tier 3 modules imported at module top level
# --------------------------------------------------------------------------- #


def test_no_tier3_modules_are_imported_at_module_top_level() -> None:
    # Parse the AST of app/core/capability.py and walk module-level
    # Import/ImportFrom nodes. Tier 3 modules must appear ONLY inside
    # function bodies.
    src = Path(capability.__file__).read_text()
    tree = ast.parse(src)
    forbidden = {
        "torch",
        "onnxruntime",
        "tensorrt",
        "cudaq",
        "cudaq_qec",
        "cuquantum",
        "cupy",
        "modelopt",
    }
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden, (
                    f"Tier 3 module {alias.name!r} imported at module top level; "
                    "move inside probe function."
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                top = node.module.split(".")[0]
                assert top not in forbidden, (
                    f"Tier 3 module {node.module!r} imported at module top level; "
                    "move inside probe function."
                )
