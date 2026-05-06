"""Acceptance tests for pyproject.toml.

These tests lock in the invariants that ticket T001 is required to satisfy.
They run against the real pyproject.toml at repo root (no mocking): the
packaging configuration is the unit under test.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _load() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def test_pyproject_file_exists() -> None:
    assert PYPROJECT_PATH.is_file(), "pyproject.toml missing at repo root"


def test_project_name_is_decoderops() -> None:
    data: dict[str, Any] = _load()
    assert data["project"]["name"] == "decoderops"


def test_project_requires_python_312_only() -> None:
    data: dict[str, Any] = _load()
    spec = data["project"]["requires-python"]
    assert spec.startswith(">=3.12")
    assert "<3.13" in spec, "Python 3.13 must not be accepted (Tier 3 stack is 3.12)"


def test_build_backend_is_setuptools() -> None:
    data: dict[str, Any] = _load()
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    # The legacy backend path does not exist; catch regressions that would
    # silently break `pip install` on a clean checkout.
    assert "_legacy" not in data["build-system"]["build-backend"]


def test_console_script_targets_cli_main() -> None:
    data: dict[str, Any] = _load()
    # Target is the `main()` function (not the `app` Typer instance) so
    # setuptools + static-analysis tools resolve it as a function.
    assert data["project"]["scripts"]["decoderops"] == "app.cli.main:main"


def test_optional_dev_dependencies_present() -> None:
    data: dict[str, Any] = _load()
    dev = data["project"]["optional-dependencies"]["dev"]
    assert "pytest" in dev
    assert "ruff" in dev
    assert "mypy" in dev


def test_ruff_configuration() -> None:
    data: dict[str, Any] = _load()
    ruff = data["tool"]["ruff"]
    assert ruff["line-length"] == 100
    assert ruff["target-version"] == "py312"
    selected = ruff["lint"]["select"]
    for code in ("E", "F", "I", "B", "UP", "SIM"):
        assert code in selected, f"ruff rule group {code} must be enabled"


def test_mypy_strict_mode() -> None:
    data: dict[str, Any] = _load()
    mypy = data["tool"]["mypy"]
    assert mypy["strict"] is True
    assert mypy["python_version"] == "3.12"
    assert any("vendor" in pattern for pattern in mypy["exclude"])


def test_pytest_markers_registered() -> None:
    data: dict[str, Any] = _load()
    ini = data["tool"]["pytest"]["ini_options"]
    assert ini["testpaths"] == ["tests"]
    marker_names = {marker.split(":", 1)[0].strip() for marker in ini["markers"]}
    for required in ("integration", "real_artefacts", "runtime_capability", "gpu"):
        assert required in marker_names, f"pytest marker {required} must be registered"


def test_packaging_find_includes_app_and_tests() -> None:
    data: dict[str, Any] = _load()
    find_conf = data["tool"]["setuptools"]["packages"]["find"]
    include = find_conf["include"]
    assert any(pat.startswith("app") for pat in include)
    assert any(pat.startswith("tests") for pat in include)
