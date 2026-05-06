"""Tests for app.config.loader.load_config (T007)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from app.config.loader import load_config
from app.core.errors import IngestionError


def test_load_json_file_populates_settings(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seed": 7, "log_level": "WARNING"}))
    s = load_config(p)
    assert s.seed == 7
    assert s.log_level == "WARNING"


def test_load_yaml_file_populates_settings(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"seed": 11, "log_level": "ERROR"}))
    s = load_config(p)
    assert s.seed == 11
    assert s.log_level == "ERROR"


def test_missing_file_raises_ingestion_error(tmp_path: Path) -> None:
    with pytest.raises(IngestionError) as exc:
        load_config(tmp_path / "nope.yaml")
    assert exc.value.reason_code == "config.file_not_found"


def test_unsupported_suffix_raises_ingestion_error(tmp_path: Path) -> None:
    p = tmp_path / "c.ini"
    p.write_text("seed=1\n")
    with pytest.raises(IngestionError) as exc:
        load_config(p)
    assert exc.value.reason_code == "config.unsupported_format"


def test_unknown_field_raises_ingestion_error(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"not_a_field": 1}))
    with pytest.raises(IngestionError) as exc:
        load_config(p)
    assert exc.value.reason_code == "config.unknown_field"
    assert exc.value.details == {"field": "not_a_field"}


def test_env_var_beats_file_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DECODEROPS_SEED", "42")
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seed": 7}))
    s = load_config(p)
    assert s.seed == 42


def test_explicit_override_beats_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECODEROPS_SEED", "42")
    s = load_config(None, overrides={"seed": 99})
    assert s.seed == 99


def test_explicit_override_beats_file_value(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seed": 7, "log_level": "WARNING"}))
    s = load_config(p, overrides={"seed": 99})
    assert s.seed == 99
    assert s.log_level == "WARNING"


def test_none_path_with_overrides_only() -> None:
    s = load_config(None, overrides={"seed": 5})
    assert s.seed == 5
