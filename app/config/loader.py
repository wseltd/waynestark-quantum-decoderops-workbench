"""Config file loader — YAML/JSON overlay on Settings with strict precedence.

Precedence (lowest to highest):
    Settings defaults < file contents < process env (DECODEROPS_*) < overrides.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from app.config.settings import Settings
from app.core.errors import IngestionError

_ENV_PREFIX = "DECODEROPS_"
# Alias map: env var name -> field name. Keeps env names stable when fields
# are renamed and lets us detect "env already set this field" regardless of
# whether the env name equals the field name.
_FIELD_TO_ENV: dict[str, str] = {}
for _fname, _finfo in Settings.model_fields.items():
    _alias = _finfo.alias
    if _alias is not None and _alias.upper().startswith(_ENV_PREFIX):
        _FIELD_TO_ENV[_fname] = _alias.upper()
    else:
        _FIELD_TO_ENV[_fname] = f"{_ENV_PREFIX}{_fname.upper()}"


def _load_file(path: Path) -> dict[str, Any]:
    """Read *path* and return its parsed mapping."""
    if not path.exists():
        raise IngestionError(
            f"config file not found: {path}",
            reason_code="config.file_not_found",
            details={"path": str(path)},
        )
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text) if text.strip() else {}
    else:
        raise IngestionError(
            f"unsupported config format: {suffix!r}",
            reason_code="config.unsupported_format",
            details={"path": str(path), "suffix": suffix},
        )
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise IngestionError(
            "config file must be a mapping at top level",
            reason_code="config.unsupported_format",
            details={"path": str(path), "top_level_type": type(data).__name__},
        )
    return data


def _validate_keys(data: dict[str, Any]) -> None:
    """Reject unknown top-level keys — Settings is flat, no deep merge."""
    known = set(Settings.model_fields.keys())
    for key in data:
        if key not in known:
            raise IngestionError(
                f"unknown config field: {key!r}",
                reason_code="config.unknown_field",
                details={"field": key},
            )


def load_config(
    path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Merge a YAML or JSON config file with env vars and explicit overrides.

    Precedence (lowest → highest): Settings defaults, file, env, overrides.

    Args:
        path: Config file path. ``None`` skips file loading.
        overrides: Explicit dict merged last; beats both file and env.

    Returns:
        A fully-validated :class:`Settings` instance.

    Raises:
        IngestionError: With ``reason_code`` one of
            ``config.file_not_found``, ``config.unsupported_format``, or
            ``config.unknown_field``.
    """
    file_dict: dict[str, Any] = {}
    if path is not None:
        file_dict = _load_file(Path(path))
        _validate_keys(file_dict)

    overrides = dict(overrides or {})
    _validate_keys(overrides)

    # env beats file: for any field with DECODEROPS_<FIELD> set in env,
    # drop the file value so pydantic-settings env resolution wins.
    merged: dict[str, Any] = dict(file_dict)
    for fname, env_name in _FIELD_TO_ENV.items():
        if env_name in os.environ and fname in merged:
            merged.pop(fname)

    # overrides beat env: pass overrides directly to Settings kwargs — these
    # take precedence over env resolution via pydantic-settings semantics.
    merged.update(overrides)

    return Settings(**merged)
