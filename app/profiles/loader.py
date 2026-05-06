"""External profile loader — JSON / YAML / dict into ProfileSpec.

The loader enforces the same validation as ProfileSpec itself.
External profiles (e.g. customer-specific YAML one day) go through
this path so every ProfileSpec instance — built-in or external — is
identically validated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from app.core.errors import IngestionError
from app.profiles.schema import ProfileSpec

__all__ = ["load_profile", "load_profile_from_dict"]


def load_profile_from_dict(data: dict[str, Any]) -> ProfileSpec:
    try:
        return ProfileSpec.model_validate(data)
    except Exception as e:  # Pydantic validation
        raise IngestionError(
            f"profile validation failed: {e}",
            reason_code="profile.invalid",
            details={"error": str(e)[:1000]},
        ) from e


def load_profile(path: Path | str) -> ProfileSpec:
    p = Path(path).expanduser()
    if not p.exists():
        raise IngestionError(
            f"profile file not found: {p}",
            reason_code="profile.file_not_found",
            details={"path": str(p)},
        )
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text) or {}
        elif suffix == ".json":
            data = json.loads(text) if text.strip() else {}
        else:
            raise IngestionError(
                f"unsupported profile format: {suffix!r}",
                reason_code="profile.unsupported_format",
                details={"path": str(p), "suffix": suffix},
            )
    except yaml.YAMLError as e:
        raise IngestionError(
            f"YAML parse failed: {e}",
            reason_code="profile.invalid",
        ) from e
    except json.JSONDecodeError as e:
        raise IngestionError(
            f"JSON parse failed: {e}",
            reason_code="profile.invalid",
        ) from e
    if not isinstance(data, dict):
        raise IngestionError(
            "profile top-level must be a mapping",
            reason_code="profile.invalid",
            details={"top_level_type": type(data).__name__},
        )
    return load_profile_from_dict(data)
