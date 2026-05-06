"""Deterministic JSON renderer (T086)."""

from __future__ import annotations

import json
from typing import Any

__all__ = ["render_json"]


def render_json(context: dict[str, Any]) -> str:
    """Serialise a report context to canonical JSON (sorted, compact)."""
    return json.dumps(
        context,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
