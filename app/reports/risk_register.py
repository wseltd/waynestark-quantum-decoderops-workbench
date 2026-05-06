"""Risk register builder (T090) — one row per blocker/degraded status."""

from __future__ import annotations

from typing import Iterable

from app.metrics.compatibility import RuntimeCompatibilityStatus

__all__ = ["build_risk_register"]


def build_risk_register(
    statuses: Iterable[RuntimeCompatibilityStatus],
) -> list[dict[str, str]]:
    """Return a sorted list of blocker rows. ``ready`` entries are omitted."""
    rows: list[dict[str, str]] = []
    for s in statuses:
        if s.status == "ready":
            continue
        rows.append(
            {
                "backend": s.backend,
                "category": s.category,
                "severity": "blocker" if s.status == "unavailable" else "warning",
                "reason": s.reason,
                "required_action": s.required_action or "",
            }
        )
    rows.sort(key=lambda r: (r["backend"], r["category"]))
    return rows
