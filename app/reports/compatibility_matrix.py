"""Compatibility matrix builder (T089).

Aggregates RuntimeCompatibilityStatus records into a 2-D matrix keyed by
(backend, category) for the deployment-readiness report.
"""

from __future__ import annotations

from typing import Iterable

from app.metrics.compatibility import RuntimeCompatibilityStatus

__all__ = ["build_compatibility_matrix"]


def build_compatibility_matrix(
    statuses: Iterable[RuntimeCompatibilityStatus],
) -> dict[str, dict[str, str]]:
    """Return {backend: {status, reason, category, required_action}}."""
    out: dict[str, dict[str, str]] = {}
    for s in statuses:
        out[s.backend] = {
            "status": s.status,
            "reason": s.reason,
            "category": s.category,
            "required_action": s.required_action or "",
        }
    return dict(sorted(out.items(), key=lambda kv: kv[0]))
