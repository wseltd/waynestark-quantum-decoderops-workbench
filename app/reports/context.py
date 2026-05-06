"""Build a typed render context from Run + Metrics + Fingerprint (T087)."""

from __future__ import annotations

from typing import Any

__all__ = ["build_context"]


def build_context(
    *,
    run: dict[str, Any],
    metrics: list[dict[str, Any]],
    artefacts: list[dict[str, Any]],
    host: dict[str, Any],
    decoders: list[dict[str, Any]],
    sweep_axes: dict[str, Any],
    shots_total: int,
    reproducibility_fingerprint_sha256: str,
) -> dict[str, Any]:
    """Assemble the dict the renderers consume.

    Pure function: no I/O, no DB, no environment reads.
    """
    return {
        "run_id": run.get("run_id", ""),
        "git_sha": run.get("git_sha", ""),
        "config_sha256": run.get("config_sha256", run.get("config_hash", "")),
        "pip_freeze_digest": run.get("pip_freeze_digest", ""),
        "rng_master_seed": run.get("rng_master_seed", 0),
        "started_at_utc": run.get("started_at_utc", ""),
        "finished_at_utc": run.get("finished_at_utc", ""),
        "host": dict(host),
        "backend": run.get("backend", {}),
        "decoders": list(decoders),
        "sweep_axes": dict(sweep_axes),
        "shots_total": int(shots_total),
        "metrics": list(metrics),
        "artefacts": list(artefacts),
        "reproducibility_fingerprint_sha256": reproducibility_fingerprint_sha256,
    }
