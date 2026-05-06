"""CUDA-Q target capability adapter (T028).

NOT a Decoder Protocol implementation — this adapter reports cudaq
target availability for the compatibility matrix and deployment-
readiness report. Consumed by the unified capability detector in
:mod:`app.core.capability`.

Design decisions:

    * ``cudaq`` is import-guarded at module top. Three module-level
      constants expose state: ``CUDAQ_AVAILABLE``, ``CUDAQ_VERSION``,
      ``CUDAQ_IMPORT_ERROR``.
    * ``enumerate_targets()`` probes each declared cudaq target in
      isolation (try/except per target) so one broken target (e.g.
      ``nvidia`` missing a GPU) does not hide others.
    * ``probe()`` calls ``cudaq.get_targets()`` and trials
      ``cudaq.set_target()`` on each; the global target is restored
      after the probe via a try/finally so the probe has no lasting
      side-effect on process state.
    * We record per-target ``unavailable_reason`` so the risk register
      can surface exactly which targets need hardware/licensing work.
    * NO decode_batch — this is a capability adapter, not a decoder.
"""

from __future__ import annotations

import time
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from app.core.capability_report import CapabilityReport

CUDAQ_AVAILABLE: bool
CUDAQ_VERSION: str | None
CUDAQ_IMPORT_ERROR: str | None
try:
    import cudaq as _cudaq

    CUDAQ_AVAILABLE = True
    CUDAQ_VERSION = str(getattr(_cudaq, "__version__", "unknown"))
    CUDAQ_IMPORT_ERROR = None
except ImportError as _exc:
    CUDAQ_AVAILABLE = False
    CUDAQ_VERSION = None
    CUDAQ_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


__all__ = [
    "CUDAQ_AVAILABLE",
    "CUDAQ_IMPORT_ERROR",
    "CUDAQ_VERSION",
    "CudaqCapabilityAdapter",
    "CudaqTargetInfo",
]


class CudaqTargetInfo(BaseModel):
    """Per-target probe result.

    Separate from :class:`app.core.capability.CapabilityReport` (which
    is the probe-level summary) — this is a per-target record the
    adapter returns in a list.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    simulator_backend: str | None
    num_qpus: int
    gpu_required: bool
    available: bool
    unavailable_reason: str | None


class CudaqCapabilityAdapter:
    """Enumerate cudaq targets and report availability per-target."""

    def available(self) -> CapabilityReport:
        start = time.perf_counter_ns()
        if not CUDAQ_AVAILABLE:
            return CapabilityReport.unavailable(
                reason=f"cudaq not installed: {CUDAQ_IMPORT_ERROR}",
                required=["cudaq"],
                category="not_installed",
            )
        # Successfully imported — claim "ready" and let the per-target
        # probe() method surface finer granularity when consumers need
        # it. Risk register joins on available() so we must return
        # available=True here, with targets cached for subsequent
        # enumerate_targets() calls.
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return CapabilityReport.ready(
            reason=f"cudaq {CUDAQ_VERSION} imported",
            required=["cudaq"],
            detected_versions={"cudaq": CUDAQ_VERSION or "unknown"},
            probe_latency_ms=elapsed_ms,
        )

    def enumerate_targets(self) -> list[CudaqTargetInfo]:
        """Run ``cudaq.get_targets()`` and test-activate each in isolation.

        Each target's own try/except means a broken target (e.g.
        nvidia without GPU) never hides the availability of the
        CPU simulators.
        """
        if not CUDAQ_AVAILABLE:
            return []
        import cudaq

        # Preserve the caller's global target across the probe.
        original_target = None
        try:
            original_target = cudaq.get_target()
        except Exception:  # pragma: no cover - defensive
            original_target = None

        results: list[CudaqTargetInfo] = []
        try:
            for tgt in cudaq.get_targets():
                name = getattr(tgt, "name", str(tgt))
                description = getattr(tgt, "description", "") or ""
                simulator_backend = getattr(tgt, "simulator", None)
                num_qpus = int(getattr(tgt, "num_qpus", lambda: 0)() if callable(
                    getattr(tgt, "num_qpus", None)) else getattr(tgt, "num_qpus", 0))
                gpu_required = "nvidia" in name.lower() or "cuda" in name.lower()
                unavailable_reason: str | None = None
                target_available: bool = True
                try:
                    cudaq.set_target(name)
                except Exception as exc:  # per-target isolation
                    target_available = False
                    unavailable_reason = f"{type(exc).__name__}: {exc}"
                results.append(
                    CudaqTargetInfo(
                        name=name,
                        description=description,
                        simulator_backend=(
                            str(simulator_backend)
                            if simulator_backend is not None
                            else None
                        ),
                        num_qpus=num_qpus,
                        gpu_required=gpu_required,
                        available=target_available,
                        unavailable_reason=unavailable_reason,
                    )
                )
        finally:
            # Restore the caller's global target, always. Failure to
            # reset would leak probe state into subsequent tests and
            # benchmark runs.
            if original_target is not None:
                try:
                    cudaq.set_target(
                        getattr(original_target, "name", original_target)
                    )
                except Exception:  # pragma: no cover - final defensive
                    pass
        # Stable ordering — target names (cudaq returns order-preserved
        # list already but sort to harden artefact stability in the
        # risk register).
        results.sort(key=lambda r: r.name)
        return results

    def probe(self) -> dict:
        """Aggregate probe result suitable for JSON embedding."""
        return {
            "cudaq_available": CUDAQ_AVAILABLE,
            "cudaq_version": CUDAQ_VERSION,
            "cudaq_import_error": CUDAQ_IMPORT_ERROR,
            "targets": [t.model_dump() for t in self.enumerate_targets()],
        }
