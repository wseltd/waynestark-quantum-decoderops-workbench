"""cuQuantum capability adapter (T030).

Capability-reporter only. Probes cuquantum import and, when possible,
a minimal tensor-network contraction on the default CUDA device. No
large contractions, no persistent GPU allocations — allocate, verify,
release.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.capability_report import CapabilityReport

CUQUANTUM_AVAILABLE: bool
CUQUANTUM_VERSION: str | None
CUQUANTUM_IMPORT_ERROR: str | None
try:
    import cuquantum as _cuq

    CUQUANTUM_AVAILABLE = True
    CUQUANTUM_VERSION = str(getattr(_cuq, "__version__", "unknown"))
    CUQUANTUM_IMPORT_ERROR = None
except ImportError as _exc:
    CUQUANTUM_AVAILABLE = False
    CUQUANTUM_VERSION = None
    CUQUANTUM_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


__all__ = [
    "CUQUANTUM_AVAILABLE",
    "CUQUANTUM_IMPORT_ERROR",
    "CUQUANTUM_VERSION",
    "CuQuantumCapability",
]


class CuQuantumCapability:
    """cuquantum availability reporter. NOT a Decoder."""

    def available(self) -> CapabilityReport:
        start = time.perf_counter_ns()
        if not CUQUANTUM_AVAILABLE:
            return CapabilityReport.unavailable(
                reason=(
                    f"cuquantum package not installed: "
                    f"{CUQUANTUM_IMPORT_ERROR}"
                ),
                required=["cuquantum-python-cu13"],
                category="not_installed",
            )
        # Package importable; probe CUDA device presence via cupy
        # (cuquantum depends on cupy for device ops).
        details = self.probe()
        if details.get("error"):
            reason = str(details["error"])
            if "cuda" in reason.lower() or "device" in reason.lower():
                cat = "machine"
                req = ["nvidia-gpu", "CUDA-13 runtime"]
            else:
                cat = "runtime"
                req = ["cuquantum-python-cu13", "cupy-cuda13x"]
            return CapabilityReport.unavailable(
                reason=f"cuquantum probe failed: {reason}",
                required=req,
                category=cat,
            )
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return CapabilityReport.ready(
            reason=f"cuquantum {CUQUANTUM_VERSION} probed successfully",
            required=["cuquantum-python-cu13"],
            detected_versions={"cuquantum": CUQUANTUM_VERSION or "unknown"},
            probe_latency_ms=elapsed_ms,
        )

    def probe(self) -> dict[str, Any]:
        """Minimal tensor-network probe (or an import-only probe if cupy
        is not usable). JSON-safe dict return; ``error`` key present
        iff probe hit a failure we want surfaced in ``available()``.
        """
        out: dict[str, Any] = {
            "cuquantum_available": CUQUANTUM_AVAILABLE,
            "cuquantum_version": CUQUANTUM_VERSION,
            "cuquantum_import_error": CUQUANTUM_IMPORT_ERROR,
        }
        if not CUQUANTUM_AVAILABLE:
            return out
        # Attempt a trivial cupy op to confirm the device is reachable
        # without allocating anything persistent.
        try:
            import cupy as cp

            out["cupy_version"] = str(getattr(cp, "__version__", "unknown"))
            # 2-element device allocation, immediate release.
            a = cp.asarray([1.0, 2.0], dtype=cp.float32)
            # Trigger a sync to surface driver errors eagerly.
            _ = float(a.sum().get())
            del a
            # Record the device index actually used.
            try:
                out["cuda_device_index"] = int(cp.cuda.runtime.getDevice())
            except Exception:  # pragma: no cover - defensive
                out["cuda_device_index"] = None
        except Exception as exc:  # noqa: BLE001 - probe wrapper
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out
