"""CUDA-Q QEC capability adapter (T029).

Capability-reporter only. Exposes ``CudaqQecCapability`` with
``available()`` returning the decoder-layer CapabilityReport and
``probe()`` returning a dict suitable for JSON embedding. Distinguishes
ImportError (package missing), AttributeError (API drift across
cudaq-qec versions), and RuntimeError (package installed but runtime
blocker) in the reason string.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.capability_report import CapabilityReport

CUDAQ_QEC_AVAILABLE: bool
CUDAQ_QEC_VERSION: str | None
CUDAQ_QEC_IMPORT_ERROR: str | None
try:
    import cudaq_qec as _cudaq_qec  # type: ignore[import-not-found]

    CUDAQ_QEC_AVAILABLE = True
    CUDAQ_QEC_VERSION = str(getattr(_cudaq_qec, "__version__", "unknown"))
    CUDAQ_QEC_IMPORT_ERROR = None
except ImportError as _exc:
    # Try the alternate dotted path used by some builds.
    try:
        import cudaq.qec as _cudaq_qec_alt  # type: ignore[import-not-found]

        CUDAQ_QEC_AVAILABLE = True
        CUDAQ_QEC_VERSION = str(getattr(_cudaq_qec_alt, "__version__", "unknown"))
        CUDAQ_QEC_IMPORT_ERROR = None
    except ImportError as _exc2:
        CUDAQ_QEC_AVAILABLE = False
        CUDAQ_QEC_VERSION = None
        CUDAQ_QEC_IMPORT_ERROR = (
            f"neither cudaq_qec nor cudaq.qec importable: "
            f"{type(_exc2).__name__}: {_exc2}"
        )


__all__ = [
    "CUDAQ_QEC_AVAILABLE",
    "CUDAQ_QEC_IMPORT_ERROR",
    "CUDAQ_QEC_VERSION",
    "CudaqQecCapability",
]


class CudaqQecCapability:
    """cudaq-qec availability reporter. NOT a Decoder."""

    def available(self) -> CapabilityReport:
        start = time.perf_counter_ns()
        if not CUDAQ_QEC_AVAILABLE:
            return CapabilityReport.unavailable(
                reason=f"cudaq-qec not installed: {CUDAQ_QEC_IMPORT_ERROR}",
                required=["cudaq-qec"],
                category="not_installed",
            )
        # Package importable — attempt the API-shape probe; distinguish
        # AttributeError (API drift) from RuntimeError (runtime blocker).
        try:
            self.probe()
        except AttributeError as exc:
            return CapabilityReport.unavailable(
                reason=(
                    f"cudaq-qec API drift: {type(exc).__name__}: {exc}"
                ),
                required=["cudaq-qec-matching-version"],
                category="version_mismatch",
            )
        except RuntimeError as exc:
            return CapabilityReport.unavailable(
                reason=f"cudaq-qec runtime error: {exc}",
                required=["cudaq-qec", "CUDA-13 runtime", "nvidia-gpu"],
                category="runtime",
            )
        except Exception as exc:  # noqa: BLE001 - wrap any other unexpected
            return CapabilityReport.unavailable(
                reason=(
                    f"cudaq-qec probe raised {type(exc).__name__}: {exc}"
                ),
                required=["cudaq-qec"],
                category="runtime",
            )
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return CapabilityReport.ready(
            reason=f"cudaq-qec {CUDAQ_QEC_VERSION} importable",
            required=["cudaq-qec"],
            detected_versions={"cudaq_qec": CUDAQ_QEC_VERSION or "unknown"},
            probe_latency_ms=elapsed_ms,
        )

    def probe(self) -> dict[str, Any]:
        """Minimal API-shape probe. Returns a JSON-safe dict.

        We exercise only read-only introspection — listing codes or
        decoders. No kernels are compiled or executed.
        """
        out: dict[str, Any] = {
            "cudaq_qec_available": CUDAQ_QEC_AVAILABLE,
            "cudaq_qec_version": CUDAQ_QEC_VERSION,
            "cudaq_qec_import_error": CUDAQ_QEC_IMPORT_ERROR,
        }
        if not CUDAQ_QEC_AVAILABLE:
            return out
        # Try primary, then alternate dotted path.
        try:
            import cudaq_qec as qec  # type: ignore[import-not-found]
        except ImportError:
            import cudaq.qec as qec  # type: ignore[import-not-found]

        # Best-effort inventory — attribute presence, not invocation.
        for attr in (
            "get_code",
            "get_decoder",
            "get_available_codes",
            "sample_memory_circuit",
            "dem_from_memory_circuit",
        ):
            out[f"has_{attr}"] = hasattr(qec, attr)
        # Real-API enumeration when the factory exists.
        if hasattr(qec, "get_available_codes"):
            try:
                out["available_codes"] = list(qec.get_available_codes())
            except Exception as exc:  # noqa: BLE001
                out["available_codes_error"] = f"{type(exc).__name__}: {exc}"
        # Real-API structural probe — build a d=3 surface code and
        # introspect its stabiliser structure. Read-only, fast.
        if hasattr(qec, "get_code"):
            try:
                code = qec.get_code("surface_code", distance=3)
                out["surface_code_d3"] = {
                    "num_data_qubits": code.get_num_data_qubits(),
                    "num_ancilla_qubits": code.get_num_ancilla_qubits(),
                    "num_x_stabilizers": code.get_num_x_stabilizers(),
                    "num_z_stabilizers": code.get_num_z_stabilizers(),
                }
            except Exception as exc:  # noqa: BLE001
                out["surface_code_d3_error"] = f"{type(exc).__name__}: {exc}"
        return out
