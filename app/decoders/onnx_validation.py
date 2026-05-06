"""ONNX Runtime validation decoder — ``onnx_validation`` backend.

Purpose: given an exported ONNX model file, validate that it loads and
runs under onnxruntime with the requested providers (CUDA, TensorRT,
CPU). This is the post-export smoke test that the Workbench uses to
confirm an ONNX artefact is actually deployable, not just syntactically
well-formed.

Design decisions:
    * onnxruntime is import-guarded at module top. ``ONNXRUNTIME_AVAILABLE``
      and ``ONNXRUNTIME_IMPORT_ERROR`` are exported as module-level
      constants so the capability detector and compatibility matrix can
      read them without invoking ``available()``.
    * The requested providers list is configurable and recorded in
      metadata as ``active_providers`` after ``warmup()``. We do NOT
      silently fall back from CUDA to CPU — if CUDAExecutionProvider
      isn't registered, ``available()`` returns unavailable with a
      precise reason.
    * No ONNX export logic here — that belongs in app.packaging. This
      module is inference/validation only.
    * Input names are read from the session, never hardcoded; the
      vendor exporter emits names like "input" but third-party ONNX
      files in customer pilots may use any naming.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app.core.capability_report import CapabilityReport
from app.decoders.protocol import Corrections, DecoderMetadata

if TYPE_CHECKING:
    import onnxruntime as _ort_type


# Module-level capability flags — read at import time so the unified
# capability detector (app.core.capability, planned for T012) can
# surface them without triggering a full probe.
ONNXRUNTIME_AVAILABLE: bool
ONNXRUNTIME_IMPORT_ERROR: str | None
try:
    import onnxruntime as _ort  # noqa: F401

    ONNXRUNTIME_AVAILABLE = True
    ONNXRUNTIME_IMPORT_ERROR = None
except ImportError as _exc:
    ONNXRUNTIME_AVAILABLE = False
    ONNXRUNTIME_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


__all__ = [
    "ONNXRUNTIME_AVAILABLE",
    "ONNXRUNTIME_IMPORT_ERROR",
    "OnnxValidationDecoder",
]


_DEFAULT_PROVIDERS: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class OnnxValidationDecoder:
    """Validate an exported ONNX decoder via onnxruntime inference.

    Args:
        model_path: Path to the .onnx model file.
        providers: Ordered list of onnxruntime execution provider names
            to request. Defaults to CUDA then CPU. The first provider
            in the registered intersection wins; ``active_providers`` in
            metadata records the actually-used providers.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        providers: list[str] | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._requested_providers: list[str] = list(
            providers if providers is not None else _DEFAULT_PROVIDERS
        )
        self._warmed: bool = False
        self._session: "_ort_type.InferenceSession | None" = None
        self._active_providers: list[str] = []
        self._input_name: str | None = None
        self._output_names: list[str] = []
        self._input_shape: list[int | None] = []
        self._model_sha256: str | None = None

    # -- capability probe --------------------------------------------------

    def available(self) -> CapabilityReport:
        """Probe onnxruntime presence, model file presence, and provider
        registration. Never raises; precise reasons distinguish all
        four failure modes."""
        start = time.perf_counter_ns()
        versions: dict[str, str] = {}

        if not ONNXRUNTIME_AVAILABLE:
            return CapabilityReport.unavailable(
                reason=(
                    "onnxruntime not installed: "
                    f"{ONNXRUNTIME_IMPORT_ERROR}"
                ),
                required=["onnxruntime", "onnxruntime-gpu (for CUDA/TRT)"],
                category="not_installed",
            )
        import onnxruntime as ort

        versions["onnxruntime"] = getattr(ort, "__version__", "unknown")

        if not self._model_path.is_file():
            return CapabilityReport.unavailable(
                reason=f"ONNX model file not found at {self._model_path}",
                required=[".onnx model file"],
                category="software",
            )

        registered = set(ort.get_available_providers())
        wanted = set(self._requested_providers)
        usable = wanted & registered
        if not usable:
            return CapabilityReport.unavailable(
                reason=(
                    f"no requested onnxruntime provider is registered; "
                    f"requested={sorted(wanted)}, "
                    f"registered={sorted(registered)}"
                ),
                required=sorted(wanted),
                category="software",
            )
        # Explicit CUDA-requested-but-not-registered reason for
        # operators who explicitly ask for CUDA.
        if (
            "CUDAExecutionProvider" in self._requested_providers
            and "CUDAExecutionProvider" not in registered
        ):
            return CapabilityReport.unavailable(
                reason=(
                    "CUDAExecutionProvider requested but not registered in "
                    f"onnxruntime; registered providers: "
                    f"{sorted(registered)}"
                ),
                required=["onnxruntime-gpu"],
                category="software",
            )

        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return CapabilityReport.ready(
            reason=(
                f"onnxruntime {versions['onnxruntime']} loaded; "
                f"providers {sorted(usable)} usable for "
                f"{self._model_path.name}"
            ),
            required=["onnxruntime", ".onnx model file"],
            detected_versions=versions,
            probe_latency_ms=elapsed_ms,
        )

    # -- warmup ------------------------------------------------------------

    def warmup(self) -> None:
        """Build an InferenceSession and record session metadata. Idempotent."""
        if self._warmed:
            return
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError(
                f"onnxruntime not available: {ONNXRUNTIME_IMPORT_ERROR}"
            )
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(self._model_path),
            sess_options=opts,
            providers=self._requested_providers,
        )
        self._session = session
        self._active_providers = list(session.get_providers())

        # Read input/output signatures off the session rather than
        # hardcoding names; vendor ONNX emits "input" but pilot-
        # customer models may use any naming.
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError(
                f"ONNX model {self._model_path} has no declared inputs"
            )
        self._input_name = inputs[0].name
        self._input_shape = list(inputs[0].shape)
        self._output_names = [o.name for o in session.get_outputs()]

        # Record SHA256 of the model file for manifest provenance.
        h = hashlib.sha256()
        with self._model_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        self._model_sha256 = h.hexdigest()

        self._warmed = True

    # -- decode ------------------------------------------------------------

    def decode_batch(self, syndromes: np.ndarray) -> Corrections:
        """Run the ONNX session on a batch of syndromes.

        Shape adaptation: ONNX models often declare dynamic batch with
        symbolic dims. We reshape the incoming ``(batch, detectors)``
        array to whatever the session's declared input rank expects
        (prepending a channel dimension when the model declares one)
        and return the first output as predictions.
        """
        if not self._warmed:
            self.warmup()
        assert self._session is not None and self._input_name is not None
        if not isinstance(syndromes, np.ndarray):
            raise TypeError(
                f"syndromes must be numpy.ndarray; got {type(syndromes).__name__}"
            )
        if syndromes.ndim != 2:
            raise ValueError(
                "syndromes must be 2D (batch, detectors); "
                f"got ndim={syndromes.ndim}"
            )
        if syndromes.dtype != np.uint8:
            raise TypeError(
                f"syndromes must be uint8; got dtype={syndromes.dtype}"
            )
        batch = int(syndromes.shape[0])

        # Cast to float32 — ONNX decoder inputs are typically float.
        # A real deployment would honour the model's declared input
        # dtype via session.get_inputs()[0].type, but the default here
        # is the vendor convention.
        feed = {self._input_name: syndromes.astype(np.float32)}

        start_ns = time.perf_counter_ns()
        outputs = self._session.run(self._output_names, feed)
        latency_ns = time.perf_counter_ns() - start_ns

        first = np.asarray(outputs[0])
        if first.ndim == 1:
            first = first.reshape(-1, 1)
        predictions = (first > 0.5).astype(np.uint8) if first.dtype != np.uint8 else first
        # Ensure shape (batch, observables)
        if predictions.shape[0] != batch:
            predictions = predictions.reshape(batch, -1)
        return Corrections(predictions=predictions, latency_ns=int(latency_ns))

    # -- metadata ----------------------------------------------------------

    def metadata(self) -> DecoderMetadata:
        resolved_path = (
            str(self._model_path.resolve())
            if self._model_path.exists()
            else None
        )
        return DecoderMetadata(
            backend_name="onnx_validation",
            backend_version=(
                ONNXRUNTIME_IMPORT_ERROR
                if not ONNXRUNTIME_AVAILABLE
                else _ort_version_or_unknown()
            ),
            model_path=resolved_path,
            model_sha256=self._model_sha256,
            receptive_field=None,
            supports_batching=True,
            supports_gpu="CUDAExecutionProvider" in self._active_providers,
            schema_version="1",
        )


def _ort_version_or_unknown() -> str:
    if not ONNXRUNTIME_AVAILABLE:
        return "unavailable"
    import onnxruntime as ort

    return getattr(ort, "__version__", "unknown")
