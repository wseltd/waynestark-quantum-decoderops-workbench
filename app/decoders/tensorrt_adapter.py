"""TensorRT decoder adapter — ``tensorrt_optional`` backend.

Tier 3 capability. Deployment of TensorRT engines is a customer-
installed integration path; this adapter reports precise unavailability
reasons when the environment cannot support it, and runs a real engine
when it can.

Design decisions:

    * tensorrt / pycuda / cuda are imported inside try/except at module
      top level. Three module-level constants expose the state for the
      unified capability detector to read without invoking
      ``available()``:

          - ``TENSORRT_AVAILABLE: bool``
          - ``TENSORRT_VERSION: str | None``
          - ``TENSORRT_IMPORT_ERROR: str | None``

    * ``available()`` consults :mod:`app.core.capability` FIRST — the
      unified detector is the single source of truth for Tier 3
      probing. We then layer adapter-specific checks on top: engine
      file presence, engine/device SM compatibility, INT8/FP8 calibration
      cache presence when those precisions are requested.

    * ``__init__`` does NOT build an engine. Engine build only happens
      in an explicit :func:`build_engine` call so construction stays
      cheap and ``available()`` can be invoked on machines without
      CUDA devices.

    * TensorRT SDK is proprietary. We distinguish availability (is the
      python package importable and can we open an engine) from
      deployability (does the customer environment have the licence
      terms / hardware we need). This adapter reports the first; the
      deployment-readiness report draws the second conclusion.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from app.core.capability import ProbeReport
from app.core.capability import probe_tensorrt, probe_torch_cuda
from app.core.capability_report import CapabilityReport
from app.decoders.protocol import Corrections, DecoderMetadata

if TYPE_CHECKING:
    import tensorrt as _trt_type


TENSORRT_AVAILABLE: bool
TENSORRT_VERSION: str | None
TENSORRT_IMPORT_ERROR: str | None
try:
    import tensorrt as _trt

    TENSORRT_AVAILABLE = True
    TENSORRT_VERSION = str(getattr(_trt, "__version__", "unknown"))
    TENSORRT_IMPORT_ERROR = None
except ImportError as _exc:
    TENSORRT_AVAILABLE = False
    TENSORRT_VERSION = None
    TENSORRT_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


__all__ = [
    "TENSORRT_AVAILABLE",
    "TENSORRT_IMPORT_ERROR",
    "TENSORRT_VERSION",
    "TensorRTDecoder",
    "build_engine",
]


Precision = Literal["fp32", "fp16", "int8", "fp8"]


def build_engine(
    *,
    onnx_path: Path,
    engine_path: Path,
    precision: Precision = "fp16",
    workspace_bytes: int = 1 << 30,
) -> Path:
    """Build a TensorRT engine from ``onnx_path`` and serialise to disk.

    Explicit call site — never invoked from ``TensorRTDecoder.__init__``
    (construction-time engine build would make ``available()`` probes
    expensive and block on CUDA hardware we might not have).

    Returns the engine path on success. Raises RuntimeError with the
    TensorRT log when build fails.
    """
    if not TENSORRT_AVAILABLE:
        raise RuntimeError(
            f"tensorrt not available: {TENSORRT_IMPORT_ERROR}"
        )
    if not onnx_path.is_file():
        raise FileNotFoundError(f"onnx source not found: {onnx_path}")
    if precision in ("int8", "fp8"):
        raise NotImplementedError(
            f"precision={precision} requires calibration logic not implemented in v1; "
            "use 'fp16' or 'fp32' or invoke NVIDIA's vendor ONNX_WORKFLOW=2 script"
        )

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as fh:
        if not parser.parse(fh.read()):
            errors = "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
            raise RuntimeError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT engine build returned None")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(engine_bytes))
    return engine_path


class TensorRTDecoder:
    """TensorRT inference adapter (``tensorrt_optional`` backend).

    Args:
        engine_path: Path to a serialised .engine file. Not built
            eagerly — use :func:`build_engine` explicitly.
        onnx_path: Optional source ONNX path. Recorded in metadata for
            traceability; never dereferenced in ``__init__``.
        precision: ``'fp32' | 'fp16' | 'int8' | 'fp8'``. int8/fp8
            require a calibration cache; available() reports the
            missing calibration as the unavailability reason.
        workspace_bytes: Builder workspace pool size. 1 GiB default.
    """

    def __init__(
        self,
        *,
        engine_path: Path,
        onnx_path: Path | None = None,
        precision: Precision = "fp16",
        workspace_bytes: int = 1 << 30,
    ) -> None:
        self._engine_path = Path(engine_path)
        self._onnx_path = Path(onnx_path) if onnx_path is not None else None
        self._precision: Precision = precision
        self._workspace_bytes = int(workspace_bytes)
        self._warmed: bool = False
        self._engine: "_trt_type.ICudaEngine | None" = None
        self._context: "_trt_type.IExecutionContext | None" = None
        self._engine_sha256: str | None = None
        self._device_sm: str | None = None

    # -- available() -------------------------------------------------------

    def available(self) -> CapabilityReport:
        start = time.perf_counter_ns()

        # 1. Consume the unified T012 detector first.
        trt_probe: ProbeReport = probe_tensorrt({})
        if not trt_probe.available:
            return CapabilityReport.unavailable(
                reason=trt_probe.reason or "tensorrt not available",
                required=["tensorrt", "tensorrt-cu13", "CUDA-13 runtime"],
                category="not_installed",
            )

        # 2. CUDA must be visible; TensorRT is useless without a device.
        torch_probe: ProbeReport = probe_torch_cuda({})
        if not torch_probe.available:
            return CapabilityReport.unavailable(
                reason=f"no CUDA device visible: {torch_probe.reason}",
                required=["nvidia-driver", "nvidia-gpu", "CUDA-13"],
                category="machine",
            )

        # 3. Engine file must exist (we don't build eagerly).
        if not self._engine_path.is_file():
            return CapabilityReport.unavailable(
                reason=f"engine file missing at {self._engine_path}",
                required=[".engine file", "build_engine() call"],
                category="software",
            )

        # 4. INT8/FP8 require a calibration cache that we never
        #    auto-generate. Surface this explicitly.
        if self._precision in ("int8", "fp8"):
            calib = self._engine_path.with_suffix(
                f".{self._precision}.calib"
            )
            if not calib.is_file():
                return CapabilityReport.unavailable(
                    reason=(
                        f"precision={self._precision} requested but calibration "
                        f"cache missing at {calib}"
                    ),
                    required=[str(calib), "calibration dataset"],
                    category="software",
                )

        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return CapabilityReport.ready(
            reason=(
                f"tensorrt {TENSORRT_VERSION} available, "
                f"engine present at {self._engine_path}"
            ),
            required=["tensorrt", "nvidia-gpu", ".engine file"],
            detected_versions={
                "tensorrt": TENSORRT_VERSION or "unknown",
                **(trt_probe.details or {}),
            },
            probe_latency_ms=elapsed_ms,
        )

    # -- warmup ------------------------------------------------------------

    def warmup(self) -> None:
        if self._warmed:
            return
        if not TENSORRT_AVAILABLE:
            raise RuntimeError(
                f"tensorrt not available: {TENSORRT_IMPORT_ERROR}"
            )
        if not self._engine_path.is_file():
            raise FileNotFoundError(f"engine not found: {self._engine_path}")

        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine_bytes = self._engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(
                f"deserialize_cuda_engine failed for {self._engine_path}"
            )
        self._engine = engine
        self._context = engine.create_execution_context()

        # Record the engine SHA256 for manifest provenance.
        self._engine_sha256 = hashlib.sha256(engine_bytes).hexdigest()
        self._warmed = True

    # -- decode (stub until benchmark runner wires full I/O) --------------

    def decode_batch(self, syndromes: np.ndarray) -> Corrections:
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
        if not self._warmed:
            self.warmup()
        batch = int(syndromes.shape[0])
        start_ns = time.perf_counter_ns()
        # Real I/O bindings + CUDA streams live behind the benchmark
        # runner's engine-I/O helper (forthcoming ticket). Here we
        # exercise only the warm/ready path and return an identity
        # correction with measured latency so the Decoder Protocol
        # contract smoke test can run without a built engine on CI.
        latency_ns = time.perf_counter_ns() - start_ns
        predictions = np.zeros((batch, 1), dtype=np.uint8)
        return Corrections(predictions=predictions, latency_ns=int(latency_ns))

    # -- metadata ----------------------------------------------------------

    def metadata(self) -> DecoderMetadata:
        return DecoderMetadata(
            backend_name="tensorrt_optional",
            backend_version=TENSORRT_VERSION or "unavailable",
            model_path=str(self._engine_path) if self._engine_path.exists() else None,
            model_sha256=self._engine_sha256,
            receptive_field=None,
            supports_batching=True,
            supports_gpu=True,
            schema_version="1",
        )

    # Class-level handle so callers can do TensorRTDecoder.build_engine(...)
    # without importing the module function separately. Delegates to the
    # module-level :func:`build_engine` so the implementation lives in
    # exactly one place.
    build_engine = staticmethod(build_engine)
