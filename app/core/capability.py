"""Unified Tier 3 capability detector.

This module is the single source of truth consumed by:

    * every decoder adapter's ``available()`` method that needs
      cross-library facts (e.g. TensorRT checking "is the cu13 CUDA
      stack loadable and the TensorRT Python package importable")
    * the ``tests/runtime_capability/`` test layer
    * the compatibility-matrix renderer
    * the deployment-readiness report's risk register

Contract:

    * Probes NEVER raise. Failure modes become a ``CapabilityReport``
      with ``available=False`` and a precise, human-readable reason.
    * Import-guarded heavyweight modules (torch, tensorrt, cudaq,
      cudaq_qec, cuquantum, cupy, modelopt, onnxruntime) are imported
      only INSIDE the probe functions, never at module top level. A
      dedicated test pins this invariant.
    * ``detect_all()`` walks every known capability name and returns a
      mapping; it must not raise no matter what the host environment
      looks like (no GPU, no cu13 stack, no proprietary Tier 3).

Name disambiguation: this module defines its OWN ``CapabilityReport``
focused on probe-result shape (``name``, ``available``, ``version``,
``reason``, ``details``). The decoder-layer ``CapabilityReport`` in
:mod:`app.core.capability_report` is a different, richer class used as
the return type of ``Decoder.available()``; same name, different
module, different contract. Import paths disambiguate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

__all__ = [
    "CAPABILITY_NAMES",
    "ProbeReport",
    "DEFAULT_ENVIRONMENT_REPORT_PATH",
    "detect_all",
    "load_environment_report",
    "probe_cudaq",
    "probe_cudaq_qec",
    "probe_cupy",
    "probe_cuquantum",
    "probe_modelopt",
    "probe_onnxruntime_cuda",
    "probe_onnxruntime_tensorrt",
    "probe_tensorrt",
    "probe_torch_cuda",
]


# Canonical ordered list of probe names. Readers (compat matrix, risk
# register) iterate in this order so artefacts are byte-stable across
# runs.
CAPABILITY_NAMES: Final[tuple[str, ...]] = (
    "torch_cuda",
    "onnxruntime_cuda",
    "onnxruntime_tensorrt",
    "tensorrt",
    "cudaq",
    "cudaq_qec",
    "cuquantum",
    "cupy",
    "modelopt",
)

DEFAULT_ENVIRONMENT_REPORT_PATH: Final[Path] = Path(".decoderops/environment_report.json")


class ProbeReport(BaseModel):
    """Result of a single Tier 3 capability probe.

    Distinct from :class:`app.core.capability_report.CapabilityReport`
    (decoder-adapter return type). This is the probe-level record used
    by the runtime capability test matrix and the compatibility report.

    Attributes:
        name: One of :data:`CAPABILITY_NAMES`.
        available: Whether the capability is usable right now.
        version: Detected library/runtime version, or ``None`` when the
            probe couldn't import the module.
        reason: Precise explanation when ``available`` is False. Also
            populated on success (e.g. "torch imported, CUDA 13.0, 2
            devices"). Never ``None`` in practice, never generic.
        details: Free-form structured facts (providers list, device
            count, driver version, error class name) for downstream
            report consumers.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str
    available: bool
    version: str | None = None
    reason: str | None = None
    details: dict[str, str] = Field(default_factory=dict)


def load_environment_report(
    path: str | Path = DEFAULT_ENVIRONMENT_REPORT_PATH,
) -> dict:
    """Parse the bootstrap-produced environment report.

    Returns an empty dict (with a WARNING log) when the file is absent
    or malformed. Never raises — capability probes must be robust to a
    missing report because the detector is invoked even on
    pre-bootstrap test environments.
    """
    p = Path(path)
    if not p.is_file():
        logger.warning("environment report not found at %s", p)
        return {}
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, dict):
                logger.warning(
                    "environment report at %s is not a JSON object", p
                )
                return {}
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to read environment report at %s: %s", p, exc)
        return {}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _gpu_device_count_from_env(env: dict) -> int:
    """Extract GPU device count from the environment_report.json shape.

    The bootstrap verify script writes::

        {"gpu": {"devices": [{...}, ...], "driver_version": "..."}}

    Returns 0 when missing — probes use this as a gate for claiming
    GPU capability.
    """
    gpu = env.get("gpu")
    if not isinstance(gpu, dict):
        return 0
    devices = gpu.get("devices", [])
    return len(devices) if isinstance(devices, list) else 0


def _tier3_package_from_env(env: dict, pkg_name: str) -> dict:
    """Pull a Tier 3 package entry from the environment_report.json.

    Shape::

        {"tier3": {"torch_cuda": {"status": "ok", "version": "...", ...}, ...}}
    """
    tier3 = env.get("tier3")
    if not isinstance(tier3, dict):
        return {}
    entry = tier3.get(pkg_name)
    return entry if isinstance(entry, dict) else {}


# --------------------------------------------------------------------------- #
# Probes — one per Tier 3 capability
# --------------------------------------------------------------------------- #


def probe_torch_cuda(env: dict) -> ProbeReport:
    try:
        import torch
    except ImportError as exc:
        return ProbeReport(
            name="torch_cuda",
            available=False,
            version=None,
            reason=f"torch not importable: {exc.name or exc}",
            details={"error_class": type(exc).__name__},
        )
    version = str(getattr(torch, "__version__", "unknown"))
    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception as exc:  # pragma: no cover - defensive; torch.cuda errors are rare
        return ProbeReport(
            name="torch_cuda",
            available=False,
            version=version,
            reason=f"torch.cuda query failed: {type(exc).__name__}: {exc}",
            details={"error_class": type(exc).__name__},
        )
    if not cuda_available:
        reason = "torch.cuda.is_available() is False"
    elif device_count == 0:
        reason = "environment_report.json reports 0 CUDA devices"
    elif _gpu_device_count_from_env(env) == 0 and env:
        # env was supplied but records no GPUs — disagreement is an
        # actionable signal, not a fatal error; surface it in reason.
        reason = (
            f"torch reports {device_count} device(s) but environment_report.json "
            "records 0; host state may have changed since bootstrap"
        )
    else:
        reason = f"torch {version} imported, {device_count} CUDA device(s) visible"
    return ProbeReport(
        name="torch_cuda",
        available=bool(cuda_available and device_count > 0),
        version=version,
        reason=reason,
        details={
            "cuda_runtime_version": str(getattr(torch.version, "cuda", "")),
            "device_count": str(device_count),
        },
    )


def _probe_onnxruntime_provider(
    env: dict, *, probe_name: str, provider: str
) -> ProbeReport:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return ProbeReport(
            name=probe_name,
            available=False,
            version=None,
            reason=f"onnxruntime not importable: {exc.name or exc}",
            details={"error_class": type(exc).__name__},
        )
    version = str(getattr(ort, "__version__", "unknown"))
    try:
        providers = list(ort.get_available_providers())
    except Exception as exc:
        return ProbeReport(
            name=probe_name,
            available=False,
            version=version,
            reason=f"ort.get_available_providers() failed: {exc}",
            details={"error_class": type(exc).__name__},
        )
    if provider not in providers:
        return ProbeReport(
            name=probe_name,
            available=False,
            version=version,
            reason=(
                f"{provider} not registered in onnxruntime available providers: "
                f"{providers}"
            ),
            details={"providers": ",".join(providers)},
        )
    return ProbeReport(
        name=probe_name,
        available=True,
        version=version,
        reason=f"onnxruntime {version} with {provider} registered",
        details={"providers": ",".join(providers)},
    )


def probe_onnxruntime_cuda(env: dict) -> ProbeReport:
    return _probe_onnxruntime_provider(
        env, probe_name="onnxruntime_cuda", provider="CUDAExecutionProvider"
    )


def probe_onnxruntime_tensorrt(env: dict) -> ProbeReport:
    return _probe_onnxruntime_provider(
        env,
        probe_name="onnxruntime_tensorrt",
        provider="TensorrtExecutionProvider",
    )


def probe_tensorrt(env: dict) -> ProbeReport:
    try:
        import tensorrt as trt
    except ImportError as exc:
        return ProbeReport(
            name="tensorrt",
            available=False,
            version=None,
            reason=f"tensorrt python package not installed: {exc.name or exc}",
            details={"error_class": type(exc).__name__},
        )
    version = str(getattr(trt, "__version__", "unknown"))
    return ProbeReport(
        name="tensorrt",
        available=True,
        version=version,
        reason=f"tensorrt {version} importable",
    )


def _simple_import_probe(name: str, module: str) -> ProbeReport:
    try:
        mod = __import__(module)
    except ImportError as exc:
        return ProbeReport(
            name=name,
            available=False,
            version=None,
            reason=f"{module} not importable: {exc.name or exc}",
            details={"error_class": type(exc).__name__},
        )
    version = str(getattr(mod, "__version__", "unknown"))
    return ProbeReport(
        name=name,
        available=True,
        version=version,
        reason=f"{module} {version} importable",
    )


def probe_cudaq(env: dict) -> ProbeReport:
    return _simple_import_probe("cudaq", "cudaq")


def probe_cudaq_qec(env: dict) -> ProbeReport:
    # Upstream sometimes exposes this as cudaq_qec and sometimes as
    # cudaq.qec; try both before declaring unavailable.
    report = _simple_import_probe("cudaq_qec", "cudaq_qec")
    if report.available:
        return report
    # Second chance via cudaq.qec
    try:
        import cudaq.qec as _qec  # type: ignore[import-not-found]

        return ProbeReport(
            name="cudaq_qec",
            available=True,
            version=str(getattr(_qec, "__version__", "unknown")),
            reason="cudaq.qec importable",
        )
    except ImportError as exc:
        return ProbeReport(
            name="cudaq_qec",
            available=False,
            version=None,
            reason=(
                f"neither cudaq_qec nor cudaq.qec importable: "
                f"{exc.name or exc}"
            ),
            details={"error_class": type(exc).__name__},
        )


def probe_cuquantum(env: dict) -> ProbeReport:
    return _simple_import_probe("cuquantum", "cuquantum")


def probe_cupy(env: dict) -> ProbeReport:
    return _simple_import_probe("cupy", "cupy")


def probe_modelopt(env: dict) -> ProbeReport:
    return _simple_import_probe("modelopt", "modelopt")


# --------------------------------------------------------------------------- #
# detect_all
# --------------------------------------------------------------------------- #


_PROBE_DISPATCH = {
    "torch_cuda": probe_torch_cuda,
    "onnxruntime_cuda": probe_onnxruntime_cuda,
    "onnxruntime_tensorrt": probe_onnxruntime_tensorrt,
    "tensorrt": probe_tensorrt,
    "cudaq": probe_cudaq,
    "cudaq_qec": probe_cudaq_qec,
    "cuquantum": probe_cuquantum,
    "cupy": probe_cupy,
    "modelopt": probe_modelopt,
}


def detect_all(
    env_report_path: str | Path = DEFAULT_ENVIRONMENT_REPORT_PATH,
) -> dict[str, CapabilityReport]:
    """Run every capability probe and return a ``{name: report}`` mapping.

    Never raises. Silently records missing or malformed environment
    reports via :func:`load_environment_report`. The returned mapping's
    key order matches :data:`CAPABILITY_NAMES` for artefact stability.
    """
    env = load_environment_report(env_report_path)
    results: dict[str, CapabilityReport] = {}
    for name in CAPABILITY_NAMES:
        probe = _PROBE_DISPATCH[name]
        try:
            results[name] = probe(env)
        except Exception as exc:  # pragma: no cover - probes must not raise
            # Last-resort wrapper: if a probe somehow raises despite
            # internal catching, record the failure rather than
            # propagating. A runaway exception would break the compat
            # matrix renderer.
            results[name] = CapabilityReport(
                name=name,
                available=False,
                version=None,
                reason=f"probe raised unexpectedly: {type(exc).__name__}: {exc}",
                details={"error_class": type(exc).__name__},
            )
    return results
