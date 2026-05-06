"""Decoder registry — single authoritative lookup for the five backends.

Imports the five concrete decoder classes from their modules (T023
pymatching_baseline, T024 ising_fast, T025 ising_accurate, T026
onnx_validation, T027 tensorrt_adapter) and returns per-call fresh
instances bound to the caller's config. No module-level instance
caching — every ``get_decoder()`` call constructs anew so different
DEMs / model paths / precisions don't accidentally share state.

Capability adapters (T028 cudaq, T029 cudaq_qec, T030 cuquantum) are
NOT registered here — they are capability reporters, not decoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Final

from app.core.capability_report import CapabilityReport
from app.decoders.ising_accurate import IsingAccurateDecoder
from app.decoders.ising_fast import IsingFastDecoder
from app.decoders.onnx_validation import OnnxValidationDecoder
from app.decoders.protocol import Decoder
from app.decoders.pymatching_baseline import PyMatchingBaseline
from app.decoders.tensorrt_adapter import TensorRTDecoder

__all__ = [
    "BACKEND_NAMES",
    "DecoderAvailability",
    "DecoderConfig",
    "UnknownDecoderError",
    "available_decoders",
    "get_decoder",
    "list_decoders",
]


BACKEND_NAMES: Final[tuple[str, ...]] = (
    "pymatching_baseline",
    "ising_fast",
    "ising_accurate",
    "onnx_validation",
    "tensorrt_optional",
)


class UnknownDecoderError(ValueError):
    """Raised when ``get_decoder`` is called with an unrecognised name.

    The valid-names list is embedded in the message so operators don't
    have to re-read the registry source to find the right spelling.
    """

    def __init__(self, *, name: str, valid: tuple[str, ...]) -> None:
        self.name = name
        self.valid = valid
        super().__init__(
            f"unknown decoder {name!r}; valid names: {sorted(valid)}"
        )


@dataclass(frozen=True)
class DecoderConfig:
    """Per-call configuration passed to ``get_decoder``.

    Each backend only consumes the subset of keys it needs; registry
    passes the whole DecoderConfig down and concrete backends ignore
    what doesn't apply. Kwargs-only so the registry remains stable
    across backend-specific parameter additions.
    """

    # pymatching_baseline
    dem: Any | None = None
    dem_path: Any | None = None
    num_threads: int = 1
    # ising_fast / ising_accurate
    model_path: Any | None = None
    asset_manifest_path: Any | None = None
    device: str = "auto"
    # onnx_validation
    onnx_model_path: Any | None = None
    providers: list[str] | None = None
    # tensorrt_adapter
    engine_path: Any | None = None
    onnx_source_path: Any | None = None
    precision: str = "fp16"
    workspace_bytes: int = 1 << 30


@dataclass(frozen=True)
class DecoderAvailability:
    """Pair of (backend name, capability report)."""

    name: str
    report: CapabilityReport


def _build_pymatching_baseline(cfg: DecoderConfig) -> Decoder:
    kwargs: dict[str, Any] = {"num_threads": cfg.num_threads}
    if cfg.dem is not None and cfg.dem_path is not None:
        raise ValueError(
            "pymatching_baseline: provide exactly one of cfg.dem or cfg.dem_path"
        )
    if cfg.dem is not None:
        kwargs["dem"] = cfg.dem
    elif cfg.dem_path is not None:
        kwargs["dem_path"] = cfg.dem_path
    else:
        raise ValueError(
            "pymatching_baseline requires cfg.dem or cfg.dem_path"
        )
    return PyMatchingBaseline(**kwargs)


def _build_ising_fast(cfg: DecoderConfig) -> Decoder:
    if cfg.model_path is None or cfg.asset_manifest_path is None:
        raise ValueError(
            "ising_fast requires cfg.model_path and cfg.asset_manifest_path"
        )
    return IsingFastDecoder(
        model_path=cfg.model_path,
        asset_manifest_path=cfg.asset_manifest_path,
        device=cfg.device,  # type: ignore[arg-type]
    )


def _build_ising_accurate(cfg: DecoderConfig) -> Decoder:
    if cfg.model_path is None or cfg.asset_manifest_path is None:
        raise ValueError(
            "ising_accurate requires cfg.model_path and cfg.asset_manifest_path"
        )
    return IsingAccurateDecoder(
        model_path=cfg.model_path,
        asset_manifest_path=cfg.asset_manifest_path,
        device=cfg.device,  # type: ignore[arg-type]
    )


def _build_onnx_validation(cfg: DecoderConfig) -> Decoder:
    if cfg.onnx_model_path is None:
        raise ValueError("onnx_validation requires cfg.onnx_model_path")
    return OnnxValidationDecoder(
        model_path=cfg.onnx_model_path,
        providers=cfg.providers,
    )


def _build_tensorrt_optional(cfg: DecoderConfig) -> Decoder:
    if cfg.engine_path is None:
        raise ValueError("tensorrt_optional requires cfg.engine_path")
    return TensorRTDecoder(
        engine_path=cfg.engine_path,
        onnx_path=cfg.onnx_source_path,
        precision=cfg.precision,  # type: ignore[arg-type]
        workspace_bytes=cfg.workspace_bytes,
    )


_BUILDERS: dict[str, Callable[[DecoderConfig], Decoder]] = {
    "pymatching_baseline": _build_pymatching_baseline,
    "ising_fast": _build_ising_fast,
    "ising_accurate": _build_ising_accurate,
    "onnx_validation": _build_onnx_validation,
    "tensorrt_optional": _build_tensorrt_optional,
}


def list_decoders() -> tuple[str, ...]:
    """Return the canonical ordered tuple of backend names."""
    return BACKEND_NAMES


def get_decoder(name: str, *, config: DecoderConfig | None = None) -> Decoder:
    """Construct a fresh backend instance keyed by name.

    Args:
        name: One of :data:`BACKEND_NAMES`.
        config: Per-call configuration. If omitted, builders raise on
            the backends that require it (e.g. pymatching_baseline
            needs a DEM; onnx_validation needs a model path). This is
            deliberate — we do NOT want a registry that silently
            constructs a broken decoder the caller then has to
            second-guess at ``warmup()`` time.

    Returns:
        A new :class:`Decoder` instance, never cached across calls.

    Raises:
        UnknownDecoderError: ``name`` is not a registered backend.
        ValueError: Required config field missing for the requested
            backend.
    """
    if name not in _BUILDERS:
        raise UnknownDecoderError(name=name, valid=BACKEND_NAMES)
    cfg = config if config is not None else DecoderConfig()
    return _BUILDERS[name](cfg)


def available_decoders(
    *, config: DecoderConfig | None = None
) -> list[DecoderAvailability]:
    """Probe every backend's ``available()`` and return results.

    Never constructs a real inference session or loads weights — only
    calls ``available()`` on a cheap constructor. Backends whose
    constructor requires config (pymatching needs DEM, ising needs
    paths, onnx needs model, tensorrt needs engine) report that
    missing-config as an availability blocker rather than raising —
    the caller gets a uniform list of availability reports for the
    compatibility matrix regardless of which configs are in scope.
    """
    cfg = config if config is not None else DecoderConfig()
    out: list[DecoderAvailability] = []
    for name in BACKEND_NAMES:
        try:
            decoder = _BUILDERS[name](cfg)
        except Exception as exc:  # constructor declined — map to unavailable
            report = CapabilityReport.unavailable(
                reason=(
                    f"{name} construction failed: {type(exc).__name__}: {exc}"
                ),
                required=["valid DecoderConfig for this backend"],
                category="software",
            )
            out.append(DecoderAvailability(name=name, report=report))
            continue
        try:
            report = decoder.available()
        except Exception as exc:  # pragma: no cover - Decoder.available must not raise
            report = CapabilityReport.unavailable(
                reason=(
                    f"{name}.available() raised unexpectedly: "
                    f"{type(exc).__name__}: {exc}"
                ),
                required=[name],
                category="runtime",
            )
        out.append(DecoderAvailability(name=name, report=report))
    return out
