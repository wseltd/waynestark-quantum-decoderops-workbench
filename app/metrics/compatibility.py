"""Runtime-compatibility status per backend (T046)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

__all__ = ["RuntimeCompatibilityStatus", "build_compatibility_status"]


_KNOWN_BACKENDS = frozenset(
    {
        "pymatching_baseline",
        "ising_fast",
        "ising_accurate",
        "onnx_validation",
        "tensorrt_optional",
        "cudaq",
        "cudaq_qec",
        "cuquantum",
        "ort_cuda",
        "ort_tensorrt",
    }
)

_KNOWN_STATUS = frozenset({"ready", "degraded", "unavailable"})
_KNOWN_CATEGORY = frozenset(
    {"machine", "software", "licensing", "runtime", "none"}
)


class RuntimeCompatibilityStatus(BaseModel):
    backend: Literal[
        "pymatching_baseline",
        "ising_fast",
        "ising_accurate",
        "onnx_validation",
        "tensorrt_optional",
        "cudaq",
        "cudaq_qec",
        "cuquantum",
        "ort_cuda",
        "ort_tensorrt",
    ]
    status: Literal["ready", "degraded", "unavailable"]
    reason: str
    category: Literal["machine", "software", "licensing", "runtime", "none"]
    required_action: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "RuntimeCompatibilityStatus":
        reason_stripped = self.reason.strip()
        if self.status == "ready":
            if self.category != "none":
                raise ValueError(
                    "status='ready' requires category='none'; "
                    f"got category={self.category!r}"
                )
            if not reason_stripped:
                raise ValueError(
                    "status='ready' requires a non-empty reason"
                )
        else:
            if self.category == "none":
                raise ValueError(
                    f"status={self.status!r} forbids category='none'"
                )
            if not reason_stripped:
                raise ValueError(
                    f"status={self.status!r} requires a non-empty reason"
                )
        return self


def build_compatibility_status(
    backend: str,
    status: str,
    reason: str,
    category: str,
    required_action: str | None = None,
) -> RuntimeCompatibilityStatus:
    if backend not in _KNOWN_BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; allowed={sorted(_KNOWN_BACKENDS)}"
        )
    if status not in _KNOWN_STATUS:
        raise ValueError(
            f"unknown status {status!r}; allowed={sorted(_KNOWN_STATUS)}"
        )
    if category not in _KNOWN_CATEGORY:
        raise ValueError(
            f"unknown category {category!r}; allowed={sorted(_KNOWN_CATEGORY)}"
        )
    return RuntimeCompatibilityStatus(
        backend=backend,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        reason=reason,
        category=category,  # type: ignore[arg-type]
        required_action=required_action,
    )
