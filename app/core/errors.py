"""Typed exception hierarchy for the DecoderOps product.

This module is a dependency-free leaf: it must not import from any other
``app/`` module. The exceptions defined here carry structured context
(``reason_code`` + ``details``) so failures can be serialised deterministically
into run manifests, API responses, the compatibility matrix, and the
deployment-readiness report's Risk Register without leaking framework
internals or raw tracebacks.

The hierarchy is intentionally shallow: one root and four domain subclasses.
Sub-cases (e.g. a schema-validation failure within ingestion) are expressed
by passing a more specific ``reason_code`` and ``details``, not by adding
new exception classes. This keeps the error surface small and stable, which
matters because report generators and API boundary mappers switch on
``reason_code`` strings rather than Python types.

Design trade-off rejected: a registry mapping ``reason_code`` -> class. Only
one implementation exists per subclass and YAGNI applies; a plain class-level
``default_reason_code`` is simpler and visible in the signature.
"""

from __future__ import annotations

from typing import Any


class DecoderOpsError(Exception):
    """Root exception for all DecoderOps domain failures.

    Every DecoderOps exception carries a human-readable ``message`` and an
    optional ``details`` dict. The ``reason_code`` is a dotted string
    (``<domain>.<reason>``) that report generators and API boundary mappers
    switch on; it defaults to the class-level ``default_reason_code`` but
    callers may pass a more specific code (e.g. ``"ingestion.invalid_schema"``)
    without subclassing.

    Args:
        message: Human-readable description of what failed.
        reason_code: Dotted machine-readable code. Defaults to the class's
            ``default_reason_code``.
        details: Optional structured context for downstream consumers (Risk
            Register rows, compatibility matrix cells, API error payloads).
            Must be JSON-serialisable for deterministic manifest writing;
            this is the caller's responsibility — no validation is performed
            here to keep this module a leaf.

    Attributes:
        message: The message as passed to the constructor.
        reason_code: The effective reason code for this instance.
        details: The details dict, or ``None`` if not provided.
    """

    default_reason_code: str = "decoderops.error"

    def __init__(
        self,
        message: str,
        *,
        reason_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.reason_code = reason_code if reason_code is not None else self.default_reason_code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        """Serialise the exception into a deterministic dict.

        Returns:
            A dict with keys ``type``, ``reason_code``, ``message``, and
            ``details``. Key order is fixed so serialised output is stable
            across runs for byte-reproducible reports. ``details`` is
            ``None`` when no details were supplied — callers that need an
            empty dict should normalise at their boundary.
        """
        return {
            "type": type(self).__name__,
            "reason_code": self.reason_code,
            "message": self.message,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"reason_code={self.reason_code!r}, "
            f"details={self.details!r})"
        )


class IngestionError(DecoderOpsError):
    """Ingestion-layer failure.

    Raised when a Stim circuit, DEM, syndrome array, Sinter shot log, or
    customer-provided bundle cannot be normalised into the internal schema.
    Use ``details`` to carry ``{"source": "<path-or-stream>", "schema_version":
    "..."}`` for the Risk Register.
    """

    default_reason_code: str = "ingestion.invalid_schema"

    def __repr__(self) -> str:
        return super().__repr__()


class DecoderError(DecoderOpsError):
    """Decoder execution-layer failure.

    Raised when a decoder backend is missing, fails to warm up, or aborts
    mid-batch. Use ``details`` to carry ``{"backend": "...", "stage":
    "warmup|decode_batch"}`` so the compatibility matrix can attribute the
    failure to a specific backend.
    """

    default_reason_code: str = "decoder.unavailable"

    def __repr__(self) -> str:
        return super().__repr__()


class PackagingError(DecoderOpsError):
    """Artefact-packaging failure.

    Raised when manifest/tarball construction detects a SHA256 mismatch,
    missing artefact, or non-reproducible byte output. Use ``details`` to
    carry ``{"artefact": "...", "expected_sha256": "...", "observed_sha256":
    "..."}``.
    """

    default_reason_code: str = "packaging.sha_mismatch"

    def __repr__(self) -> str:
        return super().__repr__()


class CapabilityError(DecoderOpsError):
    """Runtime-capability failure.

    Raised when a Tier 3 runtime (TensorRT, cudaq, cudaq-qec, cuQuantum,
    nvidia-modelopt) is missing, import-guarded off, or unusable on the
    current host. Use ``details`` to carry ``{"runtime": "...", "requirement":
    "...", "blocker_kind": "machine|software|licensing|runtime"}`` so the
    deployment-readiness report can emit a precise blocker row rather than a
    silent downgrade.
    """

    default_reason_code: str = "capability.missing_runtime"

    def __repr__(self) -> str:
        return super().__repr__()
