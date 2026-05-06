"""Decoder Protocol — the contract every decoder backend implements.

A decoder is anything that maps syndrome arrays to correction arrays.
We deliberately use ``typing.Protocol`` (structural typing) rather than
an ABC so third-party backends can conform without inheriting our class
tree. Runtime-checkable so code paths that need a boolean check
(e.g. test suites asserting structural conformance) can use
``isinstance(obj, Decoder)``.

Four required methods:

    available() -> CapabilityReport
        Cheap, import-guarded probe. Returns the CapabilityReport
        schema from ``app.core.capability_report`` (re-exported here so
        decoder-layer call sites don't need to reach across packages).
        Must NEVER raise — unavailability is a return value, not an
        exception. The ``reason`` string is required in both available
        and unavailable states and must be non-empty.

    warmup() -> None
        Load the model, allocate buffers, JIT-compile, verify
        checksums. Must be idempotent — calling it twice has the same
        observable effect as calling it once. Heavyweight imports
        (torch, tensorrt) live inside this method, not at module top.

    decode_batch(syndromes: np.ndarray) -> Corrections
        Vectorised forward pass. ``syndromes`` is ``(batch, detectors)``
        uint8. Returns ``Corrections`` wrapping ``(batch, observables)``
        uint8 plus the wall-clock latency in nanoseconds measured with
        ``time.perf_counter_ns()`` around the inference call.

    metadata() -> DecoderMetadata
        Serialisable description of the backend — frozen, JSON-safe,
        schema-versioned. Embedded verbatim into every run manifest and
        every report artefact. Changing its shape is a contract break.

Design choices:
    * ``CapabilityReport`` is imported from ``app.core.capability_report``
      and re-exported here. Redefining it would create schema drift
      across the 161 tests in the core layer that already pin the
      7-category ``blocker_category`` literal. The ticket's design
      sketch listed a 5-category variant; core's 7-category is
      strictly richer and wins. Re-export keeps the name available in
      this module for imports like ``from app.decoders.protocol import
      CapabilityReport``.
    * ``Corrections`` is a frozen ``@dataclass`` rather than a pydantic
      model because ``np.ndarray`` is its primary payload; pydantic v2
      requires ``arbitrary_types_allowed`` and custom validators to
      handle numpy, adding weight for no boundary-validation benefit on
      a throw-away per-batch result object.
    * ``DecoderMetadata`` IS a pydantic BaseModel because it travels
      into run manifests and must JSON-roundtrip deterministically with
      schema-version tagging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.core.capability_report import CapabilityReport as CapabilityReport

__all__ = [
    "CapabilityReport",
    "Corrections",
    "Decoder",
    "DecoderMetadata",
]


@dataclass(frozen=True)
class Corrections:
    """Per-batch decoder output.

    Attributes:
        predictions: ``(batch, observables)`` uint8 array of predicted
            logical-observable bit values. Callers compare these
            against the ground-truth observables from the simulator to
            compute the logical error rate.
        latency_ns: Wall-clock nanoseconds spent inside the backend's
            inference call, measured with ``time.perf_counter_ns()``
            around the vectorised decode step. Must be a non-negative
            integer.

    Frozen so callers cannot mutate the predictions array reference
    after return (the underlying numpy buffer is still mutable — the
    freeze guards the wrapper, not the data; the Decoder contract asks
    callers not to write into the returned array).
    """

    predictions: np.ndarray
    latency_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.predictions, np.ndarray):
            raise TypeError(
                f"Corrections.predictions must be numpy.ndarray, "
                f"got {type(self.predictions).__name__}"
            )
        if self.predictions.ndim != 2:
            raise ValueError(
                f"Corrections.predictions must be 2D (batch, observables); "
                f"got shape {self.predictions.shape}"
            )
        if self.predictions.dtype != np.uint8:
            raise TypeError(
                f"Corrections.predictions must be uint8; "
                f"got dtype {self.predictions.dtype}"
            )
        if not isinstance(self.latency_ns, int) or self.latency_ns < 0:
            raise ValueError(
                f"Corrections.latency_ns must be a non-negative int; "
                f"got {self.latency_ns!r}"
            )

    @property
    def batch_size(self) -> int:
        return int(self.predictions.shape[0])

    @property
    def num_observables(self) -> int:
        return int(self.predictions.shape[1])


class DecoderMetadata(BaseModel):
    """JSON-serialisable description of a decoder backend.

    Embedded in run manifests, report artefacts, and the compatibility
    matrix. Frozen and schema-version-tagged so on-disk manifests stay
    readable across minor releases; bumping ``schema_version`` away
    from ``'1'`` is an explicit contract break.

    Attributes:
        backend_name: Stable identifier used across the codebase
            (e.g. ``'pymatching_baseline'``, ``'ising_fast'``).
        backend_version: Runtime-observed version of the underlying
            library (e.g. ``pymatching.__version__``).
        model_path: Filesystem path to the loaded model artefact, or
            ``None`` for backends that do not load a model file
            (e.g. PyMatching).
        model_sha256: Hex digest of the loaded model file, recorded at
            warmup for integrity tracking. ``None`` when ``model_path``
            is ``None``.
        receptive_field: Spatial/temporal receptive field in stabiliser
            rounds for neural pre-decoders. ``None`` for non-neural
            backends.
        supports_batching: Whether ``decode_batch`` is meaningfully
            vectorised versus a per-shot loop under the hood.
        supports_gpu: Whether this backend can run on a CUDA device.
        schema_version: Literal ``'1'``. This is the single knob the
            downstream readers pin against; changing it must be
            accompanied by a migration path for stored manifests.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    backend_name: str = Field(..., min_length=1)
    backend_version: str = Field(..., min_length=1)
    model_path: str | None
    model_sha256: str | None
    receptive_field: int | None
    supports_batching: bool
    supports_gpu: bool
    schema_version: Literal["1"]


@runtime_checkable
class Decoder(Protocol):
    """Structural contract for every QEC decoder backend.

    A class conforms to ``Decoder`` if it exposes these four methods
    with compatible signatures. We DO NOT require inheritance; third-
    party decoders can register via duck typing. The benchmark runner
    (see ``app.benchmarking``) treats every backend through this
    protocol, never through its concrete class.

    Methods are non-async on purpose: real-time decoding is out of
    scope for v1 (see EXCLUSIONS), and the benchmark runner manages
    parallelism at the shot-batch level, not inside the decoder.
    """

    def available(self) -> CapabilityReport:
        """Probe whether this backend is usable in the current environment."""
        ...

    def warmup(self) -> None:
        """Load the model and allocate any per-instance state. Idempotent."""
        ...

    def decode_batch(self, syndromes: np.ndarray) -> Corrections:
        """Vectorised decode over a batch of shots. See module docstring."""
        ...

    def metadata(self) -> DecoderMetadata:
        """Serialisable description of this backend."""
        ...
