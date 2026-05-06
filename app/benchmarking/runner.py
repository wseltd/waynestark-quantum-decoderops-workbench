"""Benchmark runner — execute one RunConfig against one Decoder (T034).

Given a concrete :class:`app.benchmarking.orchestrator.RunConfig` and a
``decoder_factory`` that produces a :class:`app.decoders.protocol.Decoder`
for a given backend name, :func:`run_single` drives a full end-to-end
benchmark pass:

    1. Build the decoder via the injected factory.
    2. Call ``decoder.available()``. If unavailable → structured
       RunResult with the precise capability reason; never raises.
    3. Call ``decoder.warmup()``. Exceptions are captured into the
       RunResult rather than propagated — the upstream parallel pool
       (T035) needs a structured failure row either way.
    4. Sample deterministic syndrome batches from a Stim DEM using a
       numpy Generator seeded from
       ``derive_worker_seed(master_seed, worker_seed_slot)``.
    5. Measure ONLY the inference call (``decode_batch``) with
       ``time.perf_counter_ns()``. Sampling time is excluded.
    6. Collect per-batch latencies and fold all predictions into a
       single SHA256 digest for downstream aggregation checks.

Restraint — deliberately out of scope here:
    * No metric aggregation (no LER, no p95, no histograms).
    * No file I/O, no manifest write.
    * No process spawning.
    * No direct tensorrt / cudaq / cudaq-qec imports; decoders own
      their own capability guards (T027–T031).

Design decisions worth stating:
    * The DEM is passed in by the caller, not derived from RunConfig
      here. Circuit construction / DEM extraction is the ingestion
      layer's job; the runner takes the already-built DEM so unit
      tests never need Stim beyond the deterministic sampler.
    * Syndrome sampling uses ``stim.Circuit.compile_detector_sampler``
      when the caller provides a circuit, OR ``numpy.Generator`` with
      an explicit seed for synthetic/fake DEMs in tests. The generic
      helper :func:`generate_syndromes` takes the DEM's detector count,
      a shot count, and a seeded Generator; this keeps unit tests
      free of Stim.
    * ``decoder_factory: Callable[[str], Decoder]`` — not a registry
      module import — so tests substitute fakes without touching the
      real Ising checkpoints (which would pull torch and a ~300 MB
      weights load).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from app.benchmarking.orchestrator import RunConfig
from app.core.seeding import derive_worker_seed
from app.decoders.protocol import Corrections, Decoder

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "RunResult",
    "generate_syndromes",
    "run_single",
]


# Default batch size for decode_batch calls. Chosen so a single batch
# fits comfortably in GPU memory for the Ising models (~8 MB of
# syndromes at d=5 r=5 = 120 detectors × 1024 shots × 1 byte), and so
# per-batch latencies cover multiple millisecond ticks for stable
# timing. Callers can override via run_single(..., batch_size=...).
DEFAULT_BATCH_SIZE: int = 1024


@dataclass(frozen=True)
class RunResult:
    """Structured outcome of a single benchmark run.

    One RunResult per RunConfig. Success and failure both produce a
    RunResult; a failed run sets ``error`` to a non-empty string and
    leaves metric-bearing fields in a well-defined empty state
    (``shots_total=0``, ``per_batch_latency_ns=[]``, etc.) so the
    parallel-pool layer (T035+) can collect uniform rows.

    Attributes:
        run_id: Copied from the input RunConfig (16-hex prefix).
        config: The input RunConfig verbatim.
        shots_total: Total shots successfully decoded. Zero on failure
            or on a zero-shot config.
        batches: Number of ``decode_batch`` calls that returned.
        per_batch_latency_ns: One entry per successful decode_batch;
            value is the ``Corrections.latency_ns`` reported by the
            decoder (measured inside the backend around inference
            only, per the Decoder protocol). The runner does NOT
            substitute its own wall-clock reading here — that would
            double-count framework overhead.
        corrections_digest: ``sha256`` hex digest over the concatenated
            predictions (uint8, C-contiguous) across all batches. Empty
            runs (zero shots) produce the digest of empty input, a
            stable fixed value.
        decoder_metadata: ``DecoderMetadata.model_dump(mode='json')``
            — a JSON-safe dict, suitable for direct embedding into
            manifests.
        started_at: Unix epoch seconds (float) at the start of
            ``run_single`` BEFORE any decoder work.
        finished_at: Unix epoch seconds (float) at the end of
            ``run_single`` AFTER all work (or the failure point).
        error: ``None`` on success; a human-readable failure string
            otherwise. Unavailability (``decoder.available()`` returned
            unavailable) fills this with the capability reason.
    """

    run_id: str
    config: RunConfig
    shots_total: int
    batches: int
    per_batch_latency_ns: list[int]
    corrections_digest: str
    decoder_metadata: dict[str, Any]
    started_at: float
    finished_at: float
    error: Optional[str]

    @property
    def ok(self) -> bool:
        return self.error is None


def generate_syndromes(
    *,
    num_detectors: int,
    shots: int,
    rng: np.random.Generator,
    sampler: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None,
) -> np.ndarray:
    """Produce a ``(shots, num_detectors)`` uint8 syndrome batch.

    Two modes:

        * Real Stim mode (production): caller passes a ``sampler``
          callable that invokes
          ``stim.Circuit.compile_detector_sampler`` internally and
          returns the detector events for the requested shot count.
          The callable receives ``(shots, rng)`` so it can seed
          ``compile_detector_sampler`` deterministically from the
          numpy Generator's internal state.
        * Synthetic mode (tests): when ``sampler`` is ``None``, draws
          bits uniformly from ``rng.integers(0, 2, ...)``. This is NOT
          a valid QEC syndrome distribution — it is test noise for
          exercising the orchestration plumbing without Stim in the
          loop. Decoder accuracy on synthetic input is undefined and
          must not be interpreted as a performance signal.

    Args:
        num_detectors: Second-axis width. Must be positive.
        shots: First-axis length. Must be non-negative. ``0`` returns
            a ``(0, num_detectors)`` array — a legal degenerate case
            for runs that the scheduler decided to skip.
        rng: Seeded numpy Generator. Used directly in synthetic mode;
            passed through to the ``sampler`` callable in Stim mode so
            real runs remain deterministic per
            ``(master_seed, worker_seed_slot)``.
        sampler: Optional callable that produces detector events. When
            supplied, synthetic sampling is bypassed.

    Returns:
        ``(shots, num_detectors)`` uint8 ndarray. Always C-contiguous
        so downstream hashing of the bytes is stable across numpy
        versions.
    """
    if not isinstance(num_detectors, int) or num_detectors <= 0:
        raise ValueError(
            f"num_detectors must be a positive int; got {num_detectors!r}"
        )
    if not isinstance(shots, int) or shots < 0:
        raise ValueError(f"shots must be a non-negative int; got {shots!r}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be a numpy.random.Generator; "
            f"got {type(rng).__name__}"
        )

    if sampler is not None:
        out = sampler(shots, rng)
        if not isinstance(out, np.ndarray):
            raise TypeError(
                "sampler must return numpy.ndarray; "
                f"got {type(out).__name__}"
            )
        if out.shape != (shots, num_detectors):
            raise ValueError(
                "sampler returned wrong shape: expected "
                f"{(shots, num_detectors)}, got {out.shape}"
            )
        if out.dtype != np.uint8:
            out = out.astype(np.uint8, copy=False)
        return np.ascontiguousarray(out)

    # Synthetic path — uniform random bits. Only valid for tests.
    buf = rng.integers(0, 2, size=(shots, num_detectors), dtype=np.uint8)
    return np.ascontiguousarray(buf)


def _chunked_shot_ranges(total_shots: int, batch_size: int) -> list[int]:
    """Split ``total_shots`` into a list of batch sizes summing to total.

    Produces ``[batch_size, batch_size, ..., remainder]``. Returns an
    empty list when ``total_shots == 0``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    if total_shots == 0:
        return []
    full = total_shots // batch_size
    rem = total_shots - full * batch_size
    sizes = [batch_size] * full
    if rem:
        sizes.append(rem)
    return sizes


def run_single(
    config: RunConfig,
    *,
    decoder_factory: Callable[[str], Decoder],
    num_detectors: int,
    sampler: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RunResult:
    """Execute a single RunConfig end-to-end and return a RunResult.

    The call never raises on decoder-induced failure — every failure
    is captured into :attr:`RunResult.error`. It does raise on argument
    bugs (wrong types) because those are caller-side programmer errors,
    not runtime decoder issues.

    Args:
        config: The resolved :class:`RunConfig` (from T033).
        decoder_factory: Callable taking ``config.backend`` and
            returning a :class:`Decoder`. Injected rather than reaching
            into ``app.decoders.registry`` directly so tests don't need
            real weights on disk. Production callers pass a closure
            that binds their :class:`DecoderConfig` and delegates to
            ``app.decoders.registry.get_decoder``.
        num_detectors: Detector count of the underlying DEM. Runner-
            level knowledge because the synthetic sampler needs it and
            the real sampler must agree with it (shape-checked inside
            :func:`generate_syndromes`).
        sampler: Optional Stim-backed sampler callable; see
            :func:`generate_syndromes`.
        batch_size: Per-decode_batch shot count. Defaults to
            :data:`DEFAULT_BATCH_SIZE`.

    Returns:
        A :class:`RunResult` with timing, digest, metadata, and either
        ``error=None`` on success or a populated error string.
    """
    if not isinstance(config, RunConfig):
        raise TypeError(
            f"config must be RunConfig; got {type(config).__name__}"
        )
    if not callable(decoder_factory):
        raise TypeError("decoder_factory must be callable")

    started_at = time.time()
    seed = derive_worker_seed(config.master_seed, config.worker_seed_slot)
    rng = np.random.default_rng(seed)

    # Defaults that describe a failed / empty run; overwritten on
    # success.
    per_batch_latency_ns: list[int] = []
    batches = 0
    shots_total = 0
    digest_accum = hashlib.sha256()
    decoder_metadata: dict[str, Any] = {}
    error: Optional[str] = None

    try:
        decoder = decoder_factory(config.backend)
    except Exception as exc:
        finished_at = time.time()
        return RunResult(
            run_id=config.run_id,
            config=config,
            shots_total=0,
            batches=0,
            per_batch_latency_ns=[],
            corrections_digest=digest_accum.hexdigest(),
            decoder_metadata={},
            started_at=started_at,
            finished_at=finished_at,
            error=f"decoder_factory failed: {type(exc).__name__}: {exc}",
        )

    report = decoder.available()
    if not report.available:
        finished_at = time.time()
        return RunResult(
            run_id=config.run_id,
            config=config,
            shots_total=0,
            batches=0,
            per_batch_latency_ns=[],
            corrections_digest=digest_accum.hexdigest(),
            decoder_metadata={},
            started_at=started_at,
            finished_at=finished_at,
            error=(
                f"decoder '{config.backend}' unavailable: {report.reason} "
                f"(category={report.blocker_category})"
            ),
        )

    try:
        decoder.warmup()
        decoder_metadata = decoder.metadata().model_dump(mode="json")
    except Exception as exc:
        finished_at = time.time()
        return RunResult(
            run_id=config.run_id,
            config=config,
            shots_total=0,
            batches=0,
            per_batch_latency_ns=[],
            corrections_digest=digest_accum.hexdigest(),
            decoder_metadata=decoder_metadata,
            started_at=started_at,
            finished_at=finished_at,
            error=f"warmup failed: {type(exc).__name__}: {exc}",
        )

    try:
        for this_batch in _chunked_shot_ranges(config.num_shots, batch_size):
            syndromes = generate_syndromes(
                num_detectors=num_detectors,
                shots=this_batch,
                rng=rng,
                sampler=sampler,
            )
            corr: Corrections = decoder.decode_batch(syndromes)
            per_batch_latency_ns.append(int(corr.latency_ns))
            batches += 1
            shots_total += this_batch
            # Fold predictions into the digest; ascontiguousarray keeps
            # the byte layout stable across numpy backends.
            digest_accum.update(
                np.ascontiguousarray(corr.predictions).tobytes()
            )
    except Exception as exc:
        error = f"decode failed: {type(exc).__name__}: {exc}"

    finished_at = time.time()
    return RunResult(
        run_id=config.run_id,
        config=config,
        shots_total=shots_total,
        batches=batches,
        per_batch_latency_ns=per_batch_latency_ns,
        corrections_digest=digest_accum.hexdigest(),
        decoder_metadata=decoder_metadata,
        started_at=started_at,
        finished_at=finished_at,
        error=error,
    )
