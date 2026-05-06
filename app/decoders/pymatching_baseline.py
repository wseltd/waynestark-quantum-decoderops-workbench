"""PyMatching MWPM decoder backend — the reference baseline.

Every report in the product compares against this backend. It is
deliberately the most boring, predictable decoder in scope: pure-CPU,
no learned weights, no GPU paths, no soft decisions, no belief
propagation. Any deviation a neural pre-decoder shows in the reports
is measured against this.

The backend conforms structurally to :class:`app.decoders.protocol.Decoder`
(checked via ``isinstance(d, Decoder)`` in tests) without inheriting
from it — ``Decoder`` is a ``typing.Protocol``.

Construction contract:
    * ``dem``: a ``stim.DetectorErrorModel`` object, OR
    * ``dem_path``: path to a ``.dem`` file on disk (loaded via
      ``stim.DetectorErrorModel.from_file``).
    Exactly one must be provided. Neither is hardcoded anywhere.

``warmup()`` constructs the ``pymatching.Matching`` graph exactly once
and caches it. Subsequent calls are no-ops — the ``_warmed`` flag
guards against graph rebuild.

``decode_batch(syndromes)`` validates shape/dtype/detector-count before
dispatching to ``Matching.decode_batch``. Validation errors raise
:class:`DecoderInputError` with actual-vs-expected details so the call
site can diagnose quickly; a bare ``ValueError`` would lose that
signal.

Out of scope per T023:
    soft-decision / weighted belief propagation, GPU acceleration,
    accepting ``stim.Circuit`` directly (DEM only), persisting the
    matching graph to disk, asynchronous decoding, training/fitting,
    logging raw syndrome contents.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app.core.capability_report import CapabilityReport
from app.decoders.protocol import Corrections, DecoderMetadata

if TYPE_CHECKING:
    import stim


__all__ = ["DecoderInputError", "PyMatchingBaseline"]


class DecoderInputError(ValueError):
    """Raised when ``decode_batch`` receives a malformed input array.

    Carries structured ``expected`` and ``actual`` attributes so callers
    can format error messages or log structured events without
    re-parsing the ``ValueError``'s ``str()``. Inherits ``ValueError``
    so existing ``except ValueError:`` blocks still catch it.
    """

    def __init__(self, *, message: str, expected: object, actual: object) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"{message} (expected {expected!r}, actual {actual!r})")


class PyMatchingBaseline:
    """PyMatching MWPM reference backend.

    Args:
        dem: ``stim.DetectorErrorModel`` object; mutually exclusive with
            ``dem_path``.
        dem_path: Path to a ``.dem`` file on disk; loaded lazily on
            construction. Mutually exclusive with ``dem``.
        num_threads: Number of CPU threads for PyMatching. Defaults to
            1 — single-threaded determinism over raw speed for the
            reference backend; callers that want parallel throughput
            should batch shots across workers externally.

    Raises:
        ValueError: If neither or both of ``dem`` / ``dem_path`` are
            supplied, or if ``dem_path`` does not resolve to a file.
    """

    def __init__(
        self,
        *,
        dem: "stim.DetectorErrorModel | None" = None,
        dem_path: Path | str | None = None,
        num_threads: int = 1,
    ) -> None:
        if (dem is None) == (dem_path is None):
            raise ValueError(
                "PyMatchingBaseline requires exactly one of `dem` or "
                "`dem_path`; got both or neither."
            )
        if dem is not None:
            self._dem = dem
        else:
            resolved = Path(dem_path).resolve()  # type: ignore[arg-type]
            if not resolved.is_file():
                raise ValueError(
                    f"dem_path {resolved} does not resolve to a file"
                )
            # stim import is deferred to construction so the module can
            # be imported cheaply by tests that don't need stim loaded.
            import stim as _stim

            self._dem = _stim.DetectorErrorModel.from_file(str(resolved))

        self._num_threads = int(num_threads)
        self._warmed: bool = False
        self._matching: object | None = None
        self._num_detectors: int = int(self._dem.num_detectors)
        self._num_observables: int = int(self._dem.num_observables)

    # -- capability probe --------------------------------------------------

    def available(self) -> CapabilityReport:
        """Probe whether pymatching and stim are importable.

        Never raises. On import success returns a ready report; on
        ``ImportError`` returns an unavailable report naming the
        missing module.
        """
        start = time.perf_counter_ns()
        versions: dict[str, str] = {}
        try:
            import pymatching as _pm
            import stim as _stim
        except ImportError as exc:
            return CapabilityReport.unavailable(
                reason=f"required module not importable: {exc.name or exc}",
                required=["pymatching", "stim"],
                category="not_installed",
            )
        versions["pymatching"] = getattr(_pm, "__version__", "unknown")
        versions["stim"] = getattr(_stim, "__version__", "unknown")
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return CapabilityReport.ready(
            reason="pymatching and stim importable",
            required=["pymatching", "stim"],
            detected_versions=versions,
            probe_latency_ms=elapsed_ms,
        )

    # -- warmup ------------------------------------------------------------

    def warmup(self) -> None:
        """Build the PyMatching graph from the DEM. Idempotent."""
        if self._warmed:
            return
        import pymatching

        self._matching = pymatching.Matching.from_detector_error_model(self._dem)
        self._warmed = True

    # -- decode ------------------------------------------------------------

    def decode_batch(self, syndromes: np.ndarray) -> Corrections:
        """Vectorised MWPM decode over a batch of shots.

        Args:
            syndromes: ``(batch, detectors)`` uint8 array.

        Returns:
            :class:`Corrections` with ``(batch, observables)`` uint8
            predictions and wall-clock nanoseconds in ``latency_ns``.

        Raises:
            DecoderInputError: On any shape or dtype mismatch. Actual
                vs expected values are attached for diagnosis.
        """
        if not isinstance(syndromes, np.ndarray):
            raise DecoderInputError(
                message="syndromes must be numpy.ndarray",
                expected="numpy.ndarray",
                actual=type(syndromes).__name__,
            )
        if syndromes.ndim != 2:
            raise DecoderInputError(
                message="syndromes must be 2D (batch, detectors)",
                expected="ndim == 2",
                actual=f"ndim == {syndromes.ndim}",
            )
        if syndromes.dtype != np.uint8:
            raise DecoderInputError(
                message="syndromes must be uint8",
                expected="dtype == uint8",
                actual=str(syndromes.dtype),
            )
        if syndromes.shape[1] != self._num_detectors:
            raise DecoderInputError(
                message="detector count mismatch",
                expected=self._num_detectors,
                actual=syndromes.shape[1],
            )

        # Auto-warm on first decode — the ticket's AC allows either
        # "raise" or "autowarm" on pre-warmup decode; autowarm is the
        # friendlier contract for benchmark runner code paths.
        if not self._warmed:
            self.warmup()

        assert self._matching is not None  # noqa: S101 - guarded by _warmed above

        start_ns = time.perf_counter_ns()
        predictions = self._matching.decode_batch(syndromes)  # type: ignore[attr-defined]
        latency_ns = time.perf_counter_ns() - start_ns

        # PyMatching returns int8/bool depending on version; coerce to
        # the Corrections contract (uint8) without copying when possible.
        if predictions.dtype != np.uint8:
            predictions = predictions.astype(np.uint8, copy=False)
        # Defensive: some pymatching versions return shape (batch,) when
        # observables == 1. Always reshape to 2D.
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return Corrections(predictions=predictions, latency_ns=int(latency_ns))

    # -- metadata ----------------------------------------------------------

    def metadata(self) -> DecoderMetadata:
        """Return serialisable metadata for manifest embedding."""
        import pymatching

        return DecoderMetadata(
            backend_name="pymatching_baseline",
            backend_version=getattr(pymatching, "__version__", "unknown"),
            model_path=None,
            model_sha256=None,
            receptive_field=None,
            supports_batching=True,
            supports_gpu=False,
            schema_version="1",
        )
