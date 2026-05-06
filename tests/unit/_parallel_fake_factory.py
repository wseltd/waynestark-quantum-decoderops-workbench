"""Picklable decoder factories for multiprocessing spawn pool tests.

Must live at module scope so the spawn worker can re-import it. Cannot
live inside tests/unit/test_parallel.py because that file's fixtures and
pytest rewrites are not round-trippable across pickling.
"""

from __future__ import annotations

import numpy as np

from app.core.capability_report import CapabilityReport
from app.decoders.protocol import Corrections, DecoderMetadata


class _FakeDecoder:
    def __init__(self, num_observables: int = 1) -> None:
        self._no = num_observables

    def available(self) -> CapabilityReport:
        return CapabilityReport.ready(
            reason="fake ready",
            required=["fake"],
            detected_versions={"fake": "0"},
        )

    def warmup(self) -> None:
        return None

    def decode_batch(self, syndromes: np.ndarray) -> Corrections:
        preds = np.zeros((syndromes.shape[0], self._no), dtype=np.uint8)
        return Corrections(predictions=preds, latency_ns=1_000)

    def metadata(self) -> DecoderMetadata:
        return DecoderMetadata(
            backend_name="fake_backend",
            backend_version="fake-0",
            model_path=None,
            model_sha256=None,
            receptive_field=None,
            supports_batching=True,
            supports_gpu=False,
            schema_version="1",
        )


def make_decoder(name: str) -> _FakeDecoder:  # noqa: ARG001
    return _FakeDecoder()


def make_raising_factory(name: str) -> _FakeDecoder:  # noqa: ARG001
    raise RuntimeError("factory refuses to build decoder for test")
