"""Unit tests for app.benchmarking.runner (T034).

Uses a FakeDecoder that conforms structurally to
``app.decoders.protocol.Decoder`` so the runner can be exercised
end-to-end without pulling Stim, Torch, or any vendor weights.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pytest

from app.benchmarking.orchestrator import RunConfig
from app.benchmarking.runner import (
    DEFAULT_BATCH_SIZE,
    RunResult,
    generate_syndromes,
    run_single,
)
from app.core.capability_report import CapabilityReport
from app.decoders.protocol import Corrections, DecoderMetadata


def _config(
    *,
    run_id: str = "0" * 16,
    sweep_id: str = "s" * 16,
    num_shots: int = 2048,
    master_seed: int = 42,
    worker_seed_slot: int = 0,
    backend: str = "fake_backend",
) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        sweep_id=sweep_id,
        distance=3,
        rounds=3,
        noise={"p_error": 0.003, "model": "depolarize"},
        basis="X",
        backend=backend,
        model_variant="baseline",
        export_mode="native",
        worker_seed_slot=worker_seed_slot,
        master_seed=master_seed,
        num_shots=num_shots,
    )


# ---------------------------------------------------------------------------
# Fake decoder — conforms structurally to the Decoder protocol.
# ---------------------------------------------------------------------------


class FakeDecoder:
    """Protocol-conformant decoder for plumbing tests."""

    def __init__(
        self,
        *,
        num_detectors: int = 32,
        num_observables: int = 1,
        available_report: Optional[CapabilityReport] = None,
        warmup_raises: Optional[Exception] = None,
        decode_raises: Optional[Exception] = None,
        per_batch_latency_ns: int = 123_456,
        backend_name: str = "fake_backend",
    ) -> None:
        self._nd = num_detectors
        self._no = num_observables
        self._available = (
            available_report
            if available_report is not None
            else CapabilityReport.ready(
                reason="fake decoder ready",
                required=["fake"],
                detected_versions={"fake": "0.0"},
            )
        )
        self._warmup_raises = warmup_raises
        self._decode_raises = decode_raises
        self._per_batch_latency_ns = per_batch_latency_ns
        self._backend_name = backend_name
        self.warmup_calls: int = 0
        self.decode_calls: int = 0
        self.decode_batch_sizes: list[int] = []

    def available(self) -> CapabilityReport:
        return self._available

    def warmup(self) -> None:
        self.warmup_calls += 1
        if self._warmup_raises is not None:
            raise self._warmup_raises

    def decode_batch(self, syndromes: np.ndarray) -> Corrections:
        self.decode_calls += 1
        self.decode_batch_sizes.append(int(syndromes.shape[0]))
        if self._decode_raises is not None:
            raise self._decode_raises
        # Spend a known, small amount of real wall time so the test can
        # assert that the runner records the decoder-reported latency,
        # not its own end-to-end wall clock.
        time.sleep(0.002)
        preds = np.zeros((syndromes.shape[0], self._no), dtype=np.uint8)
        return Corrections(
            predictions=preds, latency_ns=int(self._per_batch_latency_ns)
        )

    def metadata(self) -> DecoderMetadata:
        return DecoderMetadata(
            backend_name=self._backend_name,
            backend_version="fake-1.0",
            model_path=None,
            model_sha256=None,
            receptive_field=None,
            supports_batching=True,
            supports_gpu=False,
            schema_version="1",
        )


def _factory(decoder: FakeDecoder) -> Callable[[str], Any]:
    def _f(name: str) -> Any:  # noqa: ARG001 - backend name is unused by the fake
        return decoder

    return _f


# ---------------------------------------------------------------------------
# AC tests — each name matches an acceptance_criteria entry.
# ---------------------------------------------------------------------------


def test_runner_executes_single_config_with_fake_decoder() -> None:
    cfg = _config(num_shots=2048)
    fake = FakeDecoder(num_detectors=32)
    result = run_single(
        cfg,
        decoder_factory=_factory(fake),
        num_detectors=32,
        batch_size=1024,
    )
    assert isinstance(result, RunResult)
    assert result.ok
    assert result.error is None
    assert result.shots_total == 2048
    assert result.batches == 2  # 2048 / 1024
    assert fake.decode_batch_sizes == [1024, 1024]
    assert len(result.per_batch_latency_ns) == 2
    assert result.run_id == cfg.run_id
    assert result.config == cfg


def test_runner_captures_decoder_unavailable_reason() -> None:
    unavailable = CapabilityReport.unavailable(
        reason="fake blocker for test",
        required=["fakedep>=1.0"],
        category="not_installed",
    )
    fake = FakeDecoder(available_report=unavailable)
    cfg = _config()
    result = run_single(
        cfg,
        decoder_factory=_factory(fake),
        num_detectors=16,
    )
    assert not result.ok
    assert result.error is not None
    assert "fake blocker for test" in result.error
    assert "not_installed" in result.error
    assert result.shots_total == 0
    assert result.batches == 0
    # Never warmed up if unavailable — that would be premature work.
    assert fake.warmup_calls == 0


def test_runner_captures_exception_into_result() -> None:
    fake = FakeDecoder(
        num_detectors=16,
        decode_raises=RuntimeError("simulated decode failure"),
    )
    cfg = _config(num_shots=2048)
    result = run_single(
        cfg,
        decoder_factory=_factory(fake),
        num_detectors=16,
        batch_size=1024,
    )
    assert not result.ok
    assert result.error is not None
    assert "simulated decode failure" in result.error
    assert "RuntimeError" in result.error
    # Runner does not crash — it still returns a RunResult.
    assert isinstance(result, RunResult)
    # Warmup ran (happens before decode), so metadata is captured even
    # though the run failed mid-decode.
    assert fake.warmup_calls == 1
    assert result.decoder_metadata.get("backend_name") == "fake_backend"


def test_runner_syndromes_are_deterministic_for_same_seed() -> None:
    cfg = _config(master_seed=20260421, worker_seed_slot=7, num_shots=2048)
    fake_a = FakeDecoder(num_detectors=24)
    fake_b = FakeDecoder(num_detectors=24)
    r1 = run_single(cfg, decoder_factory=_factory(fake_a), num_detectors=24)
    r2 = run_single(cfg, decoder_factory=_factory(fake_b), num_detectors=24)
    assert r1.ok and r2.ok
    # Corrections digest is predictions-only (all zeros from the fake),
    # so match that is expected. What we really want to assert is that
    # the generated syndromes are byte-identical across runs with the
    # same (master_seed, worker_seed_slot). Use generate_syndromes
    # directly to prove determinism.
    from app.core.seeding import derive_worker_seed

    seed = derive_worker_seed(cfg.master_seed, cfg.worker_seed_slot)
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed)
    s1 = generate_syndromes(num_detectors=24, shots=1024, rng=rng_a)
    s2 = generate_syndromes(num_detectors=24, shots=1024, rng=rng_b)
    assert np.array_equal(s1, s2)
    # And a different seed slot produces different bytes.
    rng_c = np.random.default_rng(
        derive_worker_seed(cfg.master_seed, cfg.worker_seed_slot + 1)
    )
    s3 = generate_syndromes(num_detectors=24, shots=1024, rng=rng_c)
    assert not np.array_equal(s1, s3)


def test_runner_measures_only_decode_batch_time() -> None:
    # Fake decoder reports a specific per-batch latency_ns. The runner
    # must copy that number verbatim — not substitute its own
    # wall-clock reading, which would include sampling + python
    # overhead.
    reported_latency = 42_000  # 42 microseconds, implausibly tight
    fake = FakeDecoder(
        num_detectors=16,
        per_batch_latency_ns=reported_latency,
    )
    cfg = _config(num_shots=3000)
    result = run_single(
        cfg,
        decoder_factory=_factory(fake),
        num_detectors=16,
        batch_size=1000,
    )
    assert result.ok
    assert result.batches == 3
    assert result.per_batch_latency_ns == [reported_latency] * 3


def test_runner_calls_warmup_before_decode() -> None:
    fake = FakeDecoder(num_detectors=16)
    cfg = _config(num_shots=1024)
    result = run_single(
        cfg,
        decoder_factory=_factory(fake),
        num_detectors=16,
        batch_size=1024,
    )
    assert result.ok
    assert fake.warmup_calls == 1
    assert fake.decode_calls == 1
    # Warmup was exercised before any decode; FakeDecoder.decode_calls
    # is incremented inside decode_batch, so if decode had run first
    # that ordering would still be 1/1. Assert the explicit ordering by
    # making warmup raise and verifying decode never ran.
    fake_fail = FakeDecoder(
        num_detectors=16, warmup_raises=RuntimeError("warmup boom")
    )
    result_fail = run_single(
        cfg,
        decoder_factory=_factory(fake_fail),
        num_detectors=16,
        batch_size=1024,
    )
    assert not result_fail.ok
    assert "warmup boom" in (result_fail.error or "")
    assert fake_fail.warmup_calls == 1
    assert fake_fail.decode_calls == 0


def test_runner_populates_decoder_metadata() -> None:
    fake = FakeDecoder(num_detectors=16, backend_name="fake_backend")
    cfg = _config(num_shots=1024)
    result = run_single(
        cfg,
        decoder_factory=_factory(fake),
        num_detectors=16,
        batch_size=1024,
    )
    assert result.ok
    md = result.decoder_metadata
    assert md["backend_name"] == "fake_backend"
    assert md["backend_version"] == "fake-1.0"
    assert md["schema_version"] == "1"
    assert md["supports_batching"] is True
    assert md["supports_gpu"] is False


def test_runner_handles_empty_shots_zero_batches() -> None:
    fake = FakeDecoder(num_detectors=16)
    cfg = _config(num_shots=0)
    result = run_single(
        cfg,
        decoder_factory=_factory(fake),
        num_detectors=16,
        batch_size=1024,
    )
    assert result.ok
    assert result.shots_total == 0
    assert result.batches == 0
    assert result.per_batch_latency_ns == []
    # Digest over empty input is the SHA256 of the empty string.
    assert result.corrections_digest == hashlib.sha256(b"").hexdigest()
    # No decode calls — but warmup still ran so the decoder is ready.
    assert fake.decode_calls == 0
    assert fake.warmup_calls == 1


def test_runner_corrections_digest_is_sha256_hex() -> None:
    fake = FakeDecoder(num_detectors=16)
    cfg = _config(num_shots=2048)
    result = run_single(
        cfg,
        decoder_factory=_factory(fake),
        num_detectors=16,
        batch_size=1024,
    )
    assert result.ok
    # SHA256 hex: exactly 64 lowercase hex characters.
    digest = result.corrections_digest
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    # Independently recompute: FakeDecoder returns all-zeros
    # (shots, 1) predictions across 2 batches of 1024 shots.
    expected = hashlib.sha256(
        np.zeros((2048, 1), dtype=np.uint8).tobytes()
    ).hexdigest()
    assert digest == expected


# ---------------------------------------------------------------------------
# Additional safety checks — not in the AC but cheap to pin down.
# ---------------------------------------------------------------------------


def test_generate_syndromes_rejects_bad_shape() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_syndromes(num_detectors=0, shots=10, rng=rng)
    with pytest.raises(ValueError):
        generate_syndromes(num_detectors=16, shots=-1, rng=rng)


def test_generate_syndromes_sampler_mode_validates_output() -> None:
    rng = np.random.default_rng(0)

    def bad_sampler(shots: int, rng: np.random.Generator) -> np.ndarray:
        return np.zeros((shots, 99), dtype=np.uint8)  # wrong detector count

    with pytest.raises(ValueError):
        generate_syndromes(
            num_detectors=16, shots=10, rng=rng, sampler=bad_sampler
        )


def test_runner_decoder_factory_failure_captured() -> None:
    def bad_factory(name: str) -> Any:
        raise RuntimeError("cannot build decoder")

    cfg = _config()
    result = run_single(
        cfg, decoder_factory=bad_factory, num_detectors=16
    )
    assert not result.ok
    assert "cannot build decoder" in (result.error or "")
    assert result.shots_total == 0


def test_runner_rejects_bad_arguments() -> None:
    fake = FakeDecoder(num_detectors=16)
    with pytest.raises(TypeError):
        run_single(
            "not-a-config",  # type: ignore[arg-type]
            decoder_factory=_factory(fake),
            num_detectors=16,
        )
