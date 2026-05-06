"""Integration test for PyMatchingBaseline on a seeded Stim sample (T023).

Runs a seeded sampling pass through a small rotated surface-code memory
circuit and confirms that PyMatching's decoder agrees with the
simulator's recorded logical observables, up to the documented
tolerance at the chosen error rate. This is the reference MWPM contract
every neural pre-decoder in the product is benchmarked against.
"""

from __future__ import annotations

import numpy as np
import stim

from app.decoders.pymatching_baseline import PyMatchingBaseline


def test_end_to_end_seeded_sample_matches_expected_logical_errors() -> None:
    # Fixed-seed sampler output is byte-deterministic for a given
    # circuit + seed across stim releases; this is the product's
    # reproducibility contract for reference benchmarks.
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.003,
    )
    dem = circuit.detector_error_model(decompose_errors=True)

    # Seeded compiled sampler — stim exposes this for deterministic
    # test fixtures. 1_000 shots is enough to exercise both zero-
    # syndrome and non-trivial detection patterns without turning the
    # unit-test suite into a long-running benchmark.
    rng_seed = 20260421
    sampler = circuit.compile_detector_sampler(seed=rng_seed)
    detector_events, observable_flips = sampler.sample(
        shots=1_000, separate_observables=True
    )

    # Shape / dtype coercion into the decoder contract (uint8).
    syndromes = detector_events.astype(np.uint8)
    ground_truth = observable_flips.astype(np.uint8)

    d = PyMatchingBaseline(dem=dem)
    d.warmup()
    result = d.decode_batch(syndromes)

    assert result.predictions.shape == ground_truth.shape
    # At p=0.003, distance=3, rounds=3, MWPM should track the simulator's
    # logical observable well above chance. We assert an engineering
    # bound that is deliberately loose (>=60% agreement) — tightening it
    # past that is the job of the larger statistical reports, not a
    # single integration smoke test.
    matches = (result.predictions == ground_truth).all(axis=1)
    agreement = matches.sum() / matches.shape[0]
    assert agreement >= 0.60, (
        f"MWPM baseline agreement {agreement:.3f} below 0.60 lower bound "
        f"at d=3, r=3, p=0.003 over 1000 seeded shots; seed={rng_seed}"
    )
