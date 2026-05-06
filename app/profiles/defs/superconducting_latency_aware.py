"""Superconducting-style latency-aware QEC profile.

Anchored to:
    - Google Willow (arXiv:2408.13687): distance-5 + distance-7
      rotated surface-code memories, d=7 reaches 0.143% per cycle,
      real-time decoder at distance 5 averages 63 μs at 1.1 μs cycle.
    - Riverlane Deltaflow 2: 16.32 μs mean decoding latency, target
      below 20 μs, up to 250 physical qubits.
    - NVIDIA Ising-Decoding public config: 25-parameter surface-code
      circuit-level noise model; d=13, rounds=104, basis=X, p=0.003
      is the canonical public test-data generation point.
    - CUDA-Q Realtime / NVQLink: few-μs FPGA-GPU round-trip envelope.

This profile tests the exact thing a builder worries about: whether an
AI-assisted path buys enough residual-syndrome reduction or logical-
error improvement to justify runtime + export complexity under
stringent timing assumptions.
"""

from __future__ import annotations

from app.profiles.schema import (
    CustomerBoundary,
    DecoderPath,
    ProfileSpec,
    ProvenanceSource,
    RuntimeBudget,
)

PROFILE = ProfileSpec(
    profile_id="superconducting_latency_aware",
    name="Superconducting-style latency-aware QEC (Willow + Ising-Decoding)",
    description=(
        "Architecture-specific profile anchored to Google Willow's "
        "below-threshold surface-code memory timing (1.1 μs cycle, "
        "63 μs decoder-latency envelope at d=5), Riverlane Deltaflow 2's "
        "<20 μs mean decoder-latency target, and NVIDIA Ising-Decoding's "
        "public circuit-level noise configuration."
    ),
    architecture="superconducting",
    intended_use=(
        "Compare classical MWPM paths and AI pre-decoder paths under a "
        "public superconducting-style latency envelope. Suitable for "
        "internal engineering reports that need a builder-shaped "
        "decision context before customer-specific detector models are "
        "available."
    ),
    limitations=(
        "Public circuit noise is not a substitute for hardware-calibrated "
        "detector models. Latency numbers measured by this profile are "
        "on the workbench test host, not on the customer's controller / "
        "FPGA / NVQLink stack. The profile's latency envelope is drawn "
        "from public primary sources but does not certify the customer's "
        "deployment will hit those numbers."
    ),
    distances=(5, 7, 13),
    rounds_by_distance={5: (5,), 7: (7,), 13: (104,)},
    bases=("X",),
    p_errors=(0.003,),
    noise_model_id="circuit_level",
    decoder_paths=(
        DecoderPath(
            label="PyMatching uncorrelated MWPM",
            backend="pymatching_baseline",
        ),
        DecoderPath(
            label="PyMatching correlated MWPM",
            backend="pymatching_correlated",
            requires=("pymatching>=2.3",),
        ),
        DecoderPath(
            label="Ising Fast (RF=9) + PyMatching global",
            backend="ising_fast",
            requires=(
                "torch",
                "vendor/Ising-Decoding checkpoints",
            ),
        ),
        DecoderPath(
            label="Ising Accurate (RF=13) + PyMatching global",
            backend="ising_accurate",
            requires=(
                "torch",
                "vendor/Ising-Decoding checkpoints",
            ),
        ),
    ),
    export_checks=(),
    runtime_budget=RuntimeBudget(
        latency_us_target=20.0,
        latency_us_hard_cap=63.0,
        cycle_time_us=1.1,
        source_notes=(
            "Target 20 μs from Riverlane Deltaflow 2 public release notes. "
            "Hard cap 63 μs from Willow's d=5 real-time decoder average. "
            "Cycle time 1.1 μs from the Willow paper."
        ),
    ),
    boundary=CustomerBoundary(
        public_proxy_can_conclude=(
            "Whether a given decoder path meets the 20 μs target and/or "
            "63 μs hard cap on this test host, under public noise.",
            "Whether the AI-predecoder paths reduce logical error rate "
            "meaningfully versus PyMatching baselines on public Ising-"
            "Decoding circuits.",
            "Which decoder path wins on a logical-error-rate-vs-latency "
            "Pareto frontier for the pinned (d, rounds, basis, p) set.",
        ),
        requires_customer_private_inputs=(
            "Whether the customer's superconducting hardware actually "
            "meets Willow-class cycle times or Deltaflow-2 latency target.",
            "Customer fabrication-specific leakage and crosstalk signatures.",
            "The customer's controller-to-accelerator interconnect "
            "(NVQLink, DGX Quantum, or other) and its round-trip envelope.",
            "Whether the AI predecoder generalises to the customer's private detector error model.",
        ),
    ),
    provenance=(
        ProvenanceSource(
            label="Google Willow paper (arXiv:2408.13687)",
            url="https://arxiv.org/abs/2408.13687",
            cites=(
                "runtime_budget.cycle_time_us",
                "runtime_budget.latency_us_hard_cap",
                "distances",
            ),
            note=(
                "d=5 real-time decoder averages 63 μs; cycle time 1.1 μs; "
                "d=7 memory reaches 0.143% per cycle."
            ),
        ),
        ProvenanceSource(
            label="Riverlane Deltaflow 2",
            url="https://www.riverlane.com/quantum-error-correction-stack/deltaflow-2",
            cites=("runtime_budget.latency_us_target",),
            note="Mean decoder latency 16.32 μs; target <20 μs.",
        ),
        ProvenanceSource(
            label="NVIDIA Ising-Decoding public config",
            url="https://github.com/NVIDIA/Ising-Decoding/blob/main/conf/config_public.yaml",
            cites=("noise_model_id", "p_errors", "distances", "rounds_by_distance"),
            note=(
                "25-parameter circuit-level noise model; public d=13 "
                "rounds=104 basis=X p=0.003 generate_test_data example."
            ),
        ),
        ProvenanceSource(
            label="NVIDIA Ising-Decoding local runner",
            url="https://github.com/NVIDIA/Ising-Decoding/blob/main/code/scripts/local_run.sh",
            cites=("decoder_paths",),
            note="Shipped public model IDs 1 (Fast) + 4 (Accurate).",
        ),
        ProvenanceSource(
            label="CUDA-Q Realtime docs (NVQLink round-trip envelope)",
            url="https://nvidia.github.io/cuda-quantum/latest/using/realtime.html",
            cites=("runtime_budget.latency_us_hard_cap",),
            note="Few-μs FPGA-GPU round-trip constraint.",
        ),
    ),
    caution_label="",
    allowed_overrides=("num_shots", "master_seed"),
)
