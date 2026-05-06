"""AI predecoder export-and-runtime profile.

Strongest public proxy for a customer-style deployment call — because
the artefacts are not just circuits and DEMs, they are runtime-facing
files: shipped NVIDIA Ising checkpoints, ONNX exports, optional
TensorRT engines, and CUDA-Q Realtime-ready .bin test data.

Anchored to:
    - NVIDIA Ising-Decoding: d=13, rounds=104, basis=X, p=0.003,
      simple_noise (canonical public test-data generation example)
    - Public model IDs 1 (Fast, RF=9) and 4 (Accurate, RF=13)
    - Local runner WORKFLOW=inference / ONNX_WORKFLOW ∈ {1,2,3}
    - CUDA-Q Realtime AI predecoder demo: 11 MB per predecoder
      TensorRT engine at d13_r104
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
    profile_id="ai_predecoder_export_runtime",
    name="AI predecoder export-and-runtime (NVIDIA Ising + ONNX/TRT)",
    description=(
        "Tests not just accuracy but deployment survivability. "
        "Exercises the full shipped NVIDIA Ising-Decoding path: real "
        "checkpoint load, real inference, ONNX export, TensorRT "
        "engine build (optional), and CUDA-Q Realtime-ready .bin "
        "test data generation."
    ),
    architecture="superconducting",
    intended_use=(
        "Pre-deployment decoder comparison for teams that intend to "
        "run an AI predecoder in a real runtime (CUDA-Q Realtime, "
        "NVQLink, or similar). Produces honest signals on whether "
        "each path survives ONNX/TRT export, and how much the "
        "predecoder helps vs the PyMatching baseline."
    ),
    limitations=(
        "The AI predecoder checkpoints are trained on the NVIDIA "
        "Ising-Decoding public 25-parameter circuit-level noise "
        "distribution. Generalisation to customer-private detector "
        "models is not guaranteed. TensorRT engine build is gated "
        "by local driver / tensorrt-cu13 availability."
    ),
    distances=(13,),
    rounds_by_distance={13: (104,)},
    bases=("X",),
    p_errors=(0.003,),
    noise_model_id="circuit_level",
    decoder_paths=(
        DecoderPath(
            label="No-op (upper bound for residual syndrome)",
            backend="no_op",
        ),
        DecoderPath(
            label="PyMatching uncorrelated MWPM (baseline)",
            backend="pymatching_baseline",
        ),
        DecoderPath(
            label="Ising Fast (RF=9) + PyMatching global",
            backend="ising_fast",
            requires=("torch", "vendor/Ising-Decoding checkpoints"),
        ),
        DecoderPath(
            label="Ising Accurate (RF=13) + PyMatching global",
            backend="ising_accurate",
            requires=("torch", "vendor/Ising-Decoding checkpoints"),
        ),
    ),
    export_checks=(
        "onnx_export_workflow_1",
        "onnx_export_workflow_2_int8",
        "onnx_validation",
        "tensorrt_engine_build",
        "cudaq_qec_test_data",
        "tarball_offline_verify",
    ),
    runtime_budget=RuntimeBudget(
        latency_us_target=20.0,
        latency_us_hard_cap=63.0,
        source_notes=(
            "Same public sources as the superconducting_latency_aware "
            "profile; reused here so the export + runtime numbers are "
            "interpreted against the same envelope."
        ),
    ),
    boundary=CustomerBoundary(
        public_proxy_can_conclude=(
            "Whether the shipped Ising Fast and Accurate checkpoints "
            "load, run on the configured GPU, and reduce LER relative "
            "to PyMatching baseline on public circuits.",
            "Whether each checkpoint exports cleanly to ONNX and "
            "optionally to a TensorRT engine on the local runtime.",
            "Residual-syndrome density (CUDA-Q Realtime metric) before "
            "and after the AI predecoder.",
            "SHA256-stamped .bin bundle for downstream CUDA-Q Realtime consumption.",
        ),
        requires_customer_private_inputs=(
            "Whether the checkpoints generalise to the customer's private detector error model.",
            "Runtime behaviour on the customer's controller + accelerator "
            "interconnect (OPX + DGX Quantum, NVQLink, etc.).",
            "Production batching / pipelining limits on the customer's hardware.",
            "Driver / container / security policies that could block "
            "TensorRT or closed NVIDIA components at deployment.",
        ),
    ),
    provenance=(
        ProvenanceSource(
            label="NVIDIA Ising-Decoding README",
            url="https://github.com/NVIDIA/Ising-Decoding/blob/main/README.md",
            cites=(
                "distances",
                "rounds_by_distance",
                "bases",
                "p_errors",
                "decoder_paths",
                "export_checks",
            ),
            note="Public test-data example: d=13 rounds=104 basis=X p=0.003.",
        ),
        ProvenanceSource(
            label="NVIDIA Ising-Decoding local_run.sh",
            url="https://github.com/NVIDIA/Ising-Decoding/blob/main/code/scripts/local_run.sh",
            cites=("export_checks",),
            note=(
                "ONNX_WORKFLOW=1 → export only; =2 → export+TRT; "
                "QUANT_FORMAT=int8/fp8; =3 → load pre-built engine."
            ),
        ),
        ProvenanceSource(
            label="NVIDIA Ising-Decoding public config",
            url="https://github.com/NVIDIA/Ising-Decoding/blob/main/conf/config_public.yaml",
            cites=("noise_model_id",),
        ),
        ProvenanceSource(
            label="CUDA-Q Realtime AI predecoder demo",
            url="https://nvidia.github.io/cudaqx/examples_rst/qec/realtime_predecoder_pymatching.html",
            cites=("export_checks", "runtime_budget"),
            note="d13_r104 TRT engine ≈11 MB per predecoder instance.",
        ),
        ProvenanceSource(
            label="TensorRT SDK licence (redistribution posture)",
            url="https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html",
            cites=("export_checks",),
            note=(
                "TensorRT engines are not bundled; operators fetch TRT "
                "separately under NVIDIA SDK licence."
            ),
        ),
    ),
    caution_label="",
    allowed_overrides=("num_shots", "master_seed"),
)
