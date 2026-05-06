"""Trapped-ion looser-latency proxy profile — CAUTION-TAGGED.

The research pack is explicit: public sources support this as a
looser-runtime, stronger-connectivity proxy, but they do NOT provide
a clean, recent, architecture-specific latency figure of the same
quality as Google's superconducting result. This profile therefore
ships with a caution_label and no hard latency budget.

Grounded in:
    - Trapped-ion PRX 2017: "Realistic trapped-ion toolbox for QEC"
    - Quantinuum QEC toolkit: real-time hybrid compute via WebAssembly
    - CUDA-Q QEC 0.6 on Quantinuum Helios-1 (partners/collaborators)
"""

from __future__ import annotations

from app.profiles.schema import (
    CustomerBoundary,
    DecoderPath,
    ProfileSpec,
    ProvenanceSource,
)

PROFILE = ProfileSpec(
    profile_id="trapped_ion_looser_latency",
    name="Trapped-ion looser-latency proxy (CAUTION — proxy-only)",
    description=(
        "Looser-runtime, stronger-connectivity proxy for trapped-ion "
        "QEC decoder comparisons. Emphasises accuracy and integration "
        "readiness under larger windows rather than hard microsecond-"
        "class latency budgets."
    ),
    architecture="trapped_ion",
    intended_use=(
        "Internal reports where a trapped-ion-style connectivity "
        "assumption is required. Use for methodology comparison; do "
        "not use for latency-budget certification."
    ),
    limitations=(
        "No public deployment-grade latency figure for trapped-ion "
        "architectures is surfaced in the research pack. The profile "
        "runs the same classical decoder comparison as the generic "
        "profile with no hard latency envelope."
    ),
    distances=(5, 7),
    rounds_by_distance={5: (5, 10), 7: (7, 25)},
    bases=("X", "Z"),
    p_errors=(0.001, 0.003),
    noise_model_id="simple_depolarizing",
    decoder_paths=(
        DecoderPath(
            label="PyMatching uncorrelated MWPM (baseline)",
            backend="pymatching_baseline",
        ),
        DecoderPath(
            label="PyMatching correlated MWPM",
            backend="pymatching_correlated",
            requires=("pymatching>=2.3",),
        ),
    ),
    export_checks=(),
    runtime_budget=None,
    boundary=CustomerBoundary(
        public_proxy_can_conclude=(
            "Classical decoder accuracy comparison on public rotated "
            "surface-code memory circuits under a trapped-ion-style "
            "assumption of looser timing tolerance.",
        ),
        requires_customer_private_inputs=(
            "Any deployment-grade latency budget for the customer's trapped-ion architecture.",
            "Private detector error model tuned to the customer's ion-trap hardware.",
            "Real-time hybrid-compute integration constraints (WebAssembly, "
            "Helios-1, or other runtime).",
        ),
    ),
    provenance=(
        ProvenanceSource(
            label="PRX 2017 — Realistic trapped-ion toolbox for QEC",
            url="https://link.aps.org/doi/10.1103/PhysRevX.7.041061",
            cites=("architecture", "decoder_paths"),
        ),
        ProvenanceSource(
            label="CUDA-Q QEC 0.6 release (real-time on Helios-1)",
            url="https://nvidia.github.io/cudaqx/components/qec/introduction.html",
            cites=("architecture",),
            note="CUDA-Q QEC supports real-time decoding on Quantinuum Helios-1.",
        ),
    ),
    caution_label=(
        "proxy-only, not deployment-grade: no public trapped-ion "
        "decoder-latency figure equivalent in quality to Willow."
    ),
    allowed_overrides=("num_shots", "master_seed"),
)
