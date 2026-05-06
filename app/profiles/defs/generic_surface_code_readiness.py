"""Generic surface-code deployment-readiness profile.

Grounded in:
    - Stim rotated_memory_x / rotated_memory_z circuit generators
    - PyMatching MWPM baseline + correlated matching (v2.3+)
    - sinter Monte Carlo collector

Parameter choices are directly sourced from the research pack:
    - distances {5, 7}: default PyMatching benchmark set
    - rounds {5, 10, 25}: Stim CLI public examples use 5/10/100
    - bases {X, Z}: both rotated_memory tasks supported
    - p_errors {0.001, 0.003, 0.005}: public Stim + PyMatching example noise
"""

from __future__ import annotations

from app.profiles.schema import (
    CustomerBoundary,
    DecoderPath,
    ProfileSpec,
    ProvenanceSource,
)

PROFILE = ProfileSpec(
    profile_id="generic_surface_code_readiness",
    name="Generic surface-code deployment-readiness (Stim + PyMatching)",
    description=(
        "Open, reproducible, free of proprietary runtime baggage. "
        "Answers: under standard rotated-surface-code assumptions, does "
        "the preferred decoder stay ahead on logical error rate without "
        "unacceptable slowdown as distance and noise rise?"
    ),
    architecture="generic",
    intended_use=(
        "Baseline decision scenario for any team comparing classical "
        "MWPM paths on public surface-code memory circuits. Use as a "
        "regression fixture and as the starting point before architecture-"
        "specific profiles."
    ),
    limitations=(
        "Uses Stim-generated example circuits that are explicitly "
        "labelled by the Stim docs as insufficient for research-grade "
        "conclusions. Do not substitute this profile for customer "
        "fabrication-specific detector models."
    ),
    distances=(5, 7),
    rounds_by_distance={5: (5, 10, 25), 7: (5, 10, 25)},
    bases=("X", "Z"),
    p_errors=(0.001, 0.003, 0.005),
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
            "Under standard rotated-surface-code assumptions the "
            "per-shot logical error rate and throughput of each "
            "compared decoder path on public circuits.",
            "Whether correlated matching improves the logical error "
            "rate on this public noise distribution.",
            "Deterministic reproducibility of the comparison given a fixed master seed.",
        ),
        requires_customer_private_inputs=(
            "Whether the customer's hardware fault channels actually "
            "contain the hyperedge correlations that correlated matching "
            "targets.",
            "Customer detector error models with real calibration data.",
            "Target deployment runtime constraints (latency budget, "
            "throughput floor, controller interconnect).",
        ),
    ),
    provenance=(
        ProvenanceSource(
            label="Stim command-line reference (rotated surface-code tasks)",
            url="https://github.com/quantumlib/Stim/blob/main/doc/usage_command_line.md",
            cites=("distances", "rounds_by_distance", "bases", "p_errors"),
        ),
        ProvenanceSource(
            label="PyMatching docs (correlated matching v2.3+)",
            url="https://pymatching.readthedocs.io/en/latest/",
            cites=("decoder_paths", "noise_model_id"),
        ),
        ProvenanceSource(
            label="sinter command-line reference",
            url="https://github.com/quantumlib/Stim/blob/main/doc/sinter_command_line.md",
            cites=("rounds_by_distance", "p_errors"),
            note="Monte Carlo collection + deterministic CSV output.",
        ),
    ),
    caution_label="",
)
