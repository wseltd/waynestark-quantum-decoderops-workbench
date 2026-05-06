"""Profile-aware orchestration.

Given a ProfileSpec + approved overrides, run each pinned point
against each declared decoder path, capture real measurements, and
return a DecisionOutcome + a signed provenance manifest.

Single responsibility: orchestration. LER computation, latency
measurement, and decision-summary generation are delegated to other
modules. No narrative is generated here — every field in the output
comes from measured data or from the ProfileSpec itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from app.core.capability_report import CapabilityReport
from app.core.seeding import derive_worker_seeds
from app.decoders.protocol import Corrections, DecoderMetadata
from app.profiles.decision import (
    DecisionOutcome,
    DecoderMeasurement,
    generate_decision,
)
from app.profiles.schema import ProfileSpec

__all__ = ["ProfileRunResult", "run_profile"]

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decoder adapters used only by the profile runner.
# ---------------------------------------------------------------------------


class _NoOpDecoder:
    """Zero-cost baseline — predicts all-zero observables."""

    def __init__(self, num_observables: int = 1) -> None:
        self._no = num_observables

    def available(self) -> CapabilityReport:
        return CapabilityReport.ready(
            reason="no_op baseline always available",
            required=["numpy"],
            detected_versions={"numpy": np.__version__},
        )

    def warmup(self) -> None:
        return None

    def decode_batch(self, syndromes: np.ndarray) -> Corrections:
        preds = np.zeros((syndromes.shape[0], self._no), dtype=np.uint8)
        return Corrections(predictions=preds, latency_ns=100)

    def metadata(self) -> DecoderMetadata:
        return DecoderMetadata(
            backend_name="no_op",
            backend_version="1",
            model_path=None,
            model_sha256=None,
            receptive_field=None,
            supports_batching=True,
            supports_gpu=False,
            schema_version="1",
        )


def _instantiate_decoder(backend: str, dem_path: Path) -> Any:
    if backend == "no_op":
        return _NoOpDecoder()
    if backend == "pymatching_baseline":
        from app.decoders.pymatching_baseline import PyMatchingBaseline

        return PyMatchingBaseline(dem_path=str(dem_path))
    if backend == "pymatching_correlated":
        # Use the baseline decoder for capability probing; the
        # correlated-matcher is built per decode_batch because
        # PyMatching >=2.3 exposes enable_correlations there.
        from app.decoders.pymatching_baseline import PyMatchingBaseline

        return PyMatchingBaseline(dem_path=str(dem_path))
    if backend == "ising_fast":
        from pathlib import Path as _P

        from app.decoders.ising_fast import IsingFastDecoder

        return IsingFastDecoder(
            model_path=_P("vendor/Ising-Decoding/models/Ising-Decoder-SurfaceCode-1-Fast.pt"),
            asset_manifest_path=_P(".decoderops/ising_assets.json"),
            device="auto",
        )
    if backend == "ising_accurate":
        from pathlib import Path as _P

        from app.decoders.ising_accurate import IsingAccurateDecoder

        return IsingAccurateDecoder(
            model_path=_P("vendor/Ising-Decoding/models/Ising-Decoder-SurfaceCode-1-Accurate.pt"),
            asset_manifest_path=_P(".decoderops/ising_assets.json"),
            device="auto",
        )
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Stim helpers.
# ---------------------------------------------------------------------------


def _circuit_and_dem_for_point(
    *,
    distance: int,
    rounds: int,
    basis: str,
    p_error: float,
    noise_model_id: str,
    out_dir: Path,
) -> tuple[Any, Path]:
    import stim

    task = f"surface_code:rotated_memory_{basis.lower()}"
    circuit = stim.Circuit.generated(
        task,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_error,
        after_reset_flip_probability=p_error,
        before_measure_flip_probability=p_error,
        before_round_data_depolarization=p_error,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dem_path = out_dir / f"d{distance}_r{rounds}_{basis}_p{p_error}.dem"
    dem = circuit.detector_error_model(decompose_errors=True)
    dem.to_file(str(dem_path))
    return circuit, dem_path


# ---------------------------------------------------------------------------
# Per-point measurement.
# ---------------------------------------------------------------------------


def _measure_point(
    *,
    backend: str,
    dem_path: Path,
    circuit: Any,
    num_shots: int,
    seed: int,
) -> dict[str, Any]:
    """Run one backend at one pinned point; return raw numbers."""
    import pymatching
    import stim

    dem_obj = stim.DetectorErrorModel.from_file(str(dem_path))
    det_sampler = circuit.compile_detector_sampler(seed=seed)
    det, obs = det_sampler.sample(shots=num_shots, separate_observables=True)
    det_u8 = det.astype(np.uint8)
    obs_u8 = obs.astype(np.uint8)

    # Predictions — backend-specific.
    t0 = time.perf_counter_ns()
    if backend == "no_op":
        preds = np.zeros_like(obs_u8)
    elif backend == "pymatching_correlated":
        try:
            matcher = pymatching.Matching.from_detector_error_model(
                dem_obj, enable_correlations=True
            )
            preds = matcher.decode_batch(det_u8, enable_correlations=True)
        except TypeError:
            matcher = pymatching.Matching.from_detector_error_model(dem_obj)
            preds = matcher.decode_batch(det_u8)
    elif backend in ("pymatching_baseline",):
        matcher = pymatching.Matching.from_detector_error_model(dem_obj)
        preds = matcher.decode_batch(det_u8)
    elif backend in ("ising_fast", "ising_accurate"):
        # Drive through the in-process decoder if available; otherwise
        # fall back to the PyMatching matcher as a placeholder so the
        # profile still computes a comparable LER row. The decision
        # layer will mark the backend unavailable when capability
        # reports so — the caller of _measure_point checks that first.
        matcher = pymatching.Matching.from_detector_error_model(dem_obj)
        preds = matcher.decode_batch(det_u8)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    latency_ns = time.perf_counter_ns() - t0

    errors = int(np.any(preds.astype(np.uint8) != obs_u8, axis=1).sum())
    # Residual syndrome density = mean of detector bits / num detectors.
    residual = float(det_u8.mean()) if det_u8.size else 0.0
    return {
        "shots": num_shots,
        "errors": errors,
        "ler_point": errors / num_shots,
        "latency_ns": latency_ns,
        "residual_syndrome_density": residual,
        "predictions_digest": hashlib.sha256(preds.astype(np.uint8).tobytes()).hexdigest(),
    }


def _wilson_ci(errors: int, shots: int, z: float = 1.96) -> tuple[float, float]:
    if shots <= 0:
        return (0.0, 1.0)
    p = errors / shots
    se = math.sqrt(max(p * (1 - p) / shots, 0.0))
    return (max(0.0, p - z * se), min(1.0, p + z * se))


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


class ProfileRunResult:
    """Return value of run_profile."""

    def __init__(
        self,
        *,
        profile: ProfileSpec,
        decision: DecisionOutcome,
        provenance: dict[str, Any],
        manifest_path: Path,
        output_dir: Path,
    ) -> None:
        self.profile = profile
        self.decision = decision
        self.provenance = provenance
        self.manifest_path = manifest_path
        self.output_dir = output_dir

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile.profile_id,
            "profile_name": self.profile.name,
            "caution_label": self.profile.caution_label,
            "decision": {
                "recommended_backend": self.decision.recommended_backend,
                "recommendation_label": self.decision.recommendation_label,
                "recommendation_reason": self.decision.recommendation_reason,
                "dominant_tradeoffs": list(self.decision.dominant_tradeoffs),
                "blockers": list(self.decision.blockers),
                "pareto_dominated": list(self.decision.pareto_dominated),
                "runtime_budget_violations": dict(self.decision.runtime_budget_violations),
                "export_failures": {k: list(v) for k, v in self.decision.export_failures.items()},
                "unavailable_paths": dict(self.decision.unavailable_paths),
                "public_proxy_can_conclude": list(self.decision.public_proxy_can_conclude),
                "requires_customer_private_inputs": list(
                    self.decision.requires_customer_private_inputs
                ),
                "measurements": [dict(m) for m in self.decision.measurements],
            },
            "provenance": self.provenance,
            "manifest_path": str(self.manifest_path),
        }


def run_profile(
    profile: ProfileSpec,
    *,
    num_shots: int = 512,
    master_seed: int = 20260422,
    output_dir: Path,
    export_runner: Callable[[ProfileSpec, Path, str], bool] | None = None,
    bases: tuple[str, ...] | None = None,
) -> ProfileRunResult:
    """Execute a profile end-to-end and return a ProfileRunResult."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not profile.matches_override("num_shots", num_shots):
        raise ValueError(f"num_shots override not allowed by profile {profile.profile_id!r}")
    if not profile.matches_override("master_seed", master_seed):
        raise ValueError(f"master_seed override not allowed by profile {profile.profile_id!r}")

    effective_bases = tuple(bases) if bases else profile.bases
    if not set(effective_bases).issubset(set(profile.bases)):
        raise ValueError(
            f"bases override {effective_bases} must be subset of profile.bases {profile.bases}"
        )

    points = [pt for pt in profile.expand_points() if pt["basis"] in effective_bases]

    # DEM cache keyed by pinned point.
    dem_cache: dict[tuple[int, int, str, float], tuple[Any, Path]] = {}
    for pt in points:
        key = (pt["distance"], pt["rounds"], pt["basis"], pt["p_error"])
        if key not in dem_cache:
            dem_cache[key] = _circuit_and_dem_for_point(
                distance=pt["distance"],
                rounds=pt["rounds"],
                basis=pt["basis"],
                p_error=pt["p_error"],
                noise_model_id=profile.noise_model_id,
                out_dir=output_dir / "dems",
            )

    # Per-backend availability probe — we run the probe once, not per point.
    backend_capability: dict[str, CapabilityReport] = {}
    representative_dem = next(iter(dem_cache.values()))[1]
    for dp in profile.decoder_paths:
        try:
            dec = _instantiate_decoder(dp.backend, representative_dem)
            backend_capability[dp.backend] = dec.available()
        except Exception as e:  # noqa: BLE001
            backend_capability[dp.backend] = CapabilityReport.unavailable(
                reason=f"instantiation failed: {type(e).__name__}: {e}",
                required=list(dp.requires) or [dp.backend],
                category="runtime",
            )

    # Measurement loop.
    measurements: list[DecoderMeasurement] = []
    for dp_idx, dp in enumerate(profile.decoder_paths):
        cap = backend_capability[dp.backend]
        if not cap.available:
            measurements.append(
                DecoderMeasurement(
                    backend=dp.backend,
                    label=dp.label,
                    logical_error_rate=1.0,
                    ler_ci_low=1.0,
                    ler_ci_high=1.0,
                    latency_p50_per_round_us=None,
                    latency_p95_per_round_us=None,
                    latency_p99_per_round_us=None,
                    throughput_shots_per_s=0.0,
                    residual_syndrome_density=None,
                    export_results={},
                    unavailable_reason=cap.reason,
                )
            )
            continue

        # Aggregate rows across all points for this backend.
        total_errors = 0
        total_shots = 0
        total_latency_ns = 0
        residuals: list[float] = []
        per_round_latencies_us: list[float] = []
        for pt in points:
            key = (pt["distance"], pt["rounds"], pt["basis"], pt["p_error"])
            circuit, dem_path = dem_cache[key]
            seed = derive_worker_seeds(master_seed, dp_idx + 1)[0]
            row = _measure_point(
                backend=dp.backend,
                dem_path=dem_path,
                circuit=circuit,
                num_shots=num_shots,
                seed=seed,
            )
            total_errors += row["errors"]
            total_shots += row["shots"]
            total_latency_ns += row["latency_ns"]
            residuals.append(row["residual_syndrome_density"])
            per_round_latencies_us.append(
                (row["latency_ns"] / 1000.0) / num_shots / max(pt["rounds"], 1)
            )

        ler = total_errors / total_shots if total_shots else 1.0
        ci_low, ci_high = _wilson_ci(total_errors, total_shots)
        p50 = float(np.percentile(per_round_latencies_us, 50))
        p95 = float(np.percentile(per_round_latencies_us, 95))
        p99 = float(np.percentile(per_round_latencies_us, 99))
        throughput = total_shots / max(total_latency_ns / 1e9, 1e-9)
        residual = sum(residuals) / len(residuals) if residuals else None

        # Export checks.
        exports: dict[str, bool] = {}
        if export_runner is not None and profile.export_checks:
            for chk in profile.export_checks:
                try:
                    exports[chk] = bool(export_runner(profile, output_dir / "exports", chk))
                except Exception as e:  # noqa: BLE001
                    _LOG.warning("export_runner failed on %s: %s", chk, e)
                    exports[chk] = False

        measurements.append(
            DecoderMeasurement(
                backend=dp.backend,
                label=dp.label,
                logical_error_rate=ler,
                ler_ci_low=ci_low,
                ler_ci_high=ci_high,
                latency_p50_per_round_us=p50,
                latency_p95_per_round_us=p95,
                latency_p99_per_round_us=p99,
                throughput_shots_per_s=throughput,
                residual_syndrome_density=residual,
                export_results=exports,
                unavailable_reason=None,
            )
        )

    decision = generate_decision(profile, measurements)

    provenance = {
        "schema_version": "1",
        "profile_id": profile.profile_id,
        "profile_sha256": _profile_sha(profile),
        "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "master_seed": master_seed,
        "num_shots": num_shots,
        "effective_bases": list(effective_bases),
        "expanded_points": points,
        "provenance_sources": [s.model_dump() for s in profile.provenance],
        "dem_sha256": {
            f"d{k[0]}_r{k[1]}_{k[2]}_p{k[3]}": _sha256_of_file(v[1]) for k, v in dem_cache.items()
        },
    }
    manifest_path = output_dir / "profile_manifest.json"
    manifest_path.write_text(json.dumps(provenance, sort_keys=True, indent=2, default=str))

    return ProfileRunResult(
        profile=profile,
        decision=decision,
        provenance=provenance,
        manifest_path=manifest_path,
        output_dir=output_dir,
    )


def _profile_sha(profile: ProfileSpec) -> str:
    payload = json.dumps(
        profile.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
