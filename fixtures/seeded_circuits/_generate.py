"""Generate deterministic Stim surface-code fixtures (T114, T115)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import stim

_DEFAULT_P = 0.001


def _build_rotated_memory_z(
    distance: int, rounds: int, p: float = _DEFAULT_P
) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def _write_with_sidecar(
    circuit: stim.Circuit,
    output_path: Path,
    *,
    distance: int,
    rounds: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    circuit.to_file(str(output_path))
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "distance": distance,
        "rounds": rounds,
        "basis": "Z",
        "p_error": _DEFAULT_P,
        "stim_version": stim.__version__,
        "sha256": digest,
    }
    meta_path.write_text(
        json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return output_path


def generate_surface_d3_r3(output_path: Path) -> Path:
    circuit = _build_rotated_memory_z(3, 3)
    return _write_with_sidecar(
        circuit, Path(output_path), distance=3, rounds=3
    )


def generate_surface_d5_r5(output_path: Path) -> Path:
    circuit = _build_rotated_memory_z(5, 5)
    return _write_with_sidecar(
        circuit, Path(output_path), distance=5, rounds=5
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture", choices=("d3_r3", "d5_r5"), default="d3_r3"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.fixture == "d3_r3":
        generate_surface_d3_r3(args.output)
    else:
        generate_surface_d5_r5(args.output)


if __name__ == "__main__":
    main()
