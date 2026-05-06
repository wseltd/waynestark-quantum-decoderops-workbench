"""Generate DEM fixtures from seeded circuits (T116)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import stim


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_dem_from_circuit(
    circuit_path: Path, output_path: Path
) -> Path:
    circuit_path = Path(circuit_path)
    output_path = Path(output_path)
    meta_path = circuit_path.with_suffix(".meta.json")
    if meta_path.exists():
        upstream = json.loads(meta_path.read_text())
        expected = upstream.get("sha256")
        actual = _sha256(circuit_path)
        if expected and expected != actual:
            raise ValueError(
                f"source circuit SHA256 mismatch: expected {expected}, got {actual}"
            )
    circuit = stim.Circuit.from_file(str(circuit_path))
    dem = circuit.detector_error_model(
        decompose_errors=True,
        flatten_loops=True,
        ignore_decomposition_failures=False,
        approximate_disjoint_errors=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dem.to_file(str(output_path))
    dem_sha = _sha256(output_path)
    circuit_sha = _sha256(circuit_path)
    meta = {
        "source_circuit": str(circuit_path),
        "source_circuit_sha256": circuit_sha,
        "dem_sha256": dem_sha,
        "stim_version": stim.__version__,
        "decompose_errors": True,
        "flatten_loops": True,
        "approximate_disjoint_errors": True,
    }
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate_dem_from_circuit(args.circuit, args.output)


if __name__ == "__main__":
    main()
