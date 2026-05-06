# Stack Decisions

## Languages and Runtimes
- **Python 3.12** — pinned via `requirements/core.lock.txt` with
  `--require-hashes`.
- **CUDA 13.0** is the Tier-3 GPU runtime (opt-in; customer-installed).

## Core Frameworks
- **FastAPI** for the HTTP surface — async-native, Pydantic-backed.
- **Typer** for the CLI — Click under the hood, type-hint-driven.
- **HTTPX** as the single HTTP client (CLI → API).

## Persistence
- **SQLAlchemy 2.x** ORM with declarative `Mapped[...]` columns.
- **DuckDB** as local-first default.
- **PostgreSQL** as service-backed option with Alembic-managed schema.

## QEC libraries
- **Stim** for circuit generation, detector sampling, and DEM derivation.
- **PyMatching** for the MWPM baseline decoder.
- **Sinter** for Monte-Carlo LER sampling (pymatching-only in v1).

## Tier-3 GPU libraries (opt-in)
- `torch` 2.11 + `onnxruntime-gpu` + `tensorrt-cu13` + `nvidia-modelopt` +
  `cudaq`, `cudaq-qec`, `cuquantum-python-cu13`, `cupy-cuda13x`.
- All are capability-detected; the workbench runs on CPU-only hardware.

## Vendor code
- NVIDIA Ising-Decoding checkpoints (Apache-2.0) under
  `vendor/Ising-Decoding/`. SHA256-verified on load.

## Determinism budget
- Every run writes a `ReproducibilityFingerprint` and the packaging
  layer emits byte-reproducible tarballs (fixed mtime 2024-01-01 UTC,
  PAX format, gzip mtime=0).

## References
- `research-and-resourced.md`, `nvidia-ising-calibration.md`.
