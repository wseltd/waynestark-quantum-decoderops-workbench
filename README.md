# Quantum DecoderOps Workbench

A local-first, on-prem-first, vendor-neutral Python workbench for
benchmarking quantum error correction (QEC) decoder pipelines. Built
around **Stim**, **PyMatching**, **sinter**, and the shipped
**NVIDIA Ising-Decoding** Apache-2.0 checkpoints (Fast RF=9 and
Accurate RF=13). FastAPI service + Typer CLI. DuckDB and PostgreSQL
persistence with byte-identical reports across both backends. Five
report types × four formats. Public-proxy decision-profile system
with explicit customer-boundary contracts in every report.

## What this product answers

> **Given a realistic builder-style benchmark profile, which decoder
> path should be preferred under those assumptions, and why?**

The workbench is the evaluation harness around QEC decoders, not a
decoder itself. Drop in a Stim circuit, a `.dem` file, a syndrome
bundle, or a built-in public profile. Pick decoder paths to compare
— PyMatching baseline, correlated MWPM, NVIDIA Ising Fast / Accurate
+ PyMatching, ONNX-validation, TensorRT. The workbench measures
logical error rate with bootstrap CIs, per-round latency p50/p95/p99,
throughput, residual syndrome density, export status, and runtime
compatibility, then emits a decision-grade report saying: under this
profile, path X is preferred, here are the trade-offs, here is what
exported cleanly, **and here is what remains blocked without
customer-private data**.

---

## Real measured numbers (single RTX PRO 6000 Blackwell, CUDA 13)

Loading the shipped Apache-2.0 NVIDIA Ising checkpoints through the
vendor's own `WORKFLOW=inference` path on the canonical public config
(d=7, rounds=7, 25-parameter circuit-level noise, 262 144 shots/basis):

| decoder | LER (Avg X/Z) | µs/round | params | wall |
|---|---|---|---|---|
| PyMatching baseline (no pre-decoder) | 0.002777 | 0.766 | — | 23 s |
| **Ising Fast (RF=9) + PyMatching** | **0.002256** (−19 %) | 0.428 | 912 772 | 23 s |
| **Ising Accurate (RF=13) + PyMatching** | **0.001348** (−51 %) | 0.403 | 1 797 764 | 26 s |

Plus a 63-cell PyMatching-only sweep (3 distances × 7 p-values × 3
variants) and a 24-cell sinter sweep — full numbers in the rendered
decision reports.

---

## Requirements

### Hardware
- **Tier 1 (CPU only)**: any 64-bit Linux box, ≥ 8 GB RAM, ~5 GB disk.
- **Tier 3 (optional GPU)**: NVIDIA GPU with CUDA 13 support
  (Hopper / Ada / Blackwell), driver ≥ 580.x. The Ising/TensorRT/
  ONNX-runtime paths require this.
- macOS / Windows are not supported as primary targets — the bootstrap
  scripts and reproducibility numbers are pinned to Ubuntu 24.04.

### Software
- **Ubuntu 22.04 / 24.04** (other distros work but are untested).
- **Python 3.12** (the `pyproject.toml` pins `>=3.12,<3.13`). Earlier
  Python 3.x versions will fail the install.
- **git** with **git-lfs** installed (needed to fetch the NVIDIA Ising
  checkpoints from the vendor repo). On Ubuntu:
  ```bash
  sudo apt-get install -y git git-lfs
  git lfs install
  ```
- **bash** (default on Ubuntu).
- *Optional:* **Docker** (only for the published container images).
- *Optional:* an NVIDIA driver + CUDA 13 stack (only for Tier 3).

### Python dependency surface

The locked dependency stack is in `requirements/`:
- `requirements/core.txt` + `core.lock.txt` — Tier 1 CPU-only
  (Stim, PyMatching, sinter, ldpc, beliefmatching, qecsim, FastAPI,
  Typer, SQLAlchemy, DuckDB, Alembic, pydantic, Jinja2, reportlab,
  pytest, ruff, mypy, plus everything they pull in). 115 packages.
- `requirements/gpu-cu13.txt` + `gpu-cu13.lock.txt` — Tier 3 GPU
  (torch 2.11+cu130, onnxruntime-gpu 1.24, tensorrt-cu13 10.16,
  nvidia-modelopt, cuquantum-python-cu13, cudaq, cudaq-qec,
  cupy-cuda13x).

Lock files are tracked and `pip install` resolves with hashes for
reproducibility. **Do not edit lock files by hand**; if you need a new
dep, add it to the corresponding `*.txt` and rerun the bootstrap.

---

## Setup — clone to first green pytest

```bash
# 1. Clone
git clone https://github.com/wseltd/quantum-decoderops-workbench.git
cd quantum-decoderops-workbench

# 2. Tier 1 — core CPU stack (creates .venv/ and installs ~115 packages)
bash scripts/bootstrap_core_env.sh

# 3. Tier 2 — fetch the Apache-2.0 NVIDIA Ising checkpoints (~11 MB via git-lfs)
bash scripts/fetch_ising_assets.sh

# 4. (Optional) Tier 3 — GPU runtime (skip if no NVIDIA GPU)
bash scripts/bootstrap_gpu_runtime_env.sh

# 5. Verify the install
bash scripts/verify_install.sh
# expect: OVERALL STATUS: READY

# 6. Run the test suite
.venv/bin/pytest -q
# expect: 680 passed, 0 skipped (Tier-1 + Tier-3 paths all green)
```

If any of the optional Tier-3 packages aren't installed, the suite
still passes — those tests skip with structured reasons rather than
fail. To see what skipped, run with `-rs`.

For a fully orchestrated single-shot install:

```bash
bash scripts/bootstrap_all.sh --with-gpu
```

---

## Running the product

### Start the FastAPI service

```bash
.venv/bin/uvicorn 'app.api.main:app' --host 127.0.0.1 --port 8000
```

13 endpoints live at `/health`, `/seed`, `/ingest/dem`,
`/ingest/syndrome`, `/benchmark/run`, `/benchmark/{run_id}`, `/runs`,
`/metrics/{run_id}`, `/artifacts/{run_id}`, `/export/onnx`,
`/reports/generate`, `/reports/{run_id}`, `/evidence/latest`. Plus
profile endpoints at `/profiles`, `/profiles/{id}`,
`/profiles/{id}/run`, `/decisions/{run_id}`. Full schema at
`/openapi.json`, Swagger UI at `/docs`.

### Use the CLI

```bash
# Health probe
.venv/bin/decoderops health

# Worker-seed derivation
.venv/bin/decoderops seed --seed 42 --num-workers 4

# List built-in public-proxy decision profiles
.venv/bin/decoderops list-profiles

# Inspect one
.venv/bin/decoderops show-profile generic_surface_code_readiness

# Run it (deterministic, byte-reproducible reports)
.venv/bin/decoderops run-profile generic_surface_code_readiness \
    --num-shots 256 --basis X
```

### Reproduce the headline Ising numbers

These require the Ising checkpoints (step 3 above) and a CUDA-13 GPU
(step 4):

```bash
# Real Ising Fast inference at d=7 r=7 25-parameter noise, 262,144 shots/basis
CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE=0 \
    PREDECODER_PYTHON=$(pwd)/.venv/bin/python \
    EXPERIMENT_NAME=demo_fast \
WORKFLOW=inference bash vendor/Ising-Decoding/code/scripts/local_run.sh

# Real Ising Accurate inference (override model_id=4 for the RF=13 architecture)
CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE=0 \
    PREDECODER_PYTHON=$(pwd)/.venv/bin/python \
    EXPERIMENT_NAME=demo_accurate \
    EXTRA_PARAMS="model_id=4" \
WORKFLOW=inference bash vendor/Ising-Decoding/code/scripts/local_run.sh
```

Expected output is the LER table from the README ("Real measured
numbers" section) on a similar Blackwell / Hopper / Ada GPU.

The same paths are wired into the Python API:

```python
from app.benchmarking.ising_local_run_inference import run_inference

rec = run_inference(
    vendor_root=Path("vendor/Ising-Decoding"),
    work_dir=Path("/tmp/ising_run"),
    model_variant="accurate",  # or "fast"
    cuda_visible_devices="0",
)
```

---

## Architecture

Seven layers, one deterministic data flow:

```
ingestion → benchmarking → decoders → metrics → packaging → reports
                                                     │
                                                     ▼
                                                api / cli (cross-cutting)
```

Full breakdown in [`docs/architecture.md`](docs/architecture.md). One-line per layer:

| layer | role |
|---|---|
| `app/ingestion` | parse `.stim` / `.dem` / `.npy` / sinter logs into a typed `NormalisedInput` envelope |
| `app/benchmarking` | sweep + run + Sinter integration + Ising vendor subprocess wrappers |
| `app/decoders` | five backends behind one `Decoder` Protocol |
| `app/metrics` | LER w/ bootstrap CIs, residual density, latency, throughput, export status, compatibility |
| `app/packaging` | byte-reproducible content-addressed tarballs + manifest + offline verifier |
| `app/reports` | 5 templates × 4 formats |
| `app/api`, `app/cli`, `app/profiles`, `app/db` | service surfaces + persistence |

---

## Public-proxy decision profiles

Four built-in profiles ship — each grounded in primary-source URLs
per pinned parameter:

- `generic_surface_code_readiness` — Stim + PyMatching open baseline.
- `superconducting_latency_aware` — Willow timing (1.1 µs cycle,
  63 µs decoder envelope, arXiv:2408.13687), Riverlane Deltaflow 2
  (<20 µs), NVIDIA Ising 25-param noise.
- `ai_predecoder_export_runtime` — full vendor ONNX/TRT export +
  CUDA-Q Realtime artefact path.
- `trapped_ion_looser_latency` — caution-tagged proxy-only.

Each profile carries a typed `CustomerBoundary` declaring what the
public data CAN conclude vs what requires customer-private inputs.
Rendered verbatim in every decision report.

---

## Docker

```bash
# Tier 1 CPU-only workbench service
docker build -f docker/Dockerfile.workbench -t decoderops-workbench .

# Reproducible benchmark runner
docker build -f docker/Dockerfile.runner -t decoderops-runner .
```

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — seven-layer architecture + reproducibility fingerprint fields.
- [`docs/stack_decisions.md`](docs/stack_decisions.md) — chosen libraries and versions.
- [`docs/api.md`](docs/api.md) — 13 mandatory HTTP endpoints + the profile endpoints.
- [`docs/cli.md`](docs/cli.md) — Typer subcommands.
- [`docs/report_schemas.md`](docs/report_schemas.md) — 5-report × 4-format matrix.
- [`docs/licensing.md`](docs/licensing.md) — licenses, redistribution posture, the TensorRT / cudaq-qec proprietary-component carve-outs.
- [`docs/caveats.md`](docs/caveats.md) — known limitations and the customer-data boundary.
- [`research-and-resourced.md`](research-and-resourced.md) — full research pack, primary-source URLs cited per profile.
- [`nvidia-ising-calibration.md`](nvidia-ising-calibration.md) — NVIDIA Ising technical reference.

---

## Test status

- **680 passed, 0 skipped** on a CUDA-13 Blackwell + Postgres-via-
  testcontainers + duckdb_engine 0.17 + psycopg v3 host.
- Tests covering: ingestion parsers, decoder protocol + each backend,
  benchmarking sweep + runner + parallel pool, sinter integration,
  vendor subprocess wrappers, every metric, packaging (manifest,
  SHA-stamp, tarball, offline verify, ONNX export, TRT engine meta,
  cudaq export), DB models + repositories + dual-backend parity,
  reports + decision report, FastAPI 13-endpoint surface, CLI seven
  subcommands, profile schema + registry + loader + decision
  generator + runner + reports + API + CLI, integration E2E (seeded
  PyMatching, real Ising checkpoint, real generate_test_data.py, real
  local_run.sh, real Postgres 16, real TensorRT engine build).

```bash
.venv/bin/pytest -q --tb=no
```

---

## What's intentionally out of scope

- Customer-private detector error models, real syndrome traces,
  hardware-specific calibration data.
- Multi-tenant auth / cookie-based sessions / public-internet exposure
  (this is a local-first product; customers front-end with their own
  auth gateway).
- Persistent job queue / multi-host orchestration.
- Bundling proprietary components: TensorRT SDK is governed by an
  NVIDIA SLA; CUDA-QX QEC's `libcudaq-qec-nv-qldpc-decoder.so` is
  closed-source under an NVIDIA Software Licence Agreement. Both must
  be installed by the operator at deployment time.

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).

This is a public-data proof. Production deployment claims require
customer-private detector error models, real syndrome traces, and
target runtime constraints — explicitly out of scope for the public
release.

If you're a quantum hardware team — superconducting, trapped-ion,
silicon-spin, or photonic — and you'd like to test this against a
real (anonymised is fine) DEM or syndrome trace, contact the
maintainer.
