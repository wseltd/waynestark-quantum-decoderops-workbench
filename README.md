# Quantum DecoderOps Workbench

A local-first, on-prem-first, vendor-neutral Python workbench for
benchmarking quantum error correction decoder pipelines. Built around
Stim + PyMatching + real NVIDIA Ising-Decoding checkpoints (Apache-2.0).
FastAPI service + Typer CLI. DuckDB + PostgreSQL persistence. Five
report types × four formats. Public-proxy decision-profile system with
explicit customer-boundary contracts in every report.

## What this product answers

> **Given a realistic builder-style benchmark profile, which decoder
> path should be preferred under those assumptions, and why?**

The workbench is the evaluation harness around QEC decoders, not a
decoder itself. You drop in a Stim circuit, a `.dem`, a syndrome bundle,
or a built-in public profile. You pick decoder paths to compare —
PyMatching baseline, correlated MWPM, NVIDIA Ising Fast / Accurate
pre-decoder paths, ONNX-validation, TensorRT. The workbench measures
logical error rate with bootstrap CIs, per-round latency p50/p95/p99,
throughput, residual syndrome density, export status, and runtime
compatibility, and emits a decision-grade report that says: under this
profile, path X is preferred, here are the trade-offs, here is what
exported cleanly, **and here is what remains blocked without
customer-private data**.

## Quick start

```bash
# Install core dependencies
bash scripts/bootstrap_core_env.sh
source .venv/bin/activate

# Verify the install
bash scripts/verify_install.sh
# → OVERALL STATUS: READY

# Run the test suite
.venv/bin/pytest -q
```

## Start the service

```bash
.venv/bin/uvicorn 'app.api.main:app' --host 0.0.0.0 --port 8000
```

Then:

```bash
.venv/bin/decoderops health
.venv/bin/decoderops seed --seed 42 --num-workers 4
.venv/bin/decoderops list-profiles
.venv/bin/decoderops run-profile generic_surface_code_readiness \
    --num-shots 256 --basis X
```

## Real measured numbers (single-GPU, RTX PRO 6000)

Loading the shipped Apache-2.0 NVIDIA Ising checkpoints through the
vendor's own `WORKFLOW=inference` path on the canonical public config
(d=7, rounds=7, 25-parameter circuit-level noise, 262 144 shots/basis):

| decoder | LER (Avg X/Z) | µs/round | params |
|---|---|---|---|
| PyMatching baseline (no pre-decoder) | 0.002777 | 0.766 | — |
| **Ising Fast (RF=9) + PyMatching** | **0.002256** (−19 %) | 0.428 | 912 772 |
| **Ising Accurate (RF=13) + PyMatching** | **0.001348** (−51 %) | 0.403 | 1 797 764 |

Plus a 63-cell PyMatching-only sweep (3 distances × 7 p-values × 3
variants) and a 24-cell sinter sweep. Full numbers in
`docs/architecture.md` and the rendered decision reports.

## Architecture
See `docs/architecture.md`.

## Docker

```bash
# Tier 1 CPU-only workbench service
docker build -f docker/Dockerfile.workbench -t decoderops-workbench .

# Reproducible benchmark runner
docker build -f docker/Dockerfile.runner -t decoderops-runner .
```

## Documentation
- `docs/architecture.md` — seven-layer architecture.
- `docs/stack_decisions.md` — chosen libraries and versions.
- `docs/api.md` — 13 mandatory HTTP endpoints.
- `docs/cli.md` — Typer subcommands.
- `docs/report_schemas.md` — 5-report × 4-format matrix.
- `docs/licensing.md` — licenses and redistribution posture.
- `docs/caveats.md` — known limitations and customer-data boundary.

## License
Apache-2.0. See `LICENSE`.

## Status

Reference implementation. The public-proxy decision profiles are
research-grounded with primary-source URLs per pinned parameter, and
every report carries a typed `CustomerBoundary` declaring what public
data CAN conclude vs what requires customer-private inputs.

This is a public-data proof. Production deployment claims require the
customer's calibrated detector error model, real syndrome traces, and
target runtime constraints — explicitly out of scope for the public
release.
