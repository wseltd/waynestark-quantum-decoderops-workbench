# Architecture — Quantum DecoderOps Workbench

A local-first, on-prem-first, vendor-neutral Python workbench for benchmarking
QEC decoder pipelines. Seven layers, one deterministic data flow.

```
+-----------+   +---------------+   +-----------+   +---------+   +-----------+   +---------+
| Ingestion | ->| Benchmarking  | ->| Decoders  | ->| Metrics | ->| Packaging | ->| Reports |
+-----------+   +---------------+   +-----------+   +---------+   +-----------+   +---------+
                      ^                                                             |
                      |                                                             v
                   +-----+   +---------+                                         +------+
                   | API | <-|  CLI    | <-------------------------------------->| docs |
                   +-----+   +---------+                                         +------+
```

## 1. Ingestion Layer (`app/ingestion`)
- **Responsibility**: parse and normalise every input source into a single
  Pydantic envelope (`NormalisedInput`) that downstream layers consume.
- **Key modules**: `schema.py`, `stim_circuit.py`, `dem_parser.py`,
  `syndrome_parser.py`, `sinter_logs.py`, `customer_dem.py`.
- **Inputs**: raw Stim programs (strings), `.dem` files, `.npy`/`.bin`
  syndrome arrays, Sinter shot logs, customer DEM bundles.
- **Outputs**: `NormalisedInput` instances with stable SHA256 provenance.
- **Boundary**: upstream layers never see raw files; downstream layers never
  see Stim-specific types beyond `NormalisedInput`.

## 2. Experiment and Benchmark Runner (`app/benchmarking`)
- **Responsibility**: expand parameter sweeps, drive the decoder stream,
  and capture timing/accuracy for one or many `RunConfig`s.
- **Key modules**: `sweep.py`, `orchestrator.py`, `runner.py`,
  `parallel.py`, `sinter_integration.py`, `ising_subprocess.py`,
  `ising_local_run_inference.py`, `ising_local_run_onnx.py`,
  `generate_test_data.py`.
- **Inputs**: `SweepSpec`, a `decoder_factory` callable.
- **Outputs**: `list[RunResult]` per sweep.
- **Boundary**: runner never writes to the DB; persistence is the
  repository layer's job.

## 3. Decoder Execution Layer (`app/decoders`)
- **Responsibility**: expose the `Decoder` protocol across five backends
  and one capability detector. The backends are:
  `pymatching_baseline`, `ising_fast`, `ising_accurate`,
  `onnx_validation`, `tensorrt_optional`.
- **Key modules**: `protocol.py`, `pymatching_baseline.py`,
  `ising_fast.py`, `ising_accurate.py`, `_ising_common.py`,
  `onnx_validation.py`, `tensorrt_adapter.py`, `cudaq_capability.py`,
  `cudaq_qec_capability.py`, `cuquantum_capability.py`, `registry.py`.
- **Boundary**: every decoder returns `Corrections` with an explicit
  latency field; capability is reported via `CapabilityReport`.

## 4. Metrics and Evaluation Layer (`app/metrics`)
- **Responsibility**: compute logical error rate with bootstrap CIs,
  residual syndrome density, latency percentiles, throughput, per-format
  export status, runtime compatibility, and the aggregate `RunMetrics`.
- **Key modules**: `logical_error_rate.py`, `residual_syndrome.py`,
  `latency.py`, `throughput.py`, `export_status.py`, `compatibility.py`,
  `aggregate.py`.
- **Boundary**: pure functions; no I/O, no DB, no time measurement.

## 5. Artefact Packaging Layer (`app/packaging`)
- **Responsibility**: schema-versioned manifest writing, SHA256 stamping,
  byte-reproducible tarball construction, CUDA-Q QEC export wrapping,
  ONNX export + sidecars, TensorRT engine metadata, and offline tarball
  verification.
- **Key modules**: `manifest.py`, `sha256_stamp.py`, `tarball.py`,
  `cudaq_export.py`, `onnx_export.py`, `tensorrt_engine_meta.py`,
  `verify.py`.
- **Boundary**: outputs are content-addressed and byte-reproducible.

## 6. Reporting Layer (`app/reports`)
- **Responsibility**: render the 5-report × 4-format matrix —
  `engineering_benchmark`, `decoder_comparison`, `deployment_readiness`,
  `artefact_manifest`, `risk_caveat` — across Markdown, HTML, JSON, and
  (optionally) a deterministic reportlab PDF.
- **Key modules**: `templates/*.j2`, `markdown_renderer.py`,
  `html_renderer.py`, `pdf_renderer.py`, `json_renderer.py`,
  `context.py`, `pipeline.py`, `compatibility_matrix.py`,
  `risk_register.py`.
- **Boundary**: renders are pure functions of the supplied context.

## 7. Local API and CLI (`app/api`, `app/cli`)
- **Responsibility**: surface the workbench as a FastAPI service
  (12+ routers) and a Typer CLI (7 subcommands). The CLI calls the local
  API over HTTPX so the two surfaces cannot drift.
- **Key modules**: `app/api/app.py`, `app/api/routers/*`,
  `app/cli/client.py`, `app/cli/main.py`, `app/cli/commands/*`.

## Persistence
- **DuckDB** is the local-first default (file-backed, single-process).
- **PostgreSQL** is the service-backed option (Alembic-managed schema).
- Both are accessed through SQLAlchemy 2.x + a repository-pattern layer
  in `app/db/repositories/`. ORM models live in `app/models/`.

## Reproducibility Fingerprint
Every run records a `ReproducibilityFingerprint` containing:
`git_sha`, `git_dirty`, `pip_freeze_digest`, `config_hash`,
`rng_master_seed`, `worker_seeds`, `cpu_model`, `cpu_count`,
`gpu_models`, `gpu_count`, `nvidia_driver_version`, `os_name`,
`os_kernel`, `python_version`, `cuda_runtime_version`, `timestamp_utc`.
Wall-clock is the only nondeterministic field — injected for tests.

## References
- `research-and-resourced.md` — product direction, artefact matrix, licensing.
- `nvidia-ising-calibration.md` — NVIDIA Ising technical reference.
