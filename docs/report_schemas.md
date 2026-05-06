# Report Schemas

Every run produces a 5-report × 4-format matrix. The five report types:

1. **engineering_benchmark** — full LER/latency/throughput table across
   decoders and sweep axes.
2. **decoder_comparison** — decoder-vs-decoder side-by-side.
3. **deployment_readiness** — runtime compatibility + risk register.
4. **artefact_manifest** — every artefact emitted by the run, with SHA256.
5. **risk_caveat** — explicit blocker list, honest and precise.

Formats:
- `markdown` (`.md`) — via Jinja2 templates in `app/reports/templates/`.
- `html` (`.html`) — autoescaped Jinja2 HTML.
- `json` (`.json`) — canonical (`sort_keys=True, separators=(",",":")`).
- `pdf` (`.pdf`) — deterministic reportlab output with frozen metadata.

## Context fields
- `run_id`, `git_sha`, `config_sha256`, `pip_freeze_digest`,
  `rng_master_seed`, `started_at_utc`, `finished_at_utc`.
- `host.{cpu_model, cpu_count, gpu_model, gpu_count, driver_version, cuda_runtime_version, os_kernel, python_version}`.
- `decoders[]`, `sweep_axes{}`, `shots_total`, `metrics[]`, `artefacts[]`,
  `reproducibility_fingerprint_sha256`.

## Byte-reproducibility
- Markdown/HTML/JSON are deterministic for a fixed context.
- PDF determinism requires a frozen timestamp and no network access
  during rendering.
