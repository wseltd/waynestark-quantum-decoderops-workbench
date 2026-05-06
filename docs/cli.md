# CLI Reference

The `decoderops` CLI is a Typer application that calls the local FastAPI
service over HTTPX. Seven subcommands:

- `decoderops health` — GET /health and print the response.
- `decoderops seed --seed N --num-workers K` — derive worker seeds.
- `decoderops run-benchmark --config PATH [--wait] [--output-dir DIR]` —
  POST /benchmark/run, persist response, print run_id.
- `decoderops export-onnx --run-id R --variant ising_fast|ising_accurate` —
  request an ONNX export.
- `decoderops generate-report --run-id R [--pdf]` — POST /reports/generate,
  render the 4-format report matrix.
- `decoderops show-run --run-id R` — GET /runs/{run_id}.
- `decoderops list-runs [--limit N] [--offset K]` — GET /runs.

Exit codes:
- 0 — success
- 1 — HTTP error (non-2xx)
- 2 — network/connection error
- 3 — --wait polling timeout

The base URL is configurable via `DECODEROPS_API_BASE_URL` (for health /
seed) or `DECODEROPS_API_URL` (for everything else). Defaults to
`http://127.0.0.1:8000`.
