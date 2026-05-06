# API Reference

All endpoints are served by the FastAPI app created via
`app.api.app.create_app()`. Swagger UI is available at `/docs` and the
OpenAPI schema at `/openapi.json`.

## Required endpoints (13)

- `GET  /health` — liveness probe. Returns `{status, db_backend, version}`.
- `POST /seed` — derive worker seeds from a master seed.
- `POST /ingest/dem` — accept a DEM file path.
- `POST /ingest/syndrome` — accept a syndrome file path.
- `POST /benchmark/run` — expand a SweepSpec and return RunConfigs.
- `GET  /benchmark/{run_id}` — fetch a benchmark run by id.
- `GET  /runs` — list persisted runs.
- `GET  /metrics/{run_id}` — fetch metrics for a run.
- `GET  /artifacts/{run_id}` — list artefacts for a run.
- `POST /export/onnx` — request an ONNX export.
- `POST /reports/generate` — render the 5-report × 4-format matrix.
- `GET  /reports/{run_id}` — list rendered report rows for a run.
- `GET  /evidence/latest` — evidence bundle for the most recent run.

## Additional (additive, not required)

- `GET  /runs/{run_id}` — fetch one run by id (convenience).
- `GET  /evidence/{run_id}` — evidence bundle for a specific run (alias).

## Error responses

RFC-7807 problem shape: `{type, title, status, detail, run_id}`. The
catch-all handler returns HTTP 500 without leaking server tracebacks.
