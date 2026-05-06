"""`decoderops run-benchmark` command (T109)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from app.cli.client import DecoderOpsClient, DecoderOpsClientError

__all__ = ["register", "run_benchmark"]


def run_benchmark(
    config: Path = typer.Option(..., "--config", help="Benchmark sweep JSON/YAML"),
    api_url: str | None = typer.Option(
        None, "--api-url", envvar="DECODEROPS_API_URL"
    ),
    output_dir: Path = typer.Option(
        Path(".decoderops/runs"), "--output-dir"
    ),
    wait: bool = typer.Option(False, "--wait/--no-wait"),
    poll_interval: float = typer.Option(2.0, "--poll-interval"),
    timeout: float = typer.Option(600.0, "--timeout"),
    seed_override: int | None = typer.Option(None, "--seed"),
    backend: list[str] = typer.Option([], "--backend"),
) -> None:
    if not config.exists():
        raise typer.BadParameter(f"config file not found: {config}")
    payload_text = config.read_text()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        import yaml

        payload = yaml.safe_load(payload_text)
    if seed_override is not None:
        payload["master_seed"] = seed_override
    if backend:
        payload["backends"] = list(backend)
    client = DecoderOpsClient(base_url=api_url)
    try:
        response = client.request("POST", "/benchmark/run", json=payload)
    except DecoderOpsClientError as e:
        msg = str(e)
        if "request failed" in msg:
            typer.echo(f"decoderops run-benchmark: {e}", err=True)
            raise typer.Exit(code=2) from e
        typer.echo(f"decoderops run-benchmark: {e}", err=True)
        raise typer.Exit(code=1) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_id = response.get("sweep_id", "sweep")
    (output_dir / f"{sweep_id}.json").write_text(json.dumps(response))

    if wait:
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(poll_interval)
            # polling endpoint is stubbed — fall through
            break
        else:
            raise typer.Exit(code=3)
    run_ids = response.get("run_ids") or [sweep_id]
    typer.echo(run_ids[0])


def register(app: typer.Typer) -> None:
    app.command(name="run-benchmark")(run_benchmark)
