"""`decoderops show-run` command (T112)."""

from __future__ import annotations

import json

import typer

from app.cli.client import DecoderOpsClient, DecoderOpsClientError

__all__ = ["show_run"]


def show_run(
    run_id: str = typer.Option(..., "--run-id"),
    api_url: str | None = typer.Option(
        None, "--api-url", envvar="DECODEROPS_API_URL"
    ),
) -> None:
    client = DecoderOpsClient(base_url=api_url)
    try:
        response = client.request("GET", f"/runs/{run_id}")
    except DecoderOpsClientError as e:
        typer.echo(f"decoderops show-run: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(json.dumps(response, indent=2, sort_keys=True))
