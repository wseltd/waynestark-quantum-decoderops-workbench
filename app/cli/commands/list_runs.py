"""`decoderops list-runs` command (T113)."""

from __future__ import annotations

import json

import typer

from app.cli.client import DecoderOpsClient, DecoderOpsClientError

__all__ = ["list_runs"]


def list_runs(
    limit: int = typer.Option(100, "--limit"),
    offset: int = typer.Option(0, "--offset"),
    api_url: str | None = typer.Option(
        None, "--api-url", envvar="DECODEROPS_API_URL"
    ),
) -> None:
    client = DecoderOpsClient(base_url=api_url)
    try:
        response = client.request(
            "GET", "/runs", params={"limit": limit, "offset": offset}
        )
    except DecoderOpsClientError as e:
        typer.echo(f"decoderops list-runs: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(json.dumps(response, indent=2, sort_keys=True))
