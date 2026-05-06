"""`decoderops list-profiles` command."""

from __future__ import annotations

import json

import typer

from app.cli.client import DecoderOpsClient, DecoderOpsClientError

__all__ = ["list_profiles"]


def list_profiles(
    api_url: str | None = typer.Option(None, "--api-url", envvar="DECODEROPS_API_URL"),
) -> None:
    client = DecoderOpsClient(base_url=api_url)
    try:
        response = client.request("GET", "/profiles")
    except DecoderOpsClientError as e:
        typer.echo(f"decoderops list-profiles: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(json.dumps(response, indent=2, sort_keys=True))
