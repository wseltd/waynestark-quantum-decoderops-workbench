"""`decoderops health` command (T107)."""

from __future__ import annotations

import json
import sys

import typer

from app.cli.client import DecoderOpsClient, DecoderOpsClientError

__all__ = ["health"]


def health(
    base_url: str | None = typer.Option(
        None, "--base-url", envvar="DECODEROPS_API_BASE_URL"
    ),
) -> None:
    """Check the API /health endpoint."""
    client = DecoderOpsClient(base_url=base_url)
    try:
        result = client.get_health()
    except DecoderOpsClientError as e:
        typer.echo(f"decoderops health: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
