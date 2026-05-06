"""`decoderops seed` command (T108)."""

from __future__ import annotations

import json

import typer

from app.cli.client import DecoderOpsClient, DecoderOpsClientError

__all__ = ["MAX_SEED", "seed"]

MAX_SEED: int = (1 << 63) - 1


def seed(
    seed: int = typer.Option(..., "--seed", help="RNG master seed"),
    num_workers: int = typer.Option(1, "--num-workers"),
    base_url: str | None = typer.Option(
        None, "--base-url", envvar="DECODEROPS_API_BASE_URL"
    ),
) -> None:
    """Derive worker seeds for a master seed via the API."""
    if seed < 0 or seed > MAX_SEED:
        raise typer.BadParameter(
            f"seed must be in [0, 2**63 - 1]; got {seed}"
        )
    client = DecoderOpsClient(base_url=base_url)
    try:
        result = client.post_seed(seed, num_workers=num_workers)
    except DecoderOpsClientError as e:
        typer.echo(f"decoderops seed: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
