"""`decoderops generate-report` command (T111)."""

from __future__ import annotations

import json

import typer

from app.cli.client import DecoderOpsClient, DecoderOpsClientError

__all__ = ["generate_report", "register"]


def generate_report(
    run_id: str = typer.Option(..., "--run-id"),
    include_pdf: bool = typer.Option(False, "--pdf/--no-pdf"),
    api_url: str | None = typer.Option(
        None, "--api-url", envvar="DECODEROPS_API_URL"
    ),
) -> None:
    client = DecoderOpsClient(base_url=api_url)
    try:
        response = client.request(
            "POST",
            "/reports/generate",
            json={"run_id": run_id, "include_pdf": include_pdf},
        )
    except DecoderOpsClientError as e:
        typer.echo(f"decoderops generate-report: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(json.dumps(response, indent=2, sort_keys=True))


def register(app: typer.Typer) -> None:
    app.command(name="generate-report")(generate_report)
