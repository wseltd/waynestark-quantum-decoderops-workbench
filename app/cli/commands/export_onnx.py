"""`decoderops export-onnx` command (T110)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from app.cli.client import DecoderOpsClient, DecoderOpsClientError

__all__ = ["export_onnx", "register"]


def export_onnx(
    run_id: str = typer.Option(..., "--run-id"),
    variant: str = typer.Option(..., "--variant"),
    workflow: int = typer.Option(1, "--workflow"),
    quant_format: str = typer.Option("none", "--quant-format"),
    output_dir: Path = typer.Option(
        Path(".decoderops/exports"), "--output-dir"
    ),
    api_url: str | None = typer.Option(
        None, "--api-url", envvar="DECODEROPS_API_URL"
    ),
) -> None:
    if variant not in ("ising_fast", "ising_accurate"):
        raise typer.BadParameter(
            f"--variant must be ising_fast or ising_accurate; got {variant!r}"
        )
    if workflow not in (1, 2):
        raise typer.BadParameter(
            f"--workflow must be 1 or 2; got {workflow}"
        )
    if quant_format not in ("none", "int8", "fp8"):
        raise typer.BadParameter(
            f"--quant-format must be none/int8/fp8; got {quant_format!r}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    request_payload = {
        "run_id": run_id,
        "output_path": str(output_dir / f"{run_id}.{variant}.onnx"),
    }
    client = DecoderOpsClient(base_url=api_url)
    try:
        response = client.request("POST", "/export/onnx", json=request_payload)
    except DecoderOpsClientError as e:
        typer.echo(f"decoderops export-onnx: {e}", err=True)
        raise typer.Exit(code=1) from e
    out_file = output_dir / f"{run_id}.{variant}.export.json"
    out_file.write_text(json.dumps(response))
    typer.echo(response.get("output_path", str(out_file)))


def register(app: typer.Typer) -> None:
    app.command(name="export-onnx")(export_onnx)
