"""`decoderops run-profile` command — execute a public proxy profile."""

from __future__ import annotations

import json

import typer

from app.cli.client import DecoderOpsClient, DecoderOpsClientError

__all__ = ["run_profile"]


def run_profile(
    profile_id: str = typer.Argument(..., help="profile_id to run"),
    num_shots: int = typer.Option(512, "--num-shots"),
    master_seed: int = typer.Option(20260422, "--master-seed"),
    basis: list[str] = typer.Option(
        [], "--basis", help="Restrict to these bases; empty = all allowed"
    ),
    include_pdf: bool = typer.Option(False, "--pdf/--no-pdf"),
    api_url: str | None = typer.Option(None, "--api-url", envvar="DECODEROPS_API_URL"),
) -> None:
    client = DecoderOpsClient(base_url=api_url)
    payload = {
        "num_shots": num_shots,
        "master_seed": master_seed,
        "bases": basis or None,
        "include_pdf": include_pdf,
    }
    try:
        response = client.request("POST", f"/profiles/{profile_id}/run", json=payload)
    except DecoderOpsClientError as e:
        typer.echo(f"decoderops run-profile: {e}", err=True)
        raise typer.Exit(code=1) from e

    decision = response.get("decision", {})
    typer.echo(
        json.dumps(
            {
                "profile_id": response.get("profile_id"),
                "decision_run_id": response.get("decision_run_id"),
                "recommended_backend": decision.get("recommended_backend"),
                "recommendation_label": decision.get("recommendation_label"),
                "recommendation_reason": decision.get("recommendation_reason"),
                "blockers": decision.get("blockers", []),
                "pareto_dominated": decision.get("pareto_dominated", []),
                "manifest_path": response.get("manifest_path"),
                "rendered_reports": response.get("rendered_reports", []),
            },
            indent=2,
            sort_keys=True,
        )
    )
