"""Typer CLI entry point for decoderops (T106)."""

from __future__ import annotations

import typer

from app.cli.commands.export_onnx import export_onnx as _cmd_export_onnx
from app.cli.commands.generate_report import (
    generate_report as _cmd_generate_report,
)
from app.cli.commands.health import health as _cmd_health
from app.cli.commands.list_profiles import list_profiles as _cmd_list_profiles
from app.cli.commands.list_runs import list_runs as _cmd_list_runs
from app.cli.commands.run_benchmark import run_benchmark as _cmd_run_benchmark
from app.cli.commands.run_profile import run_profile as _cmd_run_profile
from app.cli.commands.seed import seed as _cmd_seed
from app.cli.commands.show_profile import show_profile as _cmd_show_profile
from app.cli.commands.show_run import show_run as _cmd_show_run

__all__ = ["app", "cli", "main"]


app: typer.Typer = typer.Typer(
    name="decoderops",
    help="Quantum DecoderOps Workbench CLI.",
    no_args_is_help=True,
    add_completion=False,
)

app.command(name="health")(_cmd_health)
app.command(name="seed")(_cmd_seed)
app.command(name="run-benchmark")(_cmd_run_benchmark)
app.command(name="export-onnx")(_cmd_export_onnx)
app.command(name="generate-report")(_cmd_generate_report)
app.command(name="show-run")(_cmd_show_run)
app.command(name="list-runs")(_cmd_list_runs)
app.command(name="list-profiles")(_cmd_list_profiles)
app.command(name="show-profile")(_cmd_show_profile)
app.command(name="run-profile")(_cmd_run_profile)

# Backwards-compat alias.
cli = app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
