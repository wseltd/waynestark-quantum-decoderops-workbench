"""Consolidated CLI tests (T105-T113)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from app.cli.client import DecoderOpsClient, DecoderOpsClientError
from app.cli.main import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ------------- client -------------


def test_client_reads_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECODEROPS_API_BASE_URL", "http://127.0.0.1:9999")
    c = DecoderOpsClient()
    assert c.base_url == "http://127.0.0.1:9999"
    c.close()


def test_client_base_url_override_takes_precedence() -> None:
    c = DecoderOpsClient(base_url="http://explicit:80")
    assert c.base_url == "http://explicit:80"
    c.close()


def test_client_raises_decoderops_client_error_on_http_error() -> None:
    import httpx

    c = DecoderOpsClient(base_url="http://127.0.0.1:9")
    mock_resp = httpx.Response(500, text="err")
    with patch.object(
        c._client,
        "request",
        side_effect=httpx.HTTPStatusError(
            "500", request=httpx.Request("GET", "http://x/"), response=mock_resp
        ),
    ):
        with pytest.raises(DecoderOpsClientError):
            c.request("GET", "/health")


def test_client_raises_on_network_error() -> None:
    import httpx

    c = DecoderOpsClient(base_url="http://127.0.0.1:9")
    with patch.object(
        c._client,
        "request",
        side_effect=httpx.ConnectError("connect fail"),
    ):
        with pytest.raises(DecoderOpsClientError):
            c.request("GET", "/health")


# ------------- main -------------


def test_cli_main_registers_all_seven_commands() -> None:
    names = {c.name for c in app.registered_commands}
    assert {
        "health",
        "seed",
        "run-benchmark",
        "export-onnx",
        "generate-report",
        "show-run",
        "list-runs",
    }.issubset(names)


def test_cli_main_help_lists_every_command(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in (
        "health",
        "seed",
        "run-benchmark",
        "export-onnx",
        "generate-report",
        "show-run",
        "list-runs",
    ):
        assert name in result.stdout


# ------------- health -------------


def test_health_command_prints_json_on_success(runner: CliRunner) -> None:
    with patch(
        "app.cli.commands.health.DecoderOpsClient"
    ) as mock_client:
        instance = mock_client.return_value
        instance.get_health.return_value = {"status": "ok", "version": "1"}
        result = runner.invoke(app, ["health"])
    assert result.exit_code == 0
    assert '"status": "ok"' in result.stdout


def test_health_command_exits_one_on_client_error(runner: CliRunner) -> None:
    with patch(
        "app.cli.commands.health.DecoderOpsClient"
    ) as mock_client:
        instance = mock_client.return_value
        instance.get_health.side_effect = DecoderOpsClientError("boom")
        result = runner.invoke(app, ["health"])
    assert result.exit_code == 1


# ------------- seed -------------


def test_seed_command_requires_option(runner: CliRunner) -> None:
    result = runner.invoke(app, ["seed"])
    assert result.exit_code != 0


def test_seed_command_rejects_negative(runner: CliRunner) -> None:
    result = runner.invoke(app, ["seed", "--seed", "-1"])
    assert result.exit_code != 0


def test_seed_command_success(runner: CliRunner) -> None:
    with patch(
        "app.cli.commands.seed.DecoderOpsClient"
    ) as mock_client:
        instance = mock_client.return_value
        instance.post_seed.return_value = {"master_seed": 42, "worker_seeds": [1]}
        result = runner.invoke(app, ["seed", "--seed", "42"])
    assert result.exit_code == 0
    assert '"master_seed": 42' in result.stdout


# ------------- run-benchmark -------------


def test_run_benchmark_rejects_missing_config(
    runner: CliRunner, tmp_path
) -> None:
    result = runner.invoke(
        app, ["run-benchmark", "--config", str(tmp_path / "nope.json")]
    )
    assert result.exit_code != 0


def test_run_benchmark_success(runner: CliRunner, tmp_path) -> None:
    import json as _json

    cfg = tmp_path / "c.json"
    cfg.write_text(
        _json.dumps(
            {
                "distances": [3],
                "rounds": [3],
                "bases": ["X"],
                "p_errors": [1e-3],
                "backends": ["pymatching_baseline"],
                "num_shots": 16,
                "master_seed": 42,
            }
        )
    )
    with patch(
        "app.cli.commands.run_benchmark.DecoderOpsClient"
    ) as mock_client:
        instance = mock_client.return_value
        instance.request.return_value = {
            "sweep_id": "abc",
            "num_runs": 1,
            "run_ids": ["r-1"],
        }
        result = runner.invoke(
            app,
            [
                "run-benchmark",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
    assert result.exit_code == 0
    assert "r-1" in result.stdout


# ------------- export-onnx -------------


def test_export_onnx_rejects_invalid_variant(runner: CliRunner) -> None:
    result = runner.invoke(
        app,
        [
            "export-onnx",
            "--run-id",
            "r",
            "--variant",
            "not_a_variant",
        ],
    )
    assert result.exit_code != 0


def test_export_onnx_success(runner: CliRunner, tmp_path) -> None:
    with patch(
        "app.cli.commands.export_onnx.DecoderOpsClient"
    ) as mock_client:
        instance = mock_client.return_value
        instance.request.return_value = {
            "accepted": True,
            "output_path": "/tmp/x.onnx",
        }
        result = runner.invoke(
            app,
            [
                "export-onnx",
                "--run-id",
                "r",
                "--variant",
                "ising_fast",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
    assert result.exit_code == 0


# ------------- show-run / list-runs / generate-report -------------


def test_show_run_success(runner: CliRunner) -> None:
    with patch(
        "app.cli.commands.show_run.DecoderOpsClient"
    ) as mock_client:
        instance = mock_client.return_value
        instance.request.return_value = {"run_id": "r1", "status": "ok"}
        result = runner.invoke(app, ["show-run", "--run-id", "r1"])
    assert result.exit_code == 0


def test_list_runs_success(runner: CliRunner) -> None:
    with patch(
        "app.cli.commands.list_runs.DecoderOpsClient"
    ) as mock_client:
        instance = mock_client.return_value
        instance.request.return_value = []
        result = runner.invoke(app, ["list-runs"])
    assert result.exit_code == 0


def test_generate_report_success(runner: CliRunner) -> None:
    with patch(
        "app.cli.commands.generate_report.DecoderOpsClient"
    ) as mock_client:
        instance = mock_client.return_value
        instance.request.return_value = {"reports": []}
        result = runner.invoke(app, ["generate-report", "--run-id", "r1"])
    assert result.exit_code == 0
