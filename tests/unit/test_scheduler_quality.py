from __future__ import annotations


def test_scheduler_has_quality_check_method() -> None:
    from alphavedha.scheduler import AlphaVedhaScheduler

    scheduler = object.__new__(AlphaVedhaScheduler)
    assert hasattr(scheduler, "run_quality_check")
    assert callable(scheduler.run_quality_check)


def test_quality_check_cli_command_exists() -> None:
    from typer.testing import CliRunner

    from alphavedha.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["data", "quality-check", "--help"])
    assert result.exit_code == 0
    assert "quality" in result.output.lower() or "check" in result.output.lower()
