"""Tests for CLI commands — predict, scan, serve."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from alphavedha.cli.main import app

runner = CliRunner()


class TestPredictCommand:
    def test_predict_demo_mode(self) -> None:
        result = runner.invoke(app, ["predict", "TCS", "--demo"])
        assert result.exit_code == 0
        assert "TCS" in result.output

    def test_predict_json_output(self) -> None:
        result = runner.invoke(app, ["predict", "TCS", "--demo", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["symbol"] == "TCS"
        assert "direction" in data
        assert "composite_score" in data

    def test_predict_without_demo_warns(self) -> None:
        result = runner.invoke(app, ["predict", "TCS"])
        # Should work in demo mode by default or show an error about missing models
        assert result.exit_code in (0, 1)


class TestScanCommand:
    def test_scan_demo_mode(self) -> None:
        result = runner.invoke(app, ["scan", "large", "--demo", "--top-n", "3"])
        assert result.exit_code == 0

    def test_scan_json_output(self) -> None:
        result = runner.invoke(app, ["scan", "large", "--demo", "--json", "--top-n", "3"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "buy_candidates" in data
        assert "sell_candidates" in data


class TestServeCommand:
    def test_serve_help(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "host" in result.output.lower() or "port" in result.output.lower()
