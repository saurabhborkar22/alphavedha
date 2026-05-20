"""Tests for structured logging setup."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import structlog

from alphavedha.monitoring.logging import setup_logging


class TestSetupLogging:
    def test_dev_mode_defaults(self) -> None:
        with patch.dict("os.environ", {"ALPHAVEDHA_ENV": "development"}, clear=False):
            setup_logging()
        logger = structlog.get_logger("test")
        assert logger is not None

    def test_production_creates_log_files(self, tmp_path: Path) -> None:
        log_dir = str(tmp_path / "logs")
        with patch.dict("os.environ", {"ALPHAVEDHA_ENV": "production"}, clear=False):
            setup_logging(log_dir=log_dir, json_output=True)
        assert (tmp_path / "logs" / "alphavedha.log").exists()
        assert (tmp_path / "logs" / "alphavedha-error.log").exists()

    def test_custom_log_level(self) -> None:
        setup_logging(level="DEBUG", json_output=False)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_json_output_override(self, tmp_path: Path) -> None:
        log_dir = str(tmp_path / "logs")
        setup_logging(log_dir=log_dir, json_output=True)
        assert (tmp_path / "logs" / "alphavedha.log").exists()

    def test_console_only_no_files(self, tmp_path: Path) -> None:
        log_dir = str(tmp_path / "logs")
        setup_logging(log_dir=log_dir, json_output=False)
        assert not (tmp_path / "logs").exists()
