"""Structured logging setup — JSON file output with rotation for production."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog


def setup_logging(
    log_dir: str | None = None,
    level: str = "INFO",
    json_output: bool | None = None,
) -> None:
    """Configure structlog with console and optional file output.

    In production (ALPHAVEDHA_ENV=production), logs go to JSON files
    with rotation. In development, logs go to console with colors.
    """
    env = os.environ.get("ALPHAVEDHA_ENV", "development")
    if json_output is None:
        json_output = env == "production"
    if log_dir is None:
        log_dir = os.environ.get("ALPHAVEDHA_LOG_DIR", "logs")

    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    handlers.append(console_handler)

    if json_output:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path / "alphavedha.log",
            maxBytes=50 * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

        error_handler = RotatingFileHandler(
            log_path / "alphavedha-error.log",
            maxBytes=20 * 1024 * 1024,
            backupCount=3,
        )
        error_handler.setLevel(logging.ERROR)
        handlers.append(error_handler)

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
        force=True,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    for handler in handlers:
        handler.setFormatter(formatter)
