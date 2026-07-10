"""Tests for email alerting."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from alphavedha.monitoring.alerts import (
    AlertConfig,
    AlertLevel,
    EmailAlerter,
    detect_direction_collapse,
)


class TestAlertConfig:
    def test_defaults(self) -> None:
        config = AlertConfig()
        assert config.smtp_host == "smtp.gmail.com"
        assert config.smtp_port == 587
        assert config.enabled is False

    def test_from_env(self) -> None:
        env = {
            "ALERT_SMTP_HOST": "smtp.test.com",
            "ALERT_SMTP_PORT": "465",
            "ALERT_SMTP_SENDER": "test@test.com",
            "ALERT_SMTP_PASSWORD": "secret",
            "ALERT_SMTP_RECIPIENT": "admin@test.com",
            "ALERT_EMAIL_ENABLED": "true",
        }
        with patch.dict("os.environ", env, clear=False):
            config = AlertConfig.from_env()
        assert config.smtp_host == "smtp.test.com"
        assert config.smtp_port == 465
        assert config.sender_email == "test@test.com"
        assert config.recipient_email == "admin@test.com"
        assert config.enabled is True

    def test_from_env_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            config = AlertConfig.from_env()
        assert config.smtp_host == "smtp.gmail.com"
        assert config.enabled is False
        assert config.sender_email == ""

    def test_from_env_recipient_defaults_to_sender(self) -> None:
        env = {"ALERT_SMTP_SENDER": "me@gmail.com"}
        with patch.dict("os.environ", env, clear=True):
            config = AlertConfig.from_env()
        assert config.recipient_email == "me@gmail.com"


class TestEmailAlerter:
    def test_disabled_by_default(self) -> None:
        alerter = EmailAlerter(AlertConfig())
        assert alerter.enabled is False

    def test_disabled_without_sender(self) -> None:
        config = AlertConfig(enabled=True, sender_email="")
        alerter = EmailAlerter(config)
        assert alerter.enabled is False

    def test_enabled_with_sender(self) -> None:
        config = AlertConfig(enabled=True, sender_email="test@test.com")
        alerter = EmailAlerter(config)
        assert alerter.enabled is True

    def test_send_skipped_when_disabled(self) -> None:
        alerter = EmailAlerter(AlertConfig())
        result = alerter.send("test", "body")
        assert result is False

    @patch("alphavedha.monitoring.alerts.smtplib.SMTP")
    def test_send_success(self, mock_smtp_cls: MagicMock) -> None:
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        config = AlertConfig(
            enabled=True,
            sender_email="sender@test.com",
            sender_password="pass",
            recipient_email="recip@test.com",
        )
        alerter = EmailAlerter(config)
        result = alerter.send("Test Subject", "Test Body", AlertLevel.WARNING)

        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("sender@test.com", "pass")
        mock_server.send_message.assert_called_once()

    @patch("alphavedha.monitoring.alerts.smtplib.SMTP")
    def test_send_failure(self, mock_smtp_cls: MagicMock) -> None:
        mock_smtp_cls.return_value.__enter__ = MagicMock(
            side_effect=ConnectionRefusedError("refused")
        )
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        config = AlertConfig(
            enabled=True,
            sender_email="sender@test.com",
            sender_password="pass",
        )
        alerter = EmailAlerter(config)
        result = alerter.send("Test", "Body")
        assert result is False

    def test_scheduler_job_failed(self) -> None:
        alerter = EmailAlerter(AlertConfig())
        result = alerter.scheduler_job_failed("daily_predictions", "timeout", datetime.now(UTC))
        assert result is False

    def test_drift_detected(self) -> None:
        alerter = EmailAlerter(AlertConfig())
        result = alerter.drift_detected("technical", 0.25)
        assert result is False

    def test_accuracy_drop(self) -> None:
        alerter = EmailAlerter(AlertConfig())
        result = alerter.accuracy_drop("30d", 0.48, 0.52)
        assert result is False

    def test_api_error_spike(self) -> None:
        alerter = EmailAlerter(AlertConfig())
        result = alerter.api_error_spike(50, 5)
        assert result is False

    def test_strategy_daily_summary_disabled(self) -> None:
        alerter = EmailAlerter(AlertConfig())
        result = alerter.strategy_daily_summary("report text", "2026-06-20")
        assert result is False


class TestDirectionCollapse:
    """The daily alarm that catches an all-one-way (degenerate) prediction cohort."""

    def test_detects_dominant_direction(self) -> None:
        # 48 of 50 short → 96% > 90% threshold.
        result = detect_direction_collapse([-1] * 48 + [1, 0])
        assert result is not None
        dominant, share = result
        assert dominant == -1
        assert share == pytest.approx(0.96)

    def test_mixed_directions_no_alarm(self) -> None:
        assert detect_direction_collapse([1] * 25 + [-1] * 25) is None

    def test_empty_no_alarm(self) -> None:
        assert detect_direction_collapse([]) is None

    def test_exactly_at_threshold_does_not_fire(self) -> None:
        # 90% is not > 90%.
        assert detect_direction_collapse([-1] * 90 + [1] * 10) is None

    def test_just_over_threshold_fires(self) -> None:
        result = detect_direction_collapse([-1] * 91 + [1] * 9)
        assert result is not None
        assert result[0] == -1

    def test_alert_method_disabled_returns_false(self) -> None:
        alerter = EmailAlerter(AlertConfig())
        assert alerter.direction_collapse(-1, 0.96, 50, "2026-07-11") is False

    def test_alert_method_builds_critical_message(self) -> None:
        alerter = EmailAlerter(AlertConfig())
        with patch.object(alerter, "send", return_value=True) as mock_send:
            alerter.direction_collapse(-1, 0.96, 50, "2026-07-11")
        mock_send.assert_called_once()
        kwargs = mock_send.call_args.kwargs
        assert kwargs["level"] == AlertLevel.CRITICAL
        assert "SELL/DOWN" in kwargs["subject"]
        assert "2026-07-11" in kwargs["subject"]
