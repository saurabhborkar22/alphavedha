"""Email alerting for critical events — scheduler failures, drift, accuracy drops."""

from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from enum import StrEnum

import structlog

logger = structlog.get_logger(__name__)


class AlertLevel(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class AlertConfig:
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    recipient_email: str = ""
    enabled: bool = False

    @classmethod
    def from_env(cls) -> AlertConfig:
        sender = os.environ.get("ALERT_SMTP_SENDER", "")
        return cls(
            smtp_host=os.environ.get("ALERT_SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.environ.get("ALERT_SMTP_PORT", "587")),
            sender_email=sender,
            sender_password=os.environ.get("ALERT_SMTP_PASSWORD", ""),
            recipient_email=os.environ.get("ALERT_SMTP_RECIPIENT", sender),
            enabled=os.environ.get("ALERT_EMAIL_ENABLED", "").lower() in ("1", "true", "yes"),
        )


class EmailAlerter:
    """Sends email alerts for critical system events."""

    def __init__(self, config: AlertConfig | None = None) -> None:
        self._config = config or AlertConfig.from_env()

    @property
    def enabled(self) -> bool:
        return self._config.enabled and bool(self._config.sender_email)

    def send(self, subject: str, body: str, level: AlertLevel = AlertLevel.WARNING) -> bool:
        if not self.enabled:
            logger.debug("alert_skipped_disabled", subject=subject)
            return False

        msg = EmailMessage()
        msg["Subject"] = f"[AlphaVedha {level.value}] {subject}"
        msg["From"] = self._config.sender_email
        msg["To"] = self._config.recipient_email
        msg.set_content(body)

        try:
            with smtplib.SMTP(self._config.smtp_host, self._config.smtp_port) as server:
                server.starttls()
                server.login(self._config.sender_email, self._config.sender_password)
                server.send_message(msg)
            logger.info("alert_sent", subject=subject, level=level.value)
            return True
        except Exception as e:
            logger.error("alert_send_failed", subject=subject, error=str(e))
            return False

    def scheduler_job_failed(self, job_name: str, error: str, timestamp: datetime) -> bool:
        return self.send(
            subject=f"Scheduler job failed: {job_name}",
            body=(f"Job: {job_name}\nTime: {timestamp.isoformat()}\nError: {error}\n"),
            level=AlertLevel.CRITICAL,
        )

    def drift_detected(self, feature_group: str, psi_value: float) -> bool:
        return self.send(
            subject=f"Feature drift detected: {feature_group}",
            body=(
                f"Feature group: {feature_group}\n"
                f"PSI value: {psi_value:.4f}\n"
                f"Threshold: 0.2\n"
                f"Action: Consider retraining the model.\n"
            ),
            level=AlertLevel.WARNING,
        )

    def accuracy_drop(self, window: str, accuracy: float, threshold: float) -> bool:
        return self.send(
            subject=f"Prediction accuracy below threshold ({window})",
            body=(
                f"Rolling window: {window}\n"
                f"Current accuracy: {accuracy:.2%}\n"
                f"Threshold: {threshold:.2%}\n"
                f"Action: Review model performance and consider retraining.\n"
            ),
            level=AlertLevel.WARNING,
        )

    def api_error_spike(self, error_count: int, window_minutes: int) -> bool:
        return self.send(
            subject=f"API error spike: {error_count} errors in {window_minutes}min",
            body=(
                f"Error count: {error_count}\n"
                f"Window: last {window_minutes} minutes\n"
                f"Action: Check API logs for details.\n"
            ),
            level=AlertLevel.CRITICAL,
        )
