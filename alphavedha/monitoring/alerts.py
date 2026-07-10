"""Email alerting for critical events — scheduler failures, drift, accuracy drops."""

from __future__ import annotations

import os
import smtplib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphavedha.data.quality import QualityReport

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


# A day's ensemble predictions collapsing to one direction is the signature of a
# degenerate model (the Jun-2026 all-short failure). Mirrors the training gate's
# _DEGENERATE_MAX_CLASS_SHARE so serving and training agree on "collapsed".
DIRECTION_COLLAPSE_THRESHOLD = 0.90


def detect_direction_collapse(
    directions: Sequence[int],
    threshold: float = DIRECTION_COLLAPSE_THRESHOLD,
) -> tuple[int, float] | None:
    """Return ``(dominant_direction, share)`` when one direction exceeds
    ``threshold`` of the day's predictions — the all-one-way signature of a
    collapsed model — else ``None``. Empty input returns ``None``.
    """
    dirs = [int(d) for d in directions]
    if not dirs:
        return None
    dominant, count = Counter(dirs).most_common(1)[0]
    share = count / len(dirs)
    if share > threshold:
        return dominant, float(share)
    return None


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

    def direction_collapse(self, dominant: int, share: float, n: int, date_str: str) -> bool:
        """Alert when a day's predictions collapse to one direction (degenerate model)."""
        label = {1: "BUY/UP", -1: "SELL/DOWN", 0: "HOLD/FLAT"}.get(dominant, str(dominant))
        return self.send(
            subject=f"Direction collapse: {share:.0%} of predictions are {label} ({date_str})",
            body=(
                f"Date: {date_str}\n"
                f"{share:.1%} of {n} predictions are direction {dominant} ({label}).\n"
                f"This is the all-one-way signature of a degenerate model "
                f"(the Jun-2026 all-short failure).\n"
                f"Action: check the latest ensemble artifact and the most recent "
                f"train_all run — a healthy model produces mixed directions.\n"
            ),
            level=AlertLevel.CRITICAL,
        )

    def strategy_daily_summary(self, report_text: str, report_date: str) -> bool:
        """Send the per-strategy daily summary email."""
        return self.send(
            subject=f"Daily strategy report — {report_date}",
            body=report_text,
            level=AlertLevel.INFO,
        )

    def data_quality_failed(self, report: QualityReport) -> bool:
        """Send alert when critical data quality checks fail."""
        critical = [r for r in report.results if not r.passed and r.severity == "critical"]
        if not critical:
            return False
        subject = f"Data quality critical failures on {report.report_date}"
        lines = [
            f"Date: {report.report_date}",
            f"Critical failures: {report.n_critical}",
            f"Total checks: {len(report.results)}",
            "",
        ]
        for r in critical:
            sym_info = f" ({r.symbol})" if r.symbol else ""
            lines.append(f"  [{r.check_type}]{sym_info} {r.detail}")
        body = "\n".join(lines)
        return self.send(subject=subject, body=body, level=AlertLevel.CRITICAL)
