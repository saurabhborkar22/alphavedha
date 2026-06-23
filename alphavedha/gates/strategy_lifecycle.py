"""Strategy lifecycle manager — the factory loop.

Tracks strategies through their lifecycle:
  hypothesis -> paper -> gate_review -> live -> scale -> retire

Enforces the 5-concurrent-strategy cap. Records the kill reason for
retired strategies (dead strategies are also a track record).
Monthly retirement check: strategies that fail twice are auto-retired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import structlog

logger = structlog.get_logger(__name__)

_MAX_CONCURRENT_STRATEGIES: int = 5


class LifecycleStage(StrEnum):
    HYPOTHESIS = "hypothesis"
    PAPER = "paper"
    GATE_REVIEW = "gate_review"
    LIVE = "live"
    RETIRED = "retired"


@dataclass
class StrategyRecord:
    """Full lifecycle record for a single strategy."""

    strategy: str
    stage: LifecycleStage = LifecycleStage.HYPOTHESIS
    created_date: date = field(default_factory=date.today)
    stage_entered_date: date = field(default_factory=date.today)
    gate_review_count: int = 0
    gate_failures: int = 0
    consecutive_gate_failures: int = 0
    retirement_reason: str = ""
    notes: str = ""
    history: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.stage != LifecycleStage.RETIRED

    @property
    def days_in_stage(self) -> int:
        return (date.today() - self.stage_entered_date).days


class StrategyLifecycle:
    """Manages the factory loop for all strategies.

    Usage:
        lifecycle = StrategyLifecycle()
        lifecycle.register("ensemble_v1", notes="XGBoost + LSTM + TFT")
        lifecycle.advance("ensemble_v1", LifecycleStage.PAPER)
        lifecycle.retire("ensemble_v1", "2 consecutive G1 failures")
    """

    def __init__(
        self,
        max_concurrent: int = _MAX_CONCURRENT_STRATEGIES,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._strategies: dict[str, StrategyRecord] = {}

    @property
    def strategies(self) -> dict[str, StrategyRecord]:
        return dict(self._strategies)

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._strategies.values() if s.is_active)

    @property
    def active_strategies(self) -> list[StrategyRecord]:
        return [s for s in self._strategies.values() if s.is_active]

    @property
    def retired_strategies(self) -> list[StrategyRecord]:
        return [s for s in self._strategies.values() if not s.is_active]

    def register(
        self,
        strategy: str,
        notes: str = "",
    ) -> StrategyRecord | None:
        """Register a new strategy hypothesis.

        Returns None if at capacity (5 concurrent max).
        """
        if strategy in self._strategies:
            logger.warning("lifecycle_already_registered", strategy=strategy)
            return self._strategies[strategy]

        if self.active_count >= self._max_concurrent:
            logger.warning(
                "lifecycle_at_capacity",
                strategy=strategy,
                active=self.active_count,
                max=self._max_concurrent,
            )
            return None

        record = StrategyRecord(
            strategy=strategy,
            notes=notes,
        )
        record.history.append(f"{date.today()}: registered as hypothesis")
        self._strategies[strategy] = record

        logger.info(
            "lifecycle_registered",
            strategy=strategy,
            active_count=self.active_count,
        )

        return record

    def advance(
        self,
        strategy: str,
        to_stage: LifecycleStage,
    ) -> StrategyRecord | None:
        """Advance a strategy to the next lifecycle stage.

        Returns None if strategy not found or invalid transition.
        """
        record = self._strategies.get(strategy)
        if record is None:
            logger.warning("lifecycle_not_found", strategy=strategy)
            return None

        if record.stage == LifecycleStage.RETIRED:
            logger.warning("lifecycle_already_retired", strategy=strategy)
            return None

        valid_transitions = {
            LifecycleStage.HYPOTHESIS: {LifecycleStage.PAPER, LifecycleStage.RETIRED},
            LifecycleStage.PAPER: {LifecycleStage.GATE_REVIEW, LifecycleStage.RETIRED},
            LifecycleStage.GATE_REVIEW: {
                LifecycleStage.LIVE,
                LifecycleStage.PAPER,
                LifecycleStage.RETIRED,
            },
            LifecycleStage.LIVE: {LifecycleStage.GATE_REVIEW, LifecycleStage.RETIRED},
        }

        allowed = valid_transitions.get(record.stage, set())
        if to_stage not in allowed:
            logger.warning(
                "lifecycle_invalid_transition",
                strategy=strategy,
                from_stage=record.stage.value,
                to_stage=to_stage.value,
            )
            return None

        old_stage = record.stage
        record.stage = to_stage
        record.stage_entered_date = date.today()
        record.history.append(f"{date.today()}: {old_stage.value} -> {to_stage.value}")

        logger.info(
            "lifecycle_advanced",
            strategy=strategy,
            from_stage=old_stage.value,
            to_stage=to_stage.value,
        )

        return record

    def record_gate_result(
        self,
        strategy: str,
        passed: bool,
    ) -> StrategyRecord | None:
        """Record a gate review result. Auto-retires after 2 consecutive failures."""
        record = self._strategies.get(strategy)
        if record is None:
            return None

        record.gate_review_count += 1

        if passed:
            record.consecutive_gate_failures = 0
            record.history.append(
                f"{date.today()}: gate review PASSED (#{record.gate_review_count})"
            )
        else:
            record.gate_failures += 1
            record.consecutive_gate_failures += 1
            record.history.append(
                f"{date.today()}: gate review FAILED "
                f"(#{record.gate_review_count}, consecutive: {record.consecutive_gate_failures})"
            )

            if record.consecutive_gate_failures >= 2:
                self.retire(strategy, "2 consecutive gate failures — auto-retired")

        return record

    def retire(
        self,
        strategy: str,
        reason: str = "",
    ) -> StrategyRecord | None:
        """Retire a strategy. Dead strategies stay in the record."""
        record = self._strategies.get(strategy)
        if record is None:
            return None

        if record.stage == LifecycleStage.RETIRED:
            return record

        old_stage = record.stage
        record.stage = LifecycleStage.RETIRED
        record.stage_entered_date = date.today()
        record.retirement_reason = reason
        record.history.append(f"{date.today()}: RETIRED from {old_stage.value} ({reason})")

        logger.info(
            "lifecycle_retired",
            strategy=strategy,
            from_stage=old_stage.value,
            reason=reason,
        )

        return record

    def monthly_review(self) -> list[StrategyRecord]:
        """Run monthly retirement check on all active strategies.

        Returns list of strategies that were auto-retired.
        """
        retired: list[StrategyRecord] = []
        for record in list(self._strategies.values()):
            if not record.is_active:
                continue
            if record.consecutive_gate_failures >= 2:
                self.retire(record.strategy, "monthly review: 2+ consecutive failures")
                retired.append(record)
        return retired

    def format_status(self) -> str:
        """Format lifecycle status for display."""
        lines = [
            f"Strategy Lifecycle ({self.active_count}/{self._max_concurrent} active)",
            "",
        ]

        by_stage: dict[str, list[StrategyRecord]] = {}
        for record in self._strategies.values():
            by_stage.setdefault(record.stage.value, []).append(record)

        for stage in LifecycleStage:
            records = by_stage.get(stage.value, [])
            lines.append(f"  {stage.value}:")
            if records:
                for r in records:
                    extra = ""
                    if r.stage == LifecycleStage.RETIRED:
                        extra = f" ({r.retirement_reason})"
                    elif r.gate_review_count > 0:
                        extra = f" (reviews: {r.gate_review_count}, fails: {r.gate_failures})"
                    lines.append(f"    - {r.strategy}{extra}")
            else:
                lines.append("    (none)")

        return "\n".join(lines)
