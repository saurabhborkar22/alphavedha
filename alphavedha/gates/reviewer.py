"""Gate reviewer — applies §13 decision-gate criteria to strategy metrics.

Three gate levels with pre-committed, quantitative thresholds:
  G1: paper → small live (per strategy)
  G2: small live → scale (per strategy)
  G3: productize (sell anything)

Written BEFORE results exist so we can't move the goalposts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import structlog

logger = structlog.get_logger(__name__)


class GateLevel(StrEnum):
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"


@dataclass(frozen=True)
class G1Criteria:
    """G1 — paper → small live. All thresholds pre-committed."""

    min_cohorts: int = 30
    min_trades: int = 60
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.2
    max_drawdown_pct: float = -10.0
    bootstrap_ci_level: float = 0.90
    consecutive_failures_to_retire: int = 2


@dataclass(frozen=True)
class G2Criteria:
    """G2 — small live → scale."""

    min_weeks_live: int = 4
    min_live_fills: int = 15
    max_slippage_multiple: float = 1.5
    max_kill_switch_code_breaches: int = 0
    require_unbroken_hashes: bool = True
    require_no_missed_mornings: bool = True


@dataclass
class StrategyMetrics:
    """Observed metrics for a strategy under review."""

    strategy: str
    gate: GateLevel
    review_date: date = field(default_factory=date.today)

    # G1 metrics
    evaluated_cohorts: int = 0
    evaluated_trades: int = 0
    net_expectancy: float = 0.0
    expectancy_ci_lower: float = 0.0
    expectancy_ci_upper: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    lookahead_test_passed: bool = True
    hash_record_unbroken: bool = True

    # G2 metrics
    weeks_live: int = 0
    live_fills: int = 0
    live_net_return: float = 0.0
    paper_net_return: float = 0.0
    slippage_budget: float = 0.0
    kill_switch_code_breaches: int = 0
    missed_mornings: int = 0

    # G3 metrics
    months_public_record: int = 0
    sebi_ra_registered: bool = False


@dataclass(frozen=True)
class CriterionResult:
    """Result of evaluating a single criterion."""

    name: str
    passed: bool
    observed: str
    threshold: str
    note: str = ""


@dataclass
class GateVerdict:
    """Full verdict for a gate review."""

    strategy: str
    gate: GateLevel
    review_date: date
    passed: bool
    criteria_results: list[CriterionResult] = field(default_factory=list)
    consecutive_failures: int = 0
    recommendation: str = ""

    @property
    def failed_criteria(self) -> list[CriterionResult]:
        return [c for c in self.criteria_results if not c.passed]

    @property
    def summary(self) -> str:
        passed_count = sum(1 for c in self.criteria_results if c.passed)
        total = len(self.criteria_results)
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{self.gate.value}] {self.strategy}: {status} "
            f"({passed_count}/{total} criteria) — {self.recommendation}"
        )


class GateReviewer:
    """Evaluates strategy metrics against pre-committed gate criteria."""

    def __init__(
        self,
        g1: G1Criteria | None = None,
        g2: G2Criteria | None = None,
    ) -> None:
        self._g1 = g1 or G1Criteria()
        self._g2 = g2 or G2Criteria()

    def review(
        self,
        metrics: StrategyMetrics,
        prior_failures: int = 0,
    ) -> GateVerdict:
        if metrics.gate == GateLevel.G1:
            return self._review_g1(metrics, prior_failures)
        elif metrics.gate == GateLevel.G2:
            return self._review_g2(metrics)
        elif metrics.gate == GateLevel.G3:
            return self._review_g3(metrics)
        else:
            raise ValueError(f"Unknown gate level: {metrics.gate}")

    def _review_g1(self, m: StrategyMetrics, prior_failures: int) -> GateVerdict:
        results: list[CriterionResult] = []

        results.append(
            CriterionResult(
                name="min_cohorts",
                passed=m.evaluated_cohorts >= self._g1.min_cohorts,
                observed=str(m.evaluated_cohorts),
                threshold=f">= {self._g1.min_cohorts}",
            )
        )

        results.append(
            CriterionResult(
                name="min_trades",
                passed=m.evaluated_trades >= self._g1.min_trades,
                observed=str(m.evaluated_trades),
                threshold=f">= {self._g1.min_trades}",
            )
        )

        ci_excludes_zero = m.expectancy_ci_lower > 0
        results.append(
            CriterionResult(
                name="net_expectancy_positive",
                passed=m.net_expectancy > 0 and ci_excludes_zero,
                observed=f"{m.net_expectancy:.4f} (CI: [{m.expectancy_ci_lower:.4f}, {m.expectancy_ci_upper:.4f}])",
                threshold=f"> 0 with {self._g1.bootstrap_ci_level:.0%} CI excluding 0",
            )
        )

        results.append(
            CriterionResult(
                name="win_rate",
                passed=m.win_rate >= self._g1.min_win_rate,
                observed=f"{m.win_rate:.2%}",
                threshold=f">= {self._g1.min_win_rate:.0%}",
            )
        )

        results.append(
            CriterionResult(
                name="profit_factor",
                passed=m.profit_factor >= self._g1.min_profit_factor,
                observed=f"{m.profit_factor:.2f}",
                threshold=f">= {self._g1.min_profit_factor:.1f}",
            )
        )

        results.append(
            CriterionResult(
                name="max_drawdown",
                passed=m.max_drawdown_pct > self._g1.max_drawdown_pct,
                observed=f"{m.max_drawdown_pct:.2f}%",
                threshold=f"> {self._g1.max_drawdown_pct:.0f}%",
            )
        )

        results.append(
            CriterionResult(
                name="lookahead_test",
                passed=m.lookahead_test_passed,
                observed="passed" if m.lookahead_test_passed else "FAILED",
                threshold="no failures",
            )
        )

        results.append(
            CriterionResult(
                name="hash_record",
                passed=m.hash_record_unbroken,
                observed="unbroken" if m.hash_record_unbroken else "BROKEN",
                threshold="unbroken for period",
            )
        )

        passed = all(r.passed for r in results)
        consecutive = 0 if passed else prior_failures + 1

        if passed:
            recommendation = "PROMOTE to small live (₹50,000, Telegram-approved)"
        elif consecutive >= self._g1.consecutive_failures_to_retire:
            recommendation = f"RETIRE — {consecutive} consecutive failures"
        else:
            recommendation = f"STAY paper — failure {consecutive}/{self._g1.consecutive_failures_to_retire} before retire"

        verdict = GateVerdict(
            strategy=m.strategy,
            gate=GateLevel.G1,
            review_date=m.review_date,
            passed=passed,
            criteria_results=results,
            consecutive_failures=consecutive,
            recommendation=recommendation,
        )

        logger.info(
            "gate_review_g1",
            strategy=m.strategy,
            passed=passed,
            consecutive_failures=consecutive,
            recommendation=recommendation,
        )

        return verdict

    def _review_g2(self, m: StrategyMetrics) -> GateVerdict:
        results: list[CriterionResult] = []

        results.append(
            CriterionResult(
                name="min_weeks_live",
                passed=m.weeks_live >= self._g2.min_weeks_live,
                observed=f"{m.weeks_live} weeks",
                threshold=f">= {self._g2.min_weeks_live} weeks",
            )
        )

        results.append(
            CriterionResult(
                name="min_live_fills",
                passed=m.live_fills >= self._g2.min_live_fills,
                observed=str(m.live_fills),
                threshold=f">= {self._g2.min_live_fills}",
            )
        )

        slippage_ok = True
        if m.slippage_budget > 0 and m.paper_net_return != 0:
            divergence = abs(m.live_net_return - m.paper_net_return)
            slippage_ok = divergence <= m.slippage_budget * self._g2.max_slippage_multiple

        results.append(
            CriterionResult(
                name="slippage_within_budget",
                passed=slippage_ok,
                observed=f"live={m.live_net_return:.2%}, paper={m.paper_net_return:.2%}",
                threshold=f"divergence <= {self._g2.max_slippage_multiple}x slippage budget",
            )
        )

        results.append(
            CriterionResult(
                name="kill_switch_code_breaches",
                passed=m.kill_switch_code_breaches <= self._g2.max_kill_switch_code_breaches,
                observed=str(m.kill_switch_code_breaches),
                threshold=f"<= {self._g2.max_kill_switch_code_breaches}",
            )
        )

        if self._g2.require_unbroken_hashes:
            results.append(
                CriterionResult(
                    name="hash_record",
                    passed=m.hash_record_unbroken,
                    observed="unbroken" if m.hash_record_unbroken else "BROKEN",
                    threshold="unbroken",
                )
            )

        if self._g2.require_no_missed_mornings:
            results.append(
                CriterionResult(
                    name="missed_mornings",
                    passed=m.missed_mornings == 0,
                    observed=str(m.missed_mornings),
                    threshold="0",
                )
            )

        passed = all(r.passed for r in results)
        recommendation = (
            "SCALE — promote to next capital tier"
            if passed
            else "STAY at current size — fix issues first"
        )

        verdict = GateVerdict(
            strategy=m.strategy,
            gate=GateLevel.G2,
            review_date=m.review_date,
            passed=passed,
            criteria_results=results,
            recommendation=recommendation,
        )

        logger.info(
            "gate_review_g2",
            strategy=m.strategy,
            passed=passed,
            recommendation=recommendation,
        )

        return verdict

    def _review_g3(self, m: StrategyMetrics) -> GateVerdict:
        results: list[CriterionResult] = []

        results.append(
            CriterionResult(
                name="public_record_months",
                passed=m.months_public_record >= 6,
                observed=f"{m.months_public_record} months",
                threshold=">= 6 months",
            )
        )

        results.append(
            CriterionResult(
                name="sebi_ra_registered",
                passed=m.sebi_ra_registered,
                observed="yes" if m.sebi_ra_registered else "NO",
                threshold="registered before any paid recommendation",
            )
        )

        passed = all(r.passed for r in results)
        recommendation = (
            "PRODUCTIZE — cleared for paid offerings"
            if passed
            else "NOT READY — complete prerequisites"
        )

        verdict = GateVerdict(
            strategy=m.strategy,
            gate=GateLevel.G3,
            review_date=m.review_date,
            passed=passed,
            criteria_results=results,
            recommendation=recommendation,
        )

        logger.info(
            "gate_review_g3",
            strategy=m.strategy,
            passed=passed,
            recommendation=recommendation,
        )

        return verdict
