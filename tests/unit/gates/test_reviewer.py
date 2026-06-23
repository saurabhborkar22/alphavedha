"""Tests for the gate reviewer — G1/G2/G3 decision gates."""

from __future__ import annotations

from datetime import date

import pytest

from alphavedha.gates.reviewer import (
    G1Criteria,
    G2Criteria,
    GateLevel,
    GateReviewer,
    StrategyMetrics,
)


@pytest.fixture
def reviewer() -> GateReviewer:
    return GateReviewer()


def _passing_g1_metrics() -> StrategyMetrics:
    return StrategyMetrics(
        strategy="ensemble_v1",
        gate=GateLevel.G1,
        review_date=date(2026, 7, 20),
        evaluated_cohorts=35,
        evaluated_trades=80,
        net_expectancy=0.012,
        expectancy_ci_lower=0.003,
        expectancy_ci_upper=0.021,
        win_rate=0.52,
        profit_factor=1.4,
        max_drawdown_pct=-7.5,
        lookahead_test_passed=True,
        hash_record_unbroken=True,
    )


def _passing_g2_metrics() -> StrategyMetrics:
    return StrategyMetrics(
        strategy="ensemble_v1",
        gate=GateLevel.G2,
        review_date=date(2026, 9, 1),
        weeks_live=5,
        live_fills=20,
        live_net_return=0.08,
        paper_net_return=0.09,
        slippage_budget=0.01,
        kill_switch_code_breaches=0,
        missed_mornings=0,
        hash_record_unbroken=True,
    )


class TestG1Pass:
    def test_all_criteria_pass(self, reviewer: GateReviewer) -> None:
        metrics = _passing_g1_metrics()
        verdict = reviewer.review(metrics)
        assert verdict.passed is True
        assert verdict.gate == GateLevel.G1
        assert "PROMOTE" in verdict.recommendation
        assert len(verdict.failed_criteria) == 0

    def test_summary_format(self, reviewer: GateReviewer) -> None:
        metrics = _passing_g1_metrics()
        verdict = reviewer.review(metrics)
        assert "[G1]" in verdict.summary
        assert "PASS" in verdict.summary
        assert "8/8" in verdict.summary


class TestG1Fail:
    def test_insufficient_cohorts(self, reviewer: GateReviewer) -> None:
        metrics = _passing_g1_metrics()
        metrics = StrategyMetrics(
            strategy=metrics.strategy,
            gate=metrics.gate,
            review_date=metrics.review_date,
            evaluated_cohorts=20,
            evaluated_trades=80,
            net_expectancy=0.012,
            expectancy_ci_lower=0.003,
            expectancy_ci_upper=0.021,
            win_rate=0.52,
            profit_factor=1.4,
            max_drawdown_pct=-7.5,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "min_cohorts" in failed_names

    def test_insufficient_trades(self, reviewer: GateReviewer) -> None:
        metrics = _passing_g1_metrics()
        metrics = StrategyMetrics(
            strategy=metrics.strategy,
            gate=metrics.gate,
            evaluated_cohorts=35,
            evaluated_trades=40,
            net_expectancy=0.012,
            expectancy_ci_lower=0.003,
            expectancy_ci_upper=0.021,
            win_rate=0.52,
            profit_factor=1.4,
            max_drawdown_pct=-7.5,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "min_trades" in failed_names

    def test_negative_expectancy(self, reviewer: GateReviewer) -> None:
        metrics = _passing_g1_metrics()
        metrics = StrategyMetrics(
            strategy=metrics.strategy,
            gate=metrics.gate,
            evaluated_cohorts=35,
            evaluated_trades=80,
            net_expectancy=-0.005,
            expectancy_ci_lower=-0.015,
            expectancy_ci_upper=0.005,
            win_rate=0.52,
            profit_factor=1.4,
            max_drawdown_pct=-7.5,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "net_expectancy_positive" in failed_names

    def test_ci_includes_zero(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G1,
            evaluated_cohorts=35,
            evaluated_trades=80,
            net_expectancy=0.005,
            expectancy_ci_lower=-0.001,
            expectancy_ci_upper=0.011,
            win_rate=0.52,
            profit_factor=1.4,
            max_drawdown_pct=-7.5,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "net_expectancy_positive" in failed_names

    def test_low_win_rate(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G1,
            evaluated_cohorts=35,
            evaluated_trades=80,
            net_expectancy=0.012,
            expectancy_ci_lower=0.003,
            expectancy_ci_upper=0.021,
            win_rate=0.40,
            profit_factor=1.4,
            max_drawdown_pct=-7.5,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "win_rate" in failed_names

    def test_low_profit_factor(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G1,
            evaluated_cohorts=35,
            evaluated_trades=80,
            net_expectancy=0.012,
            expectancy_ci_lower=0.003,
            expectancy_ci_upper=0.021,
            win_rate=0.52,
            profit_factor=1.0,
            max_drawdown_pct=-7.5,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "profit_factor" in failed_names

    def test_excessive_drawdown(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G1,
            evaluated_cohorts=35,
            evaluated_trades=80,
            net_expectancy=0.012,
            expectancy_ci_lower=0.003,
            expectancy_ci_upper=0.021,
            win_rate=0.52,
            profit_factor=1.4,
            max_drawdown_pct=-12.0,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "max_drawdown" in failed_names

    def test_lookahead_failure(self, reviewer: GateReviewer) -> None:
        metrics = _passing_g1_metrics()
        metrics = StrategyMetrics(
            strategy=metrics.strategy,
            gate=metrics.gate,
            evaluated_cohorts=35,
            evaluated_trades=80,
            net_expectancy=0.012,
            expectancy_ci_lower=0.003,
            expectancy_ci_upper=0.021,
            win_rate=0.52,
            profit_factor=1.4,
            max_drawdown_pct=-7.5,
            lookahead_test_passed=False,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "lookahead_test" in failed_names

    def test_broken_hash_record(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G1,
            evaluated_cohorts=35,
            evaluated_trades=80,
            net_expectancy=0.012,
            expectancy_ci_lower=0.003,
            expectancy_ci_upper=0.021,
            win_rate=0.52,
            profit_factor=1.4,
            max_drawdown_pct=-7.5,
            hash_record_unbroken=False,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False


class TestG1Retirement:
    def test_first_failure(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="bad_strat",
            gate=GateLevel.G1,
            evaluated_cohorts=35,
            evaluated_trades=80,
            net_expectancy=-0.01,
            expectancy_ci_lower=-0.02,
            expectancy_ci_upper=0.0,
            win_rate=0.35,
            profit_factor=0.8,
            max_drawdown_pct=-15.0,
        )
        verdict = reviewer.review(metrics, prior_failures=0)
        assert verdict.passed is False
        assert verdict.consecutive_failures == 1
        assert "STAY paper" in verdict.recommendation

    def test_second_failure_retires(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="bad_strat",
            gate=GateLevel.G1,
            evaluated_cohorts=35,
            evaluated_trades=80,
            net_expectancy=-0.01,
            expectancy_ci_lower=-0.02,
            expectancy_ci_upper=0.0,
            win_rate=0.35,
            profit_factor=0.8,
            max_drawdown_pct=-15.0,
        )
        verdict = reviewer.review(metrics, prior_failures=1)
        assert verdict.passed is False
        assert verdict.consecutive_failures == 2
        assert "RETIRE" in verdict.recommendation

    def test_pass_resets_failures(self, reviewer: GateReviewer) -> None:
        metrics = _passing_g1_metrics()
        verdict = reviewer.review(metrics, prior_failures=1)
        assert verdict.passed is True
        assert verdict.consecutive_failures == 0


class TestG2Pass:
    def test_all_criteria_pass(self, reviewer: GateReviewer) -> None:
        metrics = _passing_g2_metrics()
        verdict = reviewer.review(metrics)
        assert verdict.passed is True
        assert verdict.gate == GateLevel.G2
        assert "SCALE" in verdict.recommendation


class TestG2Fail:
    def test_insufficient_weeks(self, reviewer: GateReviewer) -> None:
        metrics = _passing_g2_metrics()
        metrics = StrategyMetrics(
            strategy=metrics.strategy,
            gate=GateLevel.G2,
            weeks_live=3,
            live_fills=20,
            live_net_return=0.08,
            paper_net_return=0.09,
            slippage_budget=0.01,
            hash_record_unbroken=True,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "min_weeks_live" in failed_names

    def test_insufficient_fills(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G2,
            weeks_live=5,
            live_fills=10,
            live_net_return=0.08,
            paper_net_return=0.09,
            slippage_budget=0.01,
            hash_record_unbroken=True,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "min_live_fills" in failed_names

    def test_slippage_over_budget(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G2,
            weeks_live=5,
            live_fills=20,
            live_net_return=0.05,
            paper_net_return=0.10,
            slippage_budget=0.01,
            hash_record_unbroken=True,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "slippage_within_budget" in failed_names

    def test_kill_switch_breach(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G2,
            weeks_live=5,
            live_fills=20,
            live_net_return=0.08,
            paper_net_return=0.09,
            slippage_budget=0.01,
            kill_switch_code_breaches=1,
            hash_record_unbroken=True,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "kill_switch_code_breaches" in failed_names

    def test_missed_mornings(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G2,
            weeks_live=5,
            live_fills=20,
            live_net_return=0.08,
            paper_net_return=0.09,
            slippage_budget=0.01,
            missed_mornings=2,
            hash_record_unbroken=True,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        failed_names = [c.name for c in verdict.failed_criteria]
        assert "missed_mornings" in failed_names


class TestG3:
    def test_pass(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G3,
            months_public_record=8,
            sebi_ra_registered=True,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is True
        assert "PRODUCTIZE" in verdict.recommendation

    def test_fail_no_record(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G3,
            months_public_record=4,
            sebi_ra_registered=True,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False

    def test_fail_no_sebi(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G3,
            months_public_record=8,
            sebi_ra_registered=False,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False

    def test_fail_both(self, reviewer: GateReviewer) -> None:
        metrics = StrategyMetrics(
            strategy="ensemble_v1",
            gate=GateLevel.G3,
            months_public_record=3,
            sebi_ra_registered=False,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is False
        assert len(verdict.failed_criteria) == 2


class TestCustomCriteria:
    def test_custom_g1_thresholds(self) -> None:
        reviewer = GateReviewer(g1=G1Criteria(min_cohorts=20, min_trades=40, min_win_rate=0.40))
        metrics = StrategyMetrics(
            strategy="custom",
            gate=GateLevel.G1,
            evaluated_cohorts=25,
            evaluated_trades=50,
            net_expectancy=0.01,
            expectancy_ci_lower=0.001,
            expectancy_ci_upper=0.02,
            win_rate=0.42,
            profit_factor=1.3,
            max_drawdown_pct=-8.0,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is True

    def test_custom_g2_thresholds(self) -> None:
        reviewer = GateReviewer(g2=G2Criteria(min_weeks_live=2, min_live_fills=10))
        metrics = StrategyMetrics(
            strategy="custom",
            gate=GateLevel.G2,
            weeks_live=3,
            live_fills=12,
            live_net_return=0.05,
            paper_net_return=0.05,
            slippage_budget=0.01,
            hash_record_unbroken=True,
        )
        verdict = reviewer.review(metrics)
        assert verdict.passed is True
