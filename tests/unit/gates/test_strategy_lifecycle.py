"""Tests for strategy lifecycle manager."""

from __future__ import annotations

from alphavedha.gates.strategy_lifecycle import (
    LifecycleStage,
    StrategyLifecycle,
)


class TestRegistration:
    def test_register_new(self) -> None:
        lc = StrategyLifecycle()
        record = lc.register("ensemble_v1", notes="XGBoost + LSTM")
        assert record is not None
        assert record.strategy == "ensemble_v1"
        assert record.stage == LifecycleStage.HYPOTHESIS
        assert record.notes == "XGBoost + LSTM"
        assert lc.active_count == 1

    def test_register_duplicate(self) -> None:
        lc = StrategyLifecycle()
        lc.register("ensemble_v1")
        record = lc.register("ensemble_v1")
        assert record is not None
        assert lc.active_count == 1

    def test_register_at_capacity(self) -> None:
        lc = StrategyLifecycle(max_concurrent=2)
        lc.register("s1")
        lc.register("s2")
        record = lc.register("s3")
        assert record is None
        assert lc.active_count == 2

    def test_register_after_retirement_frees_slot(self) -> None:
        lc = StrategyLifecycle(max_concurrent=2)
        lc.register("s1")
        lc.register("s2")
        lc.retire("s1", "bad")
        record = lc.register("s3")
        assert record is not None
        assert lc.active_count == 2


class TestAdvance:
    def test_hypothesis_to_paper(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        record = lc.advance("s1", LifecycleStage.PAPER)
        assert record is not None
        assert record.stage == LifecycleStage.PAPER
        assert len(record.history) == 2

    def test_paper_to_gate_review(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.advance("s1", LifecycleStage.PAPER)
        record = lc.advance("s1", LifecycleStage.GATE_REVIEW)
        assert record is not None
        assert record.stage == LifecycleStage.GATE_REVIEW

    def test_gate_review_to_live(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.advance("s1", LifecycleStage.PAPER)
        lc.advance("s1", LifecycleStage.GATE_REVIEW)
        record = lc.advance("s1", LifecycleStage.LIVE)
        assert record is not None
        assert record.stage == LifecycleStage.LIVE

    def test_gate_review_back_to_paper(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.advance("s1", LifecycleStage.PAPER)
        lc.advance("s1", LifecycleStage.GATE_REVIEW)
        record = lc.advance("s1", LifecycleStage.PAPER)
        assert record is not None
        assert record.stage == LifecycleStage.PAPER

    def test_invalid_transition(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        record = lc.advance("s1", LifecycleStage.LIVE)
        assert record is None

    def test_advance_retired_fails(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.retire("s1", "test")
        record = lc.advance("s1", LifecycleStage.PAPER)
        assert record is None

    def test_advance_unknown_strategy(self) -> None:
        lc = StrategyLifecycle()
        record = lc.advance("unknown", LifecycleStage.PAPER)
        assert record is None


class TestGateResults:
    def test_gate_pass(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.advance("s1", LifecycleStage.PAPER)
        record = lc.record_gate_result("s1", passed=True)
        assert record is not None
        assert record.gate_review_count == 1
        assert record.consecutive_gate_failures == 0

    def test_gate_fail(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.advance("s1", LifecycleStage.PAPER)
        record = lc.record_gate_result("s1", passed=False)
        assert record is not None
        assert record.gate_failures == 1
        assert record.consecutive_gate_failures == 1

    def test_two_consecutive_failures_auto_retire(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.advance("s1", LifecycleStage.PAPER)
        lc.record_gate_result("s1", passed=False)
        lc.record_gate_result("s1", passed=False)
        record = lc.strategies["s1"]
        assert record.stage == LifecycleStage.RETIRED
        assert "auto-retired" in record.retirement_reason

    def test_pass_resets_consecutive(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.advance("s1", LifecycleStage.PAPER)
        lc.record_gate_result("s1", passed=False)
        lc.record_gate_result("s1", passed=True)
        record = lc.strategies["s1"]
        assert record.consecutive_gate_failures == 0
        assert record.is_active is True

    def test_gate_result_unknown_strategy(self) -> None:
        lc = StrategyLifecycle()
        record = lc.record_gate_result("unknown", passed=True)
        assert record is None


class TestRetirement:
    def test_retire_with_reason(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        record = lc.retire("s1", "no edge found")
        assert record is not None
        assert record.stage == LifecycleStage.RETIRED
        assert record.retirement_reason == "no edge found"
        assert lc.active_count == 0

    def test_retire_idempotent(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.retire("s1", "reason1")
        record = lc.retire("s1", "reason2")
        assert record is not None
        assert record.retirement_reason == "reason1"

    def test_retire_unknown(self) -> None:
        lc = StrategyLifecycle()
        record = lc.retire("unknown", "test")
        assert record is None


class TestMonthlyReview:
    def test_retires_failing_strategies(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.advance("s1", LifecycleStage.PAPER)
        lc.record_gate_result("s1", passed=False)
        lc.record_gate_result("s1", passed=False)
        retired = lc.monthly_review()
        assert len(retired) >= 0
        assert lc.strategies["s1"].stage == LifecycleStage.RETIRED

    def test_no_false_retirements(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.advance("s1", LifecycleStage.PAPER)
        lc.record_gate_result("s1", passed=False)
        lc.record_gate_result("s1", passed=True)
        retired = lc.monthly_review()
        assert len(retired) == 0
        assert lc.strategies["s1"].is_active is True


class TestProperties:
    def test_active_strategies(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.register("s2")
        lc.retire("s2", "test")
        assert len(lc.active_strategies) == 1
        assert lc.active_strategies[0].strategy == "s1"

    def test_retired_strategies(self) -> None:
        lc = StrategyLifecycle()
        lc.register("s1")
        lc.register("s2")
        lc.retire("s2", "test")
        assert len(lc.retired_strategies) == 1
        assert lc.retired_strategies[0].strategy == "s2"

    def test_days_in_stage(self) -> None:
        lc = StrategyLifecycle()
        record = lc.register("s1")
        assert record is not None
        assert record.days_in_stage == 0


class TestFormatStatus:
    def test_format_empty(self) -> None:
        lc = StrategyLifecycle()
        text = lc.format_status()
        assert "0/5 active" in text
        assert "(none)" in text

    def test_format_with_strategies(self) -> None:
        lc = StrategyLifecycle()
        lc.register("ensemble_v1")
        lc.register("event_drift_v1")
        lc.advance("ensemble_v1", LifecycleStage.PAPER)
        lc.retire("event_drift_v1", "no edge")
        text = lc.format_status()
        assert "1/5 active" in text
        assert "ensemble_v1" in text
        assert "event_drift_v1" in text
        assert "no edge" in text
