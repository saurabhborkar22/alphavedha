"""Tests for extraction evaluation harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphavedha.intel.extraction.eval import (
    EvalReport,
    PerTypeMetrics,
    evaluate,
    load_golden_set,
    passes_quality_gate,
)
from alphavedha.intel.extraction.schemas import DisclosureExtraction
from alphavedha.intel.extraction.taxonomy import EventType


def _make(
    event_type: EventType = EventType.OTHER,
    direction: int = 0,
    materiality: int = 5,
    confidence: float = 0.8,
    summary: str = "test",
) -> DisclosureExtraction:
    return DisclosureExtraction(
        event_type=event_type,
        direction=direction,
        materiality=materiality,
        confidence=confidence,
        summary=summary,
    )


class TestPerTypeMetrics:
    def test_precision(self) -> None:
        m = PerTypeMetrics(event_type="x", tp=8, fp=2, fn=0)
        assert m.precision == pytest.approx(0.8)

    def test_recall(self) -> None:
        m = PerTypeMetrics(event_type="x", tp=8, fp=0, fn=2)
        assert m.recall == pytest.approx(0.8)

    def test_f1(self) -> None:
        m = PerTypeMetrics(event_type="x", tp=8, fp=2, fn=2)
        assert m.f1 == pytest.approx(0.8)

    def test_zero_division(self) -> None:
        m = PerTypeMetrics(event_type="x")
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0


class TestEvaluate:
    def test_perfect_predictions(self) -> None:
        labels = [_make(EventType.ORDER_WIN, 1, 7), _make(EventType.M_AND_A, 0, 6)]
        preds = [_make(EventType.ORDER_WIN, 1, 7), _make(EventType.M_AND_A, 0, 6)]
        report = evaluate(preds, labels)
        assert report.macro_precision == 1.0
        assert report.macro_recall == 1.0
        assert report.materiality_mae == 0.0

    def test_wrong_type(self) -> None:
        labels = [_make(EventType.ORDER_WIN)]
        preds = [_make(EventType.M_AND_A)]
        report = evaluate(preds, labels)
        assert report.per_type["order_win"].fn == 1
        assert report.per_type["m_and_a"].fp == 1
        assert report.macro_precision == 0.0

    def test_materiality_mae(self) -> None:
        labels = [_make(materiality=7), _make(materiality=3)]
        preds = [_make(materiality=5), _make(materiality=5)]
        report = evaluate(preds, labels)
        assert report.materiality_mae == pytest.approx(2.0)

    def test_direction_accuracy(self) -> None:
        labels = [_make(direction=1), _make(direction=-1), _make(direction=0)]
        preds = [_make(direction=1), _make(direction=1), _make(direction=0)]
        report = evaluate(preds, labels)
        assert report.direction_total == 2
        assert report.direction_correct == 1
        assert report.direction_accuracy == pytest.approx(0.5)

    def test_red_flag_recall(self) -> None:
        labels = [
            _make(EventType.AUDITOR_RESIGNATION),
            _make(EventType.DEFAULT_OR_DELAY),
            _make(EventType.ORDER_WIN),
        ]
        preds = [
            _make(EventType.AUDITOR_RESIGNATION),
            _make(EventType.OTHER),
            _make(EventType.ORDER_WIN),
        ]
        report = evaluate(preds, labels)
        assert report.red_flag_tp == 1
        assert report.red_flag_fn == 1
        assert report.red_flag_recall == pytest.approx(0.5)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate([_make()], [_make(), _make()])

    def test_empty_lists(self) -> None:
        report = evaluate([], [])
        assert report.total == 0
        assert report.macro_precision == 0.0

    def test_summary_returns_dict(self) -> None:
        labels = [_make(EventType.ORDER_WIN, 1, 7)]
        preds = [_make(EventType.ORDER_WIN, 1, 6)]
        report = evaluate(preds, labels)
        s = report.summary()
        assert "total_samples" in s
        assert "macro_precision" in s
        assert "red_flag_recall" in s
        assert "per_type" in s


class TestQualityGate:
    def test_passes(self) -> None:
        report = EvalReport()
        report.per_type["x"] = PerTypeMetrics(event_type="x", tp=9, fp=1, fn=0)
        report.red_flag_tp = 9
        report.red_flag_fn = 1
        assert passes_quality_gate(report) is True

    def test_fails_precision(self) -> None:
        report = EvalReport()
        report.per_type["x"] = PerTypeMetrics(event_type="x", tp=5, fp=5, fn=0)
        report.red_flag_tp = 10
        report.red_flag_fn = 0
        assert passes_quality_gate(report) is False

    def test_fails_red_flag_recall(self) -> None:
        report = EvalReport()
        report.per_type["x"] = PerTypeMetrics(event_type="x", tp=9, fp=1, fn=0)
        report.red_flag_tp = 7
        report.red_flag_fn = 3
        assert passes_quality_gate(report) is False


class TestLoadGoldenSet:
    def test_loads_from_file(self) -> None:
        items = load_golden_set()
        assert len(items) == 100

    def test_each_item_has_label(self) -> None:
        items = load_golden_set()
        for item in items:
            assert "label" in item
            assert "event_type" in item["label"]
            assert "direction" in item["label"]
            assert "materiality" in item["label"]

    def test_labels_validate_as_extraction(self) -> None:
        items = load_golden_set()
        for item in items:
            e = DisclosureExtraction(**item["label"])
            assert e.event_type in EventType

    def test_has_relevant_and_boilerplate(self) -> None:
        items = load_golden_set()
        relevant = sum(1 for i in items if i["is_relevant"])
        boilerplate = sum(1 for i in items if not i["is_relevant"])
        assert relevant > 0
        assert boilerplate > 0

    def test_nonexistent_file_returns_empty(self) -> None:
        items = load_golden_set(Path("/nonexistent/path.jsonl"))
        assert items == []
