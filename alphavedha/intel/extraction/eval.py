"""Evaluation harness for LLM extraction quality.

Compares LLM outputs against a human-labelled golden set and computes:
- Precision / recall per event_type
- Macro-averaged precision / recall / F1
- Red-flag recall (the metric that matters most)
- Materiality MAE (mean absolute error)
- Direction accuracy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from alphavedha.intel.extraction.schemas import DisclosureExtraction
from alphavedha.intel.extraction.taxonomy import RED_FLAG_TYPES

logger = structlog.get_logger(__name__)

GOLDEN_SET_PATH = Path("tests/fixtures/intel_golden_set.jsonl")


@dataclass
class PerTypeMetrics:
    """Precision / recall for a single event_type."""

    event_type: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class EvalReport:
    """Full evaluation report."""

    per_type: dict[str, PerTypeMetrics] = field(default_factory=dict)
    materiality_errors: list[int] = field(default_factory=list)
    direction_correct: int = 0
    direction_total: int = 0
    red_flag_tp: int = 0
    red_flag_fn: int = 0
    total: int = 0

    @property
    def macro_precision(self) -> float:
        vals = [m.precision for m in self.per_type.values() if (m.tp + m.fp) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_recall(self) -> float:
        vals = [m.recall for m in self.per_type.values() if (m.tp + m.fn) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_f1(self) -> float:
        p, r = self.macro_precision, self.macro_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def red_flag_recall(self) -> float:
        total = self.red_flag_tp + self.red_flag_fn
        return self.red_flag_tp / total if total > 0 else 0.0

    @property
    def materiality_mae(self) -> float:
        return (
            sum(abs(e) for e in self.materiality_errors) / len(self.materiality_errors)
            if self.materiality_errors
            else 0.0
        )

    @property
    def direction_accuracy(self) -> float:
        return self.direction_correct / self.direction_total if self.direction_total > 0 else 0.0

    def summary(self) -> dict[str, object]:
        return {
            "total_samples": self.total,
            "macro_precision": round(self.macro_precision, 3),
            "macro_recall": round(self.macro_recall, 3),
            "macro_f1": round(self.macro_f1, 3),
            "red_flag_recall": round(self.red_flag_recall, 3),
            "materiality_mae": round(self.materiality_mae, 2),
            "direction_accuracy": round(self.direction_accuracy, 3),
            "per_type": {
                k: {"p": round(v.precision, 3), "r": round(v.recall, 3), "f1": round(v.f1, 3)}
                for k, v in sorted(self.per_type.items())
                if (v.tp + v.fp + v.fn) > 0
            },
        }


def evaluate(
    predictions: list[DisclosureExtraction],
    labels: list[DisclosureExtraction],
) -> EvalReport:
    """Compare predicted extractions against golden-set labels.

    Both lists must be the same length and aligned by index (i.e.,
    predictions[i] is the LLM output for the same disclosure as
    labels[i]).
    """
    if len(predictions) != len(labels):
        msg = f"Length mismatch: {len(predictions)} predictions vs {len(labels)} labels"
        raise ValueError(msg)

    report = EvalReport(total=len(labels))

    for pred, label in zip(predictions, labels, strict=True):
        label_type = label.event_type.value
        pred_type = pred.event_type.value

        if label_type not in report.per_type:
            report.per_type[label_type] = PerTypeMetrics(event_type=label_type)
        if pred_type not in report.per_type:
            report.per_type[pred_type] = PerTypeMetrics(event_type=pred_type)

        if pred_type == label_type:
            report.per_type[label_type].tp += 1
        else:
            report.per_type[label_type].fn += 1
            report.per_type[pred_type].fp += 1

        report.materiality_errors.append(pred.materiality - label.materiality)

        if label.direction != 0:
            report.direction_total += 1
            if pred.direction == label.direction:
                report.direction_correct += 1

        if label.event_type in RED_FLAG_TYPES:
            if pred.event_type == label.event_type:
                report.red_flag_tp += 1
            else:
                report.red_flag_fn += 1

    return report


def load_golden_set(path: Path | None = None) -> list[dict[str, Any]]:
    """Load golden set from JSONL file."""
    import json

    fpath = path or GOLDEN_SET_PATH
    if not fpath.exists():
        logger.warning("golden_set_not_found", path=str(fpath))
        return []

    items: list[dict[str, Any]] = []
    with fpath.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def passes_quality_gate(report: EvalReport) -> bool:
    """Check if the extraction quality meets the P2 bar."""
    return report.macro_precision >= 0.85 and report.red_flag_recall >= 0.90
