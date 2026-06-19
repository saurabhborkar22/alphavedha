"""Run extraction bake-off against the golden set.

Usage:
    GEMINI_API_KEY=... python scripts/run_bakeoff.py
    GEMINI_API_KEY=... GROQ_API_KEY=... python scripts/run_bakeoff.py --providers gemini,groq
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphavedha.intel.extraction.eval import evaluate, load_golden_set, passes_quality_gate
from alphavedha.intel.extraction.extractor import extract_one
from alphavedha.intel.extraction.llm import get_provider
from alphavedha.intel.extraction.schemas import DisclosureExtraction
from alphavedha.intel.extraction.taxonomy import EventType


def run_bakeoff(provider_names: list[str]) -> None:
    golden = load_golden_set()
    if not golden:
        print("ERROR: Golden set not found at tests/fixtures/intel_golden_set.jsonl")
        return

    relevant = [g for g in golden if g["is_relevant"]]
    print(f"Golden set: {len(golden)} total, {len(relevant)} relevant (used for eval)\n")

    labels = [DisclosureExtraction(**g["label"]) for g in relevant]

    for pname in provider_names:
        print(f"{'=' * 60}")
        print(f"Provider: {pname}")
        print(f"{'=' * 60}")

        try:
            provider = get_provider(pname)
        except Exception as e:
            print(f"  SKIP: {e}\n")
            continue

        predictions: list[DisclosureExtraction] = []
        errors = 0
        start = time.time()

        for i, g in enumerate(relevant):
            symbol = g["symbol"]
            category = g["nse_category"]
            headline = g["headline"]

            result = extract_one(provider, symbol, category, headline)

            if result is None:
                result = DisclosureExtraction(
                    event_type=EventType.OTHER,
                    direction=0,
                    materiality=0,
                    confidence=0.0,
                    summary="extraction failed",
                )
                errors += 1

            predictions.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(relevant)}...")

        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({errors} errors)\n")

        report = evaluate(predictions, labels)
        summary = report.summary()

        print(f"  Macro Precision:    {summary['macro_precision']}")
        print(f"  Macro Recall:       {summary['macro_recall']}")
        print(f"  Macro F1:           {summary['macro_f1']}")
        print(f"  Red-Flag Recall:    {summary['red_flag_recall']}")
        print(f"  Materiality MAE:    {summary['materiality_mae']}")
        print(f"  Direction Accuracy: {summary['direction_accuracy']}")
        print(f"  Quality Gate:       {'PASS' if passes_quality_gate(report) else 'FAIL'}")
        print()

        print("  Per-type breakdown:")
        for etype, metrics in sorted(summary["per_type"].items()):  # type: ignore[union-attr]
            print(
                f"    {etype:25s}  P={metrics['p']:.2f}  R={metrics['r']:.2f}  F1={metrics['f1']:.2f}"
            )  # type: ignore[index]
        print()

        out_path = Path(f"bakeoff_{pname}.json")
        raw = [
            {"id": g["id"], "predicted": p.model_dump(), "label": g["label"]}
            for g, p in zip(relevant, predictions, strict=True)
        ]
        out_path.write_text(json.dumps({"summary": summary, "details": raw}, indent=2, default=str))
        print(f"  Raw results saved to {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--providers",
        default="gemini",
        help="Comma-separated list of providers to test (default: gemini)",
    )
    args = parser.parse_args()
    providers = [p.strip() for p in args.providers.split(",")]
    run_bakeoff(providers)
