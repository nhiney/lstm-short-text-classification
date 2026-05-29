"""
error_analysis.py — Qualitative analysis of model errors.

Produces:
  - Misclassified samples table (worst-confidence errors)
  - Per-class error breakdown
  - Confusing class pairs (most common wrong predictions)
  - Attention weight visualization on misclassified examples
  - Saves report to outputs/reports/error_analysis.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F

from src.utils.config import CFG
from src.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_errors(
    model_name: str = "lstm",
    top_n: int = 20,
) -> dict:
    """Run error analysis for the specified model on the test set.

    Args:
        model_name : one of "lstm" | "dnn" | "xlmr"
        top_n      : number of worst errors to surface

    Returns:
        Dict with misclassified samples and confusion stats.
    """
    from src.inference.predict import DNNPredictor, LSTMPredictor, XLMRPredictor

    predictors = {"lstm": LSTMPredictor, "dnn": DNNPredictor, "xlmr": XLMRPredictor}
    if model_name not in predictors:
        raise ValueError(f"model_name must be one of {list(predictors.keys())}")

    predictor = predictors[model_name]()
    test_df   = pd.read_csv(Path(CFG.data.processed_dir) / "test.csv")
    classes   = CFG.labels.codes
    names     = {c: getattr(CFG.labels.names, c, c) for c in classes}

    errors = []
    for _, row in test_df.iterrows():
        result     = predictor.predict(str(row["text_raw"]))
        true_label = str(row["label"])
        pred_label = result["predicted_label"]

        if pred_label != true_label:
            errors.append({
                "text_raw":    str(row["text_raw"]),
                "text_norm":   str(row["text_norm"]),
                "true_label":  true_label,
                "true_name":   names.get(true_label, true_label),
                "pred_label":  pred_label,
                "pred_name":   names.get(pred_label, pred_label),
                "confidence":  result["confidence"],
                "probabilities": result["probabilities"],
            })

    # Sort by confidence (highest confidence wrong prediction = most revealing)
    errors.sort(key=lambda x: -x["confidence"])
    top_errors = errors[:top_n]

    # Confusion pairs
    pair_counts: dict[str, int] = {}
    for e in errors:
        key = f"{e['true_label']} → {e['pred_label']}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    confusion_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:10]

    # Per-class error rate
    total_per_class: dict[str, int] = test_df["label"].value_counts().to_dict()
    error_per_class: dict[str, int] = {}
    for e in errors:
        error_per_class[e["true_label"]] = error_per_class.get(e["true_label"], 0) + 1

    per_class_error_rate = {
        c: round(error_per_class.get(c, 0) / total_per_class.get(c, 1), 4)
        for c in classes
    }

    report = {
        "model":              model_name,
        "total_test":         len(test_df),
        "total_errors":       len(errors),
        "error_rate":         round(len(errors) / len(test_df), 4),
        "per_class_error_rate": per_class_error_rate,
        "top_confusion_pairs":  confusion_pairs,
        "top_errors":           top_errors,
    }

    out_path = Path(CFG.outputs.reports_dir) / f"error_analysis_{model_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    logger.info(
        "[ErrorAnalysis] %s — %d/%d errors (%.1f%%) | saved to %s",
        model_name, len(errors), len(test_df),
        len(errors) / len(test_df) * 100, out_path,
    )
    return report


def print_error_summary(report: dict) -> None:
    """Pretty-print the error analysis report to console."""
    print(f"\n{'='*60}")
    print(f"  Error Analysis — {report['model'].upper()}")
    print(f"{'='*60}")
    print(f"  Total test samples : {report['total_test']}")
    print(f"  Errors             : {report['total_errors']} ({report['error_rate']:.1%})")

    print(f"\n  Per-class error rate:")
    for cls, rate in sorted(report["per_class_error_rate"].items(), key=lambda x: -x[1]):
        bar = "█" * int(rate * 30)
        print(f"    {cls}  {bar:<30} {rate:.1%}")

    print(f"\n  Top confusion pairs:")
    for pair, count in report["top_confusion_pairs"]:
        print(f"    {pair:<20} {count:3d} times")

    print(f"\n  High-confidence errors (top 5):")
    for e in report["top_errors"][:5]:
        print(f"    [{e['true_label']}→{e['pred_label']}] conf={e['confidence']:.0%}  \"{e['text_raw'][:60]}\"")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    report = analyze_errors("lstm")
    print_error_summary(report)
