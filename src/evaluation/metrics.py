"""
metrics.py — Evaluation metric helpers.

Computes standard classification metrics and returns them as a structured dict
ready for JSON serialisation and visualisation.
"""
from __future__ import annotations

from typing import List, Optional

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_labels: Optional[List[str]] = None,
) -> dict:
    """Compute a full set of multi-class classification metrics.

    Args:
        y_true        : ground-truth integer labels
        y_pred        : predicted integer labels
        class_labels  : optional list of class names for the report

    Returns:
        Dict containing:
            - accuracy        : float
            - precision       : weighted average float
            - recall          : weighted average float
            - f1              : weighted average float
            - confusion_matrix: 2-D list of ints
            - report          : sklearn classification_report string
            - per_class_f1    : dict mapping class name → F1 score
    """
    label_range = list(range(len(class_labels))) if class_labels else None

    per_class_f1_arr = f1_score(
        y_true, y_pred, average=None, labels=label_range, zero_division=0
    )
    per_class_f1 = (
        {name: round(float(v), 4) for name, v in zip(class_labels, per_class_f1_arr)}
        if class_labels
        else {str(i): round(float(v), 4) for i, v in enumerate(per_class_f1_arr)}
    )

    return {
        "accuracy":         round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":        round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall":           round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1":               round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_range).tolist(),
        "per_class_f1":     per_class_f1,
        "report":           classification_report(
            y_true, y_pred,
            target_names=class_labels,
            zero_division=0,
        ),
    }
