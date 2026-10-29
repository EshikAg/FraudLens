"""Fraud-detection metrics with explicit behavior for incomplete label sets."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_predictions(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    threshold: float = 0.5,
    precision_recall_k: Iterable[int] = (10, 50, 100),
) -> dict[str, Any]:
    """Return ranking, thresholded, top-K, and confusion-matrix metrics.

    ROC-AUC and average precision are ``None`` when the evaluation window has
    only one class; they are not meaningful for that window.
    """
    raw_labels = np.asarray(list(labels))
    scores = np.asarray(list(probabilities), dtype=np.float64)
    if raw_labels.ndim != 1 or scores.ndim != 1 or raw_labels.size != scores.size:
        raise ValueError("labels and probabilities must be equal-length 1D sequences")
    if raw_labels.size == 0:
        raise ValueError("At least one labeled prediction is required")
    try:
        numeric_labels = raw_labels.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must contain only binary values 0 and 1") from exc
    if not np.isfinite(numeric_labels).all() or not np.isin(numeric_labels, (0, 1)).all():
        raise ValueError("labels must contain only binary values 0 and 1")
    y_true = numeric_labels.astype(np.int64)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    predicted = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    true_negatives, false_positives, false_negatives, true_positives = (
        int(value) for value in matrix.ravel()
    )
    positive_count = int(y_true.sum())
    metrics: dict[str, Any] = {
        "count": int(y_true.size),
        "fraud_count": positive_count,
        "fraud_ratio": float(positive_count / y_true.size),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, scores)) if np.unique(y_true).size == 2 else None
        ),
        "average_precision": (
            float(average_precision_score(y_true, scores))
            if np.unique(y_true).size == 2
            else None
        ),
        "false_positive_rate": (
            float(false_positives / (false_positives + true_negatives))
            if false_positives + true_negatives
            else None
        ),
        "confusion_matrix": {
            "true_negative": true_negatives,
            "false_positive": false_positives,
            "false_negative": false_negatives,
            "true_positive": true_positives,
        },
        "precision_at_k": {},
        "recall_at_k": {},
    }
    order = np.argsort(-scores, kind="stable")
    for requested_k in precision_recall_k:
        k = int(requested_k)
        if k < 1:
            raise ValueError("Precision@K and Recall@K values must be positive")
        selected_count = min(k, y_true.size)
        true_positives_at_k = int(y_true[order[:selected_count]].sum())
        key = str(k)
        metrics["precision_at_k"][key] = float(
            true_positives_at_k / selected_count
        )
        metrics["recall_at_k"][key] = (
            float(true_positives_at_k / positive_count) if positive_count else None
        )
    return metrics
