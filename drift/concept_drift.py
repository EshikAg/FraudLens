"""Page-Hinkley detection of degradation in delayed prediction outcomes."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _timestamp(value: Any | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    return pd.Timestamp(value).isoformat()


def _binary(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if value in (0, 1):
        return int(value)
    raise ValueError(f"{name} must be a binary value (0 or 1)")


class PageHinkleyConceptDetector:
    """Monitor binary classification error using a one-sided Page-Hinkley test."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        settings = dict(config or {})
        self.threshold = float(settings.get("threshold", 2.0))
        self.delta = float(settings.get("delta", 0.005))
        self.min_instances = int(settings.get("min_instances", 20))
        self.prediction_threshold = float(settings.get("prediction_threshold", 0.5))
        self.window_size = int(settings.get("window_size", 100))
        if self.threshold <= 0 or self.delta < 0:
            raise ValueError("Page-Hinkley threshold must be positive and delta non-negative")
        if self.min_instances < 2 or self.window_size < 2:
            raise ValueError("min_instances and window_size must be at least two")
        if not 0 <= self.prediction_threshold <= 1:
            raise ValueError("prediction_threshold must be between 0 and 1")
        self._pending_predictions: dict[str, tuple[int, float | None, str]] = {}
        self._pending_labels: dict[str, int] = {}
        self._outcomes: deque[dict[str, Any]] = deque(maxlen=self.window_size)
        self._reset_detector_state()
        self._latest = self._empty_result()

    def _reset_detector_state(self) -> None:
        self._count = 0
        self._mean_error = 0.0
        self._cumulative = 0.0
        self._minimum_cumulative = 0.0

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "drift_detected": False,
            "drift_type": None,
            "drift_score": 0.0,
            "timestamp": None,
            "affected_features": [],
            "window_statistics": {},
        }

    def update_prediction(
        self, transaction_id: str, prediction: int | float | bool, timestamp: Any | None = None
    ) -> dict[str, Any]:
        """Register a prediction before its label becomes available."""
        key = str(transaction_id)
        if not key:
            raise ValueError("transaction_id must not be empty")
        if key in self._pending_predictions:
            raise ValueError(f"Prediction already exists for transaction {key!r}")
        probability: float | None = None
        if isinstance(prediction, (float, np.floating)) and 0 <= float(prediction) <= 1:
            probability = float(prediction)
            predicted = int(probability >= self.prediction_threshold)
        else:
            predicted = _binary(prediction, "prediction")
        prediction_time = _timestamp(timestamp)
        self._pending_predictions[key] = (predicted, probability, prediction_time)
        if key in self._pending_labels:
            label = self._pending_labels.pop(key)
            self._record_outcome(key, predicted, label, probability, prediction_time)
        return dict(self._latest)

    def update_label(
        self, transaction_id: str, label: int | bool, timestamp: Any | None = None
    ) -> dict[str, Any]:
        """Join an available binary label to its prediction and update the detector."""
        key = str(transaction_id)
        if not key:
            raise ValueError("transaction_id must not be empty")
        actual = _binary(label, "label")
        prediction = self._pending_predictions.pop(key, None)
        if prediction is None:
            if key in self._pending_labels:
                raise ValueError(f"Label already exists for transaction {key!r}")
            self._pending_labels[key] = actual
            return dict(self._latest)
        predicted, probability, prediction_time = prediction
        observed_at = _timestamp(timestamp) if timestamp is not None else prediction_time
        self._record_outcome(key, predicted, actual, probability, observed_at)
        return dict(self._latest)

    def _record_outcome(
        self,
        transaction_id: str,
        predicted: int,
        label: int,
        probability: float | None,
        timestamp: str,
    ) -> None:
        error = float(predicted != label)
        self._outcomes.append(
            {
                "transaction_id": transaction_id,
                "predicted_label": predicted,
                "true_label": label,
                "error": error,
                "probability": probability,
                "timestamp": timestamp,
            }
        )
        self._count += 1
        self._mean_error += (error - self._mean_error) / self._count
        self._cumulative += error - self._mean_error - self.delta
        self._minimum_cumulative = min(self._minimum_cumulative, self._cumulative)
        score = max(0.0, self._cumulative - self._minimum_cumulative)
        detected = self._count >= self.min_instances and score >= self.threshold
        statistics = self.performance_statistics()
        self._latest = {
            "drift_detected": detected,
            "drift_type": "concept" if detected else None,
            "drift_score": float(score),
            "timestamp": timestamp,
            "affected_features": [],
            "window_statistics": {
                **statistics,
                "detector": "page_hinkley",
                "observations_since_reset": self._count,
                "mean_error_since_reset": float(self._mean_error),
                "threshold": self.threshold,
                "delta": self.delta,
            },
        }
        if detected:
            self._reset_detector_state()

    def performance_statistics(self) -> dict[str, Any]:
        """Return confusion counts and classification metrics for the recent window."""
        rows = list(self._outcomes)
        if not rows:
            return {
                "labeled_predictions": 0,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "error_rate": None,
            }
        tp = sum(row["predicted_label"] == row["true_label"] == 1 for row in rows)
        tn = sum(row["predicted_label"] == row["true_label"] == 0 for row in rows)
        fp = sum(row["predicted_label"] == 1 and row["true_label"] == 0 for row in rows)
        fn = sum(row["predicted_label"] == 0 and row["true_label"] == 1 for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        count = len(rows)
        return {
            "labeled_predictions": count,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "accuracy": float((tp + tn) / count),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "error_rate": float((fp + fn) / count),
        }

    def check_drift(self) -> dict[str, Any]:
        """Return the most recent detector status and current performance window."""
        return dict(self._latest)

    def reset(self) -> None:
        """Clear pending labels, prediction outcomes, and Page-Hinkley state."""
        self._pending_predictions.clear()
        self._pending_labels.clear()
        self._outcomes.clear()
        self._reset_detector_state()
        self._latest = self._empty_result()
