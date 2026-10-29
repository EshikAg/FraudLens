"""Measure fixed-model performance and drift outcomes using observed predictions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from evaluation.metrics import evaluate_binary_predictions


def compare_drift_periods(
    periods: Mapping[str, pd.DataFrame],
    *,
    drift_timestamp: Any,
    monitor: Any | None = None,
    reference_features: pd.DataFrame | None = None,
    feature_columns: tuple[str, ...] = (),
    label_column: str = "true_label",
    probability_column: str = "fraud_probability",
    timestamp_column: str = "timestamp",
    transaction_id_column: str = "transaction_id",
    model_version: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Report measured metrics per period and optionally update a DriftMonitor.

    Feature-drift comparisons intentionally strip labels and prediction outputs
    before passing frames to the monitor.
    """
    if not periods:
        raise ValueError("At least one labeled prediction period is required")
    if not model_version:
        raise ValueError("model_version must be provided")
    period_metrics: dict[str, Any] = {}
    concatenated: list[pd.DataFrame] = []
    for name, frame in periods.items():
        required = {label_column, probability_column, timestamp_column}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Period {name!r} is missing columns: {sorted(missing)}")
        labeled = frame.dropna(subset=[label_column])
        if labeled.empty:
            raise ValueError(f"Period {name!r} has no labeled predictions")
        period_metrics[name] = evaluate_binary_predictions(
            labeled[label_column],
            labeled[probability_column],
            threshold=threshold,
        )
        concatenated.append(labeled)

    drift_result: dict[str, Any] | None = None
    detection_delay_seconds: float | None = None
    if monitor is not None:
        if transaction_id_column not in pd.concat(concatenated, ignore_index=True):
            raise ValueError(
                f"Prediction periods must include {transaction_id_column!r} for drift monitoring"
            )
        if reference_features is not None:
            if not feature_columns:
                raise ValueError("feature_columns are required with reference_features")
            missing_features = set(feature_columns).difference(reference_features.columns)
            if missing_features:
                raise ValueError(f"Reference features are missing: {sorted(missing_features)}")
            current_frames = []
            for frame in periods.values():
                missing_current = set(feature_columns).difference(frame.columns)
                if missing_current:
                    raise ValueError(f"Current features are missing: {sorted(missing_current)}")
                current_frames.append(frame[list(feature_columns)])
            current = pd.concat(current_frames, ignore_index=True)
            reference = reference_features[list(feature_columns)].copy()
            drift_result = monitor.update_feature_distribution(
                current,
                reference_window=reference,
                timestamp=drift_timestamp,
            )
        all_predictions = pd.concat(concatenated, ignore_index=True)
        all_predictions[timestamp_column] = pd.to_datetime(
            all_predictions[timestamp_column], utc=True, errors="raise"
        )
        all_predictions = all_predictions.sort_values(
            timestamp_column, kind="stable"
        )
        for row in all_predictions.itertuples(index=False):
            values = row._asdict()
            transaction_id = str(values.get(transaction_id_column, ""))
            if not transaction_id:
                continue
            timestamp = values[timestamp_column]
            score = float(values[probability_column])
            label = int(values[label_column])
            monitor.update_prediction(transaction_id, score, timestamp)
            monitor.update_label(transaction_id, label, timestamp)
        concept_status = monitor.check_drift()
        drift_result = {
            "drift_detected": bool(
                (drift_result or {}).get("drift_detected")
                or concept_status.get("drift_detected")
            ),
            "feature_drift": drift_result,
            "concept_drift": concept_status,
        }
        observed_timestamp = concept_status.get("timestamp")
        if concept_status.get("drift_detected") and observed_timestamp:
            observed = pd.to_datetime(observed_timestamp, utc=True)
            known = pd.to_datetime(drift_timestamp, utc=True)
            if observed >= known:
                detection_delay_seconds = float((observed - known).total_seconds())

    return {
        "model_version": model_version,
        "drift_timestamp": pd.to_datetime(drift_timestamp, utc=True).isoformat(),
        "period_metrics": period_metrics,
        "performance_change": _performance_change(period_metrics),
        "drift": drift_result,
        "drift_detection_delay_seconds": detection_delay_seconds,
    }


def _performance_change(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, float | None]:
    names = list(metrics)
    if len(names) < 2:
        return {}
    first, last = metrics[names[0]], metrics[names[-1]]
    changes: dict[str, float | None] = {}
    for metric in ("average_precision", "recall", "precision", "f1", "roc_auc"):
        before, after = first.get(metric), last.get(metric)
        changes[metric] = (
            float(after - before)
            if before is not None and after is not None
            else None
        )
    return changes
