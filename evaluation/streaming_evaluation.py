"""Evaluate chronological prediction streams in fixed timestamp windows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from evaluation.metrics import evaluate_binary_predictions


def evaluate_streaming(
    predictions: pd.DataFrame,
    *,
    window_size: int,
    label_column: str = "true_label",
    probability_column: str = "fraud_probability",
    timestamp_column: str = "timestamp",
    transaction_id_column: str = "transaction_id",
    threshold: float = 0.5,
    precision_recall_k: tuple[int, ...] = (10, 50, 100),
) -> list[dict[str, Any]]:
    """Compute metrics over consecutive windows without reordering equal times."""
    required = {
        label_column,
        probability_column,
        timestamp_column,
        transaction_id_column,
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction stream is missing required columns: {sorted(missing)}")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    labeled = predictions.dropna(subset=[label_column]).copy()
    if labeled.empty:
        raise ValueError("Prediction stream contains no available labels")
    labeled[timestamp_column] = pd.to_datetime(
        labeled[timestamp_column], utc=True, errors="raise"
    )
    labeled = labeled.sort_values(timestamp_column, kind="stable")
    unique_times = list(labeled[timestamp_column].drop_duplicates())
    windows: list[dict[str, Any]] = []
    for start in range(0, len(unique_times), window_size):
        selected_times = unique_times[start : start + window_size]
        window = labeled[labeled[timestamp_column].isin(selected_times)]
        if window.empty:
            continue
        metrics = evaluate_binary_predictions(
            window[label_column],
            window[probability_column],
            threshold=threshold,
            precision_recall_k=precision_recall_k,
        )
        windows.append(
            {
                "window_index": len(windows),
                "start": window[timestamp_column].min().isoformat(),
                "end": window[timestamp_column].max().isoformat(),
                "transaction_ids": window[transaction_id_column].astype(str).tolist(),
                "model_versions": sorted(
                    window["model_version"].dropna().astype(str).unique().tolist()
                )
                if "model_version" in window
                else [],
                "metrics": metrics,
            }
        )
    return windows
