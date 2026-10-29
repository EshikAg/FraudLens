"""Ablation runner for independently supplied, reproducible model variants."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from evaluation.metrics import evaluate_binary_predictions


def run_ablation(
    variants: Mapping[str, Callable[[], pd.DataFrame]],
    *,
    label_column: str = "true_label",
    probability_column: str = "fraud_probability",
    model_version_column: str = "model_version",
    threshold: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run each configured variant and calculate metrics from returned scores."""
    if not variants:
        raise ValueError("At least one ablation variant is required")
    results: list[dict[str, Any]] = []
    for name, run_variant in variants.items():
        predictions = run_variant()
        if label_column not in predictions or probability_column not in predictions:
            raise ValueError(
                f"Ablation variant {name!r} must return labels and probabilities"
            )
        labeled = predictions.dropna(subset=[label_column])
        version = (
            str(labeled[model_version_column].iloc[0])
            if model_version_column in labeled and not labeled.empty
            else name
        )
        results.append(
            {
                "model": name,
                "model_version": version,
                "seed": int(seed),
                "metrics": evaluate_binary_predictions(
                    labeled[label_column],
                    labeled[probability_column],
                    threshold=threshold,
                ),
            }
        )
    return {"seed": int(seed), "study": "ablation", "variants": results}
