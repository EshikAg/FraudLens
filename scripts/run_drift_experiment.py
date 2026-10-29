"""Run a deterministic synthetic data- and concept-drift experiment."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOGGER = logging.getLogger("adaptive_gaap.drift_experiment")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "drift_config.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drift.drift_monitor import DriftMonitor  # noqa: E402
from drift.drift_report import save_drift_report  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the drift experiment configuration."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Drift configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Drift configuration must be a YAML mapping: {config_path}")
    for section in ("feature_drift", "concept_drift", "experiment"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Drift configuration must define {section!r}")
    return config


def _metric_summary(labels: list[int], predictions: list[int], probabilities: list[float]) -> dict[str, float]:
    """Calculate standard binary fraud metrics with explicit single-class handling."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities))
        if len(set(labels)) == 2
        else float("nan"),
        "average_precision": float(average_precision_score(labels, probabilities))
        if len(set(labels)) == 2
        else float("nan"),
    }


def _make_period(
    count: int,
    *,
    after_drift: bool,
    amount_threshold: float,
    rng: np.random.Generator,
    start: pd.Timestamp,
) -> tuple[pd.DataFrame, list[int], list[int], list[float]]:
    """Create behavior windows and fixed amount-threshold model outputs."""
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    for index in range(count):
        fraud = index % 8 == 0
        if after_drift and fraud:
            amount = float(rng.uniform(25.0, 90.0))
            velocity = 6.0
            device_usage = "new_device"
            beneficiary = "new_beneficiary"
            degree = 2.0
            transaction_type = "UPI"
            embedding = -0.6
        elif fraud:
            amount = float(rng.uniform(650.0, 1100.0))
            velocity = 1.0
            device_usage = "known_device"
            beneficiary = "known_beneficiary"
            degree = 8.0
            transaction_type = "BANK_TRANSFER"
            embedding = 0.8
        else:
            amount = float(rng.uniform(45.0, 160.0))
            velocity = float(rng.uniform(0.0, 2.0))
            device_usage = "known_device"
            beneficiary = "known_beneficiary"
            degree = float(rng.integers(4, 10))
            transaction_type = ("UPI", "IMPS", "CARD_PAYMENT")[index % 3]
            embedding = float(rng.normal(0.0, 0.15))
        probability = 0.95 if amount >= amount_threshold else 0.05
        predicted = int(probability >= 0.5)
        labels.append(int(fraud))
        predictions.append(predicted)
        probabilities.append(probability)
        rows.append(
            {
                "transaction_id": f"{'post' if after_drift else 'pre'}_{index:04d}",
                "timestamp": start + pd.Timedelta(minutes=index),
                "amount": amount,
                "transaction_velocity_1h": velocity,
                "transaction_type": transaction_type,
                "device_usage": device_usage,
                "ip_usage": "shared_ip" if after_drift and fraud else "known_ip",
                "beneficiary_behaviour": beneficiary,
                "graph_degree": degree,
                "transaction_embedding_mean": embedding,
            }
        )
    return pd.DataFrame(rows), labels, predictions, probabilities


def run_experiment(config: dict[str, Any], output_path: str | Path | None = None) -> dict[str, Any]:
    """Run a fixed, non-adaptive model against normal and shifted synthetic data."""
    experiment = config["experiment"]
    count = int(experiment.get("rows_per_period", 80))
    threshold = float(experiment.get("static_amount_threshold", 300.0))
    if count < 20:
        raise ValueError("experiment.rows_per_period must be at least 20")
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    drift_start = pd.Timestamp(experiment.get("drift_timestamp", "2025-02-01T00:00:00Z"))
    before, before_labels, before_predictions, before_probabilities = _make_period(
        count,
        after_drift=False,
        amount_threshold=threshold,
        rng=rng,
        start=drift_start - pd.Timedelta(minutes=count),
    )
    after, after_labels, after_predictions, after_probabilities = _make_period(
        count,
        after_drift=True,
        amount_threshold=threshold,
        rng=rng,
        start=drift_start,
    )

    monitor_config = {
        "feature_drift": config["feature_drift"],
        "concept_drift": config["concept_drift"],
    }
    monitor = DriftMonitor(monitor_config)
    feature_result = monitor.update_feature_distribution(
        after,
        reference_window=before,
        timestamp=after["timestamp"].max(),
    )

    for frame, labels, predictions, probabilities in (
        (before, before_labels, before_predictions, before_probabilities),
        (after, after_labels, after_predictions, after_probabilities),
    ):
        for row, label, prediction, probability in zip(
            frame.itertuples(index=False), labels, predictions, probabilities
        ):
            monitor.update_prediction(row.transaction_id, probability, row.timestamp)
            monitor.update_label(row.transaction_id, label, row.timestamp)
    concept_result = next(
        (
            event
            for event in monitor.report()["events"]
            if event["drift_type"] == "concept"
        ),
        None,
    )
    if concept_result is None:
        LOGGER.warning("Concept drift detector did not cross its configured threshold")

    report = {
        "experiment": "synthetic_static_model_drift",
        "seed": seed,
        "model": {
            "type": "fixed_amount_threshold",
            "amount_threshold": threshold,
            "retraining_performed": False,
        },
        "drift_start": drift_start.isoformat(),
        "data_drift": feature_result,
        "concept_drift": concept_result,
        "performance": {
            "before_drift": _metric_summary(
                before_labels, before_predictions, before_probabilities
            ),
            "after_drift": _metric_summary(
                after_labels, after_predictions, after_probabilities
            ),
        },
        "window_sizes": {"before_drift": len(before), "after_drift": len(after)},
        "monitor_events": monitor.report()["events"],
    }
    destination_value = output_path or config.get(
        "output_path", "artifacts/drift/synthetic_drift_report.json"
    )
    destination = Path(destination_value)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    report["report_path"] = str(save_drift_report(report, destination))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_experiment(load_config(arguments.config), arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
