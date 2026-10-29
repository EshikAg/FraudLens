"""Run a reproducible drift-triggered Adaptive-GAAP comparison experiment."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drift.drift_monitor import DriftMonitor  # noqa: E402
from drift.drift_report import save_drift_report  # noqa: E402
from models.adaptive_gaap import AdaptiveGAAP  # noqa: E402
from models.banking_gaap import BankingGAAP  # noqa: E402
from preprocessing.synthetic_data import generate_synthetic_transactions  # noqa: E402


LOGGER = logging.getLogger("adaptive_gaap.adaptation_experiment")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "adaptation_config.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Adaptation config does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Adaptation config must be a YAML mapping: {resolved}")
    return config


def _make_shifted_transactions(
    count: int,
    *,
    seed: int,
    start: pd.Timestamp,
    prefix: str,
) -> pd.DataFrame:
    """Create a post-drift fraud pattern of small, repeated related transfers."""
    frame = generate_synthetic_transactions(count, seed=seed)
    frame["transaction_id"] = [f"{prefix}_{index:05d}" for index in range(count)]
    frame["timestamp"] = [
        (start + pd.Timedelta(minutes=index * 4)).isoformat()
        for index in range(count)
    ]
    for index in frame.index[frame["label"].astype(int) == 1]:
        frame.loc[index, "amount"] = float(35 + (index % 5) * 7)
        frame.loc[index, "sender_account_id"] = f"related_account_{index % 3}"
        frame.loc[index, "receiver_account_id"] = f"new_beneficiary_{index % 2}"
        frame.loc[index, "device_id"] = f"new_device_{index % 2}"
        frame.loc[index, "ip_id"] = "new_network"
        frame.loc[index, "transaction_type"] = "UPI"
    return frame


def _metrics(
    labels: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = labels.astype(int).to_numpy()
    y_pred = (probabilities >= threshold).astype("int64")
    if set(y_true.tolist()) != {0, 1}:
        raise ValueError("Experiment evaluation window must contain both classes")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }


def run_experiment(
    config: dict[str, Any],
    *,
    output_dir: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train baseline, detect a shift, fine-tune, and compare all three models."""
    seed = int(config.get("seed", 42))
    rows = 132 if smoke else int(config.get("experiment", {}).get("rows_per_period", 220))
    epochs = 1 if smoke else int(
        config.get("experiment", {}).get("max_epochs", 3)
    )
    if rows < 100:
        raise ValueError("experiment.rows_per_period must be at least 100")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adaptive_config = copy.deepcopy(config)
    adaptive_config["registry_dir"] = str(output_dir / "registry")
    adaptive_config["incremental_training"]["max_epochs"] = epochs
    adaptive_config["incremental_training"]["patience"] = 1 if smoke else int(
        config.get("incremental_training", {}).get("patience", 2)
    )
    gaap_config = BankingGAAP(adaptive_config["gaap_config"]).config
    gaap_config["training"]["max_epochs"] = epochs
    gaap_config["training"]["patience"] = 1 if smoke else int(
        gaap_config["training"].get("patience", 2)
    )
    adaptive_config["gaap_config"] = gaap_config
    adaptive = AdaptiveGAAP(adaptive_config)
    initial = generate_synthetic_transactions(rows, seed=seed)
    adaptive.fit(initial)
    assert adaptive.baseline_model is not None
    baseline = adaptive.baseline_model

    post_start = pd.to_datetime(initial["timestamp"], utc=True).max() + pd.Timedelta(minutes=4)
    after = _make_shifted_transactions(
        rows,
        seed=seed + 1,
        start=post_start,
        prefix="post_drift",
    )
    split_index = int(len(after) * 0.8)
    recent_labeled = after.iloc[:split_index].reset_index(drop=True)
    validation = after.iloc[split_index:].reset_index(drop=True)
    target_for_drift = after.copy()
    true_labels = target_for_drift["label"].astype(int).copy()
    target_for_drift["label"] = pd.NA

    monitor = DriftMonitor(config.get("drift_monitor", {}))
    data_event = monitor.update_feature_distribution(
        target_for_drift,
        reference_window=initial,
        timestamp=pd.to_datetime(after["timestamp"], utc=True).max(),
    )
    scorer = adaptive.incremental_trainer
    old_predictions = scorer.predict_probabilities(
        baseline,
        target_for_drift,
        context_transactions=initial,
    )
    for row, label in zip(old_predictions.itertuples(index=False), true_labels):
        monitor.update_prediction(row.transaction_id, row.fraud_probability, row.timestamp)
        monitor.update_label(row.transaction_id, int(label), row.timestamp)
    drift_event = monitor.check_drift()
    if not drift_event["drift_detected"]:
        drift_event = data_event
    if not drift_event.get("drift_detected"):
        raise RuntimeError("Configured synthetic experiment did not detect drift")

    adaptation_started = time.perf_counter()
    adaptation_result = adaptive.adapt(
        recent_transactions=recent_labeled,
        validation_transactions=validation,
        drift_event=drift_event,
    )
    adaptation_seconds = time.perf_counter() - adaptation_started
    current_metrics = adaptation_result["metrics"]["current"]["validation"]
    baseline_metrics = adaptation_result["metrics"]["baseline"]["validation"]
    adaptive_metrics = adaptation_result["metrics"]["candidate"]["validation"]
    selected_adaptive_version = (
        adaptation_result["version"]
        if adaptation_result["promoted"]
        else adaptation_result["current_version"]
    )

    full_retrain_config = copy.deepcopy(adaptive.gaap_config)
    full_retrain_config["output_dir"] = str(output_dir / "full_retrain")
    full_retrain_config["training"]["max_epochs"] = epochs
    full_retrain_config["training"]["patience"] = 1 if smoke else int(
        full_retrain_config["training"].get("patience", 2)
    )
    retrained = BankingGAAP(full_retrain_config)
    retraining_rows = pd.concat([initial, recent_labeled], ignore_index=True)
    retraining_started = time.perf_counter()
    retrained.fit(retraining_rows)
    full_retrain_training_seconds = time.perf_counter() - retraining_started
    retrained_predictions = scorer.predict_probabilities(
        retrained,
        validation,
        context_transactions=retraining_rows,
    )
    retrained_metrics = _metrics(
        validation["label"],
        retrained_predictions["fraud_probability"].to_numpy(),
        retrained.gaap.threshold if retrained.gaap is not None else 0.5,
    )
    adaptive_candidate = adaptation_result.get("candidate_model")
    if adaptation_result["promoted"] and adaptive_candidate is not None:
        adaptive_model = adaptive_candidate
    else:
        adaptive_model = adaptive.current_model
    if adaptive_model is None:
        raise RuntimeError("No Adaptive-GAAP model is available for comparison")
    adapted_predictions = scorer.predict_probabilities(
        adaptive_model,
        validation,
        context_transactions=pd.concat(
            [initial, recent_labeled], ignore_index=True
        ),
    )
    adaptive_metrics = _metrics(
        validation["label"],
        adapted_predictions["fraud_probability"].to_numpy(),
        adaptive_model.gaap.threshold if adaptive_model.gaap is not None else 0.5,
    )

    def _validation_records(predictions: pd.DataFrame) -> list[dict[str, Any]]:
        labels = validation[["transaction_id", "timestamp", "label"]].copy()
        labels["transaction_id"] = labels["transaction_id"].astype(str)
        observed = predictions[["transaction_id", "fraud_probability", "model_version"]].copy()
        observed["transaction_id"] = observed["transaction_id"].astype(str)
        joined = labels.merge(observed, on="transaction_id", how="inner", validate="one_to_one")
        if len(joined) != len(validation):
            raise RuntimeError("Model predictions do not cover the full validation window")
        return [
            {
                "transaction_id": str(row.transaction_id),
                "timestamp": pd.to_datetime(row.timestamp, utc=True).isoformat(),
                "true_label": int(row.label),
                "fraud_probability": float(row.fraud_probability),
                "model_version": str(row.model_version),
            }
            for row in joined.itertuples(index=False)
        ]

    report = {
        "experiment": "adaptive_gaap_concept_drift",
        "seed": seed,
        "phases": {
            "A_initial_training_rows": int(len(initial)),
            "B_changed_behavior_rows": int(len(after)),
            "C_drift_event": drift_event,
            "D_adaptation": {
                key: value
                for key, value in adaptation_result.items()
                if key != "candidate_model"
            },
            "adaptation_time_seconds": adaptation_seconds,
            "full_retrain_training_time_seconds": full_retrain_training_seconds,
            "E_comparison_window_rows": int(len(validation)),
        },
        "drift_report": monitor.report(),
        "models": {
            "old_gaap_fixed_baseline": {
                "version": "v1",
                "metrics": baseline_metrics,
            },
            "full_retrained_gaap": {
                "metrics": retrained_metrics,
            },
            "adaptive_gaap": {
                "version": selected_adaptive_version,
                "promoted": bool(adaptation_result["promoted"]),
                "metrics": adaptive_metrics,
            },
            "adaptive_candidate": {
                "version": adaptation_result.get("version"),
                "promoted": bool(adaptation_result["promoted"]),
                "metrics": adaptation_result["metrics"]["candidate"]["validation"],
            },
            "current_before_adaptation": {
                "metrics": current_metrics,
            },
        },
        "validation_predictions": {
            "old_gaap_fixed_baseline": _validation_records(old_predictions),
            "full_retrained_gaap": _validation_records(retrained_predictions),
            "adaptive_gaap": _validation_records(adapted_predictions),
        },
        "promotion_decision": adaptation_result["promotion_decision"],
        "retraining_performed_for_candidate": False,
        "comparison_model_full_retrain_performed": True,
    }
    report_path = save_drift_report(report, output_dir / "adaptation_experiment.json")
    report["report_path"] = str(report_path)
    adaptive.save(output_dir / "adaptive_state")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "adaptation"
    )
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = run_experiment(
        _load_config(arguments.config),
        output_dir=arguments.output_dir,
        smoke=arguments.smoke,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
