"""Run chronological GAAP/adaptation experiments and generate research reports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark import run_benchmark  # noqa: E402
from evaluation.latency import measure_prediction_latency  # noqa: E402
from evaluation.metrics import evaluate_binary_predictions  # noqa: E402
from evaluation.reports import generate_reports  # noqa: E402
from evaluation.streaming_evaluation import evaluate_streaming  # noqa: E402
from models.banking_gaap import BankingGAAP  # noqa: E402
from preprocessing.synthetic_data import generate_synthetic_transactions  # noqa: E402
from scripts.run_adaptation_experiment import (  # noqa: E402
    _load_config as _load_adaptation_config,
    _make_shifted_transactions,
    run_experiment as run_adaptation_experiment,
)
from scripts.run_drift_experiment import (  # noqa: E402
    load_config as _load_drift_config,
    run_experiment as run_drift_experiment,
)

LOGGER = logging.getLogger("adaptive_gaap.evaluation")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evaluation.yaml"


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluation configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Evaluation configuration must be a mapping: {config_path}")
    return config


def _prediction_metrics(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return evaluate_binary_predictions(
        [row["true_label"] for row in records],
        [row["fraud_probability"] for row in records],
        threshold=threshold,
    )


def _graph_sample(model: BankingGAAP, limit: int = 20) -> dict[str, Any] | None:
    if model.graph is None:
        return None
    graph = model.graph
    nodes = graph.node_table.sort_values("timestamp", kind="stable").head(limit)
    node_indexes = {
        int(index): str(row.transaction_id)
        for index, row in nodes.iterrows()
    }
    snapshot = pd.to_datetime(nodes["timestamp"], utc=True).max()
    node_records = [
        {
            "transaction_id": str(row.transaction_id),
            "timestamp": pd.to_datetime(row.timestamp, utc=True).isoformat(),
            "label": None if pd.isna(getattr(row, "label", None)) else int(row.label),
        }
        for row in nodes.itertuples(index=False)
    ]
    edges = []
    for edge in graph.edge_table.itertuples(index=False):
        source = node_indexes.get(int(edge.source))
        destination = node_indexes.get(int(edge.destination))
        edge_time = pd.to_datetime(edge.edge_timestamp, utc=True)
        if source is None or destination is None or edge_time > snapshot:
            continue
        edges.append(
            {
                "source": source,
                "destination": destination,
                "relationship": str(edge.relationship_type),
                "timestamp": edge_time.isoformat(),
            }
        )
    return {
        "snapshot_timestamp": snapshot.isoformat(),
        "nodes": node_records,
        "edges": edges,
    }


def run_evaluation(
    config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train the existing Adaptive-GAAP experiment, benchmark, and report facts."""
    seed = int(config.get("seed", 42))
    benchmark_config = config.get("benchmark", {})
    experiment_config = config.get("drift_experiment", {})
    report_config = config.get("report", {})
    destination = Path(output_dir or config.get("output_dir", "reports"))
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    work = Path(work_dir or "artifacts/evaluation")
    if not work.is_absolute():
        work = PROJECT_ROOT / work
    work.mkdir(parents=True, exist_ok=True)

    adaptation_config = _load_adaptation_config(
        Path(experiment_config.get("adaptation_config", "configs/adaptation_config.yaml"))
    )
    adaptation_config["seed"] = seed
    if smoke:
        adaptation_config.setdefault("experiment", {})["max_epochs"] = int(
            experiment_config.get("smoke_epochs", 1)
        )
        adaptation_config.setdefault("experiment", {})["rows_per_period"] = int(
            experiment_config.get("smoke_rows_per_period", 132)
        )
    adaptation_report = run_adaptation_experiment(
        adaptation_config,
        output_dir=work / "adaptive_experiment",
        smoke=smoke,
    )
    adaptive_state_path = (
        work / "adaptive_experiment" / "adaptive_state" / "adaptive_gaap.json"
    )
    if not adaptive_state_path.is_file():
        raise FileNotFoundError(
            f"Adaptation experiment did not persist its effective config: {adaptive_state_path}"
        )
    adaptation_state = json.loads(adaptive_state_path.read_text(encoding="utf-8"))
    adaptation_config = adaptation_state["config"]

    drift_config = _load_drift_config(
        Path(experiment_config.get("drift_config", "configs/drift_config.yaml"))
    )
    drift_config["seed"] = seed
    drift_report = run_drift_experiment(
        drift_config,
        output_path=work / "drift_experiment.json",
    )

    row_count = int(adaptation_report["phases"]["A_initial_training_rows"])
    initial = generate_synthetic_transactions(row_count, seed=seed)
    post_start = pd.to_datetime(initial["timestamp"], utc=True).max() + pd.Timedelta(
        minutes=4
    )
    changed = _make_shifted_transactions(
        row_count,
        seed=seed + 1,
        start=post_start,
        prefix="post_drift",
    )
    validation = changed.iloc[int(len(changed) * 0.8) :].reset_index(drop=True)
    training_model = BankingGAAP.load(
        work / "adaptive_experiment" / "registry" / "versions" / "v1" / "model"
    )
    training_ids = set(training_model.split_ids["train"])
    benchmark_training = initial[
        initial["transaction_id"].astype(str).isin(training_ids)
    ].copy()
    if benchmark_training.empty:
        raise RuntimeError("GAAP baseline training partition is empty")

    configured_models = list(benchmark_config.get(
        "models", ("logistic_regression", "random_forest")
    ))
    optional_status: dict[str, str] = {}
    for optional_name in benchmark_config.get("optional_models", ("xgboost",)):
        if optional_name in configured_models:
            continue
        if importlib.util.find_spec(optional_name):
            configured_models.append(optional_name)
            optional_status[optional_name] = "available_and_enabled"
        else:
            optional_status[optional_name] = "not_installed_not_run"

    comparison_predictions: dict[str, pd.DataFrame] = {}
    model_names = {
        "old_gaap_fixed_baseline": "GAAP",
        "gaap_with_drift_detection": "GAAP + Drift Detection",
        "full_retrained_gaap": "Full-retrained-GAAP",
        "adaptive_gaap": "Adaptive-GAAP",
    }
    model_versions: dict[str, str] = {}
    for key, label in model_names.items():
        raw_predictions = adaptation_report["validation_predictions"].get(key)
        if key == "gaap_with_drift_detection":
            raw_predictions = adaptation_report["validation_predictions"].get(
                "old_gaap_fixed_baseline"
            )
        if not raw_predictions:
            raise RuntimeError(f"Missing actual validation predictions for {key}")
        frame = pd.DataFrame(raw_predictions).rename(columns={"true_label": "label"})
        if key == "gaap_with_drift_detection":
            frame["model_version"] = "v1-drift-monitored"
        comparison_predictions[label] = frame
        version_values = frame["model_version"].astype(str).unique()
        if len(version_values) != 1:
            raise RuntimeError(f"Expected one model version for comparison {key}")
        model_versions[label] = version_values[0]

    benchmark = run_benchmark(
        benchmark_training,
        validation,
        features=("amount", "sender_balance_before", "transaction_type", "channel"),
        model_names=configured_models,
        label_column="label",
        threshold=float(config.get("threshold", 0.5)),
        seed=seed,
        precision_recall_k=tuple(
            int(value)
            for value in benchmark_config.get("precision_recall_k", (10, 50, 100))
        ),
        external_predictions=comparison_predictions,
        latency_repeats=3 if smoke else int(benchmark_config.get("latency_repeats", 30)),
    )
    model_records = benchmark["models"]
    for record in model_records:
        record["model_version"] = model_versions.get(record["model"], record["model_version"])

    baseline_model = training_model
    baseline_latency = measure_prediction_latency(
        baseline_model.predict_proba,
        repeats=3 if smoke else int(benchmark_config.get("latency_repeats", 30)),
        warmup=1,
    )
    probability_distributions = {
        name: frame["fraud_probability"].astype(float).tolist()
        for name, frame in comparison_predictions.items()
    }
    for record in model_records:
        if record["model"] == "GAAP":
            record["latency"] = baseline_latency
            record["latency_scope"] = "full_saved_graph_batch_inference"

    drift_windows = drift_report["performance"]
    drift_period_metrics = {
        name: {
            key: value
            for key, value in metrics.items()
            if key in {
                "average_precision", "roc_auc", "precision", "recall",
                "f1", "false_positive_rate",
            }
        }
        for name, metrics in drift_windows.items()
    }
    streaming_windows = []
    streaming_window_size = int(
        config.get("temporal", {}).get("rolling_test_timestamps", 12)
    )
    for model_name, frame in comparison_predictions.items():
        windows = evaluate_streaming(
            frame,
            window_size=streaming_window_size,
            label_column="label",
            precision_recall_k=tuple(
                int(value)
                for value in benchmark_config.get(
                    "precision_recall_k", (10, 50, 100)
                )
            ),
        )
        for window in windows:
            streaming_windows.append(
                {
                    **window,
                    "model": model_name,
                }
            )
    event_records = drift_report.get("monitor_events", [])
    drift_scores = [
        {
            "timestamp": event.get("timestamp"),
            "drift_type": event.get("drift_type"),
            "score": event.get("drift_score"),
        }
        for event in event_records
        if event.get("drift_score") is not None
    ]
    drift_scores.sort(
        key=lambda item: pd.to_datetime(item["timestamp"], utc=True)
        if item.get("timestamp")
        else pd.Timestamp.min.tz_localize("UTC")
    )
    drift_start = pd.to_datetime(drift_report["drift_start"], utc=True)
    detected_events = [
        pd.to_datetime(item["timestamp"], utc=True)
        for item in drift_scores
        if item.get("timestamp") and pd.to_datetime(item["timestamp"], utc=True) >= drift_start
    ]
    detection_delay = (
        float((min(detected_events) - drift_start).total_seconds())
        if detected_events
        else None
    )

    recovery_model_names = {
        model_names["old_gaap_fixed_baseline"],
        model_names["full_retrained_gaap"],
        model_names["adaptive_gaap"],
    }
    recovery_records = [
        record
        for record in model_records
        if record.get("model") in recovery_model_names
    ]
    training_phases = adaptation_report["phases"]
    results: dict[str, Any] = {
        "seed": seed,
        "model_version": {
            record["model"]: record["model_version"] for record in model_records
        },
        "models": model_records,
        "validation_predictions": {
            record["model"]: record.get("prediction_records", [])
            for record in model_records
        },
        "fraud_probability_distributions": probability_distributions,
        "streaming_windows": streaming_windows,
        "drift_scores": drift_scores,
        "drift_experiment": {
            "model": drift_report["model"],
            "drift_timestamp": drift_start.isoformat(),
            "period_metrics": drift_period_metrics,
            "drift_scores": drift_scores,
            "detection_delay_seconds": detection_delay,
            "drift_events": event_records,
        },
        "adaptation_comparison": recovery_records,
        "operational": {
            "adaptation_time_seconds": training_phases.get("adaptation_time_seconds"),
            "full_retrain_training_time_seconds": training_phases.get(
                "full_retrain_training_time_seconds"
            ),
            "gaap_full_graph_inference": baseline_latency,
        },
        "example_graph": _graph_sample(baseline_model),
        "source_reports": {
            "adaptation_experiment": adaptation_report["report_path"],
            "drift_experiment": drift_report["report_path"],
        },
        "optional_model_status": optional_status,
        "ablation": {
            "status": "not_run",
            "reason": (
                "Ablation variants require separately implemented model configurations "
                "and scored outputs; no unmeasured ablation values are reported."
            ),
        },
        "evaluation_notes": {
            "gaap_with_drift_detection": (
                "Uses the fixed GAAP predictions while recording the observed drift events; "
                "monitoring does not alter the scorer."
            )
        },
    }

    # The three GAAP-family rows were already scored by the existing runner.
    # Recalculate their metrics through the shared evaluator for aligned detail.
    for record in model_records:
        rows = record.get("prediction_records", [])
        if rows:
            record["metrics"] = _prediction_metrics(
                rows, float(config.get("threshold", 0.5))
            )
    metadata = {
        "experiment": "chronological_gaap_adaptation_drift_comparison",
        "dataset": config.get("dataset", "synthetic_banking_transactions"),
        "model": "Original GAAP and Adaptive-GAAP with configurable classical baselines",
        "seed": seed,
        "threshold": float(config.get("threshold", 0.5)),
        "evaluation_window": {
            "start": pd.to_datetime(validation["timestamp"], utc=True).min().isoformat(),
            "end": pd.to_datetime(validation["timestamp"], utc=True).max().isoformat(),
            "transactions": len(validation),
        },
        "experiment_config": {
            "smoke": smoke,
            "models": configured_models,
            "effective_latency_repeats": (
                3 if smoke else int(benchmark_config.get("latency_repeats", 30))
            ),
            "ablation_status": results["ablation"]["status"],
            "evaluation": config,
            "adaptation": adaptation_config,
            "drift": drift_config,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    limitations = list(
        report_config.get("limitations", [])
    ) + [
        "The drift experiment uses a deterministic synthetic static-threshold scorer; it is a monitoring-control experiment, not a learned GAAP model.",
        "Classical baselines and the GAAP-family models use the same post-drift validation transactions, but different model-specific training procedures.",
        "GAAP latency measures batch prediction over its saved training graph; single-transaction online API latency is a separate measurement.",
        "Ablation variants are not emitted because corresponding scored model outputs were not supplied by this run.",
    ]
    paths = generate_reports(results, destination, metadata=metadata, limitations=limitations)
    return {"metadata": metadata, "results": results, "report_paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = run_evaluation(
        _load_config(arguments.config),
        output_dir=arguments.output_dir,
        work_dir=arguments.work_dir,
        smoke=arguments.smoke,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
