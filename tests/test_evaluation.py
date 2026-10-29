"""Tests for metrics, temporal safety, streaming, benchmarks, and reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.ablation import run_ablation
from evaluation.benchmark import run_benchmark
from evaluation.drift_experiment import compare_drift_periods
from evaluation.latency import measure_prediction_latency
from evaluation.metrics import evaluate_binary_predictions
from evaluation.reports import generate_reports
from evaluation.streaming_evaluation import evaluate_streaming
from evaluation.temporal_evaluation import rolling_windows, temporal_periods


def _transactions(count: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"tx-{index}" for index in range(count)],
            "timestamp": pd.date_range("2025-01-01", periods=count, freq="min", tz="UTC"),
            "amount": np.arange(count, dtype=float) + 1,
            "channel": ["mobile", "web"] * (count // 2),
            "label": [index % 2 for index in range(count)],
        }
    )


def test_metrics_include_ranking_operational_and_top_k_scores() -> None:
    result = evaluate_binary_predictions([0, 1, 0, 1], [0.1, 0.9, 0.8, 0.4], precision_recall_k=(1, 2))

    assert result["average_precision"] == pytest.approx(0.5 + 1 / 3)
    assert result["precision_at_k"]["1"] == 1
    assert result["recall_at_k"]["1"] == pytest.approx(0.5)
    assert result["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }
    assert evaluate_binary_predictions([1, 1], [0.1, 0.2])["roc_auc"] is None
    with pytest.raises(ValueError, match="binary"):
        evaluate_binary_predictions([0, 0.5], [0.1, 0.2])


def test_temporal_partitions_and_rolling_windows_never_overlap_timestamps() -> None:
    rows = _transactions()
    periods = temporal_periods(rows)
    ordered_periods = (periods.training, periods.validation, periods.test, periods.streaming)
    assert all(len(period) for period in ordered_periods)
    for earlier, later in zip(ordered_periods, ordered_periods[1:]):
        assert earlier["timestamp"].max() < later["timestamp"].min()

    windows = list(
        rolling_windows(rows, initial_train_timestamps=20, test_timestamps=5, step_timestamps=5)
    )
    assert len(windows) == 4
    for train, test in windows:
        assert train["timestamp"].max() < test["timestamp"].min()


def test_temporal_partition_keeps_duplicate_timestamps_in_one_period() -> None:
    rows = _transactions()
    rows.loc[1, "timestamp"] = rows.loc[0, "timestamp"]
    periods = temporal_periods(rows)
    memberships = {}
    for name, period in zip(
        ("training", "validation", "test", "streaming"),
        (periods.training, periods.validation, periods.test, periods.streaming),
    ):
        memberships.update({timestamp: name for timestamp in period["timestamp"].unique()})
    assert len(memberships) == rows["timestamp"].nunique()


def test_streaming_evaluation_uses_sorted_fixed_windows() -> None:
    rows = _transactions(8).rename(columns={"label": "true_label"})
    rows["fraud_probability"] = rows["true_label"].astype(float)
    rows["model_version"] = "v1"
    windows = evaluate_streaming(rows.iloc[::-1], window_size=3)

    assert [item["window_index"] for item in windows] == [0, 1, 2]
    assert windows[0]["start"] < windows[1]["start"] < windows[2]["start"]
    assert windows[0]["metrics"]["average_precision"] == 1


def test_latency_reports_measured_values() -> None:
    result = measure_prediction_latency(lambda: sum(range(20)), repeats=3, warmup=1)
    assert result["repeats"] == 3
    assert result["mean_latency_ms"] >= 0
    assert result["throughput_per_second"] > 0
    assert result["python_peak_memory_bytes"] >= 0


def test_benchmark_trains_configured_classical_baseline() -> None:
    rows = _transactions(40)
    training, test = rows.iloc[:30], rows.iloc[30:]
    result = run_benchmark(
        training,
        test,
        features=("amount", "channel"),
        model_names=("logistic_regression",),
        latency_repeats=2,
        seed=7,
    )
    assert result["models"][0]["model"] == "logistic_regression"
    assert result["models"][0]["seed"] == 7
    assert result["models"][0]["metrics"]["average_precision"] is not None


def test_benchmark_rejects_unavailable_requested_xgboost(monkeypatch) -> None:
    import evaluation.benchmark as benchmark_module

    import_module = benchmark_module.importlib.import_module

    def import_without_xgboost(name: str):
        if name == "xgboost":
            raise ImportError("simulated optional dependency unavailable")
        return import_module(name)

    monkeypatch.setattr(
        benchmark_module.importlib,
        "import_module",
        import_without_xgboost,
    )
    rows = _transactions(20)
    with pytest.raises(ImportError, match="xgboost"):
        run_benchmark(
            rows.iloc[:10],
            rows.iloc[10:],
            features=("amount",),
            model_names=("xgboost",),
        )


def test_ablation_requires_predictions_from_each_variant() -> None:
    rows = _transactions(10).rename(columns={"label": "true_label"})
    rows["fraud_probability"] = rows["true_label"].astype(float)
    rows["model_version"] = "v1"
    result = run_ablation({"gaap": lambda: rows}, seed=23)
    assert result["variants"][0]["metrics"]["f1"] == 1
    assert result["variants"][0]["model_version"] == "v1"


def test_drift_comparison_uses_labels_only_for_concept_metrics() -> None:
    class Monitor:
        def __init__(self):
            self.reference = None
            self.observed_timestamps = []

        def update_feature_distribution(self, current, reference_window, timestamp):
            assert "true_label" not in current
            assert "fraud_probability" not in current
            self.reference = reference_window
            return {"drift_detected": True, "drift_score": 0.4}

        def update_prediction(self, transaction_id, prediction, timestamp):
            self.observed_timestamps.append(pd.to_datetime(timestamp, utc=True))
            return {}

        def update_label(self, transaction_id, label, timestamp):
            return {}

        def check_drift(self):
            return {"drift_detected": True, "timestamp": "2025-01-01T00:05:00Z"}

    rows = _transactions(8).rename(columns={"label": "true_label"})
    rows["fraud_probability"] = rows["true_label"].astype(float)
    monitor = Monitor()
    result = compare_drift_periods(
        {"before": rows.iloc[:4], "after": rows.iloc[4:]},
        drift_timestamp="2025-01-01T00:04:00Z",
        monitor=monitor,
        reference_features=rows.iloc[:4],
        feature_columns=("amount",),
        model_version="gaap-v1",
    )
    assert result["drift_detection_delay_seconds"] == 60
    assert result["performance_change"]["average_precision"] == 0
    assert monitor.observed_timestamps == sorted(monitor.observed_timestamps)


def test_report_generation_writes_json_csv_markdown_and_available_figures(
    tmp_path: Path,
) -> None:
    results = {
        "seed": 3,
        "models": [
            {
                "model": "GAAP",
                "model_version": "v1",
                "seed": 3,
                "metrics": {"average_precision": 0.6, "recall": 0.5, "f1": 0.4},
                "latency": {"mean_latency_ms": 2.3},
            },
            {
                "model": "Adaptive-GAAP",
                "model_version": "v2",
                "seed": 3,
                "metrics": {"average_precision": 0.7, "recall": 0.6, "f1": 0.5},
            },
        ],
        "streaming_windows": [
            {"metrics": {"average_precision": 0.6, "recall": 0.4}},
            {"metrics": {"average_precision": 0.7, "recall": 0.6}},
        ],
        "fraud_probability_distributions": {"GAAP": [0.1, 0.7], "Adaptive-GAAP": [0.2, 0.9]},
    }
    outputs = generate_reports(
        results,
        tmp_path,
        metadata={"dataset": "synthetic", "seed": 3},
        limitations=["Synthetic test input."],
    )

    assert Path(outputs["json"]).is_file()
    assert Path(outputs["csv"]).is_file()
    assert Path(outputs["markdown"]).is_file()
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert payload["metadata"]["seed"] == 3
    assert len(payload["figures"]) >= 5
    assert all(Path(path).suffix == ".png" and Path(path).is_file() for path in payload["figures"])
    assert not any(Path(path).name == "ablation_comparison.png" for path in payload["figures"])
