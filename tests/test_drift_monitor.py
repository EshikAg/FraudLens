"""Integration tests for unified drift monitoring and reporting."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from drift.drift_monitor import DriftMonitor
from scripts.run_drift_experiment import load_config, run_experiment


def test_monitor_combines_events_saves_report_and_resets(tmp_path) -> None:
    monitor = DriftMonitor(
        {
            "feature_drift": {"features": ["amount"], "min_samples": 20},
            "concept_drift": {"threshold": 1.0, "delta": 0.0, "min_instances": 5},
        }
    )
    reference = pd.DataFrame({"amount": np.arange(40, dtype=float)})
    current = pd.DataFrame({"amount": np.arange(100, 140, dtype=float)})
    feature_status = monitor.update_feature_distribution(current, reference, "2025-01-02")
    assert feature_status["drift_detected"]
    for index in range(10):
        transaction_id = f"baseline-{index}"
        monitor.update_prediction(transaction_id, 0, "2025-01-02")
        monitor.update_label(transaction_id, 0, "2025-01-02")
    for index in range(20):
        transaction_id = f"tx-{index}"
        monitor.update_prediction(transaction_id, 0, "2025-01-03")
        monitor.update_label(transaction_id, 1, "2025-01-03")

    status = monitor.check_drift()
    assert status["drift_detected"]
    assert status["drift_type"] == "both"
    assert "amount" in status["affected_features"]

    output = monitor.save_report(tmp_path / "drift.json")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert len(saved["events"]) >= 2
    monitor.reset(reference)
    assert not monitor.check_drift()["drift_detected"]


def test_synthetic_experiment_reports_drift_and_performance(tmp_path) -> None:
    config = load_config("configs/drift_config.yaml")
    report = run_experiment(config, tmp_path / "synthetic_report.json")

    assert report["data_drift"]["drift_detected"]
    assert report["concept_drift"] is not None
    assert report["concept_drift"]["timestamp"] >= report["drift_start"]
    assert report["performance"]["before_drift"]["f1"] > report["performance"]["after_drift"]["f1"]
    assert report["model"]["retraining_performed"] is False
    assert (tmp_path / "synthetic_report.json").is_file()
