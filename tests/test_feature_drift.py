"""Tests for statistical feature-distribution drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drift.feature_drift import FeatureDriftDetector


def test_feature_drift_detects_numeric_and_categorical_changes() -> None:
    reference = pd.DataFrame(
        {
            "amount": np.linspace(10.0, 20.0, 100),
            "channel": ["mobile"] * 90 + ["web"] * 10,
        }
    )
    current = pd.DataFrame(
        {
            "amount": np.linspace(100.0, 200.0, 100),
            "channel": ["web"] * 95 + ["mobile"] * 5,
        }
    )
    detector = FeatureDriftDetector(
        {
            "features": ["amount", "channel"],
            "min_samples": 20,
            "ks_alpha": 0.01,
        }
    )
    detector.set_reference(reference)

    result = detector.compare(current, timestamp="2025-01-02T00:00:00Z")

    assert result["drift_detected"]
    assert set(result["affected_features"]) == {"amount", "channel"}
    assert result["window_statistics"]["features"]["amount"]["ks_p_value"] < 0.01
    assert result["window_statistics"]["features"]["channel"]["jensen_shannon"] > 0.1


def test_identical_windows_do_not_trigger_and_missing_features_are_reported() -> None:
    reference = pd.DataFrame({"amount": np.arange(40, dtype=float), "channel": ["upi"] * 40})
    detector = FeatureDriftDetector(
        {"features": ["amount", "channel", "device_usage"], "min_samples": 20}
    )
    detector.set_reference(reference)

    result = detector.compare(reference.copy())

    assert not result["drift_detected"]
    assert result["window_statistics"]["features"]["device_usage"]["reason"] == "feature_missing"


def test_consecutive_windows_reduce_single_window_false_alarms() -> None:
    reference = pd.DataFrame({"amount": np.arange(40, dtype=float)})
    shifted = pd.DataFrame({"amount": np.arange(100, 140, dtype=float)})
    detector = FeatureDriftDetector(
        {"features": ["amount"], "min_samples": 20, "consecutive_windows": 2}
    )
    detector.set_reference(reference)

    assert not detector.compare(shifted)["drift_detected"]
    assert detector.compare(shifted)["drift_detected"]


def test_missing_value_rate_shift_is_detected_even_with_few_observed_values() -> None:
    reference = pd.DataFrame({"amount": [10.0] * 40})
    current = pd.DataFrame({"amount": [None] * 35 + [10.0] * 5})
    detector = FeatureDriftDetector(
        {
            "features": ["amount"],
            "min_samples": 20,
            "missing_rate_threshold": 0.1,
        }
    )
    detector.set_reference(reference)

    result = detector.compare(current)

    assert result["drift_detected"]
    assert result["window_statistics"]["features"]["amount"]["triggered_methods"] == [
        "missing_rate"
    ]
