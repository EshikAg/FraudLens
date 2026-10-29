"""Tests for delayed-label concept drift monitoring."""

from __future__ import annotations

from drift.concept_drift import PageHinkleyConceptDetector


def test_page_hinkley_detects_increasing_prediction_error() -> None:
    detector = PageHinkleyConceptDetector(
        {"threshold": 1.5, "delta": 0.0, "min_instances": 10, "window_size": 100}
    )
    for index in range(30):
        detector.update_prediction(f"before-{index}", 0, f"2025-01-01T00:{index:02d}:00Z")
        detector.update_label(f"before-{index}", 0)

    detected = None
    for index in range(40):
        transaction_id = f"after-{index}"
        detector.update_prediction(transaction_id, 0, f"2025-01-02T00:{index:02d}:00Z")
        result = detector.update_label(transaction_id, 1)
        if result["drift_detected"] and detected is None:
            detected = result

    assert detected is not None
    assert detected["drift_type"] == "concept"
    assert detected["window_statistics"]["detector"] == "page_hinkley"


def test_labels_can_arrive_before_predictions() -> None:
    detector = PageHinkleyConceptDetector({"min_instances": 2})

    detector.update_label("tx-1", 1)
    result = detector.update_prediction("tx-1", 0.9)

    assert result["window_statistics"]["labeled_predictions"] == 1
    assert result["window_statistics"]["false_positives"] == 0
    assert result["window_statistics"]["false_negatives"] == 0


def test_duplicate_predictions_and_non_binary_labels_are_rejected() -> None:
    detector = PageHinkleyConceptDetector()
    detector.update_prediction("tx-1", 0)

    try:
        detector.update_prediction("tx-1", 1)
    except ValueError as error:
        assert "already exists" in str(error)
    else:
        raise AssertionError("duplicate prediction should be rejected")

    try:
        detector.update_label("tx-1", 2)
    except ValueError as error:
        assert "binary" in str(error)
    else:
        raise AssertionError("non-binary label should be rejected")
