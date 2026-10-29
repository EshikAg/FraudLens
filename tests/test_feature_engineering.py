"""Tests for deterministic and causal transaction feature engineering."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from preprocessing.feature_engineering import TransactionFeatureGenerator
from preprocessing.synthetic_data import generate_synthetic_transactions
from preprocessing.temporal_split import temporal_split
from preprocessing.transaction_schema import TransactionSchema


def _canonical(rows: list[dict[str, object]]) -> pd.DataFrame:
    return TransactionSchema.default().normalize(pd.DataFrame(rows))


def test_features_are_deterministic_and_historical() -> None:
    transactions = TransactionSchema.default().normalize(
        generate_synthetic_transactions(60)
    )
    generator = TransactionFeatureGenerator.from_yaml("configs/feature_config.yaml")

    first = generator.transform(transactions)
    second = generator.transform(transactions)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "tx_count_prev_1h"] == 0
    assert first.loc[1, "tx_count_prev_1h"] >= 0
    assert "suspicious_neighbour_count" in first.attrs["feature_columns"]


def test_future_rows_and_labels_cannot_change_earlier_features() -> None:
    original = _canonical(
        [
            {
                "transaction_id": "t1",
                "timestamp": "2025-01-01T10:00:00Z",
                "sender_account_id": "a",
                "receiver_account_id": "b",
                "amount": 10,
                "device_id": "d",
                "ip_id": "ip",
                "label": 0,
            },
            {
                "transaction_id": "t2",
                "timestamp": "2025-01-01T10:30:00Z",
                "sender_account_id": "a",
                "receiver_account_id": "c",
                "amount": 20,
                "device_id": "d",
                "ip_id": "ip",
                "label": 0,
            },
        ]
    )
    future = _canonical(
        [
            {
                "transaction_id": "future",
                "timestamp": "2025-01-02T10:00:00Z",
                "sender_account_id": "a",
                "receiver_account_id": "b",
                "amount": 9000,
                "device_id": "future_device",
                "ip_id": "future_ip",
                "label": 1,
            }
        ]
    )
    full = pd.concat([original, future], ignore_index=True)
    generator = TransactionFeatureGenerator()

    prefix_features = generator.transform(original)
    full_features = generator.transform(full).iloc[: len(original)]
    columns = prefix_features.attrs["feature_columns"]
    pd.testing.assert_frame_equal(
        prefix_features[columns], full_features[columns], check_dtype=False
    )
    assert full_features.loc[1, "suspicious_neighbour_count"] == 0


def test_equal_timestamp_rows_do_not_observe_each_other() -> None:
    transactions = _canonical(
        [
            {
                "transaction_id": "t1",
                "timestamp": "2025-01-01T10:00:00Z",
                "sender_account_id": "a",
                "receiver_account_id": "b",
                "amount": 10,
            },
            {
                "transaction_id": "t2",
                "timestamp": "2025-01-01T10:00:00Z",
                "sender_account_id": "a",
                "receiver_account_id": "c",
                "amount": 20,
            },
        ]
    )

    features = TransactionFeatureGenerator().transform(transactions)

    assert features["tx_count_prev_1h"].tolist() == [0, 0]
    assert features["new_beneficiary"].tolist() == [True, True]


def test_future_label_availability_does_not_change_earlier_features() -> None:
    prefix = _canonical(
        [
            {
                "transaction_id": "t1",
                "timestamp": "2025-01-01T10:00:00Z",
                "sender_account_id": "a",
                "receiver_account_id": "b",
                "amount": 10,
            }
        ]
    )
    future = _canonical(
        [
            {
                "transaction_id": "t2",
                "timestamp": "2025-01-01T11:00:00Z",
                "sender_account_id": "b",
                "receiver_account_id": "c",
                "amount": 20,
                "label": 1,
            }
        ]
    )
    full = pd.concat([prefix, future], ignore_index=True)
    generator = TransactionFeatureGenerator()

    prefix_features = generator.transform(prefix)
    full_features = generator.transform(full).iloc[:1]

    assert pd.isna(prefix_features.loc[0, "suspicious_neighbour_count"])
    assert pd.isna(full_features.loc[0, "suspicious_neighbour_count"])


def test_rapid_beneficiary_changes_requires_a_new_beneficiary() -> None:
    transactions = _canonical(
        [
            {
                "transaction_id": f"t{index}",
                "timestamp": f"2025-01-01T10:{index * 5:02d}:00Z",
                "sender_account_id": "a",
                "receiver_account_id": receiver,
                "amount": 10,
            }
            for index, receiver in enumerate(
                ("b1", "b2", "b3", "b4", "b4")
            )
        ]
    )

    features = TransactionFeatureGenerator().transform(transactions)

    assert features.loc[3, "rapid_beneficiary_changes"]
    assert not features.loc[4, "rapid_beneficiary_changes"]


def test_missing_features_are_reported_and_left_missing() -> None:
    transactions = _canonical(
        [
            {
                "transaction_id": "t1",
                "timestamp": "2025-01-01T10:00:00Z",
                "amount": 10,
            }
        ]
    )

    features = TransactionFeatureGenerator().transform(transactions)

    assert pd.isna(features.loc[0, "tx_count_prev_1h"])
    assert features.attrs["feature_availability"]["features"]["tx_count_prev_1h"][
        "available"
    ] is False
    assert features.attrs["feature_availability"]["features"]["amount"][
        "available"
    ] is True


def test_feature_availability_report_can_be_persisted(tmp_path) -> None:
    transactions = _canonical(
        [
            {
                "transaction_id": "t1",
                "timestamp": "2025-01-01T10:00:00Z",
                "sender_account_id": "a",
                "amount": 10,
            }
        ]
    )
    report_path = tmp_path / "reports" / "availability.json"

    TransactionFeatureGenerator().transform(
        transactions, availability_report_path=report_path
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["features"]["tx_count_prev_1h"]["available"] is True
    assert report["features"]["device_degree"]["available"] is False


def test_suspicious_numeric_label_with_fractional_storage_is_recognized() -> None:
    transactions = _canonical(
        [
            {
                "transaction_id": "t1",
                "timestamp": "2025-01-01T10:00:00Z",
                "sender_account_id": "a",
                "receiver_account_id": "b",
                "amount": 10,
                "label": 1.0,
            },
            {
                "transaction_id": "t2",
                "timestamp": "2025-01-01T11:00:00Z",
                "sender_account_id": "b",
                "receiver_account_id": "c",
                "amount": 10,
                "label": 0.0,
            },
        ]
    )

    features = TransactionFeatureGenerator().transform(transactions)

    assert features.loc[1, "suspicious_neighbour_count"] == 1


def test_temporal_split_keeps_timestamp_groups_together() -> None:
    transactions = _canonical(
        [
            {
                "transaction_id": f"t{i}",
                "timestamp": f"2025-01-01T{i:02d}:00:00Z",
                "amount": i,
            }
            for i in range(8)
        ]
    )

    split = temporal_split(transactions, 0.5, 0.25)

    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()


def test_temporal_generator_rejects_missing_timestamps() -> None:
    transactions = _canonical([{"transaction_id": "t1", "amount": 10}])

    with pytest.raises(ValueError, match="requires a timestamp"):
        TransactionFeatureGenerator().transform(transactions)
