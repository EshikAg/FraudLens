"""Tests for canonical transaction mapping and CSV loading."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from preprocessing.synthetic_data import generate_synthetic_transactions
from preprocessing.transaction_loader import load_transactions
from preprocessing.transaction_schema import CANONICAL_FIELDS, TransactionSchema


def test_schema_creates_canonical_columns_and_availability_report(
    tmp_path,
) -> None:
    input_path = tmp_path / "transactions.csv"
    report_path = tmp_path / "availability.json"
    pd.DataFrame(
        {
            "id": ["t1"],
            "date": ["2025-01-01T12:00:00Z"],
            "from_account": ["a1"],
            "value": [12.5],
            "type": ["UPI"],
        }
    ).to_csv(input_path, index=False)

    transactions = load_transactions(
        input_path,
        TransactionSchema.from_yaml("configs/transaction_schema.yaml"),
        report_path,
    )

    assert tuple(transactions.columns) == CANONICAL_FIELDS
    assert transactions.loc[0, "transaction_id"] == "t1"
    assert transactions.loc[0, "amount"] == 12.5
    assert pd.isna(transactions.loc[0, "receiver_account_id"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["fields"]["receiver_account_id"]["available"] is False


def test_missing_optional_fields_are_nullable() -> None:
    canonical = TransactionSchema.default().normalize(
        pd.DataFrame({"timestamp": ["2025-01-01"], "amount": [5]})
    )

    assert canonical.loc[0, "sender_account_id"] is pd.NA
    assert canonical.loc[0, "device_id"] is pd.NA
    assert canonical.loc[0, "timestamp"] == pd.Timestamp("2025-01-01", tz="UTC")


def test_invalid_numeric_source_value_is_reported() -> None:
    with pytest.raises(ValueError, match="Invalid numeric values"):
        TransactionSchema.default().normalize(
            pd.DataFrame({"timestamp": ["2025-01-01"], "amount": ["not-money"]})
        )


def test_synthetic_generator_is_seeded_and_writes_csv(tmp_path) -> None:
    output = tmp_path / "raw" / "synthetic.csv"
    first = generate_synthetic_transactions(40, seed=7, output_path=output)
    second = generate_synthetic_transactions(40, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert output.is_file()
    assert first["transaction_type"].nunique() > 1
    assert first["device_id"].nunique() < len(first)
    assert first["ip_id"].nunique() < len(first)
    assert first["receiver_account_id"].eq("shared_beneficiary").any()
