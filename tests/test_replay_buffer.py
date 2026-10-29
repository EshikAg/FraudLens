"""Tests for bounded historical replay storage and temporal weighting."""

from __future__ import annotations

import pandas as pd
import pytest

from adaptation.replay_buffer import ReplayBuffer


def _transactions(count: int, start: str = "2025-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=count, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "transaction_id": [f"tx-{timestamp.date()}" for timestamp in timestamps],
            "timestamp": timestamps,
            "label": [index % 2 for index in range(count)],
            "amount": [float(index) for index in range(count)],
        }
    )


def test_buffer_retains_capacity_and_idempotently_adds_rows() -> None:
    buffer = ReplayBuffer(5, seed=3)
    transactions = _transactions(7)

    assert buffer.add(transactions) == 7
    assert buffer.add(transactions.iloc[-2:]) == 0
    assert len(buffer) == 5
    assert buffer.frame()["transaction_id"].tolist() == transactions.iloc[-5:][
        "transaction_id"
    ].tolist()


def test_buffer_rejects_conflicting_duplicate_ids_and_unlabeled_rows() -> None:
    buffer = ReplayBuffer(10)
    transactions = _transactions(4)
    buffer.add(transactions)
    conflicting = transactions.iloc[[0]].copy()
    conflicting["amount"] = 999.0

    with pytest.raises(ValueError, match="Conflicting replay data"):
        buffer.add(conflicting)
    assert buffer.add(pd.DataFrame({"transaction_id": ["unlabeled"], "timestamp": ["2025-02-01"], "label": [None]})) == 0


def test_sampling_is_seeded_unique_and_excludes_reserved_rows() -> None:
    transactions = _transactions(30)
    first = ReplayBuffer(40, seed=17)
    second = ReplayBuffer(40, seed=17)
    first.add(transactions)
    second.add(transactions)

    first_sample = first.sample(10, as_of="2025-02-01", exclude_ids=["tx-2025-01-30"])
    second_sample = second.sample(10, as_of="2025-02-01", exclude_ids=["tx-2025-01-30"])

    assert first_sample["transaction_id"].tolist() == second_sample["transaction_id"].tolist()
    assert len(first_sample) == first_sample["transaction_id"].nunique() == 10
    assert "tx-2025-01-30" not in first_sample["transaction_id"].tolist()
