"""Tests for time-window snapshots and local temporal neighborhoods."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("dgl")

from graph.temporal_graph import TemporalGraph
from graph.transaction_graph import TransactionGraphBuilder
from preprocessing.transaction_schema import TransactionSchema


def _graph():
    transactions = TransactionSchema.default().normalize(
        pd.DataFrame(
            [
                {
                    "transaction_id": "t0",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "sender_account_id": "a",
                    "receiver_account_id": "b",
                    "amount": 1,
                },
                {
                    "transaction_id": "t1",
                    "timestamp": "2025-01-01T23:00:00Z",
                    "sender_account_id": "a",
                    "receiver_account_id": "c",
                    "amount": 2,
                },
                {
                    "transaction_id": "t2",
                    "timestamp": "2025-01-02T02:00:00Z",
                    "sender_account_id": "a",
                    "receiver_account_id": "d",
                    "amount": 3,
                },
                {
                    "transaction_id": "t3",
                    "timestamp": "2025-01-08T00:00:00Z",
                    "sender_account_id": "a",
                    "receiver_account_id": "e",
                    "amount": 4,
                },
            ]
        )
    )
    return TransactionGraphBuilder().build(transactions, full_history=True)


def test_full_historical_snapshot_obeys_as_of_cutoff() -> None:
    snapshot = TemporalGraph(_graph()).full_history(as_of="2025-01-02T00:00:00Z")

    assert snapshot.node_table["transaction_id"].tolist() == ["t0", "t1"]
    assert set(snapshot.edge_table["destination"]) <= {0, 1}
    assert (snapshot.edge_table["edge_timestamp"] <= pd.Timestamp("2025-01-02", tz="UTC")).all()


def test_time_window_snapshot_keeps_only_window_nodes_and_edges() -> None:
    snapshot = TemporalGraph(_graph()).time_window(
        as_of="2025-01-02T03:00:00Z", window="24h"
    )

    assert snapshot.node_table["transaction_id"].tolist() == ["t1", "t2"]
    assert snapshot.stats["nodes"] == 2
    assert all(
        snapshot.edge_table["source_timestamp"] >= pd.Timestamp("2025-01-02T03:00:00Z") - pd.Timedelta("24h")
    )


def test_custom_window_and_unavailable_history_are_explicit() -> None:
    graph = TransactionGraphBuilder().build(
        TransactionSchema.default().normalize(
            pd.DataFrame(
                [
                    {
                        "transaction_id": f"t{index}",
                        "timestamp": f"2025-01-{index + 1:02d}T00:00:00Z",
                        "sender_account_id": "account",
                        "amount": index + 1,
                    }
                    for index in range(9)
                ]
            )
        )
    )
    temporal = TemporalGraph(graph)

    with pytest.raises(ValueError, match="exceeds the graph's available edge history"):
        temporal.time_window(as_of="2025-01-09T00:00:00Z", window="8d")

    complete_graph = TransactionGraphBuilder().build(
        graph.node_table, full_history=True
    )
    snapshot = TemporalGraph(complete_graph).time_window(
        as_of="2025-01-09T00:00:00Z", window="7d"
    )
    assert snapshot.stats["nodes"] == 8


def test_local_neighborhood_uses_only_historical_incoming_edges() -> None:
    temporal = TemporalGraph(_graph())
    neighborhood = temporal.local_neighborhood(
        ["t2"], hops=2, as_of="2025-01-02T02:00:00Z"
    )

    assert set(neighborhood.node_table["transaction_id"]) == {"t0", "t1", "t2"}
    assert (neighborhood.edge_table["edge_timestamp"] <= pd.Timestamp("2025-01-02T02:00:00Z")).all()
    assert "t3" not in neighborhood.transaction_id_to_node


def test_local_neighborhood_validates_ids_and_cutoff() -> None:
    temporal = TemporalGraph(_graph())

    with pytest.raises(KeyError, match="Unknown transaction ID"):
        temporal.local_neighborhood(["unknown"])
    with pytest.raises(ValueError, match="cannot precede"):
        temporal.local_neighborhood(["t2"], as_of="2025-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="hops"):
        temporal.local_neighborhood(["t2"], hops=0)


def test_window_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        TemporalGraph(_graph()).time_window(as_of="2025-01-02", window=0)
