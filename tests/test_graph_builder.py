"""Tests for transaction nodes and causal relationship construction."""

from __future__ import annotations

import pandas as pd
import pytest

dgl = pytest.importorskip("dgl")

from graph.transaction_graph import (  # noqa: E402
    TransactionGraphBuilder,
    build_transaction_graph_from_csv,
    load_transaction_graph,
)
from graph.graph_features import to_networkx_sample
from preprocessing.synthetic_data import generate_synthetic_transactions
from preprocessing.transaction_schema import TransactionSchema


def _transactions() -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {
                "transaction_id": "t1",
                "timestamp": "2025-01-01T10:00:00Z",
                "sender_account_id": "a",
                "receiver_account_id": "b",
                "amount": 10.0,
                "transaction_type": "UPI",
                "device_id": "d",
                "ip_id": "ip",
                "label": 0,
            },
            {
                "transaction_id": "t2",
                "timestamp": "2025-01-01T10:30:00Z",
                "sender_account_id": "a",
                "receiver_account_id": "c",
                "amount": 20.0,
                "transaction_type": "IMPS",
                "device_id": "d",
                "ip_id": "ip",
                "label": 1,
            },
            {
                "transaction_id": "t3",
                "timestamp": "2025-01-02T10:00:00Z",
                "sender_account_id": "z",
                "receiver_account_id": "b",
                "amount": 30.0,
                "transaction_type": "CARD_PAYMENT",
                "device_id": "other",
                "ip_id": "other_ip",
                "label": 0,
            },
        ]
    )
    return TransactionSchema.default().normalize(raw)


def test_each_transaction_is_a_node_and_relationship_edges_are_typed() -> None:
    graph = TransactionGraphBuilder.from_yaml("configs/graph_config.yaml").build(
        _transactions()
    )

    assert graph.dgl_graph.num_nodes() == 3
    assert graph.dgl_graph.ndata["feature"].shape[0] == 3
    assert graph.dgl_graph.ndata["label"].tolist() == [0, 1, 0]
    assert graph.transaction_id_to_node == {"t1": 0, "t2": 1, "t3": 2}
    assert set(graph.edge_table["relationship_type"]) >= {
        "same_sender",
        "same_device",
        "same_ip",
        "same_receiver",
    }
    assert (graph.edge_table["source_timestamp"] < graph.edge_table["edge_timestamp"]).all()
    assert graph.stats["nodes"] == 3
    assert graph.stats["fraud_node_ratio"] == pytest.approx(1 / 3)


def test_future_transactions_cannot_add_edges_to_past_nodes() -> None:
    initial = _transactions().iloc[:2].reset_index(drop=True)
    future = TransactionSchema.default().normalize(
        pd.DataFrame(
            [
                {
                    "transaction_id": "future",
                    "timestamp": "2025-01-03T10:00:00Z",
                    "sender_account_id": "a",
                    "receiver_account_id": "new",
                    "amount": 999,
                    "device_id": "future-device",
                    "label": 1,
                }
            ]
        )
    )

    initial_graph = TransactionGraphBuilder().build(initial)
    complete_graph = TransactionGraphBuilder().build(
        pd.concat([initial, future], ignore_index=True)
    )

    past_edges = complete_graph.edge_table[
        complete_graph.edge_table["destination"] < 2
    ]
    pd.testing.assert_frame_equal(
        initial_graph.edge_table.reset_index(drop=True),
        past_edges.reset_index(drop=True),
        check_dtype=False,
    )
    assert (complete_graph.edge_table["source"] < complete_graph.edge_table["destination"]).all()


def test_same_timestamp_transactions_do_not_connect() -> None:
    transactions = _transactions().iloc[:2].copy()
    transactions.loc[1, "timestamp"] = transactions.loc[0, "timestamp"]

    graph = TransactionGraphBuilder().build(transactions)

    assert graph.dgl_graph.num_edges() == 0


def test_csv_to_dgl_graph_and_graph_round_trip(tmp_path) -> None:
    csv_path = tmp_path / "banking.csv"
    generate_synthetic_transactions(25, seed=13, output_path=csv_path)
    availability_path = tmp_path / "reports" / "available.json"
    output = tmp_path / "processed" / "graph"
    graph = build_transaction_graph_from_csv(
        csv_path,
        graph_config="configs/graph_config.yaml",
        availability_report_path=availability_path,
        output_dir=output,
    )
    restored = load_transaction_graph(output)

    assert graph.dgl_graph.num_nodes() == 25
    assert availability_path.is_file()
    assert restored.dgl_graph.num_nodes() == 25
    assert restored.transaction_id_to_node == graph.transaction_id_to_node
    assert restored.stats == graph.stats
    assert restored.dgl_graph.ndata["feature"].equal(
        graph.dgl_graph.ndata["feature"]
    )


def test_gaap_adapter_requires_disjoint_masks() -> None:
    graph = TransactionGraphBuilder().build(_transactions())
    adapted = graph.to_gaap_graph([True, False, False], [False, True, False], [False, False, True])

    assert all(
        name in adapted.ndata for name in ("feature", "label", "train_mask", "val_mask", "test_mask")
    )
    with pytest.raises(ValueError, match="disjoint"):
        graph.to_gaap_graph([True, False, False], [True, False, False], [False, False, True])


def test_configurable_rules_control_edge_types() -> None:
    graph = TransactionGraphBuilder(
        {
            "max_history_hours": 168,
            "edge_rules": [
                {"name": "sender_only", "field": "sender_account_id"},
            ],
        }
    ).build(_transactions())

    assert set(graph.edge_table["relationship_type"]) == {"sender_only"}
    assert graph.dgl_graph.edata["relationship_type"].tolist() == [0]


def test_visualization_sample_is_bounded_and_typed() -> None:
    networkx = pytest.importorskip("networkx")
    graph = TransactionGraphBuilder().build(_transactions())

    sample = to_networkx_sample(graph, max_nodes=2)

    assert isinstance(sample, networkx.MultiDiGraph)
    assert sample.number_of_nodes() == 2
    assert all("relationship_type" in values for _, _, values in sample.edges(data=True))
    with pytest.raises(ValueError, match="max_nodes"):
        to_networkx_sample(graph, max_nodes=0)
