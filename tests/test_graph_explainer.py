"""Tests for historical-only transaction graph explanations."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from explainability.graph_explainer import GraphExplainer


def _transaction_graph():
    timestamps = pd.to_datetime(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-01T01:00:00Z",
            "2025-01-01T02:00:00Z",
            "2025-01-01T03:00:00Z",
        ],
        utc=True,
    )
    nodes = pd.DataFrame(
        {
            "transaction_id": ["old-fraud", "middle", "target", "future"],
            "timestamp": timestamps,
            "label": [1, 0, 0, 1],
            "label_available_at": pd.to_datetime(
                [
                    "2025-01-01T00:30:00Z",
                    "2025-01-01T02:30:00Z",
                    "2025-01-01T02:00:00Z",
                    "2025-01-01T03:30:00Z",
                ],
                utc=True,
            ),
        }
    )
    edges = pd.DataFrame(
        [
            {
                "source": 0,
                "destination": 1,
                "relationship_type": "same_device",
                "edge_timestamp": timestamps[1],
                "time_delta_seconds": 3600,
            },
            {
                "source": 1,
                "destination": 2,
                "relationship_type": "same_receiver",
                "edge_timestamp": timestamps[2],
                "time_delta_seconds": 3600,
            },
            {
                "source": 2,
                "destination": 3,
                "relationship_type": "same_ip",
                "edge_timestamp": timestamps[3],
                "time_delta_seconds": 3600,
            },
        ]
    )
    return SimpleNamespace(node_table=nodes, edge_table=edges)


def test_graph_explanation_returns_ancestors_and_causal_paths_only() -> None:
    graph = _transaction_graph()
    explainer = GraphExplainer({"max_connection_hops": 2})

    result = explainer.explain(graph, 2)

    related = {item["transaction_id"]: item for item in result["related_transactions"]}
    assert set(related) == {"old-fraud", "middle"}
    assert related["old-fraud"]["fraud_status"] == "fraud"
    assert related["middle"]["fraud_status"] is None
    assert related["old-fraud"]["connection_path"] == [
        {
            "source_transaction_id": "old-fraud",
            "destination_transaction_id": "middle",
            "relationship": "same_device",
            "timestamp": "2025-01-01T01:00:00+00:00",
        },
        {
            "source_transaction_id": "middle",
            "destination_transaction_id": "target",
            "relationship": "same_receiver",
            "timestamp": "2025-01-01T02:00:00+00:00",
        },
    ]
    assert "future" not in related
    assert result["historical_suspicious_neighbor_ids"] == ["old-fraud"]


def test_graph_explainer_suppresses_labels_without_availability_time() -> None:
    graph = _transaction_graph()
    graph.node_table = graph.node_table.drop(columns=["label_available_at"])

    result = GraphExplainer().explain(graph, 2)

    assert all(
        related["fraud_status"] is None
        for related in result["related_transactions"]
    )
    assert result["historical_suspicious_neighbor_ids"] == []


def test_graph_explainer_honors_cutoff_and_limits() -> None:
    graph = _transaction_graph()

    result = GraphExplainer(
        {"max_related_transactions": 1, "max_connection_hops": 2}
    ).explain(graph, 2, cutoff="2025-01-01T02:00:00Z")

    assert len(result["related_transactions"]) == 1
    assert result["cutoff_timestamp"] == "2025-01-01T02:00:00+00:00"
    try:
        GraphExplainer().explain(graph, 3, cutoff="2025-01-01T02:00:00Z")
    except ValueError as error:
        assert "cannot precede" in str(error)
    else:
        raise AssertionError("A cutoff before the transaction must be rejected")
