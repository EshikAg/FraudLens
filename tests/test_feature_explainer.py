"""Tests for reproducible, historical-baseline feature ablation."""

from __future__ import annotations

from types import SimpleNamespace

import dgl
import numpy as np
import pandas as pd
import torch

from explainability.explanation_service import ExplanationService
from explainability.feature_explainer import FeatureExplainer


def _graph():
    graph = dgl.graph(([0, 1], [1, 2]), num_nodes=3)
    graph.ndata["feature"] = torch.tensor(
        [[0.1, 0.0], [0.8, 1.0], [0.99, 0.0]], dtype=torch.float32
    )
    graph.ndata["timestamp"] = torch.tensor(
        pd.to_datetime(
            [
                "2025-01-01T00:00:00Z",
                "2025-01-01T01:00:00Z",
                "2025-01-01T02:00:00Z",
            ],
            utc=True,
        ).asi8,
        dtype=torch.int64,
    )
    return graph


def _predictor(graph):
    amount = graph.ndata["feature"][:, 0].clamp(0.0, 1.0)
    return torch.stack((1.0 - amount, amount), dim=1)


def test_local_ablation_uses_only_prior_nodes_and_is_reproducible() -> None:
    graph = _graph()
    explainer = FeatureExplainer({"top_k_features": 5})
    arguments = (
        graph,
        1,
        ["amount", "new_device"],
        _predictor,
    )
    first = explainer.explain(
        *arguments,
        cutoff="2025-01-01T01:00:00Z",
    )
    second = explainer.explain(
        *arguments,
        cutoff="2025-01-01T01:00:00Z",
    )

    assert first == second
    assert first["historical_reference_count"] == 1
    amount = next(item for item in first["features"] if item["feature"] == "amount")
    assert amount["baseline_values"] == [0.10000000149011612]
    assert amount["contribution"] > 0.6


def test_explainer_rejects_bad_feature_width_and_invalid_predictor_output() -> None:
    explainer = FeatureExplainer()
    graph = _graph()
    with np.testing.assert_raises_regex(ValueError, "Feature names"):
        explainer.explain(
            graph,
            1,
            ["amount"],
            _predictor,
            cutoff="2025-01-01T01:00:00Z",
        )
    with np.testing.assert_raises_regex(ValueError, "probabilities"):
        explainer.explain(
            graph,
            1,
            ["amount", "new_device"],
            lambda value: np.full(value.num_nodes(), 2.0),
            cutoff="2025-01-01T01:00:00Z",
        )


def test_explanation_service_returns_required_analyst_fields() -> None:
    graph = _graph()
    node_table = pd.DataFrame(
        [
            {
                "transaction_id": f"tx-{index}",
                "timestamp": timestamp,
                "amount": amount,
                "historical_average_amount": 0.1,
                "amount_deviation_from_history": abs(amount - 0.1),
                "new_beneficiary": index == 1,
                "new_device": index == 1,
                "transaction_velocity_1h": 7 if index == 1 else 0,
                "account_degree": 0,
                "label": None,
            }
            for index, (timestamp, amount) in enumerate(
                zip(
                    pd.to_datetime(
                        [
                            "2025-01-01T00:00:00Z",
                            "2025-01-01T01:00:00Z",
                            "2025-01-01T02:00:00Z",
                        ],
                        utc=True,
                    ),
                    (0.1, 0.8, 0.99),
                )
            )
        ]
    )
    transaction_graph = SimpleNamespace(
        dgl_graph=graph,
        node_table=node_table,
        edge_table=pd.DataFrame(
            columns=[
                "source",
                "destination",
                "relationship_type",
                "edge_timestamp",
                "time_delta_seconds",
            ]
        ),
        transaction_id_to_node={f"tx-{index}": index for index in range(3)},
        feature_columns=("amount", "new_device"),
    )
    adapter = SimpleNamespace(
        model_version="v3",
        threshold=0.5,
        predict_proba_on=_predictor,
    )
    model = SimpleNamespace(
        _require_fitted=lambda: (transaction_graph, adapter),
    )
    service = ExplanationService(
        {
            "feature_attribution": {"top_k_features": 3},
            "reason_codes": {"amount_anomaly_multiple": 3},
            "risk_thresholds": {"low": 0.2, "medium": 0.5, "high": 0.8},
            "decision_policy": {
                "low": "approve",
                "medium": "review",
                "high": "review",
                "critical": "block",
            },
        }
    )

    result = service.explain(model, "tx-1")

    assert {
        "transaction_id",
        "fraud_probability",
        "risk_level",
        "decision",
        "top_feature_reasons",
        "top_related_transactions",
        "graph_reasons",
        "model_version",
        "explanation_confidence",
    }.issubset(result)
    assert result["risk_level"] == "critical"
    assert result["decision"] == "block"
    assert result["model_version"] == "v3"
    assert result["top_feature_reasons"]
    assert all("code" in reason and "message" in reason for reason in result["top_feature_reasons"])
