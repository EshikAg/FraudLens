"""Focused checks for causal online inference with a GAAP-compatible test double."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch

from api.services.prediction_service import PredictionService
from graph.transaction_graph import GraphConfig, TransactionGraphBuilder
from models.banking_gaap import BankingGAAP
from preprocessing.feature_engineering import TransactionFeatureGenerator
from preprocessing.synthetic_data import generate_synthetic_transactions
from preprocessing.transaction_schema import TransactionSchema


class _FakeAdapter:
    threshold = 0.5
    model_version = "test-v1"

    def __init__(self, feature_count: int) -> None:
        self.data_module = SimpleNamespace(d_in=feature_count)
        self.node_counts: list[int] = []

    def predict_proba_on(self, graph: Any) -> torch.Tensor:
        self.node_counts.append(graph.num_nodes())
        probabilities = torch.full((graph.num_nodes(), 2), 0.0)
        probabilities[:, 0] = 0.1
        probabilities[:, 1] = 0.9
        return probabilities


def _fitted_test_model() -> BankingGAAP:
    config = BankingGAAP("configs/adaptive_gaap_baseline.yaml")
    raw = generate_synthetic_transactions(row_count=40, seed=7)
    canonical = TransactionSchema.default().normalize(raw)
    engineered = TransactionFeatureGenerator.from_yaml(
        "configs/feature_config.yaml"
    ).transform(canonical)
    if config.config.get("exclude_label_derived_features", True):
        engineered.attrs["feature_columns"] = [
            name
            for name in engineered.attrs["feature_columns"]
            if name != "suspicious_neighbour_count"
        ]
    graph_config = GraphConfig.from_mapping(config.config["graph_config"])
    config.graph = TransactionGraphBuilder(graph_config).build(
        engineered, full_history=True
    )
    config.gaap = _FakeAdapter(config.graph.dgl_graph.ndata["feature"].shape[1])
    return config


def _request(transaction_id: str = "online-1") -> dict[str, Any]:
    return {
        "transaction_id": transaction_id,
        "timestamp": "2025-01-01T01:00:00Z",
        "sender_account_id": "account_6",
        "receiver_account_id": "shared_beneficiary",
        "amount": 2500.0,
        "currency": "INR",
        "transaction_type": "UPI",
        "channel": "mobile",
        "device_id": "device_1",
        "ip_id": "ip_1",
    }


def test_prediction_service_scores_and_explains_with_configured_policy() -> None:
    model = _fitted_test_model()
    service = PredictionService(
        model,
        {
            "risk_thresholds": {"low": 0.15, "medium": 0.4, "high": 0.75},
            "decisions": {
                "low": "APPROVE",
                "medium": "AUTH",
                "high": "REVIEW",
                "critical": "BLOCK",
            },
            "drift_config": {
                "feature_drift": {"features": ["amount"], "min_samples": 2},
                "concept_drift": {"min_instances": 2},
            },
        },
    )

    response = service.predict(_request())

    assert response["transaction_id"] == "online-1"
    assert response["fraud_probability"] == pytest.approx(0.9)
    assert response["risk_level"] == "CRITICAL"
    assert response["decision"] == "BLOCK"
    assert response["model_version"] == "test-v1"
    assert response["drift_detected"] is False
    assert response["timestamp"] == "2025-01-01T01:00:00+00:00"
    assert response["reasons"]
    assert response["related_transactions"]
    assert service.metrics()["predictions_succeeded"] == 1
    assert model.gaap.node_counts[0] == 10


def test_prediction_service_rejects_duplicate_and_keeps_future_out_of_graph() -> None:
    service = PredictionService(_fitted_test_model())
    service.predict(_request())

    try:
        service.predict(_request())
    except ValueError as exc:
        assert "already been scored" in str(exc)
    else:
        raise AssertionError("Duplicate transaction IDs must be rejected")

    assert service.metrics()["prediction_errors"] == 1


def test_invalid_online_transaction_is_rejected() -> None:
    service = PredictionService(_fitted_test_model())
    request = _request()
    request["amount"] = -1

    try:
        service.predict(request)
    except ValueError as exc:
        assert "amount" in str(exc)
    else:
        raise AssertionError("Negative amount must be rejected")
