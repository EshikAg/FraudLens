"""HTTP-level validation and endpoint tests for the scoring API."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from api.services.prediction_service import PredictionService
from models.banking_gaap import BankingGAAP
from tests.test_prediction_service import _fitted_test_model, _request


def _service() -> PredictionService:
    return PredictionService(
        _fitted_test_model(),
        {
            "risk_thresholds": {"low": 0.15, "medium": 0.4, "high": 0.75},
            "drift_config": {
                "feature_drift": {"features": ["amount"], "min_samples": 2},
                "concept_drift": {"min_instances": 2},
            },
        },
    )


def test_health_predict_model_drift_and_metrics_endpoints() -> None:
    app = create_app(prediction_service=_service())
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        assert health.json()["model_version"] == "test-v1"

        prediction = client.post("/predict", json=_request())
        assert prediction.status_code == 200
        result = prediction.json()
        assert result["transaction_id"] == "online-1"
        assert result["model_version"] == "test-v1"
        assert result["risk_level"] == "CRITICAL"
        assert result["decision"] == "BLOCK"
        assert result["reasons"]
        assert result["related_transactions"]
        assert "drift_detected" in result
        label_update = client.post(
            "/labels/online-1",
            json={"label": 1, "timestamp": "2025-01-01T02:00:00Z"},
        )
        assert label_update.status_code == 200

        assert client.get("/model").json()["model_loaded"] is True
        assert "drift_detected" in client.get("/drift").json()
        metrics = client.get("/metrics").json()
        assert metrics["predictions_succeeded"] == 1


def test_invalid_request_is_rejected_with_useful_validation_error() -> None:
    app = create_app(prediction_service=_service())
    invalid = _request()
    invalid["amount"] = 0
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid)
    assert response.status_code == 422
    assert any("amount" in str(error["loc"]) for error in response.json()["detail"])


def test_prediction_endpoint_rejects_ground_truth_label() -> None:
    payload = _request()
    payload["label"] = 1
    with TestClient(create_app(prediction_service=_service())) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "label must not be supplied" in response.json()["detail"]


def test_api_starts_without_model_and_reports_degraded_health() -> None:
    with TestClient(create_app(config={"model_directory": None})) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/predict", json=_request()).status_code == 503


def test_saved_gaap_model_scores_synthetic_transaction_through_api(
    tmp_path: Path,
) -> None:
    model = BankingGAAP("configs/adaptive_gaap_baseline.yaml")
    model.config["data"]["synthetic_rows"] = 66
    model.config["data"]["synthetic_csv"] = str(
        tmp_path / "raw" / "synthetic_transactions.csv"
    )
    model.config["output_dir"] = str(tmp_path / "training")
    model.config["training"]["max_epochs"] = 1
    model.config["training"]["patience"] = 1
    model.fit()
    saved_model = model.save(tmp_path / "saved_model")

    settings = {
        "model_directory": str(saved_model),
        "drift_config": {
            "feature_drift": {"features": ["amount"], "min_samples": 20},
            "concept_drift": {"min_instances": 20},
        },
    }
    service = PredictionService.from_config(settings)
    with TestClient(create_app(settings, prediction_service=service)) as client:
        response = client.post("/predict", json=_request("real-gaap-online"))

    assert response.status_code == 200
    result = response.json()
    assert result["transaction_id"] == "real-gaap-online"
    assert 0 <= result["fraud_probability"] <= 1
    assert result["model_version"] == "gaap-banking-v1"
    assert "drift_detected" in result
