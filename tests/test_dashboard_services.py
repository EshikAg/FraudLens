"""Dashboard service tests; scoring remains delegated to the HTTP API."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from dashboard.services.api_client import FraudLensAPIClient
from dashboard.services.artifact_service import ArtifactService
from dashboard.services.demo_service import (
    build_demo_transactions,
    score_demo_transactions,
)


def test_api_client_never_sends_a_transaction_label(monkeypatch) -> None:
    response = Mock()
    response.ok = True
    response.json.return_value = {
        "transaction_id": "demo-1",
        "timestamp": "2026-10-22T12:00:00+00:00",
        "fraud_probability": 0.85,
        "risk_level": "CRITICAL",
        "decision": "BLOCK",
        "drift_detected": False,
        "model_version": "v1",
        "reasons": [],
        "related_transactions": [],
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("dashboard.services.api_client.requests.post", post)
    client = FraudLensAPIClient("http://localhost:8000")

    client.predict({"transaction_id": "demo-1", "label": 1, "amount": 40})

    payload = post.call_args.kwargs["json"]
    assert "label" not in payload
    assert payload["transaction_id"] == "demo-1"


def test_demo_rows_are_scored_only_through_injected_api_client() -> None:
    frame = build_demo_transactions(
        row_count=12,
        seed=5,
        start_time=datetime(2026, 10, 22, tzinfo=timezone.utc),
    )

    class FakeAPI:
        def __init__(self) -> None:
            self.payloads = []

        def predict(self, payload):
            self.payloads.append(dict(payload))
            return {
                "transaction_id": payload["transaction_id"],
                "timestamp": payload["timestamp"],
                "fraud_probability": 0.6,
                "risk_level": "HIGH",
                "decision": "REVIEW",
                "drift_detected": False,
                "model_version": "v1",
                "reasons": [],
                "related_transactions": [],
            }

    api = FakeAPI()
    scored = score_demo_transactions(frame, api)

    assert len(scored) == 12
    assert len(api.payloads) == 12
    assert all("label" not in payload for payload in api.payloads)
    assert "sender_account_id" not in scored.columns
    assert scored.iloc[-1]["model_version"] == "v1"
    assert pd.api.types.is_datetime64_any_dtype(scored["timestamp"])


def test_artifact_service_reads_registry_without_fabricating_missing_versions(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        '{"current_version":"v2","versions":[{"version":"v1"},{"version":"v2"}]}',
        encoding="utf-8",
    )
    service = ArtifactService({"model_registry": registry_path})

    versions, active_version = service.model_versions()
    assert [item["version"] for item in versions] == ["v1", "v2"]
    assert active_version == "v2"

    absent = ArtifactService({"model_registry": tmp_path / "missing.json"})
    assert absent.model_versions() == ([], None)
