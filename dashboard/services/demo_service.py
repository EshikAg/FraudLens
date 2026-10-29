"""Create display-only demo rows and score them through the online API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pandas as pd

from dashboard.services.api_client import FraudLensAPIClient
from preprocessing.synthetic_data import generate_synthetic_transactions


def build_demo_transactions(
    *,
    row_count: int,
    seed: int,
    start_time: datetime | None = None,
) -> pd.DataFrame:
    """Generate generic sample banking rows with a later changed fraud pattern."""
    if row_count < 1:
        raise ValueError("row_count must be positive")
    frame = generate_synthetic_transactions(row_count=row_count, seed=seed)
    start = start_time or (
        datetime.now(timezone.utc) - timedelta(minutes=max(1, row_count * 4))
    )
    if start.tzinfo is None:
        raise ValueError("start_time must be timezone-aware")
    frame["timestamp"] = [
        (start + timedelta(minutes=index * 4)).isoformat()
        for index in range(row_count)
    ]
    drift_start = row_count // 2
    suspicious_after_drift = frame.index[
        (frame.index >= drift_start) & (frame["label"].astype(int) == 1)
    ]
    for index in suspicious_after_drift:
        frame.loc[index, "amount"] = float(35 + (index % 5) * 7)
        frame.loc[index, "receiver_account_id"] = f"demo_new_beneficiary_{index % 2}"
        frame.loc[index, "device_id"] = f"demo_new_device_{index % 2}"
        frame.loc[index, "ip_id"] = "demo_new_network"
        frame.loc[index, "transaction_type"] = "UPI"
    prefix = uuid4().hex[:10]
    frame["transaction_id"] = [
        f"demo_{prefix}_{index:04d}" for index in range(row_count)
    ]
    return frame


def score_demo_transactions(
    frame: pd.DataFrame, api: FraudLensAPIClient
) -> pd.DataFrame:
    """Score generated rows via HTTP, keeping labels out of the scoring request."""
    if frame.empty:
        raise ValueError("No demo transactions were provided")
    records: list[dict[str, Any]] = []
    for transaction in frame.to_dict(orient="records"):
        prediction_payload = {
            key: value for key, value in transaction.items() if key != "label"
        }
        response = api.predict(prediction_payload)
        records.append(
            {
                "transaction_id": str(response["transaction_id"]),
                "timestamp": pd.to_datetime(response["timestamp"], utc=True),
                "amount": float(transaction["amount"]),
                "currency": transaction.get("currency"),
                "transaction_type": transaction.get("transaction_type"),
                "channel": transaction.get("channel"),
                "fraud_probability": float(response["fraud_probability"]),
                "risk_level": str(response["risk_level"]),
                "decision": str(response["decision"]),
                "drift_detected": bool(response["drift_detected"]),
                "model_version": str(response["model_version"]),
                "reasons": response.get("reasons", []),
                "related_transactions": response.get("related_transactions", []),
                "transaction_details": {
                    key: transaction.get(key)
                    for key in (
                        "amount",
                        "currency",
                        "transaction_type",
                        "channel",
                        "merchant_id",
                        "timestamp",
                        "location",
                    )
                },
            }
        )
    return pd.DataFrame.from_records(records)
