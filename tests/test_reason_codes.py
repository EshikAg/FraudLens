"""Tests for stable analyst-facing reason-code generation."""

from __future__ import annotations

from explainability.reason_codes import ReasonCodeGenerator


def test_reason_codes_cover_transaction_and_graph_signals() -> None:
    reasons = ReasonCodeGenerator(
        {
            "amount_anomaly_multiple": 3,
            "high_velocity_threshold": 5,
            "multi_account_device_threshold": 3,
            "top_reason_codes": 10,
        }
    ).generate(
        {
            "amount": 900,
            "historical_average_amount": 100,
            "new_beneficiary": True,
            "new_device": True,
            "device_account_count": 4,
            "transaction_velocity_1h": 7,
            "unusual_transaction_time": True,
            "account_degree": 2,
        },
        [
            {
                "feature": "amount",
                "contribution": 0.3,
                "observed_values": [900.0],
                "baseline_values": [100.0],
            },
            {
                "feature": "channel",
                "contribution": 0.1,
                "observed_values": [1.0],
                "baseline_values": [0.2],
            },
        ],
        {
            "historical_suspicious_neighbor_ids": ["prior-fraud"],
        },
    )

    by_code = {reason["code"]: reason for reason in reasons}
    assert {
        "AMOUNT_ANOMALY",
        "NEW_BENEFICIARY",
        "NEW_DEVICE",
        "SUSPICIOUS_DEVICE_NETWORK",
        "HIGH_VELOCITY",
        "UNUSUAL_TIME",
        "SUSPICIOUS_NEIGHBOURHOOD",
        "MODEL_FEATURE_IMPACT",
    }.issubset(by_code)
    assert by_code["AMOUNT_ANOMALY"]["supporting_value"]["multiple"] == 9
    assert by_code["MODEL_FEATURE_IMPACT"]["feature"] == "channel"
    assert all(
        {"code", "message", "severity", "supporting_value"}.issubset(reason)
        for reason in reasons
    )


def test_reason_generator_does_not_fabricate_unavailable_evidence() -> None:
    reasons = ReasonCodeGenerator().generate(
        {
            "amount": 50,
            "historical_average_amount": None,
            "new_beneficiary": None,
            "new_device": None,
            "transaction_velocity_1h": None,
            "unusual_transaction_time": None,
            "account_degree": None,
        },
        [],
        {"historical_suspicious_neighbor_ids": []},
    )

    assert reasons == []
