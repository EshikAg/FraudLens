"""Stable human-readable risk reason codes derived from observed evidence."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


class ReasonCodeGenerator:
    """Translate engineered transaction signals and graph evidence into codes."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        settings = dict(config or {})
        self.amount_multiple = float(settings.get("amount_anomaly_multiple", 3.0))
        self.high_velocity_threshold = int(
            settings.get("high_velocity_threshold", 5)
        )
        self.device_account_threshold = int(
            settings.get("multi_account_device_threshold", 3)
        )
        self.high_degree_threshold = int(settings.get("high_graph_degree_threshold", 10))
        self.top_k = int(settings.get("top_reason_codes", 5))
        if self.amount_multiple <= 1:
            raise ValueError("amount_anomaly_multiple must exceed one")
        if min(
            self.high_velocity_threshold,
            self.device_account_threshold,
            self.high_degree_threshold,
            self.top_k,
        ) < 1:
            raise ValueError("Reason thresholds and top_reason_codes must be positive")

    def generate(
        self,
        transaction: Mapping[str, Any] | pd.Series,
        feature_attributions: list[Mapping[str, Any]],
        graph_explanation: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate deterministic, prioritized reasons with supporting values."""
        row = dict(transaction)
        attribution_by_feature = {
            str(item["feature"]): float(item["contribution"])
            for item in feature_attributions
        }
        reasons: list[dict[str, Any]] = []

        amount = self._number(row.get("amount"))
        historical_average = self._number(row.get("historical_average_amount"))
        if (
            amount is not None
            and historical_average is not None
            and historical_average > 0
            and amount >= historical_average * self.amount_multiple
        ):
            reasons.append(
                self._reason(
                    "AMOUNT_ANOMALY",
                    "Transaction amount is significantly higher than the account's recent average.",
                    "high",
                    {
                        "amount": amount,
                        "historical_average_amount": historical_average,
                        "multiple": amount / historical_average,
                    },
                    attribution_by_feature,
                    ("amount", "log_amount", "amount_deviation_from_history"),
                )
            )
        if self._truthy(row.get("new_beneficiary")):
            reasons.append(
                self._reason(
                    "NEW_BENEFICIARY",
                    "Beneficiary was not observed in the sender's prior transaction history.",
                    "medium",
                    {"new_beneficiary": True},
                    attribution_by_feature,
                    ("new_beneficiary",),
                )
            )
        if self._truthy(row.get("new_device")):
            reasons.append(
                self._reason(
                    "NEW_DEVICE",
                    "This device has not been observed in prior transactions.",
                    "medium",
                    {"new_device": True},
                    attribution_by_feature,
                    ("new_device",),
                )
            )
        device_accounts = self._number(row.get("device_account_count"))
        if (
            device_accounts is not None
            and device_accounts >= self.device_account_threshold
        ):
            reasons.append(
                self._reason(
                    "SUSPICIOUS_DEVICE_NETWORK",
                    "Device is associated with multiple accounts.",
                    "high",
                    {
                        "device_account_count": device_accounts,
                        "threshold": self.device_account_threshold,
                    },
                    attribution_by_feature,
                    ("device_account_count", "device_degree"),
                )
            )
        velocity = self._number(
            row.get("transaction_velocity_1h", row.get("tx_count_prev_1h"))
        )
        if velocity is not None and velocity >= self.high_velocity_threshold:
            reasons.append(
                self._reason(
                    "HIGH_VELOCITY",
                    "Transaction velocity is unusually high for the recent one-hour window.",
                    "high",
                    {
                        "transactions_in_previous_hour": velocity,
                        "threshold": self.high_velocity_threshold,
                    },
                    attribution_by_feature,
                    ("transaction_velocity_1h", "tx_count_prev_1h"),
                )
            )
        if self._truthy(row.get("unusual_transaction_time")):
            reasons.append(
                self._reason(
                    "UNUSUAL_TIME",
                    "Transaction time differs from the account's previously observed activity pattern.",
                    "medium",
                    {"unusual_transaction_time": True},
                    attribution_by_feature,
                    ("unusual_transaction_time", "hour"),
                )
            )
        suspicious_neighbors = list(
            graph_explanation.get("historical_suspicious_neighbor_ids", [])
        )
        if suspicious_neighbors:
            reasons.append(
                self._reason(
                    "SUSPICIOUS_NEIGHBOURHOOD",
                    "Transaction is connected to earlier transactions whose fraud labels were available by this time.",
                    "high",
                    {
                        "historical_suspicious_transaction_ids": suspicious_neighbors
                    },
                    attribution_by_feature,
                    ("suspicious_neighbour_count", "account_degree"),
                )
            )
        degree = self._number(row.get("account_degree"))
        if degree is not None and degree >= self.high_degree_threshold:
            reasons.append(
                self._reason(
                    "GRAPH_ASSOCIATION_ANOMALY",
                    "Transaction is associated with an unusually connected account history.",
                    "medium",
                    {"account_degree": degree, "threshold": self.high_degree_threshold},
                    attribution_by_feature,
                    ("account_degree", "beneficiary_degree"),
                )
            )
        reasoned_features = {
            feature
            for reason in reasons
            for feature in self._features_for_code(reason["code"])
        }
        for item in feature_attributions:
            feature = str(item["feature"])
            contribution = float(item["contribution"])
            if feature in reasoned_features or contribution <= 0:
                continue
            display_feature = feature.replace("_", " ")
            reasons.append(
                {
                    "code": "MODEL_FEATURE_IMPACT",
                    "message": (
                        f"The model score was sensitive to the {display_feature} "
                        "feature in a local historical-baseline comparison."
                    ),
                    "severity": "low",
                    "supporting_value": {
                        "observed_values": item["observed_values"],
                        "historical_baseline_values": item["baseline_values"],
                    },
                    "feature_contribution": contribution,
                    "feature": feature,
                }
            )
        reasons.sort(
            key=lambda item: (
                -self._severity_rank(item["severity"]),
                -float(item.get("feature_contribution") or 0.0),
                item["code"],
            )
        )
        return reasons[: self.top_k]

    @staticmethod
    def _features_for_code(code: str) -> set[str]:
        return {
            "AMOUNT_ANOMALY": {
                "amount",
                "log_amount",
                "amount_deviation_from_history",
            },
            "NEW_BENEFICIARY": {"new_beneficiary"},
            "NEW_DEVICE": {"new_device"},
            "HIGH_VELOCITY": {
                "transaction_velocity_1h",
                "tx_count_prev_1h",
            },
            "UNUSUAL_TIME": {"unusual_transaction_time", "hour"},
            "SUSPICIOUS_DEVICE_NETWORK": {
                "device_account_count",
                "device_degree",
            },
            "SUSPICIOUS_NEIGHBOURHOOD": {
                "suspicious_neighbour_count",
                "account_degree",
            },
            "GRAPH_ASSOCIATION_ANOMALY": {
                "account_degree",
                "beneficiary_degree",
            },
        }.get(code, set())

    @staticmethod
    def _reason(
        code: str,
        message: str,
        severity: str,
        supporting_value: Any,
        attributions: Mapping[str, float],
        feature_names: tuple[str, ...],
    ) -> dict[str, Any]:
        contribution = max(
            (attributions.get(name, 0.0) for name in feature_names), default=0.0
        )
        return {
            "code": code,
            "message": message,
            "severity": severity,
            "supporting_value": supporting_value,
            "feature_contribution": float(contribution),
        }

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if pd.notna(number) else None

    @staticmethod
    def _truthy(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    @staticmethod
    def _severity_rank(severity: str) -> int:
        return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(
            severity, 0
        )
