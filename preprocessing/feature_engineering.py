"""Deterministic, strictly historical transaction feature generation."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from preprocessing.validation import require_temporal_order


FEATURE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "amount": ("amount",),
    "log_amount": ("amount",),
    "transaction_type": ("transaction_type",),
    "channel": ("channel",),
    "hour": ("timestamp",),
    "day_of_week": ("timestamp",),
    "tx_count_prev_10m": ("sender_account_id", "timestamp"),
    "tx_count_prev_1h": ("sender_account_id", "timestamp"),
    "tx_count_prev_24h": ("sender_account_id", "timestamp"),
    "amount_sum_prev_1h": ("sender_account_id", "amount", "timestamp"),
    "amount_sum_prev_24h": ("sender_account_id", "amount", "timestamp"),
    "historical_average_amount": ("sender_account_id", "amount", "timestamp"),
    "historical_maximum_amount": ("sender_account_id", "amount", "timestamp"),
    "account_age_days": ("sender_account_id", "timestamp"),
    "beneficiary_age_days": ("receiver_account_id", "timestamp"),
    "new_beneficiary": ("sender_account_id", "receiver_account_id", "timestamp"),
    "new_device": ("device_id", "timestamp"),
    "device_account_count": ("device_id", "sender_account_id", "timestamp"),
    "device_transaction_count": ("device_id", "timestamp"),
    "new_ip": ("ip_id", "timestamp"),
    "ip_account_count": ("ip_id", "sender_account_id", "timestamp"),
    "ip_transaction_count": ("ip_id", "timestamp"),
    "amount_deviation_from_history": ("sender_account_id", "amount", "timestamp"),
    "transaction_velocity_1h": ("sender_account_id", "timestamp"),
    "unusual_transaction_time": ("sender_account_id", "timestamp"),
    "distinct_beneficiaries_prev_1h": (
        "sender_account_id",
        "receiver_account_id",
        "timestamp",
    ),
    "rapid_beneficiary_changes": (
        "sender_account_id",
        "receiver_account_id",
        "timestamp",
    ),
    "account_degree": ("sender_account_id", "receiver_account_id", "timestamp"),
    "device_degree": ("device_id", "sender_account_id", "timestamp"),
    "ip_degree": ("ip_id", "sender_account_id", "timestamp"),
    "beneficiary_degree": ("receiver_account_id", "sender_account_id", "timestamp"),
    "suspicious_neighbour_count": (
        "sender_account_id",
        "receiver_account_id",
        "label",
        "timestamp",
    ),
}

_DEFAULT_SUSPICIOUS_LABELS = (1, "1", True, "true", "fraud", "suspicious")


@dataclass(frozen=True)
class FeatureConfig:
    """Window and threshold settings for historical features."""

    suspicious_labels: tuple[Any, ...] = _DEFAULT_SUSPICIOUS_LABELS
    unusual_time_threshold_hours: float = 4.0
    rapid_beneficiary_threshold: int = 3
    windows_minutes: tuple[int, int, int] = (10, 60, 1440)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> FeatureConfig:
        suspicious = config.get("suspicious_labels", _DEFAULT_SUSPICIOUS_LABELS)
        windows = config.get("windows_minutes", (10, 60, 1440))
        if not isinstance(suspicious, (list, tuple)):
            raise ValueError("'suspicious_labels' must be a list")
        if not isinstance(windows, (list, tuple)) or len(windows) != 3:
            raise ValueError("'windows_minutes' must contain 10m, 1h, and 24h windows")
        normalized_windows = tuple(int(window) for window in windows)
        if any(window <= 0 for window in normalized_windows):
            raise ValueError("Feature windows must be positive")
        if tuple(sorted(normalized_windows)) != normalized_windows:
            raise ValueError("Feature windows must be in increasing order")
        unusual_threshold = float(config.get("unusual_time_threshold_hours", 4.0))
        beneficiary_threshold = int(config.get("rapid_beneficiary_threshold", 3))
        if not 0 <= unusual_threshold <= 12:
            raise ValueError("unusual_time_threshold_hours must be between 0 and 12")
        if beneficiary_threshold < 1:
            raise ValueError("rapid_beneficiary_threshold must be positive")
        return cls(
            suspicious_labels=tuple(suspicious),
            unusual_time_threshold_hours=unusual_threshold,
            rapid_beneficiary_threshold=beneficiary_threshold,
            windows_minutes=normalized_windows,  # type: ignore[arg-type]
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> FeatureConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError("Feature config must be a YAML mapping")
        return cls.from_mapping(config)


def _identifier(value: Any) -> str | None:
    if pd.isna(value):
        return None
    result = str(value)
    return result if result else None


def _number(value: Any) -> float | None:
    if pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _label_is_suspicious(value: Any, suspicious_labels: tuple[Any, ...]) -> bool:
    if pd.isna(value):
        return False

    def normalize(candidate: Any) -> str:
        if isinstance(candidate, bool):
            return str(candidate).lower()
        try:
            numeric = float(candidate)
        except (TypeError, ValueError):
            return str(candidate).strip().lower()
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
        return str(candidate).strip().lower()

    normalized = normalize(value)
    return normalized in {normalize(candidate) for candidate in suspicious_labels}


def _empty_availability(
    transactions: pd.DataFrame, feature_names: list[str]
) -> dict[str, Any]:
    fields = {
        name: name in transactions.columns
        and bool(transactions[name].notna().any())
        for name in transactions.columns
    }
    report: dict[str, Any] = {}
    for name in feature_names:
        dependencies = FEATURE_DEPENDENCIES[name]
        missing = [field for field in dependencies if not fields.get(field, False)]
        report[name] = {
            "available": not missing,
            "missing_dependencies": missing,
            "source_fields": list(dependencies),
        }
    return report


class TransactionFeatureGenerator:
    """Generate features using only transactions strictly earlier in time.

    Equal-timestamp rows are evaluated as a batch before any row in that batch
    updates history, since their relative availability cannot be established.
    """

    def __init__(self, config: FeatureConfig | Mapping[str, Any] | None = None):
        if config is None:
            self.config = FeatureConfig()
        elif isinstance(config, FeatureConfig):
            self.config = config
        else:
            self.config = FeatureConfig.from_mapping(config)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TransactionFeatureGenerator:
        return cls(FeatureConfig.from_yaml(path))

    def transform(
        self,
        transactions: pd.DataFrame,
        availability_report_path: str | Path | None = None,
    ) -> pd.DataFrame:
        require_temporal_order(transactions)
        source = transactions.reset_index(drop=True).copy()
        source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True)
        feature_names = list(FEATURE_DEPENDENCIES)
        if source.empty:
            result = source.copy()
            for name in feature_names:
                result[name] = pd.Series(index=result.index, dtype="object")
            result.attrs["feature_availability"] = {
                "row_count": 0,
                "fields": source.attrs.get("feature_availability", {}).get(
                    "fields", {}
                ),
                "features": _empty_availability(source, feature_names),
            }
            result.attrs["feature_columns"] = feature_names
            result.attrs["non_feature_columns"] = [
                name
                for name in (
                    "transaction_id",
                    "timestamp",
                    "sender_account_id",
                    "receiver_account_id",
                    "currency",
                    "merchant_id",
                    "device_id",
                    "ip_id",
                    "location",
                    "label",
                    "sender_balance_before",
                    "sender_balance_after",
                    "receiver_balance_before",
                    "receiver_balance_after",
                )
                if name in result.columns
            ]
            if availability_report_path is not None:
                self.write_availability_report(result, availability_report_path)
            return result

        ordered = source.sort_values("timestamp", kind="stable")
        records: dict[int, dict[str, Any]] = {}
        activity: dict[str, deque[tuple[pd.Timestamp, float | None, str | None]]] = (
            defaultdict(deque)
        )
        first_seen: dict[str, pd.Timestamp] = {}
        sender_beneficiaries: dict[str, set[str]] = defaultdict(set)
        account_neighbours: dict[str, set[str]] = defaultdict(set)
        beneficiary_senders: dict[str, set[str]] = defaultdict(set)
        device_accounts: dict[str, set[str]] = defaultdict(set)
        ip_accounts: dict[str, set[str]] = defaultdict(set)
        device_counts: dict[str, int] = defaultdict(int)
        ip_counts: dict[str, int] = defaultdict(int)
        historical_amount_sum: dict[str, float] = defaultdict(float)
        historical_amount_count: dict[str, int] = defaultdict(int)
        historical_amount_max: dict[str, float] = {}
        hour_vectors: dict[str, tuple[float, float, int]] = {}
        suspicious_neighbours: dict[str, set[str]] = defaultdict(set)
        historical_labels_seen = False
        ten_minutes, one_hour, one_day = (
            timedelta(minutes=value) for value in self.config.windows_minutes
        )

        for timestamp, batch in ordered.groupby("timestamp", sort=True):
            now = pd.Timestamp(timestamp)
            for original_index, row in batch.iterrows():
                sender = _identifier(row.get("sender_account_id", pd.NA))
                receiver = _identifier(row.get("receiver_account_id", pd.NA))
                device = _identifier(row.get("device_id", pd.NA))
                ip = _identifier(row.get("ip_id", pd.NA))
                amount = _number(row.get("amount", pd.NA))
                current_hour = now.hour + now.minute / 60 + now.second / 3600
                record: dict[str, Any] = {
                    "amount": amount,
                    "log_amount": (
                        math.log1p(amount) if amount is not None and amount >= 0 else np.nan
                    ),
                    "transaction_type": row.get("transaction_type", pd.NA),
                    "channel": row.get("channel", pd.NA),
                    "hour": current_hour,
                    "day_of_week": now.dayofweek,
                }

                prior_events: list[tuple[pd.Timestamp, float | None, str | None]] = []
                if sender is not None:
                    events = activity[sender]
                    while events and events[0][0] < now - one_day:
                        events.popleft()
                    prior_events = list(events)
                    record.update(
                        {
                            "tx_count_prev_10m": sum(
                                time >= now - ten_minutes
                                for time, _, _ in prior_events
                            ),
                            "tx_count_prev_1h": sum(
                                time >= now - one_hour
                                for time, _, _ in prior_events
                            ),
                            "tx_count_prev_24h": len(prior_events),
                            "amount_sum_prev_1h": sum(
                                value
                                for time, value, _ in prior_events
                                if time >= now - one_hour and value is not None
                            ),
                            "amount_sum_prev_24h": sum(
                                value for _, value, _ in prior_events if value is not None
                            ),
                            "historical_average_amount": (
                                historical_amount_sum[sender]
                                / historical_amount_count[sender]
                                if historical_amount_count[sender]
                                else np.nan
                            ),
                            "historical_maximum_amount": historical_amount_max.get(
                                sender, np.nan
                            ),
                            "account_age_days": (
                                (now - first_seen[sender]).total_seconds() / 86400
                                if sender in first_seen
                                else 0.0
                            ),
                            "new_beneficiary": (
                                receiver not in sender_beneficiaries[sender]
                                if receiver is not None
                                else np.nan
                            ),
                            "amount_deviation_from_history": np.nan,
                            "transaction_velocity_1h": sum(
                                time >= now - one_hour for time, _, _ in prior_events
                            ),
                            "unusual_transaction_time": np.nan,
                            "distinct_beneficiaries_prev_1h": len(
                                {
                                    beneficiary
                                    for time, _, beneficiary in prior_events
                                    if time >= now - one_hour and beneficiary is not None
                                }
                            ),
                            "rapid_beneficiary_changes": False,
                            "account_degree": len(account_neighbours[sender]),
                            "suspicious_neighbour_count": (
                                len(suspicious_neighbours[sender])
                                if historical_labels_seen
                                else np.nan
                            ),
                        }
                    )
                    if amount is not None and historical_amount_count[sender]:
                        record["amount_deviation_from_history"] = abs(
                            amount
                            - historical_amount_sum[sender]
                            / historical_amount_count[sender]
                        )
                    if receiver is not None:
                        record["rapid_beneficiary_changes"] = (
                            receiver not in sender_beneficiaries[sender]
                            and record["distinct_beneficiaries_prev_1h"]
                            >= self.config.rapid_beneficiary_threshold
                        )
                    vector = hour_vectors.get(sender)
                    if vector is not None and vector[2]:
                        mean_hour = (
                            math.atan2(vector[1], vector[0]) * 12 / math.pi
                        ) % 24
                        difference = abs(current_hour - mean_hour)
                        difference = min(difference, 24 - difference)
                        record["unusual_transaction_time"] = (
                            difference > self.config.unusual_time_threshold_hours
                        )
                else:
                    for name in (
                        "tx_count_prev_10m",
                        "tx_count_prev_1h",
                        "tx_count_prev_24h",
                        "amount_sum_prev_1h",
                        "amount_sum_prev_24h",
                        "historical_average_amount",
                        "historical_maximum_amount",
                        "account_age_days",
                        "new_beneficiary",
                        "amount_deviation_from_history",
                        "transaction_velocity_1h",
                        "unusual_transaction_time",
                        "distinct_beneficiaries_prev_1h",
                        "rapid_beneficiary_changes",
                        "account_degree",
                        "suspicious_neighbour_count",
                    ):
                        record[name] = np.nan

                record["beneficiary_age_days"] = (
                    (now - first_seen[receiver]).total_seconds() / 86400
                    if receiver is not None and receiver in first_seen
                    else (0.0 if receiver is not None else np.nan)
                )
                record["new_device"] = (
                    device not in device_counts if device is not None else np.nan
                )
                record["device_account_count"] = (
                    len(device_accounts[device])
                    if device is not None and sender is not None
                    else (np.nan if device is None or sender is None else 0)
                )
                record["device_transaction_count"] = (
                    device_counts.get(device, 0) if device is not None else np.nan
                )
                record["device_degree"] = record["device_account_count"]
                record["new_ip"] = ip not in ip_counts if ip is not None else np.nan
                record["ip_account_count"] = (
                    len(ip_accounts[ip])
                    if ip is not None and sender is not None
                    else (np.nan if ip is None or sender is None else 0)
                )
                record["ip_transaction_count"] = (
                    ip_counts.get(ip, 0) if ip is not None else np.nan
                )
                record["ip_degree"] = record["ip_account_count"]
                record["beneficiary_degree"] = (
                    len(beneficiary_senders[receiver])
                    if receiver is not None
                    else np.nan
                )
                records[original_index] = record

            # Update all historical state only after features for this timestamp
            # have been computed, so tied transactions cannot observe one another.
            for _, row in batch.iterrows():
                sender = _identifier(row.get("sender_account_id", pd.NA))
                receiver = _identifier(row.get("receiver_account_id", pd.NA))
                device = _identifier(row.get("device_id", pd.NA))
                ip = _identifier(row.get("ip_id", pd.NA))
                amount = _number(row.get("amount", pd.NA))
                label = row.get("label", pd.NA)
                if not pd.isna(label):
                    historical_labels_seen = True
                if sender is not None:
                    activity[sender].append((now, amount, receiver))
                    first_seen.setdefault(sender, now)
                    if receiver is not None:
                        sender_beneficiaries[sender].add(receiver)
                        beneficiary_senders[receiver].add(sender)
                        if sender != receiver:
                            account_neighbours[sender].add(receiver)
                            account_neighbours[receiver].add(sender)
                        if _label_is_suspicious(label, self.config.suspicious_labels):
                            suspicious_neighbours[sender].add(receiver)
                            suspicious_neighbours[receiver].add(sender)
                    if amount is not None:
                        historical_amount_sum[sender] += amount
                        historical_amount_count[sender] += 1
                        historical_amount_max[sender] = max(
                            amount, historical_amount_max.get(sender, amount)
                        )
                    radians = (now.hour + now.minute / 60) * math.pi / 12
                    old_x, old_y, old_count = hour_vectors.get(sender, (0.0, 0.0, 0))
                    hour_vectors[sender] = (
                        old_x + math.cos(radians),
                        old_y + math.sin(radians),
                        old_count + 1,
                    )
                if receiver is not None:
                    first_seen.setdefault(receiver, now)
                if device is not None:
                    device_counts[device] += 1
                    if sender is not None:
                        device_accounts[device].add(sender)
                if ip is not None:
                    ip_counts[ip] += 1
                    if sender is not None:
                        ip_accounts[ip].add(sender)

        feature_frame = pd.DataFrame.from_dict(records, orient="index").reindex(
            range(len(source))
        )
        result = source.copy()
        for name in feature_names:
            result[name] = feature_frame[name]
        result.attrs["feature_availability"] = {
            "row_count": len(source),
            "fields": source.attrs.get("feature_availability", {}).get("fields", {}),
            "features": _empty_availability(source, feature_names),
        }
        result.attrs["feature_columns"] = feature_names
        result.attrs["non_feature_columns"] = [
            name
            for name in (
                "transaction_id",
                "timestamp",
                "sender_account_id",
                "receiver_account_id",
                "currency",
                "merchant_id",
                "device_id",
                "ip_id",
                "location",
                "label",
                "sender_balance_before",
                "sender_balance_after",
                "receiver_balance_before",
                "receiver_balance_after",
            )
            if name in result.columns
        ]
        if availability_report_path is not None:
            self.write_availability_report(result, availability_report_path)
        return result

    @staticmethod
    def write_availability_report(
        transactions: pd.DataFrame, path: str | Path
    ) -> None:
        """Persist field and generated-feature availability as JSON."""
        report = transactions.attrs.get("feature_availability")
        if not isinstance(report, dict):
            raise ValueError("Transactions do not contain a feature availability report")
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
