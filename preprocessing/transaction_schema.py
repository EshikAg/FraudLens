"""Configurable canonical transaction schema and normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


CANONICAL_FIELDS = (
    "transaction_id",
    "timestamp",
    "sender_account_id",
    "receiver_account_id",
    "amount",
    "currency",
    "transaction_type",
    "channel",
    "merchant_id",
    "device_id",
    "ip_id",
    "location",
    "sender_balance_before",
    "sender_balance_after",
    "receiver_balance_before",
    "receiver_balance_after",
    "label",
)

_NUMERIC_FIELDS = {
    "amount",
    "sender_balance_before",
    "sender_balance_after",
    "receiver_balance_before",
    "receiver_balance_after",
}
_STRING_FIELDS = {
    "transaction_id",
    "sender_account_id",
    "receiver_account_id",
    "currency",
    "transaction_type",
    "channel",
    "merchant_id",
    "device_id",
    "ip_id",
}
_DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "transaction_id": ("transaction_id", "transactionId", "id", "txn_id"),
    "timestamp": ("timestamp", "transaction_timestamp", "datetime", "date", "time"),
    "sender_account_id": (
        "sender_account_id",
        "sender_id",
        "from_account",
        "source_account",
        "account_id",
    ),
    "receiver_account_id": (
        "receiver_account_id",
        "receiver_id",
        "to_account",
        "destination_account",
        "beneficiary_id",
    ),
    "amount": ("amount", "transaction_amount", "txn_amount", "value"),
    "currency": ("currency", "currency_code"),
    "transaction_type": ("transaction_type", "type", "txn_type"),
    "channel": ("channel", "payment_channel"),
    "merchant_id": ("merchant_id", "merchant", "payee_id"),
    "device_id": ("device_id", "device", "device_identifier"),
    "ip_id": ("ip_id", "ip_address", "ip", "source_ip"),
    "location": ("location", "geo_location", "city", "region"),
    "sender_balance_before": ("sender_balance_before", "balance_before"),
    "sender_balance_after": ("sender_balance_after", "balance_after"),
    "receiver_balance_before": ("receiver_balance_before",),
    "receiver_balance_after": ("receiver_balance_after",),
    "label": ("label", "is_fraud", "fraud_label", "is_suspicious"),
}


@dataclass(frozen=True)
class TransactionSchema:
    """Maps source columns to canonical fields without inventing values."""

    fields: Mapping[str, str | Mapping[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TransactionSchema:
        schema_path = Path(path)
        with schema_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError(f"Transaction schema must be a YAML mapping: {schema_path}")
        configured_fields = config.get("fields", {})
        if not isinstance(configured_fields, dict):
            raise ValueError("'fields' must be a YAML mapping")
        unknown_fields = set(configured_fields) - set(CANONICAL_FIELDS)
        if unknown_fields:
            raise ValueError(f"Unknown canonical fields: {sorted(unknown_fields)}")
        return cls(fields=configured_fields)

    @classmethod
    def default(cls) -> TransactionSchema:
        return cls()

    def _candidates(self, field_name: str) -> tuple[str, ...]:
        configured = self.fields.get(field_name)
        if isinstance(configured, str):
            return (configured,)
        if isinstance(configured, Mapping):
            source = configured.get("source")
            aliases = configured.get("aliases", ())
            if isinstance(aliases, str):
                aliases = (aliases,)
            if not isinstance(aliases, (list, tuple)):
                raise ValueError(f"Aliases for {field_name!r} must be a list")
            candidates = ([source] if source else []) + list(aliases)
            if not all(isinstance(candidate, str) for candidate in candidates):
                raise ValueError(f"Source names for {field_name!r} must be strings")
            return tuple(dict.fromkeys(candidates))
        if configured is not None:
            raise ValueError(f"Invalid mapping for canonical field {field_name!r}")
        return _DEFAULT_ALIASES[field_name]

    def normalize(self, source: pd.DataFrame) -> pd.DataFrame:
        """Return all canonical fields; absent source values remain nullable."""
        canonical = pd.DataFrame(index=range(len(source)))
        report: dict[str, dict[str, Any]] = {}

        for field_name in CANONICAL_FIELDS:
            candidates = self._candidates(field_name)
            source_column = next(
                (candidate for candidate in candidates if candidate in source.columns),
                None,
            )
            if source_column is None:
                values = pd.Series(pd.NA, index=canonical.index, dtype="object")
            else:
                values = source[source_column].reset_index(drop=True).copy()

            missing_count = int(values.isna().sum())
            if field_name in _NUMERIC_FIELDS:
                converted = pd.to_numeric(values, errors="coerce")
                invalid = values.notna() & converted.isna()
                if invalid.any():
                    examples = values[invalid].astype(str).head(3).tolist()
                    raise ValueError(
                        f"Invalid numeric values in {source_column!r} "
                        f"for {field_name!r}: {examples}"
                    )
                canonical[field_name] = converted
            elif field_name == "timestamp":
                converted = pd.to_datetime(values, errors="coerce", utc=True)
                invalid = values.notna() & converted.isna()
                if invalid.any():
                    examples = values[invalid].astype(str).head(3).tolist()
                    raise ValueError(
                        f"Invalid timestamps in {source_column!r}: {examples}"
                    )
                canonical[field_name] = converted
            elif field_name in _STRING_FIELDS:
                canonical[field_name] = values.astype("string")
            else:
                canonical[field_name] = values

            report[field_name] = {
                "available": source_column is not None,
                "source_column": source_column,
                "missing_value_count": missing_count,
                "missing_value_rate": (
                    missing_count / len(source) if len(source) else 0.0
                ),
            }

        canonical = canonical.loc[:, list(CANONICAL_FIELDS)]
        canonical.attrs["feature_availability"] = {
            "row_count": len(source),
            "fields": report,
        }
        return canonical

    @staticmethod
    def write_availability_report(
        transactions: pd.DataFrame, path: str | Path
    ) -> None:
        """Persist the source-field availability report as JSON."""
        report = transactions.attrs.get("feature_availability")
        if not isinstance(report, dict):
            raise ValueError("Transactions do not contain a feature availability report")
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
