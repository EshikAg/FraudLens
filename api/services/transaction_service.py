"""Validate online rows and retain bounded transaction context."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import pandas as pd

from models.banking_gaap import BankingGAAP, _resolve_path
from preprocessing.transaction_schema import CANONICAL_FIELDS, TransactionSchema


class InvalidTransactionError(ValueError):
    """Raised when a transaction is invalid for the trained feature pipeline."""


class TransactionConflictError(ValueError):
    """Raised when a transaction ID has already been scored."""


class TransactionService:
    """Canonicalize requests and expose only strictly earlier graph context."""

    def __init__(
        self,
        model: BankingGAAP,
        *,
        max_history_transactions: int = 10000,
        schema_path: str | Path = "configs/transaction_schema.yaml",
    ) -> None:
        if max_history_transactions < 1:
            raise ValueError("max_history_transactions must be positive")
        graph, _ = model._require_fitted()
        self.model = model
        self.max_history_transactions = max_history_transactions
        self.schema = TransactionSchema.from_yaml(_resolve_path(schema_path))
        self._lock = RLock()
        self._seen_ids = set(graph.node_table["transaction_id"].astype(str))
        self._online_rows: deque[dict[str, Any]] = deque(
            maxlen=max_history_transactions
        )
        self._base_rows = graph.node_table.loc[
            :, [name for name in CANONICAL_FIELDS if name in graph.node_table]
        ].copy()
        if "label" not in self._base_rows:
            self._base_rows["label"] = pd.NA

    def prepare(
        self, payload: Mapping[str, Any]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return normalized historical rows and the current canonical row."""
        if payload.get("label") is not None:
            raise InvalidTransactionError(
                "label must not be supplied to the prediction endpoint"
            )
        try:
            normalized = self.schema.normalize(pd.DataFrame([dict(payload)]))
        except (TypeError, ValueError) as exc:
            raise InvalidTransactionError(str(exc)) from exc

        row = normalized.iloc[0].to_dict()
        transaction_id = str(row["transaction_id"])
        if not transaction_id or pd.isna(row["transaction_id"]):
            raise InvalidTransactionError("transaction_id is required")
        timestamp = pd.Timestamp(row["timestamp"])
        if pd.isna(timestamp):
            raise InvalidTransactionError("timestamp is required and must be valid")
        if timestamp.tzinfo is None:
            raise InvalidTransactionError("timestamp must include a timezone")
        row["timestamp"] = timestamp.tz_convert("UTC")
        for name in (
            "sender_account_id",
            "receiver_account_id",
            "currency",
            "transaction_type",
            "channel",
        ):
            if pd.isna(row[name]) or not str(row[name]).strip():
                raise InvalidTransactionError(f"{name} is required")
        if pd.isna(row["amount"]) or float(row["amount"]) <= 0:
            raise InvalidTransactionError("amount must be greater than zero")

        with self._lock:
            if transaction_id in self._seen_ids:
                raise TransactionConflictError(
                    f"transaction_id {transaction_id!r} has already been scored"
                )
            history = pd.concat(
                [self._base_rows, pd.DataFrame(list(self._online_rows))],
                ignore_index=True,
            )
            history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
            # The inference graph contains no transaction that occurred at or
            # after the request timestamp, even if a batch arrived out of order.
            history = history.loc[history["timestamp"] < row["timestamp"]]
            history = history.tail(self.max_history_transactions).copy()
        return history.reset_index(drop=True), row

    def commit(self, row: Mapping[str, Any]) -> None:
        """Retain a successful prediction for later strictly historical context."""
        transaction_id = str(row["transaction_id"])
        with self._lock:
            if transaction_id in self._seen_ids:
                raise TransactionConflictError(
                    f"transaction_id {transaction_id!r} has already been scored"
                )
            self._seen_ids.add(transaction_id)
            self._online_rows.append(dict(row))
