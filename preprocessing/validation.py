"""Validation helpers for canonical transactions and temporal inputs."""

from __future__ import annotations

import pandas as pd

from preprocessing.transaction_schema import CANONICAL_FIELDS


def validate_transactions(transactions: pd.DataFrame) -> None:
    """Raise a clear error for malformed canonical transaction data."""
    missing_columns = [name for name in CANONICAL_FIELDS if name not in transactions]
    if missing_columns:
        raise ValueError(f"Missing canonical transaction fields: {missing_columns}")
    timestamps = pd.to_datetime(transactions["timestamp"], errors="coerce", utc=True)
    invalid = transactions["timestamp"].notna() & timestamps.isna()
    if invalid.any():
        raise ValueError("Canonical transactions contain invalid timestamps")
    amounts = pd.to_numeric(transactions["amount"], errors="coerce")
    invalid_amounts = transactions["amount"].notna() & amounts.isna()
    if invalid_amounts.any():
        raise ValueError("Canonical transactions contain non-numeric amounts")


def require_temporal_order(transactions: pd.DataFrame) -> None:
    """Require usable timestamps before temporal feature generation."""
    validate_transactions(transactions)
    if transactions["timestamp"].isna().any():
        raise ValueError("Temporal processing requires a timestamp for every row")
