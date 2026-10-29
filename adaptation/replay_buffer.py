"""Bounded, labeled transaction storage with temporal-decay replay sampling."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


class ReplayBuffer:
    """Store labeled transactions and sample them without replacement.

    Exponential temporal decay changes sampling probability, not retention:
    ``minimum_weight`` prevents older examples from becoming impossible to
    replay.
    """

    def __init__(
        self,
        capacity: int,
        *,
        seed: int = 42,
        half_life_days: float = 180.0,
        minimum_weight: float = 0.1,
    ) -> None:
        if capacity < 1:
            raise ValueError("ReplayBuffer capacity must be positive")
        if not math.isfinite(half_life_days) or half_life_days <= 0:
            raise ValueError("half_life_days must be positive and finite")
        if not 0 < minimum_weight <= 1:
            raise ValueError("minimum_weight must be in (0, 1]")
        self.capacity = int(capacity)
        self.half_life_days = float(half_life_days)
        self.minimum_weight = float(minimum_weight)
        self._rng = np.random.default_rng(seed)
        self._rows: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, transactions: pd.DataFrame) -> int:
        """Add labeled transactions; conflicting duplicate IDs are rejected."""
        required = {"transaction_id", "timestamp", "label"}
        missing = required - set(transactions.columns)
        if missing:
            raise ValueError(f"Replay rows are missing required columns: {sorted(missing)}")
        if transactions["transaction_id"].isna().any():
            raise ValueError("Replay transaction_id values cannot be missing")
        if transactions["transaction_id"].astype(str).duplicated().any():
            raise ValueError("Replay input contains duplicate transaction_id values")
        labeled = transactions.loc[transactions["label"].notna()].copy()
        if labeled.empty:
            return 0
        labeled["transaction_id"] = labeled["transaction_id"].astype(str)
        labeled["timestamp"] = pd.to_datetime(labeled["timestamp"], utc=True, errors="raise")
        incoming = {
            str(row["transaction_id"]): row
            for row in labeled.to_dict(orient="records")
        }
        for transaction_id, row in incoming.items():
            existing = self._rows.get(transaction_id)
            if existing is not None and not _rows_equal(existing, row):
                raise ValueError(
                    f"Conflicting replay data for transaction_id {transaction_id!r}"
                )
        added = 0
        for transaction_id, row in incoming.items():
            if transaction_id not in self._rows:
                self._rows[transaction_id] = row
                added += 1
        if len(self._rows) > self.capacity:
            ordered = sorted(
                self._rows.values(),
                key=lambda row: (pd.Timestamp(row["timestamp"]), row["transaction_id"]),
            )
            self._rows = {
                str(row["transaction_id"]): row
                for row in ordered[-self.capacity :]
            }
        return added

    def frame(self) -> pd.DataFrame:
        """Return a chronological copy of the currently retained rows."""
        if not self._rows:
            return pd.DataFrame(columns=["transaction_id", "timestamp", "label"])
        result = pd.DataFrame(self._rows.values())
        return result.sort_values(
            ["timestamp", "transaction_id"], kind="stable"
        ).reset_index(drop=True)

    def sample(
        self,
        count: int,
        *,
        as_of: Any | None = None,
        exclude_ids: Iterable[str] = (),
    ) -> pd.DataFrame:
        """Sample labeled rows with replacement-free temporal-decay weights."""
        if count < 0:
            raise ValueError("Replay sample count cannot be negative")
        if count == 0 or not self._rows:
            return self.frame().iloc[:0].copy()
        excluded = {str(value) for value in exclude_ids}
        candidates = [
            row for key, row in self._rows.items() if key not in excluded
        ]
        if not candidates:
            return self.frame().iloc[:0].copy()
        take = min(int(count), len(candidates))
        end = (
            pd.Timestamp(as_of)
            if as_of is not None
            else max(pd.Timestamp(row["timestamp"]) for row in candidates)
        )
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        ages_days = np.array(
            [
                max(
                    0.0,
                    (
                        end
                        - pd.Timestamp(row["timestamp"]).tz_convert("UTC")
                    ).total_seconds()
                    / 86400.0,
                )
                for row in candidates
            ],
            dtype=float,
        )
        weights = np.maximum(
            np.exp(-math.log(2.0) * ages_days / self.half_life_days),
            self.minimum_weight,
        )
        probabilities = weights / weights.sum()
        indices = self._rng.choice(
            len(candidates), size=take, replace=False, p=probabilities
        )
        result = pd.DataFrame([candidates[int(index)] for index in indices])
        return result.sort_values(
            ["timestamp", "transaction_id"], kind="stable"
        ).reset_index(drop=True)

    def remove(self, transaction_ids: Iterable[str]) -> None:
        """Remove a known set of IDs, rejecting unknown IDs."""
        identifiers = {str(value) for value in transaction_ids}
        missing = identifiers - self._rows.keys()
        if missing:
            raise KeyError(f"Cannot remove unknown replay IDs: {sorted(missing)[:5]}")
        for transaction_id in identifiers:
            del self._rows[transaction_id]

    def clear(self) -> None:
        """Remove all stored transactions."""
        self._rows.clear()


def _rows_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_frame = pd.DataFrame([left]).sort_index(axis=1)
    right_frame = pd.DataFrame([right]).sort_index(axis=1)
    try:
        pd.testing.assert_frame_equal(
            left_frame, right_frame, check_dtype=False, check_like=True
        )
    except AssertionError:
        return False
    return True
