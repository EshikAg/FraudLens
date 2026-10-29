"""Chronological train/validation/test splitting without timestamp overlap."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from preprocessing.validation import require_temporal_order


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(
    transactions: pd.DataFrame,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Split by distinct timestamps so simultaneous rows stay in one partition."""
    require_temporal_order(transactions)
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")

    ordered = transactions.sort_values("timestamp", kind="stable")
    times = pd.Index(ordered["timestamp"].drop_duplicates())
    if len(times) < 3:
        raise ValueError("At least three distinct timestamps are required to split")
    train_end = max(1, min(len(times) - 2, int(len(times) * train_fraction)))
    validation_end = max(
        train_end + 1,
        min(len(times) - 1, int(len(times) * (train_fraction + validation_fraction))),
    )
    train_cutoff = times[train_end]
    validation_cutoff = times[validation_end]
    train = ordered[ordered["timestamp"] < train_cutoff].copy()
    validation = ordered[
        (ordered["timestamp"] >= train_cutoff)
        & (ordered["timestamp"] < validation_cutoff)
    ].copy()
    test = ordered[ordered["timestamp"] >= validation_cutoff].copy()
    return TemporalSplit(train=train, validation=validation, test=test)
