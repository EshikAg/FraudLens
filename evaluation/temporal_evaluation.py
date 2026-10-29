"""Chronological experiment partitions and rolling-window iterators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class TemporalPeriods:
    """Four non-overlapping chronological partitions."""

    training: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    streaming: pd.DataFrame


def temporal_periods(
    transactions: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    fractions: tuple[float, float, float, float] = (0.5, 0.15, 0.15, 0.2),
) -> TemporalPeriods:
    """Split on unique timestamps so simultaneous transactions never cross periods."""
    if timestamp_column not in transactions:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")
    if transactions.empty or transactions[timestamp_column].isna().any():
        raise ValueError("Transactions must be non-empty and timestamps cannot be null")
    if len(fractions) != 4 or any(
        not math.isfinite(value) or value <= 0 for value in fractions
    ):
        raise ValueError("fractions must contain four positive values")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")

    ordered = transactions.copy()
    ordered[timestamp_column] = pd.to_datetime(
        ordered[timestamp_column], utc=True, errors="raise"
    )
    ordered = ordered.sort_values(timestamp_column, kind="stable")
    times = pd.Index(ordered[timestamp_column].drop_duplicates())
    if len(times) < 4:
        raise ValueError("At least four distinct timestamps are required")

    proportions = fractions[:-1]
    boundaries: list[int] = []
    cumulative = 0.0
    previous = 0
    for fraction in proportions:
        cumulative += fraction
        boundary = int(len(times) * cumulative)
        boundary = max(previous + 1, min(boundary, len(times) - (3 - len(boundaries))))
        boundaries.append(boundary)
        previous = boundary
    boundaries.append(len(times))

    time_chunks = [
        times[boundaries[index - 1] if index else 0 : boundaries[index]]
        for index in range(4)
    ]
    periods = [
        ordered[ordered[timestamp_column].isin(chunk)].copy()
        for chunk in time_chunks
    ]
    if any(frame.empty for frame in periods):
        raise ValueError("Temporal fractions produced an empty partition")
    return TemporalPeriods(*periods)


def rolling_windows(
    transactions: pd.DataFrame,
    *,
    initial_train_timestamps: int,
    test_timestamps: int,
    step_timestamps: int | None = None,
    timestamp_column: str = "timestamp",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield expanding-history train/test windows without timestamp overlap."""
    if timestamp_column not in transactions:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")
    if initial_train_timestamps < 1 or test_timestamps < 1:
        raise ValueError("Initial train and test timestamp counts must be positive")
    step = test_timestamps if step_timestamps is None else step_timestamps
    if step < 1:
        raise ValueError("step_timestamps must be positive")
    ordered = transactions.copy()
    ordered[timestamp_column] = pd.to_datetime(
        ordered[timestamp_column], utc=True, errors="raise"
    )
    ordered = ordered.sort_values(timestamp_column, kind="stable")
    times = list(ordered[timestamp_column].drop_duplicates())
    if initial_train_timestamps + test_timestamps > len(times):
        raise ValueError("Not enough distinct timestamps for one rolling window")

    for test_start in range(initial_train_timestamps, len(times) - test_timestamps + 1, step):
        train = ordered[ordered[timestamp_column].isin(times[:test_start])].copy()
        test = ordered[
            ordered[timestamp_column].isin(times[test_start : test_start + test_timestamps])
        ].copy()
        yield train, test
