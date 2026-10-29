"""Prediction latency, throughput, and optional Python allocation measurements."""

from __future__ import annotations

import statistics
import time
import tracemalloc
from collections.abc import Callable
from typing import Any


def measure_prediction_latency(
    predict: Callable[[], Any],
    *,
    repeats: int = 30,
    warmup: int = 3,
) -> dict[str, float | int | None]:
    """Measure wall-clock inference and peak Python allocations for a callable."""
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup cannot be negative")
    for _ in range(warmup):
        predict()

    durations: list[float] = []
    tracemalloc.start()
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            predict()
            durations.append(time.perf_counter() - started)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    ordered = sorted(durations)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    mean_seconds = statistics.fmean(durations)
    return {
        "repeats": repeats,
        "mean_latency_ms": mean_seconds * 1000,
        "median_latency_ms": statistics.median(durations) * 1000,
        "p95_latency_ms": ordered[p95_index] * 1000,
        "throughput_per_second": (1.0 / mean_seconds) if mean_seconds else None,
        "python_peak_memory_bytes": int(peak_bytes),
    }
