"""Expose streaming feature/prediction updates and current drift status."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import pandas as pd

from drift.drift_monitor import DriftMonitor


class DriftService:
    """Maintain a bounded online feature window for the existing detector."""

    def __init__(
        self,
        config: Mapping[str, Any] | str | Path | None = None,
        reference_window: pd.DataFrame | None = None,
        *,
        window_size: int = 500,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.monitor = DriftMonitor(config)
        self._window: deque[dict[str, Any]] = deque(maxlen=window_size)
        self._lock = RLock()
        if reference_window is not None and not reference_window.empty:
            self.monitor.feature_detector.set_reference(reference_window)

    def observe(
        self,
        features: Mapping[str, Any],
        *,
        transaction_id: str,
        probability: float,
        timestamp: Any,
    ) -> dict[str, Any]:
        """Update feature distribution and register a prediction for delayed labels."""
        with self._lock:
            self._window.append(dict(features))
            if len(self._window) >= self.monitor.feature_detector.min_samples:
                current = pd.DataFrame(list(self._window))
                self.monitor.update_feature_distribution(current, timestamp=timestamp)
            self.monitor.update_prediction(
                transaction_id, probability, timestamp=timestamp
            )
            return self.monitor.check_drift()

    def update_label(
        self, transaction_id: str, label: int, timestamp: Any | None = None
    ) -> dict[str, Any]:
        """Join a subsequently available binary label to its earlier prediction."""
        with self._lock:
            return self.monitor.update_label(
                transaction_id, label, timestamp=timestamp
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self.monitor.check_drift()
