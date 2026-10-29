"""Unified data- and concept-drift monitoring interface."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from drift.concept_drift import PageHinkleyConceptDetector
from drift.drift_report import save_drift_report
from drift.feature_drift import FeatureDriftDetector


class DriftMonitor:
    """Coordinate feature-distribution and delayed-label concept monitoring."""

    def __init__(self, config: Mapping[str, Any] | str | Path | None = None) -> None:
        if isinstance(config, (str, Path)):
            import yaml

            config_path = Path(config)
            with config_path.open("r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream)
            if not isinstance(loaded, dict):
                raise ValueError(f"Drift configuration must be a mapping: {config_path}")
            settings = loaded
        else:
            settings = dict(config or {})
        self.config = settings
        self.feature_detector = FeatureDriftDetector(settings.get("feature_drift", {}))
        self.concept_detector = PageHinkleyConceptDetector(settings.get("concept_drift", {}))
        self._events: list[dict[str, Any]] = []
        self._feature_status: dict[str, Any] | None = None

    def update_feature_distribution(
        self,
        current_window: pd.DataFrame,
        reference_window: pd.DataFrame | None = None,
        timestamp: Any | None = None,
    ) -> dict[str, Any]:
        """Set an optional reference window and compare the current window."""
        if reference_window is not None:
            self.feature_detector.set_reference(reference_window)
        result = self.feature_detector.compare(current_window, timestamp=timestamp)
        self._feature_status = result
        if result["drift_detected"]:
            self._events.append(result)
        return result

    def update_prediction(
        self, transaction_id: str, prediction: int | float | bool, timestamp: Any | None = None
    ) -> dict[str, Any]:
        """Register a model prediction whose label may arrive later."""
        return self.concept_detector.update_prediction(transaction_id, prediction, timestamp)

    def update_label(
        self, transaction_id: str, label: int | bool, timestamp: Any | None = None
    ) -> dict[str, Any]:
        """Update concept monitoring when the transaction label becomes available."""
        result = self.concept_detector.update_label(transaction_id, label, timestamp)
        if result["drift_detected"]:
            self._events.append(result)
        return result

    def check_drift(self) -> dict[str, Any]:
        """Return current combined drift status and associated window statistics."""
        concept = self.concept_detector.check_drift()
        feature = self._feature_status
        active = [event for event in self._events if event.get("drift_detected")]
        kinds = {report["drift_type"] for report in active}
        latest_timestamp = max(
            (report["timestamp"] for report in active if report.get("timestamp")),
            default=None,
        )
        if latest_timestamp is None and (feature or concept.get("timestamp")):
            latest_timestamp = (feature or concept).get("timestamp")
        return {
            "drift_detected": bool(active),
            "drift_type": "both" if len(kinds) > 1 else (next(iter(kinds)) if kinds else None),
            "drift_score": max((float(report["drift_score"]) for report in active), default=0.0),
            "timestamp": latest_timestamp or datetime.now(timezone.utc).isoformat(),
            "affected_features": sorted(
                {
                    feature_name
                    for report in active
                    for feature_name in report.get("affected_features", [])
                }
            ),
            "window_statistics": {
                "feature_drift": feature.get("window_statistics", {}) if feature else {},
                "concept_drift": concept.get("window_statistics", {}),
            },
        }

    def report(self) -> dict[str, Any]:
        """Build a JSON-serializable monitoring report with detected events."""
        return {
            "events": list(self._events),
            "latest_status": self.check_drift(),
            "concept_performance": self.concept_detector.performance_statistics(),
        }

    def save_report(self, path: str | Path) -> Path:
        """Persist the current drift report as structured JSON."""
        return save_drift_report(self.report(), path)

    def reset(self, reference_window: pd.DataFrame | None = None) -> None:
        """Clear detector state and optionally set a new reference window."""
        self.feature_detector.reset(reference_window)
        self.concept_detector.reset()
        self._events.clear()
        self._feature_status = None
