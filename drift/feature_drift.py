"""Window-based data-drift detection for numeric and categorical features."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


def _timestamp(value: Any | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    return pd.Timestamp(value).isoformat()


def _psi(reference: np.ndarray, current: np.ndarray, bins: int) -> float:
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        edges = np.array([-np.inf, float(reference[0]), np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)
    epsilon = 1e-6
    reference_share = np.clip(reference_counts / reference_counts.sum(), epsilon, None)
    current_share = np.clip(current_counts / current_counts.sum(), epsilon, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def _categorical_distributions(
    reference: pd.Series, current: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    categories = sorted(
        set(reference.astype(str).unique()) | set(current.astype(str).unique())
    )
    reference_counts = reference.astype(str).value_counts().reindex(categories, fill_value=0)
    current_counts = current.astype(str).value_counts().reindex(categories, fill_value=0)
    return reference_counts.to_numpy(dtype=float), current_counts.to_numpy(dtype=float)


class FeatureDriftDetector:
    """Compare a fixed reference window against successive current windows."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        settings = dict(config or {})
        self.features = tuple(settings.get("features", ()))
        self.methods = tuple(settings.get("methods", ("psi", "ks", "jensen_shannon")))
        unsupported = set(self.methods) - {"psi", "ks", "jensen_shannon"}
        if unsupported:
            raise ValueError(f"Unsupported feature drift methods: {sorted(unsupported)}")
        self.psi_threshold = float(settings.get("psi_threshold", 0.2))
        self.ks_alpha = float(settings.get("ks_alpha", 0.01))
        self.js_threshold = float(settings.get("js_threshold", 0.1))
        self.missing_rate_threshold = float(settings.get("missing_rate_threshold", 0.1))
        self.min_samples = int(settings.get("min_samples", 20))
        self.bins = int(settings.get("bins", 10))
        self.consecutive_windows = int(settings.get("consecutive_windows", 1))
        if self.min_samples < 2 or self.bins < 2 or self.consecutive_windows < 1:
            raise ValueError("min_samples, bins, and consecutive_windows must be positive")
        if not 0 < self.ks_alpha < 1:
            raise ValueError("ks_alpha must be between 0 and 1")
        if min(self.psi_threshold, self.js_threshold, self.missing_rate_threshold) <= 0:
            raise ValueError("Feature drift thresholds must be positive")
        if self.missing_rate_threshold > 1:
            raise ValueError("missing_rate_threshold must not exceed one")
        self._reference: pd.DataFrame | None = None
        self._streaks: dict[str, int] = defaultdict(int)

    def set_reference(self, reference_window: pd.DataFrame) -> None:
        """Set the reference distribution, rejecting empty or non-tabular input."""
        if not isinstance(reference_window, pd.DataFrame):
            raise TypeError("reference_window must be a pandas DataFrame")
        if reference_window.empty:
            raise ValueError("reference_window must contain at least one row")
        self._reference = reference_window.copy()
        self._streaks.clear()

    def compare(
        self, current_window: pd.DataFrame, timestamp: Any | None = None
    ) -> dict[str, Any]:
        """Compare a current feature window and return per-feature statistics."""
        if self._reference is None:
            raise RuntimeError("Set a reference window before comparing feature drift")
        if not isinstance(current_window, pd.DataFrame):
            raise TypeError("current_window must be a pandas DataFrame")
        if current_window.empty:
            raise ValueError("current_window must contain at least one row")

        columns = self.features or tuple(self._reference.columns)
        results: dict[str, Any] = {}
        affected: list[str] = []
        max_score = 0.0
        for name in columns:
            if name not in self._reference or name not in current_window:
                results[name] = {"available": False, "reason": "feature_missing"}
                self._streaks[name] = 0
                continue
            reference_series = self._reference[name].dropna()
            current_series = current_window[name].dropna()
            stats: dict[str, Any] = {
                "available": False,
                "reference_count": int(len(reference_series)),
                "current_count": int(len(current_series)),
                "reference_missing_rate": float(self._reference[name].isna().mean()),
                "current_missing_rate": float(current_window[name].isna().mean()),
            }
            missing_rate_change = abs(
                stats["current_missing_rate"] - stats["reference_missing_rate"]
            )
            missing_rate_triggered = missing_rate_change >= self.missing_rate_threshold
            stats["missing_rate_change"] = float(missing_rate_change)
            if min(len(reference_series), len(current_series)) < self.min_samples:
                stats["triggered_methods"] = (
                    ["missing_rate"] if missing_rate_triggered else []
                )
                self._streaks[name] = (
                    self._streaks[name] + 1
                    if missing_rate_triggered
                    else 0
                )
                stats["consecutive_windows"] = self._streaks[name]
                stats["drift_detected"] = (
                    self._streaks[name] >= self.consecutive_windows
                )
                if stats["drift_detected"]:
                    stats["available"] = True
                    affected.append(name)
                    max_score = max(
                        max_score,
                        missing_rate_change / self.missing_rate_threshold,
                    )
                    results[name] = stats
                    continue
                stats["reason"] = "insufficient_non_missing_samples"
                results[name] = stats
                continue

            numeric = pd.api.types.is_numeric_dtype(reference_series.dtype) and (
                pd.api.types.is_numeric_dtype(current_series.dtype)
            )
            if numeric:
                reference_values = reference_series.to_numpy(dtype=float)
                current_values = current_series.to_numpy(dtype=float)
                if not np.isfinite(reference_values).all() or not np.isfinite(current_values).all():
                    raise ValueError(f"Feature {name!r} contains infinite numeric values")
                if "psi" in self.methods:
                    stats["psi"] = _psi(reference_values, current_values, self.bins)
                if "ks" in self.methods:
                    ks_result = ks_2samp(reference_values, current_values)
                    stats["ks_statistic"] = float(ks_result.statistic)
                    stats["ks_p_value"] = float(ks_result.pvalue)
                if "jensen_shannon" in self.methods:
                    edges = np.unique(
                        np.quantile(
                            reference_values,
                            np.linspace(0.0, 1.0, self.bins + 1),
                        )
                    )
                    if len(edges) < 2:
                        edges = np.array([-np.inf, float(reference_values[0]), np.inf])
                    else:
                        edges[0] = -np.inf
                        edges[-1] = np.inf
                    ref_counts = np.histogram(reference_values, bins=edges)[0] + 1e-6
                    cur_counts = np.histogram(current_values, bins=edges)[0] + 1e-6
                    stats["jensen_shannon"] = float(
                        jensenshannon(ref_counts / ref_counts.sum(), cur_counts / cur_counts.sum())
                    )
                method_detections = [
                    ("psi", stats.get("psi", 0.0) >= self.psi_threshold),
                    (
                        "ks",
                        stats.get("ks_p_value", 1.0) <= self.ks_alpha,
                    ),
                    (
                        "jensen_shannon",
                        stats.get("jensen_shannon", 0.0) >= self.js_threshold,
                    ),
                ]
            else:
                ref_counts, cur_counts = _categorical_distributions(
                    reference_series, current_series
                )
                epsilon = 1e-6
                ref_share = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
                cur_share = np.clip(cur_counts / cur_counts.sum(), epsilon, None)
                ref_share /= ref_share.sum()
                cur_share /= cur_share.sum()
                if "psi" in self.methods:
                    stats["psi"] = float(
                        np.sum((cur_share - ref_share) * np.log(cur_share / ref_share))
                    )
                if "jensen_shannon" in self.methods:
                    stats["jensen_shannon"] = float(
                        jensenshannon(ref_share, cur_share)
                    )
                method_detections = [
                    ("psi", stats.get("psi", 0.0) >= self.psi_threshold),
                    (
                        "jensen_shannon",
                        stats.get("jensen_shannon", 0.0) >= self.js_threshold,
                    ),
                ]

            stats["available"] = True
            triggered = [method for method, value in method_detections if value]
            if missing_rate_triggered:
                triggered.append("missing_rate")
            self._streaks[name] = self._streaks[name] + 1 if triggered else 0
            stats["triggered_methods"] = triggered
            stats["consecutive_windows"] = self._streaks[name]
            stats["drift_detected"] = (
                self._streaks[name] >= self.consecutive_windows
            )
            if stats["drift_detected"]:
                affected.append(name)
                score_candidates = [
                    stats.get("psi", 0.0) / self.psi_threshold,
                    stats.get("jensen_shannon", 0.0) / self.js_threshold,
                    missing_rate_change / self.missing_rate_threshold,
                    (
                        1.0 - stats.get("ks_p_value", 1.0)
                    ) / (1.0 - self.ks_alpha),
                ]
                max_score = max(max_score, *score_candidates)
            results[name] = stats

        return {
            "drift_detected": bool(affected),
            "drift_type": "data" if affected else None,
            "drift_score": float(max_score),
            "timestamp": _timestamp(timestamp),
            "affected_features": affected,
            "window_statistics": {
                "reference_rows": int(len(self._reference)),
                "current_rows": int(len(current_window)),
                "features": results,
            },
        }

    def reset(self, reference_window: pd.DataFrame | None = None) -> None:
        """Clear state and optionally initialize a new reference window."""
        self._reference = None
        self._streaks.clear()
        if reference_window is not None:
            self.set_reference(reference_window)
