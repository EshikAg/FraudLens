"""Fail-closed model promotion criteria across validation and retention windows."""

from __future__ import annotations

from typing import Any, Mapping


class PromotionManager:
    """Require candidate quality to meet configured bounds on every window."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        settings = dict(config or {})
        self.metric = str(settings.get("metric", "f1"))
        self.minimum_recent_improvement = float(
            settings.get("minimum_recent_improvement", 0.0)
        )
        self.maximum_validation_drop = float(
            settings.get("maximum_validation_drop", 0.02)
        )
        self.maximum_replay_drop = float(settings.get("maximum_replay_drop", 0.02))
        self.minimum_recent_metric = settings.get("minimum_recent_metric")
        if min(
            self.maximum_validation_drop,
            self.maximum_replay_drop,
        ) < 0:
            raise ValueError("Maximum metric drop tolerances cannot be negative")
        if self.minimum_recent_metric is not None:
            self.minimum_recent_metric = float(self.minimum_recent_metric)
            if not 0 <= self.minimum_recent_metric <= 1:
                raise ValueError("minimum_recent_metric must be between 0 and 1")

    def evaluate(
        self,
        current: Mapping[str, Mapping[str, float]],
        candidate: Mapping[str, Mapping[str, float]],
    ) -> dict[str, Any]:
        """Compare candidate and current on validation, recent, and replay windows."""
        required_windows = ("validation", "recent", "replay")
        checks: dict[str, Any] = {}
        failures: list[str] = []
        for window in required_windows:
            try:
                current_score = float(current[window][self.metric])
                candidate_score = float(candidate[window][self.metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Promotion requires finite {self.metric!r} metrics for {window}"
                ) from exc
            if not (0 <= current_score <= 1 and 0 <= candidate_score <= 1):
                raise ValueError(
                    f"Promotion {self.metric!r} scores must be finite values in [0, 1]"
                )
            delta = candidate_score - current_score
            if window == "validation":
                accepted = delta >= -self.maximum_validation_drop
                requirement = f"candidate >= current - {self.maximum_validation_drop}"
            elif window == "replay":
                accepted = delta >= -self.maximum_replay_drop
                requirement = f"candidate >= current - {self.maximum_replay_drop}"
            else:
                accepted = delta >= self.minimum_recent_improvement
                requirement = f"candidate >= current + {self.minimum_recent_improvement}"
                if self.minimum_recent_metric is not None:
                    recent_floor_met = candidate_score >= self.minimum_recent_metric
                    accepted = accepted and recent_floor_met
                    if not recent_floor_met:
                        requirement += (
                            f" and candidate >= {self.minimum_recent_metric}"
                        )
            checks[window] = {
                "current": current_score,
                "candidate": candidate_score,
                "delta": delta,
                "requirement": requirement,
                "passed": bool(accepted),
            }
            if not accepted:
                failures.append(window)
        return {
            "promote": not failures,
            "metric": self.metric,
            "checks": checks,
            "failed_windows": failures,
            "replay_retention_passed": checks["replay"]["passed"],
        }
