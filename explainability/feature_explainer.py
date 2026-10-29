"""Model-agnostic local feature ablation for transaction fraud scores."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


ProbabilityPredictor = Callable[[Any], Any]


class FeatureExplainer:
    """Measure local feature-group influence by historical-baseline ablation.

    Contributions are score differences, not causal effects: positive values
    mean replacing the observed feature group with its historical baseline
    reduced the model's fraud probability.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        settings = dict(config or {})
        self.top_k = int(settings.get("top_k_features", 5))
        self.minimum_contribution = float(
            settings.get("minimum_feature_contribution", 0.0)
        )
        self.exclude_features = set(
            settings.get(
                "excluded_features",
                ("suspicious_neighbour_count", "label", "true_label"),
            )
        )
        if self.top_k < 1:
            raise ValueError("top_k_features must be positive")
        if self.minimum_contribution < 0:
            raise ValueError("minimum_feature_contribution cannot be negative")

    def explain(
        self,
        graph: Any,
        node_id: int,
        feature_columns: tuple[str, ...] | list[str],
        predictor: ProbabilityPredictor,
        *,
        cutoff: Any,
        original_probability: float | None = None,
    ) -> dict[str, Any]:
        """Return positive local ablation impacts using only earlier nodes."""
        features = graph.ndata.get("feature")
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("Explanation graph must contain a 2D feature tensor")
        if len(feature_columns) != features.shape[1]:
            raise ValueError("Feature names do not match the model feature width")
        if not 0 <= node_id < graph.num_nodes():
            raise IndexError(f"Transaction node {node_id} is outside the graph")
        if "timestamp" not in graph.ndata:
            raise ValueError("Explanation graph must include transaction timestamps")
        cutoff_timestamp = pd.Timestamp(cutoff)
        if cutoff_timestamp.tzinfo is None:
            cutoff_timestamp = cutoff_timestamp.tz_localize("UTC")
        else:
            cutoff_timestamp = cutoff_timestamp.tz_convert("UTC")
        raw_timestamps = graph.ndata["timestamp"].cpu().numpy()
        if np.issubdtype(raw_timestamps.dtype, np.number):
            finite_values = np.abs(raw_timestamps[np.isfinite(raw_timestamps)])
            magnitude = float(np.median(finite_values)) if len(finite_values) else 0.0
            unit = (
                "ns"
                if magnitude >= 1e17
                else "us"
                if magnitude >= 1e14
                else "ms"
                if magnitude >= 1e11
                else "s"
            )
            timestamps = pd.to_datetime(
                raw_timestamps, unit=unit, utc=True, errors="raise"
            )
        else:
            timestamps = pd.to_datetime(raw_timestamps, utc=True, errors="raise")
        prior_mask = np.asarray(timestamps < cutoff_timestamp, dtype=bool)
        prior_mask[node_id] = False
        prior_nodes = np.flatnonzero(prior_mask)
        baseline_values = (
            features[torch.as_tensor(prior_nodes, dtype=torch.long)].mean(dim=0)
            if len(prior_nodes)
            else torch.zeros(features.shape[1], dtype=features.dtype)
        )

        grouped_columns: dict[str, list[int]] = defaultdict(list)
        for index, name in enumerate(feature_columns):
            group = self._group_name(str(name))
            if group not in self.exclude_features and not group.startswith("suspicious_"):
                grouped_columns[group].append(index)
        baseline_features = features.clone()
        if original_probability is None:
            original_probability = self._node_probability(predictor(graph), node_id)
        if not np.isfinite(original_probability) or not 0 <= original_probability <= 1:
            raise ValueError("Model returned an invalid fraud probability")

        impacts: list[dict[str, Any]] = []
        for group, indices in sorted(grouped_columns.items()):
            perturbed = graph.clone()
            perturbed_features = baseline_features.clone()
            selected_columns = torch.as_tensor(indices, dtype=torch.long)
            perturbed_features[node_id, selected_columns] = baseline_values[
                selected_columns
            ]
            perturbed.ndata["feature"] = perturbed_features
            ablated_probability = self._node_probability(
                predictor(perturbed), node_id
            )
            contribution = float(original_probability - ablated_probability)
            if not np.isfinite(contribution):
                raise ValueError(f"Non-finite ablation impact for feature {group!r}")
            impacts.append(
                {
                    "feature": group,
                    "contribution": contribution,
                    "ablated_probability": float(ablated_probability),
                    "observed_values": [
                        float(features[node_id, index].item())
                        for index in indices
                    ],
                    "baseline_values": [
                        float(baseline_values[index].item()) for index in indices
                    ],
                    "historical_reference_count": int(len(prior_nodes)),
                }
            )

        positive = [
            item
            for item in impacts
            if item["contribution"] > self.minimum_contribution
        ]
        positive.sort(
            key=lambda item: (-item["contribution"], item["feature"])
        )
        return {
            "baseline_method": "strictly_historical_feature_group_mean",
            "historical_reference_count": int(len(prior_nodes)),
            "features": positive[: self.top_k],
            "evaluated_feature_groups": len(grouped_columns),
        }

    @staticmethod
    def _group_name(name: str) -> str:
        if "=" in name:
            return name.split("=", 1)[0]
        if name.endswith("__available"):
            return name[: -len("__available")]
        return name

    @staticmethod
    def _node_probability(output: Any, node_id: int) -> float:
        if isinstance(output, torch.Tensor):
            values = output.detach().cpu().numpy()
        else:
            values = np.asarray(output)
        if values.ndim == 2:
            if values.shape[1] < 2:
                raise ValueError("Predictor output must contain fraud probabilities")
            value = values[node_id, 1]
        elif values.ndim == 1:
            value = values[node_id]
        else:
            raise ValueError("Predictor output must be a vector or class matrix")
        probability = float(value)
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Predictor must return fraud probabilities in [0, 1]")
        return probability
