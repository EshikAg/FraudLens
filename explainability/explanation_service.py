"""Compose transaction score, local feature effects, and causal graph evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from explainability.feature_explainer import FeatureExplainer
from explainability.graph_explainer import GraphExplainer
from explainability.reason_codes import ReasonCodeGenerator
from models.banking_gaap import BankingGAAP


class ExplanationService:
    """Build reproducible analyst explanations from an existing GAAP model."""

    def __init__(self, config: Mapping[str, Any] | str | Path | None = None) -> None:
        if isinstance(config, (str, Path)):
            path = Path(config)
            with path.open("r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream)
            if not isinstance(loaded, dict):
                raise ValueError(f"Explainability config must be a YAML mapping: {path}")
            settings = loaded
        else:
            settings = dict(config or {})
        self.config = settings
        self.feature_explainer = FeatureExplainer(
            settings.get("feature_attribution", {})
        )
        self.graph_explainer = GraphExplainer(settings.get("graph_explanation", {}))
        self.reason_generator = ReasonCodeGenerator(settings.get("reason_codes", {}))
        self.risk_thresholds = dict(
            settings.get(
                "risk_thresholds",
                {"low": 0.2, "medium": 0.5, "high": 0.8},
            )
        )
        self.decision_policy = dict(
            settings.get(
                "decision_policy",
                {"low": "approve", "medium": "review", "high": "review", "critical": "block"},
            )
        )
        self.top_reasons = int(settings.get("top_feature_reasons", 5))
        self.top_related = int(settings.get("top_related_transactions", 10))
        self._validate_thresholds()

    def explain(
        self,
        model: BankingGAAP,
        transaction_id: str,
        *,
        cutoff: Any | None = None,
    ) -> dict[str, Any]:
        """Explain one scored transaction using data available at its timestamp."""
        transaction_graph, adapter = model._require_fitted()
        transaction_id = str(transaction_id)
        if transaction_id not in transaction_graph.transaction_id_to_node:
            raise KeyError(f"Unknown transaction_id: {transaction_id}")
        node_id = transaction_graph.transaction_id_to_node[transaction_id]
        row = transaction_graph.node_table.iloc[node_id]
        timestamp = pd.Timestamp(cutoff if cutoff is not None else row["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        if timestamp < pd.Timestamp(row["timestamp"]):
            raise ValueError("Explanation cutoff cannot precede the transaction timestamp")
        if timestamp > pd.Timestamp(row["timestamp"]):
            timestamp = pd.Timestamp(row["timestamp"])

        graph_result = self.graph_explainer.explain(
            transaction_graph, node_id, cutoff=timestamp
        )
        graph = transaction_graph.dgl_graph
        probability = float(
            adapter.predict_proba_on(graph)[node_id, 1].detach().cpu().item()
        )
        attribution = self.feature_explainer.explain(
            graph,
            node_id,
            transaction_graph.feature_columns,
            adapter.predict_proba_on,
            cutoff=timestamp,
            original_probability=probability,
        )
        feature_reasons = self.reason_generator.generate(
            row, attribution["features"], graph_result
        )
        risk_level = self._risk_level(probability)
        limitations = [
            "Explanation is model-supporting evidence, not proof of fraud.",
            "Feature impacts are local ablation score changes and are not causal effects.",
            "Graph evidence is limited to directed historical connections at or before the transaction time.",
        ]
        if attribution["historical_reference_count"] == 0:
            limitations.append(
                "No strictly earlier graph nodes were available for the feature baseline."
            )
        if not graph_result["related_transactions"]:
            limitations.append(
                "No eligible historical graph neighbors were found for this transaction."
            )
        historical_label_count = sum(
            item["fraud_status"] is not None
            for item in graph_result["related_transactions"]
        )
        if not historical_label_count:
            limitations.append(
                "Historical fraud statuses are hidden unless label availability time is recorded."
            )
        return {
            "transaction_id": transaction_id,
            "timestamp": timestamp.isoformat(),
            "fraud_probability": probability,
            "risk_level": risk_level,
            "decision": self.decision_policy[risk_level],
            "top_feature_reasons": feature_reasons[: self.top_reasons],
            "top_related_transactions": graph_result["related_transactions"][
                : self.top_related
            ],
            "graph_reasons": graph_result["graph_reasons"],
            "model_version": adapter.model_version,
            "explanation_confidence": {
                "level": (
                    "moderate"
                    if feature_reasons or graph_result["related_transactions"]
                    else "limited"
                ),
                "calibrated": False,
                "method": attribution["baseline_method"],
                "limitations": limitations,
                "historical_reference_count": attribution[
                    "historical_reference_count"
                ],
                "evaluated_feature_groups": attribution[
                    "evaluated_feature_groups"
                ],
            },
        }

    def explain_all(self, model: BankingGAAP) -> list[dict[str, Any]]:
        """Generate one explanation for each transaction in the fitted graph."""
        graph, _ = model._require_fitted()
        return [
            self.explain(model, str(transaction_id))
            for transaction_id in graph.node_table["transaction_id"].astype(str)
        ]

    def _risk_level(self, probability: float) -> str:
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Fraud probability must be finite and in [0, 1]")
        if probability >= float(self.risk_thresholds["high"]):
            return "critical"
        if probability >= float(self.risk_thresholds["medium"]):
            return "high"
        if probability >= float(self.risk_thresholds["low"]):
            return "medium"
        return "low"

    def _validate_thresholds(self) -> None:
        required = {"low", "medium", "high"}
        if set(self.risk_thresholds) != required:
            raise ValueError("risk_thresholds must define low, medium, and high")
        values = [float(self.risk_thresholds[name]) for name in ("low", "medium", "high")]
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("Risk thresholds must be in [0, 1]")
        if values != sorted(values):
            raise ValueError("Risk thresholds must be monotonically increasing")
        if not {"low", "medium", "high", "critical"}.issubset(self.decision_policy):
            raise ValueError(
                "decision_policy must define low, medium, high, and critical"
            )
        if self.top_reasons < 1 or self.top_related < 1:
            raise ValueError("Explanation output limits must be positive")
