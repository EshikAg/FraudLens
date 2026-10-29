"""Online feature, graph, GAAP score, drift, and explanation orchestration."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, deque
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import pandas as pd
import torch
import yaml

from api.services.drift_service import DriftService
from api.services.explanation_service import ExplanationService
from api.services.transaction_service import (
    InvalidTransactionError,
    TransactionConflictError,
    TransactionService,
)
from graph.transaction_graph import GraphConfig, TransactionGraphBuilder
from models.banking_gaap import BankingGAAP, _resolve_path
from preprocessing.feature_engineering import TransactionFeatureGenerator


LOGGER = logging.getLogger(__name__)


class PredictionExecutionError(RuntimeError):
    """A scoring stage failed; no prediction was produced."""


class PredictionService:
    """Thread-safe inference service backed by one already-loaded GAAP model."""

    def __init__(
        self,
        model: BankingGAAP,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        model._require_fitted()
        self.model = model
        self.config = dict(config or {})
        thresholds = dict(
            self.config.get(
                "risk_thresholds",
                {"low": 0.2, "medium": 0.5, "high": 0.8},
            )
        )
        decisions = dict(
            self.config.get(
                "decisions",
                {
                    "low": "APPROVE",
                    "medium": "STEP_UP_AUTHENTICATION",
                    "high": "REVIEW",
                    "critical": "BLOCK",
                },
            )
        )
        self._validate_risk_policy(thresholds, decisions)
        self.risk_thresholds = thresholds
        self.decisions = decisions
        self._lock = RLock()
        self._metrics_lock = RLock()
        self._metrics: Counter[str] = Counter()
        metrics_window_size = int(self.config.get("metrics_window_size", 1000))
        if metrics_window_size < 1:
            raise ValueError("metrics_window_size must be positive")
        self._latencies: deque[float] = deque(
            maxlen=metrics_window_size
        )
        self.transaction_service = TransactionService(
            model,
            max_history_transactions=int(
                self.config.get("max_history_transactions", 10000)
            ),
            schema_path=self.config.get(
                "schema_config", "configs/transaction_schema.yaml"
            ),
        )
        graph_config = model.config["graph_config"]
        if isinstance(graph_config, Mapping):
            self.graph_config = GraphConfig.from_mapping(graph_config)
        else:
            self.graph_config = GraphConfig.from_yaml(_resolve_path(graph_config))
        feature_config = model.config["feature_config"]
        self.feature_generator = TransactionFeatureGenerator.from_yaml(
            _resolve_path(feature_config)
        )
        self.graph_builder = TransactionGraphBuilder(self.graph_config)
        self.explanation_service = ExplanationService(
            self.config.get(
                "explainability_config", "configs/explainability_config.yaml"
            )
        )
        model_graph, _ = model._require_fitted()
        self._evaluation_metrics: dict[str, float] | None = None
        self._metrics_unavailable_reason: str | None = None
        try:
            self._evaluation_metrics = model.evaluate("test")
        except ValueError as exc:
            self._metrics_unavailable_reason = str(exc)
        self.drift_service = DriftService(
            self.config.get("drift_config", {}),
            model_graph.node_table,
            window_size=int(self.config.get("drift_window_size", 500)),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PredictionService:
        """Load the saved BankingGAAP wrapper once from configured storage."""
        model_directory = config.get("model_directory")
        if not model_directory:
            raise ValueError("api.model_directory must point to a saved BankingGAAP model")
        path = _resolve_path(model_directory)
        if not path.is_dir():
            raise FileNotFoundError(f"Saved BankingGAAP model directory not found: {path}")
        return cls(BankingGAAP.load(path), config)

    @staticmethod
    def _validate_risk_policy(
        thresholds: Mapping[str, Any], decisions: Mapping[str, Any]
    ) -> None:
        if set(thresholds) != {"low", "medium", "high"}:
            raise ValueError("risk_thresholds must define low, medium, and high")
        values = [float(thresholds[name]) for name in ("low", "medium", "high")]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Risk thresholds must be finite values in [0, 1]")
        if values != sorted(values):
            raise ValueError("Risk thresholds must be monotonically increasing")
        if not {"low", "medium", "high", "critical"}.issubset(decisions):
            raise ValueError("decisions must define low, medium, high, and critical")
        if any(not isinstance(decisions[key], str) or not decisions[key].strip() for key in decisions):
            raise ValueError("Each decision must be a non-empty string")

    @classmethod
    def load_config(cls, path: str | Path) -> dict[str, Any]:
        config_path = _resolve_path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            settings = yaml.safe_load(stream)
        if not isinstance(settings, dict):
            raise ValueError(f"API config must be a YAML mapping: {config_path}")
        return settings

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Score one transaction using only strictly earlier transaction context."""
        started = time.perf_counter()
        try:
            with self._lock:
                history, current_row = self.transaction_service.prepare(payload)
                transaction_id = str(current_row["transaction_id"])
                rows = pd.concat(
                    [history, pd.DataFrame([current_row])], ignore_index=True
                )
                rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
                rows = rows.sort_values("timestamp", kind="stable").reset_index(drop=True)
                engineered = self.feature_generator.transform(rows)

                feature_names = engineered.attrs.get("feature_columns")
                if not isinstance(feature_names, list):
                    raise RuntimeError("Feature generator did not report its feature columns")
                if self.model.config.get("exclude_label_derived_features", True):
                    engineered.attrs["feature_columns"] = [
                        name for name in feature_names
                        if name != "suspicious_neighbour_count"
                    ]

                inference_graph = self.graph_builder.build(engineered)
                self._align_feature_matrix(inference_graph)
                node_id = inference_graph.transaction_id_to_node.get(transaction_id)
                if node_id is None:
                    raise RuntimeError("Inference graph omitted the submitted transaction")
                _, adapter = self.model._require_fitted()
                probabilities = adapter.predict_proba_on(inference_graph.dgl_graph)
                probability = float(probabilities[node_id, 1].detach().cpu().item())
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise RuntimeError("GAAP returned an invalid fraud probability")

                risk_level = self._risk_level(probability)
                explanation = self.explanation_service.explain(
                    self.model, inference_graph, transaction_id
                )
                drift = self.drift_service.observe(
                    engineered.iloc[node_id].to_dict(),
                    transaction_id=transaction_id,
                    probability=probability,
                    timestamp=current_row["timestamp"],
                )
                self.transaction_service.commit(current_row)
                reasons = [
                    *explanation["top_feature_reasons"],
                    *explanation["graph_reasons"],
                ]
                response = {
                    "transaction_id": transaction_id,
                    "fraud_probability": probability,
                    "risk_level": risk_level.upper(),
                    "decision": self.decisions[risk_level],
                    "drift_detected": bool(drift["drift_detected"]),
                    "model_version": adapter.model_version,
                    "reasons": reasons,
                    "related_transactions": explanation[
                        "top_related_transactions"
                    ],
                    "timestamp": pd.Timestamp(current_row["timestamp"]).isoformat(),
                }
                with self._metrics_lock:
                    self._metrics["predictions_succeeded"] += 1
                LOGGER.info(
                    "transaction_scored",
                    extra={
                        "event": "transaction_scored",
                        "risk_level": risk_level.upper(),
                        "model_version": adapter.model_version,
                        "drift_detected": bool(drift["drift_detected"]),
                        "latency_ms": (time.perf_counter() - started) * 1000,
                    },
                )
                return response
        except (InvalidTransactionError, TransactionConflictError) as exc:
            with self._metrics_lock:
                self._metrics["prediction_errors"] += 1
            LOGGER.warning(
                "transaction_request_rejected",
                extra={
                    "event": "transaction_request_rejected",
                    "error_type": type(exc).__name__,
                },
            )
            raise
        except Exception as exc:
            with self._metrics_lock:
                self._metrics["prediction_errors"] += 1
            LOGGER.error(
                "transaction_scoring_failed",
                extra={
                    "event": "transaction_scoring_failed",
                    "error_type": type(exc).__name__,
                },
            )
            raise PredictionExecutionError(
                "Model scoring failed; no prediction was returned"
            ) from exc
        finally:
            with self._metrics_lock:
                self._latencies.append((time.perf_counter() - started) * 1000)

    def _align_feature_matrix(self, inference_graph: Any) -> None:
        """Keep trained feature semantics and width when online categories are new."""
        fitted_graph, _ = self.model._require_fitted()
        expected_columns = list(fitted_graph.feature_columns)
        actual_columns = list(inference_graph.feature_columns)
        encoded = inference_graph.dgl_graph.ndata["feature"]
        aligned = torch.zeros(
            (encoded.shape[0], len(expected_columns)),
            dtype=encoded.dtype,
            device=encoded.device,
        )
        current_positions = {
            name: index for index, name in enumerate(actual_columns)
        }
        for destination, name in enumerate(expected_columns):
            source = current_positions.get(name)
            if source is not None:
                aligned[:, destination] = encoded[:, source]
        expected_width = int(self.model.gaap.data_module.d_in)  # type: ignore[union-attr]
        if aligned.shape[1] != expected_width:
            raise RuntimeError(
                f"Aligned inference width {aligned.shape[1]} does not match "
                f"trained GAAP width {expected_width}"
            )
        inference_graph.dgl_graph.ndata["feature"] = aligned
        inference_graph.feature_columns = tuple(expected_columns)

    def _risk_level(self, probability: float) -> str:
        if probability >= float(self.risk_thresholds["high"]):
            return "critical"
        if probability >= float(self.risk_thresholds["medium"]):
            return "high"
        if probability >= float(self.risk_thresholds["low"]):
            return "medium"
        return "low"

    def model_information(self) -> dict[str, Any]:
        graph, adapter = self.model._require_fitted()
        return {
            "model_loaded": True,
            "model_version": adapter.model_version,
            "model": self.model.config["model"].get("model_name", "GAAP"),
            "feature_count": int(graph.dgl_graph.ndata["feature"].shape[1]),
            "threshold": float(adapter.threshold),
            "risk_thresholds": {
                name: float(value) for name, value in self.risk_thresholds.items()
            },
            "decisions": dict(self.decisions),
            "evaluation_split": "test",
            "metrics": self._evaluation_metrics,
            "metrics_unavailable_reason": self._metrics_unavailable_reason,
        }

    def drift_status(self) -> dict[str, Any]:
        return self.drift_service.status()

    def update_label(
        self, transaction_id: str, label: int, timestamp: Any | None = None
    ) -> dict[str, Any]:
        """Pass a delayed ground-truth label to concept-drift monitoring."""
        return self.drift_service.update_label(
            transaction_id, label, timestamp=timestamp
        )

    def metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            latencies = list(self._latencies)
            return {
                "predictions_succeeded": self._metrics["predictions_succeeded"],
                "prediction_errors": self._metrics["prediction_errors"],
                "latency_ms": {
                    "count": len(latencies),
                    "mean": sum(latencies) / len(latencies) if latencies else 0.0,
                    "max": max(latencies, default=0.0),
                },
            }
