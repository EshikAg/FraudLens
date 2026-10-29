"""Warm-start the existing GAAP model on temporally weighted replay mixtures."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from graph.transaction_graph import GraphConfig, TransactionGraphBuilder
from models.banking_gaap import BankingGAAP, _resolve_path
from models.gaap_adapter import GAAPAdapter
from preprocessing.feature_engineering import TransactionFeatureGenerator


LOGGER = logging.getLogger(__name__)


class IncrementalTrainer:
    """Fine-tune a shape-compatible checkpoint using the original GAAP trainer."""

    def __init__(
        self,
        gaap_config: Mapping[str, Any] | str | Path,
        training_config: Mapping[str, Any],
        *,
        seed: int = 42,
    ) -> None:
        self.base_model = BankingGAAP(gaap_config)
        self.training_config = dict(training_config)
        self.seed = int(seed)

    def train(
        self,
        *,
        current_model: BankingGAAP,
        baseline_model: BankingGAAP,
        recent_training: pd.DataFrame,
        replay_training: pd.DataFrame,
        validation: pd.DataFrame,
        recent_window: pd.DataFrame,
        replay_window: pd.DataFrame,
        initial_checkpoint: str | Path,
        output_dir: str | Path,
        model_version: str,
    ) -> dict[str, Any]:
        """Train and score a candidate without mutating the current model."""
        if current_model.gaap is None:
            raise RuntimeError("Current BankingGAAP must be fitted")
        groups = {
            "recent_training": recent_training,
            "replay_training": replay_training,
            "validation": validation,
            "recent": recent_window,
            "replay": replay_window,
        }
        for name, frame in groups.items():
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise ValueError(f"Adaptation requires a non-empty {name} window")
        for name, frame in groups.items():
            if "label" not in frame or frame["label"].isna().any():
                raise ValueError(f"Adaptation {name} rows must all have labels")

        ids_by_group = {
            name: set(frame["transaction_id"].astype(str))
            for name, frame in groups.items()
        }
        for left_index, left in enumerate(groups):
            for right in list(groups)[left_index + 1 :]:
                overlap = ids_by_group[left] & ids_by_group[right]
                if overlap:
                    raise ValueError(
                        f"Adaptation groups {left!r} and {right!r} overlap: "
                        f"{sorted(overlap)[:5]}"
                    )

        train_ids = ids_by_group["recent_training"] | ids_by_group["replay_training"]
        frame = pd.concat(list(groups.values()), ignore_index=True, sort=False)
        if frame["transaction_id"].astype(str).duplicated().any():
            raise ValueError("Adaptation windows contain duplicate transaction IDs")
        transaction_graph = self.build_transaction_graph(current_model, frame)

        ids = transaction_graph.node_table["transaction_id"].astype(str)

        def mask_for(selected: set[str]) -> torch.Tensor:
            return torch.tensor(ids.isin(selected).to_numpy(), dtype=torch.bool)

        train_mask = mask_for(train_ids)
        validation_mask = mask_for(ids_by_group["validation"])
        test_mask = mask_for(ids_by_group["replay"])
        graph = transaction_graph.to_gaap_graph(
            train_mask, validation_mask, test_mask
        )
        graph.ndata[dgl.NID] = torch.arange(graph.num_nodes(), dtype=torch.int64)
        output_path = Path(output_dir).resolve()
        current_adapter = current_model.gaap
        if baseline_model.gaap is None:
            raise RuntimeError("Fixed baseline BankingGAAP must be fitted")
        adapter = GAAPAdapter(
            current_model._model_config or current_model.config["model"],
            seed=self.seed,
            output_dir=output_path,
            max_epochs=int(self.training_config.get("max_epochs", 2)),
            patience=int(self.training_config.get("patience", 1)),
            threshold=current_adapter.threshold,
            model_version=model_version,
        )
        adapter.fit(graph, initial_checkpoint=initial_checkpoint)
        checkpoint_path = adapter.save_checkpoint(output_path / "gaap_state.ckpt")

        group_ids = {
            "validation": ids_by_group["validation"],
            "recent": ids_by_group["recent"],
            "replay": ids_by_group["replay"],
        }
        graph_ids = transaction_graph.node_table["transaction_id"].astype(str).tolist()
        candidate_metrics = {
            name: self._metrics(graph, adapter, selected_ids, graph_ids)
            for name, selected_ids in group_ids.items()
        }
        current_metrics = {
            name: self._metrics(graph, current_adapter, selected_ids, graph_ids)
            for name, selected_ids in group_ids.items()
        }
        baseline_metrics = {
            name: self._metrics(
                graph, baseline_model.gaap, selected_ids, graph_ids
            )
            for name, selected_ids in group_ids.items()
        }
        candidate_model = BankingGAAP(current_model.config)
        candidate_model.graph = transaction_graph
        candidate_model.gaap = adapter
        candidate_model.split_ids = {
            "train": sorted(train_ids),
            "validation": sorted(ids_by_group["validation"]),
            "test": sorted(ids_by_group["replay"]),
        }
        candidate_model._model_config = dict(adapter.model_config)
        candidate_model._model_version = model_version
        report = {
            "candidate_adapter": adapter,
            "candidate_model": candidate_model,
            "transaction_graph": transaction_graph,
            "graph": graph,
            "checkpoint_path": checkpoint_path,
            "candidate_metrics": candidate_metrics,
            "current_metrics": current_metrics,
            "baseline_metrics": baseline_metrics,
            "training_data_window": {
                "rows": int(len(train_ids)),
                "start": str(
                    pd.to_datetime(frame.loc[
                        frame["transaction_id"].astype(str).isin(train_ids), "timestamp"
                    ], utc=True).min()
                ),
                "end": str(
                    pd.to_datetime(frame.loc[
                        frame["transaction_id"].astype(str).isin(train_ids), "timestamp"
                    ], utc=True).max()
                ),
                "recent_rows": int(len(recent_training)),
                "replay_rows": int(len(replay_training)),
            },
            "feature_alignment": dict(
                transaction_graph.node_table.attrs.get("feature_alignment", {})
            ),
        }
        LOGGER.info(
            "Fine-tuned GAAP candidate %s on %d recent and %d replay transactions",
            model_version,
            len(recent_training),
            len(replay_training),
        )
        return report

    def build_transaction_graph(
        self,
        reference_model: BankingGAAP,
        transactions: pd.DataFrame,
    ) -> Any:
        """Build a causally ordered graph with the reference feature schema."""
        if reference_model.graph is None:
            raise RuntimeError("Reference BankingGAAP has no transaction graph")
        canonical = self.base_model._read_transactions(transactions)
        if "label" in canonical and canonical["label"].notna().any():
            canonical = self.base_model._binary_labels(canonical)
        else:
            canonical["label"] = pd.NA
        canonical = canonical.sort_values(
            ["timestamp", "transaction_id"], kind="stable"
        ).reset_index(drop=True)
        engineered = TransactionFeatureGenerator.from_yaml(
            _resolve_path(self.base_model.config["feature_config"])
        ).transform(canonical)
        feature_names = engineered.attrs.get("feature_columns")
        if not isinstance(feature_names, list):
            raise ValueError("Feature engineering did not report feature columns")
        if self.base_model.config.get("exclude_label_derived_features", True):
            engineered.attrs["feature_columns"] = [
                name for name in feature_names if name != "suspicious_neighbour_count"
            ]
        graph_settings = self.base_model.config["graph_config"]
        graph_config = (
            GraphConfig.from_mapping(graph_settings)
            if isinstance(graph_settings, Mapping)
            else GraphConfig.from_yaml(_resolve_path(graph_settings))
        )
        transaction_graph = TransactionGraphBuilder(graph_config).build(
            engineered, full_history=True
        )
        expected_features = tuple(reference_model.graph.feature_columns)
        observed_columns = {
            name: index
            for index, name in enumerate(transaction_graph.feature_columns)
        }
        unmapped_columns = sorted(set(observed_columns) - set(expected_features))
        missing_columns = sorted(set(expected_features) - set(observed_columns))
        alignment_report = {
            "model_feature_width": len(expected_features),
            "observed_feature_width": len(observed_columns),
            "unmapped_feature_columns": unmapped_columns,
            "missing_expected_feature_columns": missing_columns,
        }
        transaction_graph.node_table.attrs["feature_alignment"] = alignment_report
        if unmapped_columns:
            LOGGER.warning(
                "Reference feature schema cannot represent new graph columns: %s",
                unmapped_columns,
            )
        aligned_features = np.zeros(
            (transaction_graph.dgl_graph.num_nodes(), len(expected_features)),
            dtype=np.float32,
        )
        observed_tensor = transaction_graph.dgl_graph.ndata["feature"]
        for target_index, name in enumerate(expected_features):
            source_index = observed_columns.get(name)
            if source_index is not None:
                aligned_features[:, target_index] = (
                    observed_tensor[:, source_index].cpu().numpy()
                )
        transaction_graph.dgl_graph.ndata["feature"] = torch.from_numpy(
            aligned_features
        )
        transaction_graph.feature_columns = expected_features
        transaction_graph.dgl_graph.ndata["label"] = torch.tensor(
            [
                int(value) if not pd.isna(value) else -1
                for value in engineered["label"]
            ],
            dtype=torch.int64,
        )
        transaction_graph.label_mapping = {"legitimate": 0, "fraud": 1}
        return transaction_graph

    def predict_probabilities(
        self,
        model: BankingGAAP,
        transactions: pd.DataFrame,
        *,
        context_transactions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Score unseen rows with a model using its fixed encoded feature schema."""
        if model.gaap is None:
            raise RuntimeError("BankingGAAP must be fitted before scoring")
        combined = (
            pd.concat([context_transactions, transactions], ignore_index=True)
            if context_transactions is not None
            else transactions.copy()
        )
        if combined["transaction_id"].astype(str).duplicated().any():
            raise ValueError("Scoring context and target IDs must be disjoint")
        transaction_graph = self.build_transaction_graph(model, combined)
        graph = transaction_graph.dgl_graph
        probabilities = model.gaap.predict_proba_on(graph)[:, 1].cpu().numpy()
        graph_ids = transaction_graph.node_table["transaction_id"].astype(str)
        target_ids = set(transactions["transaction_id"].astype(str))
        selected = graph_ids.isin(target_ids).to_numpy()
        if int(selected.sum()) != len(transactions):
            raise RuntimeError("Scored graph did not retain all target transactions")
        result = transaction_graph.node_table.loc[
            selected, ["transaction_id", "timestamp", "label"]
        ].copy()
        result["fraud_probability"] = probabilities[selected]
        result["predicted_label"] = (
            result["fraud_probability"] >= model.gaap.threshold
        ).astype("int64")
        result["model_version"] = model.gaap.model_version
        return result.reset_index(drop=True)

    @staticmethod
    def _metrics(
        graph: Any,
        adapter: GAAPAdapter,
        selected_ids: set[str],
        transaction_ids: list[str],
    ) -> dict[str, float]:
        probabilities = adapter.predict_proba_on(graph)[:, 1].cpu().numpy()
        labels = graph.ndata["label"].cpu().numpy()
        selected_nodes = np.array(
            [
                index
                for index, transaction_id in enumerate(transaction_ids)
                if transaction_id in selected_ids
            ],
            dtype=np.int64,
        )
        if len(selected_nodes) == 0:
            raise ValueError("Cannot evaluate an empty adaptation window")
        selected_labels = labels[selected_nodes]
        selected_probabilities = probabilities[selected_nodes]
        if set(selected_labels.tolist()) != {0, 1}:
            raise ValueError("Each adaptation evaluation window must contain both classes")
        predictions = (selected_probabilities >= adapter.threshold).astype("int64")
        return {
            "accuracy": float(accuracy_score(selected_labels, predictions)),
            "precision": float(precision_score(selected_labels, predictions, zero_division=0)),
            "recall": float(recall_score(selected_labels, predictions, zero_division=0)),
            "f1": float(f1_score(selected_labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(selected_labels, selected_probabilities)),
            "average_precision": float(
                average_precision_score(selected_labels, selected_probabilities)
            ),
        }
