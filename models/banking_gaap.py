"""End-to-end transaction preprocessing and GAAP-backed fraud predictions."""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import dgl
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from graph.transaction_graph import (
    GraphConfig,
    TransactionGraph,
    TransactionGraphBuilder,
    load_transaction_graph,
)
from models.gaap_adapter import GAAPAdapter
from preprocessing.feature_engineering import TransactionFeatureGenerator
from preprocessing.synthetic_data import generate_synthetic_transactions
from preprocessing.temporal_split import temporal_split
from preprocessing.transaction_loader import load_transactions
from preprocessing.transaction_schema import TransactionSchema


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_FRAUD_LABELS = (1, "1", True, "true", "fraud", "suspicious")
_DEFAULT_LEGITIMATE_LABELS = (0, "0", False, "false", "legitimate", "legit", "genuine")


def _label_key(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip().lower()
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return str(value).strip().lower()


def _resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else (base / resolved).resolve()


class BankingGAAP:
    """Transaction-level wrapper around the original GAAP ``LitSAGE`` model.

    ``fit`` accepts raw banking rows or a CSV path. It uses the existing
    canonicalization and temporal feature pipeline, builds a transaction-node
    DGL graph, creates chronological masks, and delegates optimization to
    :class:`GAAPAdapter`.
    """

    def __init__(self, config: Mapping[str, Any] | str | Path) -> None:
        if isinstance(config, (str, Path)):
            self.config_path = _resolve_path(config)
            with self.config_path.open("r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream)
            if not isinstance(loaded, dict):
                raise ValueError(f"BankingGAAP config must be a mapping: {self.config_path}")
            self.config = loaded
        else:
            self.config_path = None
            self.config = dict(config)
        self._validate_config()
        self.graph: TransactionGraph | None = None
        self.gaap: GAAPAdapter | None = None
        self.split_ids: dict[str, list[str]] = {}
        self._model_config: dict[str, Any] | None = None
        self._model_version = str(self.config.get("model_version", "gaap-banking-v1"))

    def _validate_config(self) -> None:
        required_sections = ("data", "schema_config", "feature_config", "graph_config", "split", "model")
        missing = [name for name in required_sections if name not in self.config]
        if missing:
            raise ValueError(f"BankingGAAP config is missing sections: {missing}")
        for name in required_sections:
            if not isinstance(self.config[name], dict) and name not in (
                "schema_config",
                "feature_config",
                "graph_config",
            ):
                raise ValueError(f"BankingGAAP config section {name!r} must be a mapping")

    def _read_transactions(
        self, transactions: pd.DataFrame | str | Path
    ) -> pd.DataFrame:
        schema_settings = self.config["schema_config"]
        if not isinstance(schema_settings, Mapping):
            schema_settings = {"path": schema_settings}
        schema_path = _resolve_path(schema_settings.get("path", "configs/transaction_schema.yaml"))
        schema = TransactionSchema.from_yaml(schema_path)
        if isinstance(transactions, (str, Path)):
            return load_transactions(transactions, schema=schema)
        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("fit expects a pandas DataFrame or a CSV path")
        return schema.normalize(transactions)

    def _binary_labels(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if "label" not in transactions or transactions["label"].isna().all():
            raise ValueError("Supervised GAAP training requires transaction labels")
        label_settings = self.config.get("labels", {})
        fraud_labels = tuple(label_settings.get("fraud", _DEFAULT_FRAUD_LABELS))
        legitimate_labels = tuple(
            label_settings.get("legitimate", _DEFAULT_LEGITIMATE_LABELS)
        )
        fraud_keys = {_label_key(label) for label in fraud_labels}
        legitimate_keys = {_label_key(label) for label in legitimate_labels}
        if fraud_keys & legitimate_keys:
            raise ValueError("Fraud and legitimate label sets must be disjoint")

        mapped: list[int] = []
        unknown: set[str] = set()
        for value in transactions["label"]:
            if pd.isna(value):
                mapped.append(-1)
                continue
            key = _label_key(value)
            if key in fraud_keys:
                mapped.append(1)
            elif key in legitimate_keys:
                mapped.append(0)
            else:
                unknown.add(str(value))
                mapped.append(-1)
        if unknown:
            raise ValueError(
                "Labels are not included in configured fraud/legitimate classes: "
                f"{sorted(unknown)}"
            )
        result = transactions.copy()
        result["label"] = pd.Series(mapped, index=result.index, dtype="int64").replace(
            -1, pd.NA
        )
        return result

    @staticmethod
    def _mask_for_ids(
        graph: TransactionGraph, transaction_ids: list[str]
    ) -> torch.Tensor:
        mask = torch.zeros(graph.dgl_graph.num_nodes(), dtype=torch.bool)
        for transaction_id in transaction_ids:
            try:
                mask[graph.transaction_id_to_node[transaction_id]] = True
            except KeyError as exc:
                raise KeyError(f"Split refers to unknown transaction {transaction_id!r}") from exc
        return mask

    def fit(
        self,
        transactions: pd.DataFrame | str | Path | None = None,
    ) -> BankingGAAP:
        """Build a temporal transaction graph and train the original GAAP model."""
        if transactions is None:
            transactions = self._configured_data()
        canonical = self._read_transactions(transactions)
        canonical = self._binary_labels(canonical)

        feature_path = _resolve_path(self.config["feature_config"])
        engineered = TransactionFeatureGenerator.from_yaml(feature_path).transform(
            canonical
        )
        split_config = self.config["split"]
        split = temporal_split(
            engineered,
            train_fraction=float(split_config.get("train_fraction", 0.7)),
            validation_fraction=float(split_config.get("validation_fraction", 0.15)),
        )
        self.split_ids = {
            "train": split.train.loc[split.train["label"].notna(), "transaction_id"].astype(str).tolist(),
            "validation": split.validation.loc[
                split.validation["label"].notna(), "transaction_id"
            ].astype(str).tolist(),
            "test": split.test.loc[split.test["label"].notna(), "transaction_id"].astype(str).tolist(),
        }
        if any(not ids for ids in self.split_ids.values()):
            raise ValueError("Temporal train, validation, and test splits must be non-empty")

        graph_settings = self.config["graph_config"]
        if isinstance(graph_settings, Mapping):
            graph_config = GraphConfig.from_mapping(graph_settings)
        else:
            graph_config = GraphConfig.from_yaml(_resolve_path(graph_settings))

        exclude_label_derived = bool(
            self.config.get("exclude_label_derived_features", True)
        )
        feature_names = engineered.attrs.get("feature_columns")
        if not isinstance(feature_names, list):
            raise ValueError("Feature engineering did not report its feature columns")
        if exclude_label_derived:
            engineered.attrs["feature_columns"] = [
                name for name in feature_names if name != "suspicious_neighbour_count"
            ]
        self.graph = TransactionGraphBuilder(graph_config).build(
            engineered, full_history=True
        )
        self.graph.dgl_graph.ndata["label"] = torch.tensor(
            [
                int(value) if not pd.isna(value) else -1
                for value in engineered["label"]
            ],
            dtype=torch.int64,
        )
        self.graph.label_mapping = {"legitimate": 0, "fraud": 1}
        self.graph.stats["fraud_node_ratio"] = float(
            (engineered["label"].dropna() == 1).mean()
        )

        gaap_graph = self.graph.to_gaap_graph(
            self._mask_for_ids(self.graph, self.split_ids["train"]),
            self._mask_for_ids(self.graph, self.split_ids["validation"]),
            self._mask_for_ids(self.graph, self.split_ids["test"]),
        )
        gaap_graph.ndata[dgl.NID] = torch.arange(
            gaap_graph.num_nodes(), dtype=torch.int64
        )

        self._model_config = dict(self.config["model"])
        self._model_config["seed"] = int(self.config.get("seed", 42))
        output_dir = _resolve_path(
            self.config.get("output_dir", "artifacts/adaptive_gaap")
        )
        self.gaap = GAAPAdapter(
            self._model_config,
            seed=self._model_config["seed"],
            output_dir=output_dir,
            max_epochs=int(self.config.get("training", {}).get("max_epochs", 3)),
            patience=int(self.config.get("training", {}).get("patience", 2)),
            threshold=float(self.config.get("threshold", 0.5)),
            model_version=self._model_version,
        )
        self.gaap.fit(gaap_graph)
        LOGGER.info(
            "Trained GAAP on %d transaction nodes using temporal split sizes %s",
            gaap_graph.num_nodes(),
            {name: len(values) for name, values in self.split_ids.items()},
        )
        return self

    def _configured_data(self) -> pd.DataFrame | Path:
        data_config = self.config["data"]
        source = data_config.get("source", "synthetic")
        if source == "synthetic":
            configured_output = data_config.get("synthetic_csv")
            output = (
                _resolve_path(configured_output)
                if configured_output
                else _resolve_path(self.config.get("output_dir", "artifacts/adaptive_gaap"))
                / "synthetic_transactions.csv"
            )
            return generate_synthetic_transactions(
                row_count=int(data_config.get("synthetic_rows", 132)),
                seed=int(self.config.get("seed", 42)),
                output_path=output,
            )
        if source == "csv":
            csv_path = data_config.get("csv_path")
            if not csv_path:
                raise ValueError("data.csv_path is required when data.source is 'csv'")
            return _resolve_path(csv_path)
        raise ValueError("data.source must be 'synthetic' or 'csv'")

    def _require_fitted(self) -> tuple[TransactionGraph, GAAPAdapter]:
        if self.graph is None or self.gaap is None:
            raise RuntimeError("BankingGAAP must be fitted or loaded before inference")
        return self.graph, self.gaap

    def predict_proba(self) -> pd.DataFrame:
        """Return transaction IDs, timestamps, and GAAP fraud probabilities."""
        graph, adapter = self._require_fitted()
        probabilities = adapter.predict_proba()[:, 1].cpu().numpy()
        if len(probabilities) != len(graph.node_table):
            raise RuntimeError("GAAP prediction count does not match transaction nodes")
        return pd.DataFrame(
            {
                "transaction_id": graph.node_table["transaction_id"].astype(str),
                "timestamp": pd.to_datetime(graph.node_table["timestamp"], utc=True),
                "fraud_probability": probabilities,
            }
        )

    def predict(self) -> pd.DataFrame:
        """Return transaction-level predictions and available labels."""
        graph, adapter = self._require_fitted()
        predictions = self.predict_proba()
        predictions["predicted_label"] = (
            predictions["fraud_probability"] >= adapter.threshold
        ).astype("int64")
        if "label" in graph.node_table:
            labels = graph.node_table["label"]
            predictions["true_label"] = pd.array(labels, dtype="Int64")
        predictions["model_version"] = adapter.model_version
        return predictions

    def evaluate(self, split: str = "test") -> dict[str, float]:
        """Evaluate GAAP probabilities on one time-based split."""
        graph, _ = self._require_fitted()
        if split not in self.split_ids:
            raise ValueError("split must be one of train, validation, or test")
        mask = graph.node_table["transaction_id"].astype(str).isin(
            self.split_ids[split]
        )
        labels = graph.node_table.loc[mask, "label"].astype(int).to_numpy()
        predictions = self.predict_proba().loc[mask, "fraud_probability"].to_numpy()
        if set(labels.tolist()) != {0, 1}:
            raise ValueError(f"{split} split must contain both binary classes")
        predicted_labels = (predictions >= self.gaap.threshold).astype("int64")  # type: ignore[union-attr]
        return {
            "precision": float(
                precision_score(labels, predicted_labels, zero_division=0)
            ),
            "recall": float(recall_score(labels, predicted_labels, zero_division=0)),
            "f1": float(f1_score(labels, predicted_labels, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, predictions)),
            "average_precision": float(average_precision_score(labels, predictions)),
            "accuracy": float(accuracy_score(labels, predicted_labels)),
        }

    def save(self, directory: str | Path) -> Path:
        """Save the DGL transaction graph, GAAP weights, splits, and config."""
        graph, adapter = self._require_fitted()
        output_dir = _resolve_path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        graph.save(output_dir / "graph")
        checkpoint_path = adapter.save_checkpoint(output_dir / "gaap_state.ckpt")
        metadata = {
            "format_version": 1,
            "checkpoint": checkpoint_path.name,
            "config": self.config,
            "split_ids": self.split_ids,
            "model_version": self._model_version,
        }
        (output_dir / "banking_gaap.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return output_dir

    @classmethod
    def load(cls, directory: str | Path) -> BankingGAAP:
        """Restore a saved BankingGAAP model and its transaction graph."""
        input_dir = _resolve_path(directory)
        metadata_path = input_dir / "banking_gaap.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"BankingGAAP metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        instance = cls(metadata["config"])
        instance.split_ids = {
            str(name): [str(value) for value in values]
            for name, values in metadata["split_ids"].items()
        }
        instance._model_version = str(metadata["model_version"])
        instance.graph = load_transaction_graph(input_dir / "graph")
        graph = instance.graph
        gaap_graph = graph.to_gaap_graph(
            instance._mask_for_ids(graph, instance.split_ids["train"]),
            instance._mask_for_ids(graph, instance.split_ids["validation"]),
            instance._mask_for_ids(graph, instance.split_ids["test"]),
        )
        gaap_graph.ndata[dgl.NID] = torch.arange(
            gaap_graph.num_nodes(), dtype=torch.int64
        )
        instance._model_config = dict(instance.config["model"])
        instance._model_config["seed"] = int(instance.config.get("seed", 42))
        instance.gaap = GAAPAdapter(
            instance._model_config,
            seed=instance._model_config["seed"],
            output_dir=input_dir,
            max_epochs=int(instance.config.get("training", {}).get("max_epochs", 3)),
            patience=int(instance.config.get("training", {}).get("patience", 2)),
            threshold=float(instance.config.get("threshold", 0.5)),
            model_version=instance._model_version,
        )
        instance.gaap.load_checkpoint(gaap_graph, input_dir / metadata["checkpoint"])
        return instance
