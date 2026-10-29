"""Build and persist transaction-as-node DGL graphs."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from graph.edge_builder import EdgeBuilder, EdgeRule
from graph.graph_features import GraphFeatureEncoder
from preprocessing.transaction_loader import load_transactions
from preprocessing.transaction_schema import TransactionSchema
from preprocessing.validation import require_temporal_order


LOGGER = logging.getLogger(__name__)


def _dgl_torch() -> tuple[Any, Any]:
    try:
        import dgl
        import torch
    except ImportError as exc:
        raise ImportError(
            "DGL and PyTorch are required to build or load transaction graphs"
        ) from exc
    return dgl, torch


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def _label_key(value: Any) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip().lower()


def _is_fraud_label(value: Any, fraud_labels: Sequence[Any]) -> bool:
    if pd.isna(value):
        return False
    return _label_key(value) in {_label_key(label) for label in fraud_labels}


@dataclass(frozen=True)
class GraphConfig:
    """Settings for relationship construction and numeric node features."""

    max_history_hours: float | None = 168.0
    default_snapshot_window: str | float = "24h"
    edge_rules: tuple[EdgeRule, ...] = EdgeBuilder.default_rules()
    feature_columns: tuple[str, ...] | None = None
    categorical_columns: tuple[str, ...] = ("transaction_type", "channel")
    fraud_labels: tuple[Any, ...] = (1, "fraud", "suspicious", True)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> GraphConfig:
        history = config.get("max_history_hours", 168.0)
        if history is not None:
            history = float(history)
            if not math.isfinite(history) or history <= 0:
                raise ValueError("max_history_hours must be positive or null")
        snapshot_window = config.get("default_snapshot_window", "24h")
        if not isinstance(snapshot_window, (str, int, float)):
            raise ValueError("default_snapshot_window must be a duration string or hours")
        snapshot_delta = (
            pd.Timedelta(hours=float(snapshot_window))
            if isinstance(snapshot_window, (int, float))
            else pd.Timedelta(snapshot_window)
        )
        if pd.isna(snapshot_delta) or snapshot_delta <= pd.Timedelta(0):
            raise ValueError("default_snapshot_window must be positive")
        raw_rules = config.get("edge_rules")
        if raw_rules is not None and (
            not isinstance(raw_rules, (list, tuple))
            or not all(isinstance(rule, Mapping) for rule in raw_rules)
        ):
            raise ValueError("edge_rules must be a list of rule mappings")
        rules = (
            EdgeBuilder.default_rules()
            if raw_rules is None
            else tuple(EdgeRule.from_mapping(rule) for rule in raw_rules)
        )
        columns = config.get("feature_columns")
        categorical = config.get("categorical_columns", ("transaction_type", "channel"))
        fraud_labels = config.get("fraud_labels", (1, "fraud", "suspicious", True))
        if columns is not None and (
            not isinstance(columns, (list, tuple))
            or not all(isinstance(name, str) for name in columns)
        ):
            raise ValueError("feature_columns must be a list of strings or null")
        if not isinstance(categorical, (list, tuple)) or not all(
            isinstance(name, str) for name in categorical
        ):
            raise ValueError("categorical_columns must be a list of strings")
        if not isinstance(fraud_labels, (list, tuple)):
            raise ValueError("fraud_labels must be a list")
        return cls(
            max_history_hours=history,
            default_snapshot_window=snapshot_window,
            edge_rules=rules,
            feature_columns=tuple(columns) if columns is not None else None,
            categorical_columns=tuple(categorical),
            fraud_labels=tuple(fraud_labels),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> GraphConfig:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError(f"Graph config must be a YAML mapping: {config_path}")
        return cls.from_mapping(config)


@dataclass
class TransactionGraph:
    """A DGL graph plus transaction IDs and interpretable tabular metadata."""

    dgl_graph: Any
    node_table: pd.DataFrame
    edge_table: pd.DataFrame
    transaction_id_to_node: dict[str, int]
    feature_columns: tuple[str, ...]
    relation_mapping: dict[str, int]
    categorical_mappings: dict[str, dict[str, int]]
    label_mapping: dict[str, int]
    stats: dict[str, Any]
    fraud_labels: tuple[Any, ...]
    edge_history_hours: float | None
    default_snapshot_window: str | float
    feature_availability: dict[str, bool]
    source_field_availability: dict[str, Any]

    @property
    def graph(self) -> Any:
        """Alias for adapters expecting a DGL graph attribute."""
        return self.dgl_graph

    def to_gaap_graph(
        self,
        train_mask: Sequence[bool],
        validation_mask: Sequence[bool],
        test_mask: Sequence[bool],
    ) -> Any:
        """Return a cloned graph with the split keys expected by GAAP loaders."""
        _, torch = _dgl_torch()
        if "label" not in self.dgl_graph.ndata:
            raise ValueError("GAAP training requires transaction labels")
        masks = {
            "train_mask": train_mask,
            "val_mask": validation_mask,
            "test_mask": test_mask,
        }
        tensors = {
            name: torch.as_tensor(values, dtype=torch.bool)
            for name, values in masks.items()
        }
        expected = self.dgl_graph.num_nodes()
        if any(
            mask.ndim != 1 or mask.numel() != expected
            for mask in tensors.values()
        ):
            raise ValueError(f"Each split mask must contain exactly {expected} values")
        combined = sum(mask.to(torch.int8) for mask in tensors.values())
        if (combined > 1).any():
            raise ValueError("GAAP train/validation/test masks must be disjoint")
        adapted = self.dgl_graph.clone()
        for name, mask in tensors.items():
            adapted.ndata[name] = mask
        return adapted

    def save(self, directory: str | Path) -> Path:
        """Save the DGL graph and metadata needed to recover transaction IDs."""
        dgl, _ = _dgl_torch()
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        graph_path = output_dir / "transactions.bin"
        dgl.save_graphs(str(graph_path), [self.dgl_graph])

        node_metadata_path = output_dir / "nodes.json"
        node_records = [
            {str(key): _json_value(value) for key, value in record.items()}
            for record in self.node_table.to_dict(orient="records")
        ]
        node_metadata_path.write_text(
            json.dumps(node_records, indent=2, allow_nan=False), encoding="utf-8"
        )
        metadata = {
            "format_version": 1,
            "graph_file": graph_path.name,
            "node_metadata_file": node_metadata_path.name,
            "feature_columns": list(self.feature_columns),
            "relation_mapping": self.relation_mapping,
            "categorical_mappings": self.categorical_mappings,
            "label_mapping": self.label_mapping,
            "statistics": self.stats,
            "fraud_labels": list(self.fraud_labels),
            "edge_history_hours": self.edge_history_hours,
            "default_snapshot_window": self.default_snapshot_window,
            "feature_availability": self.feature_availability,
            "source_field_availability": self.source_field_availability,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return graph_path


class TransactionGraphBuilder:
    """Convert canonical/engineered transaction rows into a DGL graph.

    Each transaction is exactly one node. Only earlier transactions can be
    connected to later ones. Labels are stored for supervision but are never
    included in node features or used to determine edges.
    """

    def __init__(self, config: GraphConfig | Mapping[str, Any] | None = None):
        if config is None:
            self.config = GraphConfig()
        elif isinstance(config, GraphConfig):
            self.config = config
        else:
            self.config = GraphConfig.from_mapping(config)
        self.edge_builder = EdgeBuilder(
            self.config.edge_rules,
            max_history_hours=self.config.max_history_hours,
        )
        self.feature_encoder = GraphFeatureEncoder(
            feature_columns=self.config.feature_columns,
            categorical_columns=self.config.categorical_columns,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TransactionGraphBuilder:
        return cls(GraphConfig.from_yaml(path))

    def build(
        self,
        transactions: pd.DataFrame,
        *,
        full_history: bool = False,
    ) -> TransactionGraph:
        """Build a causal DGL graph and retain node/edge metadata."""
        require_temporal_order(transactions)
        source = transactions.reset_index(drop=True).copy()
        if "transaction_id" not in source:
            raise ValueError("Transactions require a transaction_id column")
        if source["transaction_id"].isna().any():
            raise ValueError("Every graph node must have a transaction_id")
        ids = source["transaction_id"].astype(str)
        if ids.duplicated().any():
            duplicates = ids[ids.duplicated()].head(5).tolist()
            raise ValueError(f"transaction_id values must be unique: {duplicates}")
        source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True)

        features = self.feature_encoder.transform(source)
        if features.values.shape[1] == 0:
            raise ValueError("At least one node feature must be available")
        edges = self.edge_builder.build(source, full_history=full_history)
        relation_names = sorted({rule.name for rule in self.config.edge_rules if rule.enabled})
        relation_mapping = {name: index for index, name in enumerate(relation_names)}
        dgl, torch = _dgl_torch()
        src = torch.tensor(edges["source"].to_numpy(dtype=np.int64), dtype=torch.int64)
        dst = torch.tensor(
            edges["destination"].to_numpy(dtype=np.int64), dtype=torch.int64
        )
        graph = dgl.graph((src, dst), num_nodes=len(source))
        graph.ndata["feature"] = torch.from_numpy(features.values.copy())
        graph.ndata["timestamp"] = torch.from_numpy(
            source["timestamp"].astype("int64").to_numpy(copy=True)
        )
        graph.edata["relationship_type"] = torch.tensor(
            [relation_mapping[name] for name in edges["relationship_type"]],
            dtype=torch.int64,
        )
        graph.edata["timestamp"] = torch.from_numpy(
            edges["edge_timestamp"].astype("int64").to_numpy(copy=True)
        )
        graph.edata["time_delta_seconds"] = torch.tensor(
            edges["time_delta_seconds"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        label_mapping: dict[str, int] = {}
        if "label" in source.columns and source["label"].notna().any():
            labels = source["label"]
            unique_values = sorted(
                labels.dropna().unique().tolist(), key=_label_key
            )
            label_mapping = {
                _label_key(value): index for index, value in enumerate(unique_values)
            }
            encoded_labels = [
                label_mapping[_label_key(value)] if not pd.isna(value) else -1
                for value in labels
            ]
            graph.ndata["label"] = torch.tensor(encoded_labels, dtype=torch.int64)

        graph.ndata["transaction_index"] = torch.arange(
            len(source), dtype=torch.int64
        )
        node_table = source.copy()
        node_table.index = pd.RangeIndex(len(node_table), name="node_id")
        transaction_id_to_node = {
            transaction_id: node_id for node_id, transaction_id in enumerate(ids)
        }
        statistics = self._statistics(graph, source, edges)
        graph_object = TransactionGraph(
            dgl_graph=graph,
            node_table=node_table,
            edge_table=edges,
            transaction_id_to_node=transaction_id_to_node,
            feature_columns=features.columns,
            relation_mapping=relation_mapping,
            categorical_mappings=features.categorical_mappings,
            label_mapping=label_mapping,
            stats=statistics,
            fraud_labels=self.config.fraud_labels,
            edge_history_hours=(
                None if full_history else self.config.max_history_hours
            ),
            default_snapshot_window=self.config.default_snapshot_window,
            feature_availability=features.availability,
            source_field_availability=source.attrs.get(
                "feature_availability", {}
            ).get("fields", {}),
        )
        LOGGER.info(
            "Built transaction graph with %d nodes and %d edges",
            graph.num_nodes(),
            graph.num_edges(),
        )
        return graph_object

    def _statistics(
        self,
        graph: Any,
        transactions: pd.DataFrame,
        edges: pd.DataFrame,
    ) -> dict[str, Any]:
        node_count = graph.num_nodes()
        edge_count = graph.num_edges()
        in_degree = graph.in_degrees().cpu().numpy()
        out_degree = graph.out_degrees().cpu().numpy()
        degree = in_degree + out_degree

        parent = list(range(node_count))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for origin, destination in zip(
            edges.get("source", []), edges.get("destination", [])
        ):
            union(int(origin), int(destination))
        components = len({find(node) for node in range(node_count)})

        labeled = (
            transactions["label"].notna()
            if "label" in transactions.columns
            else pd.Series(False, index=transactions.index)
        )
        positive = (
            transactions["label"].map(
                lambda value: _is_fraud_label(value, self.config.fraud_labels)
            )
            if labeled.any()
            else pd.Series(False, index=transactions.index)
        )
        degree_values, degree_counts = np.unique(degree, return_counts=True)
        return {
            "nodes": int(node_count),
            "edges": int(edge_count),
            "average_degree": float(degree.mean()) if node_count else 0.0,
            "degree_distribution": {
                str(int(value)): int(count)
                for value, count in zip(degree_values, degree_counts)
            },
            "connected_components": int(components),
            "edge_type_distribution": {
                str(name): int(count)
                for name, count in edges["relationship_type"].value_counts().items()
            }
            if not edges.empty
            else {},
            "fraud_node_ratio": (
                float(positive[labeled].mean()) if labeled.any() else None
            ),
            "labeled_nodes": int(labeled.sum()),
        }


def load_transaction_graph(directory: str | Path) -> TransactionGraph:
    """Load a graph previously written by :meth:`TransactionGraph.save`."""
    dgl, _ = _dgl_torch()
    input_dir = Path(directory)
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Graph metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    graph_path = input_dir / metadata["graph_file"]
    node_path = input_dir / metadata["node_metadata_file"]
    if not graph_path.is_file() or not node_path.is_file():
        raise FileNotFoundError("Saved graph is missing its graph or node metadata file")
    graphs, _ = dgl.load_graphs(str(graph_path))
    if len(graphs) != 1:
        raise ValueError(f"Expected one DGL graph in {graph_path}, found {len(graphs)}")
    graph = graphs[0]
    node_records = json.loads(node_path.read_text(encoding="utf-8"))
    node_table = pd.DataFrame(node_records)
    if "timestamp" in node_table:
        node_table["timestamp"] = pd.to_datetime(node_table["timestamp"], utc=True)
    node_table.index = pd.RangeIndex(len(node_table), name="node_id")

    src, dst = graph.edges(order="eid")
    source_ids = src.cpu().numpy()
    destination_ids = dst.cpu().numpy()
    edge_timestamp = pd.to_datetime(
        graph.edata["timestamp"].cpu().numpy(), utc=True
    )
    source_times = pd.to_datetime(
        graph.ndata["timestamp"][src].cpu().numpy(), utc=True
    )
    destination_times = pd.to_datetime(
        graph.ndata["timestamp"][dst].cpu().numpy(), utc=True
    )
    relationship_names = {
        int(value): name for name, value in metadata["relation_mapping"].items()
    }
    relationship_ids = (
        graph.edata["relationship_type"].cpu().numpy()
        if "relationship_type" in graph.edata
        else np.zeros(len(source_ids), dtype=np.int64)
    )
    deltas = (
        graph.edata["time_delta_seconds"].cpu().numpy()
        if "time_delta_seconds" in graph.edata
        else np.zeros(len(source_ids), dtype=np.float32)
    )
    edge_table = pd.DataFrame(
        {
            "source": source_ids,
            "destination": destination_ids,
            "relationship_type": [
                relationship_names[int(value)] for value in relationship_ids
            ],
            "source_timestamp": source_times,
            "destination_timestamp": destination_times,
            "edge_timestamp": edge_timestamp,
            "time_delta_seconds": deltas,
        }
    )
    transaction_ids = node_table["transaction_id"].astype(str).tolist()
    return TransactionGraph(
        dgl_graph=graph,
        node_table=node_table,
        edge_table=edge_table,
        transaction_id_to_node={
            transaction_id: index for index, transaction_id in enumerate(transaction_ids)
        },
        feature_columns=tuple(metadata["feature_columns"]),
        relation_mapping={
            str(name): int(value)
            for name, value in metadata["relation_mapping"].items()
        },
        categorical_mappings=metadata["categorical_mappings"],
        label_mapping={
            str(name): int(value) for name, value in metadata["label_mapping"].items()
        },
        stats=metadata["statistics"],
        fraud_labels=tuple(metadata.get("fraud_labels", (1, "fraud", "suspicious", True))),
        edge_history_hours=metadata.get("edge_history_hours"),
        default_snapshot_window=metadata.get("default_snapshot_window", "24h"),
        feature_availability=metadata.get("feature_availability", {}),
        source_field_availability=metadata.get("source_field_availability", {}),
    )


def build_transaction_graph_from_csv(
    csv_path: str | Path,
    *,
    graph_config: GraphConfig | Mapping[str, Any] | str | Path | None = None,
    schema: TransactionSchema | None = None,
    schema_path: str | Path | None = None,
    availability_report_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    full_history: bool = False,
) -> TransactionGraph:
    """Load a banking CSV, canonicalize it, build a DGL graph, and optionally save.

    ``schema`` and ``schema_path`` are alternatives; schema mappings remain
    independent from the graph relation and feature configuration.
    """
    if schema is not None and schema_path is not None:
        raise ValueError("Pass either schema or schema_path, not both")
    configured_schema = (
        schema
        if schema is not None
        else TransactionSchema.from_yaml(schema_path)
        if schema_path is not None
        else None
    )
    transactions = load_transactions(
        csv_path,
        schema=configured_schema,
        availability_report_path=availability_report_path,
    )
    if isinstance(graph_config, (str, Path)):
        builder = TransactionGraphBuilder.from_yaml(graph_config)
    else:
        builder = TransactionGraphBuilder(graph_config)
    result = builder.build(transactions, full_history=full_history)
    if output_dir is not None:
        result.save(output_dir)
    return result
