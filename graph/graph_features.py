"""Numeric transaction-node feature encoding for DGL/GAAP compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


_IDENTIFIER_COLUMNS = {
    "transaction_id",
    "timestamp",
    "sender_account_id",
    "receiver_account_id",
    "currency",
    "merchant_id",
    "device_id",
    "ip_id",
    "location",
    "label",
    "sender_balance_before",
    "sender_balance_after",
    "receiver_balance_before",
    "receiver_balance_after",
}


@dataclass(frozen=True)
class EncodedGraphFeatures:
    """Encoded feature matrix and the names needed to interpret its columns."""

    values: np.ndarray
    columns: tuple[str, ...]
    categorical_mappings: dict[str, dict[str, int]]
    availability: dict[str, bool]


class GraphFeatureEncoder:
    """Encode numeric and categorical values without including labels or IDs."""

    def __init__(
        self,
        feature_columns: Sequence[str] | None = None,
        categorical_columns: Sequence[str] = ("transaction_type", "channel"),
    ) -> None:
        self.feature_columns = tuple(feature_columns) if feature_columns else None
        self.categorical_columns = tuple(categorical_columns)

    def transform(self, transactions: pd.DataFrame) -> EncodedGraphFeatures:
        """Create a deterministic float32 matrix with explicit missing indicators."""
        if self.feature_columns is not None:
            requested = list(self.feature_columns)
        else:
            configured = transactions.attrs.get("feature_columns")
            if isinstance(configured, (list, tuple)):
                requested = [name for name in configured if name in transactions]
            else:
                requested = [
                    name
                    for name in transactions.columns
                    if name not in _IDENTIFIER_COLUMNS
                    and pd.api.types.is_numeric_dtype(transactions[name])
                ]
                requested.extend(
                    name
                    for name in self.categorical_columns
                    if name in transactions and name not in requested
                )

        forbidden = set(requested) & _IDENTIFIER_COLUMNS
        if forbidden:
            raise ValueError(
                f"Feature columns cannot contain labels, IDs, or timestamps: "
                f"{sorted(forbidden)}"
            )
        missing_columns = [name for name in requested if name not in transactions]
        if missing_columns:
            raise ValueError(f"Requested graph feature columns are missing: {missing_columns}")

        feature_parts: list[np.ndarray] = []
        output_columns: list[str] = []
        category_mappings: dict[str, dict[str, int]] = {}
        availability: dict[str, bool] = {}
        for name in requested:
            values = transactions[name]
            available = bool(values.notna().any())
            availability[name] = available
            if name in self.categorical_columns or not pd.api.types.is_numeric_dtype(
                values
            ):
                categories = sorted(
                    str(value) for value in values.dropna().unique().tolist()
                )
                mapping = {category: index for index, category in enumerate(categories)}
                category_mappings[name] = mapping
                encoded = np.zeros((len(values), len(categories)), dtype=np.float32)
                for row, value in enumerate(values):
                    if pd.isna(value):
                        continue
                    encoded[row, mapping[str(value)]] = 1.0
                feature_parts.append(encoded)
                output_columns.extend(f"{name}={category}" for category in categories)
                if not categories:
                    output_columns.append(f"{name}__unavailable")
                    feature_parts.append(
                        np.zeros((len(values), 1), dtype=np.float32)
                    )
                continue

            numeric = pd.to_numeric(values, errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            present = np.isfinite(numeric)
            numeric = np.where(present, numeric, 0.0)
            feature_parts.append(numeric.astype(np.float32).reshape(-1, 1))
            feature_parts.append(present.astype(np.float32).reshape(-1, 1))
            output_columns.extend((name, f"{name}__available"))

        if feature_parts:
            matrix = np.concatenate(feature_parts, axis=1)
        else:
            matrix = np.zeros((len(transactions), 0), dtype=np.float32)
        return EncodedGraphFeatures(
            values=matrix,
            columns=tuple(output_columns),
            categorical_mappings=category_mappings,
            availability=availability,
        )


def to_networkx_sample(
    transaction_graph: Any,
    max_nodes: int = 100,
) -> Any:
    """Create a bounded NetworkX MultiDiGraph for small-graph visualization.

    The sample is the first ``max_nodes`` transactions in timestamp order. This
    deterministic induced sample is intended for inspection, not model input.
    """
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "NetworkX is required for transaction graph visualization"
        ) from exc

    graph_data = transaction_graph
    nodes = graph_data.node_table.sort_values("timestamp", kind="stable").head(max_nodes)
    selected = set(nodes.index.astype(int).tolist())
    visual = nx.MultiDiGraph()
    for node_id, row in nodes.iterrows():
        attributes = {
            "transaction_id": str(row["transaction_id"]),
            "timestamp": str(row["timestamp"]),
        }
        if "label" in row and not pd.isna(row["label"]):
            attributes["label"] = row["label"]
        visual.add_node(int(node_id), **attributes)
    for edge in graph_data.edge_table.itertuples(index=False):
        source = int(edge.source)
        destination = int(edge.destination)
        if source in selected and destination in selected:
            visual.add_edge(
                source,
                destination,
                relationship_type=str(edge.relationship_type),
                timestamp=str(edge.edge_timestamp),
                time_delta_seconds=float(edge.time_delta_seconds),
            )
    return visual
