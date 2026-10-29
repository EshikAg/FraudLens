"""Point-in-time and local temporal views of transaction graphs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
import re
from typing import Any

import numpy as np
import pandas as pd

from graph.transaction_graph import (
    TransactionGraph,
    _dgl_torch,
    _is_fraud_label,
)


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    else:
        result = result.tz_convert("UTC")
    return result


def _window(value: str | int | float | pd.Timedelta | timedelta) -> pd.Timedelta:
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError("Window size in hours must be positive")
        return pd.Timedelta(hours=float(value))
    normalized = re.sub(r"(\d)d\b", r"\1D", value) if isinstance(value, str) else value
    parsed = pd.Timedelta(normalized)
    if pd.isna(parsed) or parsed <= pd.Timedelta(0):
        raise ValueError("Window must be positive")
    return parsed


class TemporalGraph:
    """Create leakage-safe snapshots from an immutable historical graph."""

    def __init__(
        self,
        transaction_graph: TransactionGraph,
        default_window: str | int | float | pd.Timedelta | None = None,
    ) -> None:
        self.transaction_graph = transaction_graph
        configured_window = (
            default_window
            if default_window is not None
            else transaction_graph.default_snapshot_window
        )
        self.default_window = _window(configured_window)

    def build(
        self,
        mode: str = "full_history",
        *,
        as_of: Any | None = None,
        window: str | int | float | pd.Timedelta | timedelta | None = None,
        transaction_ids: Sequence[str | int] | None = None,
        hops: int = 1,
    ) -> TransactionGraph:
        """Build a full snapshot, rolling window, or historical neighborhood."""
        if mode == "full_history":
            return self.full_history(as_of=as_of)
        if mode == "time_window":
            return self.time_window(as_of=as_of, window=window)
        if mode == "local_neighborhood":
            if transaction_ids is None:
                raise ValueError("transaction_ids are required for a local neighborhood")
            return self.local_neighborhood(
                transaction_ids, as_of=as_of, window=window, hops=hops
            )
        raise ValueError(
            "mode must be 'full_history', 'time_window', or 'local_neighborhood'"
        )

    def full_history(self, *, as_of: Any | None = None) -> TransactionGraph:
        """Return all transactions and edges available by the cutoff."""
        graph = self.transaction_graph
        if as_of is None:
            if graph.node_table.empty:
                return self._view([], [])
            cutoff = graph.node_table["timestamp"].max()
        else:
            cutoff = _timestamp(as_of)
        node_times = pd.to_datetime(graph.node_table["timestamp"], utc=True)
        node_ids = np.flatnonzero((node_times <= cutoff).to_numpy()).tolist()
        edge_mask = (
            pd.to_datetime(graph.edge_table["edge_timestamp"], utc=True) <= cutoff
            if not graph.edge_table.empty
            else pd.Series(dtype=bool)
        )
        edge_ids = np.flatnonzero(edge_mask.to_numpy()).tolist()
        return self._view(node_ids, edge_ids)

    def time_window(
        self,
        *,
        as_of: Any | None = None,
        window: str | int | float | pd.Timedelta | timedelta | None = None,
    ) -> TransactionGraph:
        """Return nodes and edges in a causal rolling window ending at ``as_of``."""
        graph = self.transaction_graph
        if graph.node_table.empty:
            return self._view([], [])
        cutoff = (
            _timestamp(as_of)
            if as_of is not None
            else pd.to_datetime(graph.node_table["timestamp"], utc=True).max()
        )
        duration = _window(window) if window is not None else self.default_window
        self._validate_available_history(duration)
        start = cutoff - duration
        node_times = pd.to_datetime(graph.node_table["timestamp"], utc=True)
        node_ids = np.flatnonzero(((node_times >= start) & (node_times <= cutoff)).to_numpy()).tolist()
        if graph.edge_table.empty:
            return self._view(node_ids, [])
        source_times = pd.to_datetime(graph.edge_table["source_timestamp"], utc=True)
        destination_times = pd.to_datetime(
            graph.edge_table["destination_timestamp"], utc=True
        )
        edge_mask = (
            (source_times >= start)
            & (destination_times <= cutoff)
            & (destination_times > source_times)
        )
        edge_ids = np.flatnonzero(edge_mask.to_numpy()).tolist()
        return self._view(node_ids, edge_ids)

    def local_neighborhood(
        self,
        transaction_ids: Sequence[str | int],
        *,
        as_of: Any | None = None,
        window: str | int | float | pd.Timedelta | timedelta | None = None,
        hops: int = 1,
    ) -> TransactionGraph:
        """Return anchor transactions and their incoming historical neighbors."""
        if hops < 1:
            raise ValueError("hops must be at least one")
        graph = self.transaction_graph
        if not transaction_ids:
            raise ValueError("At least one transaction ID is required")
        anchors: set[int] = set()
        for value in transaction_ids:
            if isinstance(value, int):
                node_id = value
                if node_id < 0 or node_id >= len(graph.node_table):
                    raise KeyError(f"Unknown transaction node index: {value}")
            else:
                try:
                    node_id = graph.transaction_id_to_node[str(value)]
                except KeyError as exc:
                    raise KeyError(f"Unknown transaction ID: {value}") from exc
            anchors.add(node_id)

        node_times = pd.to_datetime(graph.node_table["timestamp"], utc=True)
        anchor_cutoff = node_times.loc[list(anchors)].max()
        cutoff = _timestamp(as_of) if as_of is not None else anchor_cutoff
        if cutoff < anchor_cutoff:
            raise ValueError("as_of cannot precede an anchor transaction")
        if window is None:
            start = None
        else:
            duration = _window(window)
            self._validate_available_history(duration)
            start = cutoff - duration
        if graph.edge_table.empty:
            return self._view(sorted(anchors), [])

        edges = graph.edge_table
        source_times = pd.to_datetime(edges["source_timestamp"], utc=True)
        destination_times = pd.to_datetime(edges["destination_timestamp"], utc=True)
        eligible = (destination_times <= cutoff) & (source_times < destination_times)
        if start is not None:
            eligible &= source_times >= start
        eligible_ids = np.flatnonzero(eligible.to_numpy())
        incoming: dict[int, list[tuple[int, int]]] = {}
        for edge_id in eligible_ids:
            origin = int(edges.iloc[edge_id]["source"])
            destination = int(edges.iloc[edge_id]["destination"])
            incoming.setdefault(destination, []).append((origin, int(edge_id)))

        selected_nodes = set(anchors)
        frontier = set(anchors)
        selected_edges: set[int] = set()
        for _ in range(hops):
            previous: set[int] = set()
            for destination in frontier:
                for origin, edge_id in incoming.get(destination, []):
                    selected_edges.add(edge_id)
                    if origin not in selected_nodes:
                        previous.add(origin)
            selected_nodes.update(previous)
            frontier = previous
            if not frontier:
                break
        return self._view(sorted(selected_nodes), sorted(selected_edges))

    def _validate_available_history(self, duration: pd.Timedelta) -> None:
        history_hours = self.transaction_graph.edge_history_hours
        if history_hours is not None and duration.total_seconds() > history_hours * 3600:
            raise ValueError(
                f"Requested window ({duration}) exceeds the graph's available "
                f"edge history ({history_hours:g} hours); build with "
                "full_history=True to retain older relationships"
            )

    def _view(
        self, node_ids: Sequence[int], edge_ids: Sequence[int]
    ) -> TransactionGraph:
        """Create a relabeled DGL graph and aligned metadata tables."""
        dgl, torch = _dgl_torch()
        original = self.transaction_graph
        nodes = sorted(set(int(value) for value in node_ids))
        node_map = {original_id: new_id for new_id, original_id in enumerate(nodes)}
        edges = original.edge_table.iloc[list(edge_ids)].copy()
        if edges.empty:
            edges = original.edge_table.iloc[0:0].copy()
        else:
            edges = edges[
                edges["source"].isin(node_map) & edges["destination"].isin(node_map)
            ].copy()
        edges["source"] = edges["source"].map(node_map).astype("int64")
        edges["destination"] = edges["destination"].map(node_map).astype("int64")
        src = torch.tensor(edges["source"].to_numpy(dtype=np.int64), dtype=torch.int64)
        dst = torch.tensor(
            edges["destination"].to_numpy(dtype=np.int64), dtype=torch.int64
        )
        subgraph = dgl.graph((src, dst), num_nodes=len(nodes))
        node_indices = torch.tensor(nodes, dtype=torch.int64)
        edge_indices = torch.tensor(
            list(edges.index.map(lambda index: original.edge_table.index.get_loc(index))),
            dtype=torch.int64,
        )
        for name, values in original.dgl_graph.ndata.items():
            subgraph.ndata[name] = values[node_indices]
        for name, values in original.dgl_graph.edata.items():
            subgraph.edata[name] = values[edge_indices]

        node_table = original.node_table.iloc[nodes].copy()
        node_table.index = pd.RangeIndex(len(node_table), name="node_id")
        transaction_id_to_node = {
            str(value): index
            for index, value in enumerate(node_table["transaction_id"])
        }
        stats = _view_statistics(
            subgraph,
            node_table,
            edges,
            original.fraud_labels,
        )
        return TransactionGraph(
            dgl_graph=subgraph,
            node_table=node_table,
            edge_table=edges.reset_index(drop=True),
            transaction_id_to_node=transaction_id_to_node,
            feature_columns=original.feature_columns,
            relation_mapping=original.relation_mapping,
            categorical_mappings=original.categorical_mappings,
            label_mapping=original.label_mapping,
            stats=stats,
            fraud_labels=original.fraud_labels,
            edge_history_hours=original.edge_history_hours,
            default_snapshot_window=original.default_snapshot_window,
            feature_availability=original.feature_availability,
            source_field_availability=original.source_field_availability,
        )


def _view_statistics(
    graph: Any,
    node_table: pd.DataFrame,
    edges: pd.DataFrame,
    fraud_labels: Sequence[Any],
) -> dict[str, Any]:
    node_count = graph.num_nodes()
    degree = (
        graph.in_degrees().cpu().numpy() + graph.out_degrees().cpu().numpy()
    )
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for origin, destination in zip(
        edges.get("source", []), edges.get("destination", [])
    ):
        left, right = find(int(origin)), find(int(destination))
        if left != right:
            parent[right] = left
    values, counts = np.unique(degree, return_counts=True)
    labeled = (
        node_table["label"].notna()
        if "label" in node_table
        else pd.Series(False, index=node_table.index)
    )
    positives = (
        node_table["label"].map(
            lambda label: _is_fraud_label(label, fraud_labels)
        )
        if labeled.any()
        else pd.Series(False, index=node_table.index)
    )
    return {
        "nodes": int(node_count),
        "edges": int(graph.num_edges()),
        "average_degree": float(degree.mean()) if node_count else 0.0,
        "degree_distribution": {
            str(int(value)): int(count) for value, count in zip(values, counts)
        },
        "connected_components": len({find(node) for node in range(node_count)}),
        "edge_type_distribution": (
            {
                str(name): int(count)
                for name, count in edges["relationship_type"].value_counts().items()
            }
            if not edges.empty
            else {}
        ),
        "fraud_node_ratio": (
            float(positives[labeled].mean()) if labeled.any() else None
        ),
        "labeled_nodes": int(labeled.sum()),
    }
