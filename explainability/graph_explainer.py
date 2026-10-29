"""Causal historical-neighborhood explanations for transaction graph nodes."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import pandas as pd


class GraphExplainer:
    """Find important incoming historical relationships without future leakage."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        settings = dict(config or {})
        self.max_neighbors = int(settings.get("max_related_transactions", 10))
        self.max_hops = int(settings.get("max_connection_hops", 2))
        self.relationship_weights = dict(
            settings.get("relationship_weights", {})
        )
        for name, weight in self.relationship_weights.items():
            if not isinstance(name, str) or not isinstance(weight, (int, float)):
                raise ValueError("Relationship weights must map names to numbers")
            if not pd.notna(weight) or weight < 0:
                raise ValueError("Relationship weights must be finite and non-negative")
        self.expose_historical_labels = bool(
            settings.get("expose_historical_labels", True)
        )
        self.label_available_at_column = str(
            settings.get("label_available_at_column", "label_available_at")
        )
        self.fraud_labels = {
            self._label_key(value)
            for value in settings.get(
                "fraud_labels", (1, "1", True, "true", "fraud", "suspicious")
            )
        }
        if self.max_neighbors < 1 or self.max_hops < 1:
            raise ValueError("Graph explanation neighbor and hop limits must be positive")

    def explain(
        self,
        transaction_graph: Any,
        node_id: int,
        *,
        cutoff: Any | None = None,
    ) -> dict[str, Any]:
        """Return ranked ancestors, only following strictly earlier directed edges."""
        node_table = transaction_graph.node_table
        edge_table = transaction_graph.edge_table
        if not 0 <= node_id < len(node_table):
            raise IndexError(f"Transaction node {node_id} is outside the graph")
        target = node_table.iloc[node_id]
        target_time = pd.Timestamp(cutoff if cutoff is not None else target["timestamp"])
        if target_time.tzinfo is None:
            target_time = target_time.tz_localize("UTC")
        else:
            target_time = target_time.tz_convert("UTC")
        actual_target_time = self._row_timestamp(target)
        if target_time < actual_target_time:
            raise ValueError("Graph explanation cutoff cannot precede transaction time")
        required_edges = {"source", "destination", "relationship_type"}
        missing = required_edges - set(edge_table.columns)
        if missing:
            raise ValueError(f"Transaction edge table is missing columns: {sorted(missing)}")

        incoming: dict[int, list[dict[str, Any]]] = {}
        for edge in edge_table.to_dict(orient="records"):
            source_id = int(edge["source"])
            destination_id = int(edge["destination"])
            if source_id == destination_id or not (
                0 <= source_id < len(node_table)
                and 0 <= destination_id < len(node_table)
            ):
                continue
            source_time = self._row_timestamp(node_table.iloc[source_id])
            destination_time = self._row_timestamp(node_table.iloc[destination_id])
            edge_time_value = edge.get("edge_timestamp", destination_time)
            edge_time = pd.Timestamp(edge_time_value)
            if edge_time.tzinfo is None:
                edge_time = edge_time.tz_localize("UTC")
            else:
                edge_time = edge_time.tz_convert("UTC")
            if (
                source_time >= destination_time
                or destination_time > target_time
                or edge_time > target_time
            ):
                continue
            incoming.setdefault(destination_id, []).append(
                {
                    "source": source_id,
                    "destination": destination_id,
                    "relationship": str(edge["relationship_type"]),
                    "timestamp": edge_time,
                    "time_delta_seconds": float(
                        edge.get(
                            "time_delta_seconds",
                            (destination_time - source_time).total_seconds(),
                        )
                    ),
                }
            )

        queue: deque[tuple[int, list[dict[str, Any]]]] = deque([(node_id, [])])
        shortest_depth = {node_id: 0}
        candidates: dict[int, dict[str, Any]] = {}
        while queue:
            current, reverse_path = queue.popleft()
            depth = len(reverse_path)
            if depth >= self.max_hops:
                continue
            for edge in incoming.get(current, []):
                ancestor = edge["source"]
                path = [*reverse_path, edge]
                next_depth = len(path)
                if ancestor not in candidates or next_depth < candidates[ancestor]["hops"]:
                    candidates[ancestor] = {
                        "hops": next_depth,
                        "reverse_path": path,
                        "score": sum(
                            float(
                                self.relationship_weights.get(
                                    step["relationship"], 1.0
                                )
                            )
                            for step in path
                        )
                        / next_depth,
                    }
                if next_depth < self.max_hops and next_depth < shortest_depth.get(
                    ancestor, self.max_hops + 1
                ):
                    shortest_depth[ancestor] = next_depth
                    queue.append((ancestor, path))

        related: list[dict[str, Any]] = []
        for ancestor, candidate in candidates.items():
            row = node_table.iloc[ancestor]
            reverse_path = candidate["reverse_path"]
            forward_path = list(reversed(reverse_path))
            first_edge = forward_path[0]
            related.append(
                {
                    "transaction_id": str(row["transaction_id"]),
                    "timestamp": self._row_timestamp(row).isoformat(),
                    "relationship": first_edge["relationship"],
                    "fraud_status": self._fraud_status(row, target_time),
                    "connection_path": [
                        {
                            "source_transaction_id": str(
                                node_table.iloc[step["source"]]["transaction_id"]
                            ),
                            "destination_transaction_id": str(
                                node_table.iloc[step["destination"]]["transaction_id"]
                            ),
                            "relationship": step["relationship"],
                            "timestamp": step["timestamp"].isoformat(),
                        }
                        for step in forward_path
                    ],
                    "hops": int(candidate["hops"]),
                    "importance": float(candidate["score"]),
                    "time_delta_seconds": float(
                        (target_time - self._row_timestamp(row)).total_seconds()
                    ),
                }
            )
        related.sort(
            key=lambda item: (
                -item["importance"],
                item["hops"],
                item["time_delta_seconds"],
                item["transaction_id"],
            )
        )
        related = related[: self.max_neighbors]
        relationship_counts: dict[str, int] = {}
        for item in related:
            relationship = item["relationship"]
            relationship_counts[relationship] = (
                relationship_counts.get(relationship, 0) + 1
            )
        suspicious_ids = [
            item["transaction_id"]
            for item in related
            if item["fraud_status"] == "fraud"
        ]
        return {
            "related_transactions": related,
            "graph_reasons": [
                {
                    "code": "GRAPH_RELATIONSHIPS",
                    "message": (
                        f"Transaction has {len(related)} earlier transaction "
                        "connection(s) through the displayed relationships."
                    ),
                    "supporting_value": {
                        "historical_neighbor_count": len(related),
                        "relationship_counts": relationship_counts,
                        "max_connection_hops": self.max_hops,
                    },
                }
            ]
            if related
            else [],
            "historical_suspicious_neighbor_ids": suspicious_ids,
            "cutoff_timestamp": target_time.isoformat(),
        }

    def _fraud_status(
        self, row: pd.Series, cutoff: pd.Timestamp
    ) -> str | None:
        if (
            not self.expose_historical_labels
            or "label" not in row
            or self.label_available_at_column not in row
            or pd.isna(row["label"])
            or pd.isna(row[self.label_available_at_column])
        ):
            return None
        available_at = pd.Timestamp(row[self.label_available_at_column])
        if available_at.tzinfo is None:
            available_at = available_at.tz_localize("UTC")
        else:
            available_at = available_at.tz_convert("UTC")
        if available_at > cutoff:
            return None
        label = self._label_key(row["label"])
        if label in self.fraud_labels:
            return "fraud"
        if label in {"0", "false", "legitimate", "legit", "genuine"}:
            return "legitimate"
        return None

    @staticmethod
    def _row_timestamp(row: pd.Series) -> pd.Timestamp:
        value = pd.Timestamp(row["timestamp"])
        return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")

    @staticmethod
    def _label_key(value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value).strip().lower()
        if numeric.is_integer():
            return str(int(numeric))
        return str(value).strip().lower()
