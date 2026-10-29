"""Causal relationship edge construction between transaction nodes."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta
import math
from typing import Any, Mapping, Sequence

import pandas as pd

from preprocessing.validation import require_temporal_order


@dataclass(frozen=True)
class EdgeRule:
    """A relationship that can connect an earlier transaction to a later one."""

    name: str
    field: str | None
    enabled: bool = True
    max_age_hours: float | None = None
    connect_all_previous: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EdgeRule:
        name = value.get("name")
        field = value.get("field")
        enabled = value.get("enabled", True)
        max_age = value.get("max_age_hours")
        connect_all = value.get("connect_all_previous", False)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each edge rule requires a non-empty name")
        if field is not None and not isinstance(field, str):
            raise ValueError(f"Edge rule {name!r} field must be a string or null")
        if field is None and not connect_all:
            raise ValueError(
                f"Edge rule {name!r} needs a field or connect_all_previous=true"
            )
        if not isinstance(enabled, bool) or not isinstance(connect_all, bool):
            raise ValueError(f"Edge rule {name!r} flags must be booleans")
        if max_age is not None:
            max_age = float(max_age)
            if not math.isfinite(max_age) or max_age <= 0:
                raise ValueError(f"Edge rule {name!r} max_age_hours must be positive")
        return cls(
            name=name.strip(),
            field=field,
            enabled=enabled,
            max_age_hours=max_age,
            connect_all_previous=connect_all,
        )


class EdgeBuilder:
    """Build directed edges from strictly earlier transactions to current ones.

    Transactions with identical timestamps are processed as a batch, so they
    never create edges to one another. Labels are not used to make edges.
    """

    def __init__(
        self,
        rules: Sequence[EdgeRule | Mapping[str, Any]] | None = None,
        max_history_hours: float | None = 168.0,
    ) -> None:
        if max_history_hours is not None and (
            not math.isfinite(max_history_hours) or max_history_hours <= 0
        ):
            raise ValueError("max_history_hours must be positive or None")
        raw_rules = rules if rules is not None else self.default_rules()
        if not all(isinstance(rule, (EdgeRule, Mapping)) for rule in raw_rules):
            raise ValueError("rules must contain EdgeRule instances or mappings")
        self.rules = tuple(
            rule if isinstance(rule, EdgeRule) else EdgeRule.from_mapping(rule)
            for rule in raw_rules
        )
        names = [rule.name for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("Edge rule names must be unique")
        self.max_history_hours = max_history_hours

    @staticmethod
    def default_rules() -> tuple[EdgeRule, ...]:
        """Return generic relationship rules supported by canonical fields."""
        return (
            EdgeRule("same_sender", "sender_account_id"),
            EdgeRule("same_receiver", "receiver_account_id"),
            EdgeRule("same_beneficiary", "receiver_account_id"),
            EdgeRule("same_device", "device_id"),
            EdgeRule("same_ip", "ip_id"),
            EdgeRule("same_merchant", "merchant_id"),
            EdgeRule("same_channel", "channel", enabled=False),
            EdgeRule(
                "temporal_proximity",
                field=None,
                enabled=False,
                connect_all_previous=True,
            ),
        )

    def build(
        self,
        transactions: pd.DataFrame,
        *,
        max_history_hours: float | None = None,
        full_history: bool = False,
    ) -> pd.DataFrame:
        """Return edge records with source/destination times and relation names.

        ``full_history`` disables both the builder-level and per-rule age
        bounds. Otherwise, an explicit ``max_history_hours`` takes precedence
        over the builder default.
        """
        require_temporal_order(transactions)
        source = transactions.reset_index(drop=True).copy()
        source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True)

        if full_history:
            global_age = None
        elif max_history_hours is not None:
            if max_history_hours <= 0:
                raise ValueError("max_history_hours must be positive")
            global_age = max_history_hours
        else:
            global_age = self.max_history_hours

        edge_records: list[dict[str, Any]] = []
        enabled_rules = [rule for rule in self.rules if rule.enabled]
        for rule in enabled_rules:
            if rule.field is not None and rule.field not in source.columns:
                continue
            rule_age = None if full_history else rule.max_age_hours
            if global_age is not None:
                rule_age = (
                    global_age
                    if rule_age is None
                    else min(global_age, rule_age)
                )
            age_delta = timedelta(hours=rule_age) if rule_age is not None else None
            history_by_value: dict[Any, deque[tuple[int, pd.Timestamp]]] = defaultdict(
                deque
            )
            global_history: deque[tuple[int, pd.Timestamp]] = deque()

            for timestamp, batch in source.sort_values(
                "timestamp", kind="stable"
            ).groupby("timestamp", sort=True):
                now = pd.Timestamp(timestamp)
                if age_delta is not None:
                    cutoff = now - age_delta
                    for history in history_by_value.values():
                        while history and history[0][1] < cutoff:
                            history.popleft()
                    while global_history and global_history[0][1] < cutoff:
                        global_history.popleft()

                for destination, row in batch.iterrows():
                    if rule.connect_all_previous:
                        candidates = list(global_history)
                    else:
                        value = row[rule.field]  # type: ignore[index]
                        if pd.isna(value):
                            continue
                        try:
                            hash(value)
                        except TypeError as exc:
                            raise ValueError(
                                f"Unhashable relationship value in {rule.field!r}"
                            ) from exc
                        candidates = list(history_by_value.get(value, ()))
                    for origin, origin_time in candidates:
                        if origin_time >= now:
                            continue
                        delta_seconds = (now - origin_time).total_seconds()
                        edge_records.append(
                            {
                                "source": origin,
                                "destination": int(destination),
                                "relationship_type": rule.name,
                                "source_timestamp": origin_time,
                                "destination_timestamp": now,
                                "edge_timestamp": now,
                                "time_delta_seconds": delta_seconds,
                            }
                        )

                # Only make the current timestamp batch available after all
                # destinations at this timestamp have been processed.
                for origin, row in batch.iterrows():
                    event = (int(origin), now)
                    global_history.append(event)
                    if rule.field is None:
                        continue
                    value = row[rule.field]
                    if not pd.isna(value):
                        history_by_value[value].append(event)

        columns = [
            "source",
            "destination",
            "relationship_type",
            "source_timestamp",
            "destination_timestamp",
            "edge_timestamp",
            "time_delta_seconds",
        ]
        edges = pd.DataFrame(edge_records, columns=columns)
        if not edges.empty:
            edges = edges.sort_values(
                ["destination_timestamp", "destination", "relationship_type", "source"],
                kind="stable",
            ).reset_index(drop=True)
        return edges
