"""Safe, compact Graphviz rendering for a selected transaction's neighbors."""

from __future__ import annotations

import json
from typing import Any, Sequence

import streamlit as st


def render_transaction_graph(
    transaction_id: str, related_transactions: Sequence[dict[str, Any]]
) -> None:
    """Render related historical transactions without displaying account IDs."""
    if not related_transactions:
        st.info("No historical related transactions were returned for this row.")
        return
    lines = ["digraph transactions {", "rankdir=LR;", 'node [shape="box"];']
    target = json.dumps(f"Selected: {transaction_id}")
    lines.append(f"{target} [style=filled, fillcolor=lightcoral];")
    for index, related in enumerate(related_transactions):
        related_id = str(related.get("transaction_id", f"related-{index}"))
        source = json.dumps(f"{related_id}\\n{related.get('timestamp', '')}")
        lines.append(f"{source} -> {target} [label={json.dumps(str(related.get('relationship', 'related')))}];")
    lines.append("}")
    st.graphviz_chart("\n".join(lines), use_container_width=True)
