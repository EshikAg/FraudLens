"""Reusable compact dashboard metric components."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_metric_row(values: list[tuple[str, Any, str | None]]) -> None:
    """Render labeled metrics while preserving unavailable values as N/A."""
    columns = st.columns(len(values))
    for column, (label, value, help_text) in zip(columns, values):
        display = "N/A" if value is None else value
        column.metric(label, display, help=help_text)
