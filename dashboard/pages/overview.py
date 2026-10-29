"""Main operational summary page."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard.components.metrics import render_metric_row


def render(
    transactions: pd.DataFrame,
    model: dict[str, Any] | None,
    drift: dict[str, Any] | None,
) -> None:
    st.title("Fraud operations overview")
    st.caption("Transaction scores and model status served by Adaptive-GAAP.")
    if transactions.empty:
        st.info("Load demo transactions or score traffic through the API to populate this view.")
        return

    suspicious = transactions["risk_level"].isin(["HIGH", "CRITICAL"])
    suspicious_amounts = (
        transactions.loc[suspicious]
        .groupby("currency", dropna=False)["amount"]
        .sum()
    )
    estimated_amount = (
        ", ".join(
            f"{amount:,.2f} {currency if pd.notna(currency) else 'unknown currency'}"
            for currency, amount in suspicious_amounts.items()
        )
        if not suspicious_amounts.empty
        else "N/A"
    )
    model = model or {}
    metrics = model.get("metrics") or {}
    drift = drift or {}
    render_metric_row(
        [
            ("Total transactions", f"{len(transactions):,}", None),
            ("Suspicious transactions", f"{int(suspicious.sum()):,}", "High or critical configured risk level"),
            ("Estimated suspicious amount", estimated_amount, "High/critical amount totals are kept separate by currency"),
            ("Model version", model.get("model_version", "N/A"), None),
            ("PR-AUC", _format_metric(metrics.get("average_precision")), "Test split average precision when available"),
            ("Recall", _format_metric(metrics.get("recall")), "Test split recall when available"),
            ("Precision", _format_metric(metrics.get("precision")), "Test split precision when available"),
            ("Drift", "DETECTED" if drift.get("drift_detected") else "CLEAR", drift.get("drift_type") or "No active detector alert"),
        ]
    )
    st.subheader("Fraud probability distribution")
    histogram = transactions["risk_level"].value_counts().reindex(
        ["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0
    )
    st.bar_chart(histogram)
    thresholds = model.get("risk_thresholds")
    if isinstance(thresholds, dict):
        st.caption(
            "Risk bands use configured score thresholds: "
            + ", ".join(
                f"{name.upper()} ≥ {float(value):.3f}"
                for name, value in thresholds.items()
            )
        )
    if model and not metrics:
        st.caption(
            model.get(
                "metrics_unavailable_reason",
                "Evaluation metrics are unavailable from the loaded model.",
            )
        )


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"
