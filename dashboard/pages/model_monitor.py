"""Model lineage and GAAP-versus-Adaptive-GAAP comparison."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render(
    versions: list[dict[str, Any]],
    active_version: str | None,
    adaptation_report: dict[str, Any] | None,
    current_model: dict[str, Any] | None,
) -> None:
    st.title("Model monitor")
    records = versions
    if not records and current_model and current_model.get("model_loaded"):
        records = [
            {
                "version": current_model.get("model_version"),
                "timestamp": None,
                "training_data_window": None,
                "metrics": {"test": current_model.get("metrics")},
                "drift_event": None,
                "status": "active",
            }
        ]
    if not records:
        st.info("No model registry artifact or loaded model is available.")
        return

    rows = []
    for record in records:
        metrics = _metrics(record.get("metrics", {}))
        rows.append(
            {
                "version": record.get("version"),
                "trained_time": record.get("timestamp"),
                "training_data_window": _window(record.get("training_data_window")),
                "PR-AUC": metrics.get("average_precision"),
                "recall": metrics.get("recall"),
                "precision": metrics.get("precision"),
                "F1": metrics.get("f1"),
                "drift_event": _drift_description(record.get("drift_event")),
                "status": (
                    "active"
                    if record.get("version") == active_version
                    else record.get("status", "retired")
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("GAAP vs Adaptive-GAAP performance")
    comparison = (adaptation_report or {}).get("models", {})
    series: dict[str, dict[str, Any]] = {}
    for key, label in (
        ("old_gaap_fixed_baseline", "GAAP v1"),
        ("full_retrained_gaap", "Full retrain"),
        ("adaptive_gaap", "Adaptive-GAAP"),
    ):
        model = comparison.get(key, {})
        metrics = model.get("metrics", {})
        if isinstance(metrics, dict) and "validation" in metrics:
            metrics = metrics["validation"]
        if isinstance(metrics, dict) and metrics:
            series[label] = {
                "PR-AUC": metrics.get("average_precision"),
                "Recall": metrics.get("recall"),
                "Precision": metrics.get("precision"),
                "F1": metrics.get("f1"),
            }
    if series:
        chart = pd.DataFrame.from_dict(series, orient="index")
        st.line_chart(chart)
        st.caption("Comparison metrics are shown only when present in the stored experiment report.")
    else:
        st.info("No persisted Adaptive-GAAP comparison report is configured.")


def _metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    for key in ("validation", "test"):
        nested = metrics.get(key)
        if isinstance(nested, dict):
            return nested
    return metrics


def _window(value: Any) -> str:
    if not isinstance(value, dict):
        return "N/A"
    return f"{value.get('start', '?')} – {value.get('end', '?')} ({value.get('rows', '?')} rows)"


def _drift_description(value: Any) -> str:
    if not isinstance(value, dict):
        return "None"
    return f"{value.get('drift_type', 'drift')} @ {value.get('timestamp', 'unknown time')}"
