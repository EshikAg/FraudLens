"""Current and historical data/concept drift monitoring."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render(
    drift: dict[str, Any] | None,
    drift_report: dict[str, Any] | None,
    adaptation_report: dict[str, Any] | None,
    model: dict[str, Any] | None,
) -> None:
    st.title("Drift monitor")
    drift = drift or {}
    drift_report = drift_report or {}
    adaptation_report = adaptation_report or {}
    event = adaptation_report.get("phases", {}).get("C_drift_event", {})
    if not isinstance(event, dict):
        event = {}
    detected = bool(drift.get("drift_detected") or drift_report.get("data_drift", {}).get("drift_detected"))
    status = "DRIFT DETECTED" if detected else "NO ACTIVE DRIFT"
    columns = st.columns(4)
    columns[0].metric("Status", status)
    columns[1].metric("Score", _format(drift.get("drift_score", drift_report.get("data_drift", {}).get("drift_score"))))
    columns[2].metric("Drift type", drift.get("drift_type") or event.get("drift_type") or "N/A")
    columns[3].metric("Active model", (model or {}).get("model_version", "N/A"))
    affected = (
        drift.get("affected_features")
        or event.get("affected_features")
        or drift_report.get("data_drift", {}).get("affected_features", [])
    )
    st.write("Affected features:", ", ".join(affected) if affected else "Not available")

    events = drift_report.get("monitor_events", [])
    if not events and event:
        events = [event]
    st.subheader("Drift event history")
    if events:
        table = [
            {
                "timestamp": item.get("timestamp"),
                "type": item.get("drift_type"),
                "score": item.get("drift_score"),
                "affected_features": ", ".join(item.get("affected_features", [])),
            }
            for item in events
        ]
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    else:
        st.info("No persisted drift-event report is configured.")

    performance = drift_report.get("performance", {})
    st.subheader("Performance before and after drift")
    if performance:
        table = pd.DataFrame(performance).T
        st.dataframe(table, use_container_width=True)
        available_metrics = [
            name for name in ("precision", "recall", "f1", "average_precision")
            if name in table.columns
        ]
        if available_metrics:
            st.line_chart(table.loc[:, available_metrics])
    else:
        st.info("Before/after performance metrics are not available from configured artifacts.")

    adaptation = adaptation_report.get("phases", {}).get("D_adaptation", {})
    if isinstance(adaptation, dict) and adaptation:
        st.subheader("Adaptation event")
        st.json(
            {
                "promoted": adaptation.get("promoted"),
                "version": adaptation.get("version"),
                "current_version": adaptation.get("current_version"),
                "reason": adaptation.get("promotion_decision"),
            }
        )
    else:
        st.caption("No adaptation event is recorded in the configured experiment artifact.")


def _format(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"
