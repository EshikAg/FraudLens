"""Filterable transaction monitor table."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from datetime import datetime, time, timezone


DISPLAY_COLUMNS = [
    "transaction_id",
    "timestamp",
    "amount",
    "transaction_type",
    "fraud_probability",
    "risk_level",
    "decision",
    "model_version",
]


def render(transactions: pd.DataFrame, drift: dict | None) -> None:
    st.title("Transaction monitor")
    if transactions.empty:
        st.info("No scored transactions are available yet.")
        return
    filtered = transactions.copy()
    risk_values = sorted(filtered["risk_level"].dropna().unique().tolist())
    type_values = sorted(filtered["transaction_type"].dropna().unique().tolist())
    left, middle, right = st.columns(3)
    selected_risks = left.multiselect("Risk level", risk_values, default=risk_values)
    selected_types = middle.multiselect("Transaction type", type_values, default=type_values)
    drift_only = right.checkbox("Only transactions during/after drift")
    filtered = filtered.loc[
        filtered["risk_level"].isin(selected_risks)
        & filtered["transaction_type"].isin(selected_types)
    ]
    if filtered.empty:
        st.info("No transactions match the selected risk and transaction-type filters.")
        return

    timestamps = pd.to_datetime(filtered["timestamp"], utc=True)
    minimum = timestamps.min().date()
    maximum = timestamps.max().date()
    date_column, time_column = st.columns(2)
    date_selection = date_column.date_input(
        "Date range (UTC)",
        value=(minimum, maximum),
        min_value=minimum,
        max_value=maximum,
    )
    start_time, end_time = time_column.columns(2)
    selected_start_time = start_time.time_input(
        "From time (UTC)", value=time(hour=0, minute=0)
    )
    selected_end_time = end_time.time_input(
        "To time (UTC)", value=time(hour=23, minute=59)
    )
    if isinstance(date_selection, tuple) and len(date_selection) == 2:
        start, end = date_selection
        start_timestamp = pd.Timestamp(
            datetime.combine(start, selected_start_time), tz=timezone.utc
        )
        end_timestamp = pd.Timestamp(
            datetime.combine(end, selected_end_time), tz=timezone.utc
        )
        filtered = filtered.loc[
            pd.to_datetime(filtered["timestamp"], utc=True).between(
                start_timestamp, end_timestamp
            )
        ]

    amount_min = float(filtered["amount"].min()) if not filtered.empty else 0.0
    amount_max = float(filtered["amount"].max()) if not filtered.empty else 0.0
    low, high = st.slider(
        "Amount range",
        min_value=amount_min,
        max_value=max(amount_max, amount_min + 0.01),
        value=(amount_min, amount_max),
    )
    filtered = filtered.loc[filtered["amount"].between(low, high)]

    drift_timestamp = (drift or {}).get("timestamp")
    if drift_only:
        if not drift_timestamp or not (drift or {}).get("drift_detected"):
            st.info("No active drift timestamp is available to filter.")
            filtered = filtered.iloc[0:0]
        else:
            filtered = filtered.loc[
                pd.to_datetime(filtered["timestamp"], utc=True)
                >= pd.Timestamp(drift_timestamp)
            ]
    if filtered.empty:
        st.info("No transactions match the selected filters.")
        return
    st.dataframe(
        filtered.loc[:, DISPLAY_COLUMNS].sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
