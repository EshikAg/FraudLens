"""Selected transaction evidence and historical graph view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.transaction_graph import render_transaction_graph


def render(transactions: pd.DataFrame) -> None:
    st.title("Transaction investigation")
    if transactions.empty:
        st.info("Score or load demo transactions before opening an investigation.")
        return
    identifiers = transactions["transaction_id"].astype(str).tolist()
    default_index = next(
        (
            index
            for index, row in transactions.reset_index(drop=True).iterrows()
            if row["risk_level"] in ("HIGH", "CRITICAL")
        ),
        0,
    )
    selected = st.selectbox(
        "Select transaction",
        identifiers,
        index=default_index,
        key="investigation_transaction",
    )
    row = transactions.loc[
        transactions["transaction_id"].astype(str) == selected
    ].iloc[0]
    st.subheader(f"Transaction {selected}")
    details = row.get("transaction_details", {})
    if not isinstance(details, dict):
        details = {}
    st.json(
        {
            "timestamp": str(row["timestamp"]),
            "amount": row["amount"],
            "currency": details.get("currency"),
            "transaction_type": row["transaction_type"],
            "channel": details.get("channel"),
            "merchant_id": details.get("merchant_id"),
            "location": details.get("location"),
        }
    )
    columns = st.columns(4)
    columns[0].metric("Fraud probability", f"{row['fraud_probability']:.3f}")
    columns[1].metric("Risk level", row["risk_level"])
    columns[2].metric("Decision", row["decision"])
    columns[3].metric("Model version", row["model_version"])
    columns[0].caption(f"Prediction timestamp: {row['timestamp']}")

    reasons = row.get("reasons", [])
    st.subheader("Reasons and supporting evidence")
    if reasons:
        st.dataframe(pd.DataFrame(reasons), use_container_width=True, hide_index=True)
    else:
        st.info("The explanation service returned no ranked reasons for this transaction.")

    related = row.get("related_transactions", [])
    st.subheader("Related historical transactions")
    if related:
        safe_related = [
            {
                "transaction_id": item.get("transaction_id"),
                "timestamp": item.get("timestamp"),
                "relationship": item.get("relationship"),
                "fraud_status": item.get("fraud_status"),
                "hops": item.get("hops"),
            }
            for item in related
        ]
        st.dataframe(pd.DataFrame(safe_related), use_container_width=True, hide_index=True)
    render_transaction_graph(selected, related)
