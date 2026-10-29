"""Streamlit app for Adaptive-GAAP transaction investigation and monitoring."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from dashboard.pages import (
    drift_monitor,
    investigation,
    model_monitor,
    overview,
    transaction_monitor,
)
from dashboard.services.api_client import DashboardAPIError, FraudLensAPIClient
from dashboard.services.artifact_service import ArtifactService
from dashboard.services.demo_service import build_demo_transactions, score_demo_transactions

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
PAGE_NAMES = (
    "Overview",
    "Transaction monitor",
    "Transaction investigation",
    "Drift monitor",
    "Model monitor",
)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Dashboard configuration must be a mapping: {config_path}")
    return config


@st.cache_resource
def _api_client(
    base_url: str, timeout: float, prediction_timeout: float
) -> FraudLensAPIClient:
    return FraudLensAPIClient(base_url, timeout, prediction_timeout)


def main() -> None:
    st.set_page_config(
        page_title="Adaptive-GAAP FraudLens",
        page_icon="🔎",
        layout="wide",
    )
    config = load_config(os.environ.get("FRAUDLENS_DASHBOARD_CONFIG", DEFAULT_CONFIG))
    api = _api_client(
        os.environ.get("FRAUDLENS_API_URL", config["api_base_url"]),
        float(config.get("request_timeout_seconds", 5)),
        float(config.get("prediction_timeout_seconds", 120)),
    )
    st.sidebar.title("FraudLens")
    st.sidebar.caption("Adaptive-GAAP investigation console")
    page = st.sidebar.radio("Workspace", PAGE_NAMES)

    api_status, model, drift = _load_live_status(api)
    st.sidebar.markdown(f"**API:** {api_status}")
    if model and model.get("model_loaded"):
        st.sidebar.markdown(f"**Model:** {model.get('model_version', 'unknown')}")
    else:
        st.sidebar.warning(
            "Model unavailable. Start the API with a configured saved model."
        )

    artifacts = ArtifactService(config.get("artifacts", {}))
    drift_report = artifacts.read_json("drift_report")
    adaptation_report = artifacts.read_json("adaptation_report")
    versions, active_version = artifacts.model_versions()
    training_report = artifacts.read_json("training_report")
    if not versions and training_report:
        versions = [_training_report_version(training_report, model)]
        active_version = versions[0].get("version")

    transactions = st.session_state.get("scored_transactions")
    if not isinstance(transactions, pd.DataFrame):
        transactions = pd.DataFrame()

    st.sidebar.subheader("Synthetic demo")
    demo_config = config.get("demo", {})
    demo_count = st.sidebar.number_input(
        "Transactions to score",
        min_value=1,
        max_value=100,
        value=int(demo_config.get("rows", 24)),
        step=1,
    )
    if st.sidebar.button("Generate and score demo", type="primary"):
        try:
            with st.spinner("Scoring demo transactions through the API..."):
                demo = build_demo_transactions(
                    row_count=int(demo_count),
                    seed=int(demo_config.get("seed", 42)),
                )
                scored = score_demo_transactions(demo, api)
                st.session_state["scored_transactions"] = scored
                transactions = scored
            st.sidebar.success(f"Scored {len(scored)} demo transactions.")
        except (DashboardAPIError, ValueError) as exc:
            st.sidebar.error(str(exc))

    if page == "Overview":
        overview.render(transactions, model, drift)
    elif page == "Transaction monitor":
        transaction_monitor.render(transactions, drift)
    elif page == "Transaction investigation":
        investigation.render(transactions)
    elif page == "Drift monitor":
        drift_monitor.render(drift, drift_report, adaptation_report, model)
    else:
        model_monitor.render(versions, active_version, adaptation_report, model)

    if api_status != "connected":
        st.warning(
            "The dashboard does not fabricate model output. Start the API and "
            "configure a trained checkpoint before requesting predictions."
        )


def _load_live_status(
    api: FraudLensAPIClient,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    try:
        health = api.health()
        if not health.get("model_loaded"):
            return "model unavailable", None, None
        return "connected", api.model(), api.drift()
    except DashboardAPIError as exc:
        return str(exc), None, None


def _training_report_version(
    report: dict[str, Any], model: dict[str, Any] | None
) -> dict[str, Any]:
    """Adapt the existing trainer report into one read-only lineage row."""
    return {
        "version": report.get("model_version")
        or (model or {}).get("model_version", "current"),
        "timestamp": None,
        "training_data_window": None,
        "metrics": {"test": report.get("test_metrics")},
        "drift_event": None,
        "status": "active",
    }


if __name__ == "__main__":
    main()
