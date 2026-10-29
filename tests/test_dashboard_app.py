"""Streamlit entrypoint smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest


def test_dashboard_starts_without_a_loaded_model() -> None:
    script = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    app = AppTest.from_file(script, default_timeout=45).run()

    assert not app.exception
    assert any(element.value == "Fraud operations overview" for element in app.title)
    assert any(element.label == "Generate and score demo" for element in app.button)
    for page in (
        "Transaction monitor",
        "Transaction investigation",
        "Drift monitor",
        "Model monitor",
    ):
        app.radio[0].set_value(page).run()
        assert not app.exception


def test_model_monitor_renders_registry_and_adaptation_comparison(
    tmp_path: Path, monkeypatch
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "current_version": "v2",
                "versions": [
                    {
                        "version": "v1",
                        "timestamp": "2026-10-01T00:00:00Z",
                        "training_data_window": {
                            "start": "2025-01-01",
                            "end": "2025-02-01",
                            "rows": 100,
                        },
                        "metrics": {
                            "validation": {
                                "average_precision": 0.7,
                                "recall": 0.6,
                                "precision": 0.8,
                                "f1": 0.68,
                            }
                        },
                        "drift_event": None,
                        "status": "previous",
                    },
                    {
                        "version": "v2",
                        "timestamp": "2026-10-02T00:00:00Z",
                        "training_data_window": {"rows": 20},
                        "metrics": {
                            "validation": {
                                "average_precision": 0.8,
                                "recall": 0.7,
                                "precision": 0.82,
                                "f1": 0.75,
                            }
                        },
                        "drift_event": {"drift_type": "concept"},
                        "status": "promoted",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    adaptation = tmp_path / "adaptation.json"
    metrics = {
        "average_precision": 0.8,
        "recall": 0.7,
        "precision": 0.82,
        "f1": 0.75,
    }
    adaptation.write_text(
        json.dumps(
            {
                "models": {
                    "old_gaap_fixed_baseline": {"metrics": metrics},
                    "full_retrained_gaap": {"metrics": metrics},
                    "adaptive_gaap": {"metrics": metrics},
                }
            }
        ),
        encoding="utf-8",
    )
    dashboard_config = tmp_path / "dashboard.yaml"
    dashboard_config.write_text(
        yaml.safe_dump(
            {
                "api_base_url": "http://127.0.0.1:1",
                "request_timeout_seconds": 0.1,
                "prediction_timeout_seconds": 0.1,
                "artifacts": {
                    "model_registry": str(registry),
                    "adaptation_report": str(adaptation),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FRAUDLENS_DASHBOARD_CONFIG", str(dashboard_config))
    script = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    app = AppTest.from_file(script, default_timeout=45).run()
    app.radio[0].set_value("Model monitor").run()

    assert not app.exception
    assert len(app.dataframe) == 1
    assert list(app.dataframe[0].value["version"]) == ["v1", "v2"]
    assert len(app.get("arrow_vega_lite_chart")) == 1
