"""Run the synthetic drift/adaptation workflow and score transactions through the API."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import create_app  # noqa: E402
from api.services.prediction_service import PredictionService  # noqa: E402
from models.banking_gaap import BankingGAAP  # noqa: E402
from preprocessing.synthetic_data import generate_synthetic_transactions  # noqa: E402
from scripts.run_adaptation_experiment import (  # noqa: E402
    _load_config as _load_adaptation_config,
    run_experiment as run_adaptation_experiment,
)


LOGGER = logging.getLogger("adaptive_gaap.demo")


def _load_yaml(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {resolved}")
    return config


def _api_payload(row: pd.Series, transaction_id: str, timestamp: pd.Timestamp) -> dict[str, Any]:
    """Select only supported request fields, excluding labels and balances."""
    fields = (
        "sender_account_id",
        "receiver_account_id",
        "amount",
        "currency",
        "transaction_type",
        "channel",
        "merchant_id",
        "device_id",
        "ip_id",
        "location",
    )
    payload = {name: row[name] for name in fields}
    payload.update(
        transaction_id=transaction_id,
        timestamp=timestamp.isoformat(),
    )
    return payload


def run_demo(
    output_dir: Path,
    *,
    seed: int = 42,
    api_rows: int = 1,
) -> dict[str, Any]:
    """Execute drift-triggered adaptation and exercise API prediction/label routes."""
    if api_rows < 1:
        raise ValueError("api_rows must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)

    adaptation_config = _load_adaptation_config(
        PROJECT_ROOT / "configs" / "adaptation_config.yaml"
    )
    adaptation_config["seed"] = seed
    adaptation_report = run_adaptation_experiment(
        adaptation_config,
        output_dir=output_dir / "adaptation",
        smoke=True,
    )

    gaap_config = _load_yaml(
        PROJECT_ROOT / "configs" / "adaptive_gaap_baseline.yaml"
    )
    gaap_config["seed"] = seed
    gaap_config["model"]["seed"] = seed
    gaap_config["data"]["synthetic_rows"] = max(
        132, int(gaap_config["data"].get("synthetic_rows", 132))
    )
    gaap_config["output_dir"] = str(output_dir / "api_model_training")
    training_rows = generate_synthetic_transactions(
        row_count=int(gaap_config["data"]["synthetic_rows"]),
        seed=seed,
        output_path=output_dir / "synthetic_training_transactions.csv",
    )
    model = BankingGAAP(gaap_config).fit(training_rows)
    saved_model = model.save(output_dir / "saved_api_model")

    api_config = _load_yaml(PROJECT_ROOT / "configs" / "api_config.yaml")
    api_config["model_directory"] = str(saved_model)
    api_config["drift_window_size"] = 12
    api_config["drift_config"]["feature_drift"]["features"] = [
        "amount",
        "transaction_velocity_1h",
        "transaction_type",
        "device_id",
        "ip_id",
    ]
    api_config["drift_config"]["feature_drift"]["min_samples"] = 8
    api_config["drift_config"]["concept_drift"]["min_instances"] = 5
    service = PredictionService.from_config(api_config)
    client_app = create_app(api_config, prediction_service=service)

    demo_rows = generate_synthetic_transactions(api_rows, seed=seed + 1)
    fraud_rows = demo_rows["label"].astype(int).eq(1)
    demo_rows.loc[fraud_rows, "amount"] = 35.0
    demo_rows.loc[fraud_rows, "transaction_type"] = "UPI"
    demo_rows.loc[fraud_rows, "device_id"] = "demo_new_device"
    demo_rows.loc[fraud_rows, "ip_id"] = "demo_new_network"
    demo_rows.loc[fraud_rows, "receiver_account_id"] = "demo_new_beneficiary"
    first_timestamp = (
        pd.to_datetime(training_rows["timestamp"], utc=True).max()
        + pd.Timedelta(minutes=1)
    )

    predictions: list[dict[str, Any]] = []
    with TestClient(client_app) as client:
        health = client.get("/health")
        if health.status_code != 200 or not health.json().get("model_loaded"):
            raise RuntimeError(f"Demo API did not load its trained model: {health.text}")
        for index, row in demo_rows.iterrows():
            timestamp = first_timestamp + pd.Timedelta(minutes=index)
            transaction_id = f"demo_online_{index:04d}"
            response = client.post(
                "/predict",
                json=_api_payload(row, transaction_id, timestamp),
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Synthetic API prediction failed ({response.status_code}): "
                    f"{response.text}"
                )
            prediction = response.json()
            label_response = client.post(
                f"/labels/{transaction_id}",
                json={
                    "label": int(row["label"]),
                    "timestamp": (timestamp + pd.Timedelta(minutes=1)).isoformat(),
                },
            )
            if label_response.status_code != 200:
                raise RuntimeError(
                    f"Synthetic delayed-label update failed "
                    f"({label_response.status_code}): {label_response.text}"
                )
            predictions.append(
                {
                    "transaction_id": transaction_id,
                    "timestamp": prediction["timestamp"],
                    "fraud_probability": float(prediction["fraud_probability"]),
                    "risk_level": prediction["risk_level"],
                    "decision": prediction["decision"],
                    "model_version": prediction["model_version"],
                    "drift_detected": bool(prediction["drift_detected"]),
                    "reason_codes": [
                        reason.get("code")
                        for reason in prediction["reasons"]
                        if isinstance(reason, dict)
                    ],
                }
            )
        drift_status = client.get("/drift").json()
        api_metrics = client.get("/metrics").json()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "deterministic_synthetic_banking_transactions",
        "seed": seed,
        "model": "existing GAAP SAGE with Adaptive-GAAP offline adaptation experiment",
        "saved_api_model": str(saved_model),
        "api_health": health.json(),
        "api_metrics": api_metrics,
        "drift_status": drift_status,
        "prediction_count": len(predictions),
        "prediction_records": predictions,
        "adaptation_experiment": {
            "report_path": adaptation_report.get("report_path"),
            "drift_detected": bool(
                adaptation_report.get("phases", {})
                .get("C_drift_event", {})
                .get("drift_detected")
            ),
            "promoted": bool(
                adaptation_report.get("phases", {})
                .get("D_adaptation", {})
                .get("promoted")
            ),
            "current_version": adaptation_report.get("phases", {})
            .get("D_adaptation", {})
            .get("current_version"),
        },
        "scope_note": (
            "This demo runs offline model adaptation separately from the API. "
            "The API accepts delayed labels for drift monitoring but does not "
            "hot-promote or adapt its loaded model."
        ),
    }
    manifest_path = output_dir / "demo_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote synthetic demo manifest to %s", manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--api-rows",
        type=int,
        default=1,
        help="Number of API predictions; each rebuilds the causal history graph.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New output directory; defaults to artifacts/demo/<UTC timestamp>.",
    )
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir = arguments.output_dir
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output_dir = Path("artifacts") / "demo" / timestamp
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    manifest = run_demo(
        output_dir.resolve(),
        seed=arguments.seed,
        api_rows=arguments.api_rows,
    )
    print(
        json.dumps(
            {
                "output_directory": str(output_dir.resolve()),
                "prediction_count": manifest["prediction_count"],
                "model_version": manifest["api_health"]["model_version"],
                "drift_status": manifest["drift_status"],
                "adaptation_experiment": manifest["adaptation_experiment"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
