"""Train the existing GAAP model on synthetic or user-provided banking data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.banking_gaap import BankingGAAP  # noqa: E402


LOGGER = logging.getLogger("adaptive_gaap.train")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the original GAAP SAGE model on transaction nodes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "adaptive_gaap_baseline.yaml",
    )
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _arguments()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config: dict[str, Any] = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a YAML mapping: {config_path}")
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.seed is not None:
        config["seed"] = args.seed
        config["model"]["seed"] = args.seed
    if args.input_csv is not None:
        config["data"]["source"] = "csv"
        config["data"]["csv_path"] = str(args.input_csv)
    if args.smoke:
        config.setdefault("training", {})["max_epochs"] = 1
        config["training"]["patience"] = 1

    model = BankingGAAP(config)
    model.fit()
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    model.save(output_dir / "model")

    predictions = model.predict()
    predictions_path = output_dir / "transaction_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    report = {
        "model_version": config.get("model_version", "gaap-banking-v1"),
        "dataset": config["data"]["source"],
        "transactions": len(predictions),
        "graph_statistics": model.graph.stats if model.graph is not None else {},
        "temporal_split_sizes": {
            name: len(values) for name, values in model.split_ids.items()
        },
        "test_metrics": model.evaluate("test"),
        "predictions_path": str(predictions_path),
        "saved_model_path": str(output_dir / "model"),
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved transaction predictions to %s", predictions_path)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
