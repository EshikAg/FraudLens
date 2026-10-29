"""Run the existing GAAP experiment and persist a structured baseline report."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


LOGGER = logging.getLogger("adaptive_gaap.baseline")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
GAAP_EXPERIMENT = PROJECT_ROOT / "mycode" / "exp" / "101_retrain.py"


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config and validate its GAAP and baseline sections."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Baseline config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"Baseline config must be a YAML mapping: {config_path}")
    if not isinstance(config.get("baseline"), dict):
        raise ValueError("Baseline config must define a 'baseline' mapping")
    required = ("model_name", "loader_type", "data_name", "seed", "max_epochs")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Baseline config is missing required GAAP keys: {missing}")
    if config["max_epochs"] < 1:
        raise ValueError("'max_epochs' must be at least one")
    return config


def load_gaap_experiment() -> ModuleType:
    """Load the original GAAP 101 experiment without invoking its Hydra CLI."""
    gaap_code_dir = str(PROJECT_ROOT / "mycode")
    if gaap_code_dir not in sys.path:
        sys.path.insert(0, gaap_code_dir)

    spec = importlib.util.spec_from_file_location("gaap_101_retrain", GAAP_EXPERIMENT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load GAAP experiment from {GAAP_EXPERIMENT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GAAP did not return a numeric {name} metric") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"GAAP returned a non-finite {name} metric")
    return result


def _get_metric(
    metrics: dict[str, Any], split: str, metric: str, aliases: tuple[str, ...] = ()
) -> float:
    names = (metric, *aliases)
    candidates = [
        key
        for name in names
        for key in (f"f/{split}_{name}", f"{split}_{name}")
    ]
    for candidate in candidates:
        if candidate in metrics:
            return _as_float(metrics[candidate], candidate)
    raise KeyError(
        f"GAAP evaluation did not return {split} {metric}; "
        f"looked for {', '.join(candidates)}"
    )


def _split_metrics(metrics: dict[str, Any], split: str) -> dict[str, float]:
    """Map GAAP metric names to consistent output names."""
    return {
        "precision": _get_metric(metrics, split, "pre"),
        "recall": _get_metric(metrics, split, "rec"),
        "f1": _get_metric(metrics, split, "mf1"),
        "roc_auc": _get_metric(metrics, split, "auc"),
        "average_precision": _get_metric(metrics, split, "aps"),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a GAAP Lightning checkpoint on CPU."""
    import torch

    return torch.load(path, map_location="cpu")


def _completed_epochs(trainer: Any) -> int:
    progress = getattr(getattr(trainer, "fit_loop", None), "epoch_progress", None)
    current = getattr(progress, "current", None)
    completed = getattr(current, "completed", None)
    if completed is not None:
        return int(completed)
    return int(trainer.current_epoch)


def run_baseline(
    config_path: Path = DEFAULT_CONFIG,
    *,
    output_dir: Path | None = None,
    dataset: str | None = None,
    seed: int | None = None,
    epochs: int | None = None,
    smoke: bool = False,
    gaap_module: ModuleType | None = None,
    wandb_module: Any | None = None,
) -> dict[str, Any]:
    """Train and evaluate GAAP, then persist its config and baseline metrics."""
    config_path = config_path.resolve()
    loaded_config = load_config(config_path)
    config = copy.deepcopy(loaded_config)
    baseline_config = config.pop("baseline")

    if dataset is not None:
        config["data_name"] = dataset
    if seed is not None:
        config["seed"] = seed
    if epochs is not None:
        if epochs < 1:
            raise ValueError("--epochs must be at least one")
        config["max_epochs"] = epochs
    if smoke:
        config["max_epochs"] = 1
        config["patience"] = 1

    config["device"] = "cpu"
    destination = output_dir or Path(baseline_config.get("output_dir", "artifacts/baseline"))
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    mode = str(baseline_config.get("wandb_mode", "offline"))
    if mode not in {"offline", "disabled"}:
        raise ValueError("baseline.wandb_mode must be 'offline' or 'disabled'")

    config_snapshot = {
        "source_config": str(config_path),
        "gaap": config,
        "baseline": {
            "output_dir": str(destination),
            "wandb_mode": mode,
            "smoke": smoke,
        },
    }
    config_output = destination / "baseline_config.json"
    config_output.write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    experiment_module = gaap_module or load_gaap_experiment()
    if wandb_module is None:
        import wandb as wandb_module

    run_started = False
    try:
        wandb_module.init(
            project="GAAP",
            name=f"baseline-{config['data_name']}-seed-{config['seed']}",
            config=config,
            mode=mode,
            reinit=True,
        )
        run_started = True
        experiment = experiment_module.Experiment(
            config["model_name"], config["data_name"], config
        )

        start_time = time.perf_counter()
        experiment.train()
        training_time = time.perf_counter() - start_time

        checkpoint_value = experiment.trainer.checkpoint_callback.best_model_path
        if not checkpoint_value:
            raise FileNotFoundError("GAAP training did not produce a best checkpoint")
        checkpoint_path = Path(checkpoint_value).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"GAAP best checkpoint does not exist: {checkpoint_path}"
            )

        checkpoint = _load_checkpoint(checkpoint_path)
        if "state_dict" not in checkpoint:
            raise KeyError(f"GAAP checkpoint has no 'state_dict': {checkpoint_path}")
        experiment.model.load_state_dict(checkpoint["state_dict"])
        experiment.model = experiment.model.to("cpu")

        results = experiment.trainer.test(
            experiment.model, experiment.dm.val_dataloader(), verbose=False
        )
        if not results or not isinstance(results[0], dict):
            raise RuntimeError("GAAP test evaluation returned no metric dictionary")

        metrics_path = destination / "baseline_metrics.json"
        report = {
            "dataset": config["data_name"],
            "model": config["model_name"],
            "seed": int(config["seed"]),
            "epochs": {
                "configured_max": int(config["max_epochs"]),
                "completed": _completed_epochs(experiment.trainer),
            },
            "training_time_seconds": training_time,
            "validation_metrics": _split_metrics(results[0], "val"),
            "test_metrics": _split_metrics(results[0], "tst"),
            "f1_definition": "macro (GAAP mf1)",
            "checkpoint_path": str(checkpoint_path),
            "configuration_path": str(config_output),
            "metrics_path": str(metrics_path),
        }
        metrics_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        LOGGER.info("Baseline metrics written to %s", metrics_path)
        return report
    finally:
        if run_started:
            wandb_module.finish(quiet=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the existing GAAP experiment and save baseline metrics."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one complete training epoch with early-stopping patience one.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    report = run_baseline(
        config_path=args.config,
        output_dir=args.output_dir,
        dataset=args.dataset,
        seed=args.seed,
        epochs=args.epochs,
        smoke=args.smoke,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
