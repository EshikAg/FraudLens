"""Drift-triggered Adaptive-GAAP orchestration around the original GAAP model."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from adaptation.adaptation_manager import AdaptationManager
from adaptation.incremental_trainer import IncrementalTrainer
from adaptation.model_registry import ModelRegistry
from explainability.explanation_service import ExplanationService
from models.banking_gaap import BankingGAAP, _resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AdaptiveGAAP:
    """Keep a fixed GAAP baseline and promote only validated fine-tuned versions."""

    def __init__(self, config: Mapping[str, Any] | str | Path) -> None:
        if isinstance(config, (str, Path)):
            self.config_path = _resolve_path(config)
            with self.config_path.open("r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream)
            if not isinstance(loaded, dict):
                raise ValueError(f"Adaptive-GAAP config must be a mapping: {self.config_path}")
            self.config = loaded
        else:
            self.config_path = None
            self.config = copy.deepcopy(dict(config))
        required = ("gaap_config", "adaptation", "registry_dir")
        missing = [name for name in required if name not in self.config]
        if missing:
            raise ValueError(f"Adaptive-GAAP config is missing sections: {missing}")
        self.seed = int(self.config.get("seed", 42))
        self.feature_schema_version = str(
            self.config.get("feature_schema_version", "banking-transaction-v1")
        )
        self.registry = ModelRegistry(_resolve_path(self.config["registry_dir"]))
        base_model = BankingGAAP(self.config["gaap_config"])
        self.gaap_config = copy.deepcopy(base_model.config)
        self.incremental_trainer = IncrementalTrainer(
            self.gaap_config,
            self.config.get("incremental_training", {}),
            seed=self.seed,
        )
        self.adaptation_manager = AdaptationManager(
            self.config["adaptation"],
            registry=self.registry,
            trainer=self.incremental_trainer,
            seed=self.seed,
        )
        explainability_config = self.config.get(
            "explainability_config", "configs/explainability_config.yaml"
        )
        self.explanation_service = ExplanationService(
            _resolve_path(explainability_config)
            if isinstance(explainability_config, (str, Path))
            else explainability_config
        )
        self.baseline_model: BankingGAAP | None = None
        self.current_model: BankingGAAP | None = None

    def fit(
        self, transactions: pd.DataFrame | str | Path | None = None
    ) -> AdaptiveGAAP:
        """Train and register immutable v1; subsequent versions must be adapted."""
        if self.registry.list_versions():
            raise RuntimeError(
                "Adaptive-GAAP registry is already initialized; load it or use a new registry"
            )
        baseline_config = copy.deepcopy(self.gaap_config)
        version_directory = self.registry.directory / "versions" / "v1"
        baseline_config["output_dir"] = str(version_directory / "training")
        baseline_config["model_version"] = "v1"
        baseline = BankingGAAP(baseline_config)
        source = transactions if transactions is not None else baseline._configured_data()
        canonical = baseline._binary_labels(baseline._read_transactions(source))
        baseline.fit(canonical)
        model_directory = baseline.save(version_directory / "model")
        self.adaptation_manager.seed_historical_buffer(canonical)
        training_window = _window(canonical)
        self.registry.register(
            version="v1",
            checkpoint_path=model_directory / "gaap_state.ckpt",
            training_data_window=training_window,
            drift_event=None,
            metrics={"validation": baseline.evaluate("validation"), "test": baseline.evaluate("test")},
            feature_schema_version=self.feature_schema_version,
            metadata={
                "role": "fixed_baseline",
                "model_directory": str(model_directory),
            },
        )
        self.registry.promote("v1")
        self.baseline_model = baseline
        self.current_model = baseline
        return self

    def add_recent(self, transactions: pd.DataFrame) -> int:
        """Retain newly labeled transactions until an adaptation event occurs."""
        return self.adaptation_manager.add_recent(transactions)

    def adapt(
        self,
        *,
        recent_transactions: pd.DataFrame | None,
        validation_transactions: pd.DataFrame,
        drift_event: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Fine-tune current weights after drift and conditionally promote."""
        if self.current_model is None or self.baseline_model is None:
            raise RuntimeError("AdaptiveGAAP.fit or AdaptiveGAAP.load must run first")
        result = self.adaptation_manager.adapt(
            current_model=self.current_model,
            baseline_model=self.baseline_model,
            recent_transactions=recent_transactions,
            validation_transactions=validation_transactions,
            drift_event=drift_event,
            feature_schema_version=self.feature_schema_version,
        )
        if result.get("promoted"):
            self.current_model = result["candidate_model"]
        result.pop("candidate_model", None)
        return result

    def predict(self) -> pd.DataFrame:
        """Return transaction predictions from the currently promoted version."""
        if self.current_model is None:
            raise RuntimeError("AdaptiveGAAP has not been fitted or loaded")
        return self.current_model.predict()

    def explain(self, transaction_id: str) -> dict[str, Any]:
        """Explain one transaction scored by the currently promoted GAAP model."""
        if self.current_model is None:
            raise RuntimeError("AdaptiveGAAP has not been fitted or loaded")
        return self.explanation_service.explain(self.current_model, transaction_id)

    def explain_all(self) -> list[dict[str, Any]]:
        """Generate an explanation for every transaction in the current graph."""
        if self.current_model is None:
            raise RuntimeError("AdaptiveGAAP has not been fitted or loaded")
        return self.explanation_service.explain_all(self.current_model)

    def evaluate(self, split: str = "test") -> dict[str, float]:
        """Evaluate the currently promoted model on its saved temporal split."""
        if self.current_model is None:
            raise RuntimeError("AdaptiveGAAP has not been fitted or loaded")
        return self.current_model.evaluate(split)

    def rollback(self) -> dict[str, Any]:
        """Restore the preceding promoted checkpoint and its saved graph."""
        record = self.registry.rollback()
        directory = record.get("metadata", {}).get("model_directory")
        if not directory:
            raise ValueError(f"Registry version {record['version']} has no model directory")
        restored = BankingGAAP.load(directory)
        self.current_model = restored
        return record

    def save(self, directory: str | Path) -> Path:
        """Save registry state and replay buffers for later restoration."""
        destination = _resolve_path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "adaptive_gaap.json").write_text(
            json.dumps(
                {
                    "config": self.config,
                    "registry_dir": str(self.registry.directory),
                    "current_version": self.registry.current_version,
                    "feature_schema_version": self.feature_schema_version,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        self.adaptation_manager.recent_buffer.frame().to_csv(
            destination / "recent_buffer.csv", index=False
        )
        self.adaptation_manager.replay_buffer.frame().to_csv(
            destination / "replay_buffer.csv", index=False
        )
        return destination

    @classmethod
    def load(cls, directory: str | Path) -> AdaptiveGAAP:
        """Restore the registered fixed baseline, current model, and replay rows."""
        source = _resolve_path(directory)
        metadata_path = source / "adaptive_gaap.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Adaptive-GAAP metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata["config"]
        config["registry_dir"] = metadata["registry_dir"]
        instance = cls(config)
        if metadata.get("current_version") != instance.registry.current_version:
            raise ValueError("Saved Adaptive-GAAP current version does not match its registry")
        baseline_record = instance.registry.get("v1")
        current_record = instance.registry.current
        if baseline_record is None or current_record is None:
            raise ValueError("Adaptive-GAAP registry has no baseline or current version")
        baseline_directory = baseline_record.get("metadata", {}).get("model_directory")
        current_directory = current_record.get("metadata", {}).get("model_directory")
        if not baseline_directory or not current_directory:
            raise ValueError("Adaptive-GAAP registry is missing model directories")
        instance.baseline_model = BankingGAAP.load(baseline_directory)
        instance.current_model = BankingGAAP.load(current_directory)
        for filename, buffer in (
            ("recent_buffer.csv", instance.adaptation_manager.recent_buffer),
            ("replay_buffer.csv", instance.adaptation_manager.replay_buffer),
        ):
            path = source / filename
            if path.is_file():
                frame = pd.read_csv(path)
                if not frame.empty:
                    buffer.add(frame)
        return instance


def _window(transactions: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(transactions["timestamp"], utc=True)
    return {
        "rows": int(len(transactions)),
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
    }
