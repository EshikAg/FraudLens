"""Drift-triggered replay sampling, fine-tuning, and guarded promotion."""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from adaptation.incremental_trainer import IncrementalTrainer
from adaptation.model_registry import ModelRegistry
from adaptation.promotion_manager import PromotionManager
from adaptation.replay_buffer import ReplayBuffer
from models.banking_gaap import BankingGAAP


class AdaptationManager:
    """Adapt only after drift and only promote a candidate that passes all gates."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        registry: ModelRegistry,
        trainer: IncrementalTrainer,
        seed: int = 42,
    ) -> None:
        self.config = dict(config)
        self.seed = int(seed)
        buffer_settings = dict(self.config.get("buffers", {}))
        capacity = int(buffer_settings.get("capacity", 5000))
        self.half_life_days = float(buffer_settings.get("half_life_days", 180.0))
        self.minimum_temporal_weight = float(
            buffer_settings.get("minimum_temporal_weight", 0.1)
        )
        self.recent_buffer = ReplayBuffer(
            capacity,
            seed=self.seed,
            half_life_days=float(buffer_settings.get("recent_half_life_days", 30.0)),
            minimum_weight=self.minimum_temporal_weight,
        )
        self.replay_buffer = ReplayBuffer(
            capacity,
            seed=self.seed + 1,
            half_life_days=self.half_life_days,
            minimum_weight=self.minimum_temporal_weight,
        )
        self.registry = registry
        self.trainer = trainer
        self.promotion_manager = PromotionManager(self.config.get("promotion", {}))
        replay_ratio = float(self.config.get("replay_ratio", 0.3))
        recent_ratio = float(self.config.get("recent_ratio", 0.7))
        if not math.isclose(replay_ratio + recent_ratio, 1.0, abs_tol=1e-8):
            raise ValueError("recent_ratio and replay_ratio must sum to one")
        if min(replay_ratio, recent_ratio) <= 0:
            raise ValueError("recent_ratio and replay_ratio must both be positive")
        self.replay_ratio = replay_ratio
        self.recent_ratio = recent_ratio

    def seed_historical_buffer(self, transactions: pd.DataFrame) -> int:
        """Initialize retained historical examples from the fixed baseline run."""
        return self.replay_buffer.add(transactions)

    def add_recent(self, transactions: pd.DataFrame) -> int:
        """Add newly labeled streaming rows to the bounded recent buffer."""
        return self.recent_buffer.add(transactions)

    def adapt(
        self,
        *,
        current_model: BankingGAAP,
        baseline_model: BankingGAAP,
        recent_transactions: pd.DataFrame | None,
        validation_transactions: pd.DataFrame,
        drift_event: Mapping[str, Any],
        feature_schema_version: str,
    ) -> dict[str, Any]:
        """Warm-start GAAP on a recent/replay mixture and gate model promotion."""
        if not drift_event.get("drift_detected", False):
            return {
                "adaptation_triggered": False,
                "promoted": False,
                "reason": "no_drift_detected",
                "current_version": self.registry.current_version,
            }
        if self.registry.current_version is None:
            raise RuntimeError("Register an initial model before adapting")
        if recent_transactions is not None:
            self.add_recent(recent_transactions)
        recent_all = self.recent_buffer.frame()
        historical_all = self.replay_buffer.frame()
        if recent_all.empty or historical_all.empty:
            raise ValueError("Adaptation requires recent and historical labeled examples")
        validation_ids = set(validation_transactions["transaction_id"].astype(str))
        buffered_ids = set(recent_all["transaction_id"].astype(str)) | set(
            historical_all["transaction_id"].astype(str)
        )
        overlap = validation_ids & buffered_ids
        if overlap:
            raise ValueError(
                "Adaptation validation IDs must be disjoint from replay buffers: "
                f"{sorted(overlap)[:5]}"
            )

        fraction = float(self.config.get("evaluation_fraction", 0.2))
        if not 0 < fraction < 0.5:
            raise ValueError("evaluation_fraction must be between 0 and 0.5")
        recent_eval_count = max(2, int(len(recent_all) * fraction))
        replay_eval_count = max(2, int(len(historical_all) * fraction))
        if len(recent_all) <= recent_eval_count + 1:
            raise ValueError("Recent buffer is too small for training and evaluation")
        if len(historical_all) <= replay_eval_count + 1:
            raise ValueError("Replay buffer is too small for training and evaluation")
        recent_all = recent_all.sort_values(
            ["timestamp", "transaction_id"], kind="stable"
        ).reset_index(drop=True)
        historical_all = historical_all.sort_values(
            ["timestamp", "transaction_id"], kind="stable"
        ).reset_index(drop=True)
        recent_training_pool = recent_all.iloc[:-recent_eval_count]
        recent_window = recent_all.iloc[-recent_eval_count:]
        replay_training_pool = historical_all.iloc[:-replay_eval_count]
        replay_window = historical_all.iloc[-replay_eval_count:]

        sample_size = int(self.config.get("samples_per_adaptation", 500))
        if sample_size < 2:
            raise ValueError("samples_per_adaptation must be at least two")
        effective_sample_size = min(
            sample_size,
            int(len(recent_training_pool) / self.recent_ratio),
            int(len(replay_training_pool) / self.replay_ratio),
        )
        if effective_sample_size < 2:
            raise ValueError(
                "Recent and replay windows are too small for the configured mix"
            )
        recent_count = int(round(effective_sample_size * self.recent_ratio))
        replay_count = effective_sample_size - recent_count
        if min(recent_count, replay_count) < 1:
            raise ValueError(
                "Configured replay ratio cannot allocate at least one sample per source"
            )
        as_of = pd.to_datetime(recent_window["timestamp"], utc=True).max()
        recent_training = self._sample_recent(
            recent_training_pool, recent_count, as_of
        )
        replay_subset_buffer = ReplayBuffer(
            max(1, len(replay_training_pool)),
            seed=self.seed + len(self.registry.list_versions()) + 2,
            half_life_days=self.half_life_days,
            minimum_weight=self.minimum_temporal_weight,
        )
        replay_subset_buffer.add(replay_training_pool)
        replay_training = replay_subset_buffer.sample(
            replay_count,
            as_of=as_of,
            exclude_ids=(
                set(recent_training["transaction_id"].astype(str))
                | set(validation_transactions["transaction_id"].astype(str))
            ),
        )
        if recent_training.empty or replay_training.empty:
            raise ValueError("Configured replay ratio yielded an empty training group")

        version = self.registry.next_version()
        candidate_dir = self.registry.directory / "candidates" / version
        current_record = self.registry.current
        if current_record is None:
            raise RuntimeError("Registry current model metadata is missing")
        result = self.trainer.train(
            current_model=current_model,
            baseline_model=baseline_model,
            recent_training=recent_training,
            replay_training=replay_training,
            validation=validation_transactions,
            recent_window=recent_window,
            replay_window=replay_window,
            initial_checkpoint=self.registry.checkpoint_path(),
            output_dir=candidate_dir,
            model_version=version,
        )
        candidate_model = result["candidate_model"]
        model_directory = candidate_model.save(candidate_dir / "model")
        metrics = {
            "current": result["current_metrics"],
            "candidate": result["candidate_metrics"],
            "baseline": result["baseline_metrics"],
        }
        record = self.registry.register(
            version=version,
            checkpoint_path=model_directory / "gaap_state.ckpt",
            training_data_window=result["training_data_window"],
            drift_event=drift_event,
            metrics=metrics,
            feature_schema_version=feature_schema_version,
            metadata={
                "model_directory": str(model_directory),
                "feature_alignment": result["feature_alignment"],
            },
        )
        decision = self.promotion_manager.evaluate(
            result["current_metrics"], result["candidate_metrics"]
        )
        if decision["promote"]:
            self.registry.promote(version)
        else:
            reason = "promotion criteria failed: " + ", ".join(
                decision["failed_windows"]
            )
            self.registry.reject(version, reason)
        record = self.registry.get(version)
        if record is None:
            raise RuntimeError(f"Registered candidate {version} disappeared")
        return {
            "adaptation_triggered": True,
            "promoted": decision["promote"],
            "version": version,
            "current_version": self.registry.current_version,
            "candidate_model": candidate_model,
            "candidate_record": record,
            "promotion_decision": decision,
            "metrics": metrics,
            "feature_alignment": result["feature_alignment"],
            "training_data_window": result["training_data_window"],
            "drift_event": dict(drift_event),
        }

    def _sample_recent(
        self,
        pool: pd.DataFrame,
        count: int,
        as_of: Any,
    ) -> pd.DataFrame:
        buffer = ReplayBuffer(
            max(1, len(pool)),
            seed=self.seed + len(self.registry.list_versions()),
            half_life_days=float(
                self.config.get("buffers", {}).get("recent_half_life_days", 30.0)
            ),
            minimum_weight=self.minimum_temporal_weight,
        )
        buffer.add(pool)
        return buffer.sample(count, as_of=as_of)
