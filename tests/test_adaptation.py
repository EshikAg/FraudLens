"""Tests for drift-gated adaptation and multi-window promotion decisions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from adaptation.adaptation_manager import AdaptationManager
from adaptation.model_registry import ModelRegistry
from adaptation.promotion_manager import PromotionManager


def _rows(prefix: str, count: int, start: str) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=count, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "transaction_id": [f"{prefix}-{index}" for index in range(count)],
            "timestamp": timestamps,
            "label": [index % 2 for index in range(count)],
            "amount": [float(index + 1) for index in range(count)],
            "transaction_type": ["UPI", "IMPS"] * (count // 2),
        }
    )


class _CandidateModel:
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "gaap_state.ckpt"
        checkpoint.write_bytes(b"candidate")
        return directory


class _FakeTrainer:
    def __init__(self, replay_f1: float = 0.8) -> None:
        self.replay_f1 = replay_f1
        self.arguments = None

    def train(self, **kwargs):
        self.arguments = kwargs
        candidate = {
            "validation": {"f1": 0.79},
            "recent": {"f1": 0.85},
            "replay": {"f1": self.replay_f1},
        }
        current = {
            "validation": {"f1": 0.8},
            "recent": {"f1": 0.7},
            "replay": {"f1": 0.8},
        }
        return {
            "candidate_model": _CandidateModel(),
            "candidate_metrics": candidate,
            "current_metrics": current,
            "baseline_metrics": current,
            "training_data_window": {"rows": 10, "start": "2025-02-01", "end": "2025-02-02"},
            "feature_alignment": {
                "model_feature_width": 4,
                "observed_feature_width": 4,
                "unmapped_feature_columns": [],
                "missing_expected_feature_columns": [],
            },
        }


def _manager(tmp_path, trainer):
    registry = ModelRegistry(tmp_path / "registry")
    initial_checkpoint = tmp_path / "initial.ckpt"
    initial_checkpoint.write_bytes(b"initial")
    registry.register(
        version="v1",
        checkpoint_path=initial_checkpoint,
        training_data_window={"rows": 50},
        drift_event=None,
        metrics={"f1": 0.8},
        feature_schema_version="banking-v1",
        metadata={"role": "fixed_baseline"},
    )
    registry.promote("v1")
    manager = AdaptationManager(
        {
            "recent_ratio": 0.7,
            "replay_ratio": 0.3,
            "samples_per_adaptation": 10,
            "evaluation_fraction": 0.2,
            "buffers": {
                "capacity": 100,
                "half_life_days": 365,
                "minimum_temporal_weight": 0.1,
            },
            "promotion": {
                "metric": "f1",
                "minimum_recent_improvement": 0.0,
                "maximum_validation_drop": 0.03,
                "maximum_replay_drop": 0.03,
            },
        },
        registry=registry,
        trainer=trainer,
    )
    manager.seed_historical_buffer(_rows("history", 40, "2025-01-01"))
    return manager, registry


def test_no_drift_does_not_trigger_training(tmp_path) -> None:
    trainer = _FakeTrainer()
    manager, registry = _manager(tmp_path, trainer)
    result = manager.adapt(
        current_model=object(),
        baseline_model=object(),
        recent_transactions=_rows("recent", 40, "2025-02-01"),
        validation_transactions=_rows("validation", 10, "2025-03-01"),
        drift_event={"drift_detected": False},
        feature_schema_version="banking-v1",
    )

    assert not result["adaptation_triggered"]
    assert trainer.arguments is None
    assert registry.current_version == "v1"


def test_detected_drift_fine_tunes_replay_mixture_and_promotes(tmp_path) -> None:
    trainer = _FakeTrainer()
    manager, registry = _manager(tmp_path, trainer)
    result = manager.adapt(
        current_model=object(),
        baseline_model=object(),
        recent_transactions=_rows("recent", 40, "2025-02-01"),
        validation_transactions=_rows("validation", 10, "2025-03-01"),
        drift_event={
            "drift_detected": True,
            "drift_type": "concept",
            "timestamp": "2025-02-02T00:00:00Z",
        },
        feature_schema_version="banking-v1",
    )

    assert result["promoted"]
    assert result["version"] == "v2"
    assert registry.current_version == "v2"
    assert len(trainer.arguments["recent_training"]) > len(trainer.arguments["replay_training"])
    assert registry.get("v2")["drift_event"]["drift_type"] == "concept"


def test_replay_forgetting_prevents_candidate_promotion(tmp_path) -> None:
    trainer = _FakeTrainer(replay_f1=0.6)
    manager, registry = _manager(tmp_path, trainer)
    result = manager.adapt(
        current_model=object(),
        baseline_model=object(),
        recent_transactions=_rows("recent", 40, "2025-02-01"),
        validation_transactions=_rows("validation", 10, "2025-03-01"),
        drift_event={"drift_detected": True, "drift_type": "data"},
        feature_schema_version="banking-v1",
    )

    assert not result["promoted"]
    assert result["promotion_decision"]["replay_retention_passed"] is False
    assert registry.current_version == "v1"
    assert registry.get("v2")["status"] == "rejected"


def test_promotion_requires_recent_gain_and_all_retention_windows() -> None:
    manager = PromotionManager(
        {
            "metric": "f1",
            "minimum_recent_improvement": 0.02,
            "maximum_validation_drop": 0.01,
            "maximum_replay_drop": 0.02,
        }
    )
    current = {
        "validation": {"f1": 0.8},
        "recent": {"f1": 0.7},
        "replay": {"f1": 0.75},
    }
    candidate = {
        "validation": {"f1": 0.8},
        "recent": {"f1": 0.71},
        "replay": {"f1": 0.72},
    }

    result = manager.evaluate(current, candidate)

    assert not result["promote"]
    assert set(result["failed_windows"]) == {"recent", "replay"}
