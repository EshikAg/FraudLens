"""Tests for model version registration, promotion, and rollback."""

from __future__ import annotations

import json

import pytest

from adaptation.model_registry import ModelRegistry


def _checkpoint(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test checkpoint")
    return path


def test_registry_versions_promote_reload_and_rollback(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    v1 = _checkpoint(tmp_path / "v1.ckpt")
    v2 = _checkpoint(tmp_path / "v2.ckpt")
    record = registry.register(
        version="v1",
        checkpoint_path=v1,
        training_data_window={"rows": 80, "start": "2025-01-01", "end": "2025-01-02"},
        drift_event=None,
        metrics={"f1": 0.8},
        feature_schema_version="banking-v1",
    )
    assert record["status"] == "candidate"
    registry.promote("v1")
    registry.register(
        version="v2",
        checkpoint_path=v2,
        training_data_window={"rows": 20},
        drift_event={"drift_detected": True, "drift_type": "concept"},
        metrics={"f1": 0.85},
        feature_schema_version="banking-v1",
    )
    registry.promote("v2")

    restored = ModelRegistry(tmp_path / "registry")
    assert restored.current_version == "v2"
    assert restored.checkpoint_path().read_bytes() == b"test checkpoint"
    assert restored.rollback()["version"] == "v1"
    assert restored.current_version == "v1"
    saved = json.loads((tmp_path / "registry" / "registry.json").read_text())
    assert saved["versions"][1]["status"] == "rolled_back"


def test_registry_rejects_missing_checkpoints_and_first_rollback(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    with pytest.raises(FileNotFoundError):
        registry.register(
            version="v1",
            checkpoint_path=tmp_path / "missing.ckpt",
            training_data_window={},
            drift_event=None,
            metrics={},
            feature_schema_version="banking-v1",
        )
    checkpoint = _checkpoint(tmp_path / "only.ckpt")
    registry.register(
        version="v1",
        checkpoint_path=checkpoint,
        training_data_window={},
        drift_event=None,
        metrics={},
        feature_schema_version="banking-v1",
    )
    registry.promote("v1")
    with pytest.raises(RuntimeError, match="No previous"):
        registry.rollback()
