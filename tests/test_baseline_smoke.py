"""Tests for the baseline wrapper without running a full graph training job."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from scripts import run_baseline


class FakeModel:
    def __init__(self) -> None:
        self.loaded_state: dict[str, Any] | None = None

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loaded_state = state

    def to(self, device: str) -> FakeModel:
        assert device == "cpu"
        return self


class FakeExperiment:
    def __init__(self, model_name: str, data_name: str, config: dict[str, Any]):
        assert model_name == "SAGE"
        assert data_name == "tolokers"
        assert config["max_epochs"] == 1
        checkpoint_path = Path(config["fake_checkpoint"])
        self.trainer = SimpleNamespace(
            checkpoint_callback=SimpleNamespace(best_model_path=str(checkpoint_path)),
            fit_loop=SimpleNamespace(
                epoch_progress=SimpleNamespace(
                    current=SimpleNamespace(completed=1)
                )
            ),
            test=lambda model, dataloader, verbose: [
                {
                    "f/val_pre": 0.8,
                    "f/val_rec": 0.7,
                    "f/val_mf1": 0.75,
                    "f/val_auc": 0.9,
                    "f/val_aps": 0.85,
                    "f/tst_pre": 0.6,
                    "f/tst_rec": 0.5,
                    "f/tst_mf1": 0.55,
                    "f/tst_auc": 0.8,
                    "f/tst_aps": 0.65,
                }
            ],
        )
        self.dm = SimpleNamespace(val_dataloader=lambda: "validation-loader")
        self.model = FakeModel()
        self.trained = False

    def train(self) -> None:
        self.trained = True


class FakeWandb:
    def __init__(self) -> None:
        self.initialized = False
        self.finished = False

    def init(self, **kwargs: Any) -> None:
        assert kwargs["mode"] == "offline"
        self.initialized = True

    def finish(self, **kwargs: Any) -> None:
        assert kwargs["quiet"] is True
        self.finished = True


def _write_config(tmp_path: Path, checkpoint_path: Path) -> Path:
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        f"""
model_name: SAGE
loader_type: MiniF
data_name: tolokers
seed: 21
max_epochs: 22
patience: 3
device: cuda
fake_checkpoint: {checkpoint_path.as_posix()}
baseline:
  output_dir: run
  wandb_mode: offline
""",
        encoding="utf-8",
    )
    return config_path


def test_baseline_smoke_writes_metrics_and_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"test checkpoint")
    config_path = _write_config(tmp_path, checkpoint)
    output_dir = tmp_path / "baseline-output"

    monkeypatch.setattr(
        run_baseline,
        "_load_checkpoint",
        lambda path: {"state_dict": {"weight": 1}},
    )
    fake_gaap = ModuleType("fake_gaap")
    fake_gaap.Experiment = FakeExperiment
    fake_wandb = FakeWandb()

    report = run_baseline.run_baseline(
        config_path,
        output_dir=output_dir,
        smoke=True,
        gaap_module=fake_gaap,
        wandb_module=fake_wandb,
    )

    saved_report = json.loads(
        (output_dir / "baseline_metrics.json").read_text(encoding="utf-8")
    )
    saved_config = json.loads(
        (output_dir / "baseline_config.json").read_text(encoding="utf-8")
    )
    assert report == saved_report
    assert report["epochs"] == {"configured_max": 1, "completed": 1}
    assert report["validation_metrics"]["f1"] == 0.75
    assert report["test_metrics"]["average_precision"] == 0.65
    assert report["checkpoint_path"] == str(checkpoint.resolve())
    assert saved_config["gaap"]["device"] == "cpu"
    assert saved_config["gaap"]["max_epochs"] == 1
    assert fake_wandb.initialized and fake_wandb.finished


def test_baseline_rejects_missing_metrics() -> None:
    with pytest.raises(KeyError, match="did not return val pre"):
        run_baseline._split_metrics({"f/val_auc": 0.8}, "val")
