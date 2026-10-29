"""Thin integration between transaction DGL graphs and the original GAAP code."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from lightning import LightningDataModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAAP_SOURCE = PROJECT_ROOT / "mycode"


def load_gaap_lit_sage() -> type:
    """Import the reusable GAAP Lightning SAGE without editing baseline files."""
    source_path = str(GAAP_SOURCE)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    try:
        module = importlib.import_module("nn.litnn")
    except ImportError as exc:
        raise ImportError(
            "Could not import GAAP LitSAGE from mycode/nn/litnn.py"
        ) from exc
    try:
        return module.LitSAGE
    except AttributeError as exc:
        raise ImportError("GAAP nn.litnn does not expose LitSAGE") from exc


def _identity(value: Any) -> Any:
    return value


class TemporalTransactionDataModule(LightningDataModule):
    """Provide one full transaction graph with explicit temporal split masks."""

    def __init__(self, graph: Any, batch_size: int = 1) -> None:
        super().__init__()
        if batch_size != 1:
            raise ValueError("Full-graph GAAP batches require batch_size=1")
        required = (
            "feature",
            "label",
            "train_mask",
            "val_mask",
            "test_mask",
        )
        missing = [name for name in required if name not in graph.ndata]
        if missing:
            raise ValueError(f"GAAP transaction graph is missing node data: {missing}")
        if graph.ndata["feature"].ndim != 2:
            raise ValueError("GAAP graph features must be a two-dimensional matrix")
        if graph.ndata["label"].dtype != torch.int64:
            raise ValueError("GAAP labels must use torch.int64 class indices")
        self.g = graph
        self.d_in = int(graph.ndata["feature"].shape[1])
        labels = graph.ndata["label"]
        labeled = labels[labels >= 0]
        if labeled.numel() == 0:
            raise ValueError("GAAP transaction graph has no labeled nodes")
        self.n_classes = int(labeled.max().item()) + 1
        self.n_nodes = graph.num_nodes()
        self.bs = batch_size
        self.val_bs = batch_size
        self.device = "cpu"
        self.use_uva = False

        for mask_name in ("train_mask", "val_mask", "test_mask"):
            mask = graph.ndata[mask_name]
            if mask.dtype != torch.bool or mask.shape != (self.n_nodes,):
                raise ValueError(f"{mask_name} must be a Boolean mask per node")
            if not mask.any():
                raise ValueError(f"{mask_name} must contain labeled nodes")
            if (labels[mask] < 0).any():
                raise ValueError(f"{mask_name} cannot include unlabeled transactions")
        mask_total = sum(
            graph.ndata[name].to(torch.int8)
            for name in ("train_mask", "val_mask", "test_mask")
        )
        if (mask_total > 1).any():
            raise ValueError("GAAP train/validation/test masks must be disjoint")

    def _loader(self) -> DataLoader:
        return DataLoader(
            [self.g],
            batch_size=None,
            collate_fn=_identity,
            num_workers=0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader()

    def val_dataloader(self) -> DataLoader:
        return self._loader()


class GAAPAdapter:
    """Train and infer with GAAP's existing ``LitSAGE`` and Lightning trainer."""

    def __init__(
        self,
        model_config: Mapping[str, Any],
        *,
        seed: int = 42,
        output_dir: str | Path = "artifacts/banking_gaap",
        max_epochs: int = 3,
        patience: int = 2,
        threshold: float = 0.5,
        model_version: str = "gaap-banking-v1",
        trainer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if patience < 1:
            raise ValueError("patience must be positive")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        if not model_version:
            raise ValueError("model_version must be non-empty")
        self.model_config = dict(model_config)
        self.seed = int(seed)
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = (PROJECT_ROOT / self.output_dir).resolve()
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.model_version = model_version
        self.trainer_kwargs = dict(trainer_kwargs or {})
        self.lit_model: Any | None = None
        self.data_module: TemporalTransactionDataModule | None = None
        self.trainer: Trainer | None = None
        self.checkpoint_path: Path | None = None

    @staticmethod
    def model_parameters(
        model_config: Mapping[str, Any],
        *,
        d_in: int,
        n_nodes: int,
    ) -> dict[str, Any]:
        """Prepare constructor kwargs required by the original GAAP SAGE."""
        required = (
            "d_hidden",
            "preprocess",
            "n_bins",
            "use_dyple",
            "dy_raw_bin_width",
            "pre_n_layers",
            "pre_use_bn",
            "pre_dropout",
            "gnn_n_layers",
            "gnn_use_bn",
            "gnn_dropout",
            "gnn_agg",
            "gnn_use_res",
            "use_mha",
            "mha_n_layers",
            "mha_n_heads",
            "lr",
        )
        missing = [name for name in required if name not in model_config]
        if missing:
            raise ValueError(f"GAAP model config is missing parameters: {missing}")
        if model_config["use_dyple"] and model_config.get("init_bin_width", False):
            raise ValueError(
                "The shared GAAP LitSAGE initializer targets a different DyPLE "
                "module; set init_bin_width=false to preserve the existing model"
            )
        params = dict(model_config)
        params.update(
            d_in=d_in,
            n_classes=2,
            n_nodes=n_nodes,
        )
        return params

    def fit(
        self,
        graph: Any,
        initial_checkpoint: str | Path | None = None,
    ) -> GAAPAdapter:
        """Fit GAAP, optionally continuing from a shape-compatible checkpoint."""
        if graph.device.type != "cpu":
            raise ValueError("GAAPAdapter currently requires a CPU DGL graph")
        self.data_module = TemporalTransactionDataModule(graph)
        self._validate_split_classes(graph)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(max(1, int(self.model_config.get("num_threads", 1))))

        lit_sage = load_gaap_lit_sage()
        model_params = self.model_parameters(
            self.model_config,
            d_in=self.data_module.d_in,
            n_nodes=self.data_module.n_nodes,
        )
        self.lit_model = lit_sage(**model_params)
        if initial_checkpoint is not None:
            checkpoint_path = Path(initial_checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Initial GAAP checkpoint does not exist: {checkpoint_path}"
                )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict")
            if not isinstance(state_dict, Mapping):
                raise KeyError(
                    f"Initial GAAP checkpoint has no state_dict: {checkpoint_path}"
                )
            try:
                self.lit_model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                raise ValueError(
                    "Initial GAAP checkpoint is incompatible with the candidate "
                    "feature/model schema"
                ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.output_dir / "checkpoints"),
            filename="gaap-{epoch:03d}-{val_aps:.5f}",
            monitor="val_aps",
            mode="max",
            save_top_k=1,
        )
        early_stopping = EarlyStopping(
            monitor="val_aps",
            mode="max",
            patience=self.patience,
        )
        trainer_options: dict[str, Any] = {
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": self.max_epochs,
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "callbacks": [checkpoint_callback, early_stopping],
            "deterministic": True,
            "num_sanity_val_steps": 0,
        }
        trainer_options.update(self.trainer_kwargs)
        self.trainer = Trainer(**trainer_options)
        self.trainer.fit(self.lit_model, datamodule=self.data_module)

        best_path = checkpoint_callback.best_model_path
        if not best_path:
            raise RuntimeError("GAAP training completed without a best checkpoint")
        self.checkpoint_path = Path(best_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"GAAP checkpoint does not exist: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError(f"GAAP checkpoint has no state_dict: {self.checkpoint_path}")
        self.lit_model.load_state_dict(checkpoint["state_dict"])
        self.lit_model.eval()
        LOGGER.info("GAAP checkpoint selected: %s", self.checkpoint_path)
        return self

    @staticmethod
    def _validate_split_classes(graph: Any) -> None:
        labels = graph.ndata["label"]
        for split in ("train_mask", "val_mask", "test_mask"):
            values = labels[graph.ndata[split]]
            if set(values.tolist()) != {0, 1}:
                raise ValueError(
                    f"{split} must contain both binary classes for GAAP metrics"
                )

    def _require_fitted(self) -> tuple[Any, Any]:
        if self.lit_model is None or self.data_module is None:
            raise RuntimeError("GAAPAdapter has not been fitted or loaded")
        return self.lit_model, self.data_module.g

    def logits(self, graph: Any | None = None) -> torch.Tensor:
        """Run GAAP over its fitted graph or a compatible external graph."""
        model, fitted_graph = self._require_fitted()
        graph = fitted_graph if graph is None else graph
        if graph.device.type != "cpu":
            raise ValueError("GAAPAdapter currently requires a CPU DGL graph")
        features = graph.ndata.get("feature")
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("Inference graph must contain a 2D feature tensor")
        expected_features = int(self.data_module.d_in)  # type: ignore[union-attr]
        if int(features.shape[1]) != expected_features:
            raise ValueError(
                f"Inference graph feature width {features.shape[1]} does not "
                f"match model width {expected_features}"
            )
        model.eval()
        with torch.no_grad():
            result = model(
                [graph] * int(self.model_config["gnn_n_layers"]),
                features,
            )
        if not isinstance(result, torch.Tensor) or result.shape != (
            graph.num_nodes(),
            2,
        ):
            raise RuntimeError("GAAP produced logits with an unexpected shape")
        return result

    def predict_proba(self) -> torch.Tensor:
        """Return per-transaction class probabilities in canonical class order."""
        return self.logits().softmax(dim=-1)

    def predict_proba_on(self, graph: Any) -> torch.Tensor:
        """Predict on another graph with the same encoded feature width."""
        return self.logits(graph).softmax(dim=-1)

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Persist the fitted GAAP Lightning state dict."""
        model, _ = self._require_fitted()
        checkpoint_path = (
            Path(path) if path is not None else self.output_dir / "gaap_state.ckpt"
        )
        if not checkpoint_path.is_absolute():
            checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": self.model_config,
                "seed": self.seed,
            },
            checkpoint_path,
        )
        self.checkpoint_path = checkpoint_path
        return checkpoint_path

    def load_checkpoint(self, graph: Any, path: str | Path) -> GAAPAdapter:
        """Restore GAAP weights using a caller-provided, shape-compatible graph."""
        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"GAAP checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError(f"GAAP checkpoint has no state_dict: {checkpoint_path}")
        self.model_config = dict(checkpoint.get("model_config", self.model_config))
        self.seed = int(checkpoint.get("seed", self.seed))
        self.data_module = TemporalTransactionDataModule(graph)
        lit_sage = load_gaap_lit_sage()
        params = self.model_parameters(
            self.model_config,
            d_in=self.data_module.d_in,
            n_nodes=self.data_module.n_nodes,
        )
        self.lit_model = lit_sage(**params)
        self.lit_model.load_state_dict(checkpoint["state_dict"])
        self.lit_model.eval()
        self.checkpoint_path = checkpoint_path.resolve()
        return self
