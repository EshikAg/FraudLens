"""End-to-end CPU smoke test through real GAAP training and checkpoint restore."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dgl")
pytest.importorskip("lightning")

from models.banking_gaap import BankingGAAP


def test_synthetic_transactions_train_gaap_predict_and_reload(tmp_path: Path) -> None:
    config_path = Path("configs/adaptive_gaap_baseline.yaml")
    model = BankingGAAP(config_path)
    model.config["data"]["synthetic_rows"] = 66
    model.config["data"]["synthetic_csv"] = str(tmp_path / "raw" / "transactions.csv")
    model.config["output_dir"] = str(tmp_path / "run")
    model.config["training"]["max_epochs"] = 1
    model.config["training"]["patience"] = 1

    model.fit()

    assert model.graph is not None
    assert model.gaap is not None
    assert model.graph.dgl_graph.num_nodes() == 66
    assert model.graph.dgl_graph.ndata["feature"].shape[0] == 66
    assert set(model.split_ids) == {"train", "validation", "test"}
    assert max(
        model.graph.node_table.loc[
            model.graph.node_table["transaction_id"].isin(model.split_ids["train"]),
            "timestamp",
        ]
    ) < min(
        model.graph.node_table.loc[
            model.graph.node_table["transaction_id"].isin(model.split_ids["validation"]),
            "timestamp",
        ]
    )

    predictions = model.predict()
    probabilities = model.predict_proba()
    assert len(predictions) == 66
    assert {
        "transaction_id",
        "timestamp",
        "fraud_probability",
        "predicted_label",
        "true_label",
        "model_version",
    }.issubset(predictions.columns)
    assert probabilities["transaction_id"].equals(predictions["transaction_id"])
    assert predictions["fraud_probability"].between(0, 1).all()
    assert model.evaluate("test")["roc_auc"] >= 0

    save_path = model.save(tmp_path / "saved_model")
    assert (save_path / "gaap_state.ckpt").is_file()
    assert (save_path / "graph" / "metadata.json").is_file()
    restored = BankingGAAP.load(save_path)
    restored_predictions = restored.predict()
    np.testing.assert_allclose(
        predictions["fraud_probability"].to_numpy(),
        restored_predictions["fraud_probability"].to_numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    saved_metadata = json.loads(
        (save_path / "banking_gaap.json").read_text(encoding="utf-8")
    )
    assert saved_metadata["split_ids"] == model.split_ids
