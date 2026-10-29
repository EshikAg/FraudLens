"""Contract tests for the adapter to the existing GAAP Lightning model."""

from __future__ import annotations

import pytest
import torch

dgl = pytest.importorskip("dgl")

from models.gaap_adapter import GAAPAdapter, TemporalTransactionDataModule


def _graph():
    graph = dgl.graph(([0, 1], [1, 2]), num_nodes=3)
    graph.ndata["feature"] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    graph.ndata["label"] = torch.tensor([0, 1, 0], dtype=torch.int64)
    graph.ndata["train_mask"] = torch.tensor([True, False, False])
    graph.ndata["val_mask"] = torch.tensor([False, True, False])
    graph.ndata["test_mask"] = torch.tensor([False, False, True])
    return graph


def test_temporal_data_module_exposes_gaap_feature_and_label_contract() -> None:
    graph = _graph()

    data_module = TemporalTransactionDataModule(graph)

    assert data_module.d_in == 2
    assert data_module.n_classes == 2
    assert data_module.n_nodes == 3
    assert data_module.train_dataloader().dataset[0] is graph
    assert data_module.val_dataloader().dataset[0] is graph


def test_data_module_rejects_overlapping_or_unlabeled_masks() -> None:
    graph = _graph()
    graph.ndata["val_mask"] = torch.tensor([True, False, False])

    with pytest.raises(ValueError, match="disjoint"):
        TemporalTransactionDataModule(graph)


def test_data_module_rejects_mask_for_unlabeled_transaction() -> None:
    graph = _graph()
    graph.ndata["label"][0] = -1

    with pytest.raises(ValueError, match="cannot include unlabeled"):
        TemporalTransactionDataModule(graph)


def test_gaap_model_parameters_include_graph_feature_dimensions() -> None:
    parameters = GAAPAdapter.model_parameters(
        {
            "d_hidden": 16,
            "preprocess": "None",
            "n_bins": 8,
            "use_dyple": False,
            "dy_raw_bin_width": True,
            "pre_n_layers": 1,
            "pre_use_bn": True,
            "pre_dropout": 0.1,
            "gnn_n_layers": 2,
            "gnn_use_bn": True,
            "gnn_dropout": 0.1,
            "gnn_agg": "max_pool",
            "gnn_use_res": True,
            "use_mha": False,
            "mha_n_layers": 1,
            "mha_n_heads": 1,
            "lr": 0.001,
        },
        d_in=7,
        n_nodes=33,
    )

    assert parameters["d_in"] == 7
    assert parameters["n_classes"] == 2
    assert parameters["n_nodes"] == 33
