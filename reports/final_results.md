# Adaptive-GAAP evaluation results

- Generated: 2026-10-24T17:09:19.357151+00:00
- Dataset: synthetic_banking_transactions
- Seed: 42
- Experiment: chronological_gaap_adaptation_drift_comparison

## Experiment configuration

```json
{
  "ablation_status": "not_run",
  "adaptation": {
    "adaptation": {
      "buffers": {
        "capacity": 5000,
        "half_life_days": 180,
        "minimum_temporal_weight": 0.1,
        "recent_half_life_days": 30
      },
      "evaluation_fraction": 0.2,
      "promotion": {
        "maximum_replay_drop": 0.03,
        "maximum_validation_drop": 0.06,
        "metric": "average_precision",
        "minimum_recent_improvement": 0.0,
        "minimum_recent_metric": 0.5
      },
      "recent_ratio": 0.7,
      "replay_ratio": 0.3,
      "samples_per_adaptation": 300
    },
    "drift_monitor": {
      "concept_drift": {
        "delta": 0.0,
        "min_instances": 10,
        "prediction_threshold": 0.5,
        "threshold": 1.5
      },
      "feature_drift": {
        "features": [
          "amount",
          "transaction_type",
          "device_id",
          "ip_id",
          "receiver_account_id"
        ],
        "js_threshold": 0.1,
        "ks_alpha": 0.01,
        "methods": [
          "psi",
          "ks",
          "jensen_shannon"
        ],
        "min_samples": 20,
        "psi_threshold": 0.2
      }
    },
    "experiment": {
      "max_epochs": 1,
      "rows_per_period": 132
    },
    "explainability_config": "configs/explainability_config.yaml",
    "feature_schema_version": "banking-transaction-v1",
    "gaap_config": {
      "data": {
        "source": "synthetic",
        "synthetic_csv": "data/raw/adaptive_gaap_synthetic.csv",
        "synthetic_rows": 132
      },
      "exclude_label_derived_features": true,
      "feature_config": "configs/feature_config.yaml",
      "graph_config": {
        "categorical_columns": [
          "transaction_type",
          "channel"
        ],
        "default_snapshot_window": "24h",
        "edge_rules": [
          {
            "enabled": true,
            "field": "sender_account_id",
            "name": "same_sender"
          },
          {
            "enabled": true,
            "field": "receiver_account_id",
            "name": "same_receiver"
          },
          {
            "enabled": true,
            "field": "device_id",
            "name": "same_device"
          },
          {
            "enabled": true,
            "field": "ip_id",
            "name": "same_ip"
          },
          {
            "enabled": true,
            "field": "merchant_id",
            "name": "same_merchant"
          }
        ],
        "feature_columns": null,
        "fraud_labels": [
          1,
          "fraud",
          "suspicious",
          true
        ],
        "max_history_hours": 168
      },
      "labels": {
        "fraud": [
          1,
          "1",
          true,
          "true",
          "fraud",
          "suspicious"
        ],
        "legitimate": [
          0,
          "0",
          false,
          "false",
          "legitimate",
          "legit",
          "genuine"
        ]
      },
      "model": {
        "d_feat_emb": 8,
        "d_hidden": 16,
        "data_name": "banking_transactions",
        "dy_raw_bin_width": true,
        "gnn_agg": "max_pool",
        "gnn_dropout": 0.1,
        "gnn_n_layers": 2,
        "gnn_use_bn": true,
        "gnn_use_res": true,
        "init_bin_width": false,
        "loader_type": "MiniF",
        "lr": 0.001,
        "mha_n_heads": 1,
        "mha_n_layers": 1,
        "model_name": "SAGE",
        "n_bins": 8,
        "norm_type": "none",
        "num_threads": 1,
        "pre_dropout": 0.1,
        "pre_n_layers": 1,
        "pre_use_bn": true,
        "preprocess": "None",
        "seed": 42,
        "use_dyple": true,
        "use_mha": false,
        "weight_decay": 0.0001
      },
      "model_version": "gaap-banking-v1",
      "output_dir": "artifacts/adaptive_gaap",
      "schema_config": {
        "path": "configs/transaction_schema.yaml"
      },
      "seed": 42,
      "split": {
        "train_fraction": 0.7,
        "validation_fraction": 0.15
      },
      "threshold": 0.5,
      "training": {
        "max_epochs": 1,
        "patience": 1
      }
    },
    "incremental_training": {
      "max_epochs": 1,
      "patience": 1
    },
    "registry_dir": "D:\\FraudLens\\FraudLens\\artifacts\\evaluation\\adaptive_experiment\\registry",
    "seed": 42
  },
  "drift": {
    "concept_drift": {
      "delta": 0.005,
      "detector": "page_hinkley",
      "min_instances": 20,
      "prediction_threshold": 0.5,
      "threshold": 2.0,
      "window_size": 100
    },
    "experiment": {
      "drift_timestamp": "2025-02-01T00:00:00Z",
      "rows_per_period": 80,
      "static_amount_threshold": 300.0
    },
    "feature_drift": {
      "bins": 10,
      "consecutive_windows": 1,
      "features": [
        "amount",
        "transaction_velocity_1h",
        "transaction_type",
        "device_usage",
        "ip_usage",
        "beneficiary_behaviour",
        "graph_degree",
        "transaction_embedding_mean"
      ],
      "js_threshold": 0.1,
      "ks_alpha": 0.01,
      "methods": [
        "psi",
        "ks",
        "jensen_shannon"
      ],
      "min_samples": 20,
      "missing_rate_threshold": 0.1,
      "psi_threshold": 0.2
    },
    "output_path": "artifacts/drift/synthetic_drift_report.json",
    "seed": 42
  },
  "effective_latency_repeats": 3,
  "evaluation": {
    "ablation": {
      "variants": [
        "basic_gnn",
        "gaap",
        "gaap_temporal_graph",
        "gaap_temporal_graph_drift_detection",
        "gaap_adaptive_retraining",
        "adaptive_gaap_explainability"
      ]
    },
    "benchmark": {
      "include_graph_models": [
        "original_gaap",
        "adaptive_gaap"
      ],
      "latency_repeats": 30,
      "models": [
        "logistic_regression",
        "random_forest"
      ],
      "optional_models": [
        "xgboost"
      ],
      "precision_recall_k": [
        10,
        50,
        100
      ]
    },
    "dataset": "synthetic_banking_transactions",
    "drift_experiment": {
      "adaptation_config": "configs/adaptation_config.yaml",
      "drift_config": "configs/drift_config.yaml",
      "rows_per_period": 220,
      "smoke_epochs": 1,
      "smoke_rows_per_period": 132
    },
    "output_dir": "reports",
    "report": {
      "figures_dir": "reports/figures",
      "final_report": "reports/final_results.md",
      "limitations": [
        "Results depend on the configured dataset and chronological evaluation windows.",
        "Synthetic-data performance does not establish performance on production banking data.",
        "Model comparisons are descriptive unless repeated-seed confidence intervals are supplied.",
        "Unavailable model variants and operational measurements are reported as unavailable."
      ],
      "results_dir": "reports/results",
      "tables_dir": "reports/tables"
    },
    "seed": 42,
    "temporal": {
      "fractions": [
        0.5,
        0.15,
        0.15,
        0.2
      ],
      "rolling_initial_train_timestamps": 60,
      "rolling_step_timestamps": 12,
      "rolling_test_timestamps": 12,
      "timestamp_column": "timestamp"
    },
    "threshold": 0.5
  },
  "models": [
    "logistic_regression",
    "random_forest"
  ],
  "smoke": true
}
```

## Model metrics

| model | model_version | average_precision | recall | precision | f1 | roc_auc | false_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | logistic_regression-seed-42 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| random_forest | random_forest-seed-42 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| GAAP | v1 | 0.6081818181818182 | 0.0 | 0.0 | 0.0 | 0.6847826086956522 | 0.0 |
| GAAP + Drift Detection | v1-drift-monitored | 0.6081818181818182 | 0.0 | 0.0 | 0.0 | 0.6847826086956522 | 0.0 |
| Full-retrained-GAAP | gaap-banking-v1 | 0.7934782608695652 | 0.0 | 0.0 | 0.0 | 0.7934782608695652 | 0.0 |
| Adaptive-GAAP | v2 | 0.698051948051948 | 0.0 | 0.0 | 0.0 | 0.8804347826086957 | 0.0 |

## Observations

Highest observed PR-AUC: logistic_regression (1.0000). This is descriptive and does not establish statistical significance.
Adaptive-GAAP minus fixed GAAP on the same post-drift validation transactions: average_precision +0.0899, recall +0.0000, precision +0.0000, f1 +0.0000.

## Limitations

- Results depend on the configured dataset and chronological evaluation windows.
- Synthetic-data performance does not establish performance on production banking data.
- Model comparisons are descriptive unless repeated-seed confidence intervals are supplied.
- Unavailable model variants and operational measurements are reported as unavailable.
- The drift experiment uses a deterministic synthetic static-threshold scorer; it is a monitoring-control experiment, not a learned GAAP model.
- Classical baselines and the GAAP-family models use the same post-drift validation transactions, but different model-specific training procedures.
- GAAP latency measures batch prediction over its saved training graph; single-transaction online API latency is a separate measurement.
- Ablation variants are not emitted because corresponding scored model outputs were not supplied by this run.
