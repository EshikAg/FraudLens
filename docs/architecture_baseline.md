# GAAP baseline architecture

## Scope and source of truth

This Phase 1 layer wraps the existing GAAP benchmark experiment. It does not add
transaction ingestion, online adaptation, drift detection, explanations, or a
new fraud model.

The active experiment is the latest recorded run in
`outputs/2026-10-06/22-22-43/.hydra`: the Tolokers dataset, `SAGE` model, `MiniF`
loader, seed 42, and CPU. The recorded invocation is equivalent to
`python mycode/exp/101_retrain.py -cn tolokers device=cpu`. Its hyperparameters
are captured in [`../configs/baseline.yaml`](../configs/baseline.yaml).

## Existing architecture

| Area | Existing implementation |
| --- | --- |
| GAAP model for the active experiment | `LitSAGE` and its local `SAGE`/`SAGEConv` definitions in `mycode/exp/101_retrain.py`. The model combines DyPLE feature embedding, GraphSAGE max-pool layers, residual connections, multi-head attention, and a binary classifier. |
| Shared model implementation | `mycode/nn/litnn.py` provides Lightning modules, including `LitSAGE`; `mycode/nn/gnn.py` provides the corresponding reusable `SAGE` and other model classes. The `run_one.py` entry point selects these through `ENV.model_lit_map`. |
| Dataset loader | `MiniFDataModule` in `mycode/utils/dataloader.py`, selected through `ENV.tag_dm_map`. `read_dataset` loads DGL graphs from `datasets/<dataset>`; the active benchmark graph is `datasets/tolokers`. |
| Graph, features, labels, splits | The DGL graph stores node features in `ndata["feature"]`, class labels in `ndata["label"]`, and Boolean split masks in `train_mask`, `val_mask`, and `test_mask`. The loader materializes these from the graph's stored split masks. The active model is node classification over the benchmark graph, not transaction-level ingestion. |
| Preprocessing | `preprocess_feats` in `mycode/utils/dataloader.py` performs the configured feature normalization/encoding. The active configuration uses `norm01` and `preprocess: None`; the model's DyPLE embedding is enabled. |
| Training | `Experiment.train` in `mycode/exp/101_retrain.py` constructs the existing Lightning `Trainer`, fits the existing `LitSAGE`, and uses the existing Adam optimizer/scheduler. The active setup uses a maximum of 22,222 epochs with early stopping (patience 31), monitored on `val_aps`. |
| Evaluation | `LitSAGE`/`LitGNN` validation and test hooks use `cal_binary_metrics` in `mycode/utils/util.py`. The existing evaluator computes precision, recall, macro-F1, ROC-AUC, and Average Precision (plus other benchmark metrics) on the graph's train/validation/test masks. |
| Checkpoints | `ModelCheckpoint(save_top_k=1, monitor="val_aps", mode="max")` saves the best checkpoint. `Experiment.test` reads `trainer.checkpoint_callback.best_model_path` and restores its `state_dict` before calling `Trainer.test`. |
| Configurations and CLI | Hydra configs are in `config/` and `config/SAGE_MiniF_DyPLE_MHA/`. The latest saved resolved config is in `outputs/2026-10-06/22-22-43/.hydra/config.yaml`; the override is `device=cpu`. `101_retrain.py` defaults to `yelp.yaml` but accepts Hydra config-name and key overrides. `run_one.py` is another Hydra entry point, using `config/config.yaml`. |
| Dependencies | There is no root `requirements.txt`, `pyproject.toml`, or other dependency manifest. Source imports include PyTorch, DGL, PyTorch Lightning, Hydra/OmegaConf, W&B, NumPy, pandas, scikit-learn, `rtdl_num_embeddings`, `toad`, and tqdm. The saved run metadata reports Python 3.11.9. Phase 1 adds no dependency. |

### Existing dataset representation and limitations

The current loader expects pre-built benchmark graph files and stored masks; it
does not accept generic UPI, IMPS, NEFT, RTGS, card, bank, wallet, or merchant
transaction records. The current feature normalizer is preserved as implemented
in GAAP. This baseline documents and measures that implementation; it does not
change preprocessing or introduce temporal graph semantics.

### Reproducing the baseline

From the repository root:

```powershell
python scripts\run_baseline.py
```

The wrapper defaults to CPU and W&B offline mode, writes the effective config
to `artifacts/baseline/baseline_config.json`, writes metrics to
`artifacts/baseline/baseline_metrics.json`, and records the best checkpoint
path. Training uses the same GAAP `Experiment`, Lightning trainer, graph
loader, model, callbacks, and metric functions as the original entry point.

Run a one-epoch integration smoke run with:

```powershell
python scripts\run_baseline.py --smoke
```

To rerun the complete experiment with a chosen seed:

```powershell
python scripts\run_baseline.py --seed 123 --output-dir artifacts/baseline-seed-123
```

The smoke option sets both `max_epochs` and early-stopping patience to one; it
does not change the dataset, model, or batch/sampling semantics.

## Phase 2: generic transaction preprocessing

The reusable transaction pipeline is implemented separately from the
benchmark graph loader in `preprocessing/`. `TransactionSchema` maps configurable
source columns into the canonical fields listed in
`data/schemas/canonical_transaction_schema.json`; absent inputs remain nullable
and their availability is recorded. CSV ingestion is available through
`load_transactions`, with optional JSON availability-report output.

`TransactionFeatureGenerator` produces a dataframe containing canonical
columns and reusable transaction, account, device, IP, behavioural, and
graph-oriented features. Its temporal contract is strict: all rows with the
same timestamp are evaluated before that timestamp updates historical state.
Thus only strictly earlier transactions and labels can affect a row's
historical features. Account and beneficiary ages are first-observed ages in
the input, not asserted real-world account creation dates. The resulting
feature-column list is in dataframe attributes for downstream tensor or DGL
conversion; metadata and labels remain available but are explicitly listed as
non-feature columns.

Use `configs/transaction_schema.yaml` and `configs/feature_config.yaml` to
customize input mappings and temporal thresholds. `preprocessing/temporal_split.py`
provides chronological splits that keep equal-timestamp transactions together,
and `preprocessing/synthetic_data.py` creates deterministic example data.
Focused validation is available with `python -m pytest tests/test_transaction_schema.py tests/test_feature_engineering.py`.

## Phase 3: transaction temporal graph

`graph/transaction_graph.py` converts canonical or engineered transaction
tables into DGL graphs where each transaction is one node. Numeric and
categorical node features are encoded separately from labels and identifiers;
labels, timestamps, and transaction IDs remain available as supervision and
metadata. The directed relationship edges always point from an earlier
transaction to a later transaction. Equal-time rows do not connect, and labels
do not participate in edge construction.

`configs/graph_config.yaml` controls relationship rules, retained edge history,
feature columns, and the default snapshot window. `graph/temporal_graph.py`
provides full-history snapshots, rolling time windows, and incoming local
historical neighborhoods. A requested time window wider than the graph's
retained edge history is rejected instead of silently omitting relationships;
use `TransactionGraphBuilder.build(..., full_history=True)` when longer custom
windows are needed. Use `build_transaction_graph_from_csv` for CSV-to-DGL
conversion and optional persistence. `TransactionGraph.to_gaap_graph` adds
explicit train/validation/test masks for the existing GAAP graph interface; it
requires provided labels and disjoint masks, and does not invent a split.

Persisted graphs use DGL's graph format plus `nodes.json` and `metadata.json`
sidecars so transaction IDs and feature/relation mappings can be restored.
`to_networkx_sample` produces a deterministic, bounded sample for inspection.
Run graph tests with `python -m pytest tests/test_graph_builder.py tests/test_temporal_graph.py`.

## Phase 4: transaction graph to existing GAAP

`models/gaap_adapter.py` imports the shared `LitSAGE` implementation from
`mycode/nn/litnn.py` and supplies a single full-graph DGL batch through a thin
Lightning DataModule. It preserves GAAP's float feature tensor, integer class
labels, `train_mask`/`val_mask`/`test_mask`, two output classes, `val_aps`
checkpoint selection, and original binary metric hooks. It uses Lightning's
CPU trainer, early stopping, and best-checkpoint callback without changing
either existing GAAP training entry point.

`BankingGAAP` applies the canonical schema, historical feature generator,
chronological split, transaction graph builder, and adapter. Label-derived
historical suspicious-neighbor features are excluded by default. Prediction
tables include transaction ID, timestamp, probability, predicted and available
true labels, and model version. Saving includes DGL graph metadata, the GAAP
checkpoint, and temporal split membership; `BankingGAAP.load` reconstructs the
model and graph for prediction.

Run the synthetic CPU workflow with:

```powershell
python scripts\train_banking_gaap.py --config configs\adaptive_gaap_baseline.yaml
```

Use `--smoke` for one epoch or `--input-csv path\to\transactions.csv` for a
canonicalizable input CSV. Outputs include transaction predictions, evaluation
metrics, and a reloadable model directory under the configured output path.

## Phase 5: data- and concept-drift monitoring

`drift/feature_drift.py` compares a configured reference transaction window to
streaming windows using PSI and Jensen-Shannon divergence for numeric or
categorical features, and the two-sample KS test for numeric features.
`drift/concept_drift.py` pairs predictions with labels as they arrive and
monitors binary classification errors with a one-sided Page-Hinkley detector.
`DriftMonitor` combines both outputs, retains detected events until reset, and
can write a structured JSON report. Detection does not trigger training or
update GAAP weights.

Run the deterministic synthetic demonstration with:

```powershell
python scripts\run_drift_experiment.py --config configs\drift_config.yaml
```

The report compares a fixed amount-threshold model's fraud metrics before and
after a synthetic shift to low-value repeated transfers from new devices. This
is a monitoring demonstration, not an evaluation of a trained GAAP checkpoint.
Thresholds, minimum samples, and consecutive feature windows are configurable
in `configs/drift_config.yaml`.

## Phase 6: drift-triggered Adaptive-GAAP

`AdaptiveGAAP` preserves the first trained GAAP checkpoint as immutable `v1`.
Labeled recent transactions and historical examples are held in bounded replay
buffers; historical sampling uses configurable exponential temporal decay with
a nonzero minimum weight so old patterns remain replayable. On an explicit
drift event, the adaptation manager samples the configured recent/replay mix
and warm-starts the existing `LitSAGE` checkpoint through `GAAPAdapter`. The
GAAP implementation itself is unchanged.

Every candidate is scored against the current promoted model and fixed v1 on
validation, recent, and replay windows. Promotion is fail-closed and governed
by `configs/adaptation_config.yaml`; replay or validation regression beyond
configured tolerances rejects the candidate. The registry stores immutable
version checkpoints, drift lineage, data windows, feature-schema version,
metrics, and promotion history. `AdaptiveGAAP.rollback()` restores the prior
promoted version. This phase does not trigger adaptation from the detector
internally; callers pass a detected drift event and labeled windows explicitly.

Run the end-to-end synthetic comparison (initial GAAP, drift detection,
warm-started adaptation, and a separate full-retraining comparator) with:

```powershell
python scripts\run_adaptation_experiment.py --config configs\adaptation_config.yaml
```

Add `--smoke` for a one-epoch CPU verification. The JSON report, registry,
checkpoints, and replay state are saved under `artifacts/adaptation/` by default.
The old baseline experiment commands and GAAP source files remain unchanged.

## Phase 7: transaction explanations

GAAP is a GraphSAGE transaction-node classifier. Its learned embeddings and
hidden layers are not presented as human-readable explanations. Instead,
`ExplanationService` explains an existing score with local feature-group
ablation against a strictly earlier-node baseline, engineered transaction
signals, and causal incoming graph paths. Explanations include configured risk
and decision bands, ranked stable reason codes, related transaction IDs,
relationship paths, and explicit limitations.

Historical neighbor fraud labels are withheld unless the graph's transaction
metadata includes the configured `label_available_at` column and that timestamp
is no later than the explained transaction. Without that evidence, the status
is `null` and it cannot produce a suspicious-neighborhood reason. No future
transactions or future labels are used. Explain a fitted, promoted model with
`AdaptiveGAAP.explain(transaction_id)`; `explain_all()` is also available for
bounded offline analysis. Settings are in `configs/explainability_config.yaml`.

## Phase 8: online scoring API

`api.main:app` exposes FastAPI health, model, prediction, drift, and metrics
routes. The prediction service loads one saved `BankingGAAP` model at startup,
then processes each request through the canonical schema, historical feature
generator, causal transaction graph, existing GAAP adapter, explanation
service, and drift monitor. New categorical values are aligned to the saved
feature-column order; values with no trained column do not change model width.

For every score, the inference graph is limited to training/previous online
transactions strictly earlier than the requested timestamp plus the current
transaction. Duplicate transaction IDs and ground-truth labels on `/predict`
are rejected. `/labels/{transaction_id}` accepts delayed binary labels for the
existing concept-drift detector. Thresholds and illustrative decisions are
configurable in `configs/api_config.yaml`; the demonstration policy is not a
banking decision recommendation.

Train and save a baseline model, set `model_directory` in the API config to the
resulting model directory (default training output is
`artifacts/adaptive_gaap/model`), then run:

```powershell
python scripts\train_banking_gaap.py --config configs\adaptive_gaap_baseline.yaml
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The default API config leaves `model_directory` unset so the service can start
in a degraded state for health checks, but prediction routes return HTTP 503
until a saved model is configured. Request logs omit transaction identifiers
and financial fields.

## Phase 9: investigation and monitoring dashboard

`dashboard/app.py` provides Streamlit pages for an overview, transaction
monitor, transaction investigation, drift monitoring, and model lineage. It is
a presentation layer: online transaction scores and current model/drift status
come from the FastAPI service; registry and experiment details are loaded
read-only from configured JSON artifacts. Demo rows use the existing synthetic
transaction generator and are scored individually by the API; labels are
excluded from prediction requests. If the API is unavailable or has no loaded
model, the dashboard reports that state instead of fabricating scores.

Start the backend with a configured saved checkpoint as described in Phase 8,
then start the dashboard with:

```powershell
streamlit run dashboard\app.py --browser.gatherUsageStats false
```

The API URL and optional registry, adaptation, drift, and training report paths
are configured in `dashboard/config.yaml`; `FRAUDLENS_API_URL` and
`FRAUDLENS_DASHBOARD_CONFIG` can override the corresponding settings. The
demonstration transaction table limits exposed details and omits account,
device, and IP identifiers.

## Phase 10: research evaluation framework

`evaluation/` provides binary fraud metrics (Average Precision / PR-AUC,
ROC-AUC, precision, recall, F1, false-positive rate, confusion matrix, and
Precision@K / Recall@K), chronological four-period splitting, expanding
rolling windows, fixed-window streaming evaluation, configurable sklearn
benchmarks, callback-driven ablations, drift-period analysis, and latency /
Python-allocation measurement. Splits are made on distinct timestamps, so
same-time transactions cannot be separated across periods.

Run the complete synthetic concept-drift comparison and write CSV, JSON,
Markdown, and available PNG evidence with:

```powershell
python scripts\run_evaluation.py --smoke
```

Use `python scripts\run_evaluation.py` for the configured row count and epoch
settings. `configs/evaluation.yaml` controls the seed, optional benchmark
models, K values, drift experiment, output paths, and declared limitations.
Intermediate model and drift experiment outputs are written under
`artifacts/evaluation`; final reports are written under `reports/`, including
`reports/final_results.md`, `reports/results/evaluation_results.json`,
`reports/tables/model_metrics.csv`, and figures under `reports/figures/`.

The run compares Logistic Regression and Random Forest, when configured,
against original GAAP, full-retrained GAAP, and Adaptive-GAAP on the same
post-drift validation transactions. Classical model inputs are fitted only on
the initial GAAP training IDs. XGBoost is included only when installed and
enabled; unavailable optional models are named in the result metadata.
GAAP-family validation prediction records are produced by the existing
adaptation experiment and evaluated through the shared metric implementation.
Adaptation and full-retraining durations are wall-clock measurements. GAAP
latency is labeled as full-saved-graph batch inference, not single-request API
latency. The separate drift-control experiment uses a deterministic
amount-threshold scorer and is not presented as learned-model performance.

`evaluation/ablation.py` runs supplied variant callbacks and measures their
actual outputs. The default evaluation command does not claim ablation results
because the repository does not yet contain six distinct trained ablation
variants. Likewise, figures requiring absent model outputs are omitted and
the final report records the limitation rather than fabricating values.
Metrics from one synthetic seed/window are descriptive, not statistically
significant and not evidence of production-bank performance. To produce the
six requested ablation rows, implement and supply callbacks that return
transaction IDs, labels, probabilities, and model versions for each variant.

## Baseline outputs

The JSON report records dataset, model, seed, configured and completed epochs,
wall-clock training time, validation and test metrics, checkpoint path, and the
saved effective configuration path. F1 uses GAAP's existing macro-F1 (`mf1`)
calculation; `average_precision` is GAAP's `aps` (PR-AUC / Average Precision).
Metrics are read from GAAP's evaluator; missing metric values or checkpoint
files are errors rather than fabricated defaults.

The latest existing output directories contain Hydra configs and logs, but no
saved model checkpoint was found when this baseline layer was added. The new
runner records the checkpoint created by its own run.

## GAAP files to preserve unchanged

The following are original baseline implementation files and must remain
unchanged when extending Adaptive-GAAP:

- `mycode/nn/gnn.py`
- `mycode/nn/litnn.py`
- `mycode/utils/dataloader.py`
- `mycode/utils/util.py`
- `mycode/ENV.py`
- `mycode/exp/101_retrain.py`
- `mycode/exp/102_retrain_reallinear_att.py`
- `mycode/run_one.py`
- `config/config.yaml`
- All dataset-specific configs under `config/SAGE_MiniF_DyPLE_MHA/`

The dataset graph files under `datasets/` and historical experiment outputs
are also inputs/evidence for the original baseline and are not changed by this
Phase 1 wrapper. New functionality belongs in separate adapters, configuration,
documentation, and tests.
