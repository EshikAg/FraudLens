# Adaptive-GAAP architecture audit

## Executive summary

The repository contains the original GAAP SAGE implementation and a substantial
extension built around it. The extension reuses `mycode/nn/litnn.py` through a
thin adapter; it does not replace the original experiment entry points or model.
The transaction-centric path can canonicalize generic transaction rows, create
strictly historical features and graph edges, score through GAAP, explain the
score, and report data/concept drift. Offline adaptation supports replay,
warm-start fine-tuning, version registration, validation, promotion, and
rollback.

There is an important boundary: the API's delayed-label endpoint updates drift
monitoring only. It does not trigger the adaptation manager or hot-swap the
loaded model. Model adaptation is an explicit offline experiment/operation.

## Component inventory

| Stage | Current implementation | Audit status |
| --- | --- | --- |
| Data ingestion | `preprocessing/transaction_schema.py`, `preprocessing/transaction_loader.py` | Canonical fields, aliases, and CSV ingestion exist. There is no bank/provider connector or streaming broker integration. |
| Feature engineering | `preprocessing/feature_engineering.py` | Historical transaction, account, beneficiary, device, IP, velocity, time, and degree features. Rows at the same timestamp are treated as a batch. |
| Temporal graph | `graph/edge_builder.py`, `graph/transaction_graph.py`, `graph/graph_features.py` | One transaction per node, configurable relationship edges, strictly earlier-to-later causal edges, feature/ID mappings, graph metadata. |
| GAAP integration | `models/gaap_adapter.py`, `models/banking_gaap.py`, original `mycode/nn/gnn.py`, `mycode/nn/litnn.py` | Wrapper builds masks and delegates to the original Lightning SAGE implementation. Original files remain the baseline. |
| Fraud prediction | `models/banking_gaap.py`, `api/services/prediction_service.py` | Produces two-class probabilities; API applies a configurable risk/decision policy. |
| Data drift | `drift/feature_drift.py`, `drift/drift_monitor.py` | PSI, KS, Jensen-Shannon, missingness, and configured window thresholds. |
| Concept drift | `drift/concept_drift.py` | Prediction/label pairing and Page-Hinkley error monitoring; labels are delayed. |
| Adaptation | `models/adaptive_gaap.py`, `adaptation/` | Explicit offline fine-tuning, validation, promotion/rejection, versioning, and rollback. Not called by the API. |
| Replay | `adaptation/replay_buffer.py` | Bounded recent and historical storage with configured sampling/temporal weighting. |
| Versioning | `adaptation/model_registry.py`, `adaptation/promotion_manager.py` | JSON registry, version checkpoint paths, metrics and promotion state. |
| Explainability | `explainability/` and `api/services/explanation_service.py` | Local feature ablation/reason codes and earlier graph neighbors. Embeddings are not asserted to be human-readable. |
| API | `api/main.py`, `api/schemas.py`, `api/services/` | FastAPI health/model/predict/drift/metrics and delayed-label routes. Model loads once per service startup. |
| Dashboard | `dashboard/app.py`, `dashboard/pages/`, `dashboard/services/` | Streamlit presentation layer; live scoring and status come through the API. |
| Evaluation | `evaluation/`, `scripts/run_evaluation.py`, `scripts/run_adaptation_experiment.py` | Chronological and streaming metrics, synthetic drift/adaptation runner, latency, plots and machine-readable results. |
| Tests | `tests/` | 83 tests pass in the audited Python 3.11 environment. |
| Documentation | `docs/` | Phase-oriented baseline architecture existed before this audit; this audit adds architecture, deployment, research, experiment, and limitations guides. |

## Existing GAAP baseline and active experiment configuration

The original model is the GraphSAGE `SAGE` implementation in
`mycode/nn/gnn.py`, constructed and trained by `LitSAGE` in
`mycode/nn/litnn.py`. The currently recorded original-GAAP experiment is the
`tolokers` MiniF configuration in
`config/SAGE_MiniF_DyPLE_MHA/tolokers.yaml`: 4 graph layers, hidden width 64,
DyPLE and MHA enabled, batch normalization enabled, seed 42, and up to 22,222
epochs with early stopping. That config requests CUDA. The reproducible wrapper
config `configs/baseline.yaml` retains the architecture/seed while selecting
CPU, offline W&B, and a configured output directory. Checkpoints are managed
by the existing Lightning `ModelCheckpoint` callback; the wrapper records the
selected checkpoint and effective configuration in its JSON report.

The transaction-oriented GAAP baseline uses `configs/adaptive_gaap_baseline.yaml`
and `scripts/train_banking_gaap.py`. It defaults to 132 deterministic synthetic
rows, 70/15/15 chronological partitions, 2 SAGE layers, and CPU graph training.
Its model and graph are saved under `artifacts/adaptive_gaap/model` unless an
output directory is supplied.

## Data and prediction flow

1. The schema adapter validates/mutates provider-specific aliases into the
   canonical transaction contract. Transaction types remain configurable values
   (for example UPI, IMPS, NEFT, RTGS, card, wallet, and merchant payments).
2. The feature generator sorts by timestamp and calculates each row from
   strictly earlier observations. Equal-timestamp rows do not see each other.
3. The graph builder creates transaction nodes, encoded node features, and
   directed historical edges. The model's feature matrix excludes IDs, labels,
   timestamps, and balances by default.
4. Chronological split IDs become boolean train/validation/test masks. The
   `TemporalTransactionDataModule` passes a single full-graph batch to the
   existing GAAP SAGE model.
5. The API canonicalizes a request, excludes history at or after its
   timestamp, recomputes the feature table and graph, aligns the feature matrix
   to the saved model schema, scores the transaction, creates an explanation,
   updates drift state, and commits the row only after successful scoring.
6. Dashboard predictions and model/drift status are fetched from the API;
   artifact views are read-only.

## Audit findings and risks

### High: temporal evaluation leakage through full-graph batch normalization

The extension's train, validation, and test masks are chronological and
disjoint. However, `TemporalTransactionDataModule` returns the same graph in
the training loader. `LitGNN.training_step` forwards all its nodes, and the
SAGE preprocessor/graph layers use batch normalization. During training mode,
batch-normalization statistics therefore include features from validation and
test-period nodes. The causal edge direction prevents later nodes from sending
messages backward, but it does not prevent their feature values from affecting
global batch-normalization statistics. This compromises a strict temporal
generalization claim. The original baseline was deliberately not edited during
this audit; a research-grade fix should use train-only normalization statistics
or a training subgraph/sampling design, then repeat the chronological
evaluation.

### High: no live adaptation or online promotion

`POST /labels/{transaction_id}` pairs labels with predictions for concept
monitoring. There is no call from that route to `AdaptiveGAAP.adapt`, no
candidate validation workflow, and no live model-registry promotion/hot reload.
The implemented adaptation path is explicit and offline. Do not describe the
deployed API as self-adapting.

### High: API trust boundary is not production hardened

The API has no authentication, authorization, rate limiting, or tenant
isolation. In particular, an unauthenticated delayed-label request could
influence concept-drift state. Deploy only behind a trusted gateway until
identity, access control, audit policy, and request limits are implemented.

### Medium: per-request work grows with retained history

Each score concatenates the fitted graph's node table and retained online rows,
filters and sorts history, recomputes historical features, rebuilds a graph,
and runs inference. The configured cap bounds memory but does not make this
incremental; repeated scoring cost grows with history length and can become a
latency bottleneck.

### Medium: pending delayed labels are not bounded

The concept detector keeps unmatched prediction/label state while waiting for
labels. The transaction history and feature window are bounded, but this
pending join state has no configured expiry/cap. A production delayed-label
service needs explicit retention/expiry and metrics for unmatched predictions.
The transaction service also keeps a seen-ID set for duplicate rejection that
grows for the lifetime of the process, even when old online rows have been
evicted from the bounded history deque.

### Medium: unseen categories are unrepresented

Inference aligns the input matrix to the trained feature columns. A new
transaction type, channel, or other category without a trained column is
silently all-zero in the aligned matrix. Requests still score, but a truly new
category is not given a learned representation. Monitor/flag unknown
categories and define a schema migration/retraining policy.

### Medium: dependency manifest was incomplete

Before this audit, `requirements.txt` declared mostly API/dashboard libraries,
not the essential GAAP, DGL, Lightning, feature, configuration, or test
dependencies. It now includes the core tested stack with compatibility bounds.
XGBoost remains optional and is only used when installed and enabled. The
original implementation also imports a private scikit-learn metrics helper,
which creates an upgrade compatibility risk.

### Low: degraded API configuration is intentional but easy to miss

`configs/api_config.yaml` sets `model_directory: null`. The server can start
and answer health checks, but scoring is unavailable (HTTP 503) until a trained
model path is configured. This must be set before using the dashboard.

### Medium: drift status has no recovery/expiry policy

Detected events are retained until reset, and the API exposes no reset or
acknowledgement route. Once an event is recorded, clients can continue seeing a
historical drift event as current status. Distinguish “latest event” from
“currently drifting” and define a clear recovery/reset policy.

### Measured local API latency

The successful one-request synthetic demo measured about 16.5 seconds for a
single CPU prediction with explanation against the saved 132-row model graph.
This is a local smoke measurement, not a service-level benchmark, but it
confirms that current per-request graph reconstruction/explanation is too slow
to treat as a production real-time latency result.

## Generic online transaction support

The canonical schema can represent generic bank transactions and configurable
aliases, and the feature/graph logic is not intrinsically tied to one
transaction network. UPI, IMPS, NEFT, RTGS, card, bank, wallet, and merchant
payments can be represented as categories. This is a data contract and
synthetic demonstration, not evidence of integrations with real bank feeds,
ISO 20022, payment processors, or a regulated production environment.

## Files that define the original GAAP baseline

Keep the baseline implementation and experiment configuration intact:

- `mycode/nn/gnn.py`
- `mycode/nn/litnn.py`
- `mycode/utils/dataloader.py`
- `mycode/utils/util.py`
- `mycode/ENV.py`
- `mycode/exp/101_retrain.py`
- `mycode/exp/102_retrain_reallinear_att.py`
- `mycode/run_one.py`
- `config/config.yaml`
- `config/SAGE_MiniF_DyPLE_MHA/`
- Original dataset files and experiment outputs under `datasets/` and `outputs/`

## Validation performed

- `python -m pytest tests\test_evaluation.py -q`: 10 passed.
- `python -m pytest -q`: 83 passed; only dependency deprecation/future warnings
  were reported.
- The HTTP-level test suite covers health, prediction, invalid inputs, labels,
  drift, metrics, and scoring through a trained/saved synthetic GAAP model.
- `scripts/run_demo.py` and `scripts/reproduce_results.py` provide repeatable
  command-line flows; the demo is validated separately from the unit suite.
