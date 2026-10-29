# Limitations and risk register

## Research validity

### Full-graph batch-normalization leakage

The transaction wrapper uses chronological masks but supplies the whole graph
in each training batch. The original `LitGNN.training_step` evaluates all
nodes. The SAGE input and graph layers can use batch normalization, whose
training statistics include validation and test-period feature values. This
means a chronological split alone does not establish strict temporal
generalization. The original GAAP implementation was intentionally left
unchanged. Before publishing results, use train-only normalization or a
training-subgraph approach, then rerun all temporal comparisons and ablations.

### Limited evidence

The recorded result is one small synthetic smoke run. The test partition has
few rows; repeated seeds, confidence intervals, public fraud datasets, and
production-like drift are absent. Several requested graph benchmarks and
ablations are explicitly not run. The reported PR-AUC delta does not coincide
with improved recall/precision/F1 at the selected threshold.

### Synthetic data and category coverage

The schema represents generic banking transactions, including payment types
such as UPI, IMPS, NEFT, RTGS, card, bank, wallet, and merchant flows. This does
not demonstrate compatibility with bank feeds, settlement semantics, ISO
formats, transaction reversals, chargebacks, or institution-specific label
policies. Synthetic labels and patterns are simplified.

### Drift/adaptation scope

The concept detector depends on labels being paired with prior scores. Label
delay, missingness, and noise are not systematically evaluated. Drift reporting
does not trigger API retraining. Adaptation is an explicit offline call and
does not yet include an operational shadow/canary deployment protocol.

### Unseen feature categories

At inference, features are aligned to the training columns. A category unseen
during training contributes an all-zero encoded segment. It is accepted, but
its predictive semantics are not learned. Feature schema evolution and unknown
category alarms are needed.

## Operational and security limitations

- No authentication, authorization, rate limiting, or tenant-level isolation.
- The delayed-label route is an untrusted input to concept-drift state until
  protected; erroneous labels can poison monitoring.
- Unmatched prediction/label bookkeeping has no configured TTL or hard cap.
- The transaction service's duplicate-ID set grows for the process lifetime,
  even though retained online transaction rows are bounded.
- Drift events are retained until reset and the API has no reset/expiry
  operation, so the exposed status can become stale.
- The API stores in-memory online rows and drift state; restart loses them.
- Inference recomputes features and graph from retained history on every
  request; latency scales with retained history.
- The one-row synthetic CPU demo measured approximately 16.5 seconds per
  prediction including explanation for its small 132-row model graph; this is
  not a throughput benchmark, but rules out a real-time performance claim.
- The dashboard depends on API availability and configured artifact paths; it
  is not an audit-grade record store.
- Encryption, key/secret lifecycle, retention, privacy/de-identification,
  backups, observability, disaster recovery, and service-level objectives are
  outside the implementation.
- The risk/decision thresholds are demonstrations and require institution-
  specific approval and calibration.

## Dependency and maintenance limitations

The dependency list now includes the core tested stack, but environment
compatibility still depends on Python/PyTorch/DGL wheel availability. The
original code imports a private scikit-learn metric helper; upgrades should be
validated before deployment. XGBoost is optional.

## Recommended order of work

1. Correct and test train-only normalization to remove temporal leakage.
2. Add authn/authz, tenant isolation, bounded label queues, and append-only
   audit records.
3. Make feature/graph updates incremental and benchmark p95/p99 online latency.
4. Add unknown-category detection and explicit feature-schema version
   migration.
5. Add delayed-label, false-alarm, repeated-drift, and forgetting experiments.
6. Implement real shadow evaluation and controlled model promotion/rollback.
7. Validate on appropriate public and institution-approved datasets with
   repeated seeds and confidence intervals.
8. Establish deployment, monitoring, governance, privacy, and model-risk
   controls before any operational banking use.
