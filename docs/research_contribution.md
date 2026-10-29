# Research contribution and evidence boundary

## What is implemented

The project preserves the original GAAP SAGE model and adds a transaction-level
extension around it:

1. A configurable canonical schema for online banking transactions.
2. Strictly historical behavioral features and directed temporal transaction
   graph edges.
3. A thin adapter that reuses original GAAP `LitSAGE` and model classes.
4. Configurable data-drift measures and delayed-label concept-drift detection.
5. An explicit replay-buffer and warm-start adaptation workflow with version
   registry, validation-based promotion, and rollback.
6. Model-agnostic feature/graph explanation evidence and a configurable
   decision layer.
7. FastAPI scoring/monitoring routes, a Streamlit dashboard, and an evaluation
   framework.

This separation is deliberate: **original GAAP** is the fixed baseline;
**Adaptive-GAAP** is the proposed integration of temporal transaction context,
drift monitoring, replay-based fine-tuning, model governance, explanations, and
online scoring around that baseline.

## What is not established

The current implementation is a research prototype, not a demonstrated
production banking system. In particular:

- API labels update the drift monitor but do not trigger adaptation or deploy a
  newly promoted model.
- There is no real banking feed, broker, multi-tenant storage, or production
  label service.
- The API has no authentication or authorization.
- The temporal masks do not by themselves guarantee leakage-free GAAP training:
  full-graph training-mode batch normalization includes validation/test node
  feature statistics. Resolve this before interpreting temporal results as
  out-of-time generalization.
- Explanations are supporting model evidence and heuristics, not proof,
  causality, or a compliance determination.

## Recorded smoke-run evidence

The available report
`reports/final_results.md` describes one synthetic chronological run (seed 42,
single-epoch smoke configuration). On its shared post-drift validation window:

| Model | Average precision | ROC-AUC | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed GAAP | 0.6082 | 0.6848 | 0.0000 | 0.0000 | 0.0000 |
| Full-retrained GAAP | 0.7935 | 0.7935 | 0.0000 | 0.0000 | 0.0000 |
| Adaptive-GAAP | 0.6981 | 0.8804 | 0.0000 | 0.0000 | 0.0000 |

The measured Adaptive-GAAP PR-AUC is 0.0899 higher than fixed GAAP in this
single run, but recall, precision, and F1 remain zero at the configured
classification threshold. Classical baselines reached 1.0 AP on this
synthetic generator, which likely reflects the artificial feature/label
construction and does not indicate realistic banking performance. The
adaptation result did not outperform the full-retraining comparator on AP.

These numbers are observations from one small synthetic run, not evidence of
statistical significance or generalizable improvement. They must not be
presented as production or real-bank results. The full report records missing
ablations as not run; no absent model variant or measurement is inferred.

## Claims that are supported

- The current code can construct temporal transaction graphs and score
  synthetic/canonicalized banking transactions through the existing GAAP model.
- The offline adaptation workflow supports model warm-start, replay sampling,
  candidate evaluation, conditional promotion, registry records, and rollback.
- Data/concept drift can be monitored using configured synthetic streams.
- The API, dashboard, tests, and synthetic experiment tools exist.

## Claims that require additional experiments

- Improved fraud capture or operational utility under real concept drift.
- Superior performance against tuned graph and non-graph baselines.
- Stable gains across seeds, datasets, institutions, payment types, and drift
  severities.
- Lower adaptation cost than full retraining at comparable recall.
- Robustness to catastrophic forgetting, delayed/noisy labels, and repeated
  drift events.
- An acceptable false-positive burden at business-selected operating points.
- Production latency, throughput, memory use, availability, or security.

## Minimum next research steps

1. Eliminate full-graph normalization leakage without editing the original
   baseline; document the extension's exact normalization policy.
2. Re-run strict chronological and rolling-window tests, with split-specific
   preprocessing and train-only fitting.
3. Add repeated seeds and confidence intervals on public, legally usable fraud
   datasets plus transaction-schema-compatible datasets.
4. Report recall at a fixed false-positive rate, PR-AUC, average precision,
   calibration, and cost-sensitive metrics.
5. Implement and score the missing ablations and baseline graph models.
6. Measure online single-transaction latency and throughput separately from
   full-graph batch inference.
7. Wire candidate adaptation to a trusted delayed-label workflow only after
   authentication, auditability, validation, shadow deployment, and rollback
   controls exist.
