# Experiment guide

## Reproducibility controls

The primary seed is `seed` in `configs/evaluation.yaml`; model/adaptation
sections also carry their own seed. Keep them aligned for a paired comparison.
Every evaluation report includes the effective config, dataset name, seed,
model version, timestamps, and paths to actual results. The reproduction script
chooses a fresh output directory each run to avoid collisions with prior model
registries/checkpoints.

## Main synthetic drift experiment

Run all configured evaluation and report generation:

```powershell
python scripts\reproduce_results.py
```

Run the reduced smoke configuration:

```powershell
python scripts\reproduce_results.py --smoke
```

The experiment uses chronological training, validation, post-drift validation,
and streaming windows. Its phases create normal synthetic behavior, introduce
small repeated transfers through new devices/beneficiaries, evaluate the fixed
GAAP model, observe drift, attempt warm-start adaptation, validate a candidate,
and compare it with fixed GAAP and full retraining. Promotion depends on the
configured criteria; a drift event does not itself promote a model.

The default benchmark includes configured logistic regression and random
forest baselines, and includes XGBoost only when installed/enabled. Graph
baseline rows depend on actual graph prediction outputs. Missing model or
ablation outputs are recorded as not run, not synthesized.

## Standalone phase commands

| Goal | Command |
| --- | --- |
| Original GAAP baseline report | `python scripts\run_baseline.py` |
| Original GAAP baseline smoke | `python scripts\run_baseline.py --smoke` |
| Transaction-centric GAAP | `python scripts\train_banking_gaap.py --config configs\adaptive_gaap_baseline.yaml` |
| Drift detector synthetic report | `python scripts\run_drift_experiment.py --config configs\drift_config.yaml` |
| Adaptation comparison | `python scripts\run_adaptation_experiment.py --config configs\adaptation_config.yaml` |
| Full evaluation (fresh timestamped output) | `python scripts\reproduce_results.py` |
| Evaluation smoke | `python scripts\reproduce_results.py --smoke` |
| Complete synthetic/API demo | `python scripts\run_demo.py` |
| Full test suite | `python scripts\run_all_tests.py` |

## Output interpretation

`reports/final_results.md` is a human-readable summary. The machine-readable
results and CSV tables are written under the report directory's `results/` and
`tables/` folders, while plots are generated under `figures/`. The original
`reports/` output can be overwritten by directly running
`scripts/run_evaluation.py`; use `scripts/reproduce_results.py` for isolated
timestamped runs.

PR-AUC/Average Precision is the primary ranking metric. Recall, precision,
F1, ROC-AUC, false-positive rate, confusion matrix, and precision/recall at K
are reported when the observed data supports them. Prediction latency,
adaptation time, and retraining time are distinct measurements; the saved-graph
GAAP latency is not single-request API latency.

The recorded one-seed smoke results in `reports/final_results.md` show zero
recall/precision/F1 for all GAAP-family rows at the configured threshold. Do
not describe an AP delta as overall fraud detection improvement. Repeat across
seeds, eliminate the batch-normalization leakage risk, and report uncertainty
before making comparative research claims.

## Limitations of the current benchmark

- Synthetic fraud labels/patterns are deterministic and may be unusually easy
  for classical models.
- The stand-alone drift control uses a threshold scorer; it is not GAAP
  performance.
- Not all graph model baselines or requested ablation variants have trained
  outputs. The framework declines to invent them.
- Chronological splitting is implemented, but full-graph training-mode batch
  normalization consumes feature values from later partitions. Treat existing
  GAAP-family temporal metrics as potentially contaminated until corrected.
- One synthetic seed/window is not statistically conclusive.
