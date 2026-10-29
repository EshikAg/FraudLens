# Deployment and local setup

This guide covers a local CPU installation. The API and dashboard are
demonstration components and are not production-ready banking services.

## Install on Windows with Python 3.11

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file pins compatible version ranges for the existing GAAP
stack, DGL, Lightning, the online application, and tests. It does not install
optional XGBoost. If the configured benchmark enables XGBoost, install it into
the active virtual environment:

```powershell
python -m pip install "xgboost>=2,<4"
```

No system-wide installation is needed. For CUDA, install a PyTorch/DGL build
matching the machine and the supported DGL/PyTorch versions; the documented
default and smoke workflows use CPU.

## Train the original GAAP baseline

The wrapper uses the original GAAP experiment and the current tolokers
configuration, overrides its requested device to CPU, and writes a metrics
JSON/checkpoint report:

```powershell
python scripts\run_baseline.py
```

Use one epoch to verify the command path:

```powershell
python scripts\run_baseline.py --smoke
```

The original experiment still requires its original tolokers dataset files.
The wrapper does not fabricate or replace that research dataset. Its output
directory and configuration are controlled by `configs/baseline.yaml`.

## Train GAAP on transaction-centric synthetic or canonicalized data

```powershell
python scripts\train_banking_gaap.py --config configs\adaptive_gaap_baseline.yaml
```

For a short CPU check:

```powershell
python scripts\train_banking_gaap.py --config configs\adaptive_gaap_baseline.yaml --smoke
```

To use a CSV, supply its path and, if desired, an output directory:

```powershell
python scripts\train_banking_gaap.py --config configs\adaptive_gaap_baseline.yaml --input-csv data\my_transactions.csv --output-dir artifacts\my_gaap_model
```

The saved model is `<output-dir>\model`; the default is
`artifacts\adaptive_gaap\model`.

## Configure and run the API

Set `model_directory` in `configs/api_config.yaml` to the saved model directory,
for example:

```yaml
model_directory: artifacts/adaptive_gaap/model
```

Then run:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Check health at `http://127.0.0.1:8000/health`. The unmodified default config
uses `model_directory: null`: health is degraded and `/predict` returns
service-unavailable until a model is configured.

## Run the dashboard

Keep the API running in another terminal, then run:

```powershell
streamlit run dashboard\app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

The dashboard's API URL is set in `dashboard/config.yaml`; it can be overridden
with `FRAUDLENS_API_URL`.

## Run the synthetic demo

```powershell
python scripts\run_demo.py
```

The command trains and saves a synthetic GAAP model, exercises prediction and
delayed-label API routes in-process, runs the offline drift/adaptation
experiment, and writes a manifest and outputs to a fresh timestamped directory
under `artifacts\demo`. It requires no external banking dataset. The API's
delayed-label flow remains monitoring-only and does not adapt or hot-promote
its loaded model.

## Reproduce evaluation results

Full configured run:

```powershell
python scripts\reproduce_results.py
```

Short verification run:

```powershell
python scripts\reproduce_results.py --smoke
```

Each run gets distinct timestamped report and working directories under
`reports\reproductions` and `artifacts\reproductions`, respectively. Reports
record the seed and actual observed metrics; unavailable comparisons remain
unavailable rather than being fabricated.

## Run tests

```powershell
python scripts\run_all_tests.py
```

Equivalent direct command:

```powershell
python -m pytest -q
```

## Security and operations before deployment

- Put the API behind authenticated, authorized transport and enforce
  per-tenant access controls and rate limits.
- Do not expose the unauthenticated delayed-label route to untrusted clients.
- Configure data retention, encryption, access logging, secrets management,
  alerting, and incident response according to institutional requirements.
- Benchmark p95/p99 request latency with production-sized history and
  concurrent load; current scoring recomputes graph/features per request.
- Establish model approval, drift response, rollback, and schema migration
  procedures. No live adaptation/promotion is wired into the API.
- Run temporal-leakage checks before using reported metrics as research claims.
