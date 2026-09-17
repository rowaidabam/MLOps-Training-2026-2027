# MLOps Training 2026–2027 — Task 3

## Overview

This project turns a set of Jupyter notebooks (order-level feature engineering + a Random Forest classifier predicting late deliveries) into a production-style inference service: versioned data and models, validated input, tracked experiments, a FastAPI service, a Docker image, and a CI pipeline.

Training happens in the notebooks only. This project loads the already-fitted preprocessor and model — it never re-fits anything.

## Project Structure

```text
Task 3/
├── app/            # FastAPI service (routes, request/response schemas)
├── config/         # config.yaml — all paths and settings, no hardcoding
├── data/           # raw/processed data (DVC-tracked)
├── models/         # fitted preprocessor, model, threshold, metrics (DVC-tracked)
├── notebooks/      # original training notebooks (not used at inference time)
├── requirements/   # runtime.txt (what the service needs) vs dev.txt (full local dev freeze)
├── scripts/        # one-off scripts: verify_pipeline, register_model, set_model_alias, check_registry
├── src/            # reusable pipeline code: features, predict, validation, service, config, exceptions
├── tests/          # pytest unit + integration tests
├── artifacts/       # earlier working copies of data/model outputs (superseded by data/ and models/)
├── Dockerfile
├── .dockerignore
└── .pre-commit-config.yaml
```

## Setup, from zero

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements/dev.txt
```

Pull the DVC-tracked data and model files (requires your own Google Drive OAuth client — see "DVC remote" below):

```powershell
dvc pull
```

## Running the pipeline manually

```powershell
python scripts/verify_pipeline.py
```

Reproduces the notebook's exact test-set metrics from the raw data through the saved preprocessor and model.

## Running the tests

```powershell
pytest
```

Runs all tests. Some require extra setup (see "Known limitations" below):
- `test_features.py` — pure unit tests, no setup needed
- `test_pipeline_reproduction.py`, `test_service.py` — need `models/` and `data/` present (`dvc pull`)
- `test_registry.py`, `test_api.py` — need a running MLflow server (see below)

## MLflow — experiment tracking and model registry

Start the tracking server (in its own terminal, left running):

```powershell
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

View it at http://127.0.0.1:5000. Register the model (only needed once, or after retraining):

```powershell
python scripts/register_model.py
python scripts/set_model_alias.py
```

The service loads the model from the registry (`late_delivery_classifier@staging`) at startup, falling back to the local `models/random_forest_model.joblib` if the registry is unreachable.

## Running the API

```powershell
python -m uvicorn app.main:app --reload
```

Docs at http://127.0.0.1:8000/docs. Routes: `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch`.

## Docker

```powershell
docker build -t late-delivery-api .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 late-delivery-api
```

## DVC remote

Data and model artifacts are versioned with DVC, stored in Google Drive. Each user needs their own Google Cloud OAuth client (Desktop app type) with the Drive API enabled, then:

```powershell
dvc remote modify storage --local gdrive_client_id "YOUR_CLIENT_ID"
dvc remote modify storage --local gdrive_client_secret "YOUR_CLIENT_SECRET"
dvc pull
```

Credentials are kept in `.dvc/config.local`, which is gitignored and never committed.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main`: installs `requirements/runtime.txt`, runs `pre-commit` checks, and runs `tests/test_features.py`.

## Known limitations / not in CI

- CI runs pre-commit checks and `tests/test_features.py` only. `test_service.py`, `test_pipeline_reproduction.py`, and `test_registry.py` require DVC-pulled model artifacts or a running MLflow server, which aren't available in a clean GitHub Actions checkout — run these locally instead.
- The Docker image build is verified locally (builds successfully, container serves real predictions) but is not run in CI, since it also needs the DVC-pulled `models/` folder. A production setup would add DVC remote credentials as GitHub Secrets and a `dvc pull` step before both the tests and the build.
- The containerized API tries to load the model from the MLflow registry at startup and falls back to the local `models/` artifact if unreachable. Locally, the registry connection itself succeeds, but downloading the model artifact fails, because the MLflow server here uses local filesystem artifact storage rather than a remote/proxied store reachable from inside the container. A production setup would use a remote artifact store (S3, Azure Blob) or MLflow's `--serve-artifacts` proxy mode.
