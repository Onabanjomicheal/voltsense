# Battery SoC Estimator

![CI/CD](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci-cd.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)

Estimates State of Charge (SoC) for an LG 18650HG2 lithium-ion cell from
voltage/current/temperature sensor readings, using a trained LSTM served via
a FastAPI REST API, deployed through a staging -> production pipeline.

See `reports/model_card.md` for data sources, metrics, and known limitations.

## Pipeline

```
data/raw (Mendeley dataset)
       |
       v
[ parse_raw ]  ---------------------------> data/interim
       |          src/data_processing/raw_parser.py
       v
[ estimate_soc ]  ------------------------> data/processed
       |          src/data_processing/soc_estimation.py
       |          (pseudo-OCV + Coulomb counting)
       v
[ validate ]  -----------------------------> pandera schema check
       |          src/data/validation.py
       v
[ train ]  --------------------------------> models/lstm_model.pth
       |          src/training/train.py            models/scaler.joblib
       |          (Optuna search + MLflow tracking
       |           + registry promotion)
       v
[ feature_importance ]  -------------------> reports/feature_importance.json
       |          src/training/feature_importance.py
       v
[ test ]  ----------------------------------> pytest (unit) + Docker integration test
       |
       v
[ build & scan ]  --------------------------> Docker image, Trivy vulnerability scan
       |
       v
[ deploy: staging ]  -----------------------> auto-deploy, no approval needed
       |
       v
   (manual check of staging URL)
       |
       v
[ deploy: production ]  --------------------> requires manual approval in GitHub
```

All stages are defined as code: `dvc.yaml` (data + training), `.github/workflows/ci-cd.yml` (test + deploy).

## Project structure
```
data/
  raw/          <- original Mendeley dataset (DVC-tracked, not in git)
  interim/      <- parsed, not yet SoC-labeled
  processed/    <- final, SoC-labeled, ready for training
models/         <- trained model + scaler (DVC-tracked)
notebooks/      <- exploratory work only, no training logic
reports/        <- model_card.md, feature_importance.json, figures
src/
  data/         <- loading, feature engineering, validation, windowing
  models/       <- LSTM definition
  training/     <- train.py, feature_importance.py
  serving/      <- FastAPI app + inference wrapper
  data_processing/  <- raw -> interim -> processed pipeline stages
  utils/        <- shared logging config
tests/          <- pytest suite
demo/           <- interactive browser demo, calls the live deployed API
configs/lstm.yaml  <- all hyperparameters, paths, and tracking config in one place
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
make install          # installs deps + pre-commit hooks
cp .env.example .env  # fill in your DagsHub token and Render deploy hooks
```

## Running the pipeline

```bash
dvc pull               # fetch data + trained model from the DagsHub remote
# -- or, to reprocess from raw data / retrain from scratch --
dvc repro               # runs parse_raw -> estimate_soc -> train -> feature_importance

make test               # run the test suite
make serve               # run the API locally at http://127.0.0.1:8000/docs
```

## Deployment

Two Render services, defined as code:
- `render-staging.yaml` -- auto-deploys on every push to `main` that passes CI
- `render-production.yaml` -- deploys only after manual approval in GitHub
  (Settings -> Environments -> `production` -> required reviewers)

See `.env.example` for the secrets you need to configure in GitHub Actions
(repo Settings -> Secrets and variables -> Actions) for CI/CD to work:
`DAGSHUB_TOKEN`, `RENDER_STAGING_DEPLOY_HOOK_URL`,
`RENDER_PRODUCTION_DEPLOY_HOOK_URL`, `STAGING_URL`, `PRODUCTION_URL`.

## Demo
Open `demo/index.html` in a browser (or serve it statically) -- it calls the
live deployed API's `/predict` endpoint and visualizes predicted vs. true SoC
across a few simulated drive-cycle scenarios. Update the `API_BASE_URL`
constant at the top of the file to point at your deployed staging or
production URL.

## Key fixes vs. the original reference implementation
This project started from a public reference notebook implementation with
several real bugs, found and fixed during development:
1. **Scaler leakage** -- original fit `StandardScaler` on the full dataset
   before train/val/test splitting.
2. **Broken early stopping** -- the stopping condition was implemented but
   never actually triggered.
3. **Optuna's search result was discarded** -- hyperparameter search ran, but
   final training used a hardcoded value regardless of what was found.
4. **Cross-file window contamination** -- LSTM input sequences could splice
   together the end of one drive-cycle file and the start of an unrelated one.
5. **CC_Capacity feature broke serving** -- meaningless in short real-time
   windows; removed (see `reports/model_card.md`).

Full details of each fix and its verification are in the git history and
`reports/model_card.md`.
