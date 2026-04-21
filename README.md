# Production Churn Prediction

End-to-end customer churn prediction system using XGBoost, with SHAP explainability and FastAPI deployment.

## Dependency management (Poetry)

This repository now uses Poetry as the source of truth for dependencies.

```bash
poetry install
```

If your machine has OpenMP issues loading XGBoost on macOS, install:

```bash
brew install libomp
```

## Run the project

Train model and write artifacts:

```bash
poetry run python train.py --fast
```

Generate EDA figures:

```bash
poetry run python eda.py
```

Serve API:

```bash
poetry run uvicorn api:app --host 0.0.0.0 --port 8000
```

## Lint and format (Ruff)

```bash
poetry run ruff check . --fix
poetry run ruff format .
```
