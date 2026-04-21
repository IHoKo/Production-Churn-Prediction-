"""
Train XGBoost churn model on IBM Telco data: stratified split, F1-tuned hyperparameters
and probability threshold, evaluation, save pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
except Exception as exc:  # noqa: BLE001 — surface common macOS OpenMP issue
    err = str(exc).lower()
    if "libomp" in err or "libxgboost" in err or "openmp" in err:
        print(
            "XGBoost could not load its native library (often missing OpenMP on macOS).\n"
            "Fix: install OpenMP, e.g. `brew install libomp`, then run this script again.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    raise

from preprocess import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TelcoPreprocessor,
    churn_target_to_binary,
    load_raw_churn_csv,
)


def _column_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("num", "passthrough", NUMERIC_COLS),
        ]
    )


def build_model_pipeline(scale_pos_weight: float) -> Pipeline:
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )
    return Pipeline(
        steps=[
            ("telco", TelcoPreprocessor()),
            ("encode", _column_transformer()),
            ("model", clf),
        ]
    )


def default_strong_params(scale_pos_weight: float) -> Pipeline:
    """Hand-tuned baseline when --fast is used."""
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=550,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.88,
        colsample_bytree=0.82,
        colsample_bylevel=0.9,
        min_child_weight=2,
        gamma=0.4,
        reg_lambda=6.0,
        reg_alpha=0.3,
        max_delta_step=1,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )
    return Pipeline(
        steps=[
            ("telco", TelcoPreprocessor()),
            ("encode", _column_transformer()),
            ("model", clf),
        ]
    )


def tune_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    n_iter: int,
    cv_splits: int,
    random_state: int,
) -> RandomizedSearchCV:
    base = build_model_pipeline(scale_pos_weight)
    # Reasonable ranges; RandomizedSearchCV samples n_iter combinations
    param_distributions = {
        "model__n_estimators": randint(350, 900),
        "model__max_depth": randint(4, 10),
        "model__learning_rate": uniform(0.025, 0.11),
        "model__subsample": uniform(0.75, 0.23),
        "model__colsample_bytree": uniform(0.65, 0.3),
        "model__colsample_bylevel": uniform(0.7, 0.25),
        "model__min_child_weight": randint(1, 9),
        "model__gamma": uniform(0.0, 1.5),
        "model__reg_lambda": uniform(1.0, 12.0),
        "model__reg_alpha": uniform(0.0, 2.0),
        "model__max_delta_step": randint(0, 4),
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def best_f1_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_steps: int = 401,
) -> tuple[float, float]:
    """Return (threshold, best F1) by scanning probabilities."""
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t = 0.5
    best_f = 0.0
    for t in thresholds:
        f = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f = float(f)
            best_t = float(t)
    return best_t, best_f


def oof_threshold_for_pipeline(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> tuple[float, float]:
    """OOF probabilities then pick threshold maximizing F1 on training labels."""
    oof_proba = cross_val_predict(
        clone(pipe),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    return best_f1_threshold(y.values, oof_proba)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "Telco-Customer-Churn.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold (0–1). If omitted, chosen by OOF F1 on the training set.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip hyperparameter search; use a strong hand-tuned baseline + OOF threshold.",
    )
    parser.add_argument("--tune-iter", type=int, default=28, help="RandomizedSearchCV samples")
    parser.add_argument("--cv", type=int, default=5, help="CV folds for tuning and OOF threshold")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    raw = load_raw_churn_csv(args.data)
    y = churn_target_to_binary(raw["Churn"])
    X = raw.drop(columns=["Churn"])

    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = float(neg / max(pos, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    cv_oof = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state + 1)

    if args.fast:
        pipe = default_strong_params(scale_pos_weight)
        pipe.fit(X_train, y_train)
        print("Using fast baseline (no RandomizedSearchCV).")
    else:
        search = tune_hyperparams(
            X_train,
            y_train,
            scale_pos_weight=scale_pos_weight,
            n_iter=args.tune_iter,
            cv_splits=args.cv,
            random_state=args.random_state,
        )
        pipe = search.best_estimator_
        print("Best CV F1:", search.best_score_)
        print("Best params:", search.best_params_)

    oof_f1_at_fixed: float | None = None
    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold, oof_f1_at_fixed = oof_threshold_for_pipeline(pipe, X_train, y_train, cv_oof)
        print(f"OOF-selected threshold (max F1): {threshold:.4f} (OOF F1 ≈ {oof_f1_at_fixed:.4f})")

    proba_test = pipe.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred_test)),
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "f1": float(f1_score(y_test, pred_test, zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred_test, average="macro", zero_division=0)),
        "classification_threshold": threshold,
        "scale_pos_weight": scale_pos_weight,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate_train": float(y_train.mean()),
        "confusion_matrix": confusion_matrix(y_test, pred_test).tolist(),
        "classification_report": classification_report(
            y_test, pred_test, output_dict=True, zero_division=0
        ),
    }
    if oof_f1_at_fixed is not None:
        metrics["oof_f1_threshold_selection"] = float(oof_f1_at_fixed)

    print("Test accuracy:", metrics["accuracy"])
    print("Test ROC-AUC:", metrics["roc_auc"])
    print("Test F1 (churn=1):", metrics["f1"])
    print("Confusion matrix:\n", np.array(metrics["confusion_matrix"]))
    print(classification_report(y_test, pred_test, zero_division=0))

    artifact_path = args.out / "churn_pipeline.joblib"
    joblib.dump(pipe, artifact_path)
    with open(args.out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved pipeline to {artifact_path}")


if __name__ == "__main__":
    main()
