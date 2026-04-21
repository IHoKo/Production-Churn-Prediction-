"""Preprocessing utilities for IBM Telco Customer Churn data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def load_raw_churn_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def churn_target_to_binary(series: pd.Series) -> pd.Series:
    return series.map({"Yes": 1, "No": 0}).astype(np.int8)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tenure = out["tenure"].astype(float)
    mc = out["MonthlyCharges"].astype(float)
    tc = out["TotalCharges"].astype(float)
    out["tenure_bucket"] = pd.cut(
        tenure,
        bins=[-np.inf, 12, 24, np.inf],
        labels=["0-12", "12-24", "24+"],
        ordered=False,
    ).astype(str)
    out["charge_per_tenure"] = mc / np.maximum(tenure, 1.0)
    out["log_monthly_charges"] = np.log1p(mc)
    out["log_total_charges"] = np.log1p(np.maximum(tc, 0.0))
    # Expected lifetime bill vs actual (rough contract value signal).
    out["total_vs_expected"] = tc / np.maximum(mc * np.maximum(tenure, 1.0), 1e-6)
    return out


CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_bucket",
]

NUMERIC_COLS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charge_per_tenure",
    "log_monthly_charges",
    "log_total_charges",
    "total_vs_expected",
]


class TelcoPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocess Telco rows for training/inference.

    Drops `customerID`, imputes `TotalCharges` with the training median,
    and adds engineered numeric/categorical features.
    """

    total_charges_median_: float

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TelcoPreprocessor:
        del y
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
        self.total_charges_median_ = float(tc.median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        out = df.copy()
        if "customerID" in out.columns:
            out = out.drop(columns=["customerID"])
        if "Churn" in out.columns:
            out = out.drop(columns=["Churn"])
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(
            self.total_charges_median_
        )
        return add_engineered_features(out)
