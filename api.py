"""
FastAPI deployment: churn probability, binary prediction, optional SHAP explanations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.environ.get("CHURN_ARTIFACT_DIR", BASE_DIR / "artifacts"))
PIPELINE_PATH = ARTIFACT_DIR / "churn_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


class TelcoCustomerInput(BaseModel):
    """Raw Telco row (same schema as IBM CSV; customerID optional)."""

    customerID: str | None = Field(default=None, description="Optional row id")
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str | float | int | None = Field(
        default="",
        description="Omit or leave empty for missing; imputed like training (median).",
    )

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def empty_total_charges(cls, v):
        if v is None:
            return ""
        if isinstance(v, str) and not str(v).strip():
            return ""
        return v


class PredictResponse(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_predicted: int = Field(..., ge=0, le=1)
    threshold: float
    shap_top_features: list[dict[str, Any]] | None = None


def load_pipeline():
    if not PIPELINE_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {PIPELINE_PATH}. Run train.py first.",
        )
    return joblib.load(PIPELINE_PATH)


def load_threshold() -> float:
    if not METRICS_PATH.is_file():
        return 0.5
    with open(METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("classification_threshold", 0.5))


app = FastAPI(title="Telco Churn API", version="1.0.0")
_pipeline = None
_threshold_cache: float | None = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline()
    return _pipeline


def get_threshold() -> float:
    global _threshold_cache
    if _threshold_cache is None:
        _threshold_cache = load_threshold()
    return _threshold_cache


def row_to_dataframe(payload: TelcoCustomerInput) -> pd.DataFrame:
    d = payload.model_dump()
    return pd.DataFrame([d])


def encoded_feature_names(pipe) -> np.ndarray:
    encode_step = pipe.named_steps["encode"]
    return encode_step.get_feature_names_out()


def shap_explanation(pipe, X_df: pd.DataFrame, top_k: int = 12) -> list[dict[str, Any]]:
    import shap

    telco = pipe.named_steps["telco"]
    encode = pipe.named_steps["encode"]
    model = pipe.named_steps["model"]
    X_telco = telco.transform(X_df)
    X_enc = encode.transform(X_telco)
    fnames = encoded_feature_names(pipe)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.TreeExplainer(model.get_booster())
    sv = explainer.shap_values(X_enc)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 2:
        sv = sv[0]
    sv = sv.ravel()
    order = np.argsort(np.abs(sv))[::-1][:top_k]
    return [{"feature": str(fnames[i]), "shap_value": float(sv[i])} for i in order]


@app.get("/health")
def health():
    return {"status": "ok", "artifact": str(PIPELINE_PATH)}


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: TelcoCustomerInput,
    explain: bool = Query(False, description="Include SHAP top contributions"),
    top_k: int = Query(12, ge=1, le=50),
):
    pipe = get_pipeline()
    threshold = get_threshold()
    X = row_to_dataframe(body)
    proba = float(pipe.predict_proba(X)[0, 1])
    churn_pred = int(proba >= threshold)
    shap_part = shap_explanation(pipe, X, top_k=top_k) if explain else None
    return PredictResponse(
        churn_probability=proba,
        churn_predicted=churn_pred,
        threshold=threshold,
        shap_top_features=shap_part,
    )

@app.get("/")
def root():
    return {"message": "Telco Churn API is running. See /docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
