"""
Exploratory data analysis for IBM Telco Customer Churn (CSV in data/).
Run: python eda.py [--data path] [--out dir]
Writes summary prints and figures under eda_output/ by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from preprocess import churn_target_to_binary, load_raw_churn_csv


def _configure_matplotlib_dir() -> None:
    """Use a writable Matplotlib config dir for local/sandbox runs."""
    mpl_cfg = Path(__file__).resolve().parent / ".mplconfig"
    mpl_cfg.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))


def main() -> None:
    parser = argparse.ArgumentParser(description="Telco churn EDA plots and tables")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "Telco-Customer-Churn.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "eda_output",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    _configure_matplotlib_dir()

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120

    df = load_raw_churn_csv(args.data)
    print("=== Shape ===")
    print(df.shape)
    print("\n=== dtypes ===")
    print(df.dtypes)
    print("\n=== Missing (per column) ===")
    miss = df.isna().sum()
    print(miss[miss > 0] if miss.any() else "No pandas NA values")
    tc_num = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_bad_tc = int(tc_num.isna().sum())
    if n_bad_tc:
        print(f"TotalCharges non-numeric / empty: {n_bad_tc} rows (imputed in training)")

    y = churn_target_to_binary(df["Churn"])
    churn_rate = float(y.mean())
    print(f"\n=== Target ===\nChurn rate (1=yes): {churn_rate:.3f}")
    print(y.value_counts().rename({0: "No churn", 1: "Churn"}))

    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    plot_df = df.assign(
        Churn_bin=y,
        TotalCharges_num=tc,
        log_MonthlyCharges=np.log1p(df["MonthlyCharges"].astype(float)),
        log_TotalCharges=np.log1p(tc.fillna(tc.median())),
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        data=plot_df,
        x="Churn",
        hue="Churn",
        ax=ax,
        palette="Set2",
        order=["No", "Yes"],
        legend=False,
    )
    ax.set_title("Churn class balance")
    fig.tight_layout()
    fig.savefig(args.out / "01_churn_counts.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(
        data=plot_df,
        x="tenure",
        hue="Churn",
        bins=30,
        multiple="layer",
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[0],
    )
    axes[0].set_title("Tenure by churn")
    sns.kdeplot(
        data=plot_df,
        x="MonthlyCharges",
        hue="Churn",
        common_norm=False,
        ax=axes[1],
    )
    axes[1].set_title("MonthlyCharges by churn (density)")
    fig.tight_layout()
    fig.savefig(args.out / "02_tenure_monthly_by_churn.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=plot_df, x="Contract", y="MonthlyCharges", hue="Churn", ax=ax)
    ax.set_title("Monthly charges vs contract type")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(args.out / "03_contract_monthly_box.png")
    plt.close(fig)

    corr_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges_num", "Churn_bin"]
    cmat = plot_df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(cmat, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation (numeric + churn)")
    fig.tight_layout()
    fig.savefig(args.out / "04_correlation_heatmap.png")
    plt.close(fig)

    cat_cols = ["InternetService", "Contract", "PaymentMethod"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, cat_cols, strict=True):
        ct = pd.crosstab(plot_df[col], plot_df["Churn"], normalize="index").rename(
            columns={"No": "No churn", "Yes": "Churn"}
        )
        ct.plot(kind="bar", ax=ax, rot=15, color=["#8fd175", "#e86a6a"])
        ax.set_title(f"Churn rate by {col}")
        ax.set_ylabel("Proportion within category")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "05_categorical_churn_rate.png")
    plt.close(fig)

    print(f"\nFigures saved under: {args.out.resolve()}")


if __name__ == "__main__":
    main()
