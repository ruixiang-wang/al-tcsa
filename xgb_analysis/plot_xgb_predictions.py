#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate XGB prediction tables and scatter plots for G0/G1/G5.

Input models (per tag):
  artifacts/models/<tag>/xgboost_bayes_best.joblib

Input data (per tag):
  data/raw/NO-<tag>-all.csv

Outputs:
  artifacts/predictions/pred_G<tag>_xgb.csv
  artifacts/plots/fig_xgb_G<tag>.png
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---- basic config ----
TARGET_COL = "Gn(eV)"
TEST_SIZE = 0.2
RANDOM_SEED = 2025
TAGS = ["G0", "G1", "G5"]

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
PRED_DIR = PROJECT_ROOT / "artifacts" / "predictions"
PLOT_DIR = PROJECT_ROOT / "artifacts" / "plots"

PRED_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---- helpers ----
def load_data(path: Path, target_col: str):
    """Load CSV, drop NaN on target, drop 'catalyst' id, return X/y/feature_names."""
    df = pd.read_csv(path)
    df = df.dropna(subset=[target_col])

    drop_cols = ["catalyst"]
    drop_cols = [c for c in drop_cols if c in df.columns and c != target_col]

    y = df[target_col].values
    feat_df = df.drop(columns=[target_col] + drop_cols)

    feature_names = list(feat_df.columns)
    X = feat_df.values

    print(
        f"[load_data] {path.name} -> "
        f"samples={X.shape[0]}, features={X.shape[1]}"
    )
    print("Feature columns used:", feature_names)

    return X, y, feature_names


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def plot_scatter(
    tag: str,
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    out_path: Path,
):
    """Scatter plot: Detected vs Predicted Gn (train + test)."""
    train_mae, train_rmse, train_r2 = compute_metrics(y_train, y_train_pred)
    test_mae, test_rmse, test_r2 = compute_metrics(y_test, y_test_pred)

    all_true = np.concatenate([y_train, y_test])
    vmin = float(all_true.min())
    vmax = float(all_true.max())
    if vmax > vmin:
        margin = 0.05 * (vmax - vmin)
    else:
        margin = 1.0
    xmin, xmax = vmin - margin, vmax + margin

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    ax.scatter(
        y_train,
        y_train_pred,
        s=35,
        alpha=0.8,
        edgecolors="none",
        label="Train",
        c="#00c5ff",
    )
    ax.scatter(
        y_test,
        y_test_pred,
        s=35,
        alpha=0.8,
        edgecolors="none",
        label="Test",
        c="#ff4fb3",
    )

    ax.plot([xmin, xmax], [xmin, xmax], linestyle="-", linewidth=1.0, color="black")

    ax.set_xlabel("Detected Gn (eV)", fontsize=11)
    ax.set_ylabel("Predicted Gn (eV)", fontsize=11)
    ax.set_title(f"XGB ({tag})", fontsize=12)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_aspect("equal", "box")
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    textstr = (
        "Train:\n"
        f"$R^2$ = {train_r2:.2f}\n"
        f"RMSE = {train_rmse:.2f}\n"
        f"MAE = {train_mae:.2f}\n\n"
        "Test:\n"
        f"$R^2$ = {test_r2:.2f}\n"
        f"RMSE = {test_rmse:.2f}\n"
        f"MAE = {test_mae:.2f}"
    )

    ax.text(
        0.55,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] Saved figure to: {out_path}")


def process_tag(tag: str):
    """One full pipeline for a single tag."""
    print("\n" + "=" * 80)
    print(f"Processing tag {tag}")
    print("=" * 80)

    data_path = RAW_DATA_DIR / f"NO-{tag}-all.csv"
    model_path = MODELS_DIR / tag / "xgboost_bayes_best.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    X, y, feature_names = load_data(data_path, TARGET_COL)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    print(f"[split] Train size={X_train.shape[0]}, Test size={X_test.shape[0]}")

    model = joblib.load(model_path)
    print(f"[model] Loaded model from: {model_path}")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae, train_rmse, train_r2 = compute_metrics(y_train, y_train_pred)
    test_mae, test_rmse, test_r2 = compute_metrics(y_test, y_test_pred)
    print(
        f"[metrics][{tag}] Train: R2={train_r2:.4f}, "
        f"RMSE={train_rmse:.4f}, MAE={train_mae:.4f}"
    )
    print(
        f"[metrics][{tag}] Test : R2={test_r2:.4f}, "
        f"RMSE={test_rmse:.4f}, MAE={test_mae:.4f}"
    )

    split_col = np.array(["Train"] * len(y_train) + ["Test"] * len(y_test))
    y_true_all = np.concatenate([y_train, y_test])
    y_pred_all = np.concatenate([y_train_pred, y_test_pred])

    df_pred = pd.DataFrame(
        {
            "split": split_col,
            "Gn_true": y_true_all,
            "Gn_pred": y_pred_all,
        }
    )

    out_csv = PRED_DIR / f"pred_G{tag}_xgb.csv"
    df_pred.to_csv(out_csv, index=False)
    print(f"[table] Saved prediction table to: {out_csv}")

    out_png = PLOT_DIR / f"fig_xgb_G{tag}.png"
    plot_scatter(
        tag=tag,
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        out_path=out_png,
    )


def main():
    for tag in TAGS:
        process_tag(tag)


if __name__ == "__main__":
    main()