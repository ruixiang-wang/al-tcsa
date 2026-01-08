#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGB + SHAP force plots (G0 / G1 / G5).

For each tag:
  - load NO-<tag>-all.csv
  - load best XGB pipeline
  - compute SHAP values
  - save feature-importance table
  - draw SHAP force plots for a batch of samples

Inputs
  data/raw/NO-G0-all.csv, NO-G1-all.csv, NO-G5-all.csv
  artifacts/models/<tag>/xgboost_bayes_best.joblib

Outputs
  artifacts/shap/tables/shap_<tag>_xgb_feature_importance.csv
  artifacts/shap/force_plots/shap_<tag>_force_sample*.png
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils import (
    load_csv_dataset,
    compute_metrics,
    DEFAULT_TARGET_COL,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_SEED,
)

# ---------- paths & config ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_ROOT = PROJECT_ROOT / "artifacts" / "models"
SHAP_ROOT = PROJECT_ROOT / "artifacts" / "shap"

TABLE_DIR = SHAP_ROOT / "tables"
FORCE_PLOT_DIR = SHAP_ROOT / "force_plots"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FORCE_PLOT_DIR.mkdir(parents=True, exist_ok=True)

TAGS = ["G0", "G1", "G5"]
TEST_SIZE = DEFAULT_TEST_SIZE
RANDOM_SEED = DEFAULT_RANDOM_SEED

DATA_PATHS = {tag: RAW_DATA_DIR / f"NO-{tag}-all.csv" for tag in TAGS}
MODEL_PATHS = {
    tag: MODELS_ROOT / tag / "xgboost_bayes_best.joblib" for tag in TAGS
}

matplotlib.rcParams.update(
    {
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
    }
)


# ---------- helpers ----------
def _ensure_2d_array(shap_values):
    """Normalize SHAP return to shape (n_samples, n_features)."""
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values.mean(axis=0)
    return shap_values


def save_force_plots_for_tag(
    tag: str,
    explainer,
    shap_values_all: np.ndarray,
    X_all_orig: np.ndarray,
    feature_names,
    sample_indices,
):
    """Draw shap.force_plot for selected samples and save as PNG files."""
    shap_values_all = np.asarray(shap_values_all)
    X_all_orig = np.asarray(X_all_orig)

    base_value = explainer.expected_value

    for idx in sample_indices:
        shap_row = shap_values_all[idx, :]
        x_row = X_all_orig[idx, :]

        feature_labels = [
            f"{name}={val:.3g}" for name, val in zip(feature_names, x_row)
        ]

        plt.close("all")
        plt.figure(figsize=(8.0, 1.5))

        shap.force_plot(
            base_value,
            shap_row,
            feature_labels,
            matplotlib=True,
            show=False,
        )

        out_png = FORCE_PLOT_DIR / f"shap_{tag}_force_sample{idx}.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[plot] Saved force plot to: {out_png}")


# ---------- main per-tag logic ----------
def compute_and_plot_for_tag(tag: str):
    print("\n" + "=" * 80)
    print(f"SHAP force plots for tag {tag}")
    print("=" * 80)

    data_path = DATA_PATHS[tag]
    model_path = MODEL_PATHS[tag]

    X, y, feature_names = load_csv_dataset(
        data_path, target_col=DEFAULT_TARGET_COL
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    print(
        f"[split] Train size={X_train.shape[0]}, "
        f"Test size={X_test.shape[0]}, features={X_train.shape[1]}"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    pipe = joblib.load(model_path)
    print(f"[model] Loaded pipeline from: {model_path}")

    try:
        scaler = pipe.named_steps.get("scaler", None)
        xgb_model = pipe.named_steps["model"]
    except Exception as exc:
        raise RuntimeError(f"Pipeline does not have expected steps: {exc}")

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)
    train_mae, train_rmse, train_r2 = compute_metrics(y_train, y_train_pred)
    test_mae, test_rmse, test_r2 = compute_metrics(y_test, y_test_pred)
    print(
        f"[metrics][{tag}] Train: "
        f"R2={train_r2:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}"
    )
    print(
        f"[metrics][{tag}] Test : "
        f"R2={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}"
    )

    if scaler is not None:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    X_all_scaled = np.concatenate([X_train_scaled, X_test_scaled], axis=0)
    X_all_orig = np.concatenate([X_train, X_test], axis=0)

    print("[shap] Building explainer ...")
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values_all = explainer.shap_values(X_all_scaled)
        shap_values_all = _ensure_2d_array(shap_values_all)
        print("[shap] Using TreeExplainer.")
    except Exception as exc:
        print("[shap] TreeExplainer failed, fallback to KernelExplainer:")
        print("       ", repr(exc))
        background = X_train_scaled
        if background.shape[0] > 60:
            rng = np.random.default_rng(RANDOM_SEED)
            idx = rng.choice(background.shape[0], size=60, replace=False)
            background = background[idx]
        f = lambda data: xgb_model.predict(data)
        explainer = shap.KernelExplainer(f, background)
        shap_values_all = explainer.shap_values(X_all_scaled)
        shap_values_all = _ensure_2d_array(shap_values_all)
        print("[shap] Using KernelExplainer.")

    n_samples, n_features = shap_values_all.shape
    assert n_features == len(feature_names)
    print(
        f"[shap] Got SHAP values: samples={n_samples}, "
        f"features={n_features}"
    )

    mean_abs_all = np.mean(np.abs(shap_values_all), axis=0)
    df_shap = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap_all": mean_abs_all,
            }
        )
        .sort_values("mean_abs_shap_all", ascending=False)
        .reset_index(drop=True)
    )
    out_csv = TABLE_DIR / f"shap_{tag}_xgb_feature_importance.csv"
    df_shap.to_csv(out_csv, index=False)
    print(f"[table] Saved SHAP feature importance to: {out_csv}")

    n_plot = min(10, n_samples)
    sample_indices = list(range(n_plot))

    save_force_plots_for_tag(
        tag=tag,
        explainer=explainer,
        shap_values_all=shap_values_all,
        X_all_orig=X_all_orig,
        feature_names=feature_names,
        sample_indices=sample_indices,
    )


def main():
    for tag in TAGS:
        compute_and_plot_for_tag(tag)


if __name__ == "__main__":
    main()