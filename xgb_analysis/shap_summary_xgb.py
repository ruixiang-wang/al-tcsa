#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute SHAP values for the three XGB models (G0, G1, G5),
save feature-level SHAP tables, and draw custom SHAP beeswarm plots.

Inputs:
  data/raw/NO-G0-all.csv
  data/raw/NO-G1-all.csv
  data/raw/NO-G5-all.csv

  artifacts/models/G0/xgboost_bayes_best.joblib
  artifacts/models/G1/xgboost_bayes_best.joblib
  artifacts/models/G5/xgboost_bayes_best.joblib

Outputs:
  artifacts/shap/tables/shap_G<G>-xgb_feature_importance.csv
  artifacts/shap/summary_plots/shap_G<G>_xgb_summary.png
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# ---------- basic config ----------
TARGET_COL = "Gn(eV)"
TEST_SIZE = 0.2
RANDOM_SEED = 2025
TAGS = ["G0", "G1", "G5"]

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
SHAP_ROOT = PROJECT_ROOT / "artifacts" / "shap"
TABLE_DIR = SHAP_ROOT / "tables"
PLOT_DIR = SHAP_ROOT / "summary_plots"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
    }
)


def make_shap_cmap() -> LinearSegmentedColormap:
    """Custom colormap: blue -> off-white -> orange."""
    colors = ["#4F8CCF", "#FBF3E4", "#E37A38"]
    return LinearSegmentedColormap.from_list(
        "orange_white_blue_soft_deeper", colors
    )


SHAP_CMAP = make_shap_cmap()


# ---------- helpers ----------
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
        f"[load_data] {path.name} -> samples={X.shape[0]}, features={X.shape[1]}"
    )
    print("Feature columns:", feature_names)
    return X, y, feature_names


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def _ensure_2d_array(shap_values):
    """Normalize SHAP output to (n_samples, n_features)."""
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values.mean(axis=0)
    return shap_values


def plot_shap_beeswarm_custom(
    shap_values_all: np.ndarray,
    X_all_orig: np.ndarray,
    feature_names,
    df_shap: pd.DataFrame,
    tag: str,
    train_r2: float,
    test_r2: float,
    test_mae: float,
    max_display: int = 10,
):
    """Custom SHAP beeswarm plot with our colormap."""
    rng = np.random.default_rng(RANDOM_SEED)

    feature_order = df_shap["feature"].values[:max_display]
    feat_index = {name: i for i, name in enumerate(feature_names)}

    n_features_plot = len(feature_order)
    n_samples = shap_values_all.shape[0]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    last_scatter = None

    for row_idx, feat_name in enumerate(feature_order):
        col_idx = feat_index[feat_name]

        shap_col = shap_values_all[:, col_idx]
        feat_values = X_all_orig[:, col_idx]

        v_low, v_high = np.percentile(feat_values, [5, 95])
        if v_high == v_low:
            norm_v = np.zeros_like(feat_values)
        else:
            norm_v = (feat_values - v_low) / (v_high - v_low)
        norm_v = np.clip(norm_v, 0.0, 1.0)

        base_y = row_idx
        jitter = rng.normal(loc=0.0, scale=0.10, size=n_samples)
        y_vals = base_y + jitter

        last_scatter = ax.scatter(
            shap_col,
            y_vals,
            c=norm_v,
            cmap=SHAP_CMAP,
            s=26,
            alpha=0.7,
            linewidths=0.0,
        )

    ax.axvline(x=0.0, color="#999999", linestyle="--", linewidth=0.8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)

    ax.set_xlabel("SHAP value", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)

    ax.set_yticks(range(n_features_plot))
    ax.set_yticklabels(feature_order)
    ax.invert_yaxis()

    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=ax)
        cbar.set_label("Feature value", fontsize=10)
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(["Low", "High"])

    metrics_str = (
        f"Train R\u00b2={train_r2:.2f}, Test R\u00b2={test_r2:.2f}, "
        f"Test MAE={test_mae:.2f}"
    )
    ax.set_title(f"XGB SHAP summary ({tag})\n{metrics_str}", fontsize=11)

    plt.tight_layout()
    out_png = PLOT_DIR / f"shap_{tag}_xgb_summary.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved SHAP summary plot to: {out_png}")


def compute_and_save_shap_for_tag(tag: str):
    """Full SHAP pipeline for one tag."""
    print("\n" + "=" * 80)
    print(f"Processing SHAP for {tag}")
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
        f"[metrics][{tag}] Train: R2={train_r2:.4f}, "
        f"RMSE={train_rmse:.4f}, MAE={train_mae:.4f}"
    )
    print(
        f"[metrics][{tag}] Test : R2={test_r2:.4f}, "
        f"RMSE={test_rmse:.4f}, MAE={test_mae:.4f}"
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
        if X_train_scaled.shape[0] > 60:
            background = shap.sample(
                X_train_scaled, 60, random_state=RANDOM_SEED
            )
        else:
            background = X_train_scaled
        f = lambda data: xgb_model.predict(data)
        explainer = shap.KernelExplainer(f, background)
        shap_values_all = explainer.shap_values(X_all_scaled)
        shap_values_all = _ensure_2d_array(shap_values_all)
        print("[shap] Using KernelExplainer.")

    n_samples, n_features = shap_values_all.shape
    assert n_features == len(feature_names)

    n_train = X_train.shape[0]
    shap_train = shap_values_all[:n_train, :]
    shap_test = shap_values_all[n_train:, :]

    mean_abs_all = np.mean(np.abs(shap_values_all), axis=0)
    mean_abs_train = np.mean(np.abs(shap_train), axis=0)
    mean_abs_test = np.mean(np.abs(shap_test), axis=0)

    df_shap = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap_all": mean_abs_all,
                "mean_abs_shap_train": mean_abs_train,
                "mean_abs_shap_test": mean_abs_test,
            }
        )
        .sort_values("mean_abs_shap_all", ascending=False)
        .reset_index(drop=True)
    )

    out_csv = TABLE_DIR / f"shap_{tag}_xgb_feature_importance.csv"
    df_shap.to_csv(out_csv, index=False)
    print(f"[table] Saved SHAP feature importance to: {out_csv}")

    plot_shap_beeswarm_custom(
        shap_values_all=shap_values_all,
        X_all_orig=X_all_orig,
        feature_names=feature_names,
        df_shap=df_shap,
        tag=tag,
        train_r2=train_r2,
        test_r2=test_r2,
        test_mae=test_mae,
        max_display=min(10, len(feature_names)),
    )


def main():
    for tag in TAGS:
        compute_and_save_shap_for_tag(tag)


if __name__ == "__main__":
    main()
