#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP force plots for selected G0 catalysts.

For G0:
  - load NO-G0-all*.xlsx / NO-G0-all*.csv (prefer Excel)
  - load best XGB pipeline
  - compute SHAP for all samples
  - save global feature-importance table
  - draw shap.force_plot for a list of target catalysts

Inputs (relative to project root):
  data/raw/NO-G0-all 1.xlsx  (preferred, if present)
  data/raw/NO-G0-all 1.csv
  data/raw/NO-G0-all.xlsx
  data/raw/NO-G0-all.csv
  artifacts/models/G0/xgboost_bayes_best.joblib

Outputs:
  artifacts/shap/tables/shap_G0_xgb_feature_importance.csv
  artifacts/shap/force_plots/shap_G0_force_<catalyst>_idx<index>.png
"""

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import shap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_TARGET_COL,
    compute_metrics,
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

TARGET_COL = DEFAULT_TARGET_COL

# keep the original basename (Excel / CSV with " 1")
DATA_BASENAMES: List[str] = [
    "NO-G0-all 1",
    "NO-G0-all",
]

MODEL_PATH = MODELS_ROOT / "G0" / "xgboost_bayes_best.joblib"

TARGET_CATALYSTS = [
    "TiPy-DO_need",
    "TiPy-DF_need",
    "ScPy-DN_need",
    "ScPy-DO_need",
    "CrPy-DF_need",
    "WPy-DF_need",
]

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


def load_no_g0_dataframe() -> pd.DataFrame:
    """Prefer Excel; fall back to CSV (gbk)."""
    candidates = []
    for base in DATA_BASENAMES:
        candidates.append(RAW_DATA_DIR / f"{base}.xlsx")
        candidates.append(RAW_DATA_DIR / f"{base}.csv")

    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".xlsx":
            print(f"[data] Reading Excel: {path}")
            return pd.read_excel(path)
        else:
            print(f"[data] Reading CSV (gbk): {path}")
            return pd.read_csv(path, encoding="gbk")

    raise FileNotFoundError(
        "No G0 data file found. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def get_base_value(explainer) -> float:
    """Flatten explainer.expected_value to a scalar float."""
    base = explainer.expected_value
    if isinstance(base, (np.ndarray, list)):
        base = float(np.asarray(base).ravel()[0])
    else:
        base = float(base)
    return base


# ---------- main ----------
def main():
    # 1) load data
    df = load_no_g0_dataframe()
    df = df.dropna(subset=[TARGET_COL])

    print(f"[data] shape={df.shape}")
    print("[data] columns:", list(df.columns))

    feature_cols = [c for c in df.columns if c not in ("catalyst", TARGET_COL)]
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    # 2) load trained G0 pipeline
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    pipe = joblib.load(MODEL_PATH)
    print(f"[model] Loaded pipeline from: {MODEL_PATH}")

    try:
        scaler = pipe.named_steps.get("scaler", None)
        xgb_model = pipe.named_steps["model"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected pipeline structure (no scaler/model): {exc}")

    # 3) quick sanity metrics (optional)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=DEFAULT_RANDOM_SEED, shuffle=True
    )

    if scaler is not None:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)
    train_mae, train_rmse, train_r2 = compute_metrics(y_train, y_train_pred)
    test_mae, test_rmse, test_r2 = compute_metrics(y_test, y_test_pred)
    print(
        f"[metrics][G0] Train: "
        f"R2={train_r2:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f}"
    )
    print(
        f"[metrics][G0] Test : "
        f"R2={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}"
    )

    # 4) SHAP on all samples (use scaled features)
    if scaler is not None:
        X_all_scaled = scaler.transform(X)
    else:
        X_all_scaled = X

    print("[shap] Building TreeExplainer ...")
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values_all = explainer.shap_values(X_all_scaled)
        shap_values_all = _ensure_2d_array(shap_values_all)
        print("[shap] Using TreeExplainer.")
    except Exception as exc:
        print("[shap] TreeExplainer failed, fallback to KernelExplainer:", repr(exc))
        background = (
            X_all_scaled
            if X_all_scaled.shape[0] <= 60
            else shap.sample(X_all_scaled, 60, random_state=DEFAULT_RANDOM_SEED)
        )
        f = lambda data: xgb_model.predict(data)
        explainer = shap.KernelExplainer(f, background)
        shap_values_all = explainer.shap_values(X_all_scaled)
        shap_values_all = _ensure_2d_array(shap_values_all)
        print("[shap] Using KernelExplainer.")

    n_samples, n_features = shap_values_all.shape
    assert n_features == len(feature_cols)
    print(f"[shap] SHAP shape = {shap_values_all.shape}")

    base_value = get_base_value(explainer)
    print(f"[shap] global base_value (expected_value) = {base_value:.6f}")

    # 5) global feature importance
    mean_abs_all = np.mean(np.abs(shap_values_all), axis=0)
    df_imp = (
        pd.DataFrame(
            {"feature": feature_cols, "mean_abs_shap_all": mean_abs_all}
        )
        .sort_values("mean_abs_shap_all", ascending=False)
        .reset_index(drop=True)
    )
    out_csv = TABLE_DIR / "shap_G0_xgb_feature_importance.csv"
    df_imp.to_csv(out_csv, index=False)
    print(f"[table] Saved feature importance to: {out_csv}")

    # 6) force plots for selected catalysts
    has_md = "MD" in df.columns

    for cat in TARGET_CATALYSTS:
        mask = df["catalyst"] == cat
        if not mask.any():
            print(f"[warn] catalyst == {cat} not found, skip")
            continue

        sub = df[mask]
        if has_md and (sub["MD"] == 1).any():
            sub = sub[sub["MD"] == 1]

        row = sub.iloc[0]
        idx = row.name  # original DataFrame index

        shap_row = shap_values_all[idx, :]
        x_row = df.loc[idx, feature_cols]
        fx = pipe.predict(X[idx : idx + 1])[0]
        sum_shap = float(shap_row.sum())
        fx_from_shap = base_value + sum_shap

        print(
            f"[sample] catalyst={cat}, index={idx}, "
            f"base_value={base_value:.6f}, "
            f"sum_shap={sum_shap:.6f}, "
            f"base+sum={fx_from_shap:.6f}, "
            f"f(x)={fx:.6f}"
        )

        feature_labels = []
        for name in feature_cols:
            val = x_row[name]
            if isinstance(val, (int, float, np.floating)):
                s_val = f"{val:.3g}"
            else:
                s_val = str(val)
            feature_labels.append(f"{name}={s_val}")

        plt.close("all")
        plt.figure(figsize=(8.0, 1.8))

        shap.force_plot(
            base_value,
            shap_row,
            feature_labels,
            matplotlib=True,
            show=False,
        )

        safe_cat = cat.replace("/", "_").replace(" ", "_")
        out_png = FORCE_PLOT_DIR / f"shap_G0_force_{safe_cat}_idx{idx}.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[plot] Saved force plot for {cat} to: {out_png}")


if __name__ == "__main__":
    main()