#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP dependence plots for XGB models (G0 / G1 / G5).

For each tag:
  - load NO-<tag>-all.csv
  - load best XGB pipeline
  - compute SHAP values for all samples / features
  - save long-format CSV with (sample, feature, value, shap)
  - draw dependence plot per feature with:
      x: feature value
      y: SHAP value
      color: feature value
      curve: polynomial or piecewise polynomial fit

Inputs
  data/raw/NO-G0-all.csv, NO-G1-all.csv, NO-G5-all.csv
  artifacts/models/<tag>/xgboost_bayes_best.joblib

Outputs
  artifacts/shap/dependence_points/shap_points_<tag>.csv
  artifacts/shap/dependence_plots/shap_dep_<tag>_<feature>.png
"""

from pathlib import Path
import re

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

DEPENDENCE_POINTS_DIR = SHAP_ROOT / "dependence_points"
DEPENDENCE_PLOTS_DIR = SHAP_ROOT / "dependence_plots"
DEPENDENCE_POINTS_DIR.mkdir(parents=True, exist_ok=True)
DEPENDENCE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TAGS = ["G0", "G1", "G5"]
TEST_SIZE = DEFAULT_TEST_SIZE
RANDOM_SEED = DEFAULT_RANDOM_SEED

DATA_PATHS = {
    tag: RAW_DATA_DIR / f"NO-{tag}-all.csv" for tag in TAGS
}
MODEL_PATHS = {
    tag: MODELS_ROOT / tag / "xgboost_bayes_best.joblib" for tag in TAGS
}

# (tag, feature) pairs that should use piecewise fit
SPECIAL_PIECEWISE = {
    "G1": {"MN", "TM-N/O-0.13"},
}


# ---------- small helpers ----------
def _ensure_2d_array(shap_values):
    """Normalize SHAP return to shape (n_samples, n_features)."""
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # (n_outputs, n_samples, n_features)
        shap_values = shap_values.mean(axis=0)
    return shap_values


def safe_feat_name(name: str) -> str:
    """Convert feature name to a filesystem-safe slug."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")


def _power_to_unicode(power: int) -> str:
    """Format exponent as unicode superscript when possible."""
    if power == 2:
        return "²"
    if power == 3:
        return "³"
    return f"^{power}"


def format_poly(coefs):
    """Format poly coefficients to a short string like 'y = ax² + bx + c'."""
    coefs = list(coefs)
    deg = len(coefs) - 1
    parts = []
    for i, a in enumerate(coefs):
        power = deg - i
        if abs(a) < 1e-8:
            continue

        if not parts:
            sign_str = "-" if a < 0 else ""
        else:
            sign_str = " - " if a < 0 else " + "
        abs_a = abs(a)

        if power > 1:
            sup = _power_to_unicode(power)
            coeff_str = f"{abs_a:.4f}"
            term = f"{coeff_str}x{sup}"
        elif power == 1:
            coeff_str = f"{abs_a:.2f}"
            term = f"{coeff_str}x"
        else:
            coeff_str = f"{abs_a:.2f}"
            term = f"{coeff_str}"

        parts.append(sign_str + term)

    if not parts:
        return "y = 0"
    return "y = " + "".join(parts)


def fit_single_poly(x, y, deg: int):
    """Fit a single polynomial and return (coefs, R²)."""
    coefs = np.polyfit(x, y, deg=deg)
    y_hat = np.polyval(coefs, x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coefs, r2


def fit_piecewise_best_split(
    x,
    y,
    deg: int = 2,
    min_segment_size: int = 8,
):
    """Brute-force best 2-segment polynomial fit; return split and coefs."""
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    n = len(x_sorted)

    if n < 2 * min_segment_size:
        raise RuntimeError(f"not enough points for piecewise fit: n={n}")

    y_mean = np.mean(y_sorted)
    ss_tot = np.sum((y_sorted - y_mean) ** 2)
    if ss_tot <= 0:
        raise RuntimeError("zero variance in y, piecewise R² undefined")

    best_r2 = -np.inf
    best_split = None
    best_coefs_left = None
    best_coefs_right = None

    for split_idx in range(min_segment_size, n - min_segment_size):
        xl = x_sorted[:split_idx]
        yl = y_sorted[:split_idx]
        xr = x_sorted[split_idx:]
        yr = y_sorted[split_idx:]

        if len(xl) <= deg or len(xr) <= deg:
            continue

        coefs_l = np.polyfit(xl, yl, deg=deg)
        coefs_r = np.polyfit(xr, yr, deg=deg)

        yhat_l = np.polyval(coefs_l, xl)
        yhat_r = np.polyval(coefs_r, xr)

        ss_res = np.sum((yl - yhat_l) ** 2) + np.sum((yr - yhat_r) ** 2)
        r2 = 1 - ss_res / ss_tot

        if r2 > best_r2:
            best_r2 = r2
            best_split = split_idx
            best_coefs_left = coefs_l
            best_coefs_right = coefs_r

    if best_split is None or best_r2 == -np.inf:
        raise RuntimeError("piecewise search failed")

    split_x = x_sorted[best_split]
    return split_x, best_coefs_left, best_coefs_right, best_r2, x_sorted


# ---------- SHAP computation ----------
def build_shap_values_for_tag(tag: str):
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
        f"[split] {tag}: train={X_train.shape[0]}, "
        f"test={X_test.shape[0]}, features={X_train.shape[1]}"
    )

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    pipe = joblib.load(model_path)

    scaler = pipe.named_steps.get("scaler", None)
    xgb_model = pipe.named_steps["model"]

    if scaler is not None:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    X_all_scaled = np.concatenate([X_train_scaled, X_test_scaled], axis=0)
    X_all_orig = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    split_labels = np.array(["Train"] * len(y_train) + ["Test"] * len(y_test))

    # quick sanity metrics
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

    print(f"[shap] Building explainer for {tag} ...")
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values_all = explainer.shap_values(X_all_scaled)
        shap_values_all = _ensure_2d_array(shap_values_all)
        print(f"[shap] Using TreeExplainer for {tag}.")
    except Exception as exc:
        print(f"[{tag}] TreeExplainer failed, fallback to KernelExplainer:", repr(exc))
        background = X_train_scaled
        if background.shape[0] > 60:
            rng = np.random.default_rng(RANDOM_SEED)
            idx = rng.choice(background.shape[0], size=60, replace=False)
            background = background[idx]
        f = lambda data: xgb_model.predict(data)
        explainer = shap.KernelExplainer(f, background)
        shap_values_all = explainer.shap_values(X_all_scaled)
        shap_values_all = _ensure_2d_array(shap_values_all)
        print(f"[shap] Using KernelExplainer for {tag}.")

    return feature_names, X_all_orig, y_all, split_labels, shap_values_all


# ---------- CSV + plots ----------
def make_dependence_plots_and_csv(tag: str):
    print("\n" + "=" * 80)
    print(f"SHAP dependence for tag {tag}")
    print("=" * 80)

    feature_names, X_all_orig, _, split_labels, shap_values_all = \
        build_shap_values_for_tag(tag)

    n_samples, n_features = shap_values_all.shape

    rows = []
    for i in range(n_samples):
        split = split_labels[i]
        for j, fname in enumerate(feature_names):
            rows.append(
                {
                    "sample_index": i,
                    "split": split,
                    "feature": fname,
                    "feature_value": float(X_all_orig[i, j]),
                    "shap_value": float(shap_values_all[i, j]),
                }
            )

    df_long = pd.DataFrame(rows)
    out_csv = DEPENDENCE_POINTS_DIR / f"shap_points_{tag}.csv"
    df_long.to_csv(out_csv, index=False)
    print(f"[csv] Saved SHAP dependence points to: {out_csv}")

    for j, fname in enumerate(feature_names):
        x = X_all_orig[:, j]
        y = shap_values_all[:, j]
        cvals = x

        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
        sc = ax.scatter(
            x,
            y,
            c=cvals,
            cmap="coolwarm",
            s=40,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.3,
        )

        if len(x) >= 3:
            base_deg = 2
        else:
            base_deg = 1

        textstr = ""
        use_piecewise = tag in SPECIAL_PIECEWISE and fname in SPECIAL_PIECEWISE[tag]

        if use_piecewise:
            try:
                (
                    split_x,
                    coefs_l,
                    coefs_r,
                    r2_piece,
                    x_sorted,
                ) = fit_piecewise_best_split(
                    x,
                    y,
                    deg=base_deg,
                    min_segment_size=8,
                )

                xs_left = np.linspace(x_sorted.min(), split_x, 200)
                ys_left = np.polyval(coefs_l, xs_left)
                ax.plot(xs_left, ys_left, color="red", linewidth=1.5)

                xs_right = np.linspace(split_x, x_sorted.max(), 200)
                ys_right = np.polyval(coefs_r, xs_right)
                ax.plot(xs_right, ys_right, color="red", linewidth=1.5)

                formula_left = format_poly(coefs_l)
                formula_right = format_poly(coefs_r)
                textstr = (
                    f"{formula_left} (x ≤ {split_x:.2f})\n"
                    f"{formula_right} (x > {split_x:.2f})\n"
                    f"$R^2$ = {r2_piece:.3f}"
                )
            except Exception as exc:
                print(f"[warn] Piecewise fit failed for {tag}-{fname}: {exc}")
                use_piecewise = False

        if not use_piecewise:
            try:
                coefs, r2 = fit_single_poly(x, y, deg=base_deg)
                xs = np.linspace(x.min(), x.max(), 200)
                ys = np.polyval(coefs, xs)
                ax.plot(xs, ys, color="red", linewidth=1.5)

                formula_str = format_poly(coefs)
                textstr = formula_str + f"\n$R^2$ = {r2:.3f}"
            except Exception as exc:
                print(f"[warn] Polyfit failed for {tag}-{fname}: {exc}")
                textstr = r"$y = f(x)$"

        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(fname, fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(f"{tag}: {fname}", fontsize=10)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Feature value")

        plt.tight_layout()
        safe_name = safe_feat_name(fname)
        out_png = DEPENDENCE_PLOTS_DIR / f"shap_dep_{tag}_{safe_name}.png"
        fig.savefig(out_png)
        plt.close(fig)
        print(f"[plot] Saved: {out_png}")


def main():
    for tag in TAGS:
        make_dependence_plots_and_csv(tag)


if __name__ == "__main__":
    main()