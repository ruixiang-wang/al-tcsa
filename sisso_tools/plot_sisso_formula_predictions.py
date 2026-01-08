#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use SISSO analytical formulas (G0/G1/G5) to generate prediction tables
and Predicted vs Detected Gn(eV) scatter plots.

Inputs (train.dat from SISSO export):
  data/sisso/G0/train.dat
  data/sisso/G1/train.dat
  data/sisso/G5/train.dat

Outputs:
  artifacts/predictions/pred_G<G>_formula.csv
  artifacts/plots/fig_formula_<G>.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------- basic config ----------
TARGET_COL = "property"  # in SISSO train.dat
TEST_SIZE = 0.2
RANDOM_SEED = 2030
TAGS = ["G0", "G1", "G5"]

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

SISSO_DIR = PROJECT_ROOT / "data" / "sisso"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
PRED_DIR = ARTIFACTS_ROOT / "predictions"
PLOT_DIR = ARTIFACTS_ROOT / "plots"

PRED_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------
def load_data_from_dat(path: Path):
    """Load SISSO train.dat: name property f1 f2 ..."""
    df = pd.read_csv(path, delim_whitespace=True)
    if "name" not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"{path} does not look like a SISSO train.dat")

    feature_cols = [c for c in df.columns if c not in ["name", TARGET_COL]]
    X = df[feature_cols].values.astype(float)
    y = df[TARGET_COL].values.astype(float)

    print(
        f"[load_data] {path.name} -> samples={X.shape[0]}, features={X.shape[1]}"
    )
    print("Feature columns used:", feature_cols)
    return X, y, feature_cols


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
    """Scatter plot: Detected vs Predicted Gn(eV)."""
    train_mae, train_rmse, train_r2 = compute_metrics(y_train, y_train_pred)
    test_mae, test_rmse, test_r2 = compute_metrics(y_test, y_test_pred)

    all_true = np.concatenate([y_train, y_test])
    vmin = float(all_true.min())
    vmax = float(all_true.max())
    margin = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
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

    ax.plot(
        [xmin, xmax],
        [xmin, xmax],
        linestyle="-",
        linewidth=1.0,
        color="black",
    )

    ax.set_xlabel("Detected Gn (eV)", fontsize=11)
    ax.set_ylabel("Predicted Gn (eV)", fontsize=11)
    ax.set_title(f"SISSO ({tag})", fontsize=12)

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
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] Saved figure to: {out_path}")


# ---------- SISSO formulas ----------
def predict_formula(tag: str, X: np.ndarray, feature_names):
    """Vectorized SISSO prediction for a given tag (G0/G1/G5)."""
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    eps = 1e-12

    if tag == "G0":
        rd = X[:, name_to_idx["Rd_0_07"]]
        tmno = X[:, name_to_idx["TM_N_O_0_13"]]
        hf = X[:, name_to_idx["Hf_ox_eV__0_08"]]

        tmno_sq_safe = np.where(tmno == 0, eps, tmno) ** 2

        d001 = rd / tmno_sq_safe
        d002 = hf / tmno_sq_safe

        y = (
            -0.1056582704 * d001
            + 0.4945023781 * d002
            + 1.380148830
        )

    elif tag == "G1":
        MN = X[:, name_to_idx["MN"]]
        fai1 = X[:, name_to_idx["fai1"]]
        mode = X[:, name_to_idx["mode"]]
        tmno = X[:, name_to_idx["TM_N_O_0_13"]]
        I1 = X[:, name_to_idx["I1"]]
        hf = X[:, name_to_idx["Hf_ox_eV__0_08"]]
        rd = X[:, name_to_idx["Rd_0_07"]]

        fai1_safe = np.where(np.abs(fai1) < eps, eps, fai1)
        mode_safe = np.where(np.abs(mode) < eps, eps, mode)
        MN_safe = np.where(np.abs(MN) < eps, eps, MN)

        d001 = np.exp((mode - tmno) - (hf / fai1_safe))

        denom2 = (tmno / mode_safe) - mode
        denom2_safe = np.where(np.abs(denom2) < eps, eps, denom2)
        d002 = (I1 + hf) / denom2_safe

        denom3 = (I1 * tmno) + (MN - rd)
        denom3_safe = np.where(np.abs(denom3) < eps, eps, denom3)
        d003 = fai1 / denom3_safe

        denom4 = (fai1 / MN_safe) + (tmno - mode)
        denom4_safe = np.where(np.abs(denom4) < eps, eps, denom4)
        d004 = rd / denom4_safe

        y = (
            (-0.1344401342e+00) * d001
            + (0.7953966055e-01) * d002
            + (-0.2183691989e-01) * d003
            + (0.1346415102e-03) * d004
            + (0.9541261683e+00)
        )

    elif tag == "G5":
        hf = X[:, name_to_idx["Hf_ox_eV__0_08"]]
        tmno = X[:, name_to_idx["TM_N_O_0_13"]]
        fai2 = X[:, name_to_idx["fai2_0_59"]]
        rd = X[:, name_to_idx["Rd_0_07"]]
        d_band = X[:, name_to_idx["d_band_center"]]

        tmno_safe = np.where(tmno == 0, eps, tmno)
        fai2_safe = np.where(fai2 == 0, eps, fai2)

        d001 = (hf / tmno_safe) + fai2_safe
        d002 = (rd / fai2_safe) + d_band

        y = (
            -0.3614042479 * d001
            + 0.1173236154 * d002
            + 1.002857434
        )

    else:
        raise ValueError(f"Unknown tag: {tag}")

    return y


# ---------- per-tag pipeline ----------
def process_tag(tag: str):
    """Run SISSO formula pipeline for one tag."""
    print("\n" + "=" * 80)
    print(f"Processing tag {tag} (SISSO formula)")
    print("=" * 80)

    data_path = SISSO_DIR / tag / "train.dat"
    if not data_path.exists():
        raise FileNotFoundError(f"train.dat not found: {data_path}")

    X, y, feature_names = load_data_from_dat(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    print(
        f"[split] Train size={X_train.shape[0]}, "
        f"Test size={X_test.shape[0]}"
    )

    y_train_pred = predict_formula(tag, X_train, feature_names)
    y_test_pred = predict_formula(tag, X_test, feature_names)

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

    out_csv = PRED_DIR / f"pred_{tag}_formula.csv"
    df_pred.to_csv(out_csv, index=False)
    print(f"[table] Saved prediction table to: {out_csv}")

    out_png = PLOT_DIR / f"fig_formula_{tag}.png"
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