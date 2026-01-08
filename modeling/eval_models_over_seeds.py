#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate fixed-parameter models over many random seeds.

- Read best_params from artifacts/metrics/model_results_<TAG>.csv
- For each model, repeat train/test split with different seeds
- Save per-seed metrics + summary (mean/std/95% CI) into artifacts/metrics
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

from utils import (
    set_global_seed,
    load_csv_dataset,
    compute_regression_metrics,
    ensure_default_structure,
    raw_csv_path,
    METRICS_DIR,
)

DEFAULT_TAG = "G1"
DEFAULT_TARGET_COL = "Gn(eV)"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_RUNS = 200
DEFAULT_SEED_FOR_SEEDS = 12345
DEFAULT_N_JOBS = -1


def summarize_for_model(df_model: pd.DataFrame, model_name: str) -> dict:
    """Compute mean/std/95% CI over runs for one model."""
    summary = {
        "model_name": model_name,
        "n_runs": int(len(df_model)),
    }
    for split in ["train", "test"]:
        for metric in ["MAE", "RMSE", "R2"]:
            col = f"{split}_{metric}"
            values = df_model[col].values
            n = len(values)
            mean_val = float(values.mean())
            if n > 1:
                std_val = float(values.std(ddof=1))
                ci_half = 1.96 * std_val / np.sqrt(n)
            else:
                std_val = 0.0
                ci_half = 0.0

            summary[f"{split}_{metric}_mean"] = mean_val
            summary[f"{split}_{metric}_std"] = std_val
            summary[f"{split}_{metric}_ci95_lower"] = mean_val - ci_half
            summary[f"{split}_{metric}_ci95_upper"] = mean_val + ci_half
    return summary


def infer_tag_from_data_path(data_path: str) -> str | None:
    """Infer G0/G1/G5 from filename if possible."""
    base = os.path.basename(data_path)
    stem, _ = os.path.splitext(base)
    for t in ["G0", "G1", "G5"]:
        if t in stem:
            return t
    return None


def _fix_param_types(params: dict) -> dict:
    """Cast integer-valued floats back to int."""
    fixed = {}
    for k, v in params.items():
        if isinstance(v, float) and abs(v - round(v)) < 1e-8:
            v = int(round(v))
        fixed[k] = v
    return fixed


def load_best_params(model_results_csv: str) -> dict:
    """
    Read best_params JSON from model_results_*.csv and add 'model__' prefix
    so they can be applied to a Pipeline.
    """
    df = pd.read_csv(model_results_csv)
    if "model_name" not in df.columns or "best_params" not in df.columns:
        raise ValueError(
            f"{model_results_csv} must contain 'model_name' and 'best_params'"
        )

    best_params_dict = {}
    for _, row in df.iterrows():
        name = row["model_name"]
        bp_json = row["best_params"]
        try:
            params_clean = json.loads(bp_json)
        except Exception as e:
            raise ValueError(f"Failed to parse best_params for model {name}: {e}")
        params_clean = _fix_param_types(params_clean)
        params_prefixed = {f"model__{k}": v for k, v in params_clean.items()}
        best_params_dict[name] = params_prefixed
    return best_params_dict


def build_base_pipeline(model_name: str, n_jobs: int) -> Pipeline:
    """Base pipeline: StandardScaler + model (before applying best_params)."""
    if model_name == "extra_trees":
        model = ExtraTreesRegressor(random_state=2025, n_jobs=n_jobs)
    elif model_name == "knn":
        model = KNeighborsRegressor(n_jobs=None)
    elif model_name == "lasso":
        model = Lasso(fit_intercept=True, positive=False)
    elif model_name == "linear_regression":
        model = LinearRegression(n_jobs=n_jobs)
    elif model_name == "random_forest":
        model = RandomForestRegressor(random_state=2025, n_jobs=n_jobs)
    elif model_name == "ridge":
        model = Ridge(fit_intercept=True, positive=False)
    elif model_name == "svr":
        model = SVR()
    elif model_name == "xgboost":
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=2025,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def run_model_over_seeds(
    model_name: str,
    best_params: dict,
    X,
    y,
    seeds,
    test_size: float,
    n_jobs: int,
):
    """Train + evaluate one model over many seeds."""
    records = []
    for idx, seed in enumerate(seeds, start=1):
        print(f"\n==== [{model_name}] run {idx}/{len(seeds)} with seed = {seed} ====")
        set_global_seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

        pipe = build_base_pipeline(model_name, n_jobs)
        if best_params:
            pipe.set_params(**best_params)

        pipe.fit(X_train, y_train)

        y_train_pred = pipe.predict(X_train)
        train_mae, train_rmse, train_r2 = compute_regression_metrics(
            y_train, y_train_pred
        )

        y_test_pred = pipe.predict(X_test)
        test_mae, test_rmse, test_r2 = compute_regression_metrics(y_test, y_test_pred)

        print(
            f"[{model_name}] Train: MAE={train_mae:.6f}, RMSE={train_rmse:.6f}, R2={train_r2:.6f}"
        )
        print(
            f"[{model_name}] Test : MAE={test_mae:.6f}, RMSE={test_rmse:.6f}, R2={test_r2:.6f}"
        )

        records.append(
            {
                "model_name": model_name,
                "seed": int(seed),
                "train_MAE": train_mae,
                "train_RMSE": train_rmse,
                "train_R2": train_r2,
                "test_MAE": test_mae,
                "test_RMSE": test_rmse,
                "test_R2": test_r2,
            }
        )

    df_model = pd.DataFrame(records)
    summary = summarize_for_model(df_model, model_name)
    return df_model, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple fixed-parameter models over many random seeds "
            "(using best_params from model_results_*.csv)."
        )
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Dataset tag, e.g. G0/G1/G5. If not set, will try to infer from data_path.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to input CSV. If not set, use data/raw/NO-<tag>-all.csv.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default=DEFAULT_TARGET_COL,
        help="Target column name.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Test set ratio.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help="Number of random seeds / runs per model.",
    )
    parser.add_argument(
        "--seed_for_seeds",
        type=int,
        default=DEFAULT_SEED_FOR_SEEDS,
        help="Seed used to generate the list of seeds.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="n_jobs for models that support it.",
    )
    parser.add_argument(
        "--model_results_csv",
        type=str,
        default=None,
        help=(
            "CSV produced by train_models_bayes.py (contains best_params). "
            "Default: artifacts/metrics/model_results_<tag>.csv"
        ),
    )
    parser.add_argument(
        "--detail_csv",
        type=str,
        default=None,
        help=(
            "CSV for per-seed metrics. "
            "Default: artifacts/metrics/seed_results_detail_<tag>.csv"
        ),
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help=(
            "CSV for per-model mean/std/CI metrics. "
            "Default: artifacts/metrics/seed_results_summary_<tag>.csv"
        ),
    )
    return parser.parse_args()


def resolve_paths(args):
    """Fill in tag, data_path, and output paths."""
    # tag
    if args.tag is None:
        if args.data_path is not None:
            tag = infer_tag_from_data_path(args.data_path) or DEFAULT_TAG
        else:
            tag = DEFAULT_TAG
        args.tag = tag

    # data_path
    if args.data_path is None:
        args.data_path = str(raw_csv_path(args.tag))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # model_results_csv
    if args.model_results_csv is None:
        args.model_results_csv = str(METRICS_DIR / f"model_results_{args.tag}.csv")

    # detail / summary CSV
    if args.detail_csv is None:
        args.detail_csv = str(METRICS_DIR / f"seed_results_detail_{args.tag}.csv")
    else:
        Path(args.detail_csv).parent.mkdir(parents=True, exist_ok=True)

    if args.summary_csv is None:
        args.summary_csv = str(METRICS_DIR / f"seed_results_summary_{args.tag}.csv")
    else:
        Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    ensure_default_structure()
    resolve_paths(args)

    print(f"Tag                    : {args.tag}")
    print(f"Using data_path        : {args.data_path}")
    print(f"Using model_results_csv: {args.model_results_csv}")
    print(f"Detail CSV will be     : {args.detail_csv}")
    print(f"Summary CSV will be    : {args.summary_csv}")

    # seeds for train/test splits
    rng = np.random.default_rng(args.seed_for_seeds)
    seeds = rng.integers(low=1, high=10_000_000, size=args.n_runs).tolist()
    print(f"Total seeds: {len(seeds)}")

    # data
    set_global_seed(2025)
    print("==== Load data ====")
    X, y, _, _ = load_csv_dataset(
        path=args.data_path,
        target_col=args.target_col,
        drop_id_cols=["catalyst"],
        numeric_only=False,
        verbose=True,
    )
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # best params
    print("==== Load best params from model_results CSV ====")
    best_params_dict = load_best_params(args.model_results_csv)
    print("Models found:", list(best_params_dict.keys()))

    model_names = [
        "extra_trees",
        "knn",
        "lasso",
        "linear_regression",
        "random_forest",
        "ridge",
        "svr",
        "xgboost",
    ]

    all_detail_dfs = []
    summary_rows = []

    for name in model_names:
        if name not in best_params_dict:
            print(f"Warning: model {name} not found in {args.model_results_csv}, skip.")
            continue

        df_model, summary = run_model_over_seeds(
            model_name=name,
            best_params=best_params_dict[name],
            X=X,
            y=y,
            seeds=seeds,
            test_size=args.test_size,
            n_jobs=args.n_jobs,
        )
        df_model["tag"] = args.tag
        all_detail_dfs.append(df_model)

        summary["tag"] = args.tag
        summary_rows.append(summary)

    # save detail
    if all_detail_dfs:
        detail_df = pd.concat(all_detail_dfs, ignore_index=True)
        detail_df.to_csv(args.detail_csv, index=False)
        print(f"\nPer-seed metrics saved to: {args.detail_csv}")
    else:
        print("No detail metrics to save (no models were run).")

    # save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(args.summary_csv, index=False)
        print(f"Summary (mean/std/ci95) saved to: {args.summary_csv}")
        print("\nSummary preview:")
        print(summary_df)
    else:
        print("No summary metrics to save (no models were run).")


if __name__ == "__main__":
    main()