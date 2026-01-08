#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train multiple regression models with Bayesian hyperparameter search (BayesSearchCV)
on one G-tag dataset (G0 / G1 / G5) and save:

- best model for each algorithm under artifacts/models/<TAG>/
- a summary CSV with metrics under artifacts/metrics/model_results_<TAG>.csv

This is the refactored version for AI-TCSA, using utils.py for:
  - path management
  - global seed
  - dataset loading
  - regression metrics
  - directory structure
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedKFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR

from xgboost import XGBRegressor

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

import joblib

from utils import (
    set_global_seed,
    load_csv_dataset,
    compute_regression_metrics,
    ensure_default_structure,
    raw_csv_path,
    model_dir_for_tag,
    METRICS_DIR,
)

# ----------------------------------------------------------------------
# Global config (can be overridden by CLI)
# ----------------------------------------------------------------------

DEFAULT_TAG = "G1"
DEFAULT_TARGET_COL = "Gn(eV)"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_SEED = 2025
DEFAULT_N_JOBS = -1

USE_LEAVE_ONE_OUT = False  # if True -> LeaveOneOut, else RepeatedKFold
N_SPLITS = 5
N_REPEATS = 3


# ----------------------------------------------------------------------
# CV builder
# ----------------------------------------------------------------------
def build_cv(random_seed: int):
    """
    Build cross-validation splitter.

    If USE_LEAVE_ONE_OUT is True: use LeaveOneOut.
    Otherwise: use RepeatedKFold with N_SPLITS * N_REPEATS folds.
    """
    if USE_LEAVE_ONE_OUT:
        return LeaveOneOut()
    return RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=random_seed,
    )


# ----------------------------------------------------------------------
# Feature importance + best params post-processing
# ----------------------------------------------------------------------
def clean_param_keys(param_dict):
    """
    BayesSearchCV returns parameter names like 'model__max_depth'.

    This helper strips the 'model__' prefix so we can store them
    more cleanly in the CSV.
    """
    return {
        (k.split("__", 1)[1] if "__" in k else k): v
        for k, v in param_dict.items()
    }


def compute_feature_importances(
    best_model,
    feature_names,
    X_train,
    y_train,
    random_seed: int,
    n_jobs: int,
):
    """
    Compute feature importance for the fitted model.

    Strategy:
      - Tree-based models (ExtraTrees / RandomForest / XGBoost):
          use .feature_importances_
      - Linear models (Lasso / LinearRegression / Ridge):
          use abs(coef_)
      - Others (KNN / SVR / etc.):
          use permutation_importance on the whole pipeline

    Returns
    -------
    json_str or None
        JSON string for top-10 features, or None if something goes wrong
        (e.g. shape mismatch).
    """
    # Try to get the 'model' step from a Pipeline, otherwise use estimator itself
    try:
        est = best_model.named_steps.get("model", best_model)
    except AttributeError:
        est = best_model

    importances = None

    # 1) Tree models
    if hasattr(est, "feature_importances_"):
        importances = np.asarray(est.feature_importances_)

    # 2) Linear models
    elif isinstance(est, (Lasso, LinearRegression, Ridge)):
        coef = np.asarray(est.coef_)
        if coef.ndim > 1:
            # multi-output: average over outputs
            coef = coef.mean(axis=0)
        importances = np.abs(coef)

    # 3) Fallback: permutation importance on the full pipeline
    else:
        result = permutation_importance(
            best_model,
            X_train,
            y_train,
            n_repeats=5,
            random_state=random_seed,
            n_jobs=n_jobs,
            scoring="neg_root_mean_squared_error",
        )
        importances = np.maximum(result.importances_mean, 0.0)

    # Defensive check
    if importances is None or len(importances) != len(feature_names):
        return None

    topk = min(10, len(feature_names))
    indices = np.argsort(importances)[::-1][:topk]

    fi_list = [
        {"feature": feature_names[i], "importance": float(importances[i])}
        for i in indices
    ]
    return json.dumps(fi_list, ensure_ascii=False)


# ----------------------------------------------------------------------
# Model builders + search spaces
# ----------------------------------------------------------------------
def build_pipeline_extratrees(seed: int, n_jobs: int):
    """
    ExtraTrees: deliberately limited capacity (shallow trees, larger leaves).
    """
    model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features=0.7,
        random_state=seed,
        n_jobs=n_jobs,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def search_space_extratrees():
    return {
        "model__n_estimators": Integer(100, 300),
        "model__max_depth": Integer(3, 8),
        "model__min_samples_leaf": Integer(5, 20),
        "model__min_samples_split": Integer(10, 40),
        "model__max_features": Real(0.5, 1.0),
    }


def build_pipeline_knn(seed: int, n_jobs: int):
    """
    KNN: use relatively large k and uniform weights for smoother behavior.
    """
    model = KNeighborsRegressor(
        n_neighbors=20,
        weights="uniform",
        p=2,
        n_jobs=None,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def search_space_knn():
    """
    KNN search space:
      - large k only
      - fixed uniform weights
    """
    return {
        "model__n_neighbors": Integer(20, 40),
        "model__weights": Categorical(["uniform"]),
        "model__leaf_size": Integer(30, 80),
    }


def build_pipeline_lasso(seed: int, n_jobs: int):
    model = Lasso(
        alpha=0.1,
        max_iter=2000,
        random_state=seed,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def search_space_lasso():
    return {
        "model__alpha": Real(1e-4, 1e1, prior="log-uniform"),
        "model__max_iter": Integer(500, 5000),
    }


def build_pipeline_linreg(seed: int, n_jobs: int):
    model = LinearRegression(n_jobs=n_jobs)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def search_space_linreg():
    return {
        "model__fit_intercept": Categorical([True, False]),
        "model__positive": Categorical([False, True]),
    }


def build_pipeline_rf(seed: int, n_jobs: int):
    """
    RandomForest: also constrained to be relatively simple.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features=0.7,
        random_state=seed,
        n_jobs=n_jobs,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def search_space_rf():
    return {
        "model__n_estimators": Integer(100, 300),
        "model__max_depth": Integer(3, 8),
        "model__min_samples_leaf": Integer(5, 20),
        "model__min_samples_split": Integer(10, 40),
        "model__max_features": Real(0.5, 0.9),
    }


def build_pipeline_ridge(seed: int, n_jobs: int):
    model = Ridge(
        alpha=1.0,
        fit_intercept=True,
        positive=False,
        solver="auto",
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def search_space_ridge():
    return {
        "model__alpha": Real(1e-4, 1e3, prior="log-uniform"),
        "model__fit_intercept": Categorical([True, False]),
        "model__positive": Categorical([False, True]),
    }


def build_pipeline_svr(seed: int, n_jobs: int):
    """
    SVR: keep C / gamma relatively small; aim for smooth functions.
    """
    model = SVR(
        kernel="rbf",
        C=0.5,
        epsilon=0.1,
        gamma="scale",
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def search_space_svr():
    return {
        "model__C": Real(1e-2, 1.0, prior="log-uniform"),
        "model__epsilon": Real(0.05, 0.5, prior="log-uniform"),
        "model__gamma": Real(1e-4, 0.2, prior="log-uniform"),
    }


def build_pipeline_xgb(seed: int, n_jobs: int):
    """
    XGBoost: strongly regularised, shallow trees and large min_child_weight
    to keep train R² away from 1.0 while preserving test performance.
    """
    model = XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=20,
        gamma=0.5,
        reg_alpha=2.0,
        reg_lambda=20.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=seed,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def search_space_xgb():
    """
    XGBoost search space (still strongly regularised, slightly wider range):
      - shallow to medium trees (max_depth 2–6)
      - 150–400 trees
      - min_child_weight 1–40
      - strong but flexible L1/L2 regularisation
    """
    return {
        "model__n_estimators": Integer(150, 400),
        "model__max_depth": Integer(2, 6),
        "model__learning_rate": Real(0.02, 0.18, prior="log-uniform"),
        "model__subsample": Real(0.6, 1.0),
        "model__colsample_bytree": Real(0.6, 1.0),
        "model__min_child_weight": Integer(1, 40),
        "model__gamma": Real(0.0, 3.0),
        "model__reg_alpha": Real(1e-2, 10.0, prior="log-uniform"),
        "model__reg_lambda": Real(0.2, 50.0, prior="log-uniform"),
    }


# ----------------------------------------------------------------------
# Unified runner for one model
# ----------------------------------------------------------------------
def run_single_model(
    model_name: str,
    build_pipeline_fn,
    search_space_fn,
    n_iter: int,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    cv,
    random_seed: int,
    n_jobs: int,
    model_output_dir: str,
):
    print("\n" + "=" * 80)
    print(f"Start model: {model_name}")
    print("=" * 80)

    pipe = build_pipeline_fn(random_seed, n_jobs)
    search_space = search_space_fn()

    opt = BayesSearchCV(
        estimator=pipe,
        search_spaces=search_space,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
        random_state=random_seed,
        verbose=1,
    )

    print(f"Running BayesSearchCV for {model_name} ...")
    opt.fit(X_train, y_train)

    print(f"Best params ({model_name}):", opt.best_params_)
    cv_best_rmse = -opt.best_score_
    print(f"CV best RMSE ({model_name}): {cv_best_rmse:.6f}")

    best_model = opt.best_estimator_

    # Train metrics
    y_train_pred = best_model.predict(X_train)
    train_mae, train_rmse, train_r2 = compute_regression_metrics(y_train, y_train_pred)

    # Test metrics
    y_test_pred = best_model.predict(X_test)
    test_mae, test_rmse, test_r2 = compute_regression_metrics(y_test, y_test_pred)

    print(f"[{model_name}] Train MAE : {train_mae:.6f}")
    print(f"[{model_name}] Train RMSE: {train_rmse:.6f}")
    print(f"[{model_name}] Train R2  : {train_r2:.6f}")
    print(f"[{model_name}] Test  MAE : {test_mae:.6f}")
    print(f"[{model_name}] Test  RMSE: {test_rmse:.6f}")
    print(f"[{model_name}] Test  R2  : {test_r2:.6f}")

    # Feature importances (top-10)
    feature_importances_json = compute_feature_importances(
        best_model=best_model,
        feature_names=feature_names,
        X_train=X_train,
        y_train=y_train,
        random_seed=random_seed,
        n_jobs=n_jobs,
    )

    # Save model
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, f"{model_name}_bayes_best.joblib")
    joblib.dump(best_model, model_path)
    print(f"[{model_name}] Saved best model to: {model_path}")

    # Clean up param keys for CSV
    clean_params = clean_param_keys(opt.best_params_)

    result = {
        "model_name": model_name,
        "cv_best_rmse": cv_best_rmse,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "best_params": json.dumps(clean_params, ensure_ascii=False),
        "n_iter": n_iter,
        "feature_importances": feature_importances_json,
    }
    return result


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train multiple models with BayesSearchCV on one G-tag dataset "
            "and save metrics + best models."
        )
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=DEFAULT_TAG,
        help="Tag for dataset (e.g. G0, G1, G5). Used to find default CSV path.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to input CSV data. If not set, use data/raw/NO-<tag>-all.csv.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default=DEFAULT_TARGET_COL,
        help="Name of target column.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Test set ratio.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="Number of parallel jobs for BayesSearchCV.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help=(
            "Path to output CSV storing metrics. "
            "Default: artifacts/metrics/model_results_<tag>.csv"
        ),
    )
    parser.add_argument(
        "--model_output_dir",
        type=str,
        default=None,
        help=(
            "Directory to save best models. "
            "Default: artifacts/models/<tag>/"
        ),
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # Make sure basic directory structure exists
    ensure_default_structure()

    # Resolve data path from tag if not provided
    if args.data_path is None:
        args.data_path = str(raw_csv_path(args.tag))

    # Resolve model output directory from tag if not provided
    if args.model_output_dir is None:
        args.model_output_dir = str(model_dir_for_tag(args.tag))
    Path(args.model_output_dir).mkdir(parents=True, exist_ok=True)

    # Resolve output CSV path if not provided
    if args.output_csv is None:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        args.output_csv = str(METRICS_DIR / f"model_results_{args.tag}.csv")
    else:
        out_parent = Path(args.output_csv).parent
        out_parent.mkdir(parents=True, exist_ok=True)

    print(f"Tag: {args.tag}")
    print(f"Data path: {args.data_path}")
    print(f"Model output dir: {args.model_output_dir}")
    print(f"Metrics CSV will be saved to: {args.output_csv}")

    # Global seed
    set_global_seed(args.random_seed)

    # Load data
    print("==== Load data ====")
    X, y, feature_names, _ = load_csv_dataset(
        path=args.data_path,
        target_col=args.target_col,
        drop_id_cols=["catalyst"],
        numeric_only=False,
        verbose=True,
    )
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_seed,
        shuffle=True,
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # CV splitter
    cv = build_cv(args.random_seed)

    # Unified model config list
    model_specs = [
        {
            "name": "extra_trees",
            "builder": build_pipeline_extratrees,
            "search_space": search_space_extratrees,
            "n_iter": 50,
        },
        {
            "name": "knn",
            "builder": build_pipeline_knn,
            "search_space": search_space_knn,
            "n_iter": 30,
        },
        {
            "name": "lasso",
            "builder": build_pipeline_lasso,
            "search_space": search_space_lasso,
            "n_iter": 50,
        },
        {
            "name": "linear_regression",
            "builder": build_pipeline_linreg,
            "search_space": search_space_linreg,
            "n_iter": 20,
        },
        {
            "name": "random_forest",
            "builder": build_pipeline_rf,
            "search_space": search_space_rf,
            "n_iter": 50,
        },
        {
            "name": "ridge",
            "builder": build_pipeline_ridge,
            "search_space": search_space_ridge,
            "n_iter": 30,
        },
        {
            "name": "svr",
            "builder": build_pipeline_svr,
            "search_space": search_space_svr,
            "n_iter": 30,
        },
        {
            "name": "xgboost",
            "builder": build_pipeline_xgb,
            "search_space": search_space_xgb,
            "n_iter": 60,  # more iterations for XGB
        },
    ]

    # Run all models
    results = []
    for spec in model_specs:
        res = run_single_model(
            model_name=spec["name"],
            build_pipeline_fn=spec["builder"],
            search_space_fn=spec["search_space"],
            n_iter=spec["n_iter"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            cv=cv,
            random_seed=args.random_seed,
            n_jobs=args.n_jobs,
            model_output_dir=args.model_output_dir,
        )
        res["tag"] = args.tag
        res["data_path"] = args.data_path
        results.append(res)

    # Save metrics
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print("\nAll models finished.")
    print(f"Metrics CSV saved to: {args.output_csv}")
    print(df)


if __name__ == "__main__":
    main()
