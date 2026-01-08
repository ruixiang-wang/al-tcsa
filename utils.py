#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

Shared utilities for the AI-TCSA project.

This module centralises:
  - project path helpers
  - directory creation helpers
  - global random seed control
  - basic regression metrics
  - dataset loaders (CSV and SISSO dat)
  - naming helpers for SISSO-safe feature names and filenames
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PathLike = Union[str, Path]


def get_project_root() -> Path:
    """
    Return the absolute path to the project root.

    Assumes this file lives directly under the project root, i.e.:

        AI-TCSA/
          utils.py      <-- this file
          data/
          artifacts/
          ...

    This is safer than relying on the current working directory.
    """
    return Path(__file__).resolve().parent


# Precomputed root-level directories (for convenience)
PROJECT_ROOT: Path = get_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
SISSO_DATA_DIR: Path = DATA_DIR / "sisso"

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
PREDICTIONS_DIR: Path = ARTIFACTS_DIR / "predictions"
PLOTS_DIR: Path = ARTIFACTS_DIR / "plots"

SHAP_ROOT_DIR: Path = ARTIFACTS_DIR / "shap"
SHAP_TABLES_DIR: Path = SHAP_ROOT_DIR / "tables"
SHAP_SUMMARY_PLOTS_DIR: Path = SHAP_ROOT_DIR / "summary_plots"
SHAP_DEP_POINTS_DIR: Path = SHAP_ROOT_DIR / "dependence_points"
SHAP_DEP_PLOTS_DIR: Path = SHAP_ROOT_DIR / "dependence_plots"
SHAP_FORCE_PLOTS_DIR: Path = SHAP_ROOT_DIR / "force_plots"


def ensure_dirs(*paths: PathLike) -> None:
    """
    Create all given directories if they do not exist.

    Parameters
    ----------
    *paths:
        One or more paths (str or Path) that should be created as directories.
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def ensure_default_structure() -> None:
    """
    Create the standard AI-TCSA directory structure under artifacts/ and data/.

    This is a convenience function you can call at the start of a script, e.g.:

        from utils import ensure_default_structure

        if __name__ == "__main__":
            ensure_default_structure()
            ...

    It is safe to call multiple times.
    """
    ensure_dirs(
        RAW_DATA_DIR,
        SISSO_DATA_DIR,
        MODELS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        PLOTS_DIR,
        SHAP_TABLES_DIR,
        SHAP_SUMMARY_PLOTS_DIR,
        SHAP_DEP_POINTS_DIR,
        SHAP_DEP_PLOTS_DIR,
        SHAP_FORCE_PLOTS_DIR,
    )


def raw_csv_path(tag: str) -> Path:
    """
    Return the path to the raw CSV file for a given G-tag.

    Example
    -------
    >>> raw_csv_path("G0")
    PosixPath('.../data/raw/NO-G0-all.csv')
    """
    tag = tag.upper()
    return RAW_DATA_DIR / f"NO-{tag}-all.csv"


def sisso_train_dat_path(tag: str) -> Path:
    """
    Return the path to SISSO train.dat for a given G-tag.

    This assumes a convention like:

        data/sisso/G0/train.dat
        data/sisso/G1/train.dat
        data/sisso/G5/train.dat
    """
    tag = tag.upper()
    return SISSO_DATA_DIR / tag / "train.dat"


def model_dir_for_tag(tag: str) -> Path:
    """
    Return the model directory for a given G-tag.

    Example
    -------
    >>> model_dir_for_tag("G1")
    PosixPath('.../artifacts/models/G1')
    """
    tag = tag.upper()
    return MODELS_DIR / tag


def xgb_model_path(tag: str, filename: str = "xgboost_bayes_best.joblib") -> Path:
    """
    Return the path for a trained XGBoost model for a given G-tag.

    Parameters
    ----------
    tag:
        "G0", "G1", "G5", etc.
    filename:
        Model file name inside the model directory. Default matches
        your current naming convention: "xgboost_bayes_best.joblib".
    """
    return model_dir_for_tag(tag) / filename


# ---------------------------------------------------------------------------
# Global seed control
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy and (optionally) PyTorch.

    Parameters
    ----------
    seed:
        Integer random seed for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optionally make PyTorch deterministic if it is installed.
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Note: enabling full determinism may hurt performance;
        # uncomment below only if you really need strict reproducibility.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch is not required for this project, so we quietly ignore it.
        pass


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


def compute_regression_metrics(
    y_true: Union[Sequence[float], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
) -> Tuple[float, float, float]:
    """
    Compute basic regression metrics: MAE, RMSE, R².

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Model predictions.

    Returns
    -------
    (mae, rmse, r2):
        mean absolute error, root mean squared error, R² score.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_csv_dataset(
    path: PathLike,
    target_col: str,
    drop_id_cols: Optional[Sequence[str]] = None,
    numeric_only: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Load a CSV dataset for regression.

    The typical pattern in your project is:
      - drop rows where the target is NaN
      - drop ID or non-feature columns (e.g. "catalyst")
      - treat all remaining columns as features

    Parameters
    ----------
    path:
        Path to the CSV file.
    target_col:
        Name of the target column (e.g. "Gn(eV)").
    drop_id_cols:
        Columns to drop besides the target (e.g. ["catalyst"]).
        If None, uses ["catalyst"] by default.
    numeric_only:
        If True, keep only numeric feature columns after dropping target/id cols.
    verbose:
        If True, print a short summary (samples, features, names).

    Returns
    -------
    X:
        Feature matrix of shape (n_samples, n_features).
    y:
        Target array of shape (n_samples,).
    feature_names:
        List of column names used as features (in order).
    df_clean:
        The cleaned Pandas DataFrame (after dropping NaNs on target).
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {csv_path}")

    # Drop rows with NaN in target
    df_clean = df.dropna(subset=[target_col])

    # Decide which ID columns to drop
    if drop_id_cols is None:
        drop_id_cols = ["catalyst"]

    drop_cols = [c for c in drop_id_cols if c in df_clean.columns and c != target_col]

    # y: target
    y = df_clean[target_col].to_numpy(dtype=float)

    # Feature dataframe: drop target + ID columns
    feat_df = df_clean.drop(columns=[target_col] + drop_cols)

    if numeric_only:
        feat_df = feat_df.select_dtypes(include=[np.number])

    feature_names = list(feat_df.columns)
    X = feat_df.to_numpy(dtype=float)

    if verbose:
        print(
            f"[load_csv_dataset] {csv_path.name} -> "
            f"samples={X.shape[0]}, features={X.shape[1]}"
        )
        print("[load_csv_dataset] Feature columns:", feature_names)

    return X, y, feature_names, df_clean


def load_sisso_dat(
    path: PathLike,
    target_col: str = "property",
    name_col: str = "name",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Load a SISSO-style dat file.

    Expected format (whitespace separated):

        name property f1 f2 ...

    Parameters
    ----------
    path:
        Path to the dat file.
    target_col:
        Target column name (default "property").
    name_col:
        Name column (default "name"), which will be excluded from features.
    verbose:
        If True, print a short summary.

    Returns
    -------
    X:
        Feature matrix of shape (n_samples, n_features).
    y:
        Target array of shape (n_samples,).
    feature_names:
        List of feature column names (SISSO-safe names).
    df:
        The full DataFrame as read from the dat file.
    """
    dat_path = Path(path)
    if not dat_path.exists():
        raise FileNotFoundError(f"SISSO dat file not found: {dat_path}")

    # delim_whitespace=True lets Pandas split on any whitespace
    df = pd.read_csv(dat_path, delim_whitespace=True)

    if name_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"{dat_path} does not look like a SISSO dat file; "
            f"expected columns '{name_col}' and '{target_col}', "
            f"but got: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c not in (name_col, target_col)]

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    feature_names = feature_cols

    if verbose:
        print(
            f"[load_sisso_dat] {dat_path.name} -> "
            f"samples={X.shape[0]}, features={X.shape[1]}"
        )
        print("[load_sisso_dat] Feature columns:", feature_names)

    return X, y, feature_names, df


# ---------------------------------------------------------------------------
# Naming helpers (SISSO-safe names, file-safe names)
# ---------------------------------------------------------------------------


def to_sisso_safe_name(col_name: str) -> str:
    """
    Convert an arbitrary column name into a SISSO-safe feature name.

    Rules
    -----
    - Replace any non-alphanumeric / underscore character with '_'
    - If the name starts with a digit, prepend 'f_'
    - If the result is empty, fall back to 'feat'
    """
    safe = re.sub(r"[^0-9A-Za-z_]", "_", col_name)
    if re.match(r"^[0-9]", safe):
        safe = "f_" + safe
    if safe == "":
        safe = "feat"
    return safe


def safe_filename(name: str) -> str:
    """
    Convert an arbitrary string into a safe filename component.

    This is mainly used to build plot filenames from feature names or
    catalyst IDs, so we remove or replace risky characters.

    Rules
    -----
    - Replace any sequence of non-alphanumeric characters with '_'
    - Strip leading/trailing '_' characters
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


# ---------------------------------------------------------------------------
# Small convenience helpers
# ---------------------------------------------------------------------------


def tags_default() -> List[str]:
    """
    Return the default list of G-tags used in this project.

    Currently: ["G0", "G1", "G5"].
    """
    return ["G0", "G1", "G5"]