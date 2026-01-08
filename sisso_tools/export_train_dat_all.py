#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export NO-G*-all.csv to SISSO train.dat:

  data/raw/NO-G0-all.csv -> data/sisso/G0/train.dat
  data/raw/NO-G1-all.csv -> data/sisso/G1/train.dat
  data/raw/NO-G5-all.csv -> data/sisso/G5/train.dat

Target:  Gn(eV)
Name:    catalyst
Features: all numeric columns except name/target.
"""

from pathlib import Path
from typing import List

import pandas as pd

from utils import to_sisso_safe_name

G_TAGS = ["G0", "G1", "G5"]
TARGET_COL = "Gn(eV)"
NAME_COL = "catalyst"

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SISSO_DATA_DIR = PROJECT_ROOT / "data" / "sisso"


def write_sisso_dat(
    df: pd.DataFrame,
    out_path: Path,
    feature_cols: List[str],
    sisso_feature_names: List[str],
) -> None:
    """Write a SISSO-style .dat file."""
    n_sample = df.shape[0]
    n_feat = len(feature_cols)
    print(f"Writing {out_path} (n_samples={n_sample}, n_features={n_feat})")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        header = "name property " + " ".join(sisso_feature_names)
        f.write(header + "\n")

        for _, row in df.iterrows():
            name = str(row[NAME_COL])
            y = float(row[TARGET_COL])
            feats = [float(row[c]) for c in feature_cols]
            line = f"{name} {y:.8f} " + " ".join(f"{v:.8f}" for v in feats)
            f.write(line + "\n")


def process_one_tag(tag: str) -> None:
    """Handle a single G-tag (G0/G1/G5)."""
    csv_path = RAW_DATA_DIR / f"NO-{tag}-all.csv"
    print(f"\n==== Processing {tag} ====")
    print(f"Reading: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"{tag}: columns = {len(df.columns)}, rows = {len(df)}")

    if NAME_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(
            f"{csv_path} must contain columns '{NAME_COL}' and '{TARGET_COL}'. "
            f"Got: {list(df.columns)}"
        )

    # Select numeric feature columns except name/target
    candidate_cols = [c for c in df.columns if c not in [NAME_COL, TARGET_COL]]
    numeric_cols = [
        c for c in candidate_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        raise ValueError(
            f"No numeric feature columns in {csv_path} "
            f"after excluding {NAME_COL}/{TARGET_COL}."
        )

    feature_cols = numeric_cols
    sisso_feature_names = [to_sisso_safe_name(c) for c in feature_cols]

    cols = [NAME_COL, TARGET_COL] + feature_cols
    df_sub = df[cols].dropna().copy()

    print(f"{tag}: samples after dropna = {df_sub.shape[0]}")
    print(f"{tag}: n_features = {len(feature_cols)}")
    print(f"{tag}: first few SISSO names = {sisso_feature_names[:5]}")

    out_dir = SISSO_DATA_DIR / tag
    out_path = out_dir / "train.dat"
    write_sisso_dat(df_sub, out_path, feature_cols, sisso_feature_names)


def main() -> None:
    for tag in G_TAGS:
        process_one_tag(tag)


if __name__ == "__main__":
    main()