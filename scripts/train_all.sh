#!/usr/bin/env bash
set -e

PROJECT_ROOT="/Users/wangruixiang01/Desktop/AI-TCSA"
cd "$PROJECT_ROOT"

for tag in G0 G1 G5; do
  echo "==== Training models for ${tag} ===="
  PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python modeling/train_models_bayes.py \
    --tag "${tag}" \
    --n_jobs -1
done