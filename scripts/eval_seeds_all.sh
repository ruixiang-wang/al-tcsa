#!/usr/bin/env bash
# Evaluate seed robustness for G0/G1/G5 in AI-TCSA

set -euo pipefail

# conda env (optional)
source "/Users/wangruixiang01/miniconda3/etc/profile.d/conda.sh"
conda activate pr

PROJECT_ROOT="/Users/wangruixiang01/Desktop/AI-TCSA"
cd "$PROJECT_ROOT"

for tag in G0 G1 G5; do
  echo "==== Running seed evaluation for ${tag} ===="

  PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python modeling/eval_models_over_seeds.py \
    --data_path         "${PROJECT_ROOT}/data/raw/NO-${tag}-all.csv" \
    --model_results_csv "${PROJECT_ROOT}/artifacts/metrics/model_results_${tag}.csv" \
    --detail_csv        "${PROJECT_ROOT}/artifacts/metrics/seed_results_detail_${tag}.csv" \
    --summary_csv       "${PROJECT_ROOT}/artifacts/metrics/seed_results_summary_${tag}.csv" \
    --n_runs 200 \
    --n_jobs -1
done