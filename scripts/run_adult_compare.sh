#!/bin/bash
# Adult supIPM-vs-FREM comparison (same environment: unified representer, FREM protocol).
# Stages run sequentially; each stage = run_sweep.sh with MAX_PAR=2 (occupy <= 2 GPUs).
# Coverage-first lambda order so partial results span the range.
# Already-finished runs (results.json present) are skipped by train.py.
set -e
cd "$(dirname "$0")/.."
SEEDS_ALL="2023 2024 2025 2026 2027"

echo "[compare] === Stage A1: frem gamma_s=0.05 (core FREM curve) $(date) ==="
DATASETS=Adult ALGS=frem \
  LMDAS="0.1 0.0 1.0 0.01 0.5 0.05 3.0 0.2 2.0" \
  SEEDS="$SEEDS_ALL" MAX_PAR=2 scripts/run_sweep.sh

echo "[compare] === Stage A2: supipm (ours) $(date) ==="
DATASETS=Adult ALGS=supipm \
  LMDAS="0.1 0.0 1.0 0.01 0.3 0.03 3.0" \
  SEEDS="$SEEDS_ALL" MAX_PAR=2 scripts/run_sweep.sh

echo "[compare] === Stage B: frem gamma_s robustness (paper: gamma_s selected on val) $(date) ==="
for gs in 0.03 0.07; do
  DATASETS=Adult ALGS=frem \
    LMDAS="0.1 1.0 0.01 0.5 0.05 3.0 0.2 2.0" \
    SEEDS="$SEEDS_ALL" MAX_PAR=2 \
    EXTRA_ARGS="--gamma_s $gs" LOG_TAG="-gs$gs" scripts/run_sweep.sh
done

echo "[compare] === all stages done $(date) ==="
n_done=$(find results/Adult-age -name results.json 2>/dev/null | wc -l)
echo "[compare] finished runs with results.json: $n_done"
