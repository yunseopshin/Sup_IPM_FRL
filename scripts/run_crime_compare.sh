#!/bin/bash
# Crime campaign mirroring the Adult comparison: (A) complete the 5-seed baseline
# grids (supipm cs2 + frem gamma_s in {0.03,0.05,0.07}), (B) cs20 honest-critic runs
# (prediction from the Adult diagnosis: on Crime the pool is already honest, so
# cs20 ~ cs2 — a control for the budget-sufficiency story).
# Existing runs (results.json) are skipped automatically.
set -e
cd "$(dirname "$0")/.."
SEEDS_ALL="2023 2024 2025 2026 2027"

echo "[crime] === Stage A1: supipm baseline fill $(date) ==="
DATASETS=Crime ALGS=supipm LMDAS="0.1 0.3 1.0 0.003 0.03 0.01 3.0 0.0" \
  SEEDS="$SEEDS_ALL" MAX_PAR=2 scripts/run_sweep.sh

echo "[crime] === Stage A2: frem gs0.05 fill $(date) ==="
DATASETS=Crime ALGS=frem LMDAS="0.1 0.3 1.0 0.01 0.5 0.03 0.0" \
  SEEDS="$SEEDS_ALL" MAX_PAR=2 scripts/run_sweep.sh

echo "[crime] === Stage A3: frem gamma_s robustness $(date) ==="
for gs in 0.03 0.07; do
  DATASETS=Crime ALGS=frem LMDAS="0.1 0.3 1.0 0.01 0.5 0.03" \
    SEEDS="$SEEDS_ALL" MAX_PAR=2 \
    EXTRA_ARGS="--gamma_s $gs" LOG_TAG="-gs$gs" scripts/run_sweep.sh
done

echo "[crime] === Stage B: supipm critic 20-step $(date) ==="
DATASETS=Crime ALGS=supipm LMDAS="0.1 0.01 1.0 0.03 0.3 0.003" \
  SEEDS="$SEEDS_ALL" MAX_PAR=2 \
  EXTRA_ARGS="--critic_step 20 --critic_lr 0.01 --out_root results/critic20" \
  LOG_TAG="-cs20" scripts/run_sweep.sh

echo "[crime] === all stages done $(date) ==="
echo "[crime] baseline tree: $(find results/Crime-racepctblack -name results.json | wc -l) | cs20: $(find results/critic20/Crime-racepctblack -name results.json 2>/dev/null | wc -l)"
