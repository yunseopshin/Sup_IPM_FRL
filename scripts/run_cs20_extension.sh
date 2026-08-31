#!/bin/bash
# cs20 extension: (A) weak-lambda fill for critic20, (B) critic_step ladder at lr 0.01,
# (C) tail contest lambda in (2,3). Sequential stages, MAX_PAR=2 throughout.
# Skip logic (results.json present) makes reruns safe.
set -e
cd "$(dirname "$0")/.."
SEEDS_ALL="2023 2024 2025 2026 2027"

echo "[ext] === Stage A: cs20 weak-lambda fill $(date) ==="
DATASETS=Adult ALGS=supipm LMDAS="0.1 0.2 0.01" SEEDS="$SEEDS_ALL" MAX_PAR=2 \
  EXTRA_ARGS="--critic_step 20 --critic_lr 0.01 --out_root results/critic20" \
  LOG_TAG="-cs20w" scripts/run_sweep.sh

echo "[ext] === Stage B: critic_step ladder (lr 0.01 fixed) $(date) ==="
for cs in 2 5 10; do
  DATASETS=Adult ALGS=supipm LMDAS="1.0 2.0" SEEDS="$SEEDS_ALL" MAX_PAR=2 \
    EXTRA_ARGS="--critic_step $cs --critic_lr 0.01 --out_root results/critic_ladder/cs$cs" \
    LOG_TAG="-ladder-cs$cs" scripts/run_sweep.sh
done

echo "[ext] === Stage C: cs20 tail contest $(date) ==="
DATASETS=Adult ALGS=supipm LMDAS="2.25 2.5" SEEDS="$SEEDS_ALL" MAX_PAR=2 \
  EXTRA_ARGS="--critic_step 20 --critic_lr 0.01 --out_root results/critic20" \
  LOG_TAG="-cs20t" scripts/run_sweep.sh

echo "[ext] === all stages done $(date) ==="
echo "[ext] critic20: $(find results/critic20 -name results.json | wc -l) | ladder: $(find results/critic_ladder -name results.json 2>/dev/null | wc -l)"
