#!/bin/bash
# Adult: percritic at critic_step=2 (critic_lr 0.01), to sit beside the EXISTING
# shared critic_step ladder (results/critic_ladder/cs{2,5,10}) and cs20
# (results/critic20) — same lambdas, same seeds, same critic_lr, so the cells are
# directly comparable. Report with scripts/aggregate_smode_adult.py.
#
# One srun job (gpu:1) per run, at most MAX_PAR at once (default 2 - do not
# occupy all 4 GPUs; the count includes any other job of yours already queued).
# Finished runs are skipped, so it is restartable.
#
# Usage: [LMDAS="1.0 2.0"] [SEEDS="2023 2024"] [MAX_PAR=2] scripts/run_smode_adult.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p results/logs

LMDAS=(${LMDAS:-1.0 2.0})
SEEDS=(${SEEDS:-2023 2024 2025 2026 2027})
CS=${CS:-2}
CRITIC_LR=${CRITIC_LR:-0.01}
MAX_PAR=${MAX_PAR:-2}
OUT=results/critic_ladder/cs$CS
PY=/usr/local/miniconda3/envs/nine/bin/python

echo "== GPU usage (check for non-Slurm occupants before launching) =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

for lmda in "${LMDAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    rd="$OUT/Adult-age/supipm-percritic/lmda_f-${lmda}/seed-${seed}"
    if [ -f "$rd/results.json" ]; then echo "[skip] lmda=$lmda seed=$seed"; continue; fi
    # wait on the whole user queue, not just this script's jobs, to honour the cap
    while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$MAX_PAR" ]; do sleep 30; done
    echo "[smode-adult] lmda=$lmda seed=$seed cs=$CS $(date +%H:%M)"
    srun --gres=gpu:1 --cpus-per-task=10 --partition=idea --time=12:00:00 --mem=25G \
      --job-name="adult-pc-cs$CS-$lmda-$seed" \
      $PY src/train.py --config configs/default.yaml --dataset Adult --lmda_f "$lmda" \
      --seed "$seed" --critic_step "$CS" --critic_lr "$CRITIC_LR" --s_mode percritic \
      --out_root "$OUT" > "results/logs/smode-adult-pc-cs$CS-$lmda-$seed.log" 2>&1 &
    sleep 5
  done
done

wait
echo "[smode-adult] all jobs finished $(date)"
