#!/bin/bash
# Array task for the SynthB S~Uniform experiment: one (lambda, seed, trim-mode) per
# task. Submitted with `--array=0-17%2` so at most two GPUs are held at a time.
#
# train.py skips any run whose results.json already exists, so the full 18-task grid
# is safe to (re)submit — finished runs exit in a second and only the missing ones
# actually train.
#
# Submitted as:
#   sbatch --array=0-17%2 --begin=<time> --gres=gpu:1 --cpus-per-task=10 --mem=25G \
#     --chdir="<repo>" --output="results/logs/uni-resume-%A_%a.log" \
#     scripts/run_uniform_resume.sh
set -u
PY=/usr/local/miniconda3/envs/nine/bin/python

TASKS=()
for lm in 0.0 0.1 0.3 1.0; do                 # trimmed (alpha = 0.05)
  for sd in 2023 2024 2025; do TASKS+=("$lm $sd trim"); done
done
for lm in 0.0 0.3; do                         # no-trim counterpart (alpha = 0)
  for sd in 2023 2024 2025; do TASKS+=("$lm $sd notrim"); done
done

read -r lm sd mode <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
if [ "$mode" = "trim" ]; then
  EXTRA=(--out_root results/synthB-uniform)
else
  EXTRA=(--alpha 0.0 --out_root results/synthB-uniform-notrim)
fi

echo "[uni-resume] task=$SLURM_ARRAY_TASK_ID lmda=$lm seed=$sd mode=$mode" \
     "gpu=${CUDA_VISIBLE_DEVICES:-?} $(date '+%F %T')"
exec "$PY" src/train.py --config configs/default.yaml --dataset SynthB --alg supipm \
  --lmda_f "$lm" --seed "$sd" --weight_decay 0.0 --synthb_s_dist uniform "${EXTRA[@]}"
