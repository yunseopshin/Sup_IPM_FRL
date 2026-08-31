#!/bin/bash
# Sweep over datasets / lambdas / seeds: one srun job (gpu:1) per run, in background.
# At most MAX_PAR jobs run at once (default 2 - do not occupy all 4 GPUs).
# Edit the grids below. Usage: [MAX_PAR=2] scripts/run_sweep.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p results/logs

if [ -n "$SLURM_JOB_ID" ]; then
  echo "[warn] running inside an existing Slurm allocation ($SLURM_JOB_ID):" \
       "srun below becomes job steps INSIDE it (serialized on its GPUs)." >&2
fi

DATASETS=(${DATASETS:-Crime})
ALGS=(${ALGS:-supipm})
LMDAS=(${LMDAS:-0.0 0.1 0.3 1.0 3.0 10.0})
SEEDS=(${SEEDS:-2023 2024 2025 2026 2027})
MAX_PAR=${MAX_PAR:-2}
EXTRA_ARGS=${EXTRA_ARGS:-}   # extra train.py flags, e.g. "--gamma_s 0.03" (word-split on purpose)
LOG_TAG=${LOG_TAG:-}         # suffix for log filenames, e.g. "-gs0.03" (avoid clobbering across variants)
PY=/usr/local/miniconda3/envs/nine/bin/python

echo "== GPU usage (check for non-Slurm occupants before launching) =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo "== queue =="
squeue -u ys971217

for dataset in "${DATASETS[@]}"; do
  for alg in "${ALGS[@]}"; do
    for lmda in "${LMDAS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "[sweep] $dataset alg=$alg lmda_f=$lmda seed=$seed"
        # --mem below the 30G DefMemPerNode so 4 jobs fit alongside others (node RAM 125G)
        srun --gres=gpu:1 --cpus-per-task=10 --partition=idea --time=14:00:00 --mem=25G \
          --job-name="$alg-$dataset-$lmda-$seed" \
          bash -c "echo \"[sweep] CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"; exec $PY src/train.py --config configs/default.yaml \
          --dataset '$dataset' --alg '$alg' --lmda_f '$lmda' --seed '$seed' $EXTRA_ARGS" \
          > "results/logs/sweep-$dataset-$alg-$lmda-$seed$LOG_TAG.log" 2>&1 &
        sleep 1
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 5; done
      done
    done
  done
done

wait
echo "[sweep] all jobs finished"
