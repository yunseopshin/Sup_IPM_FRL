#!/bin/bash
# s_mode comparison grid (PERCRITIC_S.md sec.5): (critic_step 2 | 20) x
# (s_mode shared | percritic) on Crime, lmda_f=1.0, FREM seeds.
# One srun job (gpu:1) per run, at most MAX_PAR at once (default 2 - do not
# occupy all 4 GPUs). Finished runs are skipped, so it is restartable.
# Usage: [SEEDS="2023 2024"] [STEPS="2 20"] [MODES="shared percritic"] [MAX_PAR=2] \
#          scripts/run_smode_compare.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p results/logs

SEEDS=(${SEEDS:-2023 2024 2025 2026 2027})
STEPS=(${STEPS:-2 20})
MODES=(${MODES:-shared percritic})
MAX_PAR=${MAX_PAR:-2}
LMDA=${LMDA:-1.0}
PY=/usr/local/miniconda3/envs/nine/bin/python

echo "== GPU usage (check for non-Slurm occupants before launching) =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

for cs in "${STEPS[@]}"; do
  for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      sfx=''; [ "$mode" = percritic ] && sfx='-percritic'
      rd="results/smode_compare/cs${cs}/Crime-racepctblack/supipm${sfx}/lmda_f-${LMDA}/seed-${seed}"
      if [ -f "$rd/results.json" ]; then echo "[skip] cs$cs $mode $seed"; continue; fi
      echo "[smode] critic_step=$cs s_mode=$mode seed=$seed"
      tl=4:00:00; [ "$cs" -ge 10 ] && tl=12:00:00
      srun --gres=gpu:1 --cpus-per-task=10 --partition=idea --time=$tl --mem=25G \
        --job-name="cs$cs-$mode-$seed" \
        $PY src/train.py --config configs/default.yaml --dataset Crime --lmda_f "$LMDA" \
        --seed "$seed" --critic_step "$cs" --s_mode "$mode" \
        --out_root "results/smode_compare/cs${cs}" \
        > "results/logs/smode-cs$cs-$mode-$seed.log" 2>&1 &
      sleep 1
      while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 5; done
    done
  done
done

wait
echo "[smode] all jobs finished"
