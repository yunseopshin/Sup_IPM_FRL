#!/bin/bash
# One (dataset, lambda, seed) run via srun (CLAUDE.md: GPU jobs go through Slurm).
# Usage: scripts/run_single.sh [train.py args], e.g.
#   scripts/run_single.sh --dataset Crime --lmda_f 1.0 --seed 2023
set -e
cd "$(dirname "$0")/.."

srun --gres=gpu:1 --cpus-per-task=10 --partition=idea --time=14:00:00 \
  /usr/local/miniconda3/envs/nine/bin/python src/train.py --config configs/default.yaml "$@"
