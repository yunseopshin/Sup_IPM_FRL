"""Which half of sup_s sup_v is stale? Split the gap between what the TRAINED POOL
can see and what a fresh strong critic finds.

For a finished run, on one split, three numbers on the SAME data:

  A  pool_s_free : max_s max_c |Delta_c(s)| with the trained pool FIXED (v frozen)
                   and s re-optimized over a fine grid on the full split.
                   -> the best the training critic pool could ever do, given a
                      perfect search over s.
  B  eval_sup    : the eval-strength sup (v re-fit per grid point over the whole
                   ReLU class, warm-started from the pool) = results.json sup_ipm.
  C  train_ipm   : what the pool actually reported during training (train_log.csv,
                   last 10 epochs, batch-level).

Reading:
  B / A  >> 1  the POOL'S FUNCTIONS are stale: no amount of s search saves them,
               only re-fitting (theta, mu). s-diversity cannot help here.
  A / C  >> 1  the pool was fine but was being evaluated at the wrong s during
               training (a search problem, which is what s_mode addresses).

Usage: python scripts/diag_pool_vs_eval.py <run_dir> [<run_dir> ...] [--split test]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from data import apply_preset, get_loaders  # noqa: E402
from evaluate import collect  # noqa: E402
from ipm import kernel_weights, pool_ipm  # noqa: E402
from models import build_model, ReLUCritic, REP_DIM  # noqa: E402


def pool_sup_s_free(reps, s_std, critic, h, lo, hi, n_grid=4001, chunk=256):
    """A: trained pool fixed, s optimized over a fine grid on the full split."""
    device = reps.device
    with torch.no_grad():
        V = critic(reps)                                   # [n, C]
        grid = torch.linspace(lo, hi, n_grid, device=device)
        best = -float('inf')
        best_s = float('nan')
        for i in range(0, n_grid, chunk):                  # chunked: [g, n] can be large
            g = grid[i:i + chunk]
            vals = pool_ipm(kernel_weights(g, s_std, h), V)  # [g, C]
            m, k = vals.max(dim=1).values.max(dim=0), None
            if m.values.item() > best:
                best = m.values.item()
                best_s = g[m.indices].item()
    return best, best_s


def train_tail(run_dir, col='ipm_s_star', n_last=10):
    p = os.path.join(run_dir, 'train_log.csv')
    if not os.path.exists(p):
        return float('nan')
    rows = list(csv.DictReader(open(p)))
    if not rows or col not in rows[0]:
        return float('nan')
    return float(np.mean([float(r[col]) for r in rows[-n_last:]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--split', default='test', choices=['test', 'train', 'val'])
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader_key = {'train': 'traineval', 'val': 'val', 'test': 'test'}[args.split]
    cache = {}
    rows = []
    for run_dir in args.run_dirs:
        with open(os.path.join(run_dir, 'config.yaml')) as f:
            cfg = apply_preset(yaml.safe_load(f))
        key = (cfg['dataset'], cfg['seed'], cfg['batch_size'], cfg.get('mini', False))
        if key not in cache:
            cache[key] = get_loaders(cfg)
        data = cache[key]
        ckpt = torch.load(os.path.join(run_dir, 'model.pt'), map_location=device)
        if ckpt.get('critic') is None:
            print(f'[skip] no critic pool: {run_dir}')
            continue
        model = build_model(data['input_dim'], data['task']).to(device)
        model.load_state_dict(ckpt['model'])
        critic = ReLUCritic(REP_DIM, cfg['critic_num']).to(device)
        critic.load_state_dict(ckpt['critic'])

        reps, _, _, _, sens = collect(model, data[loader_key], data['input_dim'],
                                      data['task'], device,
                                      z_norm=bool(cfg.get('z_norm', True)))
        s_std = (sens.float() - data['s_mean']) / data['s_sd']
        A, A_s = pool_sup_s_free(reps, s_std.to(device), critic, float(cfg['bandwidth']),
                                 data['s_lo'], data['s_hi'])
        with open(os.path.join(run_dir, 'results.json')) as f:
            res = json.load(f)
        B = res[args.split]['sup_ipm'] if args.split in ('val', 'test') else float('nan')
        C = train_tail(run_dir)
        rows.append(dict(run=os.path.relpath(run_dir, os.getcwd()),
                         s_mode=cfg.get('s_mode', 'shared'), cs=cfg['critic_step'],
                         lmda=cfg['lmda_f'], seed=cfg['seed'],
                         pool_s_free=A, pool_s_at=A_s, eval_sup=B, train_ipm=C,
                         bacc=res[args.split].get('bacc')))
        print(json.dumps(rows[-1]), flush=True)

    print(f'\n=== split={args.split}: which half of the sup is stale ===')
    print(f"{'mode':>10} {'cs':>3} {'λ':>5} {'seed':>5} | {'C train':>8} {'A pool,s free':>14} "
          f"{'B eval(v free)':>14} | {'A/C':>6} {'B/A':>6}")
    for r in rows:
        flag = '  COLLAPSED' if (r['bacc'] is not None and r['bacc'] <= 0.5001) else ''
        print(f"{r['s_mode']:>10} {r['cs']:>3} {r['lmda']:>5} {r['seed']:>5} | "
              f"{r['train_ipm']:>8.4f} {r['pool_s_free']:>14.4f} {r['eval_sup']:>14.4f} | "
              f"{r['pool_s_free'] / max(r['train_ipm'], 1e-12):>6.2f} "
              f"{r['eval_sup'] / max(r['pool_s_free'], 1e-12):>6.2f}{flag}")

    print('\nA/C >> 1: the pool was being evaluated at the wrong s during training '
          '(a SEARCH problem - what s_mode fixes).')
    print('B/A >> 1: the pool FUNCTIONS themselves are stale; re-optimizing s cannot '
          'recover it, only re-fitting (theta, mu) can (a CAPACITY/BUDGET problem).')


if __name__ == '__main__':
    main()
