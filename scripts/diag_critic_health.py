"""Is the trained critic pool alive? Dead / constant ReLU critics carry no signal.

A critic is v_c(z) = relu(theta_c^T z + mu_c) on ball-normalized z (||z|| < 1).
Two degenerate states make a critic useless for the IPM, and one of them is
absorbing:

  DEAD      theta_c^T z_i + mu_c <= 0 for every i  ->  v_c == 0 everywhere.
            Delta_c(s) = 0 for every s, AND the gradient w.r.t. (theta_c, mu_c)
            is exactly 0, so the critic can never recover on its own - only a
            move of the encoder can revive it.
  CONSTANT  v_c is (near-)constant over the data. Since sum_i (w_i(s) - 1/B) = 0,
            a constant critic has Delta_c(s) = 0 identically. Its gradient is not
            zero, so this state is escapable, but it contributes nothing now.

This script reports, per run: the fraction of the pool in each state, the
distribution of per-critic best Delta_c over a fine s-grid, and the scale of the
representation the pool has to work on.

Usage: python scripts/diag_critic_health.py <run_dir> [...] [--split test]
"""
import argparse
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


def eff_rank(z):
    """exp(entropy of the normalized singular-value spectrum) - a soft rank."""
    s = torch.linalg.svdvals(z - z.mean(dim=0, keepdim=True))
    p = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--split', default='test', choices=['test', 'train', 'val'])
    ap.add_argument('--n_grid', type=int, default=257)
    args = ap.parse_args()

    device = torch.device('cpu')
    loader_key = {'train': 'traineval', 'val': 'val', 'test': 'test'}[args.split]
    cache, rows = {}, []
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
                                      data['task'], device, z_norm=bool(cfg.get('z_norm', True)))
        s_std = (sens.float() - data['s_mean']) / data['s_sd']
        with torch.no_grad():
            V = critic(reps)                                  # [n, C]
            C = V.size(1)
            active = (V > 0).float().mean(dim=0)              # fraction of points active
            scale = V.std(dim=0)
            dead = (V.max(dim=0).values <= 0).float().mean().item()
            # "constant" relative to the representation scale the critic could see
            near_const = (scale <= 1e-3).float().mean().item()
            grid = torch.linspace(data['s_lo'], data['s_hi'], args.n_grid)
            per_c = pool_ipm(kernel_weights(grid, s_std, cfg['bandwidth']), V).max(dim=0).values
            per_c_sorted = per_c.sort(descending=True).values
            top = per_c_sorted[0].item()
            # how concentrated is the pool's total signal?
            share_top1 = top / per_c.sum().clamp_min(1e-12).item()
            useful = (per_c >= 0.1 * top).float().mean().item()
            znorm = reps.norm(dim=1)
            rank = eff_rank(reps)
        rows.append(dict(mode=cfg.get('s_mode', 'shared'), cs=cfg['critic_step'],
                         lmda=cfg['lmda_f'], seed=cfg['seed'], dead=dead,
                         near_const=near_const, active_med=float(active.median()),
                         useful=useful, top=top, share_top1=share_top1,
                         z_norm_mean=float(znorm.mean()), z_norm_p95=float(znorm.quantile(0.95)),
                         eff_rank=rank, v_scale_med=float(scale.median())))
        print(json.dumps(rows[-1]), flush=True)

    print(f'\n=== critic-pool health on the {args.split} split (C=100) ===')
    print(f"{'mode':>10} {'cs':>3} {'λ':>5} {'seed':>5} | {'dead':>6} {'const':>6} {'useful':>7} "
          f"{'act.med':>8} | {'max Δ_c':>8} {'‖z̃‖ mean':>9} {'eff.rank':>9} {'v scale':>8}")
    for r in rows:
        print(f"{r['mode']:>10} {r['cs']:>3} {r['lmda']:>5} {r['seed']:>5} | "
              f"{r['dead']:>6.0%} {r['near_const']:>6.0%} {r['useful']:>7.0%} "
              f"{r['active_med']:>8.2f} | {r['top']:>8.4f} {r['z_norm_mean']:>9.4f} "
              f"{r['eff_rank']:>9.2f} {r['v_scale_med']:>8.4f}")
    print('\ndead   : critics that output 0 on every point (zero gradient - frozen for good)')
    print('const  : critics whose output has std <= 1e-3 (Delta_c == 0 identically)')
    print('useful : critics reaching >= 10% of the pool best Delta_c at their own best s')


if __name__ == '__main__':
    main()
