"""Does the trained pool point where the auditor looks?

Two measurements per finished run, on one split:

  POOL DIVERSITY  the 100 critic directions theta_c live on S^{d-1}. Report the
                  effective rank of the [C, d] matrix, the mean pairwise |cos|,
                  and the spread of the per-critic best Delta_c. A pool that has
                  collapsed onto one function has eff.rank ~ 1, |cos| ~ 1 and a
                  flat Delta_c profile.

  AUDITOR GAP     re-run the eval-strength ascent over the ReLU class at the
                  pool's own best s (v free, s fixed), and report where the
                  winning direction sits relative to the pool: cos to the nearest
                  pool member, and how much value the pool leaves on the table at
                  that very s. If the auditor wins with a direction the pool
                  already has, the pool is merely under-trained; if it wins with
                  a direction the pool does not contain, the pool is looking the
                  wrong way.

Usage: python scripts/diag_pool_diversity.py <run_dir> [...] [--split test]
"""
import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from data import apply_preset, get_loaders  # noqa: E402
from evaluate import collect  # noqa: E402
from ipm import kernel_weights, pool_ipm  # noqa: E402
from models import build_model, ReLUCritic, REP_DIM  # noqa: E402


def eff_rank(m):
    s = torch.linalg.svdvals(m)
    p = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()))


def best_critic_at_s(z, w_row, d, n_restarts=64, n_steps=600, lr=1e-2, generator=None):
    """Free ascent over {relu(theta^T z + mu): ||theta||=1, mu in [-1,1]} at a fixed
    s (w_row = kernel_weights(s) - 1/n). Returns (best value, theta*, mu*)."""
    theta = torch.randn(n_restarts, d, generator=generator)
    theta = theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)
    mu = torch.rand(n_restarts, 1, generator=generator) * 2.0 - 1.0
    param = torch.cat([theta, mu], dim=1).to(z.device).requires_grad_(True)
    opt = torch.optim.Adam([param], lr=lr)
    best_v, best_p = -float('inf'), None
    for step in range(n_steps + 1):
        V = torch.relu(z @ param[:, :d].t() + param[:, d])       # [n, R]
        obj = (w_row @ V).abs()                                  # [R]
        with torch.no_grad():
            k = int(obj.argmax())
            if obj[k].item() > best_v:
                best_v, best_p = obj[k].item(), param[k].detach().clone()
        if step == n_steps:
            break
        opt.zero_grad()
        (-obj.sum()).backward()
        opt.step()
        with torch.no_grad():
            th = param[:, :d]
            th.div_(th.norm(dim=1, keepdim=True).clamp_min(1e-12))
            param[:, d].clamp_(-1.0, 1.0)
    return best_v, best_p[:d], float(best_p[d])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--split', default='test', choices=['test', 'train', 'val'])
    ap.add_argument('--n_grid', type=int, default=513)
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
            continue
        model = build_model(data['input_dim'], data['task']).to(device)
        model.load_state_dict(ckpt['model'])
        critic = ReLUCritic(REP_DIM, cfg['critic_num']).to(device)
        critic.load_state_dict(ckpt['critic'])

        reps, _, _, _, sens = collect(model, data[loader_key], data['input_dim'],
                                      data['task'], device, z_norm=bool(cfg.get('z_norm', True)))
        s_std = (sens.float() - data['s_mean']) / data['s_sd']
        n, d = reps.shape
        with torch.no_grad():
            th = critic.fc.weight.data                                  # [C, d]
            thn = th / th.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cos = (thn @ thn.t()).abs()
            C = thn.size(0)
            mean_abs_cos = float((cos.sum() - C) / (C * (C - 1)))
            rank_theta = eff_rank(thn)

            V = critic(reps)
            grid = torch.linspace(data['s_lo'], data['s_hi'], args.n_grid)
            vals = pool_ipm(kernel_weights(grid, s_std, cfg['bandwidth']), V)   # [G, C]
            per_c = vals.max(dim=0).values
            pool_best = float(vals.max())
            gi = int(vals.max(dim=1).values.argmax())
            s_at = float(grid[gi])
            delta_spread = float(per_c.max() / per_c.median().clamp_min(1e-12))
            w_row = kernel_weights(grid[gi:gi + 1], s_std, cfg['bandwidth'])[0] - 1.0 / n

        gen = torch.Generator().manual_seed(12345)
        free_v, free_th, free_mu = best_critic_at_s(reps, w_row, d, generator=gen)
        with torch.no_grad():
            ft = free_th / free_th.norm().clamp_min(1e-12)
            cos_to_pool = float((thn @ ft).abs().max())
        rows.append(dict(mode=cfg.get('s_mode', 'shared'), cs=cfg['critic_step'],
                         lmda=cfg['lmda_f'], seed=cfg['seed'],
                         rank_theta=rank_theta, mean_abs_cos=mean_abs_cos,
                         delta_spread=delta_spread, pool_best=pool_best, s_at=s_at,
                         free_at_same_s=free_v, gap=free_v / max(pool_best, 1e-12),
                         cos_to_pool=cos_to_pool))
        print(json.dumps(rows[-1]), flush=True)

    print(f'\n=== pool diversity and the auditor gap ({args.split} split, C=100, d=50) ===')
    print(f"{'mode':>10} {'cs':>3} {'λ':>5} {'seed':>5} | {'rank θ':>7} {'|cos|':>6} "
          f"{'Δmax/Δmed':>10} | {'pool@s':>8} {'free@s':>8} {'free/pool':>10} {'cos(θ*,pool)':>13}")
    for r in rows:
        print(f"{r['mode']:>10} {r['cs']:>3} {r['lmda']:>5} {r['seed']:>5} | "
              f"{r['rank_theta']:>7.2f} {r['mean_abs_cos']:>6.3f} {r['delta_spread']:>10.2f} | "
              f"{r['pool_best']:>8.4f} {r['free_at_same_s']:>8.4f} {r['gap']:>10.2f}x "
              f"{r['cos_to_pool']:>13.3f}")
    print('\nrank θ ~ 1 and |cos| ~ 1: the pool has collapsed onto a single direction.')
    print('free/pool at the SAME s: what a freshly fitted critic gains over the whole pool')
    print('cos(θ*, pool): 1.0 = the pool already contains the auditor direction (under-trained);')
    print('               ~0 = the pool is looking somewhere else entirely.')


if __name__ == '__main__':
    main()
