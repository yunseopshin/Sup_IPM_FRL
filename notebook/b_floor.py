"""Scenario B helper: permutation noise floor of the sup_ipm estimator.

For each lambda=0 run (per seed), permute S against the test representation
(enforcing independence) and re-run the eval sup -> the finite-sample floor the
C1 criterion compares against. Writes <out_root>/floor.json.
Run via srun (see B_free_removal.ipynb).
"""
import json
import os
import sys

import numpy as np
import torch
import yaml

ROOT, OUT_ROOT = sys.argv[1], sys.argv[2]
os.chdir(ROOT)
sys.path.insert(0, 'src')

from data import apply_preset, get_loaders          # noqa: E402
from models import build_model                      # noqa: E402
from ipm import relu_ipm_sup_grid                   # noqa: E402
from evaluate import collect                        # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
floors = []
for seed in [2023, 2024, 2025, 2026, 2027]:
    run_dir = os.path.join(OUT_ROOT, 'SynthB-synth_s', 'supipm', 'lmda_f-0.0', f'seed-{seed}')
    with open(run_dir + '/config.yaml') as f:
        cfg = apply_preset(yaml.safe_load(f))
    data = get_loaders(cfg)
    ckpt = torch.load(run_dir + '/model.pt', map_location=device)
    model = build_model(data['input_dim'], data['task']).to(device)
    model.load_state_dict(ckpt['model'])
    reps, _, _, _, sens = collect(model, data['test'], data['input_dim'],
                                  data['task'], device)
    s_std = ((sens.float() - data['s_mean']) / data['s_sd']).to(device)
    grid = torch.linspace(data['s_lo'], data['s_hi'], cfg['eval_grid'], device=device)
    for i in range(3):
        g = torch.Generator().manual_seed(1000 * seed + i)
        perm = torch.randperm(s_std.numel(), generator=g).to(device)
        torch.manual_seed(77 + i)
        v = relu_ipm_sup_grid(reps, s_std[perm], cfg['bandwidth'], grid,
                              n_restarts=cfg['eval_restarts'],
                              n_steps=cfg['eval_steps'], lr=cfg['eval_lr'])
        floors.append(v.max().item())

out = dict(floors=floors, mean=float(np.mean(floors)), sd=float(np.std(floors)))
with open(os.path.join(OUT_ROOT, 'floor.json'), 'w') as f:
    json.dump(out, f, indent=2)
print('floor mean %.4f sd %.4f (n=%d)' % (out['mean'], out['sd'], len(floors)))
