"""Diagnostic: eval-strength sup_s IPM on the TRAIN split for finished runs.

Decomposes the train-log-vs-test sup_ipm gap:
  - training-pool IPM (train_log.csv)   : weak critic, train batches
  - THIS: strong-eval sup on train split : strong critic, train data
  - results.json val/test sup_ipm        : strong critic, held-out data
If train-split strong sup >> training-pool IPM -> weak-critic gap.
If test sup >> train-split strong sup -> data-generalization gap.

Usage: python scripts/diag_train_split_sup.py <run_dir> [<run_dir> ...]
"""
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from data import apply_preset, get_loaders  # noqa: E402
from evaluate import collect, sup_ipm_on_split  # noqa: E402
from models import build_model, ReLUCritic, REP_DIM  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader_cache = {}
out = []
for run_dir in sys.argv[1:]:
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        cfg = apply_preset(yaml.safe_load(f))
    key = (cfg['dataset'], cfg['seed'], cfg['batch_size'], cfg.get('mini', False))
    if key not in loader_cache:
        loader_cache[key] = get_loaders(cfg)
    data = loader_cache[key]
    ckpt = torch.load(os.path.join(run_dir, 'model.pt'), map_location=device)
    model = build_model(data['input_dim'], data['task']).to(device)
    model.load_state_dict(ckpt['model'])
    critic = None
    if ckpt.get('critic') is not None:
        critic = ReLUCritic(REP_DIM, cfg['critic_num']).to(device)
        critic.load_state_dict(ckpt['critic'])
    torch.manual_seed(200000 + cfg['seed'])  # reproducible restarts, distinct from run_eval
    reps, _, _, _, sens = collect(model, data['traineval'], data['input_dim'],
                                  data['task'], device, z_norm=bool(cfg.get('z_norm', True)))
    sup, _, _ = sup_ipm_on_split(reps, sens, data, cfg, critic, device)
    with open(os.path.join(run_dir, 'results.json')) as f:
        r = json.load(f)
    row = dict(run=os.path.relpath(run_dir, 'results'), alg=cfg['alg'], lmda_f=cfg['lmda_f'],
               seed=cfg['seed'], train_sup_ipm=sup['sup_ipm'], train_s_star=sup['s_star_std'],
               val_sup_ipm=r['val']['sup_ipm'], test_sup_ipm=r['test']['sup_ipm'])
    out.append(row)
    print(json.dumps(row), flush=True)

print('\n=== summary (train-split strong sup vs held-out sup) ===')
for row in out:
    print(f"{row['alg']:8s} lam={row['lmda_f']:<4g} seed={row['seed']} | "
          f"TRAIN(strong)={row['train_sup_ipm']:.4f}  val={row['val_sup_ipm']:.4f}  "
          f"test={row['test_sup_ipm']:.4f}")
