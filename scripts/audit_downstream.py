"""Downstream-head audit: what L_inf-GDP can a NEW head exhibit on a frozen representation?

For each finished run dir: freeze the encoder, precompute z_tilde on train/test, then
  (a) retrained-head audit: k fresh Linear(50,2) heads trained with the FREM supervised
      recipe (200 epochs, batch 1024, Adam 1e-3 wd 1e-2 betas (0.5,0.999)) on the frozen
      reps -> test acc + test inf_gdp of each new head (benign downstream user).
  (b) adversarial-head audit: bounded probabilistic head g(z) = sigmoid(w^T z + b)
      trained to MAXIMIZE the exact inf_gdp functional (same NW estimator / Silverman
      bandwidth / 1e-3 grid / train-trimmed range as evaluate.compute_inf_gdp) on the
      TRAIN split; reported on test via compute_inf_gdp (worst-case downstream head).

This is the experimental cash-out of the FREM-paper Theorems 1-2 promise ("any
prediction head on a fair representation is fair") that FREM itself never ran.
Prediction under the sup-IPM theory: adversarial test inf_gdp tracks the rep's residual
sup_ipm, NOT its MI(Z,S).

Usage: python scripts/audit_downstream.py <run_dir> [<run_dir> ...]
Writes one JSON line per run dir to stdout; summary table at the end.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from data import apply_preset, get_loaders  # noqa: E402
from evaluate import collect, compute_inf_gdp  # noqa: E402
from models import build_model  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_RETRAIN = 3
N_ADV_RESTARTS = 4
ADV_STEPS = 600
ADV_LR = 1e-2


def nw_weights(sens, lo, hi):
    """Exact clone of compute_inf_gdp's attention matrix (differentiable target)."""
    x = torch.arange(1e-3, 1 - 1e-3, 1e-3, device=device)
    x = x[(x >= lo) & (x <= hi)]
    n = sens.size(0)
    bw = torch.tensor((n * 3 / 4.) ** (-1. / 5)).to(device)
    w = F.softmax(-(x.repeat_interleave(n).reshape(-1, n) - sens) ** 2 / (bw ** 2) / 2, dim=1).float()
    return w


def retrain_head(reps_tr, labels_tr, seed, epochs=200, bs=1024):
    g = torch.Generator(device='cpu').manual_seed(seed)
    head = torch.nn.Linear(reps_tr.size(1), 2).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-2, betas=(0.5, 0.999))
    n = reps_tr.size(0)
    for _ in range(epochs):
        idx = torch.randperm(n, generator=g)
        for i in range(0, n - bs + 1, bs):  # drop_last=True like the protocol
            j = idx[i:i + bs].to(device)
            loss = F.cross_entropy(head(reps_tr[j]), labels_tr[j])
            opt.zero_grad(); loss.backward(); opt.step()
    return head


def adversarial_head(reps_tr, W_tr, seed):
    """Maximize dev = max_s |W @ sigmoid(w^T z + b) - mean(sigmoid(...))| on train."""
    best_obj, best = -1.0, None
    for r in range(N_ADV_RESTARTS):
        g = torch.Generator(device='cpu').manual_seed(seed * 100 + r)
        w = torch.randn(reps_tr.size(1), generator=g).to(device).requires_grad_(True)
        b = torch.zeros(1, device=device, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=ADV_LR)
        for _ in range(ADV_STEPS):
            probs = torch.sigmoid(reps_tr @ w + b)
            dev = (W_tr @ probs - probs.mean()).abs().max()
            opt.zero_grad(); (-dev).backward(); opt.step()
        with torch.no_grad():
            probs = torch.sigmoid(reps_tr @ w + b)
            obj = (W_tr @ probs - probs.mean()).abs().max().item()
        if obj > best_obj:
            best_obj, best = obj, (w.detach().clone(), b.detach().clone())
    return best, best_obj


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
    zn = bool(cfg.get('z_norm', True))
    reps_tr, _, _, labels_tr, sens_tr = collect(model, data['traineval'], data['input_dim'],
                                                data['task'], device, z_norm=zn)
    reps_te, _, _, labels_te, sens_te = collect(model, data['test'], data['input_dim'],
                                                data['task'], device, z_norm=zn)
    labels_tr = labels_tr.long()
    lo = data['s_lo'] * data['s_sd'] + data['s_mean']
    hi = data['s_hi'] * data['s_sd'] + data['s_mean']
    W_tr = nw_weights(sens_tr.float().to(device), lo, hi)
    with open(os.path.join(run_dir, 'results.json')) as f:
        r0 = json.load(f)

    torch.manual_seed(300000 + cfg['seed'])
    # (a) retrained heads
    re_gdp, re_acc = [], []
    for k in range(N_RETRAIN):
        head = retrain_head(reps_tr, labels_tr, seed=cfg['seed'] * 10 + k)
        with torch.no_grad():
            logits = head(reps_te)
            probs = torch.softmax(logits, 1)[:, 1]
            re_acc.append((logits.argmax(1) == labels_te.long()).float().mean().item())
        re_gdp.append(compute_inf_gdp(probs, sens_te.to(device), lo, hi)[0])
    # (b) adversarial head
    (w, b), train_obj = adversarial_head(reps_tr, W_tr, seed=cfg['seed'])
    with torch.no_grad():
        probs_adv = torch.sigmoid(reps_te @ w + b)
    adv_gdp, adv_s = compute_inf_gdp(probs_adv, sens_te.to(device), lo, hi)

    row = dict(run=run_dir.replace('results/', ''), alg=cfg['alg'], lmda_f=cfg['lmda_f'],
               seed=cfg['seed'], critic_step=cfg.get('critic_step'),
               joint_inf_gdp=r0['test']['inf_gdp'], sup_ipm=r0['test']['sup_ipm'],
               mi_z_s=r0['test']['mi_z_s'],
               retrain_inf_gdp_mean=float(np.mean(re_gdp)),
               retrain_inf_gdp_max=float(np.max(re_gdp)),
               retrain_acc_mean=float(np.mean(re_acc)),
               adv_inf_gdp=adv_gdp, adv_inf_gdp_train=round(train_obj, 6), adv_s=adv_s)
    out.append(row)
    print(json.dumps(row), flush=True)

print('\n=== summary (test-split, per run) ===')
print(f"{'run':58s} {'sup_ipm':>8} {'MI(Z,S)':>8} {'joint':>7} {'retrain':>8} {'ADV':>7}")
for r in out:
    print(f"{r['run']:58s} {r['sup_ipm']:8.4f} {r['mi_z_s']:8.4f} {r['joint_inf_gdp']:7.4f} "
          f"{r['retrain_inf_gdp_max']:8.4f} {r['adv_inf_gdp']:7.4f}")
