"""FREM evaluation protocol + test-set sup_s ReLU-IPM.

The metric formulas are verbatim ports of FRL-GDP-full/src/train.py
(`_validate`, `_compute_accuracy`, `_compute_fairness`, `_compute_MI`) with the
device handled explicitly instead of hardcoded `.cuda()`; `compute_hgr`/`kde` are
imported from FREM's `base.utils`. On top of FREM's metrics we report
sup_s IPM_hat(h, s) over a fine s-grid (per-s maximization over the ReLU class).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import digamma
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors, KDTree
from math import pi, sqrt

_HERE = os.path.dirname(os.path.abspath(__file__))
FREM_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'FRL-GDP-full', 'src'))
if FREM_SRC not in sys.path:
    sys.path.append(FREM_SRC)

from base.utils import compute_hgr, kde  # noqa: E402  (FREM, read-only)

from ipm import relu_ipm_sup_grid  # noqa: E402
from models import normalize_reps  # noqa: E402


def collect(model, loader, input_dim, task, device, z_norm=True):
    """Port of FREM Trainer._validate (src/train.py:550-572), device-flexible.

    With z_norm=True (default) the representation is the ball-normalized
    z_tilde = normalize_reps(extractor(x)) and the head consumes it (matching
    training) — `reps` IS the actual representation, so all metrics (incl.
    mi_z_s) are computed on it. z_norm=False restores the raw-z original."""
    model.eval()
    all_reps, all_preds, all_labels, all_sensitives = [], [], [], []
    with torch.no_grad():
        for inputs, labels, sensitives in loader:
            inputs = inputs.to(device).view(-1, input_dim)
            z = model.extractor(inputs)
            if z_norm:
                z = normalize_reps(z)
            all_reps.append(z.detach())
            all_preds.append(model.final_layer(z).detach())
            all_labels.append(labels.to(device))
            all_sensitives.append(sensitives.to(device))
    all_reps = torch.cat(all_reps)
    all_preds = torch.cat(all_preds)
    all_labels, all_sensitives = torch.cat(all_labels), torch.cat(all_sensitives)
    if task == 'cls':
        all_probs = torch.softmax(all_preds, dim=1)[:, 1].flatten()
    elif task == 'reg':
        all_probs = torch.sigmoid(all_preds.flatten())
    all_preds = torch.argmax(all_preds, dim=1).float()
    return all_reps, all_preds, all_probs, all_labels, all_sensitives


def compute_accuracy(preds, probs, labels, task):
    """Port of FREM Trainer._compute_accuracy (src/train.py:405-426)."""
    if task == 'cls':
        acc = (preds == labels).float().mean()
        bacc = (preds[labels == 0] == labels[labels == 0]).float().mean()
        bacc += (preds[labels == 1] == labels[labels == 1]).float().mean()
        bacc /= 2.0
        ap = average_precision_score(labels.detach().cpu().numpy(),
                                     probs.detach().cpu().numpy())
        acc, bacc, ap = round(acc.item(), 4), round(bacc.item(), 4), round(ap.item(), 4)
        mse, mae = None, None
    elif task == 'reg':
        acc, bacc, ap = None, None, None
        mse = round(nn.MSELoss()(probs.flatten(), labels.flatten()).item(), 4)
        mae = round(nn.L1Loss()(probs.flatten(), labels.flatten()).item(), 4)
    return acc, bacc, ap, mse, mae


def compute_MI(c, d, n_neighbors=5):
    """Port of FREM Trainer._compute_MI (src/train.py:365-402)."""
    n_samples = c.shape[0]
    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn_est = NearestNeighbors()
    for label in np.unique(d):
        mask = d == label
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn_est.set_params(n_neighbors=k)
            nn_est.fit(c[mask])
            r = nn_est.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count
    mask = label_counts > 1
    n_samples = np.sum(mask)
    if n_samples == 0:
        # fully continuous S (every value unique, e.g. SynthB): the discrete-label
        # estimator is undefined — FREM's original would crash here
        return float('nan')
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]
    kd = KDTree(c)
    m_all = kd.query_radius(c, radius, count_only=True, return_distance=False)
    m_all = np.array(m_all) - 1.0
    mi = (digamma(n_samples) + np.mean(digamma(k_all))
          - np.mean(digamma(label_counts)) - np.mean(digamma(m_all + 1)))
    return max(0, mi)


def compute_fairness(reps, probs, sensitives):
    """Port of FREM Trainer._compute_fairness (src/train.py:449-505); returns the
    rounded values (gdp_wo_kernel, gdp_w_kernel, hgr, mi_y_s, mi_z_s)."""
    n_support = probs.size(0)
    mean_probs = probs.mean()
    gdp_wo_kernel = 0.0
    for unique_sensitive in torch.unique(sensitives):
        sub_probs = probs[sensitives == unique_sensitive]
        gdp_wo_kernel += sub_probs.size(0) / n_support * (sub_probs.mean() - mean_probs).abs()

    device = probs.device
    x_approx_interval = 1e-3
    x_approx = torch.arange(x_approx_interval, 1 - x_approx_interval, x_approx_interval).to(device)
    n = sensitives.size(0)
    d = 1
    bandwidth = torch.tensor((n * (d + 2) / 4.) ** (-1. / (d + 4))).to(device)
    x_approx_repeat = x_approx.repeat_interleave(n).reshape((-1, n))
    attention_weights = F.softmax(-(x_approx_repeat - sensitives) ** 2 / (bandwidth ** 2) / 2, dim=1).float()
    y_hat = torch.matmul(attention_weights, probs)
    y_mean = torch.mean(probs)
    unsqueeze_sensitives = sensitives.unsqueeze(0)
    pdf_values = (
        torch.exp(-((x_approx_repeat - unsqueeze_sensitives) ** 2 / (bandwidth ** 2) / 2))
        ).mean(dim=-1) / sqrt(2 * pi) / bandwidth
    gdp_w_kernel = torch.sum(torch.abs(y_hat - y_mean) * pdf_values) / torch.sum(pdf_values)

    try:
        hgr = compute_hgr(probs.detach().cpu(), sensitives.detach().cpu(), kde)
        hgr_value = round(hgr.item(), 4)
    except Exception:  # SVD fails on (near-)constant predictions; keep the run alive
        hgr_value = float('nan')

    mi_y_s = mutual_info_regression(sensitives.detach().cpu().numpy().reshape(-1, 1),
                                    probs.detach().cpu().numpy())[0]
    mi_z_s = compute_MI(reps.detach().cpu().numpy(), sensitives.detach().cpu().numpy())

    return (round(float(gdp_wo_kernel), 4), round(gdp_w_kernel.item(), 4),
            hgr_value, round(float(mi_y_s), 4), round(float(mi_z_s), 4))


def compute_inf_gdp(probs, sensitives, lo, hi):
    """L_infinity generalized DP:  inf_GDP = sup_s | E[y_hat | S=s] - E[y_hat] |.

    Same Nadaraya-Watson estimator, Silverman-type bandwidth and 1e-3 grid as the
    expectation-based gdp_w_kernel metric (Jiang et al. / FREM), but the
    pdf-weighted average over s is replaced by a max over the alpha-trimmed
    TRAIN-split range [lo, hi] of raw-scale S — the same range convention as the
    training sup and the sup_ipm eval, so the sup never rides on extrapolation
    where S has no mass. Returns (inf_gdp, argmax_s_raw).
    """
    device = probs.device
    x_approx_interval = 1e-3
    x_approx = torch.arange(x_approx_interval, 1 - x_approx_interval, x_approx_interval).to(device)
    x_approx = x_approx[(x_approx >= lo) & (x_approx <= hi)]
    if x_approx.numel() == 0:  # degenerate trim range
        x_approx = torch.tensor([min(max((lo + hi) / 2, 1e-3), 1 - 1e-3)], device=device)
    n = sensitives.size(0)
    d = 1
    bandwidth = torch.tensor((n * (d + 2) / 4.) ** (-1. / (d + 4))).to(device)
    x_approx_repeat = x_approx.repeat_interleave(n).reshape((-1, n))
    attention_weights = F.softmax(-(x_approx_repeat - sensitives) ** 2 / (bandwidth ** 2) / 2, dim=1).float()
    y_hat = torch.matmul(attention_weights, probs)
    dev = (y_hat - torch.mean(probs)).abs()
    k = int(dev.argmax())
    return round(dev[k].item(), 6), round(x_approx[k].item(), 4)


def sup_ipm_on_split(reps, sensitives, data, cfg, critic, device):
    """sup_s IPM_hat(h, s) on a full split: fine grid over the SAME trimmed range
    the training sup was constrained to (train-split alpha-quantiles of
    standardized S), per-s maximization over the paper ReLU class. `reps` is the
    ball-normalized representation already (collect returns z_tilde) — do NOT
    normalize again here."""
    s_std = (sensitives.float() - data['s_mean']) / data['s_sd']
    lo, hi = data['s_lo'], data['s_hi']
    grid = torch.linspace(lo, hi, cfg['eval_grid'], device=device)
    vals = relu_ipm_sup_grid(reps, s_std.to(device), cfg['bandwidth'], grid,
                             n_restarts=cfg['eval_restarts'], n_steps=cfg['eval_steps'],
                             lr=cfg['eval_lr'], warm_critic=critic,
                             mode=cfg.get('critic_proj', 'sphere_box'))
    k = int(vals.argmax())
    s_star_std = grid[k].item()
    return dict(sup_ipm=round(vals[k].item(), 6),
                s_star_std=round(s_star_std, 4),
                s_star_raw=round(s_star_std * data['s_sd'] + data['s_mean'], 4),
                grid_lo=round(lo, 4), grid_hi=round(hi, 4)), \
        grid.cpu().numpy(), vals.cpu().numpy()


def run_eval(model, critic, data, cfg, device, run_dir=None):
    """FREM final-epoch protocol: metrics on traineval/val/test with the jointly
    trained model, plus sup_s IPM_hat on val and test."""
    torch.manual_seed(100000 + cfg['seed'])  # reproducible eval restarts
    task, input_dim = data['task'], data['input_dim']
    z_norm = bool(cfg.get('z_norm', True))
    results = {}
    curves = {}
    for split, loader_key in [('train', 'traineval'), ('val', 'val'), ('test', 'test')]:
        reps, preds, probs, labels, sens = collect(model, data[loader_key], input_dim, task,
                                                   device, z_norm=z_norm)
        loss = F.binary_cross_entropy(probs, labels.float()).item()
        acc, bacc, ap, mse, mae = compute_accuracy(preds, probs, labels, task)
        # sens passed uncast (float64 for Crime/Adult) exactly as FREM's _compute_fairness receives it
        gdp_wo, gdp_w, hgr, mi_y_s, mi_z_s = compute_fairness(reps, probs, sens)
        # inf_GDP: sup over the train-split alpha-trimmed raw-scale S range
        lo_raw = data['s_lo'] * data['s_sd'] + data['s_mean']
        hi_raw = data['s_hi'] * data['s_sd'] + data['s_mean']
        inf_gdp, inf_gdp_s = compute_inf_gdp(probs, sens, lo_raw, hi_raw)
        results[split] = dict(loss=loss, acc=acc, bacc=bacc, ap=ap, mse=mse, mae=mae,
                              gdp_wo_kernel=gdp_wo, gdp_w_kernel=gdp_w, hgr=hgr,
                              mi_y_s=mi_y_s, mi_z_s=mi_z_s,
                              inf_gdp=inf_gdp, inf_gdp_s=inf_gdp_s)
        if split in ('val', 'test'):
            sup, grid_np, vals_np = sup_ipm_on_split(reps, sens, data, cfg, critic, device)
            results[split].update(sup)
            curves[f'{split}_grid_std'] = grid_np
            curves[f'{split}_ipm'] = vals_np
    if run_dir is not None:
        np.savez(os.path.join(run_dir, 'sup_ipm_curve.npz'), **curves)
    return results


def main():
    """Standalone re-evaluation of a finished run directory."""
    import yaml
    from data import apply_preset, get_loaders
    from models import build_model, ReLUCritic, REP_DIM

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, 'config.yaml')) as f:
        cfg = apply_preset(yaml.safe_load(f))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = get_loaders(cfg)
    ckpt = torch.load(os.path.join(args.run_dir, 'model.pt'), map_location=device)
    model = build_model(data['input_dim'], data['task']).to(device)
    model.load_state_dict(ckpt['model'])
    critic = None
    if ckpt.get('critic') is not None:
        critic = ReLUCritic(REP_DIM, cfg['critic_num']).to(device)
        critic.load_state_dict(ckpt['critic'])
    results = run_eval(model, critic, data, cfg, device, run_dir=args.run_dir)
    with open(os.path.join(args.run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
