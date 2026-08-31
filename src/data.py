"""Dataset loading for sup_IPM: thin wrapper around FREM's loaders.

FREM's data code (`FRL-GDP-full/src/base/datasets.py`) is imported read-only from its
own tree so that datasets, preprocessing and splits are identical to the FREM
experimental protocol (paper `FRL_EIPM.pdf`).
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
FREM_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'FRL-GDP-full', 'src'))
if FREM_SRC not in sys.path:
    sys.path.append(FREM_SRC)

from base.datasets import load_data  # noqa: E402  (FREM, read-only)

# FREM protocol presets (notebook/runner.py / src/main.py defaults)
PRESETS = {
    'Crime': dict(source='Crime_0', target='Crime_1', task='reg',
                  sensitive_attr='racepctblack', batch_size=200),
    'Adult': dict(source='Adult_0', target='Adult_1', task='cls',
                  sensitive_attr='age', batch_size=1024),
    # Scenario B synthetic validation (notebook/B_free_removal.ipynb)
    # v3 protocol: run with --weight_decay 0.0 (see notebook); gamma no longer needed.
    # synthb_A / synthb_n control the bump amplitude and split sizes (recorded in
    # each run's config.yaml for reproducibility).
    'SynthB': dict(source='SynthB', target='SynthB', task='reg',
                   sensitive_attr='synth_s', batch_size=200,
                   synthb_gamma=0.0, synthb_A=3.0, synthb_n=5000),
}

# --- Scenario B: cost-free-removal synthetic data (v2) -------------------------
# S_latent ~ N(0,1); X = (X_task in R^4, X_leak, X_noise in R^3), d = 8;
# X_leak = A exp(-(S-s0)^2/(2 tau^2)) + gamma * beta^T X_task + 0.5 N(0,1)
# Y = sigmoid(beta^T X_task + 0.3 N(0,1))
# X_leak is the ONLY S-dependent input. The gamma-term (v2, default 0.7) carries
# task signal that is REDUNDANT given X_task, so at lambda=0 the (ridge-like,
# wd=1e-2) model actively uses X_leak -> the representation genuinely inherits
# the S-bump; yet E[Y|X] = E[Y|X_task], so dropping X_leak stays exactly
# cost-free. (v1 with gamma=0 failed: weight decay removed the task-useless leak
# on its own at lambda=0, leaving nothing to test — see the notebook.)
# s0 = 1.2 lies INSIDE the alpha=0.05 trimmed range of N(0,1) so the training sup
# can reach it. S is MinMax-scaled to [0,1] on the train split (FREM protocol).
SYNTHB = dict(n_train=2000, n_val=500, n_test=2000,
              A=1.2, s0=1.2, tau=0.2, leak_noise=0.5,
              beta=(0.8, -0.6, 0.4, 0.5), y_noise=0.3)


def _synthb_draw(rng, n, gamma, A):
    s = rng.standard_normal(n)
    x_task = rng.standard_normal((n, 4))
    x_noise = rng.standard_normal((n, 3))
    task_signal = x_task @ np.asarray(SYNTHB['beta'])
    leak = (A * np.exp(-(s - SYNTHB['s0']) ** 2 / (2 * SYNTHB['tau'] ** 2))
            + gamma * task_signal
            + SYNTHB['leak_noise'] * rng.standard_normal(n))
    logits = task_signal + SYNTHB['y_noise'] * rng.standard_normal(n)
    y = 1.0 / (1.0 + np.exp(-logits))
    x = np.concatenate([x_task, leak[:, None], x_noise], axis=1)
    return x, y, s


def _synthb_loaders(cfg):
    from torch.utils.data import TensorDataset, DataLoader
    rng = np.random.default_rng(cfg['seed'])
    gamma = cfg.get('synthb_gamma', 0.0)
    A = cfg.get('synthb_A', SYNTHB['A'])
    n = cfg.get('synthb_n', SYNTHB['n_train'])
    xtr, ytr, srt = _synthb_draw(rng, n, gamma, A)
    xva, yva, sva = _synthb_draw(rng, max(n // 5, 200), gamma, A)
    xte, yte, ste = _synthb_draw(rng, n, gamma, A)
    smin, smax = srt.min(), srt.max()
    srt, sva, ste = [(a - smin) / (smax - smin) for a in (srt, sva, ste)]

    def mk(x, y, s, shuffle, drop_last):
        dset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y),
                             torch.from_numpy(s))
        return DataLoader(dset, cfg['batch_size'], num_workers=4,
                          shuffle=shuffle, drop_last=drop_last)

    s_train = torch.from_numpy(srt).float()
    s_mean, s_sd = s_train.mean().item(), s_train.std().item()
    s_std = (s_train - s_mean) / s_sd
    alpha = cfg['alpha']
    return dict(train=mk(xtr, ytr, srt, True, True),
                traineval=mk(xtr, ytr, srt, False, False),
                val=mk(xva, yva, sva, False, False),
                test=mk(xte, yte, ste, False, False),
                input_dim=8, task='reg', s_mean=s_mean, s_sd=s_sd,
                s_lo=torch.quantile(s_std, alpha).item(),
                s_hi=torch.quantile(s_std, 1.0 - alpha).item())


def apply_preset(cfg):
    """Fill task/sensitive_attr/batch_size/source/target from the dataset preset
    when the config leaves them unset (null)."""
    preset = PRESETS[cfg['dataset']]
    for key, val in preset.items():
        if cfg.get(key) is None:
            cfg[key] = val
    return cfg


def get_loaders(cfg):
    """FREM loaders + sensitive-attribute statistics for the kernel.

    The np seed fixes FREM's unseeded Crime shuffle (affects only val membership;
    the fold-based test split is unaffected — see notes/supervised_loss_check.md).

    S statistics (mean/std and trimmed-range quantiles) are computed on the ORIGINAL
    continuous S of the full training split; the kernel bandwidth `h` is expressed in
    units of std(S) because S is standardized with these statistics.
    """
    if cfg['dataset'] == 'SynthB':
        return _synthb_loaders(cfg)
    np.random.seed(cfg['seed'])
    (train_l, traineval_l, val_l, test_l), _, (input_dim, _) = load_data(
        cfg['seed'], cfg['source'], cfg['target'], cfg['batch_size'],
        cfg['sensitive_attr'], drop_sensitive=cfg.get('drop_sensitive', False),
        mini=cfg.get('mini', False), n_bins=0)

    s_train = traineval_l.dataset.tensors[2].float()
    s_mean = s_train.mean().item()
    s_sd = s_train.std().item()
    s_std = (s_train - s_mean) / s_sd
    alpha = cfg['alpha']
    s_lo = torch.quantile(s_std, alpha).item()
    s_hi = torch.quantile(s_std, 1.0 - alpha).item()

    return dict(train=train_l, traineval=traineval_l, val=val_l, test=test_l,
                input_dim=input_dim, task=cfg['task'],
                s_mean=s_mean, s_sd=s_sd, s_lo=s_lo, s_hi=s_hi)
