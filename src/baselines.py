"""Baseline fairness regularizers, ported VERBATIM from FREM
(FRL-GDP-full/src/train.py) so that runs on the unified representer differ from
FREM's protocol only in the representation fed to them (z-tilde when z_norm is
on, raw z when off) — see README "Baselines".

Ports (formula-identical, device/self handled):
- gaussian_kernel_matrix   <- _Gaussian_kernel_matrix   (train.py:333-342)
- frlgdp_loss              <- _compute_frlgdp_loss      (train.py:280-309), the
  FREM/EIPM regularizer (default flags: no squared/squared_in/root_out); one
  numerical guard added — clamp before the inner sqrt (see comment in the code)
- reg_gdp_loss             <- _compute_reg_gdp_loss     (train.py:429-446)
- ADV baseline: critic = MLP(rep_dim, [50, 50], 1, 'SELU') (train.py:97-101);
  model-side fair_loss and the 5 post-update critic steps are in train.py here,
  copied sign-for-sign from FREM train.py:723-726 + 775-785 (note FREM's as-run
  sign convention: encoder minimizes the critic's MSE, critic ascends it).
"""
import os
import sys
from math import pi, sqrt

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
FREM_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'FRL-GDP-full', 'src'))
if FREM_SRC not in sys.path:
    sys.path.append(FREM_SRC)

from base.networks import MLP  # noqa: E402  (FREM, read-only)


def gaussian_kernel_matrix(Xi, Xj, sigma=1.0):
    matrix = - torch.cdist(Xi, Xj, p=2) ** 2
    matrix /= (2.0 * sigma ** 2)
    matrix = torch.exp(matrix)
    return matrix


def frlgdp_loss(reps, sensitives, gamma_rep=1.0, gamma_s=1.0):
    """FREM's EIPM estimator on the minibatch (kernel-weighted MMD between
    P(Z|S~s_k) and P(Z), averaged over anchors). NaN hazard at large lambda is
    inherited from FREM (sqrt of a signed quadratic form) — kept as-run."""
    minibatch_size = reps.size(0)
    if len(reps.size()) == 1:
        reps = reps.view(-1, 1)
    if len(sensitives.size()) == 1:
        sensitives = sensitives.view(-1, 1)
    weight_s = gaussian_kernel_matrix(sensitives, sensitives, sigma=gamma_s) \
        - torch.eye(sensitives.size(0)).to(sensitives.device)
    weight_s /= weight_s.sum(dim=0)
    weight_s -= 1 / (minibatch_size - 1)
    weight_s = weight_s.fill_diagonal_(0)
    weight_s = weight_s.float()

    kernel_rep = gaussian_kernel_matrix(reps, reps, sigma=gamma_rep)

    outer_product = torch.einsum('ij,ik->ijk', weight_s.T, weight_s.T)
    quad = torch.sum(outer_product * kernel_rep.unsqueeze(0), dim=(1, 2))
    # Sole deviation from the verbatim port: clamp before sqrt. The leave-one-out
    # weights make the per-anchor quadratic form occasionally negative; FREM-as-run
    # then produces NaN and kills the run (their repo at lmda>=5 on raw z; observed
    # at lmda>=0.3 on the normalized representer). Inactive wherever FREM-as-run
    # is finite.
    loss = torch.sqrt(torch.clamp(quad, min=1e-12)).sum()
    return loss / float(minibatch_size)


def reg_gdp_loss(probs, sensitives):
    """Reg-GDP: differentiable KDE estimate of GDP(y_hat, s) on the minibatch
    (prediction-level; expects S on its loaded raw [0,1]-ish scale)."""
    device = probs.device
    sensitives = sensitives.float().flatten()
    x_approx_interval = 1e-3
    x_approx = torch.arange(x_approx_interval, 1 - x_approx_interval, x_approx_interval).to(device)
    n = sensitives.size(0)
    d = 1
    bandwidth = torch.tensor((n * (d + 2) / 4.) ** (-1. / (d + 4))).to(device)
    x_approx_repeat = x_approx.repeat_interleave(n).reshape((-1, n))
    attention_weights = F.softmax(-(x_approx_repeat - sensitives) ** 2 / (bandwidth ** 2) / 2, dim=1).float()
    y_hat = torch.matmul(attention_weights, probs)
    y_mean = torch.mean(probs)
    pdf_values = (
        torch.exp(-((x_approx_repeat - sensitives.unsqueeze(0)) ** 2 / (bandwidth ** 2) / 2))
        ).mean(dim=-1) / sqrt(2 * pi) / bandwidth
    return torch.sum(torch.abs(y_hat - y_mean) * pdf_values) / torch.sum(pdf_values)


def build_adv_critic(rep_dim):
    """FREM's ADV discriminator: MLP(rep_dim, [50, 50], 1, 'SELU')."""
    return MLP(rep_dim, hidden_dims=[50, 50], output_dim=1, act='SELU')
