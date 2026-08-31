"""Encoder/head (FREM's MLP, imported) and the ReLU-IPM discriminator pool."""
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
FREM_SRC = os.path.abspath(os.path.join(_HERE, '..', '..', 'FRL-GDP-full', 'src'))
if FREM_SRC not in sys.path:
    sys.path.append(FREM_SRC)

from base.networks import MLP  # noqa: E402  (FREM, read-only)

REP_DIM = 50


def build_model(input_dim, task):
    """FREM tabular model (src/train.py:86-89): SELU encoder d->50->50 (50-dim
    representation, no output activation) + linear head 50->2 (cls) / 50->1 (reg)."""
    output_dim = 2 if task == 'cls' else 1
    return MLP(input_dim, hidden_dims=[50, REP_DIM], output_dim=output_dim, act='SELU')


class ReLUCritic(nn.Module):
    """ReLU-IPM discriminator pool: v_c(z) = relu(theta_c^T z + mu_c).

    Mirror of ReLUIPM-FRL_full/base/networks.py:59-99 (type='reluipm'): one shared
    Linear whose rows are the critics; freeze/melt toggle grads on fc only.
    """

    def __init__(self, input_dim, critic_num=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, critic_num)
        self.critic_num = critic_num

    def forward(self, x):
        return torch.relu(self.fc(x))

    def freeze(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def melt(self):
        for param in self.fc.parameters():
            param.requires_grad = True


def normalize_reps(z):
    """Soft projection of representations into the open unit ball B^d:

        z_tilde = z / sqrt(||z||_2^2 + 1)      (smooth; ||z_tilde||_2 < 1 always)

    z_tilde IS the representation: it feeds BOTH the prediction head and every
    ReLU-IPM computation (critic ascent, sup over s, encoder penalty, test-time
    sup), so the discriminator domain matches the paper class (relu IPM.pdf
    Eq.(3): z in B^d) and head/discriminator see the same Z. This adds one
    (parameter-free) normalization on top of FREM's architecture.
    """
    return z / torch.sqrt(z.pow(2).sum(dim=1, keepdim=True) + 1.0)


@torch.no_grad()
def project_critics(critic, mode='sphere_box'):
    """Post-step projection of the critic pool, two selectable constraint sets:

    - 'sphere_box' (default): the paper class (relu IPM.pdf p.5 Eq.(3)) —
      theta_c row-normalized onto the unit sphere S^{d-1} (as the authors
      describe on p.7) plus mu_c clamped to [-1, 1].
    - 'ball': the ORIGINAL scheme kept as an option — per-critic joint
      (theta_c, mu_c) l2-ball projection mirroring ReLUIPM-FRL_full
      trainer.py:70-80 (rescale only when the joint norm exceeds 1).

    See notes/relu_ipm_normalization_check.md.
    """
    w, b = critic.fc.weight.data, critic.fc.bias.data
    if mode == 'sphere_box':
        w.div_(w.norm(dim=1, keepdim=True).clamp_min(1e-12))
        b.clamp_(-1.0, 1.0)
    elif mode == 'ball':
        norms = torch.sqrt(w.pow(2).sum(dim=1) + b.pow(2))
        scale = torch.clamp(norms, min=1.0)
        w.div_(scale.unsqueeze(1))
        b.div_(scale)
    else:
        raise ValueError(f'unknown critic_proj mode: {mode}')
