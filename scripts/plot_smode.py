"""Figures for the s_mode comparison (PERCRITIC_S.md sec.5).

  fig_smode_frontier_adult.png  accuracy vs test sup_ipm frontier on Adult
  fig_smode_critic_gap.png      training IPM vs eval-strength sup_ipm (weak-critic gap)
  fig_smode_traj_adult.png      training ipm_s_star + s*_c spread over epochs

Reads the same run trees as scripts/aggregate_smode_adult.py. Outputs into
results/smode_compare/figs/.

Usage: python scripts/plot_smode.py
"""
import csv
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'results', 'smode_compare', 'figs')
SEEDS = [2023, 2024, 2025, 2026, 2027]

# categorical slots 1-3 + neutrals (dataviz reference palette, light mode);
# colour follows the ENTITY and is kept identical across all three figures
C_PERCRITIC = '#2a78d6'   # slot 1 blue  - ours
C_CS20 = '#eb6834'        # slot 2 orange - shared, 10x critic budget
C_SHARED2 = '#1baf7a'     # slot 3 aqua   - shared, same critic budget
C_MUTED = '#8a8a85'       # neutral - the intermediate ladder rungs
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#dcdcd8'
SURFACE = '#fcfcfb'


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def load_cell(base, alg):
    """{lmda: dict(acc/sup_ipm/gdp mean+sd, n, collapsed, train_ipm, s_std)}"""
    out = {}
    d = os.path.join(base, alg)
    if not os.path.isdir(d):
        return out
    for lm in sorted(os.listdir(d)):
        if not lm.startswith('lmda_f-'):
            continue
        lam = float(lm.split('lmda_f-')[1])
        ok, collapsed, seeds_ok = [], 0, []
        for s in SEEDS:
            p = os.path.join(d, lm, f'seed-{s}', 'results.json')
            if not os.path.exists(p):
                continue
            r = json.load(open(p))['test']
            if not (finite(r.get('acc')) and finite(r.get('gdp_w_kernel'))):
                continue
            if r.get('bacc') is not None and r['bacc'] <= 0.5001:
                collapsed += 1
            else:
                ok.append(r)
                seeds_ok.append(s)
        if not ok:
            continue
        entry = dict(n=len(ok), collapsed=collapsed)
        for m in ('acc', 'sup_ipm', 'gdp_w_kernel'):
            v = [r[m] for r in ok if finite(r.get(m))]
            entry[m] = float(np.mean(v))
            entry[m + '_sd'] = float(np.std(v))
        entry['train_ipm'] = train_tail(d, lm, seeds_ok, 'ipm_s_star')
        entry['s_std'] = train_tail(d, lm, seeds_ok, 's_star_std', n_last=200)
        out[lam] = entry
    return out


def train_tail(alg_dir, lm, seeds, col, n_last=10):
    vals = []
    for s in seeds:
        p = os.path.join(alg_dir, lm, f'seed-{s}', 'train_log.csv')
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p)))
        if not rows or col not in rows[0]:
            continue
        vals.append(np.mean([float(r[col]) for r in rows[-n_last:]]))
    return float(np.mean(vals)) if vals else float('nan')


def traj(alg_dir, lm, col):
    """Per-epoch mean over seeds of a train_log column."""
    curves = []
    for s in SEEDS:
        p = os.path.join(alg_dir, lm, f'seed-{s}', 'train_log.csv')
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p)))
        if not rows or col not in rows[0]:
            continue
        curves.append([float(r[col]) for r in rows])
    if not curves:
        return None, None
    n = min(len(c) for c in curves)
    arr = np.array([c[:n] for c in curves])
    return np.arange(1, n + 1), arr.mean(axis=0)


LADDER = os.path.join(ROOT, 'results', 'critic_ladder')
CELLS = {
    'percritic cs2': (os.path.join(LADDER, 'cs2', 'Adult-age'), 'supipm-percritic'),
    'shared cs2': (os.path.join(LADDER, 'cs2', 'Adult-age'), 'supipm'),
    'shared cs5': (os.path.join(LADDER, 'cs5', 'Adult-age'), 'supipm'),
    'shared cs10': (os.path.join(LADDER, 'cs10', 'Adult-age'), 'supipm'),
    'shared cs20': (os.path.join(ROOT, 'results', 'critic20', 'Adult-age'), 'supipm'),
    'shared cs2 (lr 1e-3)': (os.path.join(ROOT, 'results', 'Adult-age'), 'supipm'),
}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)
    ax.grid(True, color=GRID, lw=0.8, ls='-')
    ax.set_axisbelow(True)


def fig_frontier(data):
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=200, facecolor=SURFACE)
    style(ax)
    series = [
        ('percritic, critic_step=2 (ours)', 'percritic cs2', C_PERCRITIC, 'o'),
        ('shared, critic_step=20', 'shared cs20', C_CS20, 's'),
        ('shared, critic_step=2 (critic_lr 1e-3)', 'shared cs2 (lr 1e-3)', C_SHARED2, '^'),
    ]
    for label, key, color, marker in series:
        cell = data[key]
        lams = sorted(cell)
        x = [cell[l]['acc'] for l in lams]
        y = [cell[l]['sup_ipm'] for l in lams]
        ax.errorbar(x, y, xerr=[cell[l]['acc_sd'] for l in lams],
                    yerr=[cell[l]['sup_ipm_sd'] for l in lams], fmt='none',
                    ecolor=color, elinewidth=1.0, alpha=0.4, capsize=0, zorder=2)
        ax.plot(x, y, color=color, lw=2.0, zorder=3, solid_capstyle='round')
        # hollow marker = fewer than 5 healthy seeds (collapsed runs excluded)
        for l in lams:
            full = cell[l]['n'] == len(SEEDS)
            ax.plot([cell[l]['acc']], [cell[l]['sup_ipm']], marker=marker, ms=8, mew=2,
                    mec=SURFACE if full else color,
                    color=color if full else SURFACE, zorder=4)
        ax.plot([], [], color=color, lw=2.0, marker=marker, ms=8, mew=2, mec=SURFACE,
                label=label)

    # same critic budget AND same critic_lr as the two curves above: the direct control
    c = data['shared cs2']
    for lam in sorted(c):
        ax.plot([c[lam]['acc']], [c[lam]['sup_ipm']], marker='X', ms=10, mew=1.5,
                mec=SURFACE, color=C_SHARED2, zorder=4)
    ax.plot([], [], marker='X', ls='none', ms=10, mew=1.5, mec=SURFACE, color=C_SHARED2,
            label='shared, critic_step=2 (critic_lr 0.01)')

    # selective direct labels: only the three lambda=1 points the comparison turns on
    for key, off, ha in (('percritic cs2', (-14, 26), 'center'),
                         ('shared cs20', (34, -14), 'center'),
                         ('shared cs2 (lr 1e-3)', (-6, 30), 'center')):
        e = data[key][1.0]
        ax.annotate(f"λ=1\n{e['sup_ipm']:.3f}", (e['acc'], e['sup_ipm']),
                    textcoords='offset points', xytext=off, ha=ha, fontsize=9,
                    color=INK2, linespacing=1.3, zorder=5,
                    arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8,
                                    shrinkA=2, shrinkB=6, alpha=0.6))

    ax.set_yscale('log')
    ax.set_xlim(0.8175, 0.8455)
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.4])
    ax.set_yticklabels(['0.02', '0.05', '0.10', '0.20', '0.40'])
    ax.set_xlabel('test accuracy  (higher is better) →', color=INK2, fontsize=10)
    ax.set_ylabel('← test sup$_s$ ÎPM  (lower is fairer, log scale)', color=INK2, fontsize=10)
    ax.set_title('Adult: per-critic $s^*$ recovers most of the critic_step 2 → 20 gap',
                 color=INK, fontsize=12.5, loc='left', pad=50)
    ax.text(0, 1.012,
            'each point = one λ, mean ± sd over 5 seeds; hollow marker = 2 of 5 seeds collapsed.\n'
            'At equal λ percritic cs2 ties cs20; matched on accuracy, cs20 is still slightly ahead.\n'
            'shared cs2 (critic_lr 1e-3) continues off-scale to acc 0.784 at λ=2.',
            transform=ax.transAxes, color=INK2, fontsize=8.5, linespacing=1.5, va='bottom')
    leg = ax.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK2,
                    borderaxespad=0.8, handlelength=2.4)
    leg.set_zorder(6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_smode_frontier_adult.png'),
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def fig_critic_gap(data):
    """How badly the critic the encoder trains against is fooled, at lambda=1."""
    order = [('shared cs2 (lr 1e-3)', 'shared cs2\n(critic_lr 1e-3)', C_SHARED2),
             ('shared cs2', 'shared cs2', C_SHARED2),
             ('shared cs5', 'shared cs5', C_MUTED),
             ('shared cs10', 'shared cs10', C_MUTED),
             ('shared cs20', 'shared cs20', C_CS20),
             ('percritic cs2', 'percritic cs2\n(ours)', C_PERCRITIC)]
    labels, ratios, colors, pairs = [], [], [], []
    for key, label, color in order:
        e = data[key].get(1.0)
        if e is None or not finite(e['train_ipm']):
            continue
        labels.append(label)
        ratios.append(e['sup_ipm'] / e['train_ipm'])
        colors.append(color)
        pairs.append((e['train_ipm'], e['sup_ipm']))

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200, facecolor=SURFACE)
    style(ax)
    ax.grid(axis='y', visible=False)
    y = np.arange(len(labels))
    ax.barh(y, ratios, height=0.46, color=colors, zorder=3)
    ax.axvline(1.0, color=INK2, lw=1.2, zorder=4)
    ax.annotate('honest critic (1.0×)', (1.0, -0.72), textcoords='offset points',
                xytext=(6, 0), ha='left', va='center', fontsize=9, color=INK2,
                annotation_clip=False)
    for i, (r, (tr, te)) in enumerate(zip(ratios, pairs)):
        ax.text(r + 0.12, i, f'{r:.1f}×   (trains on {tr:.3f}, actually {te:.3f})',
                va='center', fontsize=9, color=INK2)
    ax.set_yticks(y, labels, fontsize=9, color=INK2)
    ax.set_xlim(0, max(ratios) * 1.55)
    ax.set_xlabel('eval-strength sup$_s$ ÎPM  ÷  IPM the training critic reported',
                  color=INK2, fontsize=10)
    ax.set_ylim(-1.1, len(labels) - 0.4)
    ax.set_title('Adult, λ=1: how far the training critic is fooled',
                 color=INK, fontsize=12.5, loc='left', pad=30)
    ax.text(0, 1.015,
            'ratio 1.0 = the critic the encoder trained against was as strong as the auditor',
            transform=ax.transAxes, color=INK2, fontsize=9, va='bottom')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_smode_critic_gap.png'),
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def fig_traj():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=200, facecolor=SURFACE)
    for ax in axes:
        style(ax)

    lm = 'lmda_f-1.0'
    curves = [('percritic cs2 (ours)', os.path.join(LADDER, 'cs2', 'Adult-age', 'supipm-percritic'), C_PERCRITIC),
              ('shared cs20', os.path.join(ROOT, 'results', 'critic20', 'Adult-age', 'supipm'), C_CS20),
              ('shared cs2', os.path.join(LADDER, 'cs2', 'Adult-age', 'supipm'), C_SHARED2)]
    for label, d, color in curves:
        ep, v = traj(d, lm, 'ipm_s_star')
        if ep is None:
            continue
        axes[0].plot(ep, v, color=color, lw=1.5, label=label, solid_capstyle='round')
    axes[0].annotate('shared cs2: the critic is out-run —\nthe IPM it reports falls, while the\n'
                     'auditor still finds 0.13 in the model',
                     (150, 0.0185), textcoords='offset points', xytext=(-52, -34),
                     ha='center', fontsize=8.5, color=INK2, linespacing=1.4,
                     arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8, alpha=0.6,
                                     shrinkA=2, shrinkB=4))
    axes[0].set_xlabel('epoch', color=INK2, fontsize=10)
    axes[0].set_ylabel('training ÎPM at $s^*$ (pool value)', color=INK2, fontsize=10)
    axes[0].set_title('What the training critic reports', color=INK, fontsize=11, loc='left',
                      pad=26)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=3, columnspacing=1.2,
                   handlelength=1.6, loc='lower right', bbox_to_anchor=(1.0, 1.005))

    ep, v = traj(os.path.join(LADDER, 'cs2', 'Adult-age', 'supipm-percritic'), lm, 's_star_std')
    if ep is not None:
        axes[1].plot(ep, v, color=C_PERCRITIC, lw=1.5, label='percritic (Adult, λ=1)',
                     solid_capstyle='round')
    ep2, v2 = traj(os.path.join(ROOT, 'results', 'smode_compare', 'cs2',
                                'Crime-racepctblack', 'supipm-percritic'), lm, 's_star_std')
    if ep2 is not None:
        axes[1].plot(ep2, v2, color=C_PERCRITIC, lw=1.5, ls=(0, (4, 3)),
                     label='percritic (Crime, λ=1)')
    axes[1].axhline(0.0, color=C_SHARED2, lw=2.0)
    axes[1].text(len(ep2 if ep2 is not None else ep) * 0.5, 0.02,
                 'shared: one $s^*$ for the whole pool → 0 by construction',
                 color=INK2, fontsize=9)
    axes[1].set_xlabel('epoch', color=INK2, fontsize=10)
    axes[1].set_ylabel('sd of $s^*_c$ across the pool  (units of sd(S))',
                       color=INK2, fontsize=10)
    axes[1].set_title('Where the critics sit along $s$', color=INK, fontsize=11, loc='left',
                      pad=26)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=2, columnspacing=1.2,
                   handlelength=1.6, loc='lower right', bbox_to_anchor=(1.0, 1.005))

    fig.suptitle('The weak-critic equilibrium, live (mean over 5 seeds)',
                 color=INK, fontsize=12.5, x=0.006, ha='left', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, 'fig_smode_traj_adult.png'),
                facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {k: load_cell(b, a) for k, (b, a) in CELLS.items()}
    fig_frontier(data)
    fig_critic_gap(data)
    fig_traj()
    print('wrote:', ', '.join(sorted(os.listdir(OUT))))


if __name__ == '__main__':
    main()
