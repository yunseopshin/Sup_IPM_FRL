"""Adult fairness-accuracy tradeoff, s_mode version: critic 2-step vs 20-step vs percritic.

Same figure style as scripts/aggregate_adult_compare.py's tradeoff_acc_supipm.png
(x = test sup_s IPM, y = test accuracy, one point per lambda, mean +/- sd over the
healthy seeds, lambda annotated). Collapsed seeds (test bacc <= 0.5001) are excluded
from the stats, as there.

Mean training wall-clock per run (lambda > 0 only) is measured from train_log.csv
and printed, and shown in the legend.

Usage: python scripts/plot_smode_tradeoff.py
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

SERIES = [
    ('supIPM critic 2-step', os.path.join(ROOT, 'results', 'Adult-age', 'supipm'), 'tab:blue'),
    ('supIPM critic 20-step', os.path.join(ROOT, 'results', 'critic20', 'Adult-age', 'supipm'), 'tab:green'),
    ('supIPM per-critic $s^*$, 2-step', os.path.join(ROOT, 'results', 'critic_ladder', 'cs2',
                                                     'Adult-age', 'supipm-percritic'), 'tab:orange'),
]
# FREM baseline: the val-selected gamma_s curve of scripts/aggregate_adult_compare.py
FREM_SELECTED = os.path.join(ROOT, 'results', 'Adult-age', 'compare', 'frem_selected.csv')


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


METRICS = [('sup_ipm', 'tradeoff_acc_supipm_smode.png', 'test sup$_s$ ÎPM'),
           ('gdp_w_kernel', 'tradeoff_acc_gdp_smode.png', 'test ΔGDP (kernel)')]


def collect(alg_dir, fair_m):
    """[(lmda, acc, acc_sd, fair, fair_sd, n)] + mean seconds/run over lambda > 0."""
    points, secs = [], []
    if not os.path.isdir(alg_dir):
        return points, float('nan')
    for lm in sorted(os.listdir(alg_dir)):
        if not lm.startswith('lmda_f-'):
            continue
        lam = float(lm.split('lmda_f-')[1])
        ok = []
        for s in SEEDS:
            rd = os.path.join(alg_dir, lm, f'seed-{s}')
            rj = os.path.join(rd, 'results.json')
            if not os.path.exists(rj):
                continue
            r = json.load(open(rj))['test']
            if not (finite(r.get('acc')) and finite(r.get(fair_m))):
                continue
            if r.get('bacc') is not None and r['bacc'] <= 0.5001:
                continue  # constant-predictor collapse
            ok.append(r)
            log = os.path.join(rd, 'train_log.csv')
            if lam > 0 and os.path.exists(log):
                secs.append(sum(float(row['sec']) for row in csv.DictReader(open(log))))
        if not ok:
            continue
        acc = [r['acc'] for r in ok]
        fair = [r[fair_m] for r in ok]
        points.append((lam, np.mean(acc), np.std(acc), np.mean(fair), np.std(fair), len(ok)))
    points.sort()
    return points, (float(np.mean(secs)) if secs else float('nan'))


def collect_frem(fair_m):
    """The val-selected FREM curve (one gamma_s per lambda), read from the
    aggregate_adult_compare.py selection so the baseline is identical to the
    published Adult comparison figure."""
    points, secs = [], []
    if not os.path.exists(FREM_SELECTED):
        return points, float('nan')
    for r in csv.DictReader(open(FREM_SELECTED)):
        lam = float(r['lmda_f'])
        points.append((lam, float(r['test_acc_mean']), float(r['test_acc_std']),
                       float(r[f'test_{fair_m}_mean']), float(r[f'test_{fair_m}_std']),
                       int(r['n_seeds'])))
        if lam <= 0:
            continue
        for s in SEEDS:
            log = os.path.join(ROOT, 'results', 'Adult-age', r['alg_tag'],
                               f"lmda_f-{r['lmda_f']}", f'seed-{s}', 'train_log.csv')
            if os.path.exists(log):
                secs.append(sum(float(row['sec']) for row in csv.DictReader(open(log))))
    points.sort()
    return points, (float(np.mean(secs)) if secs else float('nan'))


def draw(ax, pts, secs, label, color):
    lams = [p[0] for p in pts]
    acc, acc_sd = [p[1] for p in pts], [p[2] for p in pts]
    fair, fair_sd = [p[3] for p in pts], [p[4] for p in pts]
    ax.errorbar(fair, acc, xerr=fair_sd, yerr=acc_sd, fmt='o-', ms=4, lw=1.2, capsize=2,
                color=color, label=f'{label}  ({secs / 60.0:.0f} min/run)')
    for lam, x, y in zip(lams, fair, acc):
        ax.annotate(f'{lam:g}', (x, y), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')


def main():
    os.makedirs(OUT, exist_ok=True)
    printed = False
    for fair_m, fname, xlabel in METRICS:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        if not printed:
            print(f"{'series':<34} {'runs':>5} {'min/run':>9}")
        for label, alg_dir, color in SERIES:
            pts, secs = collect(alg_dir, fair_m)
            if not pts:
                print(f'{label:<34} (no runs found at {alg_dir})')
                continue
            draw(ax, pts, secs, label, color)
            if not printed:
                print(f'{label:<34} {sum(p[5] for p in pts):>5} {secs / 60.0:>9.1f}')

        pts, secs = collect_frem(fair_m)
        if pts:
            draw(ax, pts, secs, 'FREM (val-selected γ_s)', 'tab:red')
            if not printed:
                print(f'{"FREM (val-selected gamma_s)":<34} {sum(p[5] for p in pts):>5} '
                      f'{secs / 60.0:>9.1f}')
        printed = True

        ax.set_xlabel(xlabel)
        ax.set_ylabel('test accuracy')
        ax.set_title('Adult (S=age): fairness-accuracy tradeoff')
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(OUT, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print('wrote', path)


if __name__ == '__main__':
    main()
