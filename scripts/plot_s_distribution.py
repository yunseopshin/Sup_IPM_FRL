"""Marginal distribution of the sensitive attribute S on Crime and Adult.

S is taken exactly as the training code sees it (`src/data.get_loaders` ->
FREM `base.datasets.load_data`, FREM protocol, seeds 2023..2027):

    Crime  S = racepctblack   (UCI value, already scaled to [0,1] upstream)
    Adult  S = age            (MinMax-scaled to [0,1] with the TRAIN split min/max)

Writes, under notebook/figures/:
    s_distribution.png / .pdf   histogram (top) + ECDF (bottom), train vs test
    s_distribution.json         every number behind the figure (also feeds the
                                HTML report)

Run through Slurm (CPU only, CLAUDE.md):
    srun --cpus-per-task=4 --partition=idea python scripts/plot_s_distribution.py
"""
import json
import os
import sys

import numpy as np
import torch
from scipy import stats
from scipy.special import softmax

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from data import apply_preset, get_loaders  # noqa: E402

SEEDS = [2023, 2024, 2025, 2026, 2027]      # FREM protocol seeds
ALPHA = 0.05                                 # trim used by sup-IPM (configs/default.yaml)
BANDWIDTH = 0.25                             # kernel h, in units of std(S)
OUT_DIR = os.path.join(_ROOT, 'notebook', 'figures')

C_TRAIN = '#2a78d6'   # categorical slot 1
C_TEST = '#eb6834'    # categorical slot 2
C_MUTED = '#898781'
C_GRID = '#e1e0d9'
C_INK = '#0b0b0b'


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_split_s(dataset, seed):
    """S of the train / val / test splits + the kernel statistics of the method."""
    cfg = apply_preset(dict(dataset=dataset, seed=seed, alpha=ALPHA, mini=False,
                            drop_sensitive=False, source=None, target=None,
                            task=None, sensitive_attr=None, batch_size=None))
    d = get_loaders(cfg)
    out = {split: d[split].dataset.tensors[2].numpy().astype(float)
           for split in ('traineval', 'val', 'test')}
    out['y_train'] = d['traineval'].dataset.tensors[1].numpy().astype(float)
    out['stats'] = dict(s_mean=d['s_mean'], s_sd=d['s_sd'],
                        s_lo=d['s_lo'], s_hi=d['s_hi'])
    out['batch_size'] = cfg['batch_size']
    return out


def adult_age_years():
    """Raw ages (years) of the Adult rows, in the order the loader sees them."""
    from aif360.datasets import AdultDataset
    raw = AdultDataset(
        protected_attribute_names=['sex'],
        categorical_features=['workclass', 'education', 'marital-status',
                              'occupation', 'relationship', 'native-country', 'race'],
        privileged_classes=[['Male']],
        metadata={'label_map': {1.0: '>50K', 0.0: '<=50K'},
                  'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]})
    return raw.features[:, 0].astype(float)


def adult_train_age_range(seed, ages):
    """Train-split min/max of raw age -- the MinMax scaler the loader fits."""
    np.random.seed(seed)
    train_ids = np.random.permutation(ages.shape[0])[:32561]
    tr = ages[train_ids]
    return float(tr.min()), float(tr.max())


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
QS = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]


def describe(x):
    vals, counts = np.unique(x, return_counts=True)
    return dict(
        n=int(x.size),
        mean=float(x.mean()), sd=float(x.std(ddof=1)),
        skew=float(stats.skew(x)), kurtosis=float(stats.kurtosis(x)),
        quantiles={f'q{q}': float(np.percentile(x, q)) for q in QS},
        n_unique=int(vals.size),
        top_atom_value=float(vals[counts.argmax()]),
        top_atom_frac=float(counts.max() / x.size),
        frac_below_0p05=float((x < 0.05).mean()),
        frac_above_0p50=float((x > 0.50).mean()),
    )


def effective_n(grid_std, s_std, h):
    """Kish effective sample size of the self-normalized Gaussian kernel weights,
    1 / sum_i w_i(s)^2 -- the number of samples the conditional IPM at s really
    averages over (variance of the estimate ~ 1 / n_eff)."""
    w = softmax(-((grid_std[:, None] - s_std[None, :]) ** 2) / (2.0 * h * h), axis=1)
    return 1.0 / (w ** 2).sum(axis=1)


def ecdf_on_grid(x, grid):
    return np.searchsorted(np.sort(x), grid, side='right') / x.size


# --------------------------------------------------------------------------- #
def build_report():
    report = {'alpha': ALPHA, 'bandwidth_in_sd': BANDWIDTH, 'seeds': SEEDS,
              'datasets': {}}

    ages = adult_age_years()

    for dataset in ('Crime', 'Adult'):
        per_seed = {}
        for seed in SEEDS:
            per_seed[seed] = load_split_s(dataset, seed)
        ref = per_seed[SEEDS[0]]                       # seed 2023 = the reference draw
        s_tr, s_te = ref['traineval'], ref['test']
        st = ref['stats']

        grid = np.linspace(0.0, 1.0, 401)
        grid_std = (grid - st['s_mean']) / st['s_sd']
        s_tr_std = (s_tr - st['s_mean']) / st['s_sd']
        neff_full = effective_n(grid_std, s_tr_std, BANDWIDTH)
        # inside the trimmed range the sup over s* is taken
        trim = (grid_std >= st['s_lo']) & (grid_std <= st['s_hi'])
        # per-minibatch equivalent (the training penalty sees B samples, not n)
        scale = ref['batch_size'] / s_tr.size

        entry = dict(
            sensitive_attr={'Crime': 'racepctblack', 'Adult': 'age'}[dataset],
            batch_size=ref['batch_size'],
            kernel=dict(s_mean=st['s_mean'], s_sd=st['s_sd'],
                        trim_std=[st['s_lo'], st['s_hi']],
                        trim_raw=[st['s_mean'] + st['s_lo'] * st['s_sd'],
                                  st['s_mean'] + st['s_hi'] * st['s_sd']],
                        h_raw=BANDWIDTH * st['s_sd']),
            splits={k: describe(per_seed[SEEDS[0]][k]) for k in ('traineval', 'val', 'test')},
            corr_s_y=float(np.corrcoef(s_tr, ref['y_train'])[0, 1]),
            across_seeds={
                'train_mean': [float(per_seed[s]['traineval'].mean()) for s in SEEDS],
                'train_sd': [float(per_seed[s]['traineval'].std(ddof=1)) for s in SEEDS],
                'test_mean': [float(per_seed[s]['test'].mean()) for s in SEEDS],
                'test_n': [int(per_seed[s]['test'].size) for s in SEEDS],
            },
            n_eff=dict(
                grid=grid[trim].tolist(),
                full_train=neff_full[trim].tolist(),
                per_batch=(neff_full[trim] * scale).tolist(),
                min_per_batch=float((neff_full[trim] * scale).min()),
                median_per_batch=float(np.median(neff_full[trim] * scale)),
                at_trim_lo_per_batch=float((neff_full[trim] * scale)[0]),
                at_trim_hi_per_batch=float((neff_full[trim] * scale)[-1]),
            ),
        )

        if dataset == 'Adult':
            lo, hi = adult_train_age_range(SEEDS[0], ages)
            entry['age_years'] = dict(train_min=lo, train_max=hi)
            # the loader's MinMax scaling must reproduce from the raw ages exactly
            np.random.seed(SEEDS[0])
            replica = (ages[np.random.permutation(ages.shape[0])[:32561]] - lo) / (hi - lo)
            assert np.allclose(replica, s_tr), 'Adult age scaling does not replicate'
            # one bin per year of age, integer ages sitting at bin centres
            bins = (np.arange(int(lo), int(hi) + 2) - 0.5 - lo) / (hi - lo)
        else:
            bins = np.linspace(0.0, 1.0, 51)

        h_tr, _ = np.histogram(s_tr, bins=bins)
        h_te, _ = np.histogram(s_te, bins=bins)
        entry['hist'] = dict(edges=bins.tolist(),
                             train=h_tr.tolist(), test=h_te.tolist())
        entry['ecdf'] = dict(grid=grid.tolist(),
                             train=ecdf_on_grid(s_tr, grid).tolist(),
                             test=ecdf_on_grid(s_te, grid).tolist())
        report['datasets'][dataset] = entry

    return report


def draw_figure(report):
    """Everything here reads `report` only, so `--from-json` re-plots instantly."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6), sharex='col',
                             gridspec_kw=dict(height_ratios=[2.0, 1.0], hspace=0.14,
                                              wspace=0.20))
    titles = {'Crime': 'Crime  ·  S = racepctblack  (bins of 0.02)',
              'Adult': 'Adult  ·  S = age  (one bin per year)'}
    for col, dataset in enumerate(('Crime', 'Adult')):
        e = report['datasets'][dataset]
        lo_raw, hi_raw = e['kernel']['trim_raw']
        edges = np.asarray(e['hist']['edges'])
        grid = np.asarray(e['ecdf']['grid'])

        ax = axes[0, col]
        for v0, v1 in ((0.0, lo_raw), (hi_raw, 1.0)):        # trimmed-away tails
            ax.axvspan(v0, v1, color='#f0efec', zorder=0, linewidth=0)
        for split, c, lab in (('train', C_TRAIN, 'train'), ('test', C_TEST, 'test')):
            n = e['splits']['traineval' if split == 'train' else 'test']['n']
            pct = 100.0 * np.asarray(e['hist'][split]) / n   # share of the split, %
            ax.stairs(pct, edges, fill=True, color=c, alpha=0.28, linewidth=0)
            ax.stairs(pct, edges, color=c, linewidth=1.6, label=f'{lab}  (n = {n:,})')
        for v in (lo_raw, hi_raw):
            ax.axvline(v, color=C_MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color=C_MUTED, linestyle=(0, (4, 3)), linewidth=1.2))
        labels.append(r'trim $[q_{.05},\ q_{.95}]$ — range of $s^*$')
        ax.legend(handles, labels, frameon=False, fontsize=9, loc='upper right',
                  handlelength=1.8, borderaxespad=0.2)
        ax.set_ylabel('share of split (%)', fontsize=9.5, color='#52514e')
        ax.set_title(titles[dataset], fontsize=11, color=C_INK, pad=30)

        if dataset == 'Adult':
            lo, hi = e['age_years']['train_min'], e['age_years']['train_max']
            sec = ax.secondary_xaxis('top', functions=(lambda v: lo + v * (hi - lo),
                                                       lambda a: (a - lo) / (hi - lo)))
            sec.set_xlabel('age (years)', fontsize=8.5, color=C_MUTED, labelpad=4)
            sec.tick_params(labelsize=8.5, colors=C_MUTED)
            sec.spines['top'].set_color('#c3c2b7')

        ax2 = axes[1, col]
        for v0, v1 in ((0.0, lo_raw), (hi_raw, 1.0)):
            ax2.axvspan(v0, v1, color='#f0efec', zorder=0, linewidth=0)
        ax2.plot(grid, e['ecdf']['train'], color=C_TRAIN, linewidth=1.8)
        ax2.plot(grid, e['ecdf']['test'], color=C_TEST, linewidth=1.8)
        for v in (lo_raw, hi_raw):
            ax2.axvline(v, color=C_MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel('ECDF', fontsize=9.5, color='#52514e')
        ax2.set_xlabel('S   (min-max scaled to [0, 1])', fontsize=9.5, color='#52514e')

        for a in (ax, ax2):
            a.set_xlim(0, 1)
            a.grid(True, color=C_GRID, linewidth=0.7)
            a.set_axisbelow(True)
            for side in ('top', 'right'):
                a.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                a.spines[side].set_color('#c3c2b7')
            a.tick_params(labelsize=9, colors='#52514e')

    fig.suptitle('Sensitive attribute S — marginal distribution (FREM protocol, seed 2023)',
                 fontsize=12.5, color=C_INK, y=0.975)
    fig.subplots_adjust(top=0.845, bottom=0.095, left=0.075, right=0.985)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f's_distribution.{ext}'), dpi=200,
                    facecolor='white')
    print(f'[saved] {OUT_DIR}/s_distribution.png|.pdf|.json')


def print_tables(report):
    for dataset, e in report['datasets'].items():
        print(f"\n===== {dataset}  (S = {e['sensitive_attr']}, batch {e['batch_size']}) =====")
        print(f"  mean {e['kernel']['s_mean']:.4f}   sd {e['kernel']['s_sd']:.4f}   "
              f"trim(raw) [{e['kernel']['trim_raw'][0]:.4f}, {e['kernel']['trim_raw'][1]:.4f}]   "
              f"h_raw {e['kernel']['h_raw']:.4f}")
        print(f"  corr(S, Y) = {e['corr_s_y']:+.3f}")
        hdr = f"  {'split':10s} {'n':>7s} {'mean':>7s} {'sd':>7s} {'skew':>7s}" + \
              ''.join(f"{q:>8s}" for q in ('q1', 'q5', 'q25', 'q50', 'q75', 'q95', 'q99', 'max'))
        print(hdr)
        for split, name in (('traineval', 'train'), ('val', 'val'), ('test', 'test')):
            d = e['splits'][split]
            row = f"  {name:10s} {d['n']:7d} {d['mean']:7.3f} {d['sd']:7.3f} {d['skew']:7.2f}"
            row += ''.join(f"{d['quantiles'][q]:8.3f}" for q in
                           ('q1', 'q5', 'q25', 'q50', 'q75', 'q95', 'q99', 'q100'))
            print(row)
        d = e['splits']['traineval']
        print(f"  unique values {d['n_unique']}   largest atom: S={d['top_atom_value']:.4f} "
              f"({100 * d['top_atom_frac']:.1f}% of train)   "
              f"P(S<0.05)={100 * d['frac_below_0p05']:.1f}%   "
              f"P(S>0.50)={100 * d['frac_above_0p50']:.1f}%")
        ne = e['n_eff']
        print(f"  kernel n_eff per minibatch over the trimmed range: "
              f"min {ne['min_per_batch']:.1f}, median {ne['median_per_batch']:.1f} "
              f"(of B={e['batch_size']}); at trim ends "
              f"{ne['at_trim_lo_per_batch']:.1f} / {ne['at_trim_hi_per_batch']:.1f}")
        acr = e['across_seeds']
        print(f"  across seeds 2023-2027: train mean "
              f"[{min(acr['train_mean']):.3f}, {max(acr['train_mean']):.3f}], "
              f"test mean [{min(acr['test_mean']):.3f}, {max(acr['test_mean']):.3f}], "
              f"test n {acr['test_n']}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 's_distribution.json')
    if '--from-json' in sys.argv:          # re-plot only, no dataset loading
        with open(path) as f:
            report = json.load(f)
    else:
        report = build_report()
        with open(path, 'w') as f:
            json.dump(report, f, indent=1)
    draw_figure(report)
    print_tables(report)


if __name__ == '__main__':
    main()
