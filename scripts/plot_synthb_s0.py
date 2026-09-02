"""Section C of SYNTHB_INTERIOR.md: what the trim hides, as a function of s0.

For each SynthB run family (one true bump centre s0 per family, lambda = 0) this
reads the full-range diagnostic npz written by `plot_ipm_curve_full.py compute`
and compares three heights on the SAME curve:

    peak(s0)   the observed IPM at the TRUE bump — the violation that exists
    sup_trim   the maximum over the alpha-trimmed range — what the protocol reports
    floor(s0)  the permutation null at the true bump — the estimator's noise there

The spec asks for (sup_full - sup_trim) / sup_full. That ratio turns out to be
uninformative here: sup_full always sits on the extreme order statistic of S,
where n_eff -> 1, so it measures boundary noise rather than the bump (see the
lambda = 0 no-trim runs, where s* jumps to +-3.5..4.2 sd). Both are printed, but
the figure uses peak(s0), which is the quantity the trim can actually hide.

    python scripts/plot_synthb_s0.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
OUT_DIR = os.path.join(_ROOT, 'notebook', 'figures')
SEEDS = [2023, 2024, 2025]

C_PEAK = '#2a78d6'      # observed height at the true bump
C_TRIM = '#eb6834'      # what the trimmed protocol reports
C_SUB = '#52514e'
C_MUTED = '#898781'
C_GRID = '#e1e0d9'


def local_peak(curve, x, centre, half=0.04):
    """Height of the curve near `centre` (a window, so a small shift in the
    representation-level bump does not read as a missing peak)."""
    win = np.abs(x - centre) <= half
    return float(curve[win].max()) if win.any() else float('nan')


def collect():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'ipm_curve_full_SynthB_s0-*.npz'))):
        s0_nominal = float(re.search(r's0-([0-9.]+)\.npz$', path).group(1))
        z = np.load(path)
        x = z['grid_raw']
        lmda = str(z['lmdas'][0])
        m, sd = (float(v) for v in z['s_stats'].mean(axis=0))
        lo, hi = z['trim_raw'].mean(axis=0)
        obs, nul = z[f'test_l{lmda}_obs'], z[f'test_l{lmda}_null']
        bnul = z[f'batch_l{lmda}_null']
        n_seed = obs.shape[0]
        s0_raw = z['s0'][:, 0]                      # per-seed, in the [0,1] scale
        per_peak, per_trim, per_floor, per_bfloor, per_full, per_arg = [], [], [], [], [], []
        for i in range(n_seed):
            ins = (x >= z['trim_raw'][i, 0]) & (x <= z['trim_raw'][i, 1])
            per_peak.append(local_peak(obs[i], x, s0_raw[i]))
            per_trim.append(float(obs[i][ins].max()))
            per_full.append(float(obs[i].max()))
            per_arg.append(float((x[obs[i].argmax()] - m) / sd))   # sd units, per seed
            j = int(np.abs(x - s0_raw[i]).argmin())
            per_floor.append(float(nul.reshape(n_seed, -1, x.size)[i].mean(axis=0)[j])
                             if nul.shape[0] != n_seed else float(nul[i][j]))
            per_bfloor.append(float(bnul.reshape(n_seed, -1, x.size)[i].mean(axis=0)[j])
                              if bnul.shape[0] != n_seed else float(bnul[i][j]))
        mse = []
        for seed in SEEDS:
            f = os.path.join(_ROOT, 'results', 'synthB-s0', f's0-{s0_nominal}',
                             'SynthB-synth_s', 'supipm', f'lmda_f-{lmda}',
                             f'seed-{seed}', 'results.json')
            if os.path.exists(f):
                with open(f) as fh:
                    mse.append(json.load(fh)['test']['mse'])
        rows.append(dict(
            s0=s0_nominal, s0_raw=float(s0_raw.mean()),
            trim_std=[(lo - m) / sd, (hi - m) / sd],
            peak=float(np.mean(per_peak)), peak_sd=float(np.std(per_peak)),
            sup_trim=float(np.mean(per_trim)), sup_full=float(np.mean(per_full)),
            # the per-seed full-range argmax lands on either end, so an average of
            # the positions is meaningless — keep them all
            argmax_full_std=[round(v, 2) for v in per_arg],
            floor=float(np.mean(per_floor)), bfloor=float(np.mean(per_bfloor)),
            mse=float(np.mean(mse)) if mse else float('nan'),
            n_seed=n_seed))
    return rows


def main():
    rows = collect()
    if not rows:
        raise SystemExit('no ipm_curve_full_SynthB_s0-*.npz found')
    s0 = np.array([r['s0'] for r in rows])
    peak = np.array([r['peak'] for r in rows])
    trim = np.array([r['sup_trim'] for r in rows])
    floor = np.array([r['floor'] for r in rows])
    bfloor = np.array([r['bfloor'] for r in rows])
    edge = rows[0]['trim_std'][1]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.0, 0.62], hspace=0.12))
    ax = axes[0]
    ax.axvspan(edge, max(s0.max() + 0.3, edge + 0.1), color='#f0efec', zorder=0,
               linewidth=0)
    ax.axvline(edge, color=C_MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.plot(s0, peak, 'o-', color=C_PEAK, linewidth=2, markersize=6,
            label=r'true violation:  observed $\widehat{\mathrm{IPM}}$ at $s_0$')
    ax.plot(s0, trim, 's--', color=C_TRIM, linewidth=2, markersize=6,
            label=r'what the trimmed protocol reports:  $\sup$ over $[q_{.05}, q_{.95}]$')
    ax.plot(s0, floor, '-', color=C_SUB, linewidth=1.4,
            label=r'noise floor at $s_0$  (full split, n=5,000)')
    ax.plot(s0, bfloor, linestyle=(0, (2, 2)), color=C_SUB, linewidth=1.4,
            label=r'noise floor at $s_0$  (minibatch, B=200)')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\widehat{\mathrm{IPM}}$', fontsize=11, color=C_SUB)
    leg = ax.legend(fontsize=9, loc='lower left', frameon=True, framealpha=0.92,
                    edgecolor='none')
    leg.get_frame().set_facecolor('white')
    ax.set_title('SynthB: moving the true bump toward and past the trim boundary\n'
                 f'lambda = 0, mean of {rows[0]["n_seed"]} seeds',
                 fontsize=11.5, color='#0b0b0b', pad=10)

    ax2 = axes[1]
    ax2.axvspan(edge, max(s0.max() + 0.3, edge + 0.1), color='#f0efec', zorder=0,
                linewidth=0)
    ax2.axvline(edge, color=C_MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax2.axhline(1.0, color=C_MUTED, linewidth=1.0)
    ax2.plot(s0, trim / np.maximum(peak, 1e-9), 's-', color=C_TRIM, linewidth=2,
             markersize=6, label='reported / true  (1 = trim hides nothing)')
    ax2.plot(s0, peak / np.maximum(bfloor, 1e-9), '^-', color=C_PEAK, linewidth=2,
             markersize=6, label='true / minibatch noise floor  (1 = undetectable)')
    ax2.set_yscale('log')
    ax2.set_ylabel('ratio', fontsize=11, color=C_SUB)
    ax2.set_xlabel(r'$s_0$   (true bump centre, in units of sd(S))', fontsize=11,
                   color=C_SUB)
    ax2.legend(frameon=False, fontsize=9, loc='lower left')

    for a in axes:
        a.set_xlim(min(s0) - 0.2, max(s0.max() + 0.3, edge + 0.1))
        a.grid(True, color=C_GRID, linewidth=0.7)
        a.set_axisbelow(True)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            a.spines[side].set_color('#c3c2b7')
        a.tick_params(labelsize=9, colors=C_SUB)
    ax.text(edge - 0.05, ax.get_ylim()[1], f'trim boundary  {edge:.2f} sd  ',
            ha='right', va='top', fontsize=8.5, color=C_MUTED)

    fig.subplots_adjust(top=0.885, bottom=0.095, left=0.105, right=0.98)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(OUT_DIR, f'synthb_s0_shrinkage.{ext}'), dpi=200,
                    facecolor='white')
    print(f'[saved] {OUT_DIR}/synthb_s0_shrinkage.png|.pdf|.svg')

    hdr = (f"  {'s0':>5s} | {'peak(s0)':>9s} {'sup_trim':>9s} {'reported/true':>13s} | "
           f"{'floor':>8s} {'peak/floor':>10s} {'Bfloor':>8s} {'peak/Bfloor':>11s} | "
           f"{'sup_full':>8s} {'argmax (sd, per seed)':>22s} | {'mse':>8s}")
    print('\n' + hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['s0']:5.1f} | {r['peak']:9.4f} {r['sup_trim']:9.4f} "
              f"{r['sup_trim'] / max(r['peak'], 1e-9):13.2f} | {r['floor']:8.4f} "
              f"{r['peak'] / max(r['floor'], 1e-9):10.1f} "
              f"{r['bfloor']:8.4f} {r['peak'] / max(r['bfloor'], 1e-9):11.2f} | "
              f"{r['sup_full']:8.4f} {str(r['argmax_full_std']):>22s} | {r['mse']:8.5f}")
    print(f"\n  trim boundary = ±{edge:.2f} sd"
          "\n  peak(s0)      = 참 봉우리 위치에서의 관측값 (±0.04 창의 최대)"
          "\n  sup_trim      = 절사 구간 최대 = 프로토콜이 보고하는 값"
          "\n  Bfloor        = 그 지점의 미니배치(B=200) 순열 귀무값"
          "\n  sup_full/argmax = 전체 범위 최대와 그 위치 (양 끝 순서통계량으로 감)")


if __name__ == '__main__':
    main()
