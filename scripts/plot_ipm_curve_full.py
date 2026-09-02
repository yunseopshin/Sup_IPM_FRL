"""Conditional IPM curve  Delta_hat(s) = sup_v IPM_hat(h, s)  over the FULL S range.

The saved `sup_ipm_curve.npz` of each run only covers the alpha-trimmed range that
the training sup was constrained to. This script re-evaluates trained checkpoints
on a grid spanning the whole raw S range [0, 1], to answer: does trimming hide a
real fairness violation, or does it only cut away estimator noise?

Three curves per (dataset, lambda), all on the same grid:

  observed   Delta_hat(s) from the trained representation
  null       the same estimator after randomly permuting S (so the truth is 0 at
             every s) -- the noise floor of the estimator at that s
  n_eff(s)   1 / sum_i w_i(s)^2, how many samples the estimate at s averages over

Usage (compute needs a GPU -> Slurm, see CLAUDE.md):
    srun --gres=gpu:1 --cpus-per-task=10 --partition=idea python \\
        scripts/plot_ipm_curve_full.py compute --dataset Crime
    python scripts/plot_ipm_curve_full.py plot          # CPU, reads the npz files
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from data import apply_preset, get_loaders            # noqa: E402
from evaluate import collect                          # noqa: E402
from ipm import kernel_weights, relu_ipm_sup_grid     # noqa: E402
from models import build_model, ReLUCritic, REP_DIM   # noqa: E402

DEFAULT_ROOT = {'Crime': 'results/Crime-racepctblack/supipm',
                'Adult': 'results/Adult-age/supipm',
                'SynthB': 'results/synthB-interior/SynthB-synth_s/supipm',
                'SynthB_uniform': 'results/synthB-uniform/SynthB-synth_s/supipm'}
# display names: the npz stem is the key, the panel title is the value
TITLE = {'SynthB': 'SynthB   ·   S ~ Normal',
         'SynthB_uniform': 'SynthB   ·   S ~ Uniform'}
SEEDS = {'Crime': [2023, 2024, 2025, 2026, 2027],
         'Adult': [2023, 2024, 2025, 2026, 2027],
         'SynthB': [2023, 2024, 2025],
         'SynthB_uniform': [2023, 2024, 2025]}
# lambda = 0 and 1 as asked, plus one intermediate: on Crime lambda >= 0.1 is
# saturated (sup_ipm ~ 1e-4, MAE 0.19), so a partially-fair model is the only
# place where a violation could survive OUTSIDE the trimmed range.
LMDAS = {'Crime': ['0.0', '0.03', '1.0'], 'Adult': ['0.0', '0.3', '1.0'],
         'SynthB': ['0.0', '0.1', '0.3', '1.0'],
         'SynthB_uniform': ['0.0', '0.1', '0.3', '1.0']}
SPLITS = [('test', 'test'), ('train', 'traineval')]
OUT_DIR = os.path.join(_ROOT, 'notebook', 'figures')

# Crime/Adult: categorical slots 1 / 3 / 2 (blue / aqua / orange).
# SynthB: four ordered lambdas, so a one-hue ordinal blue ramp instead.
C = {'0.0': '#2a78d6', '0.03': '#1baf7a', '0.3': '#1baf7a', '1.0': '#eb6834'}
C_RAMP = ['#86b6ef', '#3987e5', '#256abf', '#0d366b']
C_MUTED = '#898781'
C_GRID = '#e1e0d9'
C_INK = '#0b0b0b'
C_SUB = '#52514e'


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #
def load_run(run_dir, data, device):
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        cfg = apply_preset(yaml.safe_load(f))
    ckpt = torch.load(os.path.join(run_dir, 'model.pt'), map_location=device)
    model = build_model(data['input_dim'], data['task']).to(device)
    model.load_state_dict(ckpt['model'])
    critic = None
    if ckpt.get('critic') is not None:
        critic = ReLUCritic(REP_DIM, cfg['critic_num']).to(device)
        critic.load_state_dict(ckpt['critic'])
    return cfg, model, critic


def curve(reps, s_std, cfg, critic, grid_std, device, seed, cross_eval=25):
    """cross_eval > 0 lets every grid point also use the other grid points' critics,
    so a failed restart no longer punches a spurious dip into the curve. It only
    tightens the sup, so the curve sits slightly ABOVE the `sup_ipm` recorded in
    each run's results.json (which uses the plain per-point protocol)."""
    torch.manual_seed(100000 + seed)          # same restart draw as run_eval
    vals = relu_ipm_sup_grid(reps, s_std, cfg['bandwidth'], grid_std,
                             n_restarts=cfg['eval_restarts'],
                             n_steps=cfg['eval_steps'], lr=cfg['eval_lr'],
                             warm_critic=critic,
                             mode=cfg.get('critic_proj', 'sphere_box'),
                             cross_eval=cross_eval)
    return vals.detach().cpu().numpy()


def n_eff(grid_std, s_std, h):
    """Kish effective sample size 1/sum_i w_i(s)^2 of the kernel weights, chunked
    over the grid because n can be 32k."""
    with torch.no_grad():
        out = []
        for i in range(0, grid_std.numel(), 64):
            w = kernel_weights(grid_std[i:i + 64], s_std, h)
            out.append((1.0 / (w ** 2).sum(dim=1)).cpu().numpy())
    return np.concatenate(out)


def compute(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Info] device={device}', flush=True)
    root = args.root or DEFAULT_ROOT[args.dataset]
    G = args.grid
    grid_raw = np.linspace(0.0, 1.0, G)               # full raw S range
    store = {'grid_raw': grid_raw}
    trims, sd_mean = [], []
    seeds = args.seeds or SEEDS[args.dataset]
    lmdas = args.lmdas or LMDAS[args.dataset]
    store['lmdas'] = np.asarray(lmdas)
    s0s = []

    for si, seed in enumerate(seeds):
        t0 = time.time()
        run0 = os.path.join(_ROOT, root, f'lmda_f-{lmdas[0]}', f'seed-{seed}')
        with open(os.path.join(run0, 'config.yaml')) as f:
            cfg0 = apply_preset(yaml.safe_load(f))
        data = get_loaders(cfg0)                      # one loader build per seed
        s_mean, s_sd = data['s_mean'], data['s_sd']
        trims.append([data['s_lo'] * s_sd + s_mean, data['s_hi'] * s_sd + s_mean])
        sd_mean.append([s_mean, s_sd])
        grid_std = torch.tensor((grid_raw - s_mean) / s_sd,
                                dtype=torch.float32, device=device)
        if args.dataset == 'SynthB':
            # the true input-level bump sits at s0 in the ORIGINAL N(0,1) draw;
            # _synthb_loaders then min-max scales S by the train draw's own range,
            # and the first n normals of the seeded rng ARE that train draw
            from data import SYNTHB, synthb_s_draw
            draw = synthb_s_draw(np.random.default_rng(cfg0['seed']),
                                 cfg0.get('synthb_n', 2000),
                                 cfg0.get('synthb_s_dist') or 'normal')
            s0_true = cfg0.get('synthb_s0')
            s0_true = SYNTHB['s0'] if s0_true is None else s0_true
            s0_raw = (s0_true - draw.min()) / (draw.max() - draw.min())
            s0s.append([s0_raw, (s0_raw - s_mean) / s_sd])
        print(f'[seed {seed}] loaders ready ({time.time() - t0:.0f}s), '
              f'trim_raw={trims[-1][0]:.3f}..{trims[-1][1]:.3f}'
              + (f', s0_raw={s0s[-1][0]:.3f} (std {s0s[-1][1]:.2f})' if s0s else ''),
              flush=True)

        for lmda in lmdas:
            run_dir = os.path.join(_ROOT, root, f'lmda_f-{lmda}', f'seed-{seed}')
            cfg, model, critic = load_run(run_dir, data, device)
            rng = np.random.default_rng(seed * 1000 + int(float(lmda) * 100))
            for split, loader_key in SPLITS:
                reps, _, _, _, sens = collect(model, data[loader_key],
                                              data['input_dim'], data['task'], device,
                                              z_norm=bool(cfg.get('z_norm', True)))
                s_std = ((sens.float() - s_mean) / s_sd).to(device)
                n = s_std.numel()
                t1 = time.time()
                obs = curve(reps, s_std, cfg, critic, grid_std, device, seed)
                store.setdefault(f'{split}_l{lmda}_obs', []).append(obs)

                n_null = args.n_null if split in args.null_splits else 0
                for _ in range(n_null):
                    perm = torch.as_tensor(rng.permutation(n), device=device)
                    store.setdefault(f'{split}_l{lmda}_null', []).append(
                        curve(reps, s_std[perm], cfg, critic, grid_std, device, seed))

                # what the TRAINING sup actually sees: one minibatch at a time,
                # observed and with S permuted inside the batch (its noise floor)
                if split == 'train' and args.n_batch:
                    B = int(cfg['batch_size'])
                    for _ in range(args.n_batch):
                        idx = torch.as_tensor(rng.choice(n, B, replace=False), device=device)
                        zb, sb = reps[idx], s_std[idx]
                        store.setdefault(f'batch_l{lmda}_obs', []).append(
                            curve(zb, sb, cfg, critic, grid_std, device, seed))
                        store.setdefault(f'batch_l{lmda}_null', []).append(
                            curve(zb, sb[torch.as_tensor(rng.permutation(B), device=device)],
                                  cfg, critic, grid_std, device, seed))
                    store.setdefault('batch_neff', []).append(
                        n_eff(grid_std, sb, cfg['bandwidth']))
                    store['batch_size'] = np.asarray([B])

                if lmda == lmdas[0]:      # split geometry does not depend on lambda
                    store.setdefault(f'{split}_neff', []).append(
                        n_eff(grid_std, s_std, cfg['bandwidth']))
                    store.setdefault(f'{split}_n', []).append(float(n))
                print(f'  [{split} lmda={lmda}] sup(full)={obs.max():.4f} '
                      f'at s={grid_raw[obs.argmax()]:.3f}  ({time.time() - t1:.0f}s)',
                      flush=True)
            del model, critic
            torch.cuda.empty_cache()

    store['trim_raw'] = np.asarray(trims)
    store['s_stats'] = np.asarray(sd_mean)
    store['bandwidth'] = np.asarray([cfg0['bandwidth']])
    store['seeds'] = np.asarray(seeds)
    if s0s:
        store['s0'] = np.asarray(s0s)          # [n_seeds, 2] = (raw scale, std scale)
    out = {k: np.asarray(v) for k, v in store.items()}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'ipm_curve_full_{args.dataset}{args.tag}.npz')
    np.savez(path, **out)
    print(f'[saved] {path}', flush=True)


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
def band(ax, x, arr, color, label=None, lw=1.9, ls='-', alpha=0.16):
    m = arr.mean(axis=0)
    ax.fill_between(x, arr.min(axis=0), arr.max(axis=0), color=color, alpha=alpha,
                    linewidth=0)
    ax.plot(x, m, color=color, linewidth=lw, linestyle=ls, label=label)
    return m


def _lmdas(z):
    return [str(v) for v in z['lmdas']]


def task_metric(ds, root, lmda, seeds):
    """Mean test MSE (reg) / accuracy (cls) of the same runs, from results.json."""
    key = 'acc' if ds == 'Adult' else 'mse'
    if root is None:
        return key, float('nan')
    vals = []
    for seed in seeds:
        f = os.path.join(_ROOT, root, f'lmda_f-{lmda}', f'seed-{seed}', 'results.json')
        if os.path.exists(f):
            with open(f) as fh:
                vals.append(json.load(fh)['test'][key])
    return (key, float(np.mean(vals))) if vals else (key, float('nan'))


def plot(args):
    datasets = [d for d in (args.datasets or ('Crime', 'Adult'))
                if os.path.exists(os.path.join(OUT_DIR, f'ipm_curve_full_{d}.npz'))]
    if not datasets:
        raise SystemExit('no npz found — run `compute` first')
    Z = {d: np.load(os.path.join(OUT_DIR, f'ipm_curve_full_{d}.npz')) for d in datasets}

    ncol = len(datasets)
    fig, axes = plt.subplots(3, ncol, figsize=(5.6 * ncol if ncol > 1 else 7.6, 8.6),
                             sharex='col',
                             gridspec_kw=dict(height_ratios=[1.0, 1.0, 0.66],
                                              hspace=0.14, wspace=0.22))
    axes = np.asarray(axes).reshape(3, ncol)
    summary = []

    for col, ds in enumerate(datasets):
        z = Z[ds]
        x = z['grid_raw']
        lmdas = _lmdas(z)
        colors = (dict(zip(lmdas, C_RAMP)) if ds.startswith('SynthB')
                  else {l: C[l] for l in lmdas})
        lo, hi = z['trim_raw'].mean(axis=0)
        B = int(z['batch_size'][0])
        inside = (x >= lo) & (x <= hi)
        s0 = float(z['s0'][:, 0].mean()) if 's0' in z.files else None

        def shade(ax, mark_s0=True):
            for v0, v1 in ((x[0], lo), (hi, x[-1])):
                if v1 > v0:
                    ax.axvspan(v0, v1, color='#f0efec', zorder=0, linewidth=0)
            for v in (lo, hi):
                ax.axvline(v, color=C_MUTED, linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
            if s0 is not None and mark_s0:
                ax.axvline(s0, color='#d03b3b', linewidth=1.3, zorder=2)

        # ---- row 0: observed curve on the test split ------------------------
        ax0 = axes[0, col]
        shade(ax0)
        for lmda in lmdas:
            band(ax0, x, z[f'test_l{lmda}_obs'], colors[lmda], label=f'λ = {float(lmda):g}')
        if s0 is not None:
            ax0.plot([], [], color='#d03b3b', linewidth=1.3,
                     label=f'true bump $s_0$ = {s0:.2f}')
        if ncol > 1:                    # with one column the suptitle already names it
            # a secondary top axis eats ~30pt, so the title has to clear it
            ax0.set_title(TITLE.get(ds, ds), fontsize=12.5, color=C_INK,
                          pad=8 if s0 is None else 34)
        elif s0 is not None:
            ax0.set_title(' ', fontsize=12.5, pad=26)   # room for the top axis
        ax0.set_ylabel(r'$\widehat{\mathrm{IPM}}(s)$  ·  observed (test)', fontsize=10,
                       color=C_SUB)
        ax0.legend(frameon=False, fontsize=9, loc='upper left', borderaxespad=0.3)
        if 's_stats' in z.files and ds.startswith('SynthB'):
            s_m, s_sd = (float(v) for v in z['s_stats'].mean(axis=0))
            # bind by default arg: the lambdas are called at draw time, long after
            # this loop has moved on
            sec = ax0.secondary_xaxis('top', functions=(
                lambda v, a=s_m, b=s_sd: (v - a) / b,
                lambda t, a=s_m, b=s_sd: a + t * b))
            sec.set_xlabel('s in units of sd(S)  (standardised)', fontsize=8.5,
                           color=C_MUTED, labelpad=4)
            sec.tick_params(labelsize=8.5, colors=C_MUTED)
            sec.spines['top'].set_color('#c3c2b7')

        # ---- row 1: the same estimator with S permuted (noise floor) --------
        ax1 = axes[1, col]
        shade(ax1)
        for lmda in lmdas:
            ax1.plot(x, z[f'test_l{lmda}_null'].mean(axis=0), color=colors[lmda],
                     linewidth=1.8, label=f'λ = {float(lmda):g},  full test split')
            ax1.plot(x, z[f'batch_l{lmda}_null'].mean(axis=0), color=colors[lmda],
                     linewidth=1.3, linestyle=(0, (2, 2)), alpha=0.85,
                     label=f'λ = {float(lmda):g},  minibatch B={B}')
        ax1.set_ylabel(r'$\widehat{\mathrm{IPM}}(s)$  ·  null (S permuted)', fontsize=10,
                       color=C_SUB)
        ax1.legend(frameon=False, fontsize=7.8, loc='upper left', ncol=2,
                   columnspacing=1.0, handlelength=2.0, borderaxespad=0.3)

        top = max(ax0.get_ylim()[1], ax1.get_ylim()[1])
        for a in (ax0, ax1):
            a.set_ylim(0, top)
        # rows 0 and 1 share the scale on purpose — say how small the floor is
        nmax = max(z[f'test_l{l}_null'].mean(axis=0).max() for l in lmdas)
        bmax = max(z[f'batch_l{l}_null'].mean(axis=0).max() for l in lmdas)
        ax1.text(0.985, 0.60, f'noise floor over the whole range:\n'
                              f'≤ {nmax:.2f}  (full split)   ·   ≤ {bmax:.2f}  (minibatch)',
                 transform=ax1.transAxes, ha='right', va='top', fontsize=8.4,
                 color=C_SUB)

        # ---- row 2: how many samples the estimate at s averages over --------
        ax2 = axes[2, col]
        shade(ax2)
        for key, lab, ls in (('test', f"test (n={z['test_n'].mean():,.0f})", '-'),
                             ('train', f"train (n={z['train_n'].mean():,.0f})", (0, (5, 2))),
                             ('batch', f'minibatch (B={B})', (0, (1, 2)))):
            ax2.plot(x, z[f'{key}_neff'].mean(axis=0), color=C_SUB, linewidth=1.6,
                     linestyle=ls, label=lab)
        ax2.axhline(10, color='#d03b3b', linewidth=1.0, alpha=0.8)
        ax2.text(0.985, 10, '$n_{\\mathrm{eff}}=10$ ', color='#d03b3b', fontsize=8,
                 va='bottom', ha='right')
        ax2.set_yscale('log')
        ax2.set_ylabel(r'$n_{\mathrm{eff}}(s)$   (log)', fontsize=10, color=C_SUB)
        ax2.set_xlabel('s      (S, min-max scaled to [0, 1])', fontsize=10, color=C_SUB)
        leg = ax2.legend(fontsize=8.4, loc='best', ncol=1, handlelength=2.0,
                         borderaxespad=0.4, labelspacing=0.3, frameon=True,
                         framealpha=0.92, edgecolor='none')
        leg.get_frame().set_facecolor('white')

        for r in range(3):
            a = axes[r, col]
            a.set_xlim(x[0], x[-1])
            a.grid(True, color=C_GRID, linewidth=0.7)
            a.set_axisbelow(True)
            for side in ('top', 'right'):
                a.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                a.spines[side].set_color('#c3c2b7')
            a.tick_params(labelsize=9, colors=C_SUB)

        for lmda in lmdas:
            row = dict(ds=ds, lmda=lmda, trim_hi=hi)
            for tag, key in (('obs', f'test_l{lmda}_obs'), ('null', f'test_l{lmda}_null'),
                             ('bnull', f'batch_l{lmda}_null'), ('bobs', f'batch_l{lmda}_obs'),
                             ('tr_obs', f'train_l{lmda}_obs')):
                m = z[key].mean(axis=0)
                row[f'{tag}_trim'] = float(m[inside].max())
                row[f'{tag}_full'] = float(m.max())
                row[f'{tag}_argmax'] = float(x[m.argmax()])
                row[f'{tag}_argmax_trim'] = float(x[inside][m[inside].argmax()])
            # noise-corrected curve: the permutation null is the estimator's bias
            # floor at each s, so obs - null is what survives it
            exc = z[f'test_l{lmda}_obs'].mean(axis=0) - z[f'test_l{lmda}_null'].mean(axis=0)
            row['exc_trim'] = float(exc[inside].max())
            row['exc_full'] = float(exc.max())
            row['exc_argmax'] = float(x[exc.argmax()])
            k = int(z[f'test_l{lmda}_obs'].mean(axis=0).argmax())
            row['bnull_at'] = float(z[f'batch_l{lmda}_null'].mean(axis=0)[k])
            row['null_at'] = float(z[f'test_l{lmda}_null'].mean(axis=0)[k])
            row['snr'] = row['obs_full'] / max(row['bnull_at'], 1e-9)
            row['metric_key'], row['metric'] = task_metric(
                ds, args.root or DEFAULT_ROOT.get(ds), lmda,
                z['seeds'] if 'seeds' in z.files else SEEDS.get(ds, []))
            row['s0'] = s0
            row['d_s0'] = abs(row['obs_argmax'] - s0) if s0 is not None else float('nan')
            # the full-range argmax sits on the extreme order statistic of S, so for
            # "did it find the bump" the argmax INSIDE the trim is the honest number
            row['d_s0_trim'] = (abs(row['obs_argmax_trim'] - s0) if s0 is not None
                                else float('nan'))
            # per-seed argmax, to say whether the peak location is stable
            row['argmax_seeds'] = [float(x[v.argmax()]) for v in z[f'test_l{lmda}_obs']]
            summary.append(row)

    n_seed = len(Z[datasets[0]]['test_l' + _lmdas(Z[datasets[0]])[0] + '_obs'])
    title = ('Conditional IPM over the FULL range of S   —   grey = cut away by the '
             f'5% trim   (mean of {n_seed} seeds, band = min–max)')
    if ncol == 1:
        title = (f'{datasets[0]}: conditional IPM over the FULL range of S\n'
                 f'grey = cut away by the 5% trim   ·   mean of {n_seed} seeds, '
                 'band = min–max')
    has_sec = any('s0' in Z[d].files for d in datasets)
    fig.suptitle(title, fontsize=10.5 if has_sec else 11.5, color=C_INK,
                 y=0.995 if ncol > 1 else 0.975, va='top')
    fig.subplots_adjust(top=(0.87 if has_sec else 0.925) if ncol > 1 else 0.895,
                        bottom=0.072, left=0.082 if ncol > 1 else 0.115, right=0.985)
    stem = args.out or 'ipm_curve_full'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(OUT_DIR, f'{stem}.{ext}'), dpi=200, facecolor='white')
    print(f'[saved] {OUT_DIR}/{stem}.png|.pdf|.svg')

    has_s0 = any(r['s0'] is not None for r in summary)
    hdr = (f"  {'dataset':7s} {'lmda':>5s} | {'sup_trim':>9s} {'sup_full':>9s} "
           f"{'argmax':>7s} | {'Bnull@arg':>9s} {'sig/noise':>9s} | "
           f"{'exc_full':>8s} {'metric':>9s}"
           + (f" | {'arg_trim':>8s} {'|arg_tr-s0|':>11s}" if has_s0 else ''))
    print('\n' + hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for r in summary:
        line = (f"  {r['ds']:7s} {r['lmda']:>5s} | {r['obs_trim']:9.4f} "
                f"{r['obs_full']:9.4f} {r['obs_argmax']:7.3f} | "
                f"{r['bnull_at']:9.4f} {r['snr']:9.1f} | {r['exc_full']:8.4f} "
                f"{r['metric']:9.4f}")
        if has_s0:
            line += f" | {r['obs_argmax_trim']:8.3f} {r['d_s0_trim']:11.3f}"
        print(line)
    if has_s0:
        print('\n  per-seed argmax:')
        for r in summary:
            print(f"    {r['ds']:7s} lmda={r['lmda']:>5s}  "
                  f"{[round(a, 3) for a in r['argmax_seeds']]}   (s0 = {r['s0']:.3f})")
    print('\n  sup_*      = 관측 곡선의 최대값 (절사 구간 / 전체 범위)'
          '\n  Bnull@arg  = 전체 최댓값 지점에서 미니배치 순열 귀무값 = 그 지점 잡음 바닥'
          '\n  sig/noise  = sup_full / Bnull@arg'
          '\n  exc_full   = (관측 − 전체split 귀무)의 최대값'
          '\n  metric     = test MSE (reg) / accuracy (Adult)')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('compute')
    c.add_argument('--dataset', required=True, choices=['Crime', 'Adult', 'SynthB'])
    c.add_argument('--root', default=None, help='run root, default results/<ds>/supipm')
    c.add_argument('--grid', type=int, default=401)
    c.add_argument('--n_null', type=int, default=1, help='permutation curves per split')
    c.add_argument('--n_batch', type=int, default=6, help='minibatch draws per run')
    c.add_argument('--null_splits', nargs='*', default=['test'])
    c.add_argument('--seeds', type=int, nargs='+', default=None)
    c.add_argument('--lmdas', nargs='+', default=None)
    c.add_argument('--tag', default='', help='suffix for the output npz (smoke tests)')
    q = sub.add_parser('plot')
    q.add_argument('--datasets', nargs='+', default=None,
                   help='which npz files to draw (default Crime Adult)')
    q.add_argument('--out', default=None, help='figure stem, default ipm_curve_full')
    q.add_argument('--root', default=None, help='run root for the task metric column')
    args = p.parse_args()
    (compute if args.cmd == 'compute' else plot)(args)


if __name__ == '__main__':
    main()
