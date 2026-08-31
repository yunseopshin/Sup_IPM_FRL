"""Aggregate the Crime supIPM-vs-FREM comparison (results/Crime-racepctblack/**
plus the critic20 tree), mirroring aggregate_adult_compare.py for the regression task.

Utility = test MAE; figures plot 1 - MAE (higher better), matching FREM Fig. 4's
INVERT convention for Crime. frem gamma_s in {0.03, 0.05, 0.07} selected per lambda
by LOWEST 5-seed-mean val MSE (FREM reproduction's DS_SPEC uses val_mse for Crime).
Final-epoch eval, seeds 2023-2027.

Outputs (results/Crime-racepctblack/compare/): compare_table.csv, frem_selected.csv,
compare_summary.md, tradeoff figures (MAE on y, fairness metric on x).
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(ROOT, 'results', 'Crime-racepctblack')
OUT = os.path.join(BASE, 'compare')
SEEDS = [2023, 2024, 2025, 2026, 2027]
METRICS = ['mae', 'mse', 'gdp_w_kernel', 'gdp_wo_kernel', 'inf_gdp', 'sup_ipm', 'hgr', 'mi_z_s']
ALG_KEEP = ('supipm', 'frem-')  # comparison scope (adv / reg_gdp trees left out)


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def collect():
    rows = []
    trees = [(BASE, '')]
    cs20 = os.path.join(ROOT, 'results', 'critic20', 'Crime-racepctblack')
    if os.path.isdir(cs20):
        trees.append((cs20, '-cs20'))
    for base, suffix in trees:
        for alg_tag in sorted(os.listdir(base)):
            alg_dir = os.path.join(base, alg_tag)
            if not os.path.isdir(alg_dir) or not alg_tag.startswith(ALG_KEEP):
                continue
            for lm_dir in sorted(os.listdir(alg_dir)):
                if not lm_dir.startswith('lmda_f-'):
                    continue
                lmda = float(lm_dir.split('lmda_f-')[1])
                for seed in SEEDS:
                    rj = os.path.join(alg_dir, lm_dir, f'seed-{seed}', 'results.json')
                    if not os.path.exists(rj):
                        rows.append((alg_tag + suffix, lmda, seed, None))
                        continue
                    with open(rj) as f:
                        rows.append((alg_tag + suffix, lmda, seed, json.load(f)))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = collect()
    groups = defaultdict(dict)
    for alg_tag, lmda, seed, r in rows:
        groups[(alg_tag, lmda)][seed] = r

    table = []
    for (alg_tag, lmda), by_seed in sorted(groups.items()):
        ok = {s: r for s, r in by_seed.items()
              if r is not None and finite(r['test'].get('mae'))
              and finite(r['test'].get('gdp_w_kernel'))}
        entry = dict(alg_tag=alg_tag, lmda_f=lmda,
                     n_seeds=len(ok), missing=len(SEEDS) - len(ok))
        for m in METRICS:
            vals = [r['test'][m] for r in ok.values() if finite(r['test'].get(m))]
            entry[f'test_{m}_mean'] = float(np.mean(vals)) if vals else float('nan')
            entry[f'test_{m}_std'] = float(np.std(vals)) if vals else float('nan')
        vm = [r['val']['mae'] for r in ok.values() if finite(r['val'].get('mae'))]
        entry['val_mae_mean'] = float(np.mean(vm)) if vm else float('nan')
        vs = [r['val']['mse'] for r in ok.values() if finite(r['val'].get('mse'))]
        entry['val_mse_mean'] = float(np.mean(vs)) if vs else float('nan')
        table.append(entry)

    import csv
    keys = list(table[0].keys()) if table else []
    with open(os.path.join(OUT, 'compare_table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)

    frem_tags = sorted({e['alg_tag'] for e in table
                        if e['alg_tag'].startswith('frem-') and not e['alg_tag'].endswith('-cs20')})
    frem_sel = []
    for lm in sorted({e['lmda_f'] for e in table if e['alg_tag'] in frem_tags}):
        cands = [e for e in table if e['alg_tag'] in frem_tags and e['lmda_f'] == lm
                 and e['n_seeds'] > 0]
        if not cands:
            continue
        full = [e for e in cands if e['n_seeds'] == len(SEEDS)]
        pool = full if full else cands
        frem_sel.append(min(pool, key=lambda e: e['val_mse_mean']))
    with open(os.path.join(OUT, 'frem_selected.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(frem_sel)

    supipm = sorted([e for e in table if e['alg_tag'] == 'supipm' and e['n_seeds'] > 0],
                    key=lambda e: e['lmda_f'])
    supipm_cs20 = sorted([e for e in table if e['alg_tag'] == 'supipm-cs20' and e['n_seeds'] > 0],
                         key=lambda e: e['lmda_f'])
    frem_sel = sorted(frem_sel, key=lambda e: e['lmda_f'])

    def fmt(e, m):
        return f"{e[f'test_{m}_mean']:.4f}±{e[f'test_{m}_std']:.4f}"

    lines = ['# Crime: supIPM vs FREM (unified representer, FREM protocol; utility = MAE, lower better)',
             '', f"seeds per point: {len(SEEDS)} (2023-2027); final-epoch eval", '']
    for entries, title in [(supipm, '## supIPM (critic 2-step baseline)'),
                           (supipm_cs20, '## supIPM critic_step=20, critic_lr=0.01'),
                           (frem_sel, '## FREM (gamma_s selected on val MAE per lambda)')]:
        if not entries:
            continue
        lines += [title, '',
                  '| λ | tag | n | MAE | ΔGDP(kernel) | inf_gdp | sup_ipm |', '|---|---|---|---|---|---|---|']
        for e in entries:
            lines.append(f"| {e['lmda_f']} | {e['alg_tag']} | {e['n_seeds']} | {fmt(e,'mae')} | "
                         f"{fmt(e,'gdp_w_kernel')} | {fmt(e,'inf_gdp')} | {fmt(e,'sup_ipm')} |")
        lines.append('')
    bad = [e for e in table if e['missing'] > 0]
    if bad:
        lines += ['## Incomplete configs', '']
        for e in bad:
            lines.append(f"- {e['alg_tag']} λ={e['lmda_f']}: {e['n_seeds']}/{len(SEEDS)} seeds")
    with open(os.path.join(OUT, 'compare_summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for fair_m, fname, xlabel in [('sup_ipm', 'tradeoff_mae_supipm.png', 'test sup_s ÎPM'),
                                  ('gdp_w_kernel', 'tradeoff_mae_gdp.png', 'test ΔGDP (kernel)'),
                                  ('inf_gdp', 'tradeoff_mae_infgdp.png',
                                   'test L∞-GDP  sup_s |E[ŷ|S=s] − E[ŷ]|'),
                                  ('mi_z_s', 'tradeoff_mae_mizs.png',
                                   'test MI(Z, S)  (FREM Fig. 22 proxy)')]:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        series = [(supipm, 'supIPM critic 2-step', 'tab:blue'),
                  (frem_sel, 'FREM (val-selected γ_s)', 'tab:red')]
        if supipm_cs20:
            series.append((supipm_cs20, 'supIPM critic 20-step', 'tab:green'))
        for entries, label, color in series:
            xs = [e[f'test_{fair_m}_mean'] for e in entries]
            ys = [1.0 - e['test_mae_mean'] for e in entries]  # FREM Fig.4 INVERT convention
            xerr = [e[f'test_{fair_m}_std'] for e in entries]
            yerr = [e['test_mae_std'] for e in entries]
            ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt='o-', ms=4, lw=1.2,
                        capsize=2, label=label, color=color, alpha=0.8)
            for e, x, y in zip(entries, xs, ys):
                ax.annotate(f"{e['lmda_f']:g}", (x, y), fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('test 1 − MAE')
        ax.set_title('Crime (S=racepctblack): fairness-utility tradeoff')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)
        plt.close(fig)

    print(f"[aggregate] {len([r for r in rows if r[3] is not None])} runs, "
          f"{len(table)} (alg, lambda) points -> {OUT}")


if __name__ == '__main__':
    main()
