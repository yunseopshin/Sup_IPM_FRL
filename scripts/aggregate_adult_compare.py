"""Aggregate the Adult supIPM-vs-FREM comparison (results/Adult-age/**).

FREM protocol: final-epoch eval; for the headline FREM curve, per lambda pick the
gamma_s in {0.03, 0.05, 0.07} with the highest 5-seed-mean val acc; report
5-seed mean +/- std of test metrics. Missing / NaN runs are reported, not hidden.

Outputs (results/Adult-age/compare/):
  compare_table.csv     one row per (alg_tag, lmda_f) with n_seeds and metric stats
  frem_selected.csv     the per-lambda gamma_s selection for the headline FREM curve
  compare_summary.md    readable summary table
  tradeoff_acc_supipm.png, tradeoff_acc_gdp.png
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(ROOT, 'results', 'Adult-age')
OUT = os.path.join(BASE, 'compare')
SEEDS = [2023, 2024, 2025, 2026, 2027]
METRICS = ['acc', 'bacc', 'ap', 'gdp_w_kernel', 'gdp_wo_kernel', 'inf_gdp', 'sup_ipm', 'hgr', 'mi_z_s']


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def collect():
    rows = []  # (alg_tag, lmda, seed, {split: metrics})
    # (base_dir, tag_suffix): critic20 tree = supipm with critic_step=20, critic_lr=0.01
    trees = [(BASE, '')]
    cs20 = os.path.join(ROOT, 'results', 'critic20', 'Adult-age')
    if os.path.isdir(cs20):
        trees.append((cs20, '-cs20'))
    for base, suffix in trees:
        for alg_tag in sorted(os.listdir(base)):
            alg_dir = os.path.join(base, alg_tag)
            if not os.path.isdir(alg_dir) or alg_tag == 'compare':
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
                        r = json.load(f)
                    rows.append((alg_tag + suffix, lmda, seed, r))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = collect()

    # group: (alg_tag, lmda) -> list of per-seed results (None = missing)
    groups = defaultdict(dict)
    for alg_tag, lmda, seed, r in rows:
        groups[(alg_tag, lmda)][seed] = r

    table = []
    for (alg_tag, lmda), by_seed in sorted(groups.items()):
        valid = {s: r for s, r in by_seed.items()
                 if r is not None and finite(r['test'].get('acc'))
                 and finite(r['test'].get('gdp_w_kernel'))}
        # constant-predictor collapse (bacc==0.5): report separately, keep out of the stats
        collapsed = {s: r for s, r in valid.items()
                     if r['test'].get('bacc') is not None and r['test']['bacc'] <= 0.5001}
        ok = {s: r for s, r in valid.items() if s not in collapsed}
        entry = dict(alg_tag=alg_tag, lmda_f=lmda,
                     n_seeds=len(ok), collapsed=len(collapsed),
                     missing=len(SEEDS) - len(valid))
        for m in METRICS:
            vals = [r['test'][m] for r in ok.values() if finite(r['test'].get(m))]
            entry[f'test_{m}_mean'] = float(np.mean(vals)) if vals else float('nan')
            entry[f'test_{m}_std'] = float(np.std(vals)) if vals else float('nan')
        va = [r['val']['acc'] for r in ok.values() if finite(r['val'].get('acc'))]
        entry['val_acc_mean'] = float(np.mean(va)) if va else float('nan')
        table.append(entry)

    import csv
    keys = list(table[0].keys()) if table else []
    with open(os.path.join(OUT, 'compare_table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)

    # headline FREM curve: per lambda pick gamma_s by val acc (only full-5-seed configs,
    # FREM figure convention; fall back to best available if none has 5 seeds)
    frem_tags = sorted({e['alg_tag'] for e in table if e['alg_tag'].startswith('frem-')})
    frem_sel = []
    frem_lmdas = sorted({e['lmda_f'] for e in table if e['alg_tag'] in frem_tags})
    for lm in frem_lmdas:
        cands = [e for e in table if e['alg_tag'] in frem_tags and e['lmda_f'] == lm
                 and e['n_seeds'] > 0]
        if not cands:
            continue
        full = [e for e in cands if e['n_seeds'] == len(SEEDS)]
        pool = full if full else cands
        best = max(pool, key=lambda e: e['val_acc_mean'])
        frem_sel.append(best)
    with open(os.path.join(OUT, 'frem_selected.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(frem_sel)

    supipm = sorted([e for e in table if e['alg_tag'] == 'supipm' and e['n_seeds'] > 0],
                    key=lambda e: e['lmda_f'])
    supipm_cs20 = sorted([e for e in table if e['alg_tag'] == 'supipm-cs20' and e['n_seeds'] > 0],
                         key=lambda e: e['lmda_f'])
    frem_sel = sorted(frem_sel, key=lambda e: e['lmda_f'])

    # summary markdown
    def fmt(e, m):
        return f"{e[f'test_{m}_mean']:.4f}±{e[f'test_{m}_std']:.4f}"

    lines = ['# Adult: supIPM vs FREM (unified representer, FREM protocol)', '',
             f"seeds per point: {len(SEEDS)} (2023-2027); test split n=12661; final-epoch eval", '',
             '## supIPM (ours)', '',
             '| λ | n | acc | ap | ΔGDP(kernel) | inf_gdp | sup_ipm |', '|---|---|---|---|---|---|---|']
    for e in supipm:
        lines.append(f"| {e['lmda_f']} | {e['n_seeds']} | {fmt(e,'acc')} | {fmt(e,'ap')} | "
                     f"{fmt(e,'gdp_w_kernel')} | {fmt(e,'inf_gdp')} | {fmt(e,'sup_ipm')} |")
    if supipm_cs20:
        lines += ['', '## supIPM critic_step=20, critic_lr=0.01', '',
                  '| λ | n | acc | ap | ΔGDP(kernel) | inf_gdp | sup_ipm |', '|---|---|---|---|---|---|---|']
        for e in supipm_cs20:
            lines.append(f"| {e['lmda_f']} | {e['n_seeds']} | {fmt(e,'acc')} | {fmt(e,'ap')} | "
                         f"{fmt(e,'gdp_w_kernel')} | {fmt(e,'inf_gdp')} | {fmt(e,'sup_ipm')} |")
    lines += ['', '## FREM (γ_s selected on val per λ)', '',
              '| λ | γ_s tag | n | acc | ap | ΔGDP(kernel) | inf_gdp | sup_ipm |', '|---|---|---|---|---|---|---|---|']
    for e in frem_sel:
        lines.append(f"| {e['lmda_f']} | {e['alg_tag']} | {e['n_seeds']} | {fmt(e,'acc')} | {fmt(e,'ap')} | "
                     f"{fmt(e,'gdp_w_kernel')} | {fmt(e,'inf_gdp')} | {fmt(e,'sup_ipm')} |")
    # incomplete / collapsed runs
    bad = [e for e in table if e['missing'] > 0 or e['collapsed'] > 0]
    if bad:
        lines += ['', '## Incomplete / collapsed configs (stats above exclude collapsed seeds)', '']
        for e in bad:
            lines.append(f"- {e['alg_tag']} λ={e['lmda_f']}: {e['n_seeds']}/{len(SEEDS)} healthy, "
                         f"{e['collapsed']} collapsed (bacc=0.5), {e['missing']} missing")
    with open(os.path.join(OUT, 'compare_summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # tradeoff figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for fair_m, fname, xlabel in [('sup_ipm', 'tradeoff_acc_supipm.png', 'test sup_s ÎPM'),
                                  ('gdp_w_kernel', 'tradeoff_acc_gdp.png', 'test ΔGDP (kernel)'),
                                  ('inf_gdp', 'tradeoff_acc_infgdp.png',
                                   'test L∞-GDP  sup_s |E[ŷ|S=s] − E[ŷ]|'),
                                  ('mi_z_s', 'tradeoff_acc_mizs.png',
                                   'test MI(Z, S)  (FREM Fig. 22 proxy)')]:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        series = [(supipm, 'supIPM critic 2-step', 'tab:blue'),
                  (frem_sel, 'FREM (val-selected γ_s)', 'tab:red')]
        if supipm_cs20:
            series.append((supipm_cs20, 'supIPM critic 20-step', 'tab:green'))
        for entries, label, color in series:
            xs = [e[f'test_{fair_m}_mean'] for e in entries]
            ys = [e['test_acc_mean'] for e in entries]
            xerr = [e[f'test_{fair_m}_std'] for e in entries]
            yerr = [e['test_acc_std'] for e in entries]
            ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt='o-', ms=4, lw=1.2,
                        capsize=2, label=label, color=color)
            for e, x, y in zip(entries, xs, ys):
                ax.annotate(f"{e['lmda_f']:g}", (x, y), fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('test accuracy')
        ax.set_title('Adult (S=age): fairness-accuracy tradeoff')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)
        plt.close(fig)

    print(f"[aggregate] {len([r for r in rows if r[3] is not None])} runs found, "
          f"{len(table)} (alg, lambda) points -> {OUT}")


if __name__ == '__main__':
    main()
