"""Adult critic_step ladder with s_mode: does percritic at critic_step=2 recover
what shared needs critic_step=20 for? (PERCRITIC_S.md sec.5, Adult extension.)

Cells, all at critic_lr=0.01, lambda in {1.0, 2.0}, FREM seeds 2023-2027:
  shared    cs2 / cs5 / cs10   results/critic_ladder/cs{2,5,10}
  shared    cs20               results/critic20
  percritic cs2                results/critic_ladder/cs2   (-percritic run-dir suffix)

Same conventions as scripts/aggregate_adult_compare.py: final-epoch eval,
constant-predictor collapses (test bacc <= 0.5001) reported separately and kept
out of the stats, missing runs reported.

Usage: python scripts/aggregate_smode_adult.py
"""
import json
import math
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEEDS = [2023, 2024, 2025, 2026, 2027]
LMDAS = [1.0, 2.0]
METRICS = ['acc', 'bacc', 'ap', 'gdp_w_kernel', 'inf_gdp', 'sup_ipm', 'hgr', 'mi_z_s']

# (label, results tree, alg dir)
CELLS = [
    ('shared cs2', os.path.join(ROOT, 'results', 'critic_ladder', 'cs2', 'Adult-age'), 'supipm'),
    ('percritic cs2', os.path.join(ROOT, 'results', 'critic_ladder', 'cs2', 'Adult-age'), 'supipm-percritic'),
    ('shared cs5', os.path.join(ROOT, 'results', 'critic_ladder', 'cs5', 'Adult-age'), 'supipm'),
    ('shared cs10', os.path.join(ROOT, 'results', 'critic_ladder', 'cs10', 'Adult-age'), 'supipm'),
    ('shared cs20', os.path.join(ROOT, 'results', 'critic20', 'Adult-age'), 'supipm'),
]


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def load_cell(base, alg, lmda):
    ok, collapsed, missing = {}, {}, []
    for seed in SEEDS:
        rj = os.path.join(base, alg, f'lmda_f-{lmda}', f'seed-{seed}', 'results.json')
        if not os.path.exists(rj):
            missing.append(seed)
            continue
        with open(rj) as f:
            r = json.load(f)
        if not (finite(r['test'].get('acc')) and finite(r['test'].get('gdp_w_kernel'))):
            missing.append(seed)
        elif r['test'].get('bacc') is not None and r['test']['bacc'] <= 0.5001:
            collapsed[seed] = r
        else:
            ok[seed] = r
    return ok, collapsed, missing


def stat(ok, metric):
    vals = [r['test'][metric] for r in ok.values() if finite(r['test'].get(metric))]
    if not vals:
        return float('nan'), float('nan')
    return float(np.mean(vals)), float(np.std(vals))


def train_tail(base, alg, lmda, seeds, col='ipm_s_star', n_last=10):
    """Mean over seeds of the last-n_last-epoch mean of a train_log column."""
    import csv as _csv
    out = []
    for seed in seeds:
        p = os.path.join(base, alg, f'lmda_f-{lmda}', f'seed-{seed}', 'train_log.csv')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            rows = list(_csv.DictReader(f))
        if col not in rows[0]:
            continue
        out.append(np.mean([float(r[col]) for r in rows[-n_last:]]))
    return float(np.mean(out)) if out else float('nan')


def main():
    for lmda in LMDAS:
        print(f'\n=== Adult, lambda={lmda}, critic_lr=0.01 (test metrics, mean +- sd over healthy seeds) ===')
        header = f"{'cell':>14} | {'n':>2} | {'acc':>15} | {'ap':>15} | {'dGDP':>15} | {'inf_gdp':>15} | {'sup_ipm':>15} | {'train ipm':>9} | {'s*_c std':>8}"
        print(header)
        print('-' * len(header))
        for label, base, alg in CELLS:
            ok, collapsed, missing = load_cell(base, alg, lmda)
            if not ok:
                note = f'collapsed={len(collapsed)} missing={len(missing)}'
                print(f'{label:>14} | {"0":>2} | {note}')
                continue
            cols = []
            for m in ['acc', 'ap', 'gdp_w_kernel', 'inf_gdp', 'sup_ipm']:
                mu, sd = stat(ok, m)
                cols.append(f'{mu:.4f}±{sd:.4f}')
            tr = train_tail(base, alg, lmda, ok.keys(), 'ipm_s_star')
            sstd = train_tail(base, alg, lmda, ok.keys(), 's_star_std')
            flag = ''
            if collapsed or missing:
                flag = f'   [collapsed={len(collapsed)} missing={len(missing)}]'
            print(f'{label:>14} | {len(ok):>2} | ' + ' | '.join(f'{c:>15}' for c in cols)
                  + f' | {tr:>9.5f} | {sstd:>8.4f}{flag}')
        print('  (train ipm = mean training ipm_s_star over the last 10 epochs; '
              's*_c std = mean train_log s_star_std, 0 in shared mode / absent in pre-s_mode runs)')


if __name__ == '__main__':
    main()
