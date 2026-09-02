"""Side-by-side aggregation of the s_mode comparison grid (PERCRITIC_S.md sec.5).

Cells: (critic_step 2 | 20) x (s_mode shared | percritic), FREM seeds 2023..2027,
Crime / lmda_f=1.0, produced by scripts/run_smode_compare.sh into
results/smode_compare/cs<step>/.

Usage: python scripts/aggregate_smode_compare.py [--root results/smode_compare]
"""
import argparse
import csv
import json
import os
import statistics as stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
CELLS = [('shared', 2), ('percritic', 2), ('shared', 20), ('percritic', 20)]
SEEDS = [2023, 2024, 2025, 2026, 2027]
EPOCHS = [1, 5, 10, 20, 50, 100, 150, 200]


def run_dir(root, mode, cs, seed):
    sfx = '-percritic' if mode == 'percritic' else ''
    return os.path.join(root, f'cs{cs}', 'Crime-racepctblack', f'supipm{sfx}',
                        'lmda_f-1.0', f'seed-{seed}')


def load(root, mode, cs, seed):
    rd = run_dir(root, mode, cs, seed)
    log_path, res_path = os.path.join(rd, 'train_log.csv'), os.path.join(rd, 'results.json')
    if not (os.path.exists(log_path) and os.path.exists(res_path)):
        return None
    with open(log_path) as f:
        log = {int(r['epoch']): r for r in csv.DictReader(f)}
    with open(res_path) as f:
        res = json.load(f)
    return log, res


def msd(vals):
    """mean +- sd over seeds (sd omitted for a single seed)."""
    if not vals:
        return 'n/a'
    m = stats.mean(vals)
    if len(vals) == 1:
        return f'{m:.6g}'
    return f'{m:.6g} +- {stats.stdev(vals):.2g}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(ROOT, 'results', 'smode_compare'))
    args = ap.parse_args()
    root = args.root if os.path.isabs(args.root) else os.path.join(ROOT, args.root)

    data = {}
    for mode, cs in CELLS:
        for seed in SEEDS:
            got = load(root, mode, cs, seed)
            if got is not None:
                data[(mode, cs, seed)] = got
    names = [f'{m} cs{c}' for m, c in CELLS]
    have = {(m, c): [s for s in SEEDS if (m, c, s) in data] for m, c in CELLS}
    print('=== seeds available per cell ===')
    for (m, c), ss in have.items():
        print(f'  {m:>9} cs{c:<3}: {len(ss)} {ss}')

    print('\n=== training ipm_s_star trajectory (mean over seeds) ===')
    print(f"{'epoch':>6} | " + ' | '.join(f'{n:>16}' for n in names))
    for e in EPOCHS:
        cols = []
        for m, c in CELLS:
            vals = [float(data[(m, c, s)][0][e]['ipm_s_star']) for s in have[(m, c)]
                    if e in data[(m, c, s)][0]]
            cols.append(f'{stats.mean(vals):>16.6f}' if vals else f"{'n/a':>16}")
        print(f'{e:>6} | ' + ' | '.join(cols))

    print('\n=== s_star_std (spread of s*_c across the pool; 0 by definition in shared) ===')
    for e in EPOCHS:
        cols = []
        for m, c in CELLS:
            vals = [float(data[(m, c, s)][0][e]['s_star_std']) for s in have[(m, c)]
                    if e in data[(m, c, s)][0]]
            cols.append(f'{stats.mean(vals):>16.4f}' if vals else f"{'n/a':>16}")
        print(f'{e:>6} | ' + ' | '.join(cols))

    print('\n=== per-cell summary (mean +- sd over seeds) ===')
    rows = {}
    for m, c in CELLS:
        ss = have[(m, c)]
        tail = [stats.mean([float(data[(m, c, s)][0][e]['ipm_s_star'])
                            for e in range(191, 201)]) for s in ss]
        rows[(m, c)] = dict(
            train_ipm_final=tail,
            violations=[sum(int(r['violations']) for r in data[(m, c, s)][0].values()) for s in ss],
            train_sec=[sum(float(r['sec']) for r in data[(m, c, s)][0].values()) for s in ss],
            test_sup_ipm=[data[(m, c, s)][1]['test']['sup_ipm'] for s in ss],
            test_inf_gdp=[data[(m, c, s)][1]['test']['inf_gdp'] for s in ss],
            test_mse=[data[(m, c, s)][1]['test']['mse'] for s in ss],
            test_mae=[data[(m, c, s)][1]['test']['mae'] for s in ss],
            test_gdp_w=[data[(m, c, s)][1]['test']['gdp_w_kernel'] for s in ss],
            val_sup_ipm=[data[(m, c, s)][1]['val']['sup_ipm'] for s in ss],
        )
    metrics = ['train_ipm_final', 'test_sup_ipm', 'val_sup_ipm', 'test_inf_gdp',
               'test_mse', 'test_mae', 'test_gdp_w', 'violations', 'train_sec']
    print(f"{'metric':>16} | " + ' | '.join(f'{n:>22}' for n in names))
    for k in metrics:
        print(f'{k:>16} | ' + ' | '.join(f'{msd(rows[(m, c)][k]):>22}' for m, c in CELLS))

    print('\n=== critic lag (test sup_ipm / final training ipm_s_star) ===')
    for m, c in CELLS:
        tr, te = rows[(m, c)]['train_ipm_final'], rows[(m, c)]['test_sup_ipm']
        if not tr:
            continue
        print(f'{m:>9} cs{c:<3}: train={stats.mean(tr):.6f}  test={stats.mean(te):.6f}  '
              f'ratio={stats.mean(te) / max(stats.mean(tr), 1e-12):.1f}x')


if __name__ == '__main__':
    main()
