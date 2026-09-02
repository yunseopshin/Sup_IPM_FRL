"""Numerical equivalence check: s_mode=shared vs s_mode=percritic (PERCRITIC_S.md sec.4).

The two modes compute the SAME penalty value (the sups over s and over the
critic pool commute); only the discriminator ascent target differs. This script
demonstrates that numerically, on

  (a) a random batch (z = normalize_reps(randn), synthetic S),
  (b) a real Crime batch (B=200, fresh-init critic after project_critics),
  (c) the same real batch with a TRAINED critic (multimodal case) — checks 1-3.

Checks (tol = 1e-5 unless noted, tol_ga = 1e-3):
  0. sup commutes on the fine grid (exact float equality, by construction);
  1. both GAs reach the fine-grid max, and each other (values);
  2. argmax consistency (same s*, or a legitimate tie between s-modes);
  3. encoder gradient identical under both modes (Danskin) — skipped on a tie;
  4. per-critic ascent target dominates the shared one (the INTENDED difference,
     reported, not a failure) + spread/histogram of s*_c. Cases (a), (b) only.

Run once via srun (see CLAUDE.md), env `nine`:
  srun --gres=gpu:1 --cpus-per-task=10 --partition=idea --time=0:30:00 \
    /usr/local/miniconda3/envs/nine/bin/python tests/check_s_mode.py
"""
import argparse
import os
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from data import apply_preset, get_loaders  # noqa: E402
from models import build_model, ReLUCritic, project_critics, normalize_reps, REP_DIM  # noqa: E402
from ipm import kernel_weights, pool_ipm, kernel_weights_percritic, percritic_ipm  # noqa: E402
from sup_s import find_s_star  # noqa: E402
from sup_s_percritic import find_s_star_percritic  # noqa: E402

TOL = 1e-5
TOL_GA = 1e-3
H = 0.25
SUP_KW = dict(K=8, n_s_steps=20, lr_s=0.1, n_grid=33, grid_tol=1e-6)
GEN_SEED = 777  # restart draws inside find_s_star*; re-seeded before every call

failures = []


def check(case, name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        failures.append(f'{case}/{name}')


def fresh_critic(device, seed):
    torch.manual_seed(seed)
    critic = ReLUCritic(REP_DIM, 100).to(device)
    project_critics(critic)
    return critic


def delta_percritic_at(s_vec, s_std, V):
    """Delta_c evaluated at that critic's own s: s_vec [C] -> [C]."""
    return percritic_ipm(kernel_weights_percritic(s_vec.view(1, -1), s_std, H), V)[0]


def run_case(case, z_det, s_std, s_lo, s_hi, critic, run_check4=True):
    print(f'== case: {case} (B={z_det.size(0)}, s-range=[{s_lo:.3f}, {s_hi:.3f}]) ==', flush=True)
    device = z_det.device
    with torch.no_grad():
        V = critic(z_det)                                              # [B, C]

        # --- check 0: ground truth + "sup commutes" on a finite set (exact) ---
        G_fine = torch.linspace(s_lo, s_hi, 4001, device=device)
        M = pool_ipm(kernel_weights(G_fine, s_std, H), V)              # [G, C]
        m_all = M.max().item()
        m_rows = M.max(dim=1).values.max().item()
        m_cols = M.max(dim=0).values.max().item()
    check(case, '0 sup commutes (exact)', m_all == m_rows == m_cols,
          f'M.max()={m_all:.8f} rows-first={m_rows:.8f} cols-first={m_cols:.8f}')

    gen = torch.Generator().manual_seed(GEN_SEED)
    s_sh, diag_sh = find_s_star(z_det, s_std, critic, H, s_lo, s_hi, generator=gen, **SUP_KW)
    gen = torch.Generator().manual_seed(GEN_SEED)
    s_star_c, diag_pc = find_s_star_percritic(z_det, s_std, critic, H, s_lo, s_hi,
                                              generator=gen, **SUP_KW)
    v_sh, v_pc = diag_sh['ipm'], diag_pc['ipm']

    # --- check 1: values ---
    # The 20-fixed-lr-step training GA can sit a few 1e-5 below a 4001-point
    # fine grid from polish error alone; that is not a missed s-mode. When the
    # bound narrowly fails, escalate to a converged GA (n_s_steps=200) — only a
    # GA that is STILL below the fine grid then is genuinely stuck.
    def grid_bound(tag, finder, v20):
        if v20 >= m_all - TOL:
            check(case, f'1{tag} GA >= fine-grid max', True, f'v={v20:.8f} vs grid {m_all:.8f}')
            return
        kw = dict(SUP_KW, n_s_steps=200)
        gen2 = torch.Generator().manual_seed(GEN_SEED)
        _, d200 = finder(z_det, s_std, critic, H, s_lo, s_hi, generator=gen2, **kw)
        check(case, f'1{tag} GA >= fine-grid max (converged, 200 steps)',
              d200['ipm'] >= m_all - TOL,
              f'20-step v={v20:.8f} sits {m_all - v20:.2e} below grid {m_all:.8f} '
              f'(polish error); 200-step v={d200["ipm"]:.8f}')
    grid_bound('a shared', find_s_star, v_sh)
    grid_bound('b percritic', find_s_star_percritic, v_pc)
    stuck = 'shared' if v_sh < v_pc else 'percritic'
    check(case, '1c |v_shared - v_percritic| <= 1e-3', abs(v_sh - v_pc) <= TOL_GA,
          f'|{v_sh:.8f} - {v_pc:.8f}| = {abs(v_sh - v_pc):.2e}'
          + ('' if abs(v_sh - v_pc) <= TOL_GA else f' -> {stuck} GA is stuck below the other'))

    # --- check 2: argmax consistency ---
    with torch.no_grad():
        ipm_c = delta_percritic_at(s_star_c, s_std, V)                 # [C]
        c_star = int(ipm_c.argmax())
        delta_all_at_sh = pool_ipm(kernel_weights(s_sh.view(1), s_std, H), V)[0]  # [C]
    check(case, '2a Delta_{c*}(s*_{c*}) == v_pc', abs(ipm_c[c_star].item() - v_pc) <= TOL,
          f'recomputed {ipm_c[c_star].item():.8f} vs diag {v_pc:.8f} (c*={c_star})')
    s_gap = abs(s_star_c[c_star].item() - s_sh.item())
    tie = s_gap > 1e-2
    val_at_sh = delta_all_at_sh[c_star].item()
    check(case, '2b same s* (or value-tie between s-modes)',
          (not tie) or abs(val_at_sh - v_pc) <= TOL_GA,
          f'|s*_c* - s*_sh| = {s_gap:.4f}' +
          ('' if not tie else f' (distinct modes); |Delta_c*(s*_sh) - v_pc| = {abs(val_at_sh - v_pc):.2e}'))

    # --- check 3: encoder gradient (Danskin) ---
    if tie:
        print(f'  [SKIP] 3 encoder gradient: tie between distinct s-modes '
              f'(s gap {s_gap:.4f}) -> gradients legitimately differ', flush=True)
    else:
        rep = z_det.clone().requires_grad_(True)
        gen = torch.Generator().manual_seed(GEN_SEED)
        s_star, _ = find_s_star(rep.detach(), s_std, critic, H, s_lo, s_hi, generator=gen, **SUP_KW)
        fair_sh = pool_ipm(kernel_weights(s_star.view(1), s_std, H), critic(rep)).max()
        (g_sh,) = torch.autograd.grad(fair_sh, rep)

        rep2 = z_det.clone().requires_grad_(True)
        gen = torch.Generator().manual_seed(GEN_SEED)
        s_c2, _ = find_s_star_percritic(rep2.detach(), s_std, critic, H, s_lo, s_hi,
                                        generator=gen, **SUP_KW)
        w_c = kernel_weights_percritic(s_c2.view(1, -1), s_std, H)
        fair_pc = percritic_ipm(w_c, critic(rep2)).max()
        (g_pc,) = torch.autograd.grad(fair_pc, rep2)

        rel = ((g_sh - g_pc).norm() / g_sh.norm().clamp_min(1e-12)).item()
        check(case, '3 encoder grad rel diff <= 1e-4', rel <= 1e-4,
              f'rel={rel:.2e} (fair_sh={fair_sh.item():.8f} fair_pc={fair_pc.item():.8f}, '
              f's gap={s_gap:.2e})')

    # --- check 4: the INTENDED difference in the ascent target ---
    if run_check4:
        with torch.no_grad():
            sum_pc = ipm_c.sum().item()                # sum_c Delta_c(s*_c)
            sum_sh = delta_all_at_sh.sum().item()      # sum_c Delta_c(s*_shared)
            hist = torch.histc(s_star_c.float(), bins=10, min=s_lo, max=s_hi)
        print(f'  [info] 4 s_star_c.std()={diag_pc["s_star_std"]:.4f}  '
              f'histogram over [{s_lo:.3f}, {s_hi:.3f}] (10 bins): '
              f'{[int(x) for x in hist]}', flush=True)
        print(f'  [info] 4 ascent targets: sum_c Delta_c(s*_c)={sum_pc:.6f} vs '
              f'sum_c Delta_c(s*_shared)={sum_sh:.6f} (percritic dominates by design)', flush=True)
        check(case, '4 per-critic sum dominates', sum_pc >= sum_sh - TOL,
              f'{sum_pc:.6f} >= {sum_sh:.6f} - tol')
    print(flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default=os.path.join(ROOT, 'results', 'Crime-racepctblack',
                                             'supipm', 'lmda_f-1.0', 'seed-2023'),
                        help='finished shared-mode run supplying the trained critic')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Info] device={device}', flush=True)

    # --- case (a): random batch ---------------------------------------------
    g = torch.Generator().manual_seed(2023)
    z = torch.randn(200, REP_DIM, generator=g)
    s_raw = torch.rand(200, generator=g)               # synthetic S on [0, 1]
    s_std_cpu = (s_raw - s_raw.mean()) / s_raw.std()
    s_lo = torch.quantile(s_std_cpu, 0.05).item()
    s_hi = torch.quantile(s_std_cpu, 0.95).item()
    run_case('random', normalize_reps(z).to(device), s_std_cpu.to(device),
             s_lo, s_hi, fresh_critic(device, seed=1))

    # --- case (b): real Crime batch, fresh critic ---------------------------
    with open(os.path.join(ROOT, 'configs', 'default.yaml')) as f:
        cfg = apply_preset(yaml.safe_load(f))
    cfg['seed'] = 2023
    data = get_loaders(cfg)
    torch.manual_seed(cfg['seed'])                     # deterministic shuffle
    inputs, _, sens = next(iter(data['train']))
    inputs = inputs.to(device).view(-1, data['input_dim'])
    s_std = ((sens.to(device).float() - data['s_mean']) / data['s_sd'])
    model = build_model(data['input_dim'], data['task']).to(device)
    with torch.no_grad():
        z_det = normalize_reps(model.extractor(inputs))
    run_case('crime-fresh', z_det, s_std, data['s_lo'], data['s_hi'],
             fresh_critic(device, seed=2))

    # --- case (c): real batch, TRAINED critic (multimodal), checks 1-3 ------
    ckpt_path = os.path.join(args.run_dir, 'model.pt')
    if not os.path.exists(ckpt_path):
        print(f'[warn] no trained checkpoint at {ckpt_path} — skipping case (c)', flush=True)
    else:
        with open(os.path.join(args.run_dir, 'config.yaml')) as f:
            run_cfg = apply_preset(yaml.safe_load(f))
        ckpt = torch.load(ckpt_path, map_location=device)
        model_t = build_model(data['input_dim'], data['task']).to(device)
        model_t.load_state_dict(ckpt['model'])
        critic_t = ReLUCritic(REP_DIM, run_cfg['critic_num']).to(device)
        critic_t.load_state_dict(ckpt['critic'])
        with torch.no_grad():
            z_det_t = normalize_reps(model_t.extractor(inputs))
        run_case('crime-trained', z_det_t, s_std, data['s_lo'], data['s_hi'],
                 critic_t, run_check4=False)

    if failures:
        print(f'== OVERALL: FAIL ({len(failures)}) — ' + ', '.join(failures), flush=True)
        sys.exit(1)
    print('== OVERALL: PASS ==', flush=True)


if __name__ == '__main__':
    main()
