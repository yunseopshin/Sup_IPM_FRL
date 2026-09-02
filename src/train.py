"""supIPM fair representation learning: training loop.

Objective:  L = L_sup(f(h(X)), Y) + lambda * sup_s IPM_hat(h, s)

- Supervised part (loss, networks, optimizer, batching, epochs, data, splits,
  evaluation) follows FREM (FRL-GDP-full/src/train.py train_dp) — see
  notes/supervised_loss_check.md.
- Discriminator optimization follows ReLU-IPM (ReLUIPM-FRL_full/trainer.py,
  alg='reluipm'): per batch, melt -> critic_step ascent steps on the summed pool
  objective (grad-clip 5.0, Adam, then theta row-normalized onto S^{d-1} + mu
  clamped to [-1,1] — the paper Eq.(3) class, models.project_critics) -> freeze ->
  encoder penalized by the max over the pool. The representation is
  ball-normalized (models.normalize_reps) and feeds BOTH the head and the IPM.
  Only the IPM target changes vs ReLU-IPM: marginal
  two-group difference -> conditional-vs-marginal difference at the adversarially
  found s* (sup over s by projected gradient ascent, sup_s.py).
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data import apply_preset, get_loaders
from models import build_model, ReLUCritic, project_critics, normalize_reps, REP_DIM
from ipm import kernel_weights, pool_ipm, kernel_weights_percritic, percritic_ipm
from sup_s import find_s_star
from sup_s_percritic import find_s_star_percritic
from baselines import frlgdp_loss, reg_gdp_loss, build_adv_critic
import evaluate


def rep_of(model, x, znorm):
    """The unified representation: ball-normalized z_tilde when z_norm is on,
    raw extractor output when off (the original FREM-style option)."""
    z = model.extractor(x)
    return normalize_reps(z) if znorm else z

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(ROOT, 'configs', 'default.yaml'))
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['Crime', 'Adult', 'SynthB'])
    parser.add_argument('--lmda_f', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--bandwidth', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--n_s_steps', type=int, default=None)
    parser.add_argument('--lr_s', type=float, default=None)
    parser.add_argument('--critic_num', type=int, default=None)
    parser.add_argument('--critic_step', type=int, default=None)
    parser.add_argument('--critic_lr', type=float, default=None)
    parser.add_argument('--s_mode', type=str, default=None,
                        choices=['shared', 'percritic'])
    parser.add_argument('--alg', type=str, default=None,
                        choices=['supipm', 'frem', 'reg_gdp', 'adv'])
    parser.add_argument('--z_norm', type=str, default=None, choices=['true', 'false'])
    parser.add_argument('--critic_proj', type=str, default=None,
                        choices=['sphere_box', 'ball'])
    parser.add_argument('--gamma_rep', type=float, default=None)
    parser.add_argument('--gamma_s', type=float, default=None)
    parser.add_argument('--synthb_s0', type=float, default=None,
                        help='SynthB only: centre of the leak bump, in sd(S) units')
    parser.add_argument('--synthb_s_dist', type=str, default=None,
                        choices=['normal', 'uniform'],
                        help='SynthB only: latent S distribution (both mean 0, sd 1)')
    parser.add_argument('--eval_steps', type=int, default=None)
    parser.add_argument('--eval_grid', type=int, default=None)
    parser.add_argument('--mini', action='store_true')
    parser.add_argument('--out_root', type=str, default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for key, val in vars(args).items():
        if key in ('config', 'force'):
            continue
        if key == 'mini':
            if val:
                cfg['mini'] = True
            continue
        if key == 'z_norm':
            if val is not None:
                cfg['z_norm'] = (val == 'true')
            continue
        if val is not None:
            cfg[key] = val
    cfg = apply_preset(cfg)
    return cfg, args.force


def make_run_dir(cfg):
    out_root = cfg.get('out_root') or 'results'
    if not os.path.isabs(out_root):
        out_root = os.path.join(ROOT, out_root)
    tag = f"{cfg['dataset']}-{cfg['sensitive_attr']}" + ('-mini' if cfg.get('mini') else '')
    alg_tag = cfg.get('alg', 'supipm')
    if alg_tag == 'frem':  # frem sweeps gamma too — keep runs from colliding
        alg_tag += f"-gr{cfg.get('gamma_rep', 1.0)}-gs{cfg.get('gamma_s', 0.05)}"
    if not cfg.get('z_norm', True):
        alg_tag += '-raw'
    if cfg.get('critic_proj', 'sphere_box') == 'ball':
        alg_tag += '-ball'
    if cfg.get('s_mode', 'shared') == 'percritic':
        alg_tag += '-percritic'
    run_dir = os.path.join(out_root, tag, alg_tag,
                           f"lmda_f-{cfg['lmda_f']}", f"seed-{cfg['seed']}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    cfg, force = parse_config()
    run_dir = make_run_dir(cfg)
    if os.path.exists(os.path.join(run_dir, 'results.json')) and not force:
        print(f'[Info] Already done: {run_dir}')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Info] device={device} run_dir={run_dir}', flush=True)
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # FREM order: load data (np seeded inside), then seed torch and build networks
    torch.set_num_threads(4)
    random.seed(cfg['seed'])
    data = get_loaders(cfg)
    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    task, input_dim = data['task'], data['input_dim']
    lmda = float(cfg['lmda_f'])
    alg = cfg.get('alg', 'supipm')
    if alg not in ('supipm', 'frem', 'reg_gdp', 'adv'):
        raise ValueError(f'unknown alg: {alg}')
    znorm = bool(cfg.get('z_norm', True))
    proj = cfg.get('critic_proj', 'sphere_box')
    s_mode = cfg.get('s_mode', 'shared')
    if s_mode not in ('shared', 'percritic'):
        raise ValueError(f'unknown s_mode: {s_mode}')
    print(f"[Info] alg={alg} z_norm={znorm} critic_proj={proj} s_mode={s_mode}", flush=True)
    h = float(cfg['bandwidth'])
    s_mean, s_sd = data['s_mean'], data['s_sd']
    s_lo, s_hi = data['s_lo'], data['s_hi']
    print(f"[Info] S stats: mean={s_mean:.4f} std={s_sd:.4f} "
          f"trimmed std-range=[{s_lo:.3f}, {s_hi:.3f}] h={h} (x std(S))", flush=True)

    model = build_model(input_dim, task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                 weight_decay=cfg['weight_decay'],
                                 betas=tuple(cfg['betas']))
    critic, critic_optimizer = None, None
    if lmda > 0.0 and alg == 'supipm':
        critic = ReLUCritic(REP_DIM, cfg['critic_num']).to(device)
        # ReLU-IPM critic optimizer: plain Adam, default betas, no weight decay
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr'])
    elif lmda > 0.0 and alg == 'adv':
        critic = build_adv_critic(REP_DIM).to(device)
        # FREM builds the adv critic optimizer with the model's hyperparameters
        # (FRL-GDP-full/src/train.py:631-635)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg['lr'],
                                            weight_decay=cfg['weight_decay'],
                                            betas=tuple(cfg['betas']))

    # dedicated CPU generator for the s-restart draws: keeps the global (shuffle)
    # RNG stream untouched on any device and makes runs device-portable
    s_rng = torch.Generator().manual_seed(cfg['seed'] + 1)
    sup_kwargs = dict(K=cfg['K'], n_s_steps=cfg['n_s_steps'], lr_s=cfg['lr_s'],
                      n_grid=cfg['n_grid'], grid_tol=cfg['grid_tol'], generator=s_rng)

    log_path = os.path.join(run_dir, 'train_log.csv')
    log_file = open(log_path, 'w', newline='')
    log = csv.writer(log_file)
    log.writerow(['epoch', 'task_loss', 'ipm_s_star', 's_star_mean', 's_star_last',
                  's_star_std', 'ga_max_mean', 'grid_max_mean', 'violations', 'sec'])

    t_start = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        task_sum, fair_sum, n_samp = 0.0, 0.0, 0
        s_sum, ga_sum, grid_sum, n_batch, n_viol = 0.0, 0.0, 0.0, 0, 0
        s_last, sstd_sum = 0.0, 0.0

        for inputs, labels, sensitives in data['train']:
            inputs = inputs.to(device).view(-1, input_dim)
            labels = labels.to(device)
            labels = labels.long() if task == 'cls' else labels.float()
            sens = sensitives.to(device)
            B = inputs.size(0)

            if lmda > 0.0 and alg == 'supipm':
                s_std = (sens.float() - s_mean) / s_sd
                # --- discriminator ascent (ReLU-IPM scheme, conditional target) ---
                critic.melt()
                for _ in range(cfg['critic_step']):
                    with torch.no_grad():
                        z_det = rep_of(model, inputs, znorm)
                    if s_mode == 'shared':
                        s_star, _ = find_s_star(z_det, s_std, critic, h, s_lo, s_hi, **sup_kwargs)
                        w = kernel_weights(s_star.view(1), s_std, h)                    # [1, B]
                        critic_loss = -pool_ipm(w, critic(z_det)).sum()
                    else:
                        s_star_c, _ = find_s_star_percritic(z_det, s_std, critic, h, s_lo, s_hi, **sup_kwargs)
                        w_c = kernel_weights_percritic(s_star_c.view(1, -1), s_std, h)  # [1, C, B]
                        critic_loss = -percritic_ipm(w_c, critic(z_det)).sum()
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
                    critic_optimizer.step()
                    project_critics(critic, proj)
                critic.freeze()

            # --- encoder/head update (FREM supervised part; head consumes the
            #     same unified representation as the fairness term) ---
            model.train()
            rep = rep_of(model, inputs, znorm)
            preds = model.final_layer(rep)
            if task == 'cls':
                probs = torch.softmax(preds, dim=1)[:, 1].flatten()
                task_loss = F.cross_entropy(preds, labels)
            elif task == 'reg':
                probs = torch.sigmoid(preds).flatten()
                task_loss = F.mse_loss(probs, labels.flatten())

            fair_loss = torch.zeros((), device=device)
            diag = None
            if lmda > 0.0:
                if alg == 'supipm':
                    # s* / s*_c are constants for the encoder step in both modes (Danskin)
                    if s_mode == 'shared':
                        s_star, diag = find_s_star(rep.detach(), s_std, critic, h, s_lo, s_hi, **sup_kwargs)
                        w = kernel_weights(s_star.view(1), s_std, h)
                        fair_loss = pool_ipm(w, critic(rep)).max()
                    else:
                        s_star_c, diag = find_s_star_percritic(rep.detach(), s_std, critic, h, s_lo, s_hi, **sup_kwargs)
                        w_c = kernel_weights_percritic(s_star_c.view(1, -1), s_std, h)
                        fair_loss = percritic_ipm(w_c, critic(rep)).max()
                elif alg == 'frem':
                    fair_loss = frlgdp_loss(rep, sens, gamma_rep=cfg['gamma_rep'],
                                            gamma_s=cfg['gamma_s'])
                elif alg == 'reg_gdp':
                    fair_loss = reg_gdp_loss(probs, sens)
                elif alg == 'adv':
                    fair_loss = F.mse_loss(torch.sigmoid(critic(rep)).flatten(),
                                           sens.float().flatten())

            loss = task_loss + lmda * fair_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if lmda > 0.0 and alg == 'adv':
                # FREM ADV critic updates, sign convention as-run
                # (FRL-GDP-full/src/train.py:775-785: critic ascends its own MSE)
                for _ in range(5):
                    with torch.no_grad():
                        rep_det = rep_of(model, inputs, znorm)
                    critic_probs = torch.sigmoid(critic(rep_det)).flatten()
                    critic_loss = - F.mse_loss(critic_probs, sens.float().flatten())
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

            task_sum += task_loss.item() * B
            fair_sum += fair_loss.item() * B
            n_samp += B
            n_batch += 1
            if diag is not None:
                s_sum += diag['s_star']
                s_last = diag['s_star']
                sstd_sum += diag.get('s_star_std', 0.0)  # spread of s*_c (0.0 in shared mode)
                ga_sum += diag['ga_max']
                grid_sum += diag['grid_max']
                n_viol += int(diag['violation'])

        nb = max(n_batch, 1)
        row = [epoch, task_sum / n_samp, fair_sum / n_samp, s_sum / nb, s_last,
               sstd_sum / nb, ga_sum / nb, grid_sum / nb, n_viol, round(time.time() - t0, 2)]
        log.writerow([f'{v:.6f}' if isinstance(v, float) else v for v in row])
        log_file.flush()
        if epoch == 1 or epoch % 10 == 0:
            print(f"[{epoch}/{cfg['epochs']}] task={row[1]:.4f} ipm(s*)={row[2]:.4f} "
                  f"s*={row[3]:.3f} grid_max={row[6]:.4f} viol={n_viol}", flush=True)

    log_file.close()
    print(f'[Info] training done in {time.time() - t_start:.1f}s', flush=True)

    # only the supipm ReLU pool is a member of the eval discriminator class
    pool = critic if alg == 'supipm' else None
    torch.save({'model': model.state_dict(),
                'critic': pool.state_dict() if pool is not None else None},
               os.path.join(run_dir, 'model.pt'))

    results = evaluate.run_eval(model, pool, data, cfg, device, run_dir=run_dir)
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    key = 'acc' if task == 'cls' else 'mae'
    print(f"[Done] test {key}={results['test'][key]} "
          f"gdp_w_kernel={results['test']['gdp_w_kernel']} "
          f"sup_ipm={results['test']['sup_ipm']}", flush=True)


if __name__ == '__main__':
    main()
