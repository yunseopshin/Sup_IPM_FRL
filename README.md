# sup_IPM — Fair representation learning with supIPM (ReLU discriminator)

Training objective (encoder `h`, head `f`):

```
L = L_sup(f(h(X)), Y) + λ · sup_s ÎPM(h, s)
ÎPM(h, s) = sup_{v∈V} | Σ_i v(h(X_i)) K_h(s − S_i) / D_n(s) − (1/n) Σ_i v(h(X_i)) |
```

with `V` the paper-exact ReLU class `v(z̃) = (θᵀz̃ + μ)₊`, `θ ∈ S^{d−1}`, `μ ∈ [−1,1]`
(relu IPM.pdf Eq.(3)), evaluated on ball-normalized representations
`z̃ = z / √(‖z‖² + 1)` (`‖z̃‖ < 1`, so the discriminator domain is B^d as the theory
requires). `z̃` IS the representation: it feeds the prediction head AND every IPM
computation (critic ascent, sup over s, encoder penalty, test-time sup) — one
parameter-free normalization added on top of FREM's architecture, so the λ=0 baseline
is FREM's unfair model plus this normalization (checked to match within seed noise).
The *experimental protocol* (datasets, preprocessing, splits, metrics, final-epoch
evaluation) follows FREM (`../FRL-GDP-full`, paper `FRL_EIPM.pdf`); the *discriminator
optimization scheme* (alternating structure, steps, lrs, grad-clip) follows ReLU-IPM
(`../ReLUIPM-FRL_full`, `alg='reluipm'`), with the constraint set deliberately replaced
by the paper class above — see `notes/relu_ipm_normalization_check.md` and
`notes/supervised_loss_check.md`.

Per batch (λ > 0): melt critic pool → `critic_step`× [find s* by projected GA
(`src/sup_s.py`: K batched restarts + coarse-grid seed, s* ∈ [q_α(S), q_{1−α}(S)]) →
ascend the summed pool objective at s* (grad-clip 5.0, Adam, θ row-normalization onto
S^{d−1} + μ clamp to [−1,1])] →
freeze → encoder penalized by the pool max at the re-found s* (s* treated as constant).
S is standardized by train mean/std, so `bandwidth` is in units of std(S).

## Run

Everything goes through Slurm (see `CLAUDE.md`). One run:

```bash
scripts/run_single.sh --dataset Crime --lmda_f 1.0 --seed 2023
```

Sweep (one `srun --gres=gpu:1` job per (dataset, λ, seed), backgrounded, then `wait`):

```bash
DATASETS="Crime" LMDAS="0.0 0.1 1.0" SEEDS="2023 2024" scripts/run_sweep.sh
```

Re-evaluate a finished run (recomputes `results.json` + `sup_ipm_curve.npz`):

```bash
srun --gres=gpu:1 --cpus-per-task=10 --partition=idea --time=1:00:00 \
  /usr/local/miniconda3/envs/nine/bin/python src/evaluate.py --run_dir results/Crime-racepctblack/lmda_f-1.0/seed-2023
```

Outputs per run (`results/<dataset>-<S>/lmda_f-<λ>/seed-<seed>/`; `--mini` runs get a
separate `<dataset>-<S>-mini/` tree): `config.yaml`
(resolved), `train_log.csv` (per epoch: `task_loss`, `ipm_s_star`, `s_star_mean`,
`ga_max_mean`, `grid_max_mean`, `violations` — the GD-vs-grid sanity check), `model.pt`,
`results.json` (FREM metrics on train/val/test — plus `inf_gdp`/`inf_gdp_s`, the
L∞ generalized DP `sup_s |E[ŷ|S=s] − E[ŷ]|` over the trimmed train-S range, same NW
estimator as ΔGDP — and `sup_ipm`, `s_star_*` on val/test),
`sup_ipm_curve.npz` (the per-s maximized ÎPM curves).

## Config options (`configs/default.yaml`)

| key | meaning | default |
|---|---|---|
| `dataset` | `Crime` (reg, S=racepctblack) or `Adult` (cls, S=age); FREM presets fill `source/target/task/sensitive_attr/batch_size` | `Crime` |
| `alg` | `supipm` (ours) or a FREM-verbatim baseline on the SAME unified representer: `frem` (EIPM, `src/baselines.py::frlgdp_loss`), `reg_gdp` (prediction-level kernel GDP), `adv` (FREM's adversarial baseline, as-run sign convention). Run dirs get an `<alg>` path segment (`-raw`/`-ball` suffixes for non-default toggles) | `supipm` |
| `z_norm` | `true`: representation `z̃ = z/√(‖z‖²+1)` feeds head + fairness term (unified-representer comparison); `false`: raw `z` everywhere — the ORIGINAL FREM-style option | `true` |
| `critic_proj` | supipm discriminator constraint: `sphere_box` (paper Eq.(3): θ∈S^{d−1}, μ∈[−1,1]) or `ball` — the ORIGINAL joint (θ,μ) ℓ₂-ball option (applies to training pool and eval sup) | `sphere_box` |
| `gamma_rep`, `gamma_s` | `frem` only: RBF bandwidth on representations / kernel bandwidth on raw-scale S (paper: 1.0 / {0.03,0.05,0.07}) | 1.0, 0.05 |
| `seed` | FREM seeds 2023–2027 (Crime test fold = seed−2022) | 2023 |
| `epochs`, `lr`, `weight_decay`, `betas`, `batch_size` | FREM supervised part | 200, 1e-3, 1e-2, (0.5, 0.999), preset |
| `lmda_f` | λ (0 = FREM unfair baseline; fairness machinery skipped) | 0.0 |
| `bandwidth` | kernel h in units of std(S) | 0.25 |
| `alpha` | s-range trim quantile for the sup | 0.05 |
| `K`, `n_s_steps`, `lr_s` | GA restarts / steps / step size for sup over s | 8, 20, 0.1 |
| `n_grid`, `grid_tol` | coarse seeding grid + sanity tolerance | 33, 1e-6 |
| `critic_num`, `critic_step`, `critic_lr` | ReLU-IPM discriminator pool (as-run sweep values) | 100, 2, 1e-3 |
| `eval_grid`, `eval_restarts`, `eval_steps`, `eval_lr` | test-time sup_s ÎPM (fine grid over the TRAIN-split trimmed s-range, per-s ReLU-class ascent, warm-started from the trained pool; at λ=0 one extra random restart keeps the budget equal) | 401, 4, 400, 1e-2 |

CLI overrides exist for the common knobs (`src/train.py --help`).
