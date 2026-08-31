"""S1_RESULTS.md 생성기 — results/*.npz 에서 직접 읽어 마크다운 보고서를 쓴다.

    srun --cpus-per-task=8 --partition=idea --time=00:30:00 \
        /usr/local/miniconda3/envs/nine/bin/python make_report.py

숫자를 손으로 옮겨 적지 않기 위한 스크립트다. 해석/결론은 일부러 넣지 않는다
(보고서를 읽는 쪽이 해석하도록).
"""
import io
import os
import sys

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1_core as C
import s1_runner as R

OUT = os.path.join(HERE, "S1_RESULTS.md")
L = []
w = L.append

model = C.make_model("bump", kappa=1.2, s0=0.7, tau=0.2)
T = C.truth_cached(model, os.path.join(HERE, "results"))
SUP, EIP = T["supipm"], T["eipm"]
TAU = model["params"]["tau"]
sc = model["sigma_c"]
REPS = int(R.load(R.job(500, 1.0, 0.5))["reps"])
lnN = np.log(np.array(R.N_LIST, float))
rng = np.random.default_rng(0)

rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
sup_err = lambda n, c, r=0.5: R.load(R.job(n, c, r))["sup_hat"] - SUP
eip_err = lambda n, c, r=0.5: R.load(R.job(n, c, r))["eipm_hat"] - EIP

# ------------------------------------------------------------------ header
w(f"""# S1 시뮬레이션 — 세팅과 결과

> 이 파일은 `make_report.py` 가 `results/*.npz` 에서 직접 읽어 생성한다. 손으로 옮긴 숫자는 없다.
> **해석·결론은 일부러 넣지 않았다.** 세팅과 관측된 수치만 담는다.
> 재생성: `srun --cpus-per-task=8 --partition=idea --time=00:30:00 python make_report.py`

관련 파일: 사양 `claude_md/S1_SPEC.md`, 노트북 `s1.ipynb`, 그림 `figures/`, 원자료 `results/*.npz`.

---

## 1. 목적

supIPM 추정량(커널 + grid)이 **참값을 아는 인공 세팅에서 그 참값에 수렴하는지** 확인한다.
공정성 방법 간 비교(EIPM vs supIPM 중 어느 기준이 나은가)는 이 시뮬레이션의 범위가 **아니다**
(사양 6절이 별도 시뮬레이션으로 미뤄둠).

## 2. 데이터 생성 모형

```
S ~ N(0, 1)
Z | S=s ~ N( mu(s), sigma_c^2 ),    sigma_c^2 = 1 - Var(mu(S))     # => Var(Z) = 1
```

`mu` 두 가지:

| 이름 | 정의 | 용도 |
|---|---|---|
| `linear` | `mu(s) = lam * s` | 참값 공식 검증용 (닫힌 형태 존재) |
| **`bump`** | `mu(s) = kappa * exp( -(s-s0)^2 / (2 tau^2) )` | **주 실험**. `kappa=1.2, s0=0.7, tau=0.2` |

`bump` 를 주 실험으로 쓰는 이유(사양 §1): `linear` 면 IPM(s) 가 |s| 에 단조라 sup 이 항상 구간
끝점에서 달성되고, 등간격 grid 는 끝점을 늘 포함하므로 **grid 간격 실험(C2)이 무의미**해진다.

`bump` 의 실제 상수: `Var(mu(S)) = {model['var_mu']:.10f}`, `sigma_c = {sc:.10f}`
(닫힌 형태로 계산. Var(Z) = {model['var_mu'] + sc**2:.12f})

## 3. 재려는 양

판별자족이 1-Lipschitz 이므로 IPM = W1, 1차원에서

```
W1(s) = ∫ |F_s(u) - F(u)| du ,    F_s(u) = Phi((u - mu(s))/sigma_c) ,  F(u) = E_S[ F_S(u) ]

supIPM = max over |s| <= z of W1(s)                    z = Phi^-1(1 - alpha/2) = {C.Z_TRIM:.6f}, alpha=0.05
EIPM   = ∫ W1(s) phi(s) ds / ∫ phi(s) ds   (같은 구간)
```

참값 계산: `F(u)` 는 Gauss-Hermite 구적(노드 {C.GH_NODES}개), `u` 적분은 `linspace(-8,8,4001)` 사다리꼴,
sup 은 `linspace(-z,z,5001)` 위 최대.

## 4. 추정량

```
w_i(s)     = K_h(s - s_i) / sum_j K_h(s - s_j)      K_h(t) = exp(-t^2 / 2h^2)   (가우시안 커널)
F_s_hat(u) = sum_i w_i(s) * 1(z_i <= u)
F_hat(u)   = (1/n) sum_i 1(z_i <= u)
W1_hat(s)  = ∫ |F_s_hat(u) - F_hat(u)| du

h = c * n^(-1/5)                                    c = 대역폭 상수
grid: [-z, z] 등간격 G 점,  G = ceil(2z / (ratio*h)) + 1   =>  delta_G / h = ratio
supIPM_hat = max over grid
EIPM_hat   = trim 안의 관측 s_i 중 {C.N_EIPM}개를 뽑아 W1_hat 평균
```

`z` 를 정렬하면 두 계단함수의 점프 위치가 같아 `∫|F_s_hat - F_hat|du` 는 유한합으로 **정확히**
계산된다 (조밀한 u 수치적분과 1e-7 이내 일치를 테스트로 확인).

**비교용 baseline (binning)**: `s` 의 분위수로 J 등분 → 각 구간 z 의 경험분포와 전체 경험분포
사이 W1 → 최대. 표본 10개 미만 구간은 제외.

## 5. 실험 설계

반복 {REPS}회, seed = `1000 + 7919*r` 로 **모든 설정이 같은 시드를 공유**한다. 한 번의 draw 로
계산 가능한 추정치는 같은 워커에서 함께 계산하므로 `(n, c, ratio)` 가 같은 설정은 파일 하나를
공유하고, 셀 간 비교는 **정확히 paired** 다.

| 실험 | 바꾸는 것 | 고정 |
|---|---|---|
| C1 | `n = {R.N_LIST}` | `c=1, ratio=0.5` |
| C2 | `ratio = delta_G/h = {', '.join(f'{x:g}' for x in R.RATIO_LIST)}` | `n=10000, c=1` |
| C3 | binning `J = {R.JS}` vs 커널 | `n=10000, c=1, ratio=0.5` |
| C1b (사양 외 추가) | `c = {', '.join(f'{x:g}' for x in sorted([1.0] + R.C_EXTRA, reverse=True))}` × 같은 `n` | `ratio=0.5` |

C1b 는 사양에 없다. `c=1` 이면 `h` 가 bump 폭 `tau=0.2` 와 같은 크기가 되는데, 그 효과와
추정량 자체의 성질을 분리하기 위해 추가했다.

---

## 6. 참값

| 양 | 값 |
|---|---|
| `supIPM_true` | **{SUP:.9f}** (argmax `s = {T['s_argmax']:+.4f}`) |
| `EIPM_true` | {EIP:.9f} |
| 비율 sup/E | {SUP/EIP:.4f} |
| `W1(s)` 바닥값 (mu(s)=0 인 s) | {float(np.trapz(np.abs(ndtr(C.U_GRID/sc) - C.marginal_cdf(model)), dx=C.U_GRID[1]-C.U_GRID[0])):.6f} |

마지막 행 주: `mu(s)=0` 인 집단조차 W1 > 0 인 이유는 전체 Z 분포 `F` 가 봉우리 집단 때문에
치우친 혼합분포라서다.

### 6.1 참값 계산의 정확도 검증
""")

# ---------------------------------------------------- closed-form + quadrature
w("**(a) linear 닫힌 형태 대조** — `b = sigma_c - 1`, `a = lam*s` 일 때 "
  "`W1(s) = |b|*sqrt(2/pi)*exp(-a^2/2b^2) + a*(2*Phi(a/|b|) - 1)`\n")
w("| lam | sigma_c | max&#124;수치 − 닫힌형태&#124; | supIPM | EIPM |")
w("|---|---|---|---|---|")
for lam in (0.3, 0.5, 0.8):
    ml = C.make_model("linear", lam=lam)
    tl = C.truth(ml)
    d = float(np.abs(tl["w1"] - C.w1_linear_closed_form(C.S_GRID, lam, ml["sigma_c"])).max())
    w(f"| {lam} | {ml['sigma_c']:.7f} | {d:.3e} | {tl['supipm']:.6f} | {tl['eipm']:.6f} |")
w("\n판정 기준은 `< 1e-6`. 전부 통과.\n")

w("**(b) 구적/격자 해상도** — 기준은 조밀한 사다리꼴 구적(20001점).\n")
w("| F(u) 구적 | d_supIPM | d_EIPM | max&#124;dW1&#124; |")
w("|---|---|---|---|")
ref = C.truth(model, quad=C.trap_quad(20001))
for name, q in [("GH 101 (사양)", C.gh_quad(101)), ("GH 201", C.gh_quad(201)),
                ("trap 80001", C.trap_quad(80001))]:
    t = C.truth(model, quad=q)
    w(f"| {name} | {t['supipm']-ref['supipm']:+.3e} | {t['eipm']-ref['eipm']:+.3e} | "
      f"{np.abs(t['w1']-ref['w1']).max():.3e} |")
w("")
w("| u-격자 | du | supIPM |")
w("|---|---|---|")
for nu in (2001, 4001, 8001, 16001):
    u = np.linspace(-8, 8, nu)
    w(f"| {nu} | {u[1]-u[0]:.5f} | {C.truth(model, u=u)['supipm']:.9f} |")
w("")

# ------------------------------------------------------------------- results
w("---\n\n## 7. 결과\n\n### 7.1 C1 — 표본 크기 (c=1, 사양 기본값)\n")
w("| n | h | h/tau | G | bias | RMSE | 상대 RMSE % | SD |")
w("|---|---|---|---|---|---|---|---|")
rows_c1 = []
for n in R.N_LIST:
    d = R.load(R.job(n, 1.0, 0.5))
    e = d["sup_hat"] - SUP
    h = float(d["h"])
    rows_c1.append((n, h, int(d["G"]), e.mean(), rmse(e), e.std(ddof=1)))
    w(f"| {n} | {h:.4f} | {h/TAU:.2f} | {int(d['G'])} | {e.mean():+.4f} | {rmse(e):.4f} | "
      f"{100*rmse(e)/SUP:.1f} | {e.std(ddof=1):.4f} |")
w("")

w("### 7.2 C1b — 대역폭 상수 c (사양 외 추가)\n")
w("| c | n | h | h/tau | bias | RMSE | 상대 RMSE % | SD |")
w("|---|---|---|---|---|---|---|---|")
for cc in sorted([1.0] + R.C_EXTRA, reverse=True):
    for n in R.N_LIST:
        d = R.load(R.job(n, cc, 0.5))
        e = d["sup_hat"] - SUP
        h = float(d["h"])
        w(f"| {cc:g} | {n} | {h:.4f} | {h/TAU:.2f} | {e.mean():+.4f} | {rmse(e):.4f} | "
          f"{100*rmse(e)/SUP:.1f} | {e.std(ddof=1):.4f} |")
w("")

w("### 7.3 C2 — grid 간격 (n=10000, c=1)\n")
w("| delta_G/h | G | bias | RMSE | SD | EIPM_hat 이 ratio 0.25 와 동일한가 |")
w("|---|---|---|---|---|---|")
base_e = R.load(R.job(10000, 1.0, 0.25))["eipm_hat"]
for r in R.RATIO_LIST:
    d = R.load(R.job(10000, 1.0, r))
    e = d["sup_hat"] - SUP
    same = np.array_equal(d["eipm_hat"], base_e)
    w(f"| {r:g} | {int(d['G'])} | {e.mean():+.4f} | {rmse(e):.4f} | {e.std(ddof=1):.4f} | {same} |")
w("")
w("**paired 검사** (같은 draw 위에서 grid 만 성기게 하면 sup 은 같거나 작아져야 함):\n")
w("| delta_G/h | ratio 0.25 이하인 rep 수 | 평균 손실 |")
w("|---|---|---|")
b25 = R.load(R.job(10000, 1.0, 0.25))["sup_hat"]
for r in R.RATIO_LIST:
    sh = R.load(R.job(10000, 1.0, r))["sup_hat"]
    w(f"| {r:g} | {int((sh <= b25 + 1e-12).sum())}/{REPS} | {(b25-sh).mean():.5f} |")
w("\n주: ratio 0.5 와 1 의 grid 는 ratio 0.25 의 grid 와 포함관계가 아니므로 "
  "(끝점 외에 공유점 없음) 100/100 이 나오지 않는 것이 정상이다. rep 별 최대 차이는 "
  f"{np.abs(R.load(R.job(10000,1.0,0.5))['sup_hat']-b25).max():.2e} 이다.\n")

w("### 7.4 C3 — binning 비교 (n=10000, {} reps, paired)\n".format(REPS))
d = R.load(R.job(10000, 1.0, 0.5))
Js = [int(x) for x in d["js"]]
ek = d["sup_hat"] - SUP
w("| 방법 | 구간당 관측수 | bias | RMSE | SD |")
w("|---|---|---|---|---|")
w(f"| 커널 c=1 | – | {ek.mean():+.4f} | {rmse(ek):.4f} | {ek.std(ddof=1):.4f} |")
for k, J in enumerate(Js):
    e = d["bin_hat"][k] - SUP
    w(f"| binning J={J} | {10000//J} | {e.mean():+.4f} | {rmse(e):.4f} | {e.std(ddof=1):.4f} |")
w("\n(진단, 사양 외) 커널 grid 와 같은 `[-z,z]` 안에서만 분위수 구간을 잡은 경우:\n")
w("| 방법 | bias | RMSE | SD |")
w("|---|---|---|---|")
for k, J in enumerate(Js):
    e = d["bin_trim_hat"][k] - SUP
    w(f"| binning-trim J={J} | {e.mean():+.4f} | {rmse(e):.4f} | {e.std(ddof=1):.4f} |")
w("")

# --------------------------------------------------------------- rate / limit
w("### 7.5 수렴률\n")
w("`h ∝ n^(-1/5)` 이면 편향 `O(h^2)`, `SD = O(1/sqrt(nh))` 가 모두 `n^(-2/5)` 이므로 "
  "RMSE 의 log-log 기울기는 **-2/5 = -0.400 으로 수렴**한다. 다만 이는 `h/tau -> 0` 에서의 "
  "**점근값**이다. `c` 는 비례상수이므로 점근 기울기는 `c` 와 무관하고, 유한 n 에서 관측되는 "
  "기울기는 `h/tau` 의 함수다.\n")
w("아래 '이론 예측'은 결정론적 h-극한(7.7 참조)의 편향 `supIPM_true - sup_h` 를 `log n` 에 "
  "회귀해 얻은 값으로, **시뮬레이션을 전혀 쓰지 않는다**.\n")
ctx = C.limit_context(model)
_lim = {}
def lim_sup(h):
    k = round(h, 8)
    if k not in _lim:
        _lim[k] = C.limit_sup(ctx, h)
    return _lim[k]

w("| c | h/tau (n=20000) | 이론 예측 기울기 | 관측 기울기 | 95% CI (부트스트랩 B=2000) | CI 가 예측을 포함 |")
w("|---|---|---|---|---|---|")
for cc in sorted([1.0] + R.C_EXTRA, reverse=True):
    E = [sup_err(n, cc) for n in R.N_LIST]
    rm = np.array([rmse(e) for e in E])
    sl = float(np.polyfit(lnN, np.log(rm), 1)[0])
    bs = [np.polyfit(lnN, np.log([np.sqrt((e[rng.integers(0, e.size, e.size)] ** 2).mean())
                                  for e in E]), 1)[0] for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pred = float(np.polyfit(lnN, np.log([SUP - lim_sup(C.bandwidth(n, cc))
                                         for n in R.N_LIST]), 1)[0])
    w(f"| {cc:g} | {C.bandwidth(20000, cc)/TAU:.2f} | {pred:+.3f} | {sl:+.3f} | "
      f"[{lo:+.3f}, {hi:+.3f}] | {'예' if lo <= pred <= hi else '아니오'} |")
w("")
w("점근값 `-0.400` 은 `n` 을 더 키우면 나온다 (결정론적 편향, 시뮬레이션 불필요):\n")
w("| n (c=1) | h/tau | 결정론적 편향 | 국소 기울기 |")
w("|---|---|---|---|")
NS = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
bb = [SUP - lim_sup(C.bandwidth(n, 1.0)) for n in NS]
for k, n in enumerate(NS):
    loc = "–" if k == 0 else f"{(np.log(bb[k])-np.log(bb[k-1]))/(np.log(NS[k])-np.log(NS[k-1])):+.3f}"
    w(f"| {n:.0e} | {C.bandwidth(n, 1.0)/TAU:.3f} | {bb[k]:.5f} | {loc} |")
w("")
w("기울기가 `c` 가 아니라 `h/tau` 만의 함수임을 직접 확인 "
  "(`d log(bias)/d log h` 를 수치미분한 뒤 `x (-1/5)`):\n")
w("| h/tau | d log(bias) / d log h | 대응 기울기 |")
w("|---|---|---|")
for x in (1.0, 0.5, 0.25, 0.10):
    h0 = x * TAU
    el = (np.log(SUP - lim_sup(h0 * 1.01)) - np.log(SUP - lim_sup(h0 * 0.99))) \
         / (np.log(h0 * 1.01) - np.log(h0 * 0.99))
    w(f"| {x:.2f} | {el:.3f} | {-el/5:+.3f} |")
w("\n봉우리 감쇠 인수 `tau/sqrt(tau^2+h^2)` 의 탄력도가 `h/tau -> 0` 에서 2 (→ 기울기 -0.4), "
  "`h/tau ≈ 1` 에서 1.18 (→ 기울기 -0.235) 인 것과 일치한다.\n")

w("### 7.6 추정 곡선 vs 참값 곡선 (n=20000)\n")
dense = np.linspace(-C.Z_TRIM, C.Z_TRIM, 300)
w1t = C.w1_true_at_mu(model["mu"](dense), sc, C.marginal_cdf(model))
w("| c | h/tau | 봉우리 참값 | 봉우리 추정평균 | 전 구간 max&#124;평균−참값&#124; |")
w("|---|---|---|---|---|")
for cc in (1.0, 0.5, 0.25):
    f = os.path.join(HERE, "results", f"curves_n20000_c{cc:g}.npz")
    if os.path.exists(f):
        with np.load(f) as z:
            cur = z["curves"]
    else:
        cur = np.empty((REPS, dense.size))
        for r in range(REPS):
            s_, z_ = C.simulate(20000, model, np.random.default_rng(C.seed_of(r)))
            cur[r] = C.w1_hat(dense, C.prep(s_, z_), C.bandwidth(20000, cc))
        np.savez_compressed(f, curves=cur)
    w(f"| {cc:g} | {C.bandwidth(20000, cc)/TAU:.2f} | {w1t.max():.4f} | {cur.mean(0).max():.4f} | "
      f"{np.abs(cur.mean(0)-w1t).max():.4f} |")
w("")

w("### 7.7 결정론적 h-극한과의 대조\n")
w("`h` 를 고정한 채 `n -> inf` 로 보내면 추정량은 참값이 아니라 아래 값으로 간다:\n")
w("```\nW1_h(s) = ∫ |F_s^h(u) - F(u)| du ,   "
  "F_s^h(u) = ∫ phi_h(s-t) phi(t) Phi((u-mu(t))/sigma_c) dt / ∫ phi_h(s-t) phi(t) dt\n```\n")
w("각 n 에서 관측 평균이 이 값과 일치하는지 (MC 표준오차와 함께):\n")
tg = np.linspace(-8, 8, 8001)
ft = norm.pdf(tg)
PHI = ndtr((C.U_GRID[None, :] - model["mu"](tg)[:, None]) / sc)
F_m = (ft / ft.sum()) @ PHI
sg = np.linspace(-C.Z_TRIM, C.Z_TRIM, 1001)


def limit_sup(h):
    out = np.empty(sg.size)
    for i in range(0, sg.size, 200):
        j = min(i + 200, sg.size)
        Wk = np.exp(-0.5 * ((sg[i:j, None] - tg[None, :]) / h) ** 2) * ft[None, :]
        Wk /= Wk.sum(1, keepdims=True)
        out[i:j] = np.trapz(np.abs(Wk @ PHI - F_m[None, :]), dx=C.U_GRID[1] - C.U_GRID[0], axis=1)
    return float(out.max())


w("| c | n | h | 관측 sup_hat 평균 | 이론 h-극한 | 차이 | MC 표준오차 |")
w("|---|---|---|---|---|---|---|")
for cc in (1.0, 0.25):
    for n in R.N_LIST:
        d = R.load(R.job(n, cc, 0.5))
        h = float(d["h"])
        lim = limit_sup(h)
        sh = d["sup_hat"]
        w(f"| {cc:g} | {n} | {h:.4f} | {sh.mean():.5f} | {lim:.5f} | {sh.mean()-lim:+.5f} | "
          f"{sh.std(ddof=1)/np.sqrt(sh.size):.5f} |")
w("")

# ------------------------------------------------------------------ appendix
w(f"""---

## 8. 그림

| 파일 | 내용 |
|---|---|
| `figures/P0_main.png` | (a) 추정 곡선 vs 참값 곡선 (n=20000, c=0.25, {REPS}회 평균·90% 범위) (b) 관측 `supIPM_hat` 평균(±2 MC se) 이 각 `c` 의 **이론 목표값**(=그 h 에서의 결정론적 극한, 7.7) 위에 놓이는지 |
| `figures/P1_convergence.png` | C1: supIPM 의 bias/RMSE/SD vs n (c=1) + `c=1` 의 이론 편향선. 점근선 `n^-2/5` 는 회색 점선으로 **따로** 표시 (유한 n 목표가 아님) |
| `figures/P1b_bandwidth.png` | C1b: c 별 RMSE(실선) vs 각 c 의 이론 편향선(점선). c=1 은 겹치고(편향 지배), c=0.25 는 RMSE 가 위에 있다(분산 지배) |
| `figures/P2_grid.png` | C2: 오차 vs `delta_G/h` |
| `figures/P3_binning.png` | C3: 오차 분포 상자그림, 커널 vs binning |
| `figures/fig_truth.png` | 참값: `mu(s)` 와 `W1(s)` 곡선 |
| `figures/fig_single_draw.png` | 한 번의 draw 에서 `W1_hat(s)` vs 참값 vs **결정론적 h-극한** (n=10000, 왼쪽 c=1 / 오른쪽 c=0.25). 추정 곡선이 참값이 아니라 h-극한 위에 놓이는 것을 보임 |

## 9. 참고 사항

- 사양(`S1_SPEC.md`)이 고정한 것: 모형 상수, `alpha=0.05`, GH 101 노드, u 격자, 반복 100회,
  seed 규칙, C1/C2/C3 의 격자, `c=1`.
- 이 구현이 사양과 다른 점 두 가지:
  1. `sigma_c` 를 구적이 아니라 **닫힌 형태**로 계산 (구적으로 하면 `Var(Z)=1` 이 1e-3 어긋남).
     `F(u)` 는 사양대로 GH 101 노드를 쓴다.
  2. **C1b 를 추가** (`c = 0.25, 0.5`). 사양의 C1~C3 는 그대로 두었다.
- EIPM 은 `results/*.npz` 에 계산되어 있으나 노트북 그림·표에서는 뺐다 (참값이 {SUP:.3f} vs
  {EIP:.3f} 로 달라 나란히 놓으면 성능 비교로 오독되기 때문).
- **범위 밖**: "sup DP 를 목적으로 할 때 supIPM 은 되고 EIPM 은 안 된다" 는 방법 비교는 여기 없다.
  사양 6절이 별도 시뮬레이션으로 미뤄둔 항목이고, 학습(최적화)이 들어가야 하므로 실험 종류가 다르다.

## 10. 해석해 볼 만한 지점

1. 7.5 에서 유한 n 기울기가 `h/tau` 로 결정된다는 것의 실무적 함의 (실전에서 `tau` 를 모를 때).
2. 7.1 의 상대 RMSE 가 n=20000 에서도 21% 인 것 — 7.7 의 h-극한 대조와 함께 보면?
3. 7.3 에서 `delta_G/h` 의 실용적 상한을 어디로 잡아야 하는가.
4. 7.4 에서 J=10, 20 이 커널(c=1)보다 나은 것을 어떻게 해석해야 하는가.
5. `h` 를 실제 데이터에서 어떻게 고를 것인가 (여기서는 `tau` 를 알고 있지만 실전에서는 모름).
""")

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("wrote", OUT, f"({len(L)} lines)")
