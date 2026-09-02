"""Per-critic gradient-ascent sup over s: s*_c = argmax_s Delta_c(s) for every critic c."""
import torch

from ipm import kernel_weights, pool_ipm, kernel_weights_percritic, percritic_ipm


def find_s_star_percritic(z_det, s_std, critic, h, s_lo, s_hi, K=8, n_s_steps=20, lr_s=0.1,
                          n_grid=33, grid_tol=1e-6, generator=None):
    """For every critic c maximize Delta_c(s) = |sum_i (w_i(s) - 1/B) v_c(z_i)|
    over s in [s_lo, s_hi] by projected gradient ascent.

    Same scheme as sup_s.find_s_star, batched over the pool: K restarts per
    critic (one seeded at that critic's own coarse-grid argmax, the rest
    uniform), the elementwise best (s, value) over every start and every ascent
    iterate is kept. The penalty value max_c Delta_c(s*_c) equals the shared
    mode's max_s max_c (the two sups commute); what differs is that each critic
    is anchored at its own worst s, so the pool can cover several s-modes at
    once instead of collapsing onto the single s* of the currently-strongest
    critic.

    - The critic pool and the representations are treated as constants
      (gradients flow through the kernel weights only); summing obj over [K, C]
      before autograd is a batching trick — each s_{k,c} receives only its own
      gradient.

    Returns (s_star_c: detached [C] tensor, diag: dict with the same
    ipm/ga_max/grid_max/s_star/violation keys as find_s_star — reduced over the
    pool so train.py logging works unchanged — plus s_star_std/s_star_c).
    """
    device = z_det.device
    with torch.no_grad():
        V = critic(z_det)                                       # [B, C]
        C = V.size(1)
        grid = torch.linspace(s_lo, s_hi, n_grid, device=device)
        grid_vals = pool_ipm(kernel_weights(grid, s_std, h), V)  # [G, C]
        grid_max_c = grid_vals.max(dim=0).values                 # [C]
        grid_arg_c = grid[grid_vals.argmax(dim=0)]               # [C]

    init = torch.empty(K, C, device=device)
    init[0] = grid_arg_c
    if K > 1:
        # drawn on CPU (optionally from a dedicated generator) so the draw is
        # device-independent and never touches the DataLoader's global RNG stream
        u = torch.rand(K - 1, C, generator=generator).to(device)
        init[1:] = s_lo + (s_hi - s_lo) * u

    s = init.clone().requires_grad_(True)
    best_val = torch.full((K, C), -float('inf'), device=device)
    best_s = init.clone()

    for t in range(n_s_steps + 1):
        obj = percritic_ipm(kernel_weights_percritic(s, s_std, h), V)  # [K, C]
        with torch.no_grad():
            improved = obj > best_val
            best_val[improved] = obj.detach()[improved]
            best_s[improved] = s.detach()[improved]
        if t == n_s_steps:
            break
        (grad,) = torch.autograd.grad(obj.sum(), s)
        with torch.no_grad():
            s.add_(lr_s * grad)
            s.clamp_(s_lo, s_hi)

    k_c = best_val.argmax(dim=0)                                # [C]
    cols = torch.arange(C, device=device)
    s_star_c = best_s[k_c, cols]                                # [C]
    ipm_c = best_val[k_c, cols]                                 # [C]
    diag = dict(s_star=s_star_c[ipm_c.argmax()].item(), ipm=ipm_c.max().item(),
                ga_max=ipm_c.max().item(), grid_max=grid_max_c.max().item(),
                violation=bool((ipm_c < grid_max_c - grid_tol).any()),
                s_star_std=s_star_c.std().item(), s_star_c=s_star_c.detach())
    return s_star_c.detach(), diag
