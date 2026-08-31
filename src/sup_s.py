"""Gradient-ascent sup over the sensitive attribute s (batched restarts + grid seed)."""
import torch

from ipm import kernel_weights, pool_ipm


def find_s_star(z_det, s_std, critic, h, s_lo, s_hi, K=8, n_s_steps=20, lr_s=0.1,
                n_grid=33, grid_tol=1e-6, generator=None):
    """Maximize IPM_pool(s) = max_c |sum_i (w_i(s) - 1/B) v_c(z_i)| over
    s in [s_lo, s_hi] by projected gradient ascent.

    - K batched starts: one seeded at the argmax of a coarse `n_grid`-point grid,
      the rest uniform on [s_lo, s_hi]; the best (s, value) over every start and
      every ascent iterate is kept, so the result is >= the grid max by
      construction — the logged `violation` flag only fires on a numerical bug.
    - The critic pool and the representations are treated as constants (gradients
      flow through the kernel weights w_i(s) only).

    Returns (s_star: detached 0-dim tensor, diag: dict with ipm/ga_max/grid_max/
    s_star/violation).
    """
    device = z_det.device
    with torch.no_grad():
        V = critic(z_det)                                       # [B, C]
        grid = torch.linspace(s_lo, s_hi, n_grid, device=device)
        grid_vals = pool_ipm(kernel_weights(grid, s_std, h), V).max(dim=1).values
        grid_max = grid_vals.max()
        grid_arg = grid[grid_vals.argmax()]

    init = torch.empty(K, device=device)
    init[0] = grid_arg
    if K > 1:
        # drawn on CPU (optionally from a dedicated generator) so the draw is
        # device-independent and never touches the DataLoader's global RNG stream
        u = torch.rand(K - 1, generator=generator).to(device)
        init[1:] = s_lo + (s_hi - s_lo) * u

    s = init.clone().requires_grad_(True)
    best_val = torch.full((K,), -float('inf'), device=device)
    best_s = init.clone()

    for t in range(n_s_steps + 1):
        obj = pool_ipm(kernel_weights(s, s_std, h), V).max(dim=1).values  # [K]
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

    k = best_val.argmax()
    ga_max = best_val[k].item()
    diag = dict(s_star=best_s[k].item(), ipm=ga_max, ga_max=ga_max,
                grid_max=grid_max.item(),
                violation=bool(ga_max < grid_max.item() - grid_tol))
    return best_s[k].detach(), diag
