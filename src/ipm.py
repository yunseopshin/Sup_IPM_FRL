"""Kernel weights, the conditional ReLU-IPM estimator, and the test-time grid sup.

    IPM_hat(h, s) = sup_{v in V} | sum_i v(z_i) K_h(s - S_i) / D_n(s) - (1/n) sum_i v(z_i) |
                  = sup_{v in V} | sum_i (w_i(s) - 1/n) v(z_i) |,
    w_i(s) = K_h(s - S_i) / D_n(s),   K_h Gaussian,   D_n(s) = sum_j K_h(s - S_j).

S enters standardized by the train mean/std, so the bandwidth `h` is in units of
std(S). The self-normalized Gaussian weights are exactly a softmax of
-(s - S_i)^2 / (2 h^2), which is the numerically stable way to compute them.
"""
import torch
import torch.nn.functional as F


def kernel_weights(s_query, s_batch, h):
    """Nadaraya-Watson kernel weights.

    s_query: [K] query points, s_batch: [B] standardized sensitive values.
    Returns w: [K, B], rows summing to 1.
    """
    e = -((s_query.unsqueeze(1) - s_batch.unsqueeze(0)) ** 2) / (2.0 * h * h)
    return F.softmax(e, dim=1)


def pool_ipm(w, V):
    """Per-(query, critic) conditional IPM values |(w - 1/B) @ V|.

    w: [K, B] kernel weights, V: [B, C] critic outputs. Returns [K, C].
    """
    B = V.size(0)
    return torch.matmul(w - 1.0 / B, V).abs()


def relu_ipm_sup_grid(z, s_std, h, grid, n_restarts=4, n_steps=400, lr=1e-2,
                      warm_critic=None, mode='sphere_box'):
    """Test-time sup_v IPM_hat(h, s) on a fine s-grid over the full evaluation set.

    For every grid point, maximizes over the paper's ReLU class
    {v(z) = relu(theta^T z + mu) : theta in S^{d-1}, mu in [-1, 1]}
    (relu IPM.pdf Eq.(3)) with Adam ascent on `n_restarts` random restarts per grid
    point (batched over all grid points and restarts), projecting after each step
    (row-wise theta normalization + mu clamp) and keeping the best value seen
    (start points included). mode='ball' instead uses the ORIGINAL joint
    (theta, mu) l2-ball class (kept as an option; matches critic_proj='ball'
    training). When `warm_critic` (a trained ReLUCritic) is given, its
    per-grid-point best pool member is added as one extra warm-started restart.

    z: [n, d] representations, expected ALREADY normalized into B^d by
    models.normalize_reps (treated as constants). s_std: [n] standardized S,
    grid: [G] standardized query points. Returns ipm_vals: [G] (detached).
    """
    device = z.device
    z = z.detach()
    n, d = z.shape
    G = grid.numel()
    with torch.no_grad():
        W = kernel_weights(grid, s_std, h) - 1.0 / n          # [G, n]

    # random restarts: theta uniform on S^{d-1}, mu uniform on [-1, 1]; drawn on
    # CPU so the draw is device-independent. When no trained pool exists
    # (lambda=0) one extra random restart keeps the total budget equal to the
    # warm-started case.
    R0 = n_restarts if warm_critic is not None else n_restarts + 1
    if mode == 'sphere_box':
        theta0 = torch.randn(G, R0, d)
        theta0 = theta0 / theta0.norm(dim=2, keepdim=True).clamp_min(1e-12)
        mu0 = torch.rand(G, R0, 1) * 2.0 - 1.0
        joint = torch.cat([theta0, mu0], dim=2).to(device)
    elif mode == 'ball':
        joint = torch.randn(G, R0, d + 1)
        joint = (joint / joint.norm(dim=2, keepdim=True).clamp_min(1e-12)).to(device)
    else:
        raise ValueError(f'unknown mode: {mode}')

    if warm_critic is not None:
        with torch.no_grad():
            V_pool = warm_critic(z)                            # [n, C]
            pool_vals = torch.matmul(W, V_pool).abs()          # [G, C]
            best_c = pool_vals.argmax(dim=1)                   # [G]
            warm = torch.cat([warm_critic.fc.weight.data[best_c],
                              warm_critic.fc.bias.data[best_c].unsqueeze(1)], dim=1)
        joint = torch.cat([joint, warm.unsqueeze(1)], dim=1)   # [G, R+1, d+1]

    R = joint.size(1)
    param = joint.clone().requires_grad_(True)
    opt = torch.optim.Adam([param], lr=lr)
    best = torch.full((G,), -float('inf'), device=device)

    for step in range(n_steps + 1):
        theta = param[..., :d].reshape(G * R, d)
        mu = param[..., d].reshape(G * R)
        V = torch.relu(torch.matmul(z, theta.t()) + mu)        # [n, G*R]
        obj = torch.einsum('gn,ngr->gr', W, V.view(n, G, R)).abs()
        best = torch.maximum(best, obj.max(dim=1).values.detach())
        if step == n_steps:
            break
        opt.zero_grad()
        (-obj.sum()).backward()
        opt.step()
        with torch.no_grad():
            if mode == 'sphere_box':
                th = param[..., :d]
                th.div_(th.norm(dim=2, keepdim=True).clamp_min(1e-12))
                param[..., d].clamp_(-1.0, 1.0)
            else:
                norms = param.norm(dim=2, keepdim=True)
                param.div_(torch.clamp(norms, min=1.0))

    return best
