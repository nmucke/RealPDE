"""AX (experiments/ax_adaptive, 2026-09-24): two more causal per-trajectory choices on top of AW (cb_adapt.py).

(A) KWAdapt -- W2 half-width multiplier kappa per lead group (default leads 1-2 / 3-20). The emitted interval
    c +- h becomes c +- kappa h (output only: the cond-width net's features never read the emitted bounds). Score
    of each grid kappa_j on the revealed previous window (stride 20 == horizon 20):
        s_gj = sum_{scored elem of window n-1, leads of group g} exp(-2 kappa_j h / sigma_g) 1[|T - c| <= kappa_j h]
               / #scored(group g)
    c = the centre actually emitted on window n-1 (ctr_out_{n-1} + w_{n-1} d_{n-1}, w = AW's weight), h = its
    pre-kappa half-width. S_g <- rho S_g + s_g ; kappa_g = grid[argmax S_g]. AW's own score keeps the pre-kappa h.
(B) EBAdapt -- the energy-target blend e_blend_w b (e_tgt = b Khat + (1-b) R_run). Every step the energy/shape
    stage is evaluated for all K grid values at once (batched along dim 0; same kernels), the selected one is
    emitted, all K outputs are kept; next step each is scored on the revealed truth by
        J = 0.48*100/(1+0.5 rel_l2) + 0.07*100/(1+0.5 tke) + 0.10*100/(1+0.5 mvpe)   (official per-sample errors)
    S <- rho S + valid * J ; b = grid[argmax S]. `valid` is 0 on the first whole-step window of a trajectory
    (window 1 ran outside this graph, so its K candidates do not exist) -> b is adaptive from window 3 on.
Everything is device-side (argmax + index_select): no host sync, no data-dependent branch -> CUDA-graph safe.
Ties with no evidence resolve to the fixed value through a 1e-7 prior."""
from __future__ import annotations

import torch


def _prior(g, v, shape, device):
    s0 = torch.zeros(shape, dtype=torch.float32, device=device)
    s0[..., int(torch.argmin((g - float(v)).abs()).item())] = 1e-7
    return s0


class KWAdapt:
    def __init__(self, grid, groups, rho, std_tgt, sigma_g, T, device, k_fixed=1.0):
        g = torch.tensor([float(v) for v in grid], dtype=torch.float32, device=device)
        self.K = int(g.numel())
        self.G = len(groups)
        self.rho = float(rho)
        self.kv = g.view(self.K, 1, 1, 1, 1, 1)
        self.coef2 = (-2.0 * std_tgt[:2] / sigma_g).clone()
        gm = torch.zeros(T, self.G, dtype=torch.float32)
        l2g = torch.zeros(T, dtype=torch.long)
        for gi, leads in enumerate(groups):
            for l in leads:
                gm[l, gi] = 1.0
                l2g[l] = gi
        assert (gm.sum(1) == 1).all(), "every lead must be in exactly one group"
        self.gmat = gm.to(device)
        self.l2g = l2g.to(device)
        self.ftab = (0.5 * (1.0 - g)).contiguous()                    # bound shift per unit width
        self.S0 = _prior(g, k_fixed, (self.G, self.K), device)


def kw_select(ad: KWAdapt, S, y, co, d_prev, wd_prev, w_prev, scored):
    """y, co, d_prev: (1,T,H,W,3) normalised; wd_prev (1,T,H,W,2) pre-shift width of window n-1; w_prev
    (1,T,1,1,3) the CB weight emitted on window n-1; scored (1,T,H,W,2) 0/1. Updates S (G,K) in place.
    Returns f (1,T,1,1,1): lower += f * wd, upper -= f * wd (f = (1 - kappa)/2 per lead)."""
    hh = wd_prev * 0.5
    R = torch.addcmul(y[..., :2] - co[..., :2], w_prev[..., :2], d_prev[..., :2], value=-1.0).abs_()
    KH = ad.kv * hh.unsqueeze(0)                                       # (K,1,T,H,W,2)
    e = torch.exp(KH * ad.coef2) * scored
    s = ((R.unsqueeze(0) <= KH) * e).sum(dim=(1, 3, 4, 5))             # (K,T)
    n = scored.sum(dim=(0, 2, 3, 4)) @ ad.gmat                         # (G,)
    sg = (s @ ad.gmat) / n.clamp_min(1.0)                              # (K,G)
    torch.add(sg.t(), S, alpha=ad.rho, out=S)
    k = torch.argmax(S, dim=1)                                         # (G,)
    return ad.ftab.index_select(0, k).index_select(0, ad.l2g).view(1, -1, 1, 1, 1)


class EBAdapt:
    def __init__(self, grid, rho, std_tgt, mean_tgt, H, W, device, b_fixed=0.85, sub_s_real=2):
        g = torch.tensor([float(v) for v in grid], dtype=torch.float32, device=device)
        self.K = int(g.numel())
        self.rho = float(rho)
        self.grid = [float(v) for v in grid]
        self.wv = g.view(self.K, 1, 1, 1, 1)
        self.omv = torch.tensor([1.0 - float(v) for v in grid], dtype=torch.float32, device=device).view(self.K, 1, 1, 1, 1)
        self.std2 = std_tgt[:2].clone()
        self.mean2 = mean_tgt[:2].clone()
        self.std2sq = (std_tgt[:2] ** 2).clone()
        # probe geometry: verbatim from starting_kit/scoring.py::mvpe_rel_l2_per_sample
        d, center_x, center_y, n_probe = 16, 10, 32, 9
        pcy = int(center_y / sub_s_real)
        iy = min(2, int(H / (n_probe + 1)))
        py = [pcy + iy * j for j in range(-(n_probe - 1) // 2, n_probe - (n_probe - 1) // 2)]
        py = [v for v in py if 0 <= v < H]
        px = []
        for i in range(4):
            if int((2 * d + center_x) / sub_s_real) < W:
                p = int(((i + 1) * d + center_x) / sub_s_real)
            else:
                p = int((0.5 * (i + 2) * d + center_x) / sub_s_real)
            if 0 <= p < W:
                px.append(p)
        assert py and px
        self.py = torch.tensor(py, dtype=torch.long, device=device)
        self.px = torch.tensor(px, dtype=torch.long, device=device)
        self.S0 = _prior(g, b_fixed, (self.K,), device)
        self.jc = torch.tensor([48.0, 7.0, 10.0], dtype=torch.float32, device=device).view(3, 1)
        self.one3 = torch.ones(3, 1, dtype=torch.float32, device=device)


def eb_select(ad: EBAdapt, S, valid, y, outK):
    """y (1,T,H,W,3) normalised prev_target; outK (K,T,H,W,3) normalised outputs of every candidate on window
    n-1; valid 0-dim float (0 = no candidate outputs yet). Updates S (K,) in place; returns k (1,) long.
    Kernel diet (v2): candidates and target stacked for one var_mean; norms via vector_norm; the three errors
    stacked for one J expression. Same quantities as v1 (official formulas) up to float rounding."""
    K = ad.K
    t = y[..., :2]
    Z = torch.cat([outK[..., :2], t], dim=0)                            # (K+1,T,H,W,2)
    var, mean = torch.var_mean(Z, dim=1, unbiased=False)                # (K+1,H,W,2)
    # rel_l2 (u, v, physical): ||(p - t) std|| / ||t std + mean||
    num = torch.linalg.vector_norm((Z[:K] - t) * ad.std2, dim=(1, 2, 3, 4))
    den = torch.linalg.vector_norm(torch.addcmul(ad.mean2, t, ad.std2))
    # tke: per-pixel 0.5 sum_ch var_t (physical); the 0.5 cancels in the ratio
    ke = (var * ad.std2sq).sum(-1)                                     # (K+1,H,W)
    tk_num = torch.linalg.vector_norm(ke[:K] - ke[K:], dim=(1, 2))
    tk_den = torch.linalg.vector_norm(ke[K])
    # mvpe: physical time-mean at the probes, rel-L2 per probe column (over probe rows x channels), mean over columns
    M = torch.addcmul(ad.mean2, mean.index_select(1, ad.py).index_select(2, ad.px), ad.std2)   # (K+1,ny,nx,2)
    mv = (torch.linalg.vector_norm(M[:K] - M[K:], dim=(1, 3))
          / torch.linalg.vector_norm(M[K:], dim=(1, 3)).clamp_min(1e-8)).mean(1)               # (K,)
    E = torch.stack([num / den.clamp_min(1e-8), tk_num / tk_den.clamp_min(2e-8), mv])        # (3,K)
    J = torch.div(ad.jc, torch.add(ad.one3, E, alpha=0.5)).sum(0)                               # (K,)
    # guard: a non-finite candidate score must never win the argmax (torch.argmax treats NaN as the max);
    # value-preserving for finite J
    J = torch.nan_to_num(J, nan=-1e6, posinf=-1e6, neginf=-1e6)
    torch.add(J * valid, S, alpha=ad.rho, out=S)
    return torch.argmax(S).view(1)


def eb_select_v1(ad: EBAdapt, S, valid, y, outK):
    """v1 (reference for the v2 equivalence test only)."""
    p = outK[..., :2]
    t = y[..., :2]
    tph = t * ad.std2 + ad.mean2
    # rel_l2 (u, v, physical)
    num = ((p - t) * ad.std2).square().sum(dim=(1, 2, 3, 4))
    rel = (num / tph.square().sum().clamp_min(1e-16)).sqrt()
    # tke: 0.5 * sum_ch var_t (physical) per pixel, rel-L2 over the map
    kp = (p.var(dim=1, unbiased=False) * ad.std2sq).sum(-1)            # (K,H,W)  (0.5 cancels in the ratio)
    kt = (t.var(dim=1, unbiased=False) * ad.std2sq).sum(-1)            # (1,H,W)
    tke = ((kp - kt).square().sum(dim=(1, 2)) / kt.square().sum().clamp_min(4e-16)).sqrt()
    # mvpe: time-mean at the probes, rel-L2 per probe column, mean over columns
    pm = p.mean(dim=1).index_select(1, ad.py).index_select(2, ad.px) * ad.std2 + ad.mean2    # (K,ny,nx,2)
    tm = tph.mean(dim=1).index_select(1, ad.py).index_select(2, ad.px)                      # (1,ny,nx,2)
    mv = ((pm - tm).square().sum(dim=(1, 3)).sqrt() / tm.square().sum(dim=(1, 3)).sqrt().clamp_min(1e-8)).mean(1)
    J = 48.0 / (1.0 + 0.5 * rel) + 7.0 / (1.0 + 0.5 * tke) + 10.0 / (1.0 + 0.5 * mv)
    torch.add(J * valid, S, alpha=ad.rho, out=S)
    return torch.argmax(S).view(1)
