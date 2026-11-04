"""AW (experiments/aw_adaptive_cb, 2026-09-24): causal per-trajectory CB weight.

CB shifts the emitted interval by w * (raw - ctr_out) (half-widths, pred and all
internal state untouched). Instead of one fixed w, each trajectory picks w from a
small grid by the score each grid w WOULD have earned on the already-revealed
previous windows (stride 20 == horizon 20: window n's prev_target is window n-1's
full target):
    s_k = sum_{scored elements of window n-1} exp(-2 h / sigma_g) * 1[|T - c_k| <= h] / #scored
    c_k = ctr_out_{n-1} + w_k (raw_{n-1} - ctr_out_{n-1}),  h = half-width of window n-1
    S  <- rho * S + s ;  w_n = grid[argmax S]
(exactly the official per-element sps term without the per-window accuracy factor;
u, v only; scored = physical prev_target != 0, the same mask the cond-width net uses).
Window 0 has no CB (ctr_out is None) and window 1 uses the fixed CB weight; window 1's
(d, width) and a tie-break prior on the fixed-w grid point seed S, so the first
adaptive window is window 2. Everything is device-side (argmax + gather): no host
sync, no data-dependent branching -> CUDA-graph safe. The SAME function runs in the
eager path and inside the whole-step graph (bit-identical values)."""
from __future__ import annotations

import torch


class CBAdapt:
    """Static per-run constants (device tensors), built once before any capture."""

    def __init__(self, grid, rho, w_fixed, std_tgt, mean_tgt, sigma_g, device):
        g = torch.tensor([float(v) for v in grid], dtype=torch.float32, device=device)
        self.K = int(g.numel())
        self.grid = g
        self.gridv = g.view(self.K, 1, 1, 1, 1, 1)
        self.rho = float(rho)
        self.std2 = std_tgt[:2].clone()
        self.mean2 = mean_tgt[:2].clone()
        self.coef2 = (-2.0 * std_tgt[:2] / sigma_g).clone()          # exp(-2 h_phys / sigma), h in norm units
        self.cmask = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32, device=device).view(1, 1, 1, 1, 3)
        self.gridc = (g.view(self.K, 1) * self.cmask.view(1, 3)).contiguous()   # (K, 3): [w_k, w_k, 0]
        # tie-break prior: exact ties (no evidence, e.g. d == 0) resolve to the fixed-w grid point
        s0 = torch.zeros(self.K, dtype=torch.float32, device=device)
        s0[int(torch.argmin((g - float(w_fixed)).abs()).item())] = 1e-7
        self.S0 = s0


def cb_scored(ad: CBAdapt, y):
    """(1,T,H,W,2) float 0/1: physical prev_target != 0 (same expression / values as t2_cw_update's mask)."""
    import cond_width_net as _CW   # MSK: exact_mask switch (same mask as t2_cw_update's)
    return _CW._scored(y[..., :2] * ad.std2 + ad.mean2, y.dtype)


def cb_adapt_select(ad: CBAdapt, S: torch.Tensor, y, co, d_prev, wd_prev, scored=None):
    """y, co, d_prev: (1,T,H,W,3) normalised units (prev_target, previous ctr_out,
    previous raw - ctr_out); wd_prev: (1,T,H,W,2) previous upper - lower (pre-shift);
    scored: optional (1,T,H,W,2) 0/1 mask view (the whole-step graph passes t2_cw_update's).
    Updates S (K,) in place; returns the (1,1,1,1,3) weight [w*, w*, 0].
    16 kernels (+3 when scored is computed here)."""
    if scored is None:
        scored = cb_scored(ad, y)
    hh = wd_prev * 0.5
    e = torch.exp(hh * ad.coef2) * scored
    R = y[..., :2] - co[..., :2]
    X = torch.addcmul(R.unsqueeze(0), ad.gridv, d_prev[..., :2].unsqueeze(0), value=-1.0)
    s = ((X.abs_() <= hh.unsqueeze(0)) * e).sum(dim=(1, 2, 3, 4, 5))
    n = scored.sum().clamp_min(1.0)
    torch.add(s / n, S, alpha=ad.rho, out=S)
    k = torch.argmax(S).view(1)
    return ad.gridc.index_select(0, k).view(1, 1, 1, 1, 3)
