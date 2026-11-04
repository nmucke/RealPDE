"""AXE (experiments/axe_extended, 2026-09-24): causal per-trajectory choice of `blend_ridge` (the ridge pulling the
per-region LS blend (a, b) toward (1, 0)), FACTORISED with AX-B's energy-blend choice (ax_adapt.EBAdapt).

At every whole-step window the tail is evaluated for Kc = K_eb + K_br candidate rows, batched along dim 0:
    rows 0 .. K_eb-1           : (b_k, lambda*)   -- AX-B's energy-blend candidates at the current ridge
    rows K_eb .. K_eb+K_br-1   : (b*, lambda_j)   -- the ridge candidates at the current energy blend
The emitted row is (b*, lambda*) = row k_eb (first block). Next step every row is scored on the revealed window by
AX-B's composite J (official per-sample rel_l2 / tke / mvpe); S_eb <- rho S_eb + valid J[:K_eb],
S_br <- rho S_br + valid J[K_eb:]; b* = grid_eb[argmax S_eb], lambda* = grid_br[argmax S_br].
Device-side only (argmax + index_select + cat), no host sync -> CUDA-graph safe. Ties with no evidence resolve to
CONFIG["blend_ridge"] through a 1e-7 prior (ax_adapt._prior)."""
from __future__ import annotations

import torch

from ax_adapt import EBAdapt, _prior


class BRAdapt:
    def __init__(self, grid, rho, eb: EBAdapt, device, br_fixed=1.0):
        g = torch.tensor([float(v) for v in grid], dtype=torch.float32, device=device)
        self.K = int(g.numel())
        self.rho = float(rho)
        self.grid = [float(v) for v in grid]
        self.lamv = g.view(self.K, 1, 1, 1, 1)                        # lam per candidate, broadcast in _solve
        self.S0 = _prior(g, br_fixed, (self.K,), device)
        self.Keb = eb.K
        self.Kc = eb.K + self.K
        self.ar_eb = torch.arange(eb.K, device=device, dtype=torch.long)
        self.ar_br = torch.arange(self.K, device=device, dtype=torch.long)


def ebr_scores(ad: EBAdapt, y, outK):
    """AX-B's J (ax_adapt.eb_select v2, verbatim maths) for every one of the outK.shape[0] rows."""
    K = outK.shape[0]
    t = y[..., :2]
    Z = torch.cat([outK[..., :2], t], dim=0)                            # (K+1,T,H,W,2)
    var, mean = torch.var_mean(Z, dim=1, unbiased=False)                # (K+1,H,W,2)
    num = torch.linalg.vector_norm((Z[:K] - t) * ad.std2, dim=(1, 2, 3, 4))
    den = torch.linalg.vector_norm(torch.addcmul(ad.mean2, t, ad.std2))
    ke = (var * ad.std2sq).sum(-1)                                     # (K+1,H,W)
    tk_num = torch.linalg.vector_norm(ke[:K] - ke[K:], dim=(1, 2))
    tk_den = torch.linalg.vector_norm(ke[K])
    M = torch.addcmul(ad.mean2, mean.index_select(1, ad.py).index_select(2, ad.px), ad.std2)   # (K+1,ny,nx,2)
    mv = (torch.linalg.vector_norm(M[:K] - M[K:], dim=(1, 3))
          / torch.linalg.vector_norm(M[K:], dim=(1, 3)).clamp_min(1e-8)).mean(1)               # (K,)
    E = torch.stack([num / den.clamp_min(1e-8), tk_num / tk_den.clamp_min(2e-8), mv])        # (3,K)
    J = torch.div(ad.jc, torch.add(ad.one3, E, alpha=0.5)).sum(0)                               # (K,)
    return torch.nan_to_num(J, nan=-1e6, posinf=-1e6, neginf=-1e6)


def ebr_select(eb: EBAdapt, br: BRAdapt, S_eb, S_br, valid, y, outK):
    """outK (Kc,T,H,W,3): every row of window n-1. Updates S_eb (K_eb,), S_br (K_br,) in place.
    Returns (k_eb (1,) long = emitted row, zidx (Kc,) long = ridge candidate per row, eidx (Kc,) long = blend per row)."""
    J = ebr_scores(eb, y, outK) * valid
    torch.add(J[:br.Keb], S_eb, alpha=eb.rho, out=S_eb)
    torch.add(J[br.Keb:], S_br, alpha=br.rho, out=S_br)
    k_eb = torch.argmax(S_eb).view(1)
    k_br = torch.argmax(S_br).view(1)
    zidx = torch.cat([k_br.expand(br.Keb), br.ar_br])
    eidx = torch.cat([br.ar_eb, k_eb.expand(br.K)])
    return k_eb, zidx, eidx
