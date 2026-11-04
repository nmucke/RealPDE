"""KD2 (experiments/kd2_adapt_diet, 2026-09-24): the adaptive selectors AW (cb_adapt.py) + AX (ax_adapt.py: kw, eb)
and their CB / MSK glue as ONE fused block of the whole-step graph (cuda_graph_wholestep.py, CONFIG "kd2_adapt").

Same maths, same decisions, far fewer kernels (the selectors were ~78 tiny launch-bound kernels per step):
  * AW's 5 centres and KW's centre (the CB weight emitted on the previous window) are ONE addcmul over a 6-vector of
    weights [grid..., w_prev]; KW's 5 half-width multipliers are ONE product (0.5 kappa) * width (== kappa * hh bit
    for bit); the exp weight is computed once for the 5 kappas and AW reuses the kappa == 1 row (== AW's own e).
    Every per-element indicator * weight product is the parent's value bit for bit.
  * The scored mask is written (as float, by the comparison itself) into row 0 of the product tensor Q, so ONE
    reduction gives the per-lead sums of every AW / KW candidate AND the per-lead scored counts; ONE matmul with a
    0/1 group matrix turns them into AW totals, KW group totals (broadcast back to every lead) and the counts.
  * EB's candidates and the revealed target sit in consecutive rows of one static buffer (no cat); the three
    official errors are reduced into columns of the same small matrix, so ONE clamp (maximum) and ONE division
    serve AW, KW and EB alike.
  * The three score EMAs live in ONE (2, K, L) buffer: one update, one argmax, one gather from a static table that
    yields, in one launch: AW's [w, w, 0], KW's per-lead +f / -f, EB's `valid` flag (always 1 after a graphed
    step) and -- in place -- next step's w_prev.
  * The CB / kappa bound shifts act on lower and upper in one launch each (the output pack is contiguous).
Only the ORDER of the float sums inside the scores changes (per-lead partial sums + matmul instead of one
reduction; physical-unit differences for EB): the scores move by float rounding, the argmax decisions (and hence
pred / bounds) are unchanged on every stream checked -- see the notes. Every constant table is built once, before
any capture; everything in the graph is device-side (no host sync, no data-dependent branch).

AXE variant (br_adapt, axe_adapt.py) -- EXACT mode: the ridge candidates often give near-identical outputs, so their
score EMAs sit on knife-edge ties that ANY change of float summation order flips (measured: 1 flip per LCO stream with the
fused EB errors). Here the (blend, ridge) J is therefore computed by axe_adapt.ebr_scores' own ops on the same tensor
layouts (bit for bit), and AW's total is its own single reduction (as in cb_adapt.py); only the EMA update / argmax /
index outputs are fused: J feeds two column groups of the same EMA buffer and a second (long) gather from a static table
yields, in one launch, the emitted row k_eb and the per-row candidate indices zidx / eidx (= ebr_select's outputs)."""
from __future__ import annotations

import torch


class KD2Plan:
    """Static tables (device tensors). Raises if the configuration is not the one the fused block covers; the
    caller then keeps the parent code path."""

    def __init__(self, cb, kw, eb, cb_w, br=None, T=20, H=32, W=64, device="cuda"):
        if cb is None or kw is None or eb is None or cb_w is None:
            raise RuntimeError("kd2: AW + KW + EB must all be on")
        K = int(cb.K)
        Keb = int(eb.K)
        Kbr = 0 if br is None else int(br.K)
        Kc = Keb + Kbr
        if int(kw.K) != K or Keb > K or Kbr > K or Kc > 2 * K or 5 + 2 * Kc > 5 + 2 * T:
            raise RuntimeError("kd2: grid sizes not covered")
        if not (cb.rho == kw.rho == eb.rho) or (br is not None and br.rho != cb.rho):
            raise RuntimeError("kd2: one EMA rate expected")
        if not torch.equal(cb.coef2, kw.coef2):
            raise RuntimeError("kd2: AW / KW exp coefficients differ")
        if not (torch.equal(cb.std2, eb.std2) and torch.equal(cb.mean2, eb.mean2)):
            raise RuntimeError("kd2: AW / EB statistics differ")
        kap = kw.kv.flatten()
        one = (kap == 1.0).nonzero().flatten()
        if one.numel() != 1:
            raise RuntimeError("kd2: kappa grid must contain 1.0 exactly once")
        cbw = cb_w.reshape(-1, 3)
        if cbw.shape[0] != T or not (bool((cbw[:, 0] == cbw[0, 0]).all()) and bool((cbw[:, 1] == cbw[0, 0]).all())
                                     and bool((cbw[:, 2] == 0).all())):
            raise RuntimeError("kd2: cb_blend_w must be one value for every lead / channel")
        G = int(kw.G)
        l2g = kw.l2g.cpu()
        dev = torch.device(device)
        f32 = torch.float32
        self.K, self.Ke, self.G, self.T = K, Kc, G, T          # Ke = number of candidate rows scored by J
        self.Keb, self.Kbr, self.br = Keb, Kbr, br is not None
        self.rho = float(cb.rho)
        self.j1 = int(one.item())
        # score-matrix columns: 0,1,2 AW [w, w, 0] | 3 EB | 4 valid | 5.. KW +f per lead | 5+T.. KW -f | reps per group
        self.C_EB, self.C_VAL, self.C_KW, self.C_REP = 3, 4, 5, 5 + 2 * T
        L = 5 + 2 * T + G
        self.L = L
        gm = kw.gmat.cpu()                                                  # (T, G)
        GG = torch.zeros(T, L, dtype=f32)
        GG[:, 0:5] = 1.0
        for t in range(T):
            GG[:, 5 + t] = gm[:, int(l2g[t])]
            GG[:, 5 + T + t] = gm[:, int(l2g[t])]
        GG[:, self.C_REP:self.C_REP + G] = gm
        self.GG = GG.to(dev)
        self.grid = cb.grid.clone()
        g = cb.grid.cpu()
        ft = kw.ftab.cpu()
        TAB = torch.zeros(2, K, L, dtype=f32)
        TAB[0, :, 0] = g
        TAB[0, :, 1] = g
        TAB[0, :, self.C_VAL] = 1.0
        TAB[1, :, 5:5 + T] = ft.view(K, 1)
        TAB[1, :, 5 + T:5 + 2 * T] = -ft.view(K, 1)
        self.TAB = TAB.to(dev)
        if br is not None:
            # plane-0 columns 5 .. 5+2Kc-1 carry the AXE indices (long table, second gather):
            #   [BR dup x Keb | 0..Kbr-1 | 0..Keb-1 | EB dup x Kbr] -> zidx = cols 5..5+Kc-1, eidx = cols 5+Kc..5+2Kc-1
            TL = torch.zeros(2, K, L, dtype=torch.long)
            kk = torch.arange(K).view(K, 1)
            self.C_BRD = 5
            self.C_EBD = 5 + Kc + Keb
            TL[0, :, self.C_BRD:self.C_BRD + Keb] = kk
            TL[0, :, 5 + Keb:5 + Kc] = torch.arange(Kbr).view(1, Kbr)
            TL[0, :, 5 + Kc:5 + Kc + Keb] = torch.arange(Keb).view(1, Keb)
            TL[0, :, self.C_EBD:self.C_EBD + Kbr] = kk
            self.TABL = TL.to(dev)
            self.br_S0 = br.S0
        # per-lead / per-group S seeds
        self.kwS0_rows = kw.S0.cpu()                                        # (G, K)
        self.l2g = l2g
        self.cbw3 = cbw[0].clone().to(dev)                                  # [w, w, 0]
        self.kaph = (0.5 * kap).view(K, 1, 1, 1, 1).contiguous().to(dev)    # 0.5 kappa (exact)
        self.coef2 = cb.coef2
        self.std2, self.mean2 = eb.std2, eb.mean2
        Ke = self.Ke
        self.mvec = torch.tensor([0.0] + [1.0] * Ke, dtype=f32, device=dev)
        self.flat = (eb.py.view(-1, 1) * W + eb.px.view(1, -1)).flatten().contiguous()
        self.ny, self.nx = int(eb.py.numel()), int(eb.px.numel())
        NC = L + 3 + self.nx
        self.NC = NC
        cv = torch.ones(1, NC, dtype=f32)
        cv[0, L] = 1e-8
        cv[0, L + 1] = 2e-8
        cv[0, L + 3:] = 1e-8
        self.cvec = cv.to(dev)
        self.one3 = torch.ones(1, 3, dtype=f32, device=dev)
        self.jc = torch.tensor([[48.0, 7.0, 10.0]], dtype=f32, device=dev)
        self.eb_S0 = eb.S0
        self.eb = eb
        self.cb_S0 = cb.S0
        # self-test: cuBLAS must give bit-identical outputs for identical 0/1 columns (the AW [w, w] pair and the
        # per-lead KW columns of one group rely on it)
        q = torch.randn(1 + 2 * K, T, device=dev) * 100.0
        M = q @ self.GG
        ok = torch.equal(M[:, 0], M[:, 1])
        for t in range(T):
            gi = int(l2g[t])
            ok = ok and torch.equal(M[:, 5 + t], M[:, self.C_REP + gi]) and torch.equal(M[:, 5 + T + t], M[:, self.C_REP + gi])
        if not ok:
            raise RuntimeError("kd2: matmul column self-test failed")


class KD2State:
    """Per-wrapper static buffers of the fused block (allocated once, before capture)."""

    def __init__(self, plan: KD2Plan, T, H, W, device):
        p = plan
        f = lambda *s: torch.zeros(*s, device=device, dtype=torch.float32)  # noqa: E731
        K, L = p.K, p.L
        self.Qall = f(2 + 2 * K, T, H, W, 2)                 # [resid * scored | scored | AW products | KW products]
        self.Q = self.Qall[1:]                               # [scored | AW | KW]: one per-lead reduction
        self.Q0 = self.Q[0]                                  # (T,H,W,2) scored
        self.Q0cw = self.Q[0].permute(3, 0, 1, 2)            # (2,T,H,W) view for the cond-width update
        self.RScw = self.Qall[0].permute(3, 0, 1, 2)         # (2,T,H,W) resid * scored
        self.RS2 = self.Qall[0:2].permute(0, 4, 1, 2, 3)     # (2, 2, T, H, W): [resid*scored, scored] -> one T-sum
        self.Gm = f(1 + 2 * K, p.NC)                          # sums / norms; row 0 = denominators
        self.RR = f(2 * K, p.NC)                             # ratios; [:, :L] viewed (2, K, L) = this step's scores
        self.SC = self.RR[:, 0:L].view(2, K, L)
        self.S = f(2, K, L)                                  # score EMAs
        self.WB = f(K + 2 * L)                               # [grid (K) | gather output (2, 1, L)]
        self.WB[:K].copy_(p.grid)
        self.Wv = self.WB[0:K + 1].view(K + 1, 1, 1, 1, 1)   # [grid..., w_prev]
        self.VALS = self.WB[K:].view(2, 1, L)
        self.cbw = self.VALS[0, 0, 0:3].view(1, 1, 1, 1, 3)  # [w, w, 0] of this step
        self.valid = self.VALS[0, 0, p.C_VAL]                # 0-dim: 0 on the first graphed window, else 1
        self.f2 = self.VALS[1, 0, p.C_KW:p.C_KW + 2 * T].view(2, T, 1, 1, 1)   # [+f, -f] per lead
        self.J5 = f(2 * K)                                   # [J_eb (Keb) | J_br (Kbr)] then zero pad
        if p.br:
            self.Jeb = self.J5[0:K]                          # (K,): J_eb + zero pad
            self.VL = torch.zeros(2, 1, L, device=device, dtype=torch.long)
            Kc = p.Ke
            self.zidx = self.VL[0, 0, 5:5 + Kc]
            self.eidx = self.VL[0, 0, 5 + Kc:5 + 2 * Kc]
            self.keb = self.VL[0, 0, p.C_EBD:p.C_EBD + 1]
            eb = p.eb
            nx, ny = int(eb.px.numel()), int(eb.py.numel())
            self.eNT = torch.zeros(2, 2 * K, device=device)               # [num; tk_num] (row stride 2K >= Kc)
            self.eDEN = torch.zeros(2 + nx, device=device)                # [den, tk_den, probe norms...]
            self.eE = torch.zeros(3, 2 * K, device=device)                # [rel; tke; mv]
            self.ecv = torch.tensor([1e-8, 2e-8] + [1e-8] * nx, device=device)
            self.eflat = (eb.py.view(-1, 1) * W + eb.px.view(1, -1)).flatten().contiguous()
            self.eny, self.enx = ny, nx
        # views for inspection / probes (same names as the parent's buffers)
        self.cb_S = self.S[0, :, 0]
        self.kw_S = self.S[1, :, p.C_REP:p.C_REP + p.G].t()
        if p.br:
            self.eb_S = self.S[0, 0:p.Keb, p.C_EBD]
            self.br_S = self.S[0, 0:p.Kbr, p.C_BRD]
        else:
            self.eb_S = self.S[0, 0:p.Ke, p.C_EB]


def cw_update(m, cw, pc, prev2, ema2, scored_cw, st=None):
    """t2_cw_graph_body.t2_cw_update with (a) prev_target and the previous centre de-standardised in ONE pass (they
    are rows [y, ctr] of one static buffer, `pc` (2,T,H,W,3)), (b) the mask written as float by the comparison itself
    into `scored_cw` (the view of Q row 0), (c) e_all / the EMA written straight into prev2 / ema2 (no copies),
    (d) with `st`: resid * scored written next to the mask, so ONE sum over T gives num and den, and ONE box filter
    (pad + avg_pool) smooths both (per-output arithmetic unchanged: bit-identical, checked).
    Every value is the parent's bit for bit (same element-wise ops, same reduction layouts)."""
    import cond_width_net as _CW
    ph = pc[..., :2] * m.std_tgt[:2] + m.mean_tgt[:2]                # (2,T,H,W,2)
    pt = ph[0].permute(3, 0, 1, 2)
    ctr = ph[1].permute(3, 0, 1, 2)
    resid = (pt - ctr).abs()
    if _CW.EXACT_MASK:
        torch.gt(pt.abs(), _CW.SCORED_TOL, out=scored_cw)
    else:
        torch.ne(pt, 0.0, out=scored_cw)
    scored = scored_cw
    if st is not None:
        torch.mul(resid, scored, out=st.RScw)
        nd = st.RS2.sum(dim=2)                                           # (2 [num, den], 2, H, W)
        ndf = _CW._box(nd.reshape(4, nd.shape[-2], nd.shape[-1]), cw.n_px)
        numf, denf = ndf[0:2], ndf[2:4]
    else:
        num = (resid * scored).sum(dim=1)
        den = scored.sum(dim=1)
        numf = _CW._box(num, cw.n_px)
        denf = _CW._box(den, cw.n_px)
    e_all = torch.clamp_min(numf / denf.clamp_min(1e-6), 0.0, out=prev2)
    torch.add(cw.rho * ema2, (1.0 - cw.rho) * e_all, out=ema2)


def ebr_J_into(ad, y, outK, out):
    """axe_adapt.ebr_scores, same ops on the same layouts (bit for bit); the final nan_to_num writes into `out`."""
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
    return torch.nan_to_num(J, nan=-1e6, posinf=-1e6, neginf=-1e6, out=out)


def ebr_J_into_v2(ad, Z, out, st):
    """ebr_J_into with the same reductions on the same layouts (bit for bit) and 5 fewer element-wise kernels:
    the three denominators land in one buffer (one clamp = maximum), rel / tke numerators in one (2, K) buffer (one
    division), the three errors are written straight into the (3, K) E rows (no stack), the probe gather is one flat
    index_select (pure gather, same values). Z (K+1,T,H,W,2) = (candidates, target): a strided view of the static
    buffer in place of the parent's torch.cat copy (same values; var_mean on it is bit-identical)."""
    K = Z.shape[0] - 1
    t = Z[K:K + 1]
    var, mean = torch.var_mean(Z, dim=1, unbiased=False)                # (K+1,H,W,2)
    NT, DEN, E = st.eNT[:, 0:K], st.eDEN, st.eE[:, 0:K]
    torch.linalg.vector_norm((Z[:K] - t) * ad.std2, dim=(1, 2, 3, 4), out=NT[0])
    torch.linalg.vector_norm(torch.addcmul(ad.mean2, t, ad.std2), out=DEN[0])
    ke = (var * ad.std2sq).sum(-1)                                     # (K+1,H,W)
    torch.linalg.vector_norm(ke[:K] - ke[K:], dim=(1, 2), out=NT[1])
    torch.linalg.vector_norm(ke[K], out=DEN[1])
    M = torch.addcmul(ad.mean2, mean.reshape(K + 1, -1, 2).index_select(1, st.eflat), ad.std2).view(K + 1, st.eny, st.enx, 2)
    torch.linalg.vector_norm(M[K:], dim=(1, 3), out=DEN[2:].view(1, -1))
    Dc = torch.maximum(DEN, st.ecv)
    torch.mean(torch.linalg.vector_norm(M[:K] - M[K:], dim=(1, 3)) / Dc[2:].view(1, -1), 1, out=E[2])
    torch.div(NT, Dc[0:2].view(2, 1), out=E[0:2])
    J = torch.div(ad.jc, torch.add(ad.one3, E, alpha=0.5)).sum(0)                               # (K,)
    return torch.nan_to_num(J, nan=-1e6, posinf=-1e6, neginf=-1e6, out=out)


def select(p: KD2Plan, st: KD2State, y2, co2, d2, wd, Z, y=None, outK=None):
    """y2 / co2 / d2 (T,H,W,2): prev_target, previous ctr_out, previous raw - ctr_out (normalised); wd (T,H,W,2)
    previous pre-shift width; Z (1+Ke, T,H,W,2) = [prev_target, EB candidate outputs of the previous window].
    Updates st.S in place and writes st.VALS; returns EB's index (1,) long."""
    K, Ke, L = p.K, p.Ke, p.L
    Q, Gm, RR = st.Q, st.Gm, st.RR
    # ---- AW + KW
    R = y2 - co2
    X = torch.addcmul(R, st.Wv, d2, value=-1.0).abs_()                  # (K+1, T,H,W,2)
    KH = p.kaph * wd                                                    # (K, ...) == kappa * hh
    E = torch.exp(KH * p.coef2) * st.Q0
    torch.mul(X[0:K] <= KH[p.j1], E[p.j1], out=Q[1:1 + K])             # AW
    torch.mul(X[K] <= KH, E, out=Q[1 + K:1 + 2 * K])                     # KW
    torch.mm(Q.sum(dim=(2, 3, 4)), p.GG, out=Gm[:, 0:L])
    if p.br:
        # EXACT mode: AW's total as its own single reduction (cb_adapt.py's), J by ebr_scores' ops
        torch.sum(Q[1:1 + K], dim=(1, 2, 3, 4), out=Gm[1:1 + K, 0])
        Gm[1:1 + K, 1].copy_(Gm[1:1 + K, 0])
        torch.div(Gm[1:], torch.maximum(Gm[0:1], p.cvec), out=RR)
        Keb, Kbr = p.Keb, p.Kbr
        ebr_J_into_v2(p.eb, Z, st.J5[0:Ke], st)
        torch.mul(st.J5[0:Keb].view(Keb, 1).expand(Keb, Kbr), st.valid, out=st.SC[0, 0:Keb, p.C_EBD:p.C_EBD + Kbr])
        torch.mul(st.J5[Keb:Keb + K].view(K, 1).expand(K, Keb), st.valid, out=st.SC[0, :, p.C_BRD:p.C_BRD + Keb])
        torch.add(st.SC, st.S, alpha=p.rho, out=st.S)
        r = torch.argmax(st.S, dim=1)
        torch.gather(p.TAB, 1, r.view(2, 1, L), out=st.VALS)
        torch.gather(p.TABL, 1, r.view(2, 1, L), out=st.VL)
        return st.keb
    # ---- EB (physical units)
    Pz = torch.addcmul(p.mean2, Z, p.std2)
    var, mean = torch.var_mean(Pz, dim=1, unbiased=False)
    D = torch.addcmul(Pz, Pz[0:1], p.mvec.view(-1, 1, 1, 1, 1), value=-1.0)
    torch.linalg.vector_norm(D, dim=(1, 2, 3, 4), out=Gm[0:1 + Ke, L])
    ke = var.sum(-1)
    Dk = torch.addcmul(ke, ke[0:1], p.mvec.view(-1, 1, 1), value=-1.0)
    torch.linalg.vector_norm(Dk, dim=(1, 2), out=Gm[0:1 + Ke, L + 1])
    Mp = mean.reshape(1 + Ke, -1, 2).index_select(1, p.flat)
    DM = torch.addcmul(Mp, Mp[0:1], p.mvec.view(-1, 1, 1), value=-1.0)
    torch.linalg.vector_norm(DM.view(1 + Ke, p.ny, p.nx, 2), dim=(1, 3), out=Gm[0:1 + Ke, L + 3:])
    # ---- shared: clamp the denominators, divide, EB's J, one EMA update, one argmax, one gather
    torch.div(Gm[1:], torch.maximum(Gm[0:1], p.cvec), out=RR)
    torch.mean(RR[0:Ke, L + 3:], dim=1, out=RR[0:Ke, L + 2])
    J = torch.div(p.jc, torch.add(p.one3, RR[0:Ke, L:L + 3], alpha=0.5)).sum(1)
    if not p.br:
        torch.nan_to_num(J, nan=-1e6, posinf=-1e6, neginf=-1e6, out=st.J5[0:Ke])
        torch.mul(st.J5[0:K], st.valid, out=st.SC[0, :, p.C_EB])
        torch.add(st.SC, st.S, alpha=p.rho, out=st.S)
        r = torch.argmax(st.S, dim=1)
        torch.gather(p.TAB, 1, r.view(2, 1, L), out=st.VALS)
        return r[0, p.C_EB:p.C_EB + 1]
    # AXE: J of the Keb blend rows -> the Kbr EB-dup columns, J of the Kbr ridge rows -> the Keb BR-dup columns
    Keb, Kbr = p.Keb, p.Kbr
    torch.nan_to_num(J, nan=-1e6, posinf=-1e6, neginf=-1e6, out=st.J5[0:Ke])
    torch.mul(st.J5[0:Keb].view(Keb, 1).expand(Keb, Kbr), st.valid, out=st.SC[0, 0:Keb, p.C_EBD:p.C_EBD + Kbr])
    torch.mul(st.J5[Keb:Keb + K].view(K, 1).expand(K, Keb), st.valid, out=st.SC[0, :, p.C_BRD:p.C_BRD + Keb])
    torch.add(st.SC, st.S, alpha=p.rho, out=st.S)
    r = torch.argmax(st.S, dim=1)
    torch.gather(p.TAB, 1, r.view(2, 1, L), out=st.VALS)
    torch.gather(p.TABL, 1, r.view(2, 1, L), out=st.VL)
    return st.keb


def seed(p: KD2Plan, st: KD2State, cb_S):
    """Capturable (copy_ / fill_ only). cb_S: AW's (K,) score state of window 1 (or the prior)."""
    K, Ke, T = p.K, p.Ke, p.T
    S = st.S
    S.zero_()
    S[0, :, 0:3].copy_(cb_S.view(K, 1).expand(K, 3))
    if p.br:
        Keb, Kbr = p.Keb, p.Kbr
        S[0, 0:Keb, p.C_EBD:p.C_EBD + Kbr].copy_(p.eb_S0.view(Keb, 1).expand(Keb, Kbr))
        if Keb < K:
            S[0, Keb:, p.C_EBD:p.C_EBD + Kbr].fill_(float("-inf"))
        S[0, 0:Kbr, p.C_BRD:p.C_BRD + Keb].copy_(p.br_S0.view(Kbr, 1).expand(Kbr, Keb))
        if Kbr < K:
            S[0, Kbr:, p.C_BRD:p.C_BRD + Keb].fill_(float("-inf"))
    else:
        S[0, 0:Ke, p.C_EB].copy_(p.eb_S0)
        if Ke < K:
            S[0, Ke:, p.C_EB].fill_(float("-inf"))
    S[1, :, p.C_REP:p.C_REP + p.G].copy_(st._kwS0_t)
    S[1, :, p.C_KW:p.C_KW + T].copy_(st._kwS0_lead)
    S[1, :, p.C_KW + T:p.C_KW + 2 * T].copy_(st._kwS0_lead)
    st.VALS[0, 0, 0:3].copy_(p.cbw3)
    st.valid.fill_(0.0)


def prepare_seed_tables(p: KD2Plan, st: KD2State, device):
    st._kwS0_t = p.kwS0_rows.t().contiguous().to(device)                             # (K, G)
    st._kwS0_lead = p.kwS0_rows.t()[:, p.l2g].contiguous().to(device)                  # (K, T)
