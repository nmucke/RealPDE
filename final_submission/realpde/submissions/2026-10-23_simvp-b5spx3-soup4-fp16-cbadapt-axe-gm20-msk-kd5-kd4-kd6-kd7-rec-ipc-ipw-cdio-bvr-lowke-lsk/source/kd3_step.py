"""KD3 (experiments/kd3_step_diet, 2026-09-24): bit-exact kernel diet of the whole-step graph OUTSIDE the adaptive
modules (CONFIG "kd3_step"; used only by cuda_graph_wholestep.py's captured body -- windows 0/1, W01 and every
eager / fallback path keep the parent code).

Same ops on the same values, fewer launches:
  * backbone (packed fp16 core): every 1x1-conv GEMM is `baddbmm(C, W, X, out=C)` on a static output buffer that
    already holds the expanded bias -- the SAME cuBLAS call on the SAME values as `baddbmm(bias, W, X)`, whose
    first kernel is exactly that expand-copy. All 32 bias images are written by 2 copies at the top of the forward
    (one per spatial size) instead of 32.
  * backbone input: u, v are standardised straight into the fp16 input buffer (the division stores its float32
    quotient rounded to half, == div -> cat -> fold -> .half()); the pressure channel and the four context maps
    likewise -- no cat / fold copy / cast passes. Output de-standardised straight into a static buffer (no cat).
  * mega PRE: the three centre moments M0, M1, M2 and their inputs (a0, prev_basis, prev_basis2) live stacked,
    so the three differences and the three EMA updates are one launch each; the six second moments
    (Saa, Sab, S02, Sbb, S22, S12 -- products reordered to match) are ONE compound EMA over the stacked
    time-mean (was 18 kernels).
  * mega TAIL: the centre gains (g1, g2) are written into one stacked buffer; r*Saa once; g*basis/g*b2,
    g*M1/g*M2, 2g*Sab/2g*S02, g^2*Sbb/g^2*S22 as one launch each -- every product / sum in the parent's order.
All state stays trajectory-scoped (seeded by the whole-step wrapper's `_seed`).

K3E port (experiments/v22_verify, 2026-09-24): mega_body / _coefficients also carry AXE's br_adapt branches of
cuda_graph_megatail._body (ridge-candidate rows: e_t2 / z row gathers, constK, emitted row k_eb), verbatim ops."""
from __future__ import annotations

import torch
import torch.nn.functional as F

import lsk_head as _LSK  # LSK: head-input hook + s buffer (inert when off)

_ROWS_A = ("proj_in", "gate", "proj_out", "mlp1", "mlp2")


class KD3State:
    """Static buffers of the KD3 path. Raises if the configuration is not the covered one (the caller then keeps
    the parent code path)."""

    def __init__(self, model, T, H, W, C, device):
        base = model.base
        if type(base).__name__ != "GraphedCtxNormWrapper":
            raise RuntimeError("kd3: graphed ctx-norm wrapper expected")
        w = base.wrapped
        cu = getattr(w, "ctx_unet", None)
        pk = getattr(cu, "packed", None)
        if type(cu).__name__ != "PackedGenericCore" or type(pk).__name__ != "PackedSimVP1StageFP16" \
                or getattr(pk, "scope", None) != "all":
            raise RuntimeError("kd3: packed fp16 core (scope 'all') expected")
        cfg = model.cfg
        if not cfg.get("mu_vp") or not cfg.get("ctr_basis2", True):
            raise RuntimeError("kd3: mu_vp / ctr_basis2 expected on")
        self.pk = pk
        K = pk.K
        hp = pk._hp
        Hb, Wb = (H + 2 - pk.down_w.shape[-2]) // 2 + 1, (W + 2 - pk.down_w.shape[-1]) // 2 + 1
        self.hwA, self.hwB = Hb * Wb, H * W
        # ---- bias images: rows = (member, out-channel) of every 1x1 GEMM, grouped by spatial size
        self.out = {}
        colsA, colsB, spanA, spanB = [], [], [], []
        for i in range(pk.n_blocks):
            for n in _ROWS_A:
                b3 = hp(f"b{i}_{n}_B3")                    # (K, C_out, 1) half
                spanA.append((f"b{i}_{n}", b3.shape[1]))
                colsA.append(b3.reshape(-1, 1))
        for n in ("fuse", "head"):
            b3 = hp(f"{n}_B3")
            spanB.append((n, b3.shape[1]))
            colsB.append(b3.reshape(-1, 1))
        # KD5 (experiments/kd5_more, CONFIG "kd5_near"): NEAR-EXACT variants of the stem / down convs (unfold + one
        # GEMM, bias from the same prefilled images), dw2 (dilated depthwise as a space-to-batch plain depthwise conv)
        # and the W2 net's first conv (bias as a ones channel, 16 input channels). Accuracy-checked, not bit-exact.
        self.near = bool(cfg.get("kd5_near", False))
        if self.near:
            d2, p2 = pk.dil2, pk.pad2
            if Hb % d2 or Wb % d2 or p2 % d2 or pk.down_w.shape[-1] != 4 or pk.stem_w.shape[-1] != 3:
                raise RuntimeError("kd5: conv geometry not covered")
            if pk.down_w.shape[1] * K != pk.stem_w.shape[0]:
                raise RuntimeError("kd5: down conv groups != K")
            if hp("fuse_W3").shape[2] != 2 * pk.enc:
                raise RuntimeError("kd5: fuse input != [up, skip]")
            spanA.append(("down", pk.hid))
            colsA.append(hp("down_b").reshape(-1, 1))
            spanB.append(("stem", pk.stem_w.shape[0] // K))
            colsB.append(hp("stem_b").reshape(-1, 1))
            self.Wstem = hp("stem_w").reshape(1, pk.stem_w.shape[0], -1).contiguous()
            self.Wdown = hp("down_w").reshape(K, pk.hid, -1).contiguous()
            self.d2, self.hd, self.wd, self.Hb, self.Wb = d2, Hb // d2, Wb // d2, Hb, Wb
            Cb = K * pk.hid
            self.SB = torch.zeros(d2, d2, Cb, self.hd, self.wd, device=device, dtype=torch.float16)   # s2b dw2 input
            self.SBv = self.SB.permute(2, 3, 0, 4, 1)                                  # (C, hd, d, wd, d) == (C, h, w)
            self.AG = torch.zeros(1, Cb, Hb, Wb, device=device, dtype=torch.float16)  # dw2 * gate
        self.biasA = torch.cat(colsA, 0).contiguous()
        self.biasB = torch.cat(colsB, 0).contiguous()
        self.bufA = torch.zeros(self.biasA.shape[0], self.hwA, device=device, dtype=torch.float16)
        self.bufB = torch.zeros(self.biasB.shape[0], self.hwB, device=device, dtype=torch.float16)
        for buf, spans, hw in ((self.bufA, spanA, self.hwA), (self.bufB, spanB, self.hwB)):
            r = 0
            for n, co in spans:
                self.out[n] = buf[r:r + K * co].view(K, co, hw)
                r += K * co
        # ---- backbone input (fp16, channel-first) and output (float, de-standardised)
        n_in = pk.stem_w.shape[1]
        if n_in != T * C + 4:
            raise RuntimeError("kd3: stem input channels != T*C + 4")
        self.zh = torch.zeros(1, n_in, H, W, device=device, dtype=torch.float16)
        self.zV = self.zh[:, :T * C].view(1, T, C, H, W).permute(0, 1, 3, 4, 2)      # (1,T,H,W,C) view
        self.zE = self.zh[:, T * C:].permute(0, 2, 3, 1)                              # (1,H,W,4) view
        self.raw = torch.zeros(1, T, H, W, C, device=device, dtype=torch.float32)
        # ---- mega stacks
        f = lambda *s: torch.zeros(*s, device=device, dtype=torch.float32)  # noqa: E731
        self.ABC = f(3, 1, T, H, W, C)      # [a0 scratch, prev_basis, prev_basis2]
        self.M3 = f(3, 1, T, H, W, C)       # [M0, M1, M2]
        self.S6 = f(6, 1, 1, H, W, C)       # [Saa, Sab, S02, Sbb, S22, S12]
        self.G = f(2, 1, 1, H, W, C)        # [g1, g2]
        self.E2 = f(2, 1, 1, H, W, C)       # [e_true, e_pred]
        self.KP = f(1, 1, H, W, 1)          # next step's keep_prev = float(~mask)
        self.W2b = torch.zeros(2, 1, T, H, W, 1, device=device, dtype=torch.bool)   # [valid & keep, ~valid & keep]
        if cfg["blend"] != "region":
            raise RuntimeError("kd3: blend='region' expected")
        self.lam = float(cfg["blend_ridge"])
        # ke_net_flow features, written in place (fields | energy | context | summary)
        kf = model.ke_flow
        self.k_in = int(kf.cfg["k_in"])
        self.n_leads = len(list(kf.cfg["leads"]))
        n_feat = 2 * self.k_in + 2 * self.n_leads + 8 + 4 + 8
        if kf.net.stem.in_channels != n_feat or not kf.cfg.get("summary"):
            raise RuntimeError("kd3: ke_flow feature layout not covered")
        self.FEAT = f(1, n_feat, H, W)
        # cond-width net input (fp16), written by the normalising division
        self.INH = torch.zeros(2, T, 12, H, W, device=device, dtype=torch.float16)
        if self.near:
            # KD5: W2 c1 on 16 input channels: 12 features | ones (bias through the always-in-bounds centre tap) | 3 zeros
            c1 = model.cw.net.c1
            if tuple(c1.weight.shape[1:]) != (12, 3, 3) or c1.padding[0] != 1 or c1.dilation[0] != 1 or c1.bias is None:
                raise RuntimeError("kd5: W2 c1 geometry not covered")
            # input channel order [f0, l1, l2, l3, l4, l7, l8, lp, le, lr (T-constant) | f5 | f6 (static) | ones | 0 0 0]
            self.ORD = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 5, 6]
            cw = model.cw
            mu, sd = cw.mu.detach().float().reshape(-1), cw.sd.detach().float().reshape(-1)
            if mu.numel() != 12:
                raise RuntimeError("kd5: W2 feature count != 12")
            idx = torch.tensor(self.ORD[:10], device=mu.device)
            self.mu10 = mu.index_select(0, idx).view(10, 1, 1, 1).to(device).contiguous()
            self.sd10 = sd.index_select(0, idx).view(10, 1, 1, 1).to(device).contiguous()
            self.mu5, self.sd5 = mu[5].to(device).clone(), sd[5].to(device).clone()
            self.CT = f(10, 2, H, W)
            self.INH16 = torch.zeros(2, T, 16, H, W, device=device, dtype=torch.float16)
            lead = torch.arange(T, device=device, dtype=torch.float32) / (T - 1.0)       # == cond_width_net's f6
            f6n = (lead - mu[6].to(device)) / sd[6].to(device)
            self.INH16[:, :, 11].copy_(f6n.view(1, T, 1, 1).expand(2, T, H, W))
            self.INH16[:, :, 12].fill_(1.0)
            w16 = torch.zeros(c1.weight.shape[0], 16, 3, 3, device=device, dtype=torch.float16)
            w16[:, :12] = c1.weight.detach()[:, self.ORD].half()
            w16[:, 12, 1, 1] = c1.bias.detach().half()
            self.W2c1 = w16
        # [prev2, ema2, ratio] of the cond-width state (prev2 / ema2 are the whole-step wrapper's persistent maps)
        self.PE3 = f(3, 2, H, W)
        self.dla = (model.log_a[1] - model.log_a[0]).clone()      # constant k-table grid step (same op, once)
        self.kf_scale = float(model.ke_flow.scale)
        self.ridge = float(cfg["ctr_g_ridge"])
        lo, hi = cfg["ctr_g_clip"]
        self.lo, self.hi = float(lo), float(hi)


# ----------------------------------------------------------------------------------------------- backbone
def _pw(pk, st, x, name):
    B, KC, H, W = x.shape
    K = pk.K
    if B != 1:
        raise RuntimeError("kd3: B == 1 expected")
    Cb = st.out[name]
    y = torch.baddbmm(Cb, pk._hp(f"{name}_W3"), x.reshape(K, KC // K, H * W), out=Cb)
    return y.reshape(B, -1, H, W)


def _block(pk, st, i, x):
    K, hid = pk.K, pk.hid
    g = lambda n: pk._hp(f"b{i}_{n}")  # noqa: E731
    u = F.gelu(_pw(pk, st, F.group_norm(x, 8 * K, g("norm1_w"), g("norm1_b"), 1e-5), f"b{i}_proj_in"))
    a = F.conv2d(u, g("dw1_w"), g("dw1_b"), padding=pk.pad1, groups=K * hid)
    a = F.conv2d(a, g("dw2_w"), g("dw2_b"), padding=pk.pad2, dilation=pk.dil2, groups=K * hid)
    gt = torch.sigmoid(_pw(pk, st, u, f"b{i}_gate"))
    x = x + _pw(pk, st, a * gt, f"b{i}_proj_out")
    h = F.gelu(_pw(pk, st, F.group_norm(x, 8 * K, g("norm2_w"), g("norm2_b"), 1e-5), f"b{i}_mlp1"))
    x = x + _pw(pk, st, h, f"b{i}_mlp2")
    return x


def _block_near(pk, st, i, x):
    """KD5 near-exact twin of _block: dw1 bias written straight into the space-to-batch layout of dw2's input; dw2
    (dilation d) as a plain depthwise conv over the d*d phase sub-images (same taps, cudnn kernel); `a * gate`
    written back into the standard layout by the multiplication itself (no extra copies)."""
    K, hid = pk.K, pk.hid
    C = K * hid
    d, hd, wd = st.d2, st.hd, st.wd
    g = lambda n: pk._hp(f"b{i}_{n}")  # noqa: E731
    u = F.gelu(_pw(pk, st, F.group_norm(x, 8 * K, g("norm1_w"), g("norm1_b"), 1e-5), f"b{i}_proj_in"))
    y1 = F.conv2d(u, g("dw1_w"), None, padding=pk.pad1, groups=C)
    torch.add(y1.view(C, hd, d, wd, d), g("dw1_b").view(C, 1, 1, 1, 1), out=st.SBv)
    a = F.conv2d(st.SB.view(d * d, C, hd, wd), g("dw2_w"), g("dw2_b"), padding=pk.pad2 // d, groups=C)
    gt = torch.sigmoid(_pw(pk, st, u, f"b{i}_gate"))
    torch.mul(a.view(d, d, C, hd, wd).permute(2, 3, 0, 4, 1), gt.view(C, hd, d, wd, d), out=st.AG.view(C, hd, d, wd, d))
    x = x + _pw(pk, st, st.AG, f"b{i}_proj_out")
    h = F.gelu(_pw(pk, st, F.group_norm(x, 8 * K, g("norm2_w"), g("norm2_b"), 1e-5), f"b{i}_mlp1"))
    x = x + _pw(pk, st, h, f"b{i}_mlp2")
    return x


def packed_forward(pk, st, zh):
    """== PackedGenericCore.forward -> PackedSimVP1StageFP16.forward (scope "all") on the already-assembled
    fp16 input `zh`; returns the (1,T,H,W,3) standardised output (a view)."""
    m = torch.backends.cuda.matmul
    old = (m.allow_tf32, m.allow_fp16_reduced_precision_reduction)
    m.allow_tf32 = True
    m.allow_fp16_reduced_precision_reduction = False
    try:
        K, enc = pk.K, pk.enc
        B, _, H, W = zh.shape
        hp = pk._hp
        st.bufA.copy_(st.biasA.expand(-1, st.hwA))
        st.bufB.copy_(st.biasB.expand(-1, st.hwB))
        if st.near:
            # KD5: stem / down convs as unfold + one GEMM on the prefilled bias images (near-exact)
            Cs = st.out["stem"].view(1, -1, st.hwB)
            ys = torch.baddbmm(Cs, st.Wstem, F.unfold(zh, 3, padding=1), out=Cs).view(B, -1, H, W)
            skip = F.gelu(F.group_norm(ys, 8 * K, hp("stem_gw"), hp("stem_gb"), 1e-5))
            Cd = st.out["down"]
            yd = torch.baddbmm(Cd, st.Wdown, F.unfold(skip.view(K, enc, H, W), 4, padding=1, stride=2), out=Cd)
            h = F.gelu(F.group_norm(yd.view(B, -1, st.Hb, st.Wb), 8 * K, hp("down_gw"), hp("down_gb"), 1e-5))
            for i in range(pk.n_blocks):
                h = _block_near(pk, st, i, h)
        else:
            skip = F.gelu(F.group_norm(F.conv2d(zh, hp("stem_w"), hp("stem_b"), padding=1),
                                       8 * K, hp("stem_gw"), hp("stem_gb"), 1e-5))
            h = F.gelu(F.group_norm(F.conv2d(skip, hp("down_w"), hp("down_b"), stride=2, padding=1, groups=K),
                                    8 * K, hp("down_gw"), hp("down_gb"), 1e-5))
            for i in range(pk.n_blocks):
                h = _block(pk, st, i, h)
        h = F.conv_transpose2d(h, hp("up_w"), hp("up_b"), stride=2, padding=1, groups=K)
        if st.near:
            # KD5: fuse 1x1 GEMM over [up, skip] as two accumulating GEMMs on the column halves of W (no cat copy)
            Cf = st.out["fuse"]
            Wf = hp("fuse_W3")
            torch.baddbmm(Cf, Wf[:, :, :enc], h.view(K, enc, H * W), out=Cf)
            yf = torch.baddbmm(Cf, Wf[:, :, enc:], skip.view(K, enc, H * W), out=Cf).view(B, -1, H, W)
        else:
            hc = torch.cat([h.view(B, K, enc, H, W), skip.view(B, K, enc, H, W)], dim=2)
            hc = hc.view(B, K * 2 * enc, H, W)
            yf = _pw(pk, st, hc, "fuse")
        h = F.gelu(F.group_norm(yf, 8 * K, hp("fuse_gw"), hp("fuse_gb"), 1e-5))
        if _LSK.HOOK is not None:
            _LSK.HOOK(h)   # LSK: head input -> phi
        out = torch.mean(_pw(pk, st, h, "head").view(B, K, pk.out_ch, H, W), dim=1, dtype=torch.float32)
    finally:
        m.allow_tf32, m.allow_fp16_reduced_precision_reduction = old
    b, tc, h_, w_ = out.shape
    return out.reshape(b, tc // 3, 3, h_, w_).permute(0, 1, 3, 4, 2)


def ctx_body(base, x, st, rec=None):
    """== GraphedCtxNormWrapper._body(x) (causal ctx-norm + packed backbone + de-standardisation); returns the
    static `st.raw` buffer holding the same values as the parent's fresh output.
    REC (rec_detect.py): with `rec`, the accumulation is `acc = win + acc * keep` (keep = 0 restarts the accumulator
    from this window; keep = 1 is the parent's `acc + win` bit for bit: acc * 1.0 is exact, then one rounding), the window
    sum and the causal mean m are written into rec.MW (same ops), and the detector runs between the two."""
    from cuda_graph_ctxnorm import _causal_ms_graphed
    w = base.wrapped
    T = x.shape[1]
    uv = x[..., :2]
    if rec is None:
        window_sum1 = uv.sum(dim=1)
    else:
        window_sum1 = torch.sum(uv, dim=1, out=rec.MW[1])    # same reduction, into the REC buffer
        rec.detect()                                          # reads MW = [previous m, this window's sum]
    window_sum2 = (uv * uv).sum(dim=1)
    if rec is None:
        base._static_S1.add_(window_sum1)
        base._static_S2.add_(window_sum2)
        base._static_n.add_(float(T))
        m, s = _causal_ms_graphed(base._static_S1, base._static_S2, base._static_n, w._ms_floor)
        _LSK.write_s(s)   # LSK
    else:
        torch.addcmul(window_sum1, base._static_S1, rec.keep, out=base._static_S1)
        torch.addcmul(window_sum2, base._static_S2, rec.keep, out=base._static_S2)
        torch.addcmul(rec.T_t, base._static_n, rec.keep, out=base._static_n)
        recip = base._static_n.reciprocal()                   # == _causal_ms_graphed, m written into rec.MW[0]
        m = torch.mul(base._static_S1, recip, out=rec.MW[0])
        var = base._static_S2 * recip - m * m
        var = torch.clamp_min(var, w._ms_floor)
        s = torch.sqrt(var) if _LSK.S_BUF is None else torch.sqrt(var, out=_LSK.S_BUF)   # LSK: s -> S_BUF (same kernel)
    ms, ss = m.unsqueeze(1), s.unsqueeze(1)
    # standardise_input + _fold_window + cat(extra) + .half(), written in place
    torch.div(x[..., :2] - ms, ss, out=st.zV[..., :2])
    st.zV[..., 2:3].copy_(x[..., 2:3])
    logs = torch.log(s)
    torch.div(m - w._m_mean, w._m_std, out=st.zE[..., 0:2])
    torch.div(logs - w._logs_mean, w._logs_std, out=st.zE[..., 2:4])
    y_s = packed_forward(st.pk, st, st.zh)
    # unstandardise_output, written in place
    raw = st.raw
    torch.add(y_s[..., :2] * ss, ms, out=raw[..., :2])
    raw[..., 2:3].copy_(y_s[..., 2:3])
    return raw


# ----------------------------------------------------------------------------------------------- mega
def sync_state_in(mega, st):
    """== GraphedMegaWrapper._sync_state_in, with M0..M2 / the six second moments written into the stacks."""
    m = mega.model
    mega._mu.copy_(m.mu)
    mega._var.copy_(m.var)
    st.M3[0].copy_(m.M0)
    st.M3[1].copy_(m.M1)
    st.M3[2].copy_(m.M2)
    st.S6[0].copy_(m.Saa)
    st.S6[1].copy_(m.Sab)
    st.S6[2].copy_(m.S02)
    st.S6[3].copy_(m.Sbb)
    st.S6[4].copy_(m.S22)
    st.S6[5].copy_(m.S12)
    for k in mega._S:
        mega._S[k].copy_(m.S[k])
    st.E2[0].copy_(m.e_true)
    st.E2[1].copy_(m.e_pred)
    mega._ex_prev.copy_(m.ke_flow.ex_prev)
    mega._ef_prev.copy_(m.ke_flow.ef_prev)


def bind_io(io, st):
    """Point the whole-step I/O namespace's moment state at the stacks."""
    io._M0, io._M1, io._M2 = st.M3[0], st.M3[1], st.M3[2]
    io._e_true, io._e_pred = st.E2[0], st.E2[1]
    io._Saa, io._Sab, io._S02, io._Sbb, io._S22, io._S12 = st.S6.unbind(0)


def _advect(s, fl_last, obstacle, nm, t, out_adv):
    """== GainMapImprovedModel._advect with `~mask` passed in (nm) and the result written into `out_adv`."""
    w = s.shift.w
    uc = s.uc if s.uc_eff is None else s.uc_eff
    d = uc * torch.arange(1, t + 1, device=fl_last.device, dtype=fl_last.dtype)   # (T,)
    src = s.shift.cols.view(1, -1) - d.view(-1, 1)                                   # (T, W)
    i0 = torch.floor(src)
    wgt = (src - i0).view(1, 1, t, w, 1)
    i0 = i0.long()
    valid = (i0.unsqueeze(1) >= obstacle.long().unsqueeze(0) + 1) & (i0 >= 0).unsqueeze(1) & nm.unsqueeze(0)
    i0c, i1c = i0.clamp(0, w - 1), (i0 + 1).clamp(0, w - 1)
    adv = (1.0 - wgt) * fl_last[:, :, i0c, :] + wgt * fl_last[:, :, i1c, :]              # (1, H, T, W, C)
    torch.mul(adv.permute(0, 2, 1, 3, 4), valid.view(1, t, *nm.shape, 1).to(fl_last.dtype), out=out_adv)
    out_adv[..., 2] = 0.0
    return out_adv, valid.view(1, t, *nm.shape, 1)


def _coefficients(st, valid, nm, S5, lam=None):
    """== cuda_graph_megatail._kd_coefficients (GainMapImprovedModel._coefficients, "region") with the two
    regions' shared p11 / r1 computed once on the stacked sums, the region masks written into one bool buffer,
    `* a0` (a0 = 1.0, an exact identity) dropped, the dead zeros of the invalid region's b not built, and
    `where(valid, b_v, zeros_like)` as `where(valid, b_v, 0.0)`. Also stores next step's keep_prev.
    AXE (V22 port): lam may be br_adapt's (K_br,1,1,1,1) ridge-candidate tensor (== the parent's
    `_kd_coefficients(..., lam=br.lamv)`: same element-wise ops, one output row per candidate); None = CONFIG float."""
    lam = st.lam if lam is None else lam
    keep = nm.view(1, 1, *nm.shape, 1)
    W2b = st.W2b
    torch.bitwise_and(valid, keep, out=W2b[0])
    torch.bitwise_and(~valid, keep, out=W2b[1])
    w2 = W2b.to(S5.dtype)                                                   # == stack([v, i], 0)
    allsums = (S5.unsqueeze(1) * w2.unsqueeze(0)).sum(dim=(4, 5), keepdim=True)   # (5, 2, 1, T, 1, 1, C)
    ff, aa, fa, fy, ay = allsums.unbind(0)                                  # [region v, region i] each
    p11 = ff * (1.0 + lam)
    r1 = fy + lam * ff
    p11v, r1v, fav, aav = p11[0], r1[0], fa[0], aa[0]
    p22 = aav * (1.0 + lam)
    r2 = ay[0] + lam * aav * 0.0
    det = (p11v * p22 - fav * fav).clamp_min(1e-3 * p11v * p22 + 1e-30)
    av = (p22 * r1v - fav * r2) / det
    bv = (p11v * r2 - fav * r1v) / det
    ai = r1[1] / (p11[1] + 1e-30)
    a = torch.where(valid, av, ai)
    b = torch.where(valid, bv, 0.0)
    a[..., 2] = 1.0
    b[..., 2] = 0.0
    return a, b


def features_kd3(st, xw, ffl, ex, ef, runx, runf, mask, wfrac, leads_idx, halves, ex_prev, ef_prev, mxk):
    """== ke_net_flow.features_kd(..., summary=True) written straight into the static (1, 42, H, W) buffer
    (no stack / cat copies); `mxk` = xw's temporal mean (passed in: == the TAIL's x mean, channels :2)."""
    import ke_net_flow as KF
    EPS, FC, LC = KF.EPS, KF.FIELD_CLAMP, KF.LOG_CLAMP
    FT = st.FEAT
    H, W = xw.shape[2], xw.shape[3]
    k, L = st.k_in, st.n_leads
    mrx = runx.mean(dim=(1, 2), keepdim=True)
    sx = mrx.clamp_min(EPS).sqrt()
    sf = runf.mean(dim=(1, 2), keepdim=True).clamp_min(EPS).sqrt()
    nk, nl = 2 * k, 2 * L
    slot = lambda c0, n: FT[:, c0:c0 + 2 * n].view(1, n, 2, H, W).permute(0, 1, 3, 4, 2)  # noqa: E731
    torch.div(xw[:, -k:] - mxk, sx.unsqueeze(1), out=slot(0, k))
    torch.div(ffl.index_select(1, leads_idx), sf.unsqueeze(1), out=slot(nk, L))
    FT[:, 0:nk + nl].clamp_(-FC, FC)
    ex1, ex2, ef1, ef2 = halves
    nums = [ef, ex, runx, runx, ex2, ef2, ex, ex]
    dens = [runf, runx, runf, mrx.expand_as(runx), ex1, ef1, ef_prev, ex_prev]
    lg = torch.log((torch.stack(nums, 0) + EPS) / (torch.stack(dens, 0) + EPS))     # (8, 1, H, W, 2)
    c0 = nk + nl
    en = lambda c: FT[:, c:c + 8].view(1, 4, 2, H, W).permute(1, 0, 3, 4, 2)          # noqa: E731
    torch.clamp(lg[0:4], -LC, LC, out=en(c0))
    mx = mxk.squeeze(1)
    FT[:, c0 + 8].copy_(mask)
    FT[:, c0 + 9].copy_(mx[..., 0])
    torch.mul(mx[..., 1], 6.0, out=FT[:, c0 + 10])
    FT[:, c0 + 11].copy_(wfrac.view(-1, 1, 1).expand_as(mask))
    torch.clamp(lg[4:8], -LC, LC, out=en(c0 + 12))
    return FT


def _kd5_features(st, x_phys, pred_phys, sp, prev2, T, H, W):
    """KD5 (near path): cond_width_net._kd_base_feature_list + the three state logs + the normalisation, with the
    ten T-constant features normalised ONCE on their compact (10, 2, H, W) stack and broadcast over T by one copy
    into the fp16 net input (per element the same float ops as the parent: (log(.) - mu_i) / sd_i, then rounded
    to half); f5 normalised straight into its channel; f6 (lead time) is static (written at construction)."""
    import cond_width_net as _CW
    EPS = _CW.EPS
    b = x_phys.shape[0]
    xp = torch.stack([x_phys, pred_phys], 0)                 # (2,b,T,H,W)
    means = xp.mean(dim=2)
    stds = xp.std(dim=2, unbiased=True)
    xm, meanpred = means[0], means[1]
    sx, pred_std = stds[0], stds[1]
    last = x_phys[:, -1]
    gm = _CW._t2_grad_mag(torch.stack([last, meanpred], 0))
    bx = _CW._box(torch.stack([sx, sx * sx], 0).reshape(2 * b, H, W), 3).view(2, b, H, W)
    var_local = (bx[1] - bx[0] * bx[0]).clamp_min(0.0)
    CT = st.CT                                               # rows [f0, l1, l2, l3, l4, l7, l8, lp, le, lr]
    torch.log(torch.stack([sx, (last - xm).abs(), gm[0], pred_std, torch.sqrt(var_local), gm[1]], 0) + EPS, out=CT[1:7])
    torch.log(sp[:, 0] + EPS, out=CT[0])
    PE3 = st.PE3                                             # rows [prev2, ema2, ratio]: prev2 / ema2 ARE rows 0, 1
    torch.clamp_min(prev2 / sp[:, 0].clamp_min(EPS), 0.0, out=PE3[2])
    torch.log(PE3 + EPS, out=CT[7:10])
    dp = torch.empty_like(pred_phys)
    dp[:, 0] = pred_phys[:, 0] - last
    dp[:, 1:] = pred_phys[:, 1:] - pred_phys[:, :-1]
    f5 = torch.log(dp.abs() + EPS)
    CTn = torch.div(CT - st.mu10, st.sd10)
    st.INH16[:, :, 0:10].copy_(CTn.permute(1, 0, 2, 3).unsqueeze(1).expand(b, T, 10, H, W))
    torch.div(f5 - st.mu5, st.sd5, out=st.INH16[:, :, 10])


def cw_apply(m, cw, k_scale, mean, var, x, pred_out, prev2, ema2, out_lower, out_upper, st):
    """== t2_cw_graph_body.t2_cw_apply with the normalised features divided straight into the fp16 net input
    (the float32 quotient rounded to half on store == the parent's div -> .half())."""
    import cond_width_net as _CW
    from submission import _SIGMA_GLOBAL
    std1 = var.sqrt()
    a = 2.0 * std1 * m.std_tgt / _SIGMA_GLOBAL
    pos = (a.clamp_min(1e-12).log() - m.log_a[0]) / st.dla
    k = m.k_tab[pos.round().long().clamp(0, len(m.k_tab) - 1)]
    half1 = k_scale * k * std1
    T, H, W = mean.shape[1], mean.shape[2], mean.shape[3]
    sig_pipe_phys = std1[..., :2] * m.std_tgt[:2]
    x_phys = (x[..., :2] * m.std_in[:2] + m.mean_in[:2]).squeeze(0).permute(3, 0, 1, 2)
    pred_phys = (pred_out[..., :2] * m.std_tgt[:2] + m.mean_tgt[:2]).squeeze(0).permute(3, 0, 1, 2)
    sp = sig_pipe_phys.squeeze(0).permute(3, 0, 1, 2)
    if st.near:
        _kd5_features(st, x_phys, pred_phys, sp, prev2, T, H, W)
    else:
        base = _CW._kd_base_feature_list(x_phys, pred_phys, sp.expand(-1, T, -1, -1))
        PE3 = st.PE3                                   # rows [prev2, ema2, ratio]: prev2 / ema2 ARE rows 0, 1
        torch.clamp_min(prev2 / sp[:, 0].clamp_min(_CW.EPS), 0.0, out=PE3[2])
        lp, le, lr = torch.log(PE3 + _CW.EPS).unbind(0)                    # == _kd_logs([prev2, ema2, ratio]) minus the stack
        ex = lambda t: t.unsqueeze(1).expand(-1, T, -1, -1)  # noqa: E731
        feats = torch.stack(base + [ex(lp), ex(le), ex(lr)], dim=2)
        torch.div(feats - cw.mu.view(1, 1, -1, 1, 1), cw.sd.view(1, 1, -1, 1, 1), out=st.INH)
    net_h = getattr(cw, "_kd_net_half", None)
    if net_h is None:
        import copy
        net_h = copy.deepcopy(cw.net).half()
        cw._kd_net_half = net_h
    if st.near:
        # KD5: c1 on [features | ones | zeros] with the bias in the ones channel's centre tap (near-exact)
        n_ = net_h
        hh = n_.act(n_.g1(F.conv2d(st.INH16.view(2 * T, 16, H, W), st.W2c1, None, padding=1)))
        hh = n_.act(n_.g2(n_.c2(hh)))
        hh = n_.act(n_.g3(n_.c3(hh)))
        mm = n_.out(hh).float().view(2, T, H, W)
    else:
        inp = st.INH.view(2 * T, -1, H, W)
        mm = net_h(inp).float().view(2, T, H, W)
    mult = torch.exp(cw.alpha * mm).clamp(cw.clip_lo, cw.clip_hi) * cw.lam
    mult2 = mult.permute(1, 2, 3, 0).unsqueeze(0)
    out_half = torch.cat([half1[..., :2] * mult2, half1[..., 2:].expand(1, T, H, W, 1)], dim=-1)
    torch.sub(mean, out_half, out=out_lower)
    torch.add(mean, out_half, out=out_upper)


def _ctr_gain2(st):
    """== GainMapImprovedModel._ctr_gain2 on the stacked moments (r*Saa once); writes (g1, g2) into st.G."""
    S6, r = st.S6, st.ridge
    rS = r * S6[0]
    P = S6[3:5] + rS                                   # [Sbb + r Saa, S22 + r Saa] = [p11, p22]
    p11, p22 = P[0], P[1]
    r1 = S6[1] + rS
    r2 = S6[2]
    S12 = S6[5]
    det = (p11 * p22 - S12 * S12).clamp_min(1e-3 * p11 * p22 + 1e-30)
    torch.clamp((p22 * r1 - S12 * r2) / det, st.lo, st.hi, out=st.G[0])
    torch.clamp((p11 * r2 - S12 * r1) / det, -st.hi, st.hi, out=st.G[1])
    return st.G[0], st.G[1]


def mega_body(mega, b, st):
    """== GraphedMegaWrapper._body(b) (PRE + energy + TAIL) with the stacked moment state (see module doc)."""
    from cuda_graph_megatail import _ke
    GainMapImprovedModel = mega._GainMapImprovedModel
    s = b.shadow
    x, mask, prev_mask = b._x, b._mask_in, b._prev_mask
    fl_f, mf = b._fl_f, b._mf
    err, y, blend_y = b._err, b._y, b._blend_y
    prev_fl, prev_adv = b._prev_fl, b._prev_adv
    prev_const = b._prev_const
    ABC, M3, S6 = st.ABC, st.M3, st.S6

    # ================= PRE half =================
    P = mega._P7
    d = err - b._mu
    new_mu = torch.add(b._mu, b._a_err_f32 * d, out=b._mu)
    torch.mul(d, d, out=P[0])

    torch.sub(y, prev_const, out=ABC[0])                  # a0; rows 1, 2 are prev_basis, prev_basis2
    dABC = ABC - M3                                        # [da, db, dc]
    torch.add(M3, b._a_ctr_f32 * dABC, out=M3)             # new_M0, new_M1, new_M2 in place
    da, db, dc = dABC[0], dABC[1], dABC[2]
    new_M0, new_M1, new_M2 = M3[0], M3[1], M3[2]
    # second-moment products in S6 order [Saa, Sab, S02, Sbb, S22, S12]
    torch.mul(da, da, out=P[1])
    torch.mul(da, db, out=P[2])
    torch.mul(da, dc, out=P[3])
    torch.mul(db, db, out=P[4])
    torch.mul(dc, dc, out=P[5])
    torch.mul(db, dc, out=P[6])
    Pm = P.mean(dim=2, keepdim=True)                       # (7,B,1,H,W,C)

    om4 = b._om4                                           # == 1.0 - [a_err, a_ctr, ...] (float64, one launch)
    new_var = torch.mul(om4[0], b._var + b._a_err_f64 * Pm[0], out=b._var)  # noqa: F841
    a64c = b._a_ctr_f64
    om = om4[1]
    torch.mul(om, S6 + a64c * Pm[1:7], out=S6)
    new_Saa, new_Sab, new_S02, new_Sbb, new_S22, new_S12 = S6.unbind(0)

    keep_prev = st.KP                                      # == (~prev_mask).view(1,1,H,W,1).to(float), set last step
    Cs = mega._C5
    torch.mul(prev_fl, prev_fl, out=Cs[0])
    torch.mul(prev_adv, prev_adv, out=Cs[1])
    torch.mul(prev_fl, prev_adv, out=Cs[2])
    torch.mul(prev_fl, blend_y, out=Cs[3])
    torch.mul(prev_adv, blend_y, out=Cs[4])
    Cs.mul_(keep_prev)
    S5 = b._S5
    torch.add(S5, b._a_bld_f32 * (Cs - S5), out=S5)
    new_S = b._S

    # ================= energy / ke_net_flow half (verbatim) =================
    kx = getattr(b, "_kx", None)
    if kx is None:
        kx = _ke(x)
    if mega._mu_vp:
        mu_p = torch.sub(new_mu, new_mu.mean(dim=1, keepdim=True), out=b._fm[1:2])
        v2 = b._fm.var(dim=1, keepdim=True, unbiased=False)
        e_f, vM = v2[0:1], v2[1:2]
    else:
        e_f = _ke(fl_f)
    d_et = kx - b._e_true
    new_e_true = torch.add(b._e_true, b._a_e_f32 * d_et, out=b._e_true)
    d_ep = e_f - b._e_pred
    new_e_pred = torch.add(b._e_pred, b._a_e_f32 * d_ep, out=b._e_pred)

    E8 = st.E2 + 1e-8                                      # [e_true + 1e-8, e_pred + 1e-8] (E2 = [new_e_true, new_e_pred])
    g2 = (E8[0] / E8[1]).pow(mega._gain_power)

    ex, ef = kx[:, 0, ..., :2], e_f[:, 0, ..., :2]
    xw, ffl = x[..., :2], fl_f[..., :2]
    rx, rf = new_e_true[:, 0, ..., :2], new_e_pred[:, 0, ..., :2]
    Hh, Wh, Ch = x.shape[2], x.shape[3], x.shape[4]
    xh = x.view(1, 2, 10, Hh, Wh, Ch)[..., :2].var(dim=2, unbiased=False)
    fh = fl_f.view(1, 2, 10, Hh, Wh, Ch)[..., :2].var(dim=2, unbiased=False)
    halves = (xh[:, 0], xh[:, 1], fh[:, 0], fh[:, 1])
    import ke_net_flow
    mx_ = x.mean(dim=1, keepdim=True)                      # (TAIL's; also == features' xw temporal mean)
    feat = features_kd3(st, xw, ffl, ex, ef, rx, rf, mask[None], b._wfrac, mega._leads_idx, halves,
                        b._ex_prev, b._ef_prev, mx_[..., :2])
    nm = ~mask                                             # shared by apply_correction, _advect, _coefficients
    st.KP.copy_(nm.view(st.KP.shape))                      # float(~mask): here and next step's keep_prev (PRE read above)
    EPS = ke_net_flow.EPS
    LR = torch.log(st.E2[:, :, 0, ..., :2] + EPS)          # [log(runx + EPS), log(runf + EPS)] in one pass
    log_rule = torch.log(ef + EPS) + 0.75 * (LR[0] - LR[1])            # == rule_log_energy(ef, rx, rf)
    corr = mega._kf_net(feat).clamp(-1.5, 1.5)                          # == apply_correction(corr, log_rule, mask[None], scale)
    if st.kf_scale != 1.0:
        corr = corr * st.kf_scale
    corr = corr * st.KP.view(1, *mask.shape, 1)
    e_new = (log_rule + corr).exp()
    b._ex_prev.copy_(ex)
    b._ef_prev.copy_(ef)
    e_t2 = e_new.unsqueeze(1)
    eb = getattr(b, "_eb_ad", None)
    br = getattr(b, "_br_ad", None)                        # AXE (axe_adapt.py): + K_br ridge rows (factorised); None = off
    if eb is not None:
        e_t2 = eb.wv * e_t2 + eb.omv * b._R_run[..., :2]
        if br is not None:
            e_t2 = e_t2.index_select(0, b._br_eidx)        # (Kc,1,H,W,2)
    elif mega._e_blend_w is not None:
        e_t2 = mega._e_blend_w * e_t2 + (1.0 - mega._e_blend_w) * b._R_run[..., :2]
    g_flow = (e_t2 + 1e-10) / (e_f[..., :2] + 1e-10)
    if eb is not None:
        g2 = torch.cat([g_flow, g2[..., 2:].expand(eb.K if br is None else br.Kc, -1, -1, -1, -1)], dim=-1)
    else:
        g2 = torch.cat([g_flow, g2[..., 2:]], dim=-1)
    g_max = mega._gain_max
    gamma = g2.clamp(1.0 / g_max ** 2, g_max ** 2).sqrt()

    # ================= TAIL half =================
    s.S = new_S
    s.M0, s.M1, s.M2 = new_M0, new_M1, new_M2
    s.Saa, s.Sab, s.Sbb = new_Saa, new_Sab, new_Sbb
    s.S02, s.S12, s.S22 = new_S02, new_S12, new_S22

    T = x.shape[1]
    obstacle = s.shift.last_obstacle(mask)
    fl_x = x - mx_
    adv, valid = _advect(s, fl_x[:, -1], obstacle, nm, T, b._out_adv)    # adv IS b._out_adv (prev_adv, read in PRE only)
    axe = eb is not None and br is not None
    if axe:
        # AXE: (a, b) for every ridge candidate at once, then one z row per candidate row (factorised)
        a_, b_ = _coefficients(st, valid, nm, b._S5, lam=br.lamv)
        z = (a_ * fl_f + b_ * adv).index_select(0, b._br_zidx)            # (Kc,T,H,W,C)
    else:
        a_, b_ = _coefficients(st, valid, nm, b._S5)
        z = a_ * fl_f + b_ * adv

    mz = z.mean(dim=1, keepdim=True)
    zc = zc_ls = z - mz
    if axe:
        constK = mf + mz                                                  # (Kc,1,H,W,C)
    else:
        const = torch.add(mf, mz, out=b._out_const)
    e_z = _ke(zc)
    c_zf = (zc * fl_f).mean(dim=1, keepdim=True)
    e_tgt2 = gamma * gamma * e_f
    fluc = GainMapImprovedModel._shape(s, zc, fl_f, e_z, e_f, c_zf, e_tgt2)
    if mega._mu_vp:
        e_tgt_ach = _ke(fluc)
        cFM = (fluc * mu_p).mean(dim=1, keepdim=True)
        disc = (cFM * cFM - e_tgt_ach * vM + e_tgt_ach * e_tgt_ach).clamp_min(0.0)
        c = ((-cFM + disc.sqrt()) / e_tgt_ach.clamp_min(1e-12)).clamp_min(0.0)
        fluc = c * fluc
        if axe:
            # AXE: keep every row's output for the next step's scores; emit row k_eb = (b*, lambda*)
            outK = torch.add(constK + fluc, new_mu, out=b._eb_outK)
            torch.index_select(outK, 0, b._eb_k, out=b._out_out)
            fluc = fluc.index_select(0, b._eb_k)
            zc_ls = zc.index_select(0, b._eb_k)
            const = torch.index_select(constK, 0, b._eb_k, out=b._out_const)
        elif eb is not None:
            outK = torch.add(const + fluc, new_mu, out=b._eb_outK)
            torch.index_select(outK, 0, b._eb_k, out=b._out_out)
            fluc = fluc.index_select(0, b._eb_k)
        else:
            torch.add(const + fluc, new_mu, out=b._out_out)
    else:
        raise RuntimeError("kd3: mu_vp expected")

    w = mega._ctr_w
    torch.add((1.0 - w) * fluc, w * zc_ls, out=b._out_basis)       # == ABC[1]
    torch.sub(adv, adv.mean(dim=1, keepdim=True), out=b._out_b2)   # == ABC[2]
    G = st.G
    g1, g2c = _ctr_gain2(st)                                        # noqa: F841
    GB = G * ABC[1:3]                                               # [g1 * basis, g2 * b2]
    ctr = const + GB[0] + GB[1]
    GM = G * M3[1:3]                                                # [g1 * M1, g2 * M2]
    torch.sub(ctr + new_M0 - GM[0], GM[1], out=b._out_ctr_out)
    G2 = 2.0 * G                                                    # [2 g1, 2 g2]
    AB = G2 * S6[1:3]                                               # [2 g1 Sab, 2 g2 S02]
    GG = G * G
    CE = GG * S6[3:5]                                               # [g1 g1 Sbb, g2 g2 S22]
    D = G2[0] * g2c * new_S12
    torch.clamp_min(new_Saa - AB[0] - AB[1] + CE[0] + D + CE[1], 1e-12, out=b._out_ctr_var)
