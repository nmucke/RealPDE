"""KD6 (experiments/kd6_concurrency, 2026-09-25): fork / join concurrency inside the steady whole-step CUDA graph
(cuda_graph_wholestep.py, noscan graph; CONFIG "kd6_concurrency"). The same values, a different schedule (the same
kernels, bar one row-split of the cond-width feature statistics).

The KD2 + KD3 (+ KD5) + REC whole-step body is one serial stream of ~700 latency-bound kernels on a mostly idle GPU.
About 280 of them do not depend on the backbone output at all (oracle: experiments/kd6_concurrency/scripts/
kd6_oracle.py -- per-op read / write byte extents -> dependency DAG). Here they run on a SIDE stream, concurrently
with the ctx-norm + backbone forward on the main stream:
  side, forked at the top of the body:
    * the step() head: err, adapt_loss, blend_y, counters +1, kxy (Welford var of the input pair), KD2's cond-width
      update and the fused AW / KW / EB / BR selection;
    * the mask;
    * after an event recorded on main right after REC's detector (inside ctx_body, before the backbone):
      REC rescale of (mu, M0..M2), counters / EMA coefficients, R_run;
    * mega's PRE half (mu, var, centre moments, second moments, blend statistics), e_true, the x halves, the x mean,
      the x / mask / wfrac channels of the ke-flow features, ~mask and next step's keep_prev, obstacle, _advect,
      the blend coefficients, out_b2, the centre gains, the ctr_var chain, cw_apply's k-table / half1 / x_phys, and
      the x / sig_pipe / prev2-only part of the KD5 cond-width features (x row of the stacked [x_phys, pred_phys]
      buffer with its per-row T-mean / T-std, grad-magnitude of the last frame, box / local variance, CT[0], the
      ratio row PE3[2] and CT[7:10]; per-row reductions == the parent's joint ones bit for bit, checked).
  main: ctx_body (window sums, detector [event], accumulators, input assembly, packed backbone, output), then --
    after waiting for the side's blend_y (it reads the previous mf) -- mf, fl_f; then the join; then everything that
    reads the backbone output, in the parent's order (mu_p / e_f, e_pred, features' fl_f part, ke net, the blend /
    _shape / fluc / outputs, ctr_out, W2 features + net, CB shifts).
  second fork (after the ke-flow features are complete), side || the ke_net_flow net on main: the gain-2 ratio, the
    rule log-energy, the (ex_prev, ef_prev) carry, the blend candidates z / zc / constK / e_z / c_zf, the prev_fl
    carry; joined before e_new (the first reader of the net's output).
  third fork (after the emitted prediction is written), side || the cond-width features + W2 net on main: the
    emitted row's (fluc, zc), const, next step's basis, the interval centre ctr_out and the CB offset raw - ctr_out;
    joined before lower / upper (the first readers of ctr_out).
  in-backbone forks (a SECOND side stream, so they never queue behind the side stream's work): in every backbone
    block the gate branch (proj GEMM + sigmoid, reads only u) runs || the two depthwise convs on main, joined before
    a * gate (_packed_forward / _block / _block_near below == kd3_step's, plus the fork / join lines).
Every statement below is the parent's (cuda_graph_wholestep._forward_body, kd3_step.ctx_body / mega_body / cw_apply /
features_kd3 / _kd5_features) verbatim; each stream issues its statements in the parent's relative order; the only
change is which stream issues them -- except in _kd5_features, whose stacked [x_phys, pred_phys] T-statistics and
[last, meanpred] grad-magnitude are computed per row (same values: experiments/kd6_concurrency/scripts/
kd6_split_check.py) so that the x row can run on the side stream. Correctness therefore reduces to: nothing on
one stream between fork and join touches (RAW / WAR / WAW) what the other stream touches there, except through the
recorded edges (fork, detect -> side, blend_y -> mf, join, fork 2 / join 2, fork 3 / join 3, the per-block
fork_b / join_b). Checked mechanically
(experiments/kd6_concurrency/scripts/kd6_race.py: every op of one eager concurrent body logged with its byte extents
and stream, happens-before from the recorded edges, 0 unordered conflicts) and by the build-time self-test below
(KD6 graph vs the parent graph: every reachable CUDA storage and the outputs byte-identical over 3 replays from the
same state).
Allocations: while capturing, the side stream's allocations are routed to the capture's private pool (KD4's
mechanism, torch._C._cuda_beginAllocateCurrentStreamToPool; each side stream registered at its first fork, released
after the body);
blocks are stream-tagged (FR26), so main never receives a side block; a side block whose last reader is on main is
freed only after that read was issued, and the next side work (fork 2) waits for everything main issued before it, so
a recycled side block is never written while a main kernel may still read it (same argument for main blocks read by
the side: they stay referenced until the body returns, after the final join). Warm-ups run the same body with the
same fork / join on the real streams (eager).
TKA LOWKE ("tka_lowke", when the whole-step wrapper carries it): the head's innovation reads the un-damped previous pred
(_pred_raw) and _lowke_apply runs last on main, exactly as in the parent body.
KD7 (kd7_fuse.py, CONFIG "kd7"): when the wrapper carries compiled + self-tested fused kernels (w._kd7j), the main
stream's post-join elementwise chains (e_new .. fluc .. outK; pred_phys, the grad-magnitude, dp / f5 / CT
normalisation, the width multiplier, LOWKE's gain) run as bit-exact jiterator kernels instead; the native lines are
kept verbatim in the else-branches (w._kd7j None = the KD6 body above, op for op).
Scope: KD2 (AXE exact mode) + KD3 + REC on, noscan graph only (the scan graph is never replayed with the shipped
uc_warm / uc_every and keeps the parent body). Anything else -> not covered -> the parent graph."""
from __future__ import annotations

import contextlib

import torch

import lsk_head as _LSK  # LSK: head-input hook + s buffer (inert when off)


def _lsk_issue_pending(S):
    """LSK: issue the LS moment update on side_c (once per body). Called from the backbone right after stem / down, so
    its ~40 launches enter the graph's dispatch while the backbone runs its dependent block chain, not ahead of it."""
    fn = getattr(S, "lsk_pending", None) if S is not None else None
    if fn is not None:
        S.lsk_pending = None
        fn()


def covered(w):
    """None if the whole-step wrapper `w` is in the configuration this body transcribes; else the reason."""
    m = w.model
    if not (w._kd2 and w._kd3 and w._t2):
        return "kd6: KD2 + KD3 + T2 path required"
    if w._rec is None:
        return "kd6: REC path required"
    if w._kd2p is None or not w._kd2p.br or w._br_ad is None or w._eb_ad is None:
        return "kd6: AXE (eb + br) required"
    if not getattr(m._mega, "_mu_vp", False):
        return "kd6: mu_vp required"
    if getattr(m, "cb_w", None) is None:
        return "kd6: cb_w required"
    import kd4_w01 as _KD4
    if not _KD4.pool_routing_available():
        return "kd6: pool routing API absent"
    return None


def _event(stream):
    ev = torch.cuda.Event()
    ev.record(stream)
    return ev


def _wait(stream, ev):
    stream.wait_event(ev)


class Streams:
    """Fork / join bookkeeping of ONE body (KD4's pattern). concurrent=False: every side block runs on main."""

    def __init__(self, side, pool=None, concurrent=True, side_b=None, side_c=None):
        self.main = torch.cuda.current_stream()
        self.side = side if concurrent else self.main
        # side_b: a SECOND side stream for the short forks inside the backbone (gate branch of every block); None ->
        # those run on main (no in-backbone concurrency)
        self.side_b = side_b if (concurrent and side_b is not None) else None
        # LSK side_c: a THIRD side stream for the LS moment update (forked at the top, waited for by main right
        # after the backbone); None -> the update runs on main
        self.side_c = side_c if (concurrent and side_c is not None) else None
        self.registered_c = False
        self.forked_c = False
        self.lsk_pending = None   # LSK: the update is issued inside the backbone (after stem / down), see below
        self.pool = pool
        self.concurrent = bool(concurrent)
        self.registered = False
        self.registered_b = False
        self.forked = False
        self.forked_b = False
        if self.concurrent and self.side.cuda_stream == self.main.cuda_stream:
            raise RuntimeError("kd6: side stream aliases the main stream")
        if self.side_b is not None and self.side_b.cuda_stream in (self.main.cuda_stream, self.side.cuda_stream):
            raise RuntimeError("kd6: second side stream aliases main / side")

    def fork(self):
        if not self.concurrent:
            return
        if self.pool is not None and not self.registered:
            with torch.cuda.stream(self.side):
                torch._C._cuda_beginAllocateCurrentStreamToPool(self.side.device.index, self.pool)
            self.registered = True
        _wait(self.side, _event(self.main))
        self.forked = True

    def on_side(self):
        return torch.cuda.stream(self.side) if self.concurrent else contextlib.nullcontext()

    def fork_b(self):
        """in-backbone fork: the second side stream waits for everything issued on main so far."""
        if self.side_b is None:
            return
        if self.pool is not None and not self.registered_b:
            with torch.cuda.stream(self.side_b):
                torch._C._cuda_beginAllocateCurrentStreamToPool(self.side_b.device.index, self.pool)
            self.registered_b = True
        _wait(self.side_b, _event(self.main))
        self.forked_b = True

    def fork_c(self):
        if self.side_c is None:
            return
        if self.pool is not None and not self.registered_c:
            with torch.cuda.stream(self.side_c):
                torch._C._cuda_beginAllocateCurrentStreamToPool(self.side_c.device.index, self.pool)
            self.registered_c = True
        _wait(self.side_c, _event(self.main))
        self.forked_c = True

    def on_side_c(self):
        return torch.cuda.stream(self.side_c) if self.side_c is not None else contextlib.nullcontext()

    def join_c(self):
        if self.forked_c:
            _wait(self.main, _event(self.side_c))
            self.forked_c = False

    def on_side_b(self):
        return torch.cuda.stream(self.side_b) if self.side_b is not None else contextlib.nullcontext()

    def join_b(self):
        if self.forked_b:
            _wait(self.main, _event(self.side_b))
            self.forked_b = False

    def record(self, stream):
        return _event(stream) if self.concurrent else None

    def wait(self, stream, ev):
        if ev is not None:
            _wait(stream, ev)

    def join(self):
        if self.forked:
            _wait(self.main, _event(self.side))
            self.forked = False

    def _unregister(self):
        if self.registered:
            self.registered = False
            with torch.cuda.stream(self.side):
                torch._C._cuda_endAllocateCurrentStreamToPool(self.side.device.index)
            torch._C._cuda_releasePool(self.side.device.index, self.pool)
        if self.registered_c:
            self.registered_c = False
            with torch.cuda.stream(self.side_c):
                torch._C._cuda_endAllocateCurrentStreamToPool(self.side_c.device.index)
            torch._C._cuda_releasePool(self.side_c.device.index, self.pool)
        if self.registered_b:
            self.registered_b = False
            with torch.cuda.stream(self.side_b):
                torch._C._cuda_endAllocateCurrentStreamToPool(self.side_b.device.index)
            torch._C._cuda_releasePool(self.side_b.device.index, self.pool)

    def close(self):
        torch.cuda.set_stream(self.main)
        try:
            self.join_c()
            self.join_b()
            self.join()
        finally:
            self._unregister()


def _block(pk, st, i, x, S):
    """== kd3_step._block with the gate branch (proj GEMM + sigmoid, reads only u) on the second side stream,
    concurrently with the two depthwise convs on main; joined before a * gate."""
    import kd3_step as _KD3
    import torch.nn.functional as F
    K, hid = pk.K, pk.hid
    g = lambda n: pk._hp(f"b{i}_{n}")  # noqa: E731
    u = F.gelu(_KD3._pw(pk, st, F.group_norm(x, 8 * K, g("norm1_w"), g("norm1_b"), 1e-5), f"b{i}_proj_in"))
    S.fork_b()
    with S.on_side_b():
        gt = torch.sigmoid(_KD3._pw(pk, st, u, f"b{i}_gate"))
    a = F.conv2d(u, g("dw1_w"), g("dw1_b"), padding=pk.pad1, groups=K * hid)
    a = F.conv2d(a, g("dw2_w"), g("dw2_b"), padding=pk.pad2, dilation=pk.dil2, groups=K * hid)
    S.join_b()
    x = x + _KD3._pw(pk, st, a * gt, f"b{i}_proj_out")
    h = F.gelu(_KD3._pw(pk, st, F.group_norm(x, 8 * K, g("norm2_w"), g("norm2_b"), 1e-5), f"b{i}_mlp1"))
    x = x + _KD3._pw(pk, st, h, f"b{i}_mlp2")
    return x


def _block_near(pk, st, i, x, S):
    """== kd3_step._block_near (KD5) with the gate branch on the second side stream (see _block)."""
    import kd3_step as _KD3
    import torch.nn.functional as F
    K, hid = pk.K, pk.hid
    C = K * hid
    d, hd, wd = st.d2, st.hd, st.wd
    g = lambda n: pk._hp(f"b{i}_{n}")  # noqa: E731
    u = F.gelu(_KD3._pw(pk, st, F.group_norm(x, 8 * K, g("norm1_w"), g("norm1_b"), 1e-5), f"b{i}_proj_in"))
    S.fork_b()
    with S.on_side_b():
        gt = torch.sigmoid(_KD3._pw(pk, st, u, f"b{i}_gate"))
    y1 = F.conv2d(u, g("dw1_w"), None, padding=pk.pad1, groups=C)
    torch.add(y1.view(C, hd, d, wd, d), g("dw1_b").view(C, 1, 1, 1, 1), out=st.SBv)
    a = F.conv2d(st.SB.view(d * d, C, hd, wd), g("dw2_w"), g("dw2_b"), padding=pk.pad2 // d, groups=C)
    S.join_b()
    torch.mul(a.view(d, d, C, hd, wd).permute(2, 3, 0, 4, 1), gt.view(C, hd, d, wd, d), out=st.AG.view(C, hd, d, wd, d))
    x = x + _KD3._pw(pk, st, st.AG, f"b{i}_proj_out")
    h = F.gelu(_KD3._pw(pk, st, F.group_norm(x, 8 * K, g("norm2_w"), g("norm2_b"), 1e-5), f"b{i}_mlp1"))
    x = x + _KD3._pw(pk, st, h, f"b{i}_mlp2")
    return x


def _packed_forward(pk, st, zh, S):
    """== kd3_step.packed_forward (same ops, same TF32 / fp16-reduction flags) with the blocks' gate branches
    forked onto the second side stream."""
    import kd3_step as _KD3
    import torch.nn.functional as F
    mm_ = torch.backends.cuda.matmul
    old = (mm_.allow_tf32, mm_.allow_fp16_reduced_precision_reduction)
    mm_.allow_tf32 = True
    mm_.allow_fp16_reduced_precision_reduction = False
    try:
        K, enc = pk.K, pk.enc
        B, _, H, W = zh.shape
        hp = pk._hp
        st.bufA.copy_(st.biasA.expand(-1, st.hwA))
        st.bufB.copy_(st.biasB.expand(-1, st.hwB))
        if st.near:
            Cs = st.out["stem"].view(1, -1, st.hwB)
            ys = torch.baddbmm(Cs, st.Wstem, F.unfold(zh, 3, padding=1), out=Cs).view(B, -1, H, W)
            skip = F.gelu(F.group_norm(ys, 8 * K, hp("stem_gw"), hp("stem_gb"), 1e-5))
            Cd = st.out["down"]
            yd = torch.baddbmm(Cd, st.Wdown, F.unfold(skip.view(K, enc, H, W), 4, padding=1, stride=2), out=Cd)
            h = F.gelu(F.group_norm(yd.view(B, -1, st.Hb, st.Wb), 8 * K, hp("down_gw"), hp("down_gb"), 1e-5))
            _lsk_issue_pending(S)   # LSK
            for i in range(pk.n_blocks):
                h = _block_near(pk, st, i, h, S)
        else:
            skip = F.gelu(F.group_norm(F.conv2d(zh, hp("stem_w"), hp("stem_b"), padding=1),
                                       8 * K, hp("stem_gw"), hp("stem_gb"), 1e-5))
            h = F.gelu(F.group_norm(F.conv2d(skip, hp("down_w"), hp("down_b"), stride=2, padding=1, groups=K),
                                    8 * K, hp("down_gw"), hp("down_gb"), 1e-5))
            _lsk_issue_pending(S)   # LSK
            for i in range(pk.n_blocks):
                h = _block(pk, st, i, h, S)
        h = F.conv_transpose2d(h, hp("up_w"), hp("up_b"), stride=2, padding=1, groups=K)
        if st.near:
            Cf = st.out["fuse"]
            Wf = hp("fuse_W3")
            torch.baddbmm(Cf, Wf[:, :, :enc], h.view(K, enc, H * W), out=Cf)
            yf = torch.baddbmm(Cf, Wf[:, :, enc:], skip.view(K, enc, H * W), out=Cf).view(B, -1, H, W)
        else:
            hc = torch.cat([h.view(B, K, enc, H, W), skip.view(B, K, enc, H, W)], dim=2)
            hc = hc.view(B, K * 2 * enc, H, W)
            yf = _KD3._pw(pk, st, hc, "fuse")
        h = F.gelu(F.group_norm(yf, 8 * K, hp("fuse_gw"), hp("fuse_gb"), 1e-5))
        if _LSK.HOOK is not None:
            _LSK.HOOK(h)   # LSK: head input -> phi
        out = torch.mean(_KD3._pw(pk, st, h, "head").view(B, K, pk.out_ch, H, W), dim=1, dtype=torch.float32)
    finally:
        mm_.allow_tf32, mm_.allow_fp16_reduced_precision_reduction = old
    b, tc, h_, w_ = out.shape
    return out.reshape(b, tc // 3, 3, h_, w_).permute(0, 1, 3, 4, 2)


def _ctx_body(base, x, st, rec, after_detect, S=None):
    """== kd3_step.ctx_body(base, x, st, rec) (REC branch), with `after_detect()` called right after the detector."""
    from cuda_graph_ctxnorm import _causal_ms_graphed  # noqa: F401  (same import as the parent)
    import kd3_step as _KD3
    w = base.wrapped
    T = x.shape[1]
    uv = x[..., :2]
    window_sum1 = torch.sum(uv, dim=1, out=rec.MW[1])    # same reduction, into the REC buffer
    rec.detect()                                          # reads MW = [previous m, this window's sum]
    after_detect()
    window_sum2 = (uv * uv).sum(dim=1)
    torch.addcmul(window_sum1, base._static_S1, rec.keep, out=base._static_S1)
    torch.addcmul(window_sum2, base._static_S2, rec.keep, out=base._static_S2)
    torch.addcmul(rec.T_t, base._static_n, rec.keep, out=base._static_n)
    recip = base._static_n.reciprocal()                   # == _causal_ms_graphed, m written into rec.MW[0]
    m = torch.mul(base._static_S1, recip, out=rec.MW[0])
    var = base._static_S2 * recip - m * m
    var = torch.clamp_min(var, w._ms_floor)
    s = torch.sqrt(var) if _LSK.S_BUF is None else torch.sqrt(var, out=_LSK.S_BUF)   # LSK: s -> S_BUF (same kernel)
    ms, ss = m.unsqueeze(1), s.unsqueeze(1)
    torch.div(x[..., :2] - ms, ss, out=st.zV[..., :2])
    st.zV[..., 2:3].copy_(x[..., 2:3])
    logs = torch.log(s)
    torch.div(m - w._m_mean, w._m_std, out=st.zE[..., 0:2])
    torch.div(logs - w._logs_mean, w._logs_std, out=st.zE[..., 2:4])
    y_s = _KD3.packed_forward(st.pk, st, st.zh) if S is None else _packed_forward(st.pk, st, st.zh, S)
    raw = st.raw
    torch.add(y_s[..., :2] * ss, ms, out=raw[..., :2])
    raw[..., 2:3].copy_(y_s[..., 2:3])
    return raw


def forward_body(w, S: Streams):
    """== GraphedWholeStepWrapper._forward_body(scan=False) on the KD2 + KD3 + REC (+ KD5) path, two streams."""
    import kd2_adapt as _KD2
    import kd3_step as _KD3
    import ke_net_flow
    import cond_width_net as _CW
    from cuda_graph_megatail import _ke
    from submission import _SIGMA_GLOBAL
    m = w.model
    base, mega, maskg = m.base, m._mega, m._maskg
    io = w._io
    b = io
    st = w._kd3s
    kp, ks = w._kd2p, w._kd2s
    rec = w._rec
    x = w._dev_in[w._ix:w._ix + 1]
    y = w._dev_in[w._iy:w._iy + 1]
    prev_out = w._dev_out_views[0]          # still holds the previous step's pred
    GainMapImprovedModel = mega._GainMapImprovedModel
    s = b.shadow
    eb, br = b._eb_ad, b._br_ad
    ABC, M3, S6 = st.ABC, st.M3, st.S6
    T = x.shape[1]
    Hh, Wh, Ch = x.shape[2], x.shape[3], x.shape[4]

    _lsk = getattr(m, "_lsk", None)
    if _lsk is not None:
        # LSK: the LS moment update reads only y and the previous window's snapshots -> third side stream, issued
        # from inside the backbone (_lsk_issue_pending after stem / down)
        def _lsk_issue():
            S.fork_c()
            with S.on_side_c():
                _lsk.update(y)
        S.lsk_pending = _lsk_issue
    S.fork()
    # =========================== SIDE (1): step() head + mask ===========================
    with S.on_side():
        # TKA LOWKE (cuda_graph_wholestep "tka_lowke"): the innovation reads the un-damped previous pred
        err = torch.sub(y, w._pred_raw if getattr(w, "_lowke", None) else prev_out, out=w._err)
        torch.mean(err ** 2, dim=(0, 1, 2, 3, 4), out=w._dev_loss)
        torch.sub(y, w._mf, out=w._blend_y)   # w._mf still holds prev_mf here
        ev_blend = S.record(S.side)           # main's mf overwrites w._mf only after this
        cnt = w._cnt
        cnt.add_(1.0)
        kxy = w._dev_in.var(dim=1, keepdim=True, unbiased=False)
        ky = kxy[w._iy:w._iy + 1]
        io._kx = kxy[w._ix:w._ix + 1]
        _KD2.cw_update(m, m.cw, w._pc, w._prev2, w._ema2, ks.Q0cw, st=ks)
        io._eb_k = _KD2.select(kp, ks, y[0, ..., :2], w._ctr_out[0, ..., :2], w._cb_d[0, ..., :2],
                               w._cb_wd[0], w._zt, y=y, outK=w._eb_outK)
        io._br_zidx, io._br_eidx = ks.zidx, ks.eidx
        mask = maskg._body(x)

    # =========================== MAIN: ctx-norm + backbone ===========================
    det = {}
    raw = _ctx_body(base, x, st, rec, lambda: det.__setitem__("ev", S.record(S.main)), S)
    if _lsk is not None:
        _lsk_issue_pending(S)      # LSK: no-op (already issued inside the backbone) unless that path was skipped
        S.join_c()                 # LSK: M ready
        if _lsk.rec_reset:
            _lsk.rec_reset_(rec.keep)   # LSX: REC reset clears the LS state (x 1.0 when REC does not fire)
        _lsk.apply_(raw)           # LSK: 2 kernels on the critical path, before any reader of raw

    # =========================== SIDE (2): everything else that does not read the backbone ===========================
    with S.on_side():
        S.wait(S.side, det["ev"])
        rec.rescale_mean([mega._mu, st.M3])   # REC: mean-type state x r (x 1.0 when no reset)
        a4, _ = rec.counters_a4(w._prior_n_t, w._alpha_min_t)
        torch.add(w._R_run, (ky - w._R_run) * torch.reciprocal(cnt[5].to(ky.dtype)), out=w._R_run)
        a4f = a4.to(torch.float32)
        io._a_err_f32, io._a_err_f64 = a4f[0], a4[0]
        io._om4 = rec.om4
        io._a_ctr_f32, io._a_ctr_f64 = a4f[1], a4[1]
        io._a_bld_f32 = a4f[2]
        io._a_e_f32 = a4f[3]
        io._wfrac = (torch.clamp_max(cnt[4], 20.0) / 20.0).to(torch.float32).view(1)
        io._mask_in = mask
        prev_const = b._prev_const
        prev_fl, prev_adv = b._prev_fl, b._prev_adv
        blend_y = b._blend_y
        # ---- mega PRE half (kd3_step.mega_body, verbatim)
        P = mega._P7
        d = err - b._mu
        new_mu = torch.add(b._mu, b._a_err_f32 * d, out=b._mu)
        torch.mul(d, d, out=P[0])
        torch.sub(y, prev_const, out=ABC[0])
        dABC = ABC - M3
        torch.add(M3, b._a_ctr_f32 * dABC, out=M3)
        da, db, dc = dABC[0], dABC[1], dABC[2]
        new_M0 = M3[0]
        torch.mul(da, da, out=P[1])
        torch.mul(da, db, out=P[2])
        torch.mul(da, dc, out=P[3])
        torch.mul(db, db, out=P[4])
        torch.mul(dc, dc, out=P[5])
        torch.mul(db, dc, out=P[6])
        Pm = P.mean(dim=2, keepdim=True)
        om4 = b._om4
        torch.mul(om4[0], b._var + b._a_err_f64 * Pm[0], out=b._var)
        a64c = b._a_ctr_f64
        om = om4[1]
        torch.mul(om, S6 + a64c * Pm[1:7], out=S6)
        new_Saa, new_Sab, new_S02, new_Sbb, new_S22, new_S12 = S6.unbind(0)
        keep_prev = st.KP
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
        # ---- energy half: the parts that read only x / kx / state
        kx = b._kx
        d_et = kx - b._e_true
        new_e_true = torch.add(b._e_true, b._a_e_f32 * d_et, out=b._e_true)
        xh = x.view(1, 2, 10, Hh, Wh, Ch)[..., :2].var(dim=2, unbiased=False)
        mx_ = x.mean(dim=1, keepdim=True)
        # features_kd3: x / mask / wfrac channels (FEAT channels [0, 2k) and c0+8 .. c0+11)
        import ke_net_flow as KF
        EPS_F = KF.EPS
        FT = st.FEAT
        k_, L_ = st.k_in, st.n_leads
        runx = new_e_true[:, 0, ..., :2]
        mxk = mx_[..., :2]
        mrx = runx.mean(dim=(1, 2), keepdim=True)
        sx = mrx.clamp_min(EPS_F).sqrt()
        nk, nl = 2 * k_, 2 * L_
        slot = lambda c0_, n: FT[:, c0_:c0_ + 2 * n].view(1, n, 2, Hh, Wh).permute(0, 1, 3, 4, 2)  # noqa: E731
        xw = x[..., :2]
        torch.div(xw[:, -k_:] - mxk, sx.unsqueeze(1), out=slot(0, k_))
        c0 = nk + nl
        maskf = mask[None]
        mxs = mxk.squeeze(1)
        FT[:, c0 + 8].copy_(maskf)
        FT[:, c0 + 9].copy_(mxs[..., 0])
        torch.mul(mxs[..., 1], 6.0, out=FT[:, c0 + 10])
        FT[:, c0 + 11].copy_(b._wfrac.view(-1, 1, 1).expand_as(maskf))
        nm = ~mask
        st.KP.copy_(nm.view(st.KP.shape))     # float(~mask): here and next step's keep_prev (PRE read above)
        # ---- TAIL: the parts that read only x / mask / state
        obstacle = s.shift.last_obstacle(mask)
        fl_x = x - mx_
        adv, valid = _KD3._advect(s, fl_x[:, -1], obstacle, nm, T, b._out_adv)
        a_, b_ = _KD3._coefficients(st, valid, nm, b._S5, lam=br.lamv)
        torch.sub(adv, adv.mean(dim=1, keepdim=True), out=b._out_b2)   # == ABC[2]
        G = st.G
        g1, g2c = _KD3._ctr_gain2(st)                                    # noqa: F841
        GM = G * M3[1:3]
        G2 = 2.0 * G
        AB = G2 * S6[1:3]
        GG = G * G
        CE = GG * S6[3:5]
        D = G2[0] * g2c * new_S12
        torch.clamp_min(new_Saa - AB[0] - AB[1] + CE[0] + D + CE[1], 1e-12, out=b._out_ctr_var)
        # ---- cw_apply: k-table / half1 / x_phys (read only ctr_var and x)
        var = w._ctr_var
        std1 = var.sqrt()
        a = 2.0 * std1 * m.std_tgt / _SIGMA_GLOBAL
        pos = (a.clamp_min(1e-12).log() - m.log_a[0]) / st.dla
        kk = m.k_tab[pos.round().long().clamp(0, len(m.k_tab) - 1)]
        half1 = m._cwg._k_scale * kk * std1
        sig_pipe_phys = std1[..., :2] * m.std_tgt[:2]
        x_phys = (x[..., :2] * m.std_in[:2] + m.mean_in[:2]).squeeze(0).permute(3, 0, 1, 2)
        if st.near:
            # ---- kd3_step._kd5_features: the x / sig_pipe / prev2-only part (verbatim ops; the parent's stacked
            # [x_phys, pred_phys] tensor is built row by row -- the x row here, the pred row on main -- and its T-mean /
            # T-std are taken per row: same values (bit-exact: experiments/kd6_concurrency/scripts/kd6_split_check.py);
            # the grad-magnitude of `last` alone == row 0 of the parent's stacked [last, meanpred] call (elementwise))
            EPS_W = _CW.EPS
            sp = sig_pipe_phys.squeeze(0).permute(3, 0, 1, 2)
            bW = x_phys.shape[0]
            XP = torch.empty((2,) + tuple(x_phys.shape), device=x_phys.device, dtype=x_phys.dtype)
            XP[0].copy_(x_phys)
            xm = XP[0].mean(dim=1)
            sxw = XP[0].std(dim=1, unbiased=True)
            last = x_phys[:, -1]
            gm0 = _CW._t2_grad_mag(torch.stack([last], 0))
            bx = _CW._box(torch.stack([sxw, sxw * sxw], 0).reshape(2 * bW, Hh, Wh), 3).view(2, bW, Hh, Wh)
            var_local = (bx[1] - bx[0] * bx[0]).clamp_min(0.0)
            L6 = torch.empty(6, bW, Hh, Wh, device=x_phys.device, dtype=x_phys.dtype)   # == the parent's 6-row stack
            L6[0].copy_(sxw)
            torch.abs(last - xm, out=L6[1])
            L6[2].copy_(gm0[0])
            torch.sqrt(var_local, out=L6[4])
            CT = st.CT
            torch.log(sp[:, 0] + EPS_W, out=CT[0])
            PE3 = st.PE3
            torch.clamp_min(w._prev2 / sp[:, 0].clamp_min(EPS_W), 0.0, out=PE3[2])
            torch.log(PE3 + EPS_W, out=CT[7:10])

    # =========================== MAIN: mf, fl_f, join, backbone-dependent rest ===========================
    S.wait(S.main, ev_blend)
    mf = torch.mean(raw, dim=1, keepdim=True, out=w._mf)
    fl_f = torch.sub(raw, mf, out=mega._fm[0:1])   # slot 0 of mega's (fl_f, mu_p) pair
    S.join()
    io._fl_f = fl_f
    # ---- mega energy half (rest, verbatim order)
    mu_p = torch.sub(new_mu, new_mu.mean(dim=1, keepdim=True), out=b._fm[1:2])
    v2 = b._fm.var(dim=1, keepdim=True, unbiased=False)
    e_f, vM = v2[0:1], v2[1:2]
    d_ep = e_f - b._e_pred
    new_e_pred = torch.add(b._e_pred, b._a_e_f32 * d_ep, out=b._e_pred)
    ex, ef = kx[:, 0, ..., :2], e_f[:, 0, ..., :2]
    ffl = fl_f[..., :2]
    rf = new_e_pred[:, 0, ..., :2]
    fh = fl_f.view(1, 2, 10, Hh, Wh, Ch)[..., :2].var(dim=2, unbiased=False)
    ex1, ex2, ef1, ef2 = xh[:, 0], xh[:, 1], fh[:, 0], fh[:, 1]
    # features_kd3: fl_f / energy channels
    FC, LC = KF.FIELD_CLAMP, KF.LOG_CLAMP
    runf = rf
    sf = runf.mean(dim=(1, 2), keepdim=True).clamp_min(EPS_F).sqrt()
    torch.div(ffl.index_select(1, mega._leads_idx), sf.unsqueeze(1), out=slot(nk, L_))
    FT[:, 0:nk + nl].clamp_(-FC, FC)
    nums = [ef, ex, runx, runx, ex2, ef2, ex, ex]
    dens = [runf, runx, runf, mrx.expand_as(runx), ex1, ef1, b._ef_prev, b._ex_prev]
    lg = torch.log((torch.stack(nums, 0) + EPS_F) / (torch.stack(dens, 0) + EPS_F))     # (8, 1, H, W, 2)
    en = lambda c: FT[:, c:c + 8].view(1, 4, 2, Hh, Wh).permute(1, 0, 3, 4, 2)          # noqa: E731
    torch.clamp(lg[0:4], -LC, LC, out=en(c0))
    torch.clamp(lg[4:8], -LC, LC, out=en(c0 + 12))
    feat = FT
    EPS = ke_net_flow.EPS
    # =========================== SECOND FORK: side || ke_net_flow net ===========================
    # side: the gain-2 ratio, the rule log-energy, the (ex_prev, ef_prev) carry (their only reader, lg, is above),
    # the blend candidates z / zc / constK / e_z / c_zf and the prev_fl carry -- none reads the ke net's output.
    S.fork()
    with S.on_side():
        E8 = st.E2 + 1e-8
        g2 = (E8[0] / E8[1]).pow(mega._gain_power)
        LR = torch.log(st.E2[:, :, 0, ..., :2] + EPS)
        log_rule = torch.log(ef + EPS) + 0.75 * (LR[0] - LR[1])
        b._ex_prev.copy_(ex)
        b._ef_prev.copy_(ef)
        z = (a_ * fl_f + b_ * adv).index_select(0, b._br_zidx)
        mz = z.mean(dim=1, keepdim=True)
        zc = zc_ls = z - mz
        constK = mf + mz
        e_z = _ke(zc)
        c_zf = (zc * fl_f).mean(dim=1, keepdim=True)
        w._prev_fl.copy_(fl_f)
    J = getattr(w, "_kd7j", None)     # KD7 (kd7_fuse.py): fused bit-exact jiterator kernels, None = the KD6 ops
    if J is not None:
        net_o = mega._kf_net(feat)
        S.join()
        # A: clamp / kf_scale / keep-mask / exp / blend with R_run in one kernel; the eidx gather lands in E3[..., :2]
        e_t5 = J.A(net_o.unsqueeze(1), st.KP.view(1, 1, *mask.shape, 1), log_rule.unsqueeze(1), eb.wv, eb.omv,
                   b._R_run[..., :2])
        E3 = J.E3()
        torch.index_select(e_t5, 0, b._br_eidx, out=E3[..., :2])
        # ---- TAIL (rest)
        s.S = new_S
        s.M0, s.M1, s.M2 = M3[0], M3[1], M3[2]
        s.Saa, s.Sab, s.Sbb = new_Saa, new_Sab, new_Sbb
        s.S02, s.S12, s.S22 = new_S02, new_S12, new_S22
        # B: g_flow / cat / clamp / sqrt / e_tgt2 + _shape's per-pixel coefficients; C: _shape's fluc
        gam_s, s_cl = J.B(E3, e_f, g2, e_z, c_zf)
        fluc0 = J.C(gam_s, zc, s_cl, fl_f)
        e_tgt_ach = _ke(fluc0)
        cFM = (fluc0 * mu_p).mean(dim=1, keepdim=True)
        c = J.D(cFM, e_tgt_ach, vM)
        outK = torch.add(J.E(constK, c, fluc0), new_mu, out=b._eb_outK)
        torch.index_select(outK, 0, b._eb_k, out=b._out_out)
    else:
        corr = mega._kf_net(feat).clamp(-1.5, 1.5)
        if st.kf_scale != 1.0:
            corr = corr * st.kf_scale
        corr = corr * st.KP.view(1, *mask.shape, 1)
        S.join()
        e_new = (log_rule + corr).exp()
        e_t2 = e_new.unsqueeze(1)
        e_t2 = eb.wv * e_t2 + eb.omv * b._R_run[..., :2]
        e_t2 = e_t2.index_select(0, b._br_eidx)
        g_flow = (e_t2 + 1e-10) / (e_f[..., :2] + 1e-10)
        g2 = torch.cat([g_flow, g2[..., 2:].expand(br.Kc, -1, -1, -1, -1)], dim=-1)
        g_max = mega._gain_max
        gamma = g2.clamp(1.0 / g_max ** 2, g_max ** 2).sqrt()
        # ---- TAIL (rest)
        s.S = new_S
        s.M0, s.M1, s.M2 = M3[0], M3[1], M3[2]
        s.Saa, s.Sab, s.Sbb = new_Saa, new_Sab, new_Sbb
        s.S02, s.S12, s.S22 = new_S02, new_S12, new_S22
        e_tgt2 = gamma * gamma * e_f
        fluc = GainMapImprovedModel._shape(s, zc, fl_f, e_z, e_f, c_zf, e_tgt2)
        e_tgt_ach = _ke(fluc)
        cFM = (fluc * mu_p).mean(dim=1, keepdim=True)
        disc = (cFM * cFM - e_tgt_ach * vM + e_tgt_ach * e_tgt_ach).clamp_min(0.0)
        c = ((-cFM + disc.sqrt()) / e_tgt_ach.clamp_min(1e-12)).clamp_min(0.0)
        fluc = c * fluc
        outK = torch.add(constK + fluc, new_mu, out=b._eb_outK)
        torch.index_select(outK, 0, b._eb_k, out=b._out_out)
    # =========================== THIRD FORK: side || the cond-width features + W2 net ===========================
    # side: the emitted row's (fluc, zc) selection, const, next step's basis, the interval centre ctr_out and the CB
    # offset d = raw - ctr_out; nothing in the width net reads them (it reads x, this step's pred, sig_pipe, prev2).
    # (new names: the parent rebinds `fluc` / `zc_ls` -- here the main-stream tensors stay referenced until the end.)
    S.fork()
    with S.on_side():
        if J is not None:     # KD7: (c * fluc0)[k] == c[k] * fluc0[k] (elementwise)
            fluc_k = torch.mul(c.index_select(0, b._eb_k), fluc0.index_select(0, b._eb_k))
        else:
            fluc_k = fluc.index_select(0, b._eb_k)
        zc_k = zc.index_select(0, b._eb_k)
        const = torch.index_select(constK, 0, b._eb_k, out=b._out_const)
        wc = mega._ctr_w
        torch.add((1.0 - wc) * fluc_k, wc * zc_k, out=b._out_basis)       # == ABC[1]
        GB = G * ABC[1:3]
        ctr = const + GB[0] + GB[1]
        torch.sub(ctr + new_M0 - GM[0], GM[1], out=b._out_ctr_out)
        d_cb = torch.sub(raw, w._ctr_out, out=w._cb_d)
        if _lsk is not None:
            _lsk.snapshot(raw)     # LSK: phi / s / base raw of this window for the next update (third fork, side)
    # ---- whole-step: bounds (kd3_step.cw_apply rest), CB shifts  (prev_fl carry: second fork)
    cw = m.cw
    mean = w._ctr_out
    T2 = J is not None and J.t2          # KD7 tier 2: W2 features / width multiplier / LOWKE kernels
    if T2:
        pred_phys = J.P(prev_out[..., :2], m.std_tgt[:2], m.mean_tgt[:2]).squeeze(0).permute(3, 0, 1, 2)
    else:
        pred_phys = (prev_out[..., :2] * m.std_tgt[:2] + m.mean_tgt[:2]).squeeze(0).permute(3, 0, 1, 2)
    sp = sig_pipe_phys.squeeze(0).permute(3, 0, 1, 2)
    if st.near:
        # ---- kd3_step._kd5_features: the pred part (x part on the side stream above)
        XP[1].copy_(pred_phys)
        meanpred = XP[1].mean(dim=1)
        pred_std = XP[1].std(dim=1, unbiased=True)
        if T2:     # KD7: replicate pad + 1 kernel (the stack of one row is a copy: dropped, same values)
            gm1_0 = J.grad_mag(meanpred)
        else:
            gm1_0 = _CW._t2_grad_mag(torch.stack([meanpred], 0))[0]
        L6[3].copy_(pred_std)
        L6[5].copy_(gm1_0)
        torch.log(L6 + EPS_W, out=CT[1:7])
        if T2:     # KD7: dp = pred_phys - [last, pred_phys[:, :-1]]; log |dp| + eps, (- mu5) / sd5 in one kernel
            prev_t = torch.cat([last.unsqueeze(1), pred_phys[:, :-1]], dim=1)
            f5n = J.F(pred_phys, prev_t, st.mu5, st.sd5)
            CTn = J.N(CT, st.mu10, st.sd10)
            st.INH16[:, :, 0:10].copy_(CTn.permute(1, 0, 2, 3).unsqueeze(1).expand(bW, T, 10, Hh, Wh))
            st.INH16[:, :, 10].copy_(f5n)
        else:
            dp = torch.empty_like(pred_phys)
            dp[:, 0] = pred_phys[:, 0] - last
            dp[:, 1:] = pred_phys[:, 1:] - pred_phys[:, :-1]
            f5 = torch.log(dp.abs() + EPS_W)
            CTn = torch.div(CT - st.mu10, st.sd10)
            st.INH16[:, :, 0:10].copy_(CTn.permute(1, 0, 2, 3).unsqueeze(1).expand(bW, T, 10, Hh, Wh))
            torch.div(f5 - st.mu5, st.sd5, out=st.INH16[:, :, 10])
    else:
        base_f = _CW._kd_base_feature_list(x_phys, pred_phys, sp.expand(-1, T, -1, -1))
        PE3 = st.PE3
        torch.clamp_min(w._prev2 / sp[:, 0].clamp_min(_CW.EPS), 0.0, out=PE3[2])
        lp, le, lr = torch.log(PE3 + _CW.EPS).unbind(0)
        exT = lambda t: t.unsqueeze(1).expand(-1, T, -1, -1)  # noqa: E731
        feats = torch.stack(base_f + [exT(lp), exT(le), exT(lr)], dim=2)
        torch.div(feats - cw.mu.view(1, 1, -1, 1, 1), cw.sd.view(1, 1, -1, 1, 1), out=st.INH)
    net_h = getattr(cw, "_kd_net_half", None)
    if net_h is None:
        raise RuntimeError("kd6: fp16 cond-width net not built yet")
    if st.near:
        import torch.nn.functional as F
        n_ = net_h
        hh = n_.act(n_.g1(F.conv2d(st.INH16.view(2 * T, 16, Hh, Wh), st.W2c1, None, padding=1)))
        hh = n_.act(n_.g2(n_.c2(hh)))
        hh = n_.act(n_.g3(n_.c3(hh)))
        mm = n_.out(hh).float().view(2, T, Hh, Wh)
    else:
        inp = st.INH.view(2 * T, -1, Hh, Wh)
        mm = net_h(inp).float().view(2, T, Hh, Wh)
    if T2:     # KD7: half1[..., :2] * (clamp(exp(alpha mm), lo, hi) * lam) in one kernel
        out_half = torch.cat([J.M(half1[..., :2], mm.permute(1, 2, 3, 0).unsqueeze(0)),
                              half1[..., 2:].expand(1, T, Hh, Wh, 1)], dim=-1)
    else:
        mult = torch.exp(cw.alpha * mm).clamp(cw.clip_lo, cw.clip_hi) * cw.lam
        mult2 = mult.permute(1, 2, 3, 0).unsqueeze(0)
        out_half = torch.cat([half1[..., :2] * mult2, half1[..., 2:].expand(1, T, Hh, Wh, 1)], dim=-1)
    S.join()
    torch.sub(mean, out_half, out=w._dev_out_views[1])
    torch.add(mean, out_half, out=w._dev_out_views[2])
    torch.sub(w._dev_out_views[2][..., :2], w._dev_out_views[1][..., :2], out=w._cb_wd)
    w._pack12.addcmul_(ks.cbw, d_cb)
    w._pack12[..., :2].addcmul_(ks.f2, w._cb_wd)
    if getattr(w, "_lowke", None):
        if T2 and J.lk is not None:
            J.lowke_apply(w)      # KD7: == w._lowke_apply() with its scalar / gain chain in one kernel
        else:
            w._lowke_apply()      # TKA LOWKE: output-only damping of the emitted pred (main, after every join)


# ----------------------------------------------------------------------------------------------- build / self-test
# per-step temporaries the body parks on the I/O namespace (they live in the capturing graph's private pool, so the
# parent graph and the KD6 graph write DIFFERENT blocks for them): not state, excluded from the self-test
_IO_TEMPS = ("_kx", "_eb_k", "_a_err_f32", "_a_err_f64", "_a_ctr_f32", "_a_ctr_f64", "_a_bld_f32", "_a_e_f32", "_wfrac",
             "_mask_in")


def _reachable_storages(root, max_depth=8):
    """Every CUDA storage reachable from `root` through __dict__ / nn.Module / dict / list / tuple (by data_ptr),
    except the I/O namespace's per-step temporaries."""
    import torch.nn as nn
    seen, out = set(), {}
    io = getattr(root, "_io", None)
    if io is not None:
        for k in _IO_TEMPS:
            v = io.__dict__.get(k)
            if isinstance(v, torch.Tensor):
                seen.add(id(v))

    def visit(o, depth):
        if depth > max_depth or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            if o.device.type == "cuda" and o.untyped_storage().nbytes() > 0:
                stg = o.untyped_storage()
                out.setdefault(stg.data_ptr(), o)
            return
        if isinstance(o, (list, tuple, set)):
            for v in o:
                visit(v, depth + 1)
            return
        if isinstance(o, dict):
            for v in o.values():
                visit(v, depth + 1)
            return
        if isinstance(o, (torch.cuda.Stream, torch.cuda.Event, torch.cuda.CUDAGraph, str, bytes, int, float, bool,
                          type(None))):
            return
        if isinstance(o, nn.Module):
            for v in list(o._parameters.values()) + list(o._buffers.values()) + list(o._modules.values()):
                visit(v, depth + 1)
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict):
            for v in d.values():
                visit(v, depth + 1)
    visit(root, 0)
    return list(out.values())


def _bytes(t):
    u = torch.empty(0, dtype=torch.uint8, device=t.device)
    u.set_(t.untyped_storage())
    return u


def snapshot(ts):
    return [_bytes(t).clone() for t in ts]


def restore(ts, snap):
    for t, s_ in zip(ts, snap):
        _bytes(t).copy_(s_)


def equal(ts, snap):
    bad = 0
    for t, s_ in zip(ts, snap):
        if not torch.equal(_bytes(t), s_):
            bad += 1
    return bad


def build(w, warmup_iters: int = 3, n_test: int = 3):
    """Capture the KD6 noscan graph, self-test it against the parent's `w.graph_noscan` (same buffers, same start
    state, n_test consecutive replays: every reachable CUDA storage byte-identical after each), restore the
    pre-KD6 state and return (graph, stats). Raises on any failure (caller keeps the parent graph)."""
    why = covered(w)
    if why:
        raise RuntimeError(why)
    stats = {}
    ts = _reachable_storages(w)
    stats["n_storages"] = len(ts)
    stats["mb"] = round(sum(t.untyped_storage().nbytes() for t in ts) / 2 ** 20, 1)
    torch.cuda.synchronize()
    rng0 = torch.cuda.get_rng_state()
    pre = snapshot(ts)
    side = side_b = side_c = None
    cur = torch.cuda.current_stream().cuda_stream
    for _ in range(4):
        side = torch.cuda.Stream()
        if side.cuda_stream != cur:
            break
    for _ in range(4):
        side_b = torch.cuda.Stream()
        if side_b.cuda_stream not in (cur, side.cuda_stream):
            break
    if getattr(w.model, "_lsk", None) is not None:
        for _ in range(4):
            side_c = torch.cuda.Stream()
            if side_c.cuda_stream not in (cur, side.cuda_stream, side_b.cuda_stream):
                break
    graph = None
    try:
        s_stream = torch.cuda.Stream()
        s_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_stream), torch.no_grad():
            for _ in range(warmup_iters):
                S = Streams(side, None, concurrent=True, side_b=side_b, side_c=side_c)
                try:
                    forward_body(w, S)
                finally:
                    S.close()
        torch.cuda.current_stream().wait_stream(s_stream)
        torch.cuda.synchronize()
        restore(ts, pre)
        pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, pool=pool):
            if torch.cuda.current_stream().cuda_stream in (side.cuda_stream, side_b.cuda_stream) + (
                    (side_c.cuda_stream,) if side_c is not None else ()):
                raise RuntimeError("kd6: side stream aliases the capture stream")
            S = Streams(side, pool, concurrent=True, side_b=side_b, side_c=side_c)
            try:
                forward_body(w, S)
            finally:
                S.close()
        torch.cuda.synchronize()
        restore(ts, pre)
        # ---- self-test: parent graph vs KD6 graph from the same state, n_test consecutive steps
        ref = w.graph_noscan
        bad_total = 0
        for k in range(n_test):
            torch.cuda.synchronize()
            st0 = snapshot(ts)
            ref.replay()
            torch.cuda.synchronize()
            post = snapshot(ts)
            restore(ts, st0)
            graph.replay()
            torch.cuda.synchronize()
            bad = equal(ts, post)
            bad_total += bad
            del st0, post
        stats["selftest_bad_storages"] = bad_total
        if bad_total:
            raise RuntimeError(f"kd6: self-test mismatch ({bad_total} storages)")
        stats["selftest"] = "bit_exact"
    finally:
        torch.cuda.synchronize()
        restore(ts, pre)
        torch.cuda.set_rng_state(rng0)
        torch.cuda.synchronize()
    stats["side_stream"] = int(side.cuda_stream)
    stats["side_stream_b"] = int(side_b.cuda_stream)
    if side_c is not None:
        stats["side_stream_c"] = int(side_c.cuda_stream)
    return graph, (side, side_b, side_c), stats
