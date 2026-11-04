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
KD8 (experiments/kd8, CONFIG "kd8", on top of KD6 + KD7 + LSK; w._kd8): a further critical-path diet -- the same values;
work moved off the main stream and 6 more bit-exact jiterator kernels (kd8_fuse.py, w._kd8j) on it. Parts (KD8_PARTS;
CONFIG "kd8": True = all, a list = a diagnostic subset):
  bias   the packed backbone's bias preloads (bufA / bufB copies) -> side_c, forked at the top; main waits before the stem;
  ws2    ctx_body's window sum of squares (mul + sum) and the zV[..., 2] copy -> side_b at the top;
  ze     the zE chain (log s, (m - m_mean) / m_std, (log s - logs_mean) / logs_std) [+ the S_BUF copy under "v"] ->
         side_b after main has written m / s; joined before the backbone;
  v      ctx-norm std: mul / mul / sub / clamp_min / sqrt -> kd8 V (S_BUF <- s copied on side_b);
  lskc   LSK: the head-input hook GEMM (phi) and the corr baddbmm -> side_c (same matmul flags as the serial code), main
         joins before the u, v update = kd7 P (y * ss + ms) + a native addcmul out=raw[..., :2] (== the in-place addcmul_);
  mup    mu_p (mean + sub, reads only new_mu) -> side, end of the first side section;
  pj     after the first join: fh + the leads gather -> side_b, the x-only log-energy rows (1, 3, 4, 6, 7) -> side_c;
         main: e_pred (kd8 EP; b._e_pred written back at the start of the second fork's side section, before E2's
         readers), sf (kd8 SF), the three backbone rows (kd8 LG) + 3 copies into FT;
  ke2    _ke(fluc0) (T-variance) on side_b || cFM on main;
  e2     kd7 E + the new_mu add -> kd8 E2 (b._eb_outK written on the third fork's side stream);
  lowke  TKA LOWKE on the third fork's side stream; its in-place update of the emitted pred waits for an event recorded
         on main right after main's last reader of the un-damped pred (pred_phys);
  w2f    the pred row's T-std (-> L6[3]) on side_b, the dp / f5 branch (-> INH16 channel 10) on side_c, || the T-mean /
         grad-magnitude on main; joined before the log of L6 / the INH16 channels 0:10 copy;
  mh     kd7 M reads the fp16 W2-net output directly (drops the .float() copy; the load cast is exact);
  bd     lower / upper / width / both CB shifts -> kd8 BD + one copy into the output pack (cb_wd copy on side_b).
Checked like KD6 / KD7: experiments/kd8/scripts/kd8_race.py (transitive happens-before, jiterator launches logged:
0 unordered conflicts, state == the parent serial body) and the build-time KD6 graph self-test (vs the parent graph);
kd8_fuse's unit self-test + kd8_selftest below. Any KD8 build failure -> the KD7 / KD6 body exactly as without "kd8".

Scope: KD2 (AXE exact mode) + KD3 + REC on, noscan graph only (the scan graph is never replayed with the shipped
uc_warm / uc_every and keeps the parent body). Anything else -> not covered -> the parent graph."""
from __future__ import annotations

import contextlib

import torch

import lsk_head as _LSK  # LSK: head-input hook + s buffer (inert when off)


KD8_PARTS = frozenset(("bias", "ws2", "ze", "lskc", "mup", "lowke", "v", "pj", "ke2", "e2", "w2f", "bd", "mh"))


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
        self.k8 = None            # KD8: events / tensors of the moved pre-backbone work (forward_body)
        self.keep = []            # KD8: side-stream temporaries read on another stream, referenced to the end of the body
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
        k8 = S.k8 if S is not None else None
        if k8 is not None and k8["bias"]:      # KD8: the bias preloads were issued on side_c at the top
            S.wait(S.main, k8["evA"])
            S.wait(S.main, k8["evB"])
        else:
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
            if k8 is not None and k8.get("lsk") is not None:     # KD8: phi on side_c (same matmul flags: inside this block)
                S.fork_c()
                with S.on_side_c():
                    _LSK.HOOK(h)
                S.keep.append(h)   # FR34/MRG: side_c reads h -> keep it alive to the end of the body (no pool reuse)
            else:
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
    k8 = S.k8 if S is not None else None
    window_sum1 = torch.sum(uv, dim=1, out=rec.MW[1])    # same reduction, into the REC buffer
    rec.detect()                                          # reads MW = [previous m, this window's sum]
    after_detect()
    if k8 is not None and k8["ws2"] is not None:
        S.wait(S.main, k8["ev_ws2"])                      # KD8: (uv * uv).sum(dim=1) + the zV[..., 2] copy ran on side_b
        window_sum2 = k8["ws2"]
    else:
        window_sum2 = (uv * uv).sum(dim=1)
    torch.addcmul(window_sum1, base._static_S1, rec.keep, out=base._static_S1)
    torch.addcmul(window_sum2, base._static_S2, rec.keep, out=base._static_S2)
    torch.addcmul(rec.T_t, base._static_n, rec.keep, out=base._static_n)
    recip = base._static_n.reciprocal()                   # == _causal_ms_graphed, m written into rec.MW[0]
    m = torch.mul(base._static_S1, recip, out=rec.MW[0])
    J8f = k8["J8"] if (k8 is not None and k8["v"]) else None
    sbuf8 = None
    if J8f is not None:      # KD8 V: mul / mul / sub / clamp_min / sqrt in one kernel; S_BUF <- s copied (side_b if "ze")
        s = J8f.V(base._static_S2, recip, m, w._ms_floor)
        sbuf8 = _LSK.S_BUF
    else:
        var = base._static_S2 * recip - m * m
        var = torch.clamp_min(var, w._ms_floor)
        s = torch.sqrt(var) if _LSK.S_BUF is None else torch.sqrt(var, out=_LSK.S_BUF)   # LSK: s -> S_BUF (same kernel)
    ms, ss = m.unsqueeze(1), s.unsqueeze(1)
    zv2 = k8 is None or k8["ws2"] is None                 # zV[..., 2] copy still on main (KD8 "ws2" part off)
    if sbuf8 is not None and not k8["ze"]:
        sbuf8.copy_(s)
    if k8 is not None and k8["ze"]:
        S.fork_b()
        with S.on_side_b():                               # KD8: the zE chain on side_b (reads only m, s)
            if sbuf8 is not None:
                sbuf8.copy_(s)
                S.keep.append(s)
            logs = torch.log(s)
            torch.div(m - w._m_mean, w._m_std, out=st.zE[..., 0:2])
            torch.div(logs - w._logs_mean, w._logs_std, out=st.zE[..., 2:4])
            S.keep.append(logs)
            ev_ze = S.record(S.side_b or S.main)
        torch.div(x[..., :2] - ms, ss, out=st.zV[..., :2])
        if zv2:
            st.zV[..., 2:3].copy_(x[..., 2:3])
        S.wait(S.main, ev_ze)
    else:
        torch.div(x[..., :2] - ms, ss, out=st.zV[..., :2])
        if zv2:
            st.zV[..., 2:3].copy_(x[..., 2:3])
        logs = torch.log(s)
        torch.div(m - w._m_mean, w._m_std, out=st.zE[..., 0:2])
        torch.div(logs - w._logs_mean, w._logs_std, out=st.zE[..., 2:4])
    y_s = _KD3.packed_forward(st.pk, st, st.zh) if S is None else _packed_forward(st.pk, st, st.zh, S)
    raw = st.raw
    lsk8 = k8.get("lsk") if k8 is not None else None
    if lsk8 is not None:
        # KD8: LSKHead.apply_ split -- corr = baddbmm(Mb, M8, phi) on side_c (after the hook GEMM, default matmul flags
        # as in apply_), then main: kd7 P (y * ss + ms) and the native addcmul (corr * s) out= raw[..., :2]
        r = lsk8.r
        _lsk_issue_pending(S)      # no-op: the update was issued inside the backbone
        with S.on_side_c():
            torch.baddbmm(lsk8.Mh[:, :, r:], lsk8.Mh[:, :, :r], lsk8.phi.expand(2, r, _LSK.P), out=lsk8.corr)
        J8 = k8["J"]
        cv = lsk8.corr.permute(1, 2, 0).view(_LSK.T, _LSK.H, _LSK.W, 2)
        if J8 is not None:
            t8 = J8.P(y_s[..., :2], ss, ms)
            S.join_c()
            torch.addcmul(t8, cv, lsk8.s_buf[0], out=raw[..., :2])
        else:
            torch.add(y_s[..., :2] * ss, ms, out=raw[..., :2])
            S.join_c()
            raw[0, ..., :2].addcmul_(cv, lsk8.s_buf[0])
    else:
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
        # MRG (LSR x KD8): under KD8 "lskc" the corr baddbmm (the apply's read of Mh) runs on side_c, so LSR's REC
        # reset runs there too, right after the update that writes _state / Mh (same stream -> ordered); rec.keep was
        # written on main by rec.detect (ctx-norm) before the fork_c below. Without "lskc": main, after join_c (LSR).
        rr8 = _lsk.rec_reset and bool(getattr(w, "_kd8", False)) and \
            "lskc" in (getattr(w, "_kd8_parts", None) or KD8_PARTS)
        def _lsk_issue():
            S.fork_c()
            with S.on_side_c():
                _lsk.update(y)
                if rr8:
                    _lsk.rec_reset_(rec.keep)   # LSX: REC reset clears the LS state (x 1.0 when REC does not fire)
        S.lsk_pending = _lsk_issue
    K8 = bool(getattr(w, "_kd8", False))
    P8 = getattr(w, "_kd8_parts", None) or KD8_PARTS      # diagnostic subset (CONFIG "kd8" as a list); True = all
    J8 = getattr(w, "_kd8j", None) if K8 else None       # kd8_fuse.Fused8 (compiled + self-tested), None -> native
    E2 = False
    if K8:
        # KD8: backbone-independent pre-backbone work off the main stream (see the module docstring)
        if S.concurrent and (S.side_b is None or S.side_c is None):
            raise RuntimeError("kd8: side_b and side_c required")
        J7 = getattr(w, "_kd7j", None)
        evA = evB = ws2 = ev_ws2 = None
        if "bias" in P8:
            S.fork_c()
            with S.on_side_c():
                st.bufA.copy_(st.biasA.expand(-1, st.hwA))
                evA = S.record(S.side_c or S.main)
                st.bufB.copy_(st.biasB.expand(-1, st.hwB))
                evB = S.record(S.side_c or S.main)
        if "ws2" in P8:
            S.fork_b()
            with S.on_side_b():
                st.zV[..., 2:3].copy_(x[..., 2:3])
                uv8 = x[..., :2]
                ws2 = (uv8 * uv8).sum(dim=1)
                ev_ws2 = S.record(S.side_b or S.main)
            S.keep.append(ws2)
        S.k8 = dict(bias="bias" in P8, evA=evA, evB=evB, ws2=ws2, ev_ws2=ev_ws2, ze="ze" in P8, v="v" in P8, J8=J8,
                    lsk=_lsk if "lskc" in P8 else None, J=J7 if (J7 is not None and J7.t2) else None)
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
    if _lsk is not None and not (K8 and "lskc" in P8):
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
        if K8 and "mup" in P8:     # KD8: mu_p reads only new_mu (this section) -> slot 1 of the (fl_f, mu_p) pair, here
            mu_p = torch.sub(new_mu, new_mu.mean(dim=1, keepdim=True), out=b._fm[1:2])

    # =========================== MAIN: mf, fl_f, join, backbone-dependent rest ===========================
    S.wait(S.main, ev_blend)
    mf = torch.mean(raw, dim=1, keepdim=True, out=w._mf)
    fl_f = torch.sub(raw, mf, out=mega._fm[0:1])   # slot 0 of mega's (fl_f, mu_p) pair
    S.join()
    io._fl_f = fl_f
    PJ = K8 and "pj" in P8 and J8 is not None
    if PJ:
        # KD8: after the join -- side_b: fl_f's T-halves variance (fh) + the leads gather; side_c: the x-only rows of
        # the log-energy features (1, 3, 4, 6, 7: read ex / runx / mrx / xh / the (ef, ex)_prev carry, all final here);
        # main: e_f, e_pred (EP), sf (SF), the fl slot, the three backbone rows (LG) -- the same values row by row
        FC, LC = KF.FIELD_CLAMP, KF.LOG_CLAMP
        en = lambda c: FT[:, c:c + 8].view(1, 4, 2, Hh, Wh).permute(1, 0, 3, 4, 2)          # noqa: E731
        S.fork_b()
        with S.on_side_b():
            fh = fl_f.view(1, 2, 10, Hh, Wh, Ch)[..., :2].var(dim=2, unbiased=False)
            ffl_sel = fl_f[..., :2].index_select(1, mega._leads_idx)
            ev_fh = S.record(S.side_b or S.main)
        S.fork_c()
        with S.on_side_c():
            ex_ = kx[:, 0, ..., :2]
            ex1_, ex2_ = xh[:, 0], xh[:, 1]
            lgx = torch.log((torch.stack([ex_, runx, ex2_, ex_, ex_], 0) + EPS_F) /
                            (torch.stack([runx, mrx.expand_as(runx), ex1_, b._ef_prev, b._ex_prev], 0) + EPS_F))
            torch.clamp(lgx[0:1], -LC, LC, out=en(c0)[1:2])
            torch.clamp(lgx[1:2], -LC, LC, out=en(c0)[3:4])
            torch.clamp(lgx[2:3], -LC, LC, out=en(c0 + 12)[0:1])
            torch.clamp(lgx[3:5], -LC, LC, out=en(c0 + 12)[2:4])
            ev_lgx = S.record(S.side_c or S.main)
        S.keep.extend([fh, ffl_sel, lgx])
    # ---- mega energy half (rest, verbatim order)
    if not (K8 and "mup" in P8):
        mu_p = torch.sub(new_mu, new_mu.mean(dim=1, keepdim=True), out=b._fm[1:2])
    v2 = b._fm.var(dim=1, keepdim=True, unbiased=False)
    e_f, vM = v2[0:1], v2[1:2]
    if PJ:
        new_e_pred = J8.EP(b._e_pred, b._a_e_f32, e_f)      # b._e_pred <- new_e_pred: second fork, side
        ex, ef = kx[:, 0, ..., :2], e_f[:, 0, ..., :2]
        rf = new_e_pred[:, 0, ..., :2]
        runf = rf
        sf = J8.SF(runf.mean(dim=(1, 2), keepdim=True))
        S.wait(S.main, ev_fh)
        torch.div(ffl_sel, sf.unsqueeze(1), out=slot(nk, L_))
        FT[:, 0:nk + nl].clamp_(-FC, FC)
        ef1, ef2 = fh[:, 0], fh[:, 1]
        o0, o2, o5 = J8.LG(ef, runx, ef2, runf, ef1)
        en(c0)[0].copy_(o0)
        en(c0)[2].copy_(o2)
        en(c0 + 12)[1].copy_(o5)
        S.wait(S.main, ev_lgx)
    else:
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
        if PJ:
            b._e_pred.copy_(new_e_pred)     # KD8: the state write of EP (b._e_pred == st.E2[1], read right below)
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
        if K8 and "ke2" in P8:     # KD8: the T-variance of fluc0 on side_b || cFM on main
            S.fork_b()
            with S.on_side_b():
                e_tgt_ach = _ke(fluc0)
                ev_ke = S.record(S.side_b or S.main)
            S.keep.append(e_tgt_ach)
            cFM = (fluc0 * mu_p).mean(dim=1, keepdim=True)
            S.wait(S.main, ev_ke)
        else:
            e_tgt_ach = _ke(fluc0)
            cFM = (fluc0 * mu_p).mean(dim=1, keepdim=True)
        c = J.D(cFM, e_tgt_ach, vM)
        E2 = K8 and "e2" in P8 and J8 is not None
        if E2:     # KD8: E + the new_mu add in one kernel; b._eb_outK <- outK on the third fork's side stream
            outK = J8.E2(constK, c, fluc0, new_mu)
        else:
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
        if J is not None and E2:
            b._eb_outK.copy_(outK)     # KD8: the state write of E2
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
    lk8 = K8 and "lowke" in P8 and bool(getattr(w, "_lowke", None))
    if lk8:
        # KD8: TKA LOWKE on the side stream of the third fork. pred_phys (just issued) is main's last reader of the
        # un-damped pred P; the side's in-place update of P waits for it. == w._lowke_apply() / J.lowke_apply(w).
        ev_pp = S.record(S.main)
        with S.on_side():
            S.wait(S.side, ev_pp)
            Pk = w._dev_out_views[0]
            w._pred_raw.copy_(Pk)
            var_k, mean_k = torch.var_mean(Pk, dim=1, keepdim=True, unbiased=False)
            Kk = torch.matmul(var_k, w._lk_kw)
            if T2 and J.lk is not None:
                Gk = J.L(Kk, torch.mean(Kk), w._lk_negchm, w._lk_chm)
            else:
                kb = torch.mean(Kk).clamp_min_(1e-20).mul_(w._lk_theta)
                g = torch.div(Kk, kb).pow_(w._lk_delta).clamp_(w._lk_gmin, 1.0)
                Gk = torch.addcmul(w._lk_negchm, g, w._lk_chm)
            Pk.addcmul_(Gk, torch.sub(Pk, mean_k))
    sp = sig_pipe_phys.squeeze(0).permute(3, 0, 1, 2)
    if st.near:
        # ---- kd3_step._kd5_features: the pred part (x part on the side stream above)
        XP[1].copy_(pred_phys)
        W2F = K8 and "w2f" in P8 and T2
        if W2F:
            # KD8: the pred row's T-std (-> L6[3]) on side_b and the dp / f5 branch (-> INH16 channel 10) on side_c,
            # || the T-mean / grad-magnitude on main; joined before the log of L6 / the INH16 channels 0:10 copy
            S.fork_b()
            with S.on_side_b():
                pred_std = XP[1].std(dim=1, unbiased=True)
                L6[3].copy_(pred_std)
                ev_sd = S.record(S.side_b or S.main)
            S.fork_c()
            with S.on_side_c():
                prev_t = torch.cat([last.unsqueeze(1), pred_phys[:, :-1]], dim=1)
                f5n = J.F(pred_phys, prev_t, st.mu5, st.sd5)
                st.INH16[:, :, 10].copy_(f5n)
                ev_f5 = S.record(S.side_c or S.main)
            S.keep.extend([pred_std, prev_t, f5n])
            meanpred = XP[1].mean(dim=1)
            gm1_0 = J.grad_mag(meanpred)
            L6[5].copy_(gm1_0)
            S.wait(S.main, ev_sd)
            torch.log(L6 + EPS_W, out=CT[1:7])
            CTn = J.N(CT, st.mu10, st.sd10)
            S.wait(S.main, ev_f5)
            st.INH16[:, :, 0:10].copy_(CTn.permute(1, 0, 2, 3).unsqueeze(1).expand(bW, T, 10, Hh, Wh))
        else:
            meanpred = XP[1].mean(dim=1)
            pred_std = XP[1].std(dim=1, unbiased=True)
            if T2:     # KD7: replicate pad + 1 kernel (the stack of one row is a copy: dropped, same values)
                gm1_0 = J.grad_mag(meanpred)
            else:
                gm1_0 = _CW._t2_grad_mag(torch.stack([meanpred], 0))[0]
            L6[3].copy_(pred_std)
            L6[5].copy_(gm1_0)
            torch.log(L6 + EPS_W, out=CT[1:7])
        if W2F:
            pass
        elif T2:     # KD7: dp = pred_phys - [last, pred_phys[:, :-1]]; log |dp| + eps, (- mu5) / sd5 in one kernel
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
        MH = K8 and "mh" in P8 and J8 is not None and T2
        if MH:     # KD8: the fp16 net output read by kd7 M directly (the kernel's load cast == .float(), exact)
            mm = n_.out(hh).view(2, T, Hh, Wh)
        else:
            mm = n_.out(hh).float().view(2, T, Hh, Wh)
    else:
        MH = False
        inp = st.INH.view(2 * T, -1, Hh, Wh)
        mm = net_h(inp).float().view(2, T, Hh, Wh)
    if T2:     # KD7: half1[..., :2] * (clamp(exp(alpha mm), lo, hi) * lam) in one kernel
        mh_ = J8.MH(J, half1[..., :2], mm.permute(1, 2, 3, 0).unsqueeze(0)) if MH else \
            J.M(half1[..., :2], mm.permute(1, 2, 3, 0).unsqueeze(0))
        out_half = torch.cat([mh_, half1[..., 2:].expand(1, T, Hh, Wh, 1)], dim=-1)
    else:
        mult = torch.exp(cw.alpha * mm).clamp(cw.clip_lo, cw.clip_hi) * cw.lam
        mult2 = mult.permute(1, 2, 3, 0).unsqueeze(0)
        out_half = torch.cat([half1[..., :2] * mult2, half1[..., 2:].expand(1, T, Hh, Wh, 1)], dim=-1)
    S.join()
    if K8 and "bd" in P8 and J8 is not None:
        # KD8 BD: lower / upper / width / both CB shifts in one kernel -> one copy into the output pack (main) and the
        # width state (side_b)
        pk8, wd8 = J8.BD(mean, out_half, d_cb, ks.cbw, ks.f2)
        w._pack12.copy_(pk8)
        S.fork_b()
        with S.on_side_b():
            w._cb_wd.copy_(wd8[0:1][..., :2])
        S.keep.append(wd8)
    else:
        torch.sub(mean, out_half, out=w._dev_out_views[1])
        torch.add(mean, out_half, out=w._dev_out_views[2])
        torch.sub(w._dev_out_views[2][..., :2], w._dev_out_views[1][..., :2], out=w._cb_wd)
        w._pack12.addcmul_(ks.cbw, d_cb)
        w._pack12[..., :2].addcmul_(ks.f2, w._cb_wd)
    if getattr(w, "_lowke", None) and not lk8:
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
    if getattr(w.model, "_lsk", None) is not None or getattr(w, "_kd8", False):
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
    stats["kd8"] = bool(getattr(w, "_kd8", False))
    stats["side_stream"] = int(side.cuda_stream)
    stats["side_stream_b"] = int(side_b.cuda_stream)
    if side_c is not None:
        stats["side_stream_c"] = int(side_c.cuda_stream)
    return graph, (side, side_b, side_c), stats


def kd8_selftest(w, rounds: int = 4, seed: int = 20260927):
    """KD8's one reused KD7 kernel at its new operands: P(y_s[..., :2], ss, ms) (y_s = the packed output's permuted
    (1,T,H,W,3) view, ss / ms = (1,1,H,W,2) broadcasts) vs the native y_s[..., :2] * ss + ms (mul, add), bitwise, on
    random draws with NaN / +-inf / +-0 / denormal / huge / tiny values; then the native addcmul out= vs the in-place
    addcmul_ it replaces (fp16 corr, broadcast s). Raises on any difference."""
    import kd7_fuse as _KD7
    J = getattr(w, "_kd7j", None)
    if J is None or not J.t2 or getattr(w.model, "_lsk", None) is None:
        return dict(p_checked=False)
    dev = w._dev_in.device
    T, H, W_ = w._dev_in.shape[1], w._dev_in.shape[2], w._dev_in.shape[3]
    gen = torch.Generator().manual_seed(seed)
    n_el = 0
    for r in range(rounds):
        N = lambda *sh, **kw: _KD7._nasty(sh, gen, dev, **kw)  # noqa: E731
        y60 = N(1, T, 3, H, W_)                          # packed output layout: (1, T*3, H, W) -> (1,T,H,W,3) view
        y_s = y60.permute(0, 1, 3, 4, 2)
        s_ = N(1, H, W_, 2, positive=r % 2 == 1)
        m_ = N(1, H, W_, 2)
        ms, ss = m_.unsqueeze(1), s_.unsqueeze(1)
        ref = torch.empty(1, T, H, W_, 3, device=dev)
        torch.add(y_s[..., :2] * ss, ms, out=ref[..., :2])
        t8 = J.P(y_s[..., :2], ss, ms)
        if not _KD7._bits_equal(t8, ref[..., :2].contiguous()):
            raise RuntimeError(f"kd8: self-test P mismatch (round {r})")
        corr = N(2, T, H * W_).to(torch.float16)
        cv = corr.permute(1, 2, 0).view(T, H, W_, 2)
        a = ref.clone()
        a[0, ..., :2].addcmul_(cv, s_[0])
        b = ref.clone()
        torch.addcmul(t8, cv, s_[0], out=b[..., :2])
        if not _KD7._bits_equal(a, b):
            raise RuntimeError(f"kd8: self-test addcmul mismatch (round {r})")
        n_el += t8.numel()
    torch.cuda.synchronize()
    return dict(p_checked=True, rounds=rounds, elements=n_el)
