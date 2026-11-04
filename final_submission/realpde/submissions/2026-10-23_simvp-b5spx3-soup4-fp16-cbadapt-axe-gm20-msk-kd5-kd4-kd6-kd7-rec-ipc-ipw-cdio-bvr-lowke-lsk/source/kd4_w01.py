"""KD4 (experiments/kd4_w01_diet, 2026-09-24): bit-exact diet + fork/join concurrency of the W01 window-0 / window-1
CUDA graphs (w01_warmup_graph.py; CONFIG "kd4_w01"). Used ONLY while W01 captures G0 / G1: every hook below is an
instance (or module) attribute installed for the duration of one capture body and removed afterwards, so the eager
step, every fallback path and the whole-step graph run the unchanged code.

What changes inside G0 / G1 (same ops on the same values -- or KD / KD3's verified bit-exact twins of them --
fewer copies / kernels, shorter critical path):
  * mask: `GraphedMaskWrapper._body(x)` called directly (no copy in / copy out / clone).
  * backbone: `kd3_step.ctx_body` (the KD3 twin of `GraphedCtxNormWrapper._body` the steady whole-step graph already
    uses: GEMM bias prefill, in-place input assembly) on the whole-step wrapper's KD3 buffers; without them,
    `base._body(x)` (no copies).
  * convection-speed scan: `GraphedScanUcWrapper._body` on the step's own (fl_x, obstacle, mask), no copies, on a
    SIDE STREAM forked right after the mask and joined where the step asks for the scan (`_scan_uc_dispatch`), i.e.
    concurrently with the backbone (the scan reads only x and the mask, the backbone only x and its own buffers).
  * window 0 is captured as TWO graphs sharing one pool (W01 `_capture_g0_split`): g0 = mask + backbone || scan,
    split at the scan request; g0b starts with the device part of the (Re, AoA) prior (moved there: it reads only
    the prior tables and the two scalars, and nothing before the split point reads its outputs) and holds the rest.
    At runtime g0 is launched BEFORE the host (Re, AoA) estimate, which then overlaps g0's GPU work.
  * window 1: the PRE-stage block of step() (blend_y, _update_error / _update_ctr / _update_blend, _update_R, the
    cond-width seed update) runs on the side stream: switched to right after `adapt_loss` is formed, switched back
    at the mask. Nothing the main stream runs between the fork and the join (mask, backbone, fl_f / fl_x) reads what
    that block writes, and the block reads nothing the main stream writes there.
  * ke_net_flow features: `ke_net_flow.features_kd` (KD's bit-exact twin, same signature) in place of `features`;
    window 1 also takes `_KeFlow.__call__`'s own eager branch (the op sequence the keflow graph was transcribed
    from) instead of the keflow graph wrapper's copy-in / copy-out forward.
  * window 1: the tail graph's `_forward_body` with its static inputs / state rebound to the live tensors (same
    shape / strides / 16-B alignment -- else a fresh contiguous copy, == the parent's copy_) and fresh outputs (no
    copy in, no state sync, no clones).
  * bounds (both windows, cond-width branch, fp32 width net): `bounds_fp32` = the steady graph's
    t2_cw_apply construction (k-table on var's own shape, KD fused features, channel-first stack) with the SAME
    fp32 net `cw.net` (NOT the graph-only fp16 copy) and window 0's neutral extra features.
Fork / join only while a capture is active (warm-ups run the same hooks on one stream). All side-stream work of a
capture precedes its single join, so no side-stream block is recycled while a main-stream kernel still reads it.
Guards: W01 self-test (graph vs the unchanged eager code, bit-exact) + any failure -> W01 without KD4."""
from __future__ import annotations

import torch

_TAIL_IN = ("_x", "_mask_in", "_fl_f", "_mf", "_e_f", "_gamma")
_TAIL_STATE = ("_mu", "_M0", "_M1", "_M2", "_Saa", "_Sab", "_Sbb", "_S02", "_S12", "_S22")
_TAIL_OUT = ("_out_adv", "_out_const", "_out_basis", "_out_b2", "_out_out", "_out_ctr_out", "_out_ctr_var")
_SHADOW = ("M0", "M1", "M2", "Saa", "Sab", "Sbb", "S02", "S12", "S22", "uc", "uc_eff")


def _like(t, buf) -> bool:
    return (isinstance(t, torch.Tensor) and t.shape == buf.shape and t.dtype == buf.dtype and t.device == buf.device
            and t.stride() == buf.stride() and t.data_ptr() % 16 == 0 and not t.requires_grad)


def preconditions(model) -> None:
    cfg = model.cfg
    if cfg["uc_fixed"] is not None or cfg["uc_guard"] or not cfg["advect"]:
        raise RuntimeError("kd4: uc_fixed / uc_guard off and advect on expected")
    for name in ("_maskg", "_scang", "_tail"):
        w = getattr(model, name, None)
        if w is None or not getattr(w, "ok", False):
            raise RuntimeError(f"kd4: {name} expected")


class Hooks:
    """Context manager around ONE W01 capture body (`window` 0 or 1)."""

    def __init__(self, warm, window: int, stats: dict):
        self.w, self.m, self.window, self.stats = warm, warm.model, window, stats
        self.side = warm._kd4_side
        self.kd3s = warm._kd4_kd3s
        # fork / join only inside a capture, and only when the side stream's allocations can be routed to the
        # capture's private pool (torch 2.2 routes ONLY the capturing stream's allocations there by default: a side
        # stream would otherwise allocate from the regular pool -> freed / reused / released behind the graph's back)
        self.pool = getattr(warm, "_cur_pool", None)
        self.concurrent = (bool(torch.cuda.is_current_stream_capturing()) and warm._kd4_conc and self.pool is not None
                           and pool_routing_available())
        self.registered = False
        self.main = None
        self.on_side = False
        self.forked = False
        self.scan_res = None
        self.mask = None
        self.keep = []
        self._installed = []
        self.prior_fn = None                                # window 0 (split): the deferred device prior
        self.split = bool(getattr(warm, "_kd4_split", False)) and window == 0

    # ------------------------------------------------------------ install / remove
    _MISSING = object()

    def _set(self, obj, name, val):
        self._installed.append((obj, name, obj.__dict__.get(name, self._MISSING)))
        obj.__dict__[name] = val

    def __enter__(self):
        m = self.m
        self.main = torch.cuda.current_stream()
        self._set(m._maskg, "forward", self._mask_fwd)
        self._set(m.base, "forward", self._base_fwd)
        self._set(m, "_scan_uc_dispatch", self._dispatch)
        import ke_net_flow as _KF
        if getattr(_KF, "features_kd", None) is not None:
            self._set(_KF, "features", _KF.features_kd)      # KD's bit-exact fewer-kernel twin (same signature)
        if bounds_covered(m):
            self._set(m, "bounds", self._bounds)
        if self.window == 1:
            kfw = m.ke_flow._graph
            if kfw is not None:
                self._set(kfw, "should_use", lambda: False)     # -> _KeFlow.__call__'s own eager branch
            self._set(m._tail, "forward", self._tail_fwd)
        return self

    def __exit__(self, *exc):
        pending = self.prior_fn is not None
        self.prior_fn = None
        try:
            if self.on_side:
                torch.cuda.set_stream(self.main)
                self.on_side = False
            try:
                self.join()
            finally:
                self._unregister()
        except Exception:  # noqa: BLE001 - never mask the original error; hooks are restored below either way
            if exc[0] is None:
                raise
        finally:
            for obj, name, old in reversed(self._installed):
                if old is self._MISSING:
                    obj.__dict__.pop(name, None)
                else:
                    obj.__dict__[name] = old
            self._installed = []
            torch.cuda.set_stream(self.main)
        if pending and exc[0] is None:
            raise RuntimeError("kd4: deferred prior never ran (split point not reached)")
        return False

    # ------------------------------------------------------------ streams
    def fork(self):
        if not self.registered:
            # route the side stream's allocations to this capture's private pool (released again at the join)
            with torch.cuda.stream(self.side):
                torch._C._cuda_beginAllocateCurrentStreamToPool(self.side.device.index, self.pool)
            self.registered = True
        self.side.wait_stream(self.main)
        self.forked = True

    def join(self):
        if self.forked:
            self.main.wait_stream(self.side)
            self.forked = False
        self._unregister()

    def _unregister(self):
        if self.registered:
            self.registered = False
            with torch.cuda.stream(self.side):
                torch._C._cuda_endAllocateCurrentStreamToPool(self.side.device.index)
            torch._C._cuda_releasePool(self.side.device.index, self.pool)

    def prior_side(self, fn):
        """window 0: the device part of the prior -- deferred to the split point (split mode, it needs the host
        estimate), else on the side stream (forked here)."""
        if self.split:
            self.prior_fn = fn
            return None
        if not self.concurrent:
            return fn()
        self.fork()
        with torch.cuda.stream(self.side):
            fn()
        self.stats["prior_side"] = self.stats.get("prior_side", 0) + 1

    def pre_to_side(self):
        """window 1: called right after adapt_loss is formed; the rest of the PRE block runs on the side stream."""
        if not self.concurrent:
            return
        self.fork()
        torch.cuda.set_stream(self.side)
        self.on_side = True
        self.stats["pre_side"] = self.stats.get("pre_side", 0) + 1

    # ------------------------------------------------------------ hooks
    def _scan_will_run(self) -> bool:
        m = self.m
        cfg = m.cfg
        return (cfg["uc_fixed"] is None and m._scang is not None and m._scang.should_use()
                and (m.uc is None or m.n_win < int(cfg["uc_warm"]) or m.n_win % int(cfg["uc_every"]) == 0))

    def _mask_fwd(self, x):
        m = self.m
        if self.on_side:
            torch.cuda.set_stream(self.main)
            self.on_side = False
        mask = m._maskg._body(x)
        self.mask = mask
        if self.concurrent and self._scan_will_run():
            self.fork()                                   # side: (prior | PRE) then the scan, after the mask
            with torch.cuda.stream(self.side):
                obstacle = m.shift.last_obstacle(mask)
                mx = x.mean(dim=1, keepdim=True)
                fl_x = x - mx
                self.scan_res = m._scang._body(fl_x, obstacle, mask)
                self.keep += [obstacle, mx, fl_x]
            self.stats["scan_side"] = self.stats.get("scan_side", 0) + 1
        else:
            self.join()                                   # nothing to overlap with: join before the backbone
        return mask

    def _base_fwd(self, x):
        base = self.m.base
        if self.kd3s is not None:
            import kd3_step as _KD3
            self.stats["kd3_ctx"] = self.stats.get("kd3_ctx", 0) + 1
            st = self.kd3s
            if getattr(st, "near", False):
                # FZ (experiments/fz_freeze): with KD5 ("kd5_near") the shared KD3 state carries the NEAR-EXACT
                # backbone branches of the steady graph; G0 / G1 must stay bit-exact to the eager code (W01
                # self-test), so they are captured through KD3's exact branches (same state, near off while tracing).
                st.near = False
                try:
                    return _KD3.ctx_body(base, x, st)
                finally:
                    st.near = True
            return _KD3.ctx_body(base, x, st)
        return base._body(x)

    def _dispatch(self, fl, obstacle, mask):
        m = self.m
        if self.scan_res is not None and mask is self.mask:
            self.join()
            r, self.scan_res = self.scan_res, None
        else:
            self.join()
            self.stats["scan_main"] = self.stats.get("scan_main", 0) + 1
            r = m._scang._body(fl, obstacle, mask)
        if self.prior_fn is not None:
            # window 0, split mode: end the first graph here, the second starts with the device prior
            if self.w._kd4_split_now():
                self.stats["split"] = self.stats.get("split", 0) + 1
            fn, self.prior_fn = self.prior_fn, None
            fn()
        return r

    def _bounds(self, mean, x, var=None, k_scale=None, pred_out=None):
        m = self.m
        if (var is None or k_scale is None or pred_out is None or var.dim() != 5 or var.shape[1] != 1
                or tuple(var.shape[2:]) != tuple(mean.shape[2:]) or mean.dim() != 5):
            self.stats["bounds_parent"] = self.stats.get("bounds_parent", 0) + 1
            return type(m).bounds(m, mean, x, var, k_scale, pred_out)
        self.stats["bounds_kd"] = self.stats.get("bounds_kd", 0) + 1
        return bounds_fp32(m, mean, x, var, float(k_scale), pred_out)

    def _tail_fwd(self, x, mask, fl_f, mf, e_f, gamma):
        m, tw = self.m, self.m._tail
        s = tw.shadow
        staged = {}                                        # (id(live), layout) -> tensor the body reads

        def stage(t, buf, key):
            """The live tensor itself when it has the static buffer's layout; else a fresh contiguous copy of it
            (== the parent's `buf.copy_(t)`); None -> not covered (parent forward)."""
            key_ = (id(t), tuple(buf.shape), buf.stride(), buf.dtype)
            if key_ in staged:
                return staged[key_]
            if _like(t, buf):
                r = t
            elif (isinstance(t, torch.Tensor) and t.shape == buf.shape and t.dtype == buf.dtype
                  and t.device == buf.device and not t.requires_grad):
                r = torch.empty_like(buf)
                r.copy_(t)
                self.stats["tail_staged"] = sorted(set(self.stats.get("tail_staged", [])) | {key})
            else:
                return None
            staged[key_] = r
            return r
        ins = {}
        for k, v in zip(_TAIL_IN, (x, mask, fl_f, mf, e_f, gamma)):
            ins[k] = stage(v, getattr(tw, k), k)
        for k in _TAIL_STATE:
            ins[k] = stage(getattr(m, k[1:]), getattr(tw, k), k)
        sh = {k: stage(getattr(m, k), getattr(s, k), "s." + k) for k in _SHADOW}
        S = None
        if isinstance(m.S, dict) and m.S.keys() == s.S.keys():
            S = {k: stage(m.S[k], s.S[k], "S." + k) for k in s.S}
        bad = [k for k, v in list(ins.items()) + list(sh.items()) if v is None]
        if S is None or any(v is None for v in S.values()):
            bad.append("S")
        if bad:
            self.stats["tail_parent"] = self.stats.get("tail_parent", 0) + 1
            self.stats["tail_parent_why"] = bad
            return type(tw).forward(tw, x, mask, fl_f, mf, e_f, gamma)
        outs = {k: torch.empty_like(getattr(tw, k)) for k in _TAIL_OUT}
        saved_t = {k: tw.__dict__[k] for k in list(ins) + list(outs)}
        saved_s = {k: s.__dict__[k] for k in list(sh) + ["S"]}
        try:
            tw.__dict__.update(ins)
            tw.__dict__.update(outs)
            s.__dict__.update(sh)
            s.S = S
            tw._forward_body()
        finally:
            tw.__dict__.update(saved_t)
            s.__dict__.update(saved_s)
        self.stats["tail_direct"] = self.stats.get("tail_direct", 0) + 1
        return tuple(outs[k] for k in _TAIL_OUT)


def pool_routing_available() -> bool:
    return all(hasattr(torch._C, n) for n in ("_cuda_beginAllocateCurrentStreamToPool",
                                             "_cuda_endAllocateCurrentStreamToPool", "_cuda_releasePool"))


def bounds_covered(m) -> bool:
    import cond_width_net as _CW
    cw = getattr(m, "cw", None)
    return (bool(m.cfg.get("cond_width")) and cw is not None and type(cw).apply is _CW._t2_apply
            and getattr(_CW, "_kd_base_feature_list", None) is not None)


def bounds_fp32(m, mean, x, var, k_scale, pred_out):
    """== GainMapImprovedModel.bounds(mean, x, var, k_scale, pred_out) with the cond-width branch
    (cond_width_net._t2_apply, fp32 net) -- windows 0 / 1 -- built the way the steady graph's
    t2_cw_graph_body.t2_cw_apply builds it (k-table chain on var's (1,1,H,W,3) shape then broadcast; the
    KD fused feature builder; features stacked in the net's channel-first layout), with the SAME fp32 net
    (`cw.net`, not the graph-only fp16 copy). Window 0 (no error map yet): the neutral extra features of
    _t2_apply (log(sig_pipe + EPS) twice, log1p(EPS)). Every value equals the eager one element for element;
    the net sees the same (2T, 12, H, W) contiguous input."""
    import cond_width_net as _CW
    from submission import _SIGMA_GLOBAL
    cw = m.cw
    std1 = var.sqrt()                                                    # (1,1,H,W,3)
    a = 2.0 * std1 * m.std_tgt / _SIGMA_GLOBAL
    pos = (a.clamp_min(1e-12).log() - m.log_a[0]) / (m.log_a[1] - m.log_a[0])
    k = m.k_tab[pos.round().long().clamp(0, len(m.k_tab) - 1)]
    half1 = k_scale * k * std1                                           # (1,1,H,W,3)
    T, H, W = mean.shape[1], mean.shape[2], mean.shape[3]
    sig_pipe_phys = std1[..., :2] * m.std_tgt[:2]                        # (1,1,H,W,2)
    x_phys = (x[..., :2] * m.std_in[:2] + m.mean_in[:2]).squeeze(0).permute(3, 0, 1, 2)
    pred_phys = (pred_out[..., :2] * m.std_tgt[:2] + m.mean_tgt[:2]).squeeze(0).permute(3, 0, 1, 2)
    sp = sig_pipe_phys.squeeze(0).permute(3, 0, 1, 2)                    # (2,1,H,W)
    base = _CW._kd_base_feature_list(x_phys, pred_phys, sp.expand(-1, T, -1, -1))
    prev2 = getattr(cw, "_prev2", None)
    if prev2 is None:
        neutral = base[0]                                                # == log(sig_pipe + EPS) at full T
        extra = [neutral, neutral, torch.full_like(neutral, _CW.LOG1P_EPS)]
    else:
        ratio = (prev2 / sp[:, 0].clamp_min(_CW.EPS)).clamp_min(0.0)
        lp, le, lr = _CW._kd_logs([prev2, cw._ema2, ratio])
        ex = lambda t: t.unsqueeze(1).expand(-1, T, -1, -1)  # noqa: E731
        extra = [ex(lp), ex(le), ex(lr)]
    feats = torch.stack(base + extra, dim=2)                             # (2,T,12,H,W)
    feats = (feats - cw.mu.view(1, 1, -1, 1, 1)) / cw.sd.view(1, 1, -1, 1, 1)
    inp = feats.view(2 * T, -1, H, W)
    mm = cw.net(inp).view(2, T, H, W)
    mult = torch.exp(cw.alpha * mm).clamp(cw.clip_lo, cw.clip_hi) * cw.lam
    mult2 = mult.permute(1, 2, 3, 0).unsqueeze(0)                        # (1,T,H,W,2)
    out_half = torch.cat([half1[..., :2] * mult2, half1[..., 2:].expand(1, T, H, W, 1)], dim=-1)
    return mean - out_half, mean + out_half
