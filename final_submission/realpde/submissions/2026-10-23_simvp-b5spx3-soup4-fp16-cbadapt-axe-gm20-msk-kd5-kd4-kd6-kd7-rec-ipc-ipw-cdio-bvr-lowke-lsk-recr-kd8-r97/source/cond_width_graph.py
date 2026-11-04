"""P4 (v2, follow-up pass) -- fold the P3 "conditional width" stage
(cond_width_net.py) into ONE CUDA graph, closing both the launch-count
regression (submissions/2026-09-28_.../manifest.md S5, +318 launches/step)
AND, in this v2 pass, the CPU-side host-dispatch cost the v1 two-graph
design still carried (see the coordinator follow-up: +1.0-1.15 ms/step on
only +2 `cudaLaunchKernel` events -- profiled root cause: `cudaMemcpyAsync`
dispatch for the `.copy_()` calls staging graph inputs, NOT kernel-launch
count; `torch.profiler` self_cpu_time breakdown, 300 steady-state steps,
shim_test: bundle 4.567 ms/step vs v1 graphed 5.603 ms/step, delta
+1.036 ms/step, of which `cudaMemcpyAsync` alone is +0.824 ms/step (~80%)
and `cudaGraphLaunch` (2 replays/step in v1) is +0.090 ms/step).

v1 used TWO separate `CUDAGraph`s (`update_graph`, `apply_graph`), called
from two different program points (`step()`'s cw update block; `bounds()`
via `ExperimentTTT.ttt_step`) -- this v2 merges them into ONE graph
(`self.graph`), replayed ONCE per step from `ttt_step` (after `step()`
returns), via a values-preserving REORDER of the update computation:

  - The update math (`CondWidthState.update_channel`'s BLEND branch) only
    ever reads `prev_target` and `ctr_old` (= `self.ctr_out` as it stood
    BEFORE `step()` reassigns it for the new window) plus this wrapper's own
    persistent state -- nothing computed LATER in `step()` (mask, raw
    backbone output, ctr reassignment, ...) feeds it, and nothing between
    the ORIGINAL call site and `ttt_step`'s `bounds()` call reads
    `cw.prev_err`/`ema_err` in between (only `CondWidthState.apply` does,
    and that always runs strictly after). So computing the update AFTER
    `step()` returns instead of at its original mid-`step()` position changes
    no observable value -- the same "reordering (values-preserving)"
    argument `cuda_graph_tail.py`'s own module docstring already uses for
    its `gamma`/energy_gain reorder.
  - `ctr_old` is captured via a PLAIN PYTHON REFERENCE ASSIGNMENT inside
    `step()`, at the ORIGINAL call site, immediately before `self.ctr_out`
    is reassigned (`self._cwg._ctr_old = self.ctr_out` -- zero GPU cost, no
    kernel, no copy; a python attribute now points at the same tensor
    object). `ttt_step` reads it back after `step()` returns, alongside the
    NEW `self.ctr_out`/`self.ctr_var`/`x`/`out` values `bounds()` already
    needs -- ONE `combined_step(prev_target, ctr_old, mean, var, x,
    pred_out)` call does update-then-apply as one captured sequence, one
    `.replay()`.
  - The gating condition also collapses cleanly: the original
    `prev_target is not None and self.prev_out is not None` (checked inside
    `step()`) is equivalent to simply `prev_target_norm is not None` at the
    `ttt_step` level, since `self.prev_out` is None only before the very
    first step of a trajectory -- exactly the same window (0) where
    `prev_target` is also None (checked directly against `docs/track2.md`'s
    own contract: `prev_target_norm` is `None` only on a trajectory's first
    step). `self.ctr_out is not None` (checked pre- or post-reassignment,
    same truth value both ways, per the established invariant "`ctr_out`
    non-None from window 0 onward", already relied on by
    `cuda_graph_bounds.py`'s own module docstring).

Net effect: BOTH graphed windows (previously window >= 2 for update,
window >= 1 for apply -- v1's `apply` actually started one window earlier
than documented there, since `seed()` flips `_live` mid-`step()` at window 1
and `ttt_step`'s SAME-step `bounds()` call already saw it; harmless, since
the math is identical regardless of "freshly seeded" vs "been live a
while", but confusingly worded in the v1 manifest) are now UNIFIED: window 0
and window 1 both take the fully eager path (matching P3 byte-for-byte);
window >= 2 takes ONE merged graph replay. `self._live` semantics unchanged
(one host flag, cleared by `reset()`, same rationale as v1).

Per-step host-side call count: v1 = 6 copy_in + 2 replay + 2 clone_out = 10.
v2 = 6 copy_in + 1 replay + 2 clone_out = 9 -- saves exactly one
`cudaGraphLaunch` (~0.045 ms/step per the profile above). The DOMINANT cost
(`cudaMemcpyAsync`, ~0.8 ms/step) is NOT reduced by this merge: it is
proportional to the NUMBER OF DISTINCT EXTERNAL TENSORS entering the graph
(6: prev_target, ctr_old, mean, var, x, pred_out), all genuinely new
per-step information with no existing graph-owned buffer to alias against
without coupling this wrapper's correctness to another wrapper's own
capture-success state (breaking the "own try/except, independent of every
other graph" contract every wrapper in this file relies on, and which
robustness check (g) -- per-graph + blanket capture-failure injection --
directly tests). Eliminating more of it was assessed and explicitly NOT
attempted in this follow-up pass; see the submission manifest's S9 for the
full writeup and the profiler evidence.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import cond_width_net as _CW

STATIC_MEAN_SHAPE = (1, 20, 32, 64, 3)
STATIC_VAR_SHAPE = (1, 1, 32, 64, 3)
STATIC_STATE_SHAPE = (1, 32, 64)


class GraphedCondWidthWrapper(nn.Module):
    def __init__(self, model, warmup_iters: int = 3):
        super().__init__()
        self.model = model
        self.cw = model.cw
        self.ok = False
        self._live = False
        self._ctr_old = None  # set by step()'s cw block each live step
        # `_ctr_old_valid` is DELIBERATELY a separate flag from `_live`:
        # `_live` flips False->True mid-`step()` at window 1 (via `seed()`,
        # called from the eager branch), but window 1 NEVER sets `_ctr_old`
        # (only the graphed `if should_use()` branch does) -- if
        # `combined_should_use()` used `_live` alone, window 1's OWN
        # same-step `ttt_step` apply dispatch would incorrectly try the
        # graphed path against a stale/unset `_ctr_old`. This flag is set
        # True ONLY in the same branch that sets `_ctr_old`, and explicitly
        # False everywhere else (including the eager branch and `reset()`),
        # so it is synchronized with `_ctr_old` by construction.
        self._ctr_old_valid = False
        self.graph = None
        self.capture_error = None
        self._k_scale = float(model.cfg["ctr_k_scale"])
        self._t2 = hasattr(type(self.cw), "update_both")  # T2 batched cond-width present
        self._rho = float(self.cw.rho)
        self._n_px = int(self.cw.n_px)
        device = model.device
        if not (isinstance(device, str) and device.startswith("cuda")) and not (
            hasattr(device, "type") and device.type == "cuda"
        ):
            return
        try:
            self._capture(warmup_iters)
            self.ok = True
        except Exception as e:  # noqa: BLE001
            self.ok = False
            self.graph = None
            self.capture_error = repr(e)

    # ------------------------------------------------------------ alloc
    def _alloc_buffers(self, device, dtype):
        self._prev_target = torch.zeros(*STATIC_MEAN_SHAPE, device=device, dtype=dtype)
        self._ctr_old_buf = torch.zeros(*STATIC_MEAN_SHAPE, device=device, dtype=dtype)
        self._mean = torch.zeros(*STATIC_MEAN_SHAPE, device=device, dtype=dtype)
        self._var = torch.zeros(*STATIC_VAR_SHAPE, device=device, dtype=dtype)
        self._x = torch.zeros(*STATIC_MEAN_SHAPE, device=device, dtype=dtype)
        self._pred_out = torch.zeros(*STATIC_MEAN_SHAPE, device=device, dtype=dtype)
        self._out_lower = torch.zeros(*STATIC_MEAN_SHAPE, device=device, dtype=dtype)
        self._out_upper = torch.zeros(*STATIC_MEAN_SHAPE, device=device, dtype=dtype)
        # persistent per-channel state, updated in place by THIS SAME graph
        # every replay, read by the apply half right after.
        self._prev_buf = [torch.zeros(*STATIC_STATE_SHAPE, device=device, dtype=dtype)
                          for _ in range(2)]
        self._ema_buf = [torch.zeros(*STATIC_STATE_SHAPE, device=device, dtype=dtype)
                         for _ in range(2)]

    # ------------------------------------------------------- update half
    def _update_body(self):
        m = self.model
        for c in (0, 1):
            pt_phys = self._prev_target[..., c] * m.std_tgt[c] + m.mean_tgt[c]
            ctr_phys = self._ctr_old_buf[..., c] * m.std_tgt[c] + m.mean_tgt[c]
            resid = (pt_phys - ctr_phys).abs()
            scored = _CW._scored(pt_phys, resid.dtype)   # MSK: exact_mask switch
            num = (resid * scored).sum(dim=1)
            den = scored.sum(dim=1)
            numf = _CW._box(num, self._n_px)
            denf = _CW._box(den, self._n_px)
            e_all = (numf / denf.clamp_min(1e-6)).clamp_min(0.0)
            blended = self._rho * self._ema_buf[c] + (1.0 - self._rho) * e_all
            self._ema_buf[c].copy_(blended)
            self._prev_buf[c].copy_(e_all)

    # -------------------------------------------------------- apply half
    def _apply_body(self):
        from submission import _SIGMA_GLOBAL

        m = self.model
        mean, var = self._mean, self._var
        std = var.expand_as(mean).sqrt()
        a = 2.0 * std * m.std_tgt / _SIGMA_GLOBAL
        pos = (a.clamp_min(1e-12).log() - m.log_a[0]) / (m.log_a[1] - m.log_a[0])
        k = m.k_tab[pos.round().long().clamp(0, len(m.k_tab) - 1)]
        half = self._k_scale * k * std

        T = half.shape[1]
        sig_pipe_phys = std[..., :2] * m.std_tgt[:2]
        mults = []
        for c in (0, 1):
            x_phys = self._x[..., c] * m.std_in[c] + m.mean_in[c]
            pred_phys = self._pred_out[..., c] * m.std_tgt[c] + m.mean_tgt[c]
            sp_c = sig_pipe_phys[..., c]
            base = _CW.causal_base_features(x_phys, pred_phys, sp_c)
            e_prev_map = self._prev_buf[c]
            e_ema_map = self._ema_buf[c]
            e_prev = e_prev_map.unsqueeze(1).expand(-1, T, -1, -1)
            e_ema = e_ema_map.unsqueeze(1).expand(-1, T, -1, -1)
            ratio = (e_prev_map / sp_c[:, 0].clamp_min(_CW.EPS)).clamp_min(0.0)
            prevwin = torch.log(e_prev + _CW.EPS)
            ema = torch.log(e_ema + _CW.EPS)
            ratio_log = torch.log(ratio + _CW.EPS).unsqueeze(1).expand(-1, T, -1, -1)
            extra = torch.stack([prevwin, ema, ratio_log], dim=-1)
            feats = torch.cat([base, extra], dim=-1)
            feats = (feats - self.cw.mu) / self.cw.sd
            inp = feats.squeeze(0).permute(0, 3, 1, 2).contiguous()
            net_out = self.cw.net(inp).squeeze(1)
            mult = torch.exp(self.cw.alpha * net_out).clamp(self.cw.clip_lo, self.cw.clip_hi) * self.cw.lam
            mults.append(mult.unsqueeze(0))
        mult2 = torch.stack(mults, dim=-1)
        out_half = half.clone()
        out_half[..., :2] = half[..., :2] * mult2
        self._out_lower.copy_(mean - out_half)
        self._out_upper.copy_(mean + out_half)

    def _forward_body(self):
        # update MUST run first: apply reads the buffers update just wrote
        # (same relative order as the eager reference: step()'s cw update
        # block runs strictly before ttt_step's bounds() call).
        if self._t2:
            # T2 batched cond-width (cond_width_net.py carries the appended
            # batched patch): one shared capture body, stacked (2,H,W) state
            # kept as the two (1,H,W) persistent buffers via cat/copy-back.
            from t2_cw_graph_body import t2_cw_body
            prev2 = torch.cat([self._prev_buf[0], self._prev_buf[1]], 0)
            ema2 = torch.cat([self._ema_buf[0], self._ema_buf[1]], 0)
            t2_cw_body(self.model, self.cw, self._k_scale, self._prev_target, self._ctr_old_buf,
                       self._mean, self._var, self._x, self._pred_out, prev2, ema2,
                       self._out_lower, self._out_upper)
            for c in (0, 1):
                self._prev_buf[c].copy_(prev2[c:c + 1])
                self._ema_buf[c].copy_(ema2[c:c + 1])
            return
        self._update_body()
        self._apply_body()

    # ------------------------------------------------------------ capture
    def _capture(self, warmup_iters: int) -> None:
        m = self.model
        ref = m.std_tgt
        device, dtype = ref.device, ref.dtype
        self._alloc_buffers(device, dtype)

        # deterministic, finite dummy data -- values never matter (every
        # real replay overwrites every fresh-input buffer before use; the
        # persistent state buffers are re-seeded for real once should_use()
        # first fires, see seed()), only shapes/dtypes/addresses do. var and
        # the state maps must stay positive (sqrt/log/div downstream).
        self._prev_target.normal_()
        self._ctr_old_buf.normal_()
        self._mean.normal_()
        self._var.uniform_(1e-4, 1.0)
        self._x.normal_()
        self._pred_out.normal_()
        for c in (0, 1):
            self._prev_buf[c].uniform_(1e-3, 1.0)
            self._ema_buf[c].uniform_(1e-3, 1.0)

        s_stream = torch.cuda.Stream()
        s_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_stream), torch.no_grad():
            for _ in range(warmup_iters):
                self._forward_body()
        torch.cuda.current_stream().wait_stream(s_stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self._forward_body()

    # -------------------------------------------------------------- API
    def should_use(self) -> bool:
        """Call from step()'s cw block, BEFORE this step's own update
        decision. Returns True iff this is at least the 2nd update of the
        trajectory (cw.ema_err[0] already non-None) -- window 1's first-ever
        update always stays eager (EMA-seed branch), and so does its
        SAME-step apply (this graph is not invoked at all until window 2)."""
        return self.ok and self.cw.ema_err[0] is not None

    def seed(self) -> None:
        """One-time (per trajectory) sync from the eager CondWidthState's
        current prev_err/ema_err (as left by window 1's eager update) into
        this wrapper's persistent buffers. Call exactly once, right after
        window 1's eager update."""
        for c in (0, 1):
            self._prev_buf[c].copy_(self.cw.prev_err[c])
            self._ema_buf[c].copy_(self.cw.ema_err[c])
        self._live = True

    def combined_should_use(self) -> bool:
        """Deliberately gated on `_ctr_old_valid`, NOT `_live` -- see the
        `__init__` comment: `_live` alone would misfire at window 1 (the
        exact step `seed()` flips it True), since `_ctr_old` is only ever
        set by the SAME branch that sets `_ctr_old_valid`."""
        return self.ok and self._ctr_old_valid

    def combined_step(self, prev_target: torch.Tensor, ctr_old: torch.Tensor,
                      mean: torch.Tensor, var: torch.Tensor,
                      x: torch.Tensor, pred_out: torch.Tensor):
        self._prev_target.copy_(prev_target)
        self._ctr_old_buf.copy_(ctr_old)
        self._mean.copy_(mean)
        self._var.copy_(var)
        self._x.copy_(x)
        self._pred_out.copy_(pred_out)
        self.graph.replay()
        return self._out_lower.clone(), self._out_upper.clone()
