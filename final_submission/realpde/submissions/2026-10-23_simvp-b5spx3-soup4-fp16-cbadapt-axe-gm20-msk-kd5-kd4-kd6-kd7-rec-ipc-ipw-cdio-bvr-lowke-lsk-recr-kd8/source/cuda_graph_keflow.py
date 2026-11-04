"""c8t (launch-cost, phase 3): a THIRD CUDA graph capturing `_KeFlow.__call__`
(the ke_net_flow feature build + forward + correction), the single largest
still-eager stage per `experiments/c6t2_launch/stage_profile.py`
(POST_energy_gain, ~174 launches/step, dominated by this call).

Phase 2 (`cuda_graph_tail.py`) explicitly left this stage eager because its
feature builder reads a plain PYTHON INT counter (`_KeFlow.n`, for the
`wfrac = min(n, 20) / 20` feature) that changes every call -- baking it into
a graph at capture time would silently freeze it at the capture-time value.

The fix (per this task's brief, and unlike phase 1's `causal_ms` divide
finding): `wfrac`'s arithmetic (`min(n, 20) / 20.0`) is ALREADY pure host
(Python int/float) arithmetic in the shipped code -- `torch.full((1,), value,
...)` only ever constructs a tensor FROM an already-computed host float, it
never divides a tensor by a python int. So there is no tensor-kernel
divide-vs-scalar-divide hazard here at all (unlike `causal_ms`'s `s1 / n`):
we keep the exact same host computation (`min(self.kf.n, 20) / 20.0`, one
Python line, re-evaluated fresh every call) and write it into a fixed
0-dim-shaped static tensor via `.fill_()` BEFORE each replay, outside the
captured region -- `debug_wfrac_fill.py` verifies `tensor.fill_(v)` and
`torch.full((1,), v, ...)` are bit-identical for all 21 values `v` can take
(`min(n,20)/20` for n = 0..25), i.e. the value that lands in the graph's
input buffer is bit-for-bit what the original inline `torch.full` call would
have produced.

The OTHER piece of state `_KeFlow.__call__` carries across calls,
`ex_prev`/`ef_prev` (last window's `(ex, ef)`, used by `summary=True`'s two
lagged log-ratio features), is NOT baked into the graph as internal
accumulator state (unlike phase 1's `causal_ms` S1/S2, which genuinely
accumulate forever). Instead -- matching `cuda_graph_tail.py`'s own stated
preference for "no persistent graph-internal-only state" -- this graph reads
`ex_prev`/`ef_prev` as ordinary INPUT buffers, synced in from the live
`_KeFlow` object every call (`forward()`, below), and produces this call's
own `(ex, ef)` as OUTPUTS, cloned out and written back onto the live object
after replay -- exactly mirroring the original code's own
`self.ex_prev, self.ef_prev, self.n = ex, ef, self.n + 1` state update, just
moved outside the graph. This sidesteps any trajectory-boundary subtlety:
`reset_ttt_state()` already zeroes `_KeFlow.n` / sets `ex_prev = ef_prev =
None` (unchanged, byte-for-byte -- this file does not touch `_KeFlow.reset`),
and this graph is simply never invoked (see `should_use`) until the live
object's `ex_prev`/`ef_prev` are real tensors again.

Static-config preconditions (asserted at capture time; capture aborts -> ok
= False -> eager fallback if violated): `cfg["summary"] is True`,
`cfg.get("lead_e")` and `cfg.get("phase")` both falsy -- true for the
`ke_net_flow.pt` this package ships (`k_in=6, leads=(0,4,9,14,19),
summary=True, lead_e=False, phase=False`), which is what makes `_KeFlow`'s
own `if c["summary"]: ...` / `if c.get("lead_e") or c.get("phase"): ...`
branches resolvable at construction (`summary` hard-coded True in
`_forward_body`; the `lead_e`/`phase` extra-feature branch dropped entirely
-- it is dead code for this config).

Window 0 of every trajectory (`_KeFlow.n == 0`, `ex_prev is None`) is NOT
graphed -- same "structurally different, vanishing fraction of steps" call
phase 1/2 made for their own window-0 exclusions; `should_use()` gates on
`kf.n >= 1`, equivalent to `ex_prev is not None`.

Correctness contract identical to `cuda_graph_ctxnorm.py` / `cuda_graph_tail.py`:
fixed-address static buffers, `.copy_()` in / `.clone()` out, no allocation
inside the replayed graph, try/except at construction only (never in the
timed path). Unlike the tail wrapper, this file does NOT hand-transcribe any
tensor math: `ke_net_flow.features`, `ke_net_flow.apply_correction`, and
`self.kf.net.forward` are called UNMODIFIED on the static buffers, so there
is no hand-transcription risk for the math itself -- only the wfrac/ex_prev
plumbing above is new.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import ke_net_flow

STATIC_X_SHAPE = (1, 20, 32, 64, 3)
STATIC_E_SHAPE = (1, 1, 32, 64, 3)
STATIC_MASK_SHAPE = (32, 64)


class GraphedKeFlowWrapper(nn.Module):
    def __init__(self, keflow, warmup_iters: int = 3):
        """`keflow`: the live `_KeFlow` instance (`GainMapImprovedModel.ke_flow`),
        already called through at least one real window (`keflow.n >= 1`,
        `keflow.ex_prev`/`ef_prev` not None) so there is something real and
        correctly-shaped to warm up / capture against -- exactly the state
        `get_ttt_model`'s existing 2-window dummy warm-up trajectory already
        produces for `GraphedTailWrapper` (same call site, `_KeFlow.__call__`
        runs unconditionally every `step()`, both dummy windows included)."""
        super().__init__()
        self.kf = keflow
        self.ok = False
        self.graph = None
        self.capture_error = None
        try:
            ref = keflow.net.stem.weight
            device, dtype = ref.device, ref.dtype
            if device.type != "cuda":
                return
            c = keflow.cfg
            if not c.get("summary"):
                raise RuntimeError("GraphedKeFlowWrapper: cfg['summary'] must be True")
            if c.get("lead_e") or c.get("phase"):
                raise RuntimeError("GraphedKeFlowWrapper: lead_e/phase must be falsy")
            if keflow.n < 1 or keflow.ex_prev is None or keflow.ef_prev is None:
                raise RuntimeError("GraphedKeFlowWrapper: keflow has no post-window-0 state yet")
            self._capture(device, dtype, warmup_iters)
            self.ok = True
        except Exception as e:  # noqa: BLE001
            self.ok = False
            self.graph = None
            self.capture_error = repr(e)

    # ------------------------------------------------------------ capture
    def _alloc_buffers(self, device, dtype):
        f = lambda shp: torch.zeros(*shp, device=device, dtype=dtype)
        # precomputed once, outside the graph: `ke_net_flow.features`'s
        # `leads` fancy-index needs to be a pre-existing device LongTensor,
        # not a fresh-per-call python-list-derived one, to be capturable
        # (see ke_net_flow.py's `index_select` comment for the full finding).
        self._leads_idx = torch.as_tensor(list(self.kf.cfg["leads"]), device=device, dtype=torch.long)
        self._x = f(STATIC_X_SHAPE)
        self._fl_f = f(STATIC_X_SHAPE)
        self._e_x = f(STATIC_E_SHAPE)
        self._e_f = f(STATIC_E_SHAPE)
        self._runx = f(STATIC_E_SHAPE)
        self._runf = f(STATIC_E_SHAPE)
        self._mask_in = torch.zeros(STATIC_MASK_SHAPE, device=device, dtype=torch.bool)
        ex_shape = (STATIC_E_SHAPE[0], STATIC_E_SHAPE[2], STATIC_E_SHAPE[3], 2)  # (B,H,W,2)
        self._ex_prev = f(ex_shape)
        self._ef_prev = f(ex_shape)
        self._wfrac = f((1,))
        # outputs
        self._out = f(STATIC_E_SHAPE)
        self._ex_out = f(ex_shape)
        self._ef_out = f(ex_shape)

    def _forward_body(self):
        kf = self.kf
        c = kf.cfg
        x, fl_f, e_x, e_f = self._x, self._fl_f, self._e_x, self._e_f
        runx, runf, mask = self._runx, self._runf, self._mask_in

        ex, ef = e_x[:, 0, ..., :2], e_f[:, 0, ..., :2]
        xw, ffl = x[..., :2], fl_f[..., :2]
        rx, rf = runx[:, 0, ..., :2], runf[:, 0, ..., :2]
        # cfg["summary"] verified True at construction time (see __init__)
        halves = (xw[:, :10].var(1, unbiased=False), xw[:, 10:].var(1, unbiased=False),
                  ffl[:, :10].var(1, unbiased=False), ffl[:, 10:].var(1, unbiased=False))
        feat = ke_net_flow.features(
            xw, ffl, ex, ef, rx, rf, mask[None], self._wfrac,
            k_in=c["k_in"], leads=self._leads_idx, summary=True, halves=halves,
            ex_prev=self._ex_prev, ef_prev=self._ef_prev)
        # cfg.get("lead_e")/cfg.get("phase") verified falsy at construction
        # time -- the extra-feature branch is dead code for this config,
        # dropped here rather than replicated.
        e_new = ke_net_flow.apply_correction(
            kf.net(feat), ke_net_flow.rule_log_energy(ef, rx, rf), mask[None], kf.scale)
        out = e_f.clone()
        out[:, 0, ..., :2] = e_new

        self._out.copy_(out)
        self._ex_out.copy_(ex)
        self._ef_out.copy_(ef)

    def _capture(self, device, dtype, warmup_iters: int) -> None:
        self._alloc_buffers(device, dtype)
        # deterministic, finite dummy data for warm-up/capture -- values never
        # matter (every real replay overwrites every input buffer before
        # use), only shapes/dtypes/addresses do.
        self._x.normal_()
        self._fl_f.normal_()
        self._e_x.uniform_(0.01, 1.0)
        self._e_f.uniform_(0.01, 1.0)
        self._runx.uniform_(0.01, 1.0)
        self._runf.uniform_(0.01, 1.0)
        self._mask_in.copy_(torch.rand(STATIC_MASK_SHAPE, device=device) > 0.5)
        self._ex_prev.uniform_(0.01, 1.0)
        self._ef_prev.uniform_(0.01, 1.0)
        self._wfrac.fill_(0.5)

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
        return self.ok and self.kf.n >= 1 and self.kf.ex_prev is not None

    def forward(self, x, fl_f, e_x, e_f, runx, runf, mask):
        self._x.copy_(x)
        self._fl_f.copy_(fl_f)
        self._e_x.copy_(e_x)
        self._e_f.copy_(e_f)
        self._runx.copy_(runx)
        self._runf.copy_(runf)
        self._mask_in.copy_(mask)
        self._ex_prev.copy_(self.kf.ex_prev)
        self._ef_prev.copy_(self.kf.ef_prev)
        self._wfrac.fill_(min(self.kf.n, 20) / 20.0)
        self.graph.replay()
        out = self._out.clone()
        ex_new = self._ex_out.clone()
        ef_new = self._ef_out.clone()
        self.kf.ex_prev, self.kf.ef_prev, self.kf.n = ex_new, ef_new, self.kf.n + 1
        return out
