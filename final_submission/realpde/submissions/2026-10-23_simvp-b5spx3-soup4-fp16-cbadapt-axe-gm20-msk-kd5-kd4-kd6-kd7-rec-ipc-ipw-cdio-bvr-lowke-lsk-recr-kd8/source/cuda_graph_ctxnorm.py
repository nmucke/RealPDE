"""c6t (launch-cost): CUDA-graph-captured replacement for CtxNormWrapper's
own `forward` -- the single largest per-step block in the pipeline (~52% of
`ttt_step` median wall time per `experiments/c5t_speed/notes.md` S1: ctx-norm
accumulate + standardise + UNet forward + un-standardise).

This is a NEW file, only referenced by this copy's `submission.py`
(`experiments/c6t_launch/graphed_pkg/`); `ship_ctxnorm_models.py` itself is
untouched -- every numeric op below is that module's own `causal_ms` /
`standardise_input` / `build_extra_maps` / `unstandardise_output`, called
with the exact same arguments as `CtxNormWrapper.forward` itself uses, so
the graphed path computes bit-for-bit the same function of its inputs; only
*how many kernels the CPU has to launch to get there* changes.

Why this region and not more (see experiments/c6t_launch/notes.md S2 for the
full reasoning): everything captured here runs unconditionally, every step,
at the one fixed static shape (1, 20, 32, 64, 3) the evaluation protocol
guarantees (docs/track2.md: windows are always 20 frames, short trajectory
tails are padded to full windows). Every other pipeline stage this task's
brief also names (`ke_net_flow`, `_scan_uc`, `_advect`, the online energy
candidates) has at least one of: data-dependent Python control flow (`if
self.uc is None or self.n_win < uc_warm ...`), a variable-length loop bound
(`_advect`'s `t = x.shape[1]`-successor range, `torch.arange(1, t+1, ...)`
sized by a *window index*, not the fixed input shape), or host-side list/
scalar bookkeeping (`self.uc_hist`, `self.er_acc`) that would need a redesign
(not just a capture) to make graph-safe inside this task's budget -- so they
are left eager, exactly as shipped.

Correctness contract (state tensors at fixed addresses, no realloc):
  - `_static_S1` / `_static_S2` / `_static_n` are the trajectory-scoped
    causal accumulator, now living in THIS wrapper (not in the wrapped
    `CtxNormWrapper`, whose own `.S1/.S2/.n` are left untouched/unused once
    the graph is active) as pre-allocated buffers, zeroed once at capture
    time and thereafter only ever `.zero_()` (reset_ttt_state) or `.add_()`
    (every step, in place) -- never reassigned, so their addresses (and the
    graph nodes that read/write them) stay valid for the wrapper's whole
    lifetime, across arbitrarily many trajectories.
  - `_static_x` / `_static_out` are the graph's fixed input/output buffers;
    `forward(x)` copies the caller's tensor in (works whether `x` is a CPU
    or CUDA tensor -- `Tensor.copy_` handles the device transfer) and
    returns a **clone** of `_static_out` (the one deliberate non-captured
    allocation per step -- required because `GainMapImprovedModel.step`
    caches `raw`-derived tensors, e.g. `self.prev_out`, across steps, and the
    next `replay()` would otherwise silently overwrite them in place; this
    single small copy is negligible next to the dozens of kernels the graph
    replaces).
  - No allocation happens *inside* the replayed graph itself: every op in
    `_forward_body` either writes into a pre-allocated buffer (`add_`,
    `copy_`) or is a pure function of already-allocated tensors whose own
    intermediate outputs are captured as part of the graph's private pool
    (CUDA graphs own that memory themselves; that pool is allocated once at
    capture, not on each replay -- standard `torch.cuda.graph` behaviour).

Failure handling: any exception during construction (CUDA graphs unsupported
by the driver, capture fails, no CUDA at all) is caught in `__init__`; `.ok`
stays False and `forward`/`reset` transparently fall back to the wrapped
module's own eager path -- never inside the timed loop, only at construction
(matches the task brief: "graceful fallback ... try/except at construction,
never inside the timed path").
"""
from __future__ import annotations

import torch
import torch.nn as nn

import lsk_head as _LSK  # LSK: s buffer (inert when off)

from ship_ctxnorm_models import (
    standardise_input,
    build_extra_maps,
    unstandardise_output,
)

STATIC_SHAPE = (1, 20, 32, 64, 3)   # (B, T, H, W, C) -- fixed by docs/track2.md


def _causal_ms_graphed(s1: torch.Tensor, s2: torch.Tensor, n: torch.Tensor,
                        floor: torch.Tensor):
    """Same numerics as `ship_ctxnorm_models.causal_ms`, but `n` is a 0-dim
    DEVICE TENSOR (required so the graph reads its live, replay-time value
    instead of baking in whatever python int `n` held at capture time) --
    divided via `n.reciprocal()` + multiply rather than `s1 / n`.

    This is NOT a cosmetic rewrite: empirically, `tensor / python_int`
    (what the eager wrapper does) and `tensor / tensor` are DIFFERENT CUDA
    kernels with different rounding -- PyTorch lowers scalar division to a
    host-computed reciprocal times the tensor (`s1 * (1.0/n)`), while
    tensor-by-tensor division computes a true elementwise divide; the two
    differ by up to 1 ULP per element, verified to grow to ~3.7e-4 max abs
    diff after propagating through the UNet's conv/GELU stack (measured
    directly, `experiments/c6t_launch/debug_causal_ms_divide.py`) -- past
    the task's 1e-5 tolerance-tier bound. `tensor.reciprocal()` followed by
    a tensor*tensor MULTIPLY, however, was verified bit-identical to
    `tensor / python_int` for every `n` tested (20 through 19999, and
    non-multiples of 20): both compute `s1 * (1.0/n)` via the same
    reciprocal-then-multiply path, just with the reciprocal taken on-device
    instead of on-host. That equivalence is what makes this graphable AND
    bit-exact at the same time.
    """
    recip = n.reciprocal()
    m = s1 * recip
    var = s2 * recip - m * m
    var = torch.clamp_min(var, floor)
    return m, torch.sqrt(var)


class GraphedCtxNormWrapper(nn.Module):
    def __init__(self, wrapped: nn.Module, warmup_iters: int = 3):
        super().__init__()
        self.wrapped = wrapped  # real params/buffers live here; kept as a submodule
        self.ok = False
        self.graph = None
        self.capture_error = None
        try:
            ref = next(wrapped.parameters())
        except StopIteration:  # pragma: no cover
            return
        if ref.device.type != "cuda":
            return  # CPU-only box: leave `ok=False`, caller uses eager wrapped.forward
        try:
            self._capture(ref.device, ref.dtype, warmup_iters)
            self.ok = True
        except Exception as e:  # noqa: BLE001 - any capture failure -> eager fallback
            self.ok = False
            self.graph = None
            self.capture_error = repr(e)

    # ------------------------------------------------------------ capture
    def _capture(self, device, dtype, warmup_iters: int) -> None:
        w = self.wrapped
        B, T, H, W, C = STATIC_SHAPE
        # plain instance attributes, NOT registered as nn buffers: `.to()` /
        # `.eval()` calls made later by the surrounding pipeline (e.g.
        # `GainMapImprovedModel.__init__`'s `base.to(device).eval()`) must
        # not touch -- let alone reallocate -- these fixed-address tensors.
        self._static_x = torch.zeros(B, T, H, W, C, device=device, dtype=dtype)
        self._static_S1 = torch.zeros(B, H, W, 2, device=device, dtype=dtype)
        self._static_S2 = torch.zeros(B, H, W, 2, device=device, dtype=dtype)
        self._static_n = torch.zeros((), device=device, dtype=dtype)
        self._static_out = torch.zeros(B, T, H, W, C, device=device, dtype=dtype)

        # warm-up on a side stream (torch.cuda.graph's own documented
        # prerequisite): run the exact op sequence a few times so cudnn algo
        # selection / caching allocator settle before capture -- untimed,
        # inside get_ttt_model, exactly like the O3 optimisation this
        # supersedes.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(warmup_iters):
                self._forward_body()
        torch.cuda.current_stream().wait_stream(s)
        self._static_S1.zero_()
        self._static_S2.zero_()
        self._static_n.zero_()

        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self._forward_body()
        # the capture pass itself ran the body for real once more (CUDA
        # graph capture executes the ops, it does not just trace them) --
        # undo that before any real trajectory starts.
        self._static_S1.zero_()
        self._static_S2.zero_()
        self._static_n.zero_()

    def _forward_body(self) -> torch.Tensor:
        """Exactly `CtxNormWrapper.forward`'s own math (see that module's
        docstring), rewritten so the running accumulator lives in the fixed
        buffers above and is updated in place. Starting S1/S2/n at an
        all-zero tensor and always `.add_()`-ing is bit-identical to the
        original `if self.S1 is None: clone else: add` branch (`0.0 + x ==
        x` exactly, IEEE-754 addition with the additive identity never
        rounds) -- removing the branch is what makes this graph-capturable
        with a single, unconditional op sequence."""
        out = self._body(self._static_x)
        self._static_out.copy_(out)
        return self._static_out

    def _body(self, x: torch.Tensor) -> torch.Tensor:
        """KD: `_forward_body`'s math on any (1,T,H,W,3) input tensor, returning
        the (fresh, graph-pool) output -- the whole-step graph reads it
        directly instead of copying x in and the output out."""
        w = self.wrapped
        uv = x[..., :2]
        window_sum1 = uv.sum(dim=1)
        window_sum2 = (uv * uv).sum(dim=1)
        self._static_S1.add_(window_sum1)
        self._static_S2.add_(window_sum2)
        self._static_n.add_(float(STATIC_SHAPE[1]))
        m, s = _causal_ms_graphed(self._static_S1, self._static_S2,
                                   self._static_n, w._ms_floor)
        _LSK.write_s(s)   # LSK
        xs = standardise_input(x, m, s)
        extra = build_extra_maps(m, s, w._m_mean, w._m_std, w._logs_mean, w._logs_std)
        y_s = w.ctx_unet(xs, extra)
        return unstandardise_output(y_s, m, s)

    # -------------------------------------------------------------- API
    def reset(self) -> None:
        """Trajectory boundary: zero the accumulator IN PLACE (never
        reallocate -- the graph's captured nodes point at these exact
        buffers for the wrapper's whole lifetime, across every trajectory
        the stream contains)."""
        if self.ok:
            self._static_S1.zero_()
            self._static_S2.zero_()
            self._static_n.zero_()
        else:
            self.wrapped.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.ok:
            return self.wrapped.forward(x)
        self._static_x.copy_(x)
        self.graph.replay()
        # one small, deliberate, non-captured allocation: the caller
        # (GainMapImprovedModel.step) keeps `raw`-derived tensors alive
        # across steps (e.g. `self.prev_out`), so the next replay must not
        # be able to mutate what a previous step handed out.
        return self._static_out.clone()
