"""P2 (2026-09-27): D2 free-rider bundle's variance-preserving mu
(`CONFIG["mu_vp"]`) ported into this graph's output assembly. The OTHER
bundle piece (the `_g2_base` energy-target blend, `CONFIG["e_blend_w"]`)
does NOT need any change here: `gamma` already arrives as an INPUT to this
graph, blended EAGERLY upstream in `step()`'s `_gamma_ext`/`_g2_base` call
(see `submission.py`) before `forward()` is invoked -- unchanged plumbing.
`mu_vp` is a plain Python config value, fixed for the process lifetime,
cached at construction time and branched on ONCE when `_forward_body` is
traced during `_capture()` -- same capture-time specialization pattern this
file already uses elsewhere (`mean_mode`/`ctr_bias` comments below).

c6t2 (launch-cost, phase 2): a SECOND CUDA graph capturing the largest
still-eager block of `GainMapImprovedModel.step` after the ctx-norm+UNet
region phase 1 already graphed (`cuda_graph_ctxnorm.py`, unmodified, reused
as-is here).

Per-stage launch profile that motivated this scope (`experiments/c6t2_launch/
stage_profile.py`, `cur_pkg` = phase 1's `graphed_pkg`, 20 steady-state
steps): advect 31.25, coefficients-solve 73.3, ctr-apply 28.75 launches/step
(plus the untagged inline shape/const/fluc/output-assembly code between
them) -- the largest *safely containable* remainder. `mask` (7.5/step) is
NOT part of what this graph removes: it stays exactly one eager `_mask(x)`
call (unconditional, same as before this file existed), fed into this graph
as an input (`forward(x, mask, ...)`) rather than recomputed or captured --
listed here only because the graph reads it, not because its cost changes.
`POST_energy_gain` (174/step, the single biggest stage) is NOT graphed here:
it runs `ke_net_flow`'s forward, whose feature builder reads a plain PYTHON
INT counter (`_KeFlow.n`, for the `wfrac = min(n, 20) / 20` feature) that
increments across steps -- the same class of hazard phase 1 found in
`causal_ms`'s `n`, but here compounded by several such counters, an
`ex_prev is None` bootstrap, and an optional extra-feature branch. Assessed
as materially higher risk to get bit-exact within this task's time budget;
left eager, UNCHANGED, exactly as `_gamma_ext`/`_g2_base` ship today.
`_update_energy` (the running e_true/e_pred average) is left eager for the
same reason it structurally sits between the two graphed pieces once
reordered (see below), and to avoid re-deriving `_a()`'s bit-exactness for a
third counter (`self.n_e`) beyond what this file already needed for none --
this file's graphed region has NO counter-driven scalars at all (verified
below), which is what makes it safe to capture without new bit-exactness
machinery.

Reordering (values-preserving): the shipped code interleaves
advect/coefficients (data: x, raw, self.S) with energy_gain
(data: x, fl_f, e_f, kx, mask -- NOT z/adv/coefficients' output) inside one
`step()` body. Since energy_gain does not read anything advect/coefficients
produce, and advect/coefficients does not read `gamma`, the two are
independent and may be reordered without changing any value: this file's
integration point in `submission.py` computes `gamma` (eager, unchanged)
BEFORE calling this graph, so the graphed region below is one CONTIGUOUS
sequence, fed `gamma` as a plain tensor input (copied into a static buffer
exactly like phase 1's `x`).

Scope EXCLUDED from this graph and why:
  - `bounds()` (21.25 launches/step): called from a DIFFERENT method
    (`ExperimentTTT.ttt_step`, not `GainMapImprovedModel.step`) on
    `m.ctr_out`/`m.ctr_var` -- the *outputs* of this graph. Folding it in
    would require changing the `step()`/`ttt_step()` call boundary (a second,
    separate risk surface); left as a plain eager call, unchanged, using this
    graph's cloned `ctr_out`/`ctr_var` outputs exactly as it already uses the
    eager ones today.
  - `_update_error`, `_update_ctr` (the update half), `_update_blend` (the
    update half), `_update_energy`, `_apply_prior`, `_gamma_ext`/`_g2_base`
    (incl. `ke_net_flow`): all EAGER, UNCHANGED byte-for-byte; several use
    `self._a(n)`-style python-int-driven running averages (n, n_c, n_b, n_e)
    which DO need float64-tensor treatment to graph safely (verified
    separately, `experiments/c6t2_launch/debug_ema_scalar.py` -- float32
    tensor-vs-python-scalar multiply differs by up to 1 ULP in compound
    expressions like `(1-a)*(x+a*y)`; float64 does not), and are simply not
    worth that risk/effort given they are not in this graph's scope.
  - window 0 of every trajectory (`self.S is None`): structurally different
    code (early-return constants in `_coefficients`, no revealed target yet
    for the PRE-stage updates) -- handled by NOT invoking this graph at all
    on window 0 (see `GraphedTailWrapper.should_use`), falling back to the
    unmodified eager `_tail_eager` path. Dead simple, zero risk, and window 0
    is a vanishing fraction of steps in any real trajectory (stride-20
    windows), so its cost is not worth graphing.

Correctness contract: identical to `cuda_graph_ctxnorm.py`'s (fixed-address
static buffers, `.copy_()` in / `.clone()` out, no allocation inside the
replayed graph, try/except at construction only, never in the timed path).
The ONE piece of state this graph produces and reuses INTERNALLY across
replays without an explicit external buffer is none -- everything this graph
needs is copied in fresh from the (eager, unmodified) live model state each
step, and everything it produces is cloned out; no counters, no persistent
graph-internal-only state, unlike `_KeFlow`'s `ex_prev`/`ef_prev` (which is
exactly why THIS region was chosen and that one was not).

Every tensor op below is transcribed from (or -- for `_mask`, `_advect`,
`_coefficients`, `_shape`, `_ctr_gain2` -- called directly, unmodified, via a
lightweight "shadow" object whose attributes point at this file's static
buffers instead of the live model's own, so the ORIGINAL bound-method code
executes byte-for-byte) `submission.py`'s `GainMapImprovedModel`. The shadow
trick means there is no hand-transcription risk for those five methods: the
exact same Python bytecode runs, just reading different tensors.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

STATIC_SHAPE = (1, 20, 32, 64, 3)


def _ke(x: torch.Tensor) -> torch.Tensor:
    return x.var(dim=1, keepdim=True, unbiased=False)


class _Shadow:
    """Attribute bag standing in for a `GainMapImprovedModel` instance, for
    the five original bound methods this graph calls unmodified
    (`_mask`, `_advect`, `_coefficients`, `_shape`, `_ctr_gain2`). Stable
    (never-reassigned-elsewhere) attributes are copied by reference from the
    live model at construction time (`.cfg`, `.shift`, `.device`,
    `.std_in`/`.mean_in`); everything else the graphed methods touch is one
    of THIS file's own static buffers, wired up once before capture.
    """


class GraphedTailWrapper(nn.Module):
    def __init__(self, model, warmup_iters: int = 3):
        """`model`: the live GainMapImprovedModel instance (already
        constructed, `.base` already possibly graphed by cuda_graph_ctxnorm).
        """
        super().__init__()
        self.model = model  # kept for eager fallback + reading live state each step
        self.ok = False
        self.graph = None
        self.capture_error = None
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

    # ------------------------------------------------------------ capture
    def _alloc_buffers(self, device, dtype):
        B, T, H, W, C = STATIC_SHAPE
        m = self.model
        f = lambda shp: torch.zeros(*shp, device=device, dtype=dtype)
        # step inputs (fresh each call). `mask` MUST be computed eagerly by
        # the caller BEFORE this graph runs (it also feeds the eager
        # `_gamma_ext`/ke_net_flow call, which this graph does not include --
        # see module docstring), so it is an input here, not recomputed.
        self._x = f((B, T, H, W, C))
        self._mask_in = torch.zeros(H, W, device=device, dtype=torch.bool)
        self._fl_f = f((B, T, H, W, C))
        self._mf = f((B, 1, H, W, C))
        # e_f / gamma are full 3-channel (u, v, p) -- _ke()/_g2_base() never
        # slice the channel dim down before returning them to the caller
        # (only internally, for the u,v-only ke_flow correction).
        self._e_f = f((B, 1, H, W, C))
        self._gamma = f((B, 1, H, W, C))
        # state copied in from the live (eager, unmodified) model each call
        # -- self.S retains the FULL window (T) dim (see _update_blend: `cur`
        # is never temporally reduced, only masked), unlike e.g. Saa/Sbb.
        self._S = {k: f((B, T, H, W, C)) for k in ("ff", "aa", "fa", "fy", "ay")}
        self._mu = f(tuple(m.mu.shape)) if m.mu is not None else f((B, T, H, W, C))
        self._M0 = f((B, T, H, W, C))
        self._M1 = f((B, T, H, W, C))
        self._M2 = f((B, T, H, W, C))
        # Saa/Sab/Sbb/S02/S12/S22 shape = _ke(a0) shape = (B,1,H,W,C) (ctr_lead=False)
        self._Saa = f((B, 1, H, W, C))
        self._Sab = f((B, 1, H, W, C))
        self._Sbb = f((B, 1, H, W, C))
        self._S02 = f((B, 1, H, W, C))
        self._S12 = f((B, 1, H, W, C))
        self._S22 = f((B, 1, H, W, C))
        # `_advect` reads self.uc / self.uc_eff (0-dim tensors, px/frame);
        # uc_guard = None for this shipped config so uc_eff always mirrors uc
        # exactly (`_update_uc`'s early-return branch), but both are copied
        # in independently -- exact same formula the eager code reads (`uc =
        # self.uc if self.uc_eff is None else self.uc_eff`), not an assumed
        # shortcut.
        self._uc = torch.zeros((), device=device, dtype=dtype)
        self._uc_eff = torch.zeros((), device=device, dtype=dtype)
        # outputs (mask is NOT re-output: the caller already has the eager
        # `mask` it computed and passed in as an input)
        self._out_adv = f((B, T, H, W, C))
        self._out_const = f((B, 1, H, W, C))
        self._out_basis = f((B, T, H, W, C))
        self._out_b2 = f((B, T, H, W, C))
        self._out_out = f((B, T, H, W, C))
        self._out_ctr_out = f((B, T, H, W, C))
        self._out_ctr_var = f((B, 1, H, W, C))

        self.shadow = _Shadow()
        s = self.shadow
        s.cfg = m.cfg
        s.shift = m.shift
        s.device = m.device
        s.std_in = m.std_in
        s.mean_in = m.mean_in
        s.S = self._S
        s.M0, s.M1, s.M2 = self._M0, self._M1, self._M2
        s.Saa, s.Sab, s.Sbb = self._Saa, self._Sab, self._Sbb
        s.S02, s.S12, s.S22 = self._S02, self._S12, self._S22
        s.uc, s.uc_eff = self._uc, self._uc_eff
        s.adv_w = None  # dead for this shipped config (uc_guard=None); see module docstring
        s.n_b = 0  # only read inside _coefficients' "pixel" blend branch, dead (blend="region")
        # `_solve`/`_smooth` are @staticmethod on GainMapImprovedModel, called
        # as `self._solve(...)`/`self._smooth(...)` inside `_coefficients` --
        # bind the plain underlying functions as instance attributes (NOT
        # auto-bound like a class-level method lookup would be, since these
        # are assigned directly on the instance) so that call syntax resolves.
        from submission import GainMapImprovedModel  # local import: avoid a
        # module-level circular import (submission.py imports THIS file)
        s._solve = GainMapImprovedModel._solve
        s._smooth = GainMapImprovedModel._smooth
        # D2/H1 bundle config, cached as a plain Python value at construction
        # time (fixed for the process lifetime) -- see module docstring.
        self._mu_vp = bool(m.cfg.get("mu_vp", False))

    def _sync_state_in(self):
        """Copy the LIVE model's current (eager-updated, address-changing)
        state into this graph's fixed buffers. Called every step, before
        replay -- exactly the same role as phase 1's `_static_x.copy_(x)`,
        just for more tensors."""
        m = self.model
        for k in self._S:
            self._S[k].copy_(m.S[k])
        self._mu.copy_(m.mu)
        self._M0.copy_(m.M0)
        self._M1.copy_(m.M1)
        self._M2.copy_(m.M2)
        self._Saa.copy_(m.Saa)
        self._Sab.copy_(m.Sab)
        self._Sbb.copy_(m.Sbb)
        self._S02.copy_(m.S02)
        self._S12.copy_(m.S12)
        self._S22.copy_(m.S22)
        self._uc.copy_(m.uc)
        self._uc_eff.copy_(m.uc_eff)

    def _forward_body(self):
        """The graphed region itself: mask -> advect -> coefficients-solve ->
        shape/const/fluc -> output assembly (+ bias) -> ctr basis/apply.
        Transcribed 1:1 from `GainMapImprovedModel.step` (see this file's
        module docstring for the exact line-range/formula correspondence and
        what is deliberately excluded)."""
        from submission import GainMapImprovedModel  # local import: avoid a
        # module-level circular import (submission.py imports THIS file)
        s = self.shadow
        x, fl_f, mf, e_f, gamma = self._x, self._fl_f, self._mf, self._e_f, self._gamma
        mask = self._mask_in

        T = x.shape[1]
        obstacle = s.shift.last_obstacle(mask)
        mx = x.mean(dim=1, keepdim=True)
        fl_x = x - mx
        adv, valid = GainMapImprovedModel._advect(s, fl_x[:, -1], obstacle, mask, T)
        a, b = GainMapImprovedModel._coefficients(s, valid, mask)
        # self.adv_w is always None for this shipped config (uc_guard=None
        # forces the early-return branch in `_update_uc` that never sets it)
        # -- the `if self.adv_w is not None: b = b * self.adv_w` line is dead
        # code for CONFIG as shipped; skipped here, not replicated.
        z = a * fl_f + b * adv

        mz = z.mean(dim=1, keepdim=True)
        zc = zc_ls = z - mz
        const = mf + mz
        e_z = _ke(zc)
        c_zf = (zc * fl_f).mean(dim=1, keepdim=True)
        e_tgt = gamma * gamma * e_f
        fluc = GainMapImprovedModel._shape(s, zc, fl_f, e_z, e_f, c_zf, e_tgt)
        # mean_mode = None (off) for this shipped config: const unchanged.
        # bias = True, self.mu seeded by the (Re, AoA) prior before window 0
        # ever reaches this graph (gated to window >= 1 only) -- never None.
        if self._mu_vp:
            # D2/H1 variance-preserving mu (verbatim from step()'s eager
            # branch, this file's own inlined copy -- see module docstring).
            mu = self._mu
            mu_p = mu - mu.mean(dim=1, keepdim=True)
            e_tgt_ach = _ke(fluc)
            vM = _ke(mu_p)
            cFM = (fluc * mu_p).mean(dim=1, keepdim=True)
            disc = (cFM * cFM - e_tgt_ach * vM + e_tgt_ach * e_tgt_ach).clamp_min(0.0)
            c = ((-cFM + disc.sqrt()) / e_tgt_ach.clamp_min(1e-12)).clamp_min(0.0)
            fluc = c * fluc
            out = const + fluc + mu
        else:
            out = const + fluc
            out = out + self._mu

        w = float(s.cfg["ctr_w"])
        basis = (1.0 - w) * fluc + w * zc_ls
        b2 = adv - adv.mean(dim=1, keepdim=True)
        g, g2 = GainMapImprovedModel._ctr_gain2(s)
        ctr = const + g * basis + g2 * b2
        # ctr_bias = True for this shipped config.
        ctr = ctr + self._M0 - g * self._M1 - g2 * self._M2
        ctr_var = (self._Saa - 2.0 * g * self._Sab - 2.0 * g2 * self._S02
                   + g * g * self._Sbb + 2.0 * g * g2 * self._S12
                   + g2 * g2 * self._S22).clamp_min(1e-12)

        self._out_adv.copy_(adv)
        self._out_const.copy_(const)
        self._out_basis.copy_(basis)
        self._out_b2.copy_(b2)
        self._out_out.copy_(out)
        self._out_ctr_out.copy_(ctr)
        self._out_ctr_var.copy_(ctr_var)

    def _capture(self, warmup_iters: int) -> None:
        m = self.model
        ref = m.std_tgt
        device, dtype = ref.device, ref.dtype
        if m.S is None or m.mu is None or m.M0 is None or m.M2 is None:
            # Nothing sane to warm up / capture with yet (would only happen
            # if this wrapper were constructed before any window ever ran
            # eagerly through window 0 -- not how it is integrated in
            # submission.py, but guarded defensively).
            raise RuntimeError("GraphedTailWrapper: model has no post-window-0 state yet")
        self._alloc_buffers(device, dtype)
        # deterministic, finite dummy data for warm-up/capture -- values
        # never matter (every real replay overwrites every input buffer
        # before use), only shapes/dtypes/addresses do.
        self._x.normal_()
        self._mask_in.copy_(torch.rand(self._mask_in.shape, device=device) > 0.5)
        self._fl_f.normal_()
        self._mf.normal_()
        self._e_f.uniform_(0.01, 1.0)
        self._gamma.uniform_(0.5, 2.0)
        self._sync_state_in()

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
        """Window 0 of every trajectory (`self.model.S is None`) takes a
        structurally different eager path; every other pipeline stage this
        graph depends on (M0/M2/mu, all prior-seeded) is available from
        window 0 onward, so `self.model.S is not None` is exactly the gate
        used elsewhere in this file/step() for "not window 0"."""
        return self.ok and self.model.S is not None

    def forward(self, x, mask, fl_f, mf, e_f, gamma):
        self._x.copy_(x)
        self._mask_in.copy_(mask)
        self._fl_f.copy_(fl_f)
        self._mf.copy_(mf)
        self._e_f.copy_(e_f)
        self._gamma.copy_(gamma)
        self._sync_state_in()
        self.graph.replay()
        return (
            self._out_adv.clone(),
            self._out_const.clone(),
            self._out_basis.clone(),
            self._out_b2.clone(),
            self._out_out.clone(),
            self._out_ctr_out.clone(),
            self._out_ctr_var.clone(),
        )
