"""c10t_launch (launch-cost, phase 4): a THIRD structural CUDA graph, capturing
the PRE-stage running-statistics updates at the TOP of `GainMapImprovedModel.
step` -- `_update_error` (mu/var), `_update_ctr` (M0/M1/M2/Saa/Sab/Sbb/S02/
S12/S22), `_update_blend` (the S dict) -- the largest still-eager block per
`experiments/c10t_launch/logs/stage_profile_base.log`:
PRE_ctr_update 48.00, PRE_blend_update 31.65, PRE_error_bias 9.60
launches/step (sum ~89.25/step, ~41% of the base package's ~218 launches/step).

Correctness contract: identical to `cuda_graph_ctxnorm.py`/`cuda_graph_tail.py`
(fixed-address static buffers, `.copy_()` in / `.clone()` out, no allocation
crossing the graph's input/output boundary, try/except at construction only,
never in the timed path).

## Bit-exactness: the EMA "a" hazard, take 3

`self._a(n) = max(1/(n+prior_n), alpha_min)` is host-side python-float
arithmetic (cheap, stays on the host, never itself graphed) but its RESULT is
multiplied into tensors inside the running averages -- and CONFIG's `n`/`n_c`/
`n_b` counters must become host-written static scalar TENSORS for a graph to
read a fresh value each replay (baking `a` in as a capture-time constant
would silently freeze it at its first value forever). Two counter-driven
shapes appear in this scope:

  * SIMPLE form `x + a*d` (mu, M0, M1, M2, and the blend S dict): verified
    (`experiments/c10t_launch/scripts/debug_pre_ema.py`, 305 tested `n`,
    0 mismatches) bit-exact for `a` as a **float32** 0-dim device tensor vs
    the eager python-float multiply -- reproduces phase 1/2's own finding
    that `tensor * python_float == tensor * float32_tensor` bit-for-bit.
  * COMPOUND form `(1-a)*(x + a*y)` (var, Saa/Sab/Sbb, S02/S12/S22): the SAME
    script found float32 mismatches (2/305 tested `n`, max abs diff 4.8e-7)
    but **zero** mismatches for `a` as a **float64** 0-dim device tensor --
    reproducing (independently, not just citing) `experiments/c6t2_launch/
    notes.md` sec 3's `debug_ema_scalar.py` finding, now verified against the
    EXACT formulas (incl. the extra `mean(dim=1)` reduction `_update_ctr`
    composes it with) this graph actually captures, not just a synthetic
    analogue.

So every counter this graph reads gets TWO static scalar buffers (f32 for the
simple updates, f64 for the compound ones), filled from the SAME host-computed
python float, once per step, OUTSIDE the captured region (the `.fill_()` calls
themselves are plain eager kernel launches -- a handful, versus the ~89
launches/step this graph removes).

## Scope / gating

`_mask` is NOT included (same reasoning as `cuda_graph_tail.py`: cheap,
7.5 launches/step, and -- unlike this graph -- needed on window 0 too; stays
exactly one eager call).

Gated on `self.model.S is not None`, READ AT THE TOP OF `step()`, i.e. BEFORE
this step's own PRE-stage runs -- one window LATER than `cuda_graph_tail.py`'s
own `should_use()` (which is checked AFTER the PRE-stage has possibly just
seeded `S` for the first time). This is deliberate, not an oversight:
`_update_blend`'s FIRST call after `S` is `None` takes a structurally
DIFFERENT branch (`self.S = cur; return` -- no EMA, no `n_b` increment, no
read of a prior `S`), which this graph does not (and structurally cannot,
without host-visible data-dependent control flow) express. Since `S` starts
`None` and is only ever seeded by that first call, "S is not None as of the
START of this step" is exactly "this is not the first post-window-0 step of
the trajectory" -- window 1 (where the seed happens) falls back to the
unmodified eager path (dead simple, zero risk, and -- like window 0 -- a
vanishing fraction of steps on any real stride-20 trajectory); `_update_error`
and `_update_ctr` do NOT have an analogous seed branch for this shipped
config (their own `is None` branches are dead from window 0 onward, since
`_apply_prior` seeds `mu`/`var`/`M0`/`M1`/`Saa`/`Sab`/`Sbb`/`M2`/`S02`/`S12`/
`S22` before window 0 ever reaches here) -- gating all three together on the
more conservative (blend's) condition costs nothing beyond leaving one extra
window/trajectory eager, which was already true of `cuda_graph_tail.py` at
window 0.

Every tensor op below is transcribed 1:1 from `submission.py`'s
`_update_error`/`_update_ctr`/`_update_blend`, specialised to this shipped
CONFIG's compile-time-fixed branches (verified by reading `submission.py`
directly, not assumed): `pool_t=True` (not `"sep"`, so the `prof` update is
dead), `mu_alpha_min=None` (so `am == a` always, the `mu_floor` clamp is
dead), `ctr_lead=False` (`m()` reduces `dim=1`), `ctr_basis2=True` (the
`M2`/`S02`/`S12`/`S22` second-basis branch is always taken), `blend`
region-only S (the `_update_blend` method itself has no config branch: the
"pixel" blend smoothing lives in `_coefficients`, already inside
`cuda_graph_tail.py`, untouched here).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

STATIC_SHAPE = (1, 20, 32, 64, 3)


def _prior_n_alpha_min(cfg: Dict[str, Any]):
    return float(cfg["prior_n"]), float(cfg["alpha_min"])


class GraphedPreWrapper(nn.Module):
    def __init__(self, model, warmup_iters: int = 3):
        super().__init__()
        self.model = model
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
        f = lambda shp, dt=dtype: torch.zeros(*shp, device=device, dtype=dt)
        # ---- inputs (fresh each call) ----
        self._err = f((B, T, H, W, C))
        self._y = f((B, T, H, W, C))            # prev_target, for ctr's a0
        self._blend_y = f((B, T, H, W, C))       # prev_target - prev_mf, for blend
        self._prev_fl = f((B, T, H, W, C))
        self._prev_adv = f((B, T, H, W, C))
        self._prev_mask = torch.zeros(H, W, device=device, dtype=torch.bool)
        self._prev_const = f((B, 1, H, W, C))
        self._prev_basis = f((B, T, H, W, C))
        self._prev_basis2 = f((B, T, H, W, C))
        # ---- state, copied in, overwritten in place, cloned out after replay ----
        self._mu = f((B, T, H, W, C))
        # `self.var` is `_ke(err).clone()` (pool_t=True -> variance already
        # reduced over T at seed time) -- (B, 1, H, W, C), NOT the full-T
        # shape (a first version of this file wrongly allocated (B,T,H,W,C)
        # here, which silently broadcast-copied on `.copy_()` and would have
        # written a wrong-shaped value back to the live model).
        self._var = f((B, 1, H, W, C))
        self._M0 = f((B, T, H, W, C))
        self._M1 = f((B, T, H, W, C))
        self._M2 = f((B, T, H, W, C))
        self._Saa = f((B, 1, H, W, C))
        self._Sab = f((B, 1, H, W, C))
        self._Sbb = f((B, 1, H, W, C))
        self._S02 = f((B, 1, H, W, C))
        self._S12 = f((B, 1, H, W, C))
        self._S22 = f((B, 1, H, W, C))
        self._S = {k: f((B, T, H, W, C)) for k in ("ff", "aa", "fa", "fy", "ay")}
        # ---- host-written scalar coefficients (see module docstring) ----
        self._a_err_f32 = torch.zeros((), device=device, dtype=torch.float32)
        self._a_err_f64 = torch.zeros((), device=device, dtype=torch.float64)
        self._a_ctr_f32 = torch.zeros((), device=device, dtype=torch.float32)
        self._a_ctr_f64 = torch.zeros((), device=device, dtype=torch.float64)
        self._a_bld_f32 = torch.zeros((), device=device, dtype=torch.float32)

    def _sync_state_in(self):
        m = self.model
        self._mu.copy_(m.mu)
        self._var.copy_(m.var)
        self._M0.copy_(m.M0)
        self._M1.copy_(m.M1)
        self._M2.copy_(m.M2)
        self._Saa.copy_(m.Saa)
        self._Sab.copy_(m.Sab)
        self._Sbb.copy_(m.Sbb)
        self._S02.copy_(m.S02)
        self._S12.copy_(m.S12)
        self._S22.copy_(m.S22)
        for k in self._S:
            self._S[k].copy_(m.S[k])

    def _forward_body(self):
        """Transcribed 1:1 from `_update_error`/`_update_ctr`/`_update_blend`
        (see module docstring for the exact CONFIG-branch correspondence)."""
        err, y, blend_y = self._err, self._y, self._blend_y
        prev_fl, prev_adv, prev_mask = self._prev_fl, self._prev_adv, self._prev_mask
        prev_const, prev_basis, prev_basis2 = self._prev_const, self._prev_basis, self._prev_basis2

        # ---- _update_error (pool_t=True, mu_alpha_min=None) ----
        # NOTE: no explicit `.double()` anywhere below. `experiments/
        # c10t_launch/scripts/debug_pre_ema.py`'s verified recipe is: leave
        # every tensor operand at its NATIVE float32 dtype and let a
        # float64 0-DIM tensor scalar (`_a_*_f64`) participate directly --
        # PyTorch's 0-dim "wrapped scalar" promotion then uses the scalar's
        # full double precision for the intermediate multiply/add, exactly
        # like a python float does, while the RESULT stays float32 (no
        # dtype-widening of the big tensors, no final `.to()` needed). An
        # earlier version of this file explicitly cast tensors to `.double()`
        # before combining with the f64 scalar -- that computes the `mean()`
        # reduction itself in float64 (the eager/python-float path never
        # does that), which is a DIFFERENT computation and was caught by the
        # end-to-end equivalence gate (bounds mismatched from window 2
        # onward, ~1e-6). Fixed by removing every `.double()` call below.
        d = err - self._mu
        new_mu = self._mu + self._a_err_f32 * d          # simple form: f32 exact
        d2 = d * d
        new_var = (1.0 - self._a_err_f64) * (self._var + self._a_err_f64 * d2.mean(dim=1, keepdim=True))

        # ---- _update_ctr (ctr_lead=False -> m() means over dim=1, ctr_basis2=True) ----
        a0 = y - prev_const
        b0 = prev_basis
        c0 = prev_basis2
        da = a0 - self._M0
        db = b0 - self._M1
        dc = c0 - self._M2
        new_M0 = self._M0 + self._a_ctr_f32 * da
        new_M1 = self._M1 + self._a_ctr_f32 * db
        new_M2 = self._M2 + self._a_ctr_f32 * dc

        def m(t):
            return t.mean(dim=1, keepdim=True)

        new_Saa = (1.0 - self._a_ctr_f64) * (self._Saa + self._a_ctr_f64 * m(da * da))
        new_Sab = (1.0 - self._a_ctr_f64) * (self._Sab + self._a_ctr_f64 * m(da * db))
        new_Sbb = (1.0 - self._a_ctr_f64) * (self._Sbb + self._a_ctr_f64 * m(db * db))
        new_S02 = (1.0 - self._a_ctr_f64) * (self._S02 + self._a_ctr_f64 * m(da * dc))
        new_S12 = (1.0 - self._a_ctr_f64) * (self._S12 + self._a_ctr_f64 * m(db * dc))
        new_S22 = (1.0 - self._a_ctr_f64) * (self._S22 + self._a_ctr_f64 * m(dc * dc))

        # ---- _update_blend (EMA branch only; seed branch handled eagerly, see should_use) ----
        keep = (~prev_mask).view(1, 1, *prev_mask.shape, 1).to(prev_fl.dtype)
        cur = {
            "ff": prev_fl * prev_fl, "aa": prev_adv * prev_adv, "fa": prev_fl * prev_adv,
            "fy": prev_fl * blend_y, "ay": prev_adv * blend_y,
        }
        cur = {k: v * keep for k, v in cur.items()}
        new_S = {k: self._S[k] + self._a_bld_f32 * (cur[k] - self._S[k]) for k in cur}

        self._mu.copy_(new_mu)
        self._var.copy_(new_var)
        self._M0.copy_(new_M0)
        self._M1.copy_(new_M1)
        self._M2.copy_(new_M2)
        self._Saa.copy_(new_Saa.to(self._Saa.dtype))
        self._Sab.copy_(new_Sab.to(self._Sab.dtype))
        self._Sbb.copy_(new_Sbb.to(self._Sbb.dtype))
        self._S02.copy_(new_S02.to(self._S02.dtype))
        self._S12.copy_(new_S12.to(self._S12.dtype))
        self._S22.copy_(new_S22.to(self._S22.dtype))
        for k in self._S:
            self._S[k].copy_(new_S[k])

    def _capture(self, warmup_iters: int) -> None:
        m = self.model
        ref = m.std_tgt
        device, dtype = ref.device, ref.dtype
        if (m.S is None or m.mu is None or m.var is None or m.M0 is None or m.M2 is None
                or m.prev_const is None or m.prev_basis is None or m.prev_basis2 is None):
            raise RuntimeError("GraphedPreWrapper: model has no post-window-1 state yet")
        self._alloc_buffers(device, dtype)
        # deterministic, finite dummy data (values never matter, shapes/dtypes/addresses do)
        for t in (self._err, self._y, self._blend_y, self._prev_fl, self._prev_adv,
                  self._prev_basis, self._prev_basis2):
            t.normal_()
        self._prev_const.normal_()
        self._prev_mask.copy_(torch.rand(self._prev_mask.shape, device=device) > 0.5)
        self._a_err_f32.fill_(0.1)
        self._a_err_f64.fill_(0.1)
        self._a_ctr_f32.fill_(0.1)
        self._a_ctr_f64.fill_(0.1)
        self._a_bld_f32.fill_(0.1)
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
        """Read at the TOP of step(), i.e. BEFORE this step's own PRE-stage
        would run -- "S is not None as of the end of the PREVIOUS step" is
        exactly "not the first post-window-0 step" (see module docstring)."""
        return self.ok and self.model.S is not None

    def forward(self, err, y, blend_y, prev_fl, prev_adv, prev_mask,
                prev_const, prev_basis, prev_basis2):
        m = self.model
        self._err.copy_(err)
        self._y.copy_(y)
        self._blend_y.copy_(blend_y)
        self._prev_fl.copy_(prev_fl)
        self._prev_adv.copy_(prev_adv)
        self._prev_mask.copy_(prev_mask)
        self._prev_const.copy_(prev_const)
        self._prev_basis.copy_(prev_basis)
        self._prev_basis2.copy_(prev_basis2)
        self._sync_state_in()

        # host-side counters, incremented exactly where the eager methods
        # increment them (once per firing, unconditionally within this gate)
        m.n += 1
        m.n_c += 1
        m.n_b += 1
        prior_n, alpha_min = _prior_n_alpha_min(m.cfg)
        a_err = max(1.0 / (m.n + prior_n), alpha_min)
        a_ctr = max(1.0 / (m.n_c + prior_n), alpha_min)
        a_bld = max(1.0 / (m.n_b + prior_n), alpha_min)
        self._a_err_f32.fill_(a_err)
        self._a_err_f64.fill_(a_err)
        self._a_ctr_f32.fill_(a_ctr)
        self._a_ctr_f64.fill_(a_ctr)
        self._a_bld_f32.fill_(a_bld)

        self.graph.replay()

        m.mu = self._mu.clone()
        m.var = self._var.clone()
        m.M0 = self._M0.clone()
        m.M1 = self._M1.clone()
        m.M2 = self._M2.clone()
        m.Saa = self._Saa.clone()
        m.Sab = self._Sab.clone()
        m.Sbb = self._Sbb.clone()
        m.S02 = self._S02.clone()
        m.S12 = self._S12.clone()
        m.S22 = self._S22.clone()
        m.S = {k: v.clone() for k, v in self._S.items()}
