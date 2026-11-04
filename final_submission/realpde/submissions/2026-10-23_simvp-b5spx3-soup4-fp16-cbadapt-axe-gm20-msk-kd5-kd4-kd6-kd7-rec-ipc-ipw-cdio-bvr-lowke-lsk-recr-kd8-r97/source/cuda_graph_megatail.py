"""P2 (2026-09-27): D2 free-rider bundle ported onto this graph -- the
`_g2_base` energy-target blend (`CONFIG["e_blend_w"]`) and the variance-
preserving mu (`CONFIG["mu_vp"]`), both from `experiments/d2_bundle/
fold0_c/submission.py`. `R_run` (the causal running mean of revealed target
ke, `submission.py`'s `GainMapImprovedModel.R_run`) is a NEW fresh-per-step
INPUT to this graph -- like `x`/`mask`/`fl_f`, copied in via `.copy_()` each
`forward()` call, never persistent graph-internal state -- because its own
update/seeding (`_update_R`, window-0 seed) stays entirely EAGER in
`step()`, unconditionally on `use_mega`, so by the time this graph ever
replays R_run already holds the value the eager reference would have used.
No new counter/EMA-scalar machinery needed for it (contrast `self.n`/
`self.n_c`/`self.n_b`/`self.n_e`, which ARE read INSIDE this capture and so
DO need the host-written-scalar-tensor treatment below). `e_blend_w`/`mu_vp`
are plain Python config values, fixed for the life of the process -- same
capture-time specialization pattern already used throughout this file (e.g.
"ke=None for this shipped config" below): a `if bool(cfg[...])` branch
resolved ONCE when `_forward_body` is traced during `_capture()`, not
re-evaluated per replay.

c12t_launch (launch-cost pass, phase 5): a FIFTH CUDA graph that MERGES the
PRE-stage running-statistics update (`cuda_graph_pre.py`, phase 4),
the energy-gain/ke_net_flow computation (`_update_energy` + `_g2_base` +
`_gamma_ext`, previously eager glue around `cuda_graph_keflow.py`, phase 3),
and the TAIL stage (`cuda_graph_tail.py`, phase 2) into ONE `torch.cuda.graph`
capture and ONE `.replay()` call per firing step.

## Why merge instead of just graphing what was still eager

`experiments/c12t_launch/logs/stage_profile_base.log` (this task, 20
steady-state steps, `pkgs/base` = phase 4's shipped package) found the
single largest remaining costs were NOT bare eager tensor math but the
GLUE around the three existing graphs:
  - POST_energy_gain (46.55 launches/step): `_update_energy`'s few ops +
    `_g2_base`'s rule arithmetic + `GraphedKeFlowWrapper.forward`'s OWN
    7 copies-in + 1 fill + 1 replay + 3 clones-out + `_gamma_ext`'s
    clamp/sqrt.
  - UNATTRIBUTED (48.25 launches/step, the profiler's residual for
    `model.step()` minus every STAGE:: marker): dominated by
    `GraphedPreWrapper.forward`'s 15 copies-in + ~6 scalar fills + 1 replay
    + 16 clones-out, and `GraphedTailWrapper.forward`'s 17 copies-in + 1
    replay + 7 clones-out -- much of which is PRE's own output (mu, var,
    M0, M1, M2, Saa, Sab, Sbb, S02, S12, S22, the S dict) being CLONED OUT
    by `GraphedPreWrapper` only for `GraphedTailWrapper` to immediately
    COPY it straight back IN, unchanged, a few Python lines later in the
    SAME `step()` call.

Per this task's brief (item a): "can adjacent graphs share static buffers
so that no copy kernel is needed? can output assembly write directly into
the returned tensors?" -- the answer here is stronger than sharing buffers
between two SEPARATELY captured graphs (which still requires each to be
captured against pre-existing, correctly-shaped buffers and leaves TWO
`cudaGraphLaunch` calls): capturing PRE's and TAIL's (and the energy/keflow
glue's) tensor math in ONE Python function under ONE `torch.cuda.graph()`
block makes every intermediate value (`new_mu`, `new_M0`, ..., `gamma`,
...) a plain Python-scope local variable inside the SAME capture -- there
is no buffer to copy into or out of between the three stages at all, only
at the TRUE input/output boundary of the merged region (this graph's own
fresh-per-step inputs, and its final 7 outputs, identical in shape/meaning
to `cuda_graph_tail.py`'s own 7).

## Scope / gating

Fires when `self.ok and self.model.S is not None` -- READ AT THE TOP OF
`step()`, i.e. the SAME window range `cuda_graph_pre.py` already uses
(window >= 2 of every trajectory; window 1 has `self.model.S is None` as of
the top of that step, since `_update_blend`'s SEED branch -- `self.S = cur;
return`, structurally uncapturable -- is what sets `S` non-None for the
first time, part way through window 1's own (eager) PRE-stage). Window 1
and window 0 are UNTOUCHED by this file: `submission.py`'s `step()` falls
back to the existing, independently-verified chain (`self._pre` if present
/ eager PRE, then `self._tail` if present / eager TAIL, `_KeFlow`'s own
`self._graph` if present / eager) for those two windows, exactly as phase
4 shipped it -- this graph is a strict ADDITIVE fast path, never a
replacement for the existing fallback chain, so an individual-graph
capture-failure injection (this task's robustness suite item g) degrades
gracefully to phase 4's own already-verified behaviour.

`_scan_uc`/`_update_uc` (task item c, NOT attempted this task -- see
`notes.md` -- fires on a host-visible schedule, not every step) stay
EAGER, UNCHANGED, called by `submission.py` immediately BEFORE this
graph's `forward()`, exactly as they are called before `cuda_graph_tail.py`
today; `uc`/`uc_eff` are plain fresh-per-step INPUT buffers here, same
role as in `cuda_graph_tail.py`. `_mask` (task item d) is graphed
SEPARATELY (`cuda_graph_mask.py`, new, this task) -- its own file, since
it is needed on window 0 too (unlike everything else here).

## Correctness contract

Identical to phases 1-4 (`cuda_graph_ctxnorm.py`/`cuda_graph_tail.py`/
`cuda_graph_keflow.py`/`cuda_graph_pre.py`): fixed-address static buffers,
`.copy_()` in / `.clone()` out only at the TRUE boundary of the captured
region, no allocation inside the replayed graph, try/except at
construction only, never in the timed path.

Every tensor formula below is copy-pasted VERBATIM from the four files
above (not re-derived), specialised to the same compile-time-fixed CONFIG
branches those files already established and independently re-verified by
reading `submission.py` directly for this task (`pool_t=True`,
`mu_alpha_min=None`, `ctr_lead=False`, `ctr_basis2=True`, `blend="region"`
-> `_coefficients`'s "pixel" sub-branch dead, `ke=None` -> `_g2_base`'s
`ke_net` branch dead, `er_online=False` -> `_gamma_ext`'s candidate-scaling
branch dead, `ke_flow` is NOT None -> its branch IS live, `mean_mode=None`,
`ke_steps=0`). The three EMA counters this graph reads (`self.n`,
`self.n_c`, `self.n_b`, PLUS a fourth new to this file, `self.n_e`) get
host-written scalar tensors exactly like `cuda_graph_pre.py`'s own -- SIMPLE
`x + a*d` forms (mu, M0, M1, M2, the S dict, AND `_update_energy`'s
e_true/e_pred, which is the SAME simple shape) use an `a` tensor that is a
float32 0-dim device scalar; COMPOUND `(1-a)*(x+a*y)` forms (var, Saa, Sab,
Sbb, S02, S12, S22) use a float64 0-dim device scalar, with NO explicit
`.double()` cast anywhere on the big tensors -- `experiments/c10t_launch/
scripts/debug_pre_ema.py`'s verified recipe, reused unchanged (this file
introduces no NEW counter-driven arithmetic shape beyond what phase 4
already validated: `_update_energy`'s `self.e_true = self.e_true + a*(e_true
- self.e_true)` is the exact same SIMPLE `x + a*d` shape already in that
script's tested table, 0/305 mismatches for an f32-tensor `a`).

`ke_net_flow.features`/`apply_correction`/`rule_log_energy` and
`self.net.forward` (the small KeFlowNet) are called UNMODIFIED on local
tensors built the same way `cuda_graph_keflow.py`'s `_forward_body` already
does (including its own `leads` `index_select`-not-fancy-indexing fix), so
there is no new hand-transcription risk for the net-forward/feature-build
math itself, only for the plumbing connecting it to this graph's own
inputs/outputs.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

STATIC_SHAPE = (1, 20, 32, 64, 3)
STATIC_E_SHAPE = (1, 1, 32, 64, 3)


def _ke(x: torch.Tensor) -> torch.Tensor:
    return x.var(dim=1, keepdim=True, unbiased=False)


def _kd_coefficients(s, valid, mask, st, lam=None):
    """KD: `GainMapImprovedModel._coefficients` ("region" branch, S present)
    with the blend statistics passed as the already-stacked (5,1,T,H,W,C)
    tensor `st` -- identical values to `torch.stack([S[k] for k in keys])`,
    without the copy. Same op sequence otherwise (verbatim)."""
    if s.cfg["blend"] != "region":
        raise RuntimeError("KD mega: only blend='region' is specialised")
    # AXE: lam may be a (K_br,1,1,1,1) tensor of ridge candidates (br_adapt); None = the CONFIG float
    lam = float(s.cfg["blend_ridge"]) if lam is None else lam
    keep = (~mask).view(1, 1, *mask.shape, 1)
    v = (valid & keep).to(st.dtype)
    i = (~valid & keep).to(st.dtype)
    pooled = {}
    w2 = torch.stack([v, i], 0)                                   # (2,1,T,H,W,3)
    allsums = (st.unsqueeze(1) * w2.unsqueeze(0)).sum(dim=(4, 5), keepdim=True)
    for j, (region, with_b) in enumerate((("v", True), ("i", False))):
        pooled[region] = s._solve(*[allsums[k, j] for k in range(5)], lam=lam, with_b=with_b)
    a = torch.where(valid, pooled["v"][0], pooled["i"][0])
    b = torch.where(valid, pooled["v"][1], torch.zeros_like(pooled["v"][1]))
    a[..., 2] = 1.0
    b[..., 2] = 0.0
    return a, b


class _Shadow:
    """Same role as `cuda_graph_tail.py`'s `_Shadow`: an attribute bag standing
    in for a `GainMapImprovedModel` instance so the original bound methods
    (`_advect`, `_coefficients`, `_shape`, `_ctr_gain2`) run byte-for-byte,
    reading this graph's local variables instead of the live model's own."""


class GraphedMegaWrapper(nn.Module):
    def __init__(self, model, warmup_iters: int = 3):
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
        kf = m.ke_flow
        f = lambda shp, dt=dtype: torch.zeros(*shp, device=device, dtype=dt)

        # ---- fresh-per-step inputs ----
        self._x = f((B, T, H, W, C))
        self._mask_in = torch.zeros(H, W, device=device, dtype=torch.bool)
        self._prev_mask = torch.zeros(H, W, device=device, dtype=torch.bool)
        # KD: fl_f is slot 0 of a (2,T,H,W,C) pair whose slot 1 receives the
        # demeaned bias mu_p, so both temporal energies come from ONE var.
        self._fm = f((2, T, H, W, C))
        self._fl_f = self._fm[0:1]
        self._mf = f((B, 1, H, W, C))
        self._err = f((B, T, H, W, C))
        self._y = f((B, T, H, W, C))                 # prev_target
        self._blend_y = f((B, T, H, W, C))
        self._prev_fl = f((B, T, H, W, C))
        self._prev_adv = f((B, T, H, W, C))
        self._prev_const = f((B, 1, H, W, C))
        self._prev_basis = f((B, T, H, W, C))
        self._prev_basis2 = f((B, T, H, W, C))
        self._uc = torch.zeros((), device=device, dtype=dtype)
        self._uc_eff = torch.zeros((), device=device, dtype=dtype)
        # D2/H1 bundle: causal running mean of revealed target ke, a fresh-
        # per-step INPUT (see module docstring) -- same shape as kx/e_f.
        self._R_run = f(STATIC_E_SHAPE)

        # ---- persistent state, copied in from the live model each firing
        # call (same role/cost as cuda_graph_pre.py's own _sync_state_in --
        # NOT eliminated, since window 1's EAGER PRE-stage is what seeds
        # these for the first firing call of every trajectory), overwritten
        # in place by this graph's own replay from then on ----
        self._mu = f((B, T, H, W, C))
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
        # KD: the blend statistics stored STACKED (5,B,T,H,W,C); `_S` keeps the
        # per-key dict interface as views of it.
        self._S5 = f((5, B, T, H, W, C))
        self._S = {k: self._S5[i] for i, k in enumerate(("ff", "aa", "fa", "fy", "ay"))}
        # KD scratch (shared by this wrapper's graph and the whole-step graph,
        # which never run concurrently): the 5 blend products and the 7
        # second-moment products, each reduced in ONE launch.
        self._C5 = f((5, B, T, H, W, C))
        self._P7 = f((7, B, T, H, W, C))
        self._e_true = f(STATIC_E_SHAPE)
        self._e_pred = f(STATIC_E_SHAPE)
        ex_shape = (STATIC_E_SHAPE[0], STATIC_E_SHAPE[2], STATIC_E_SHAPE[3], 2)
        self._ex_prev = f(ex_shape)
        self._ef_prev = f(ex_shape)

        # ---- ke_net_flow plumbing (mirrors cuda_graph_keflow.py) ----
        self._leads_idx = torch.as_tensor(list(kf.cfg["leads"]), device=device, dtype=torch.long)
        self._wfrac = f((1,))

        # ---- host-written EMA scalar coefficients (see module docstring) ----
        self._a_err_f32 = torch.zeros((), device=device, dtype=torch.float32)
        self._a_err_f64 = torch.zeros((), device=device, dtype=torch.float64)
        self._a_ctr_f32 = torch.zeros((), device=device, dtype=torch.float32)
        self._a_ctr_f64 = torch.zeros((), device=device, dtype=torch.float64)
        self._a_bld_f32 = torch.zeros((), device=device, dtype=torch.float32)
        self._a_e_f32 = torch.zeros((), device=device, dtype=torch.float32)

        # ---- outputs (identical set to cuda_graph_tail.py's own 7) ----
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
        s.uc, s.uc_eff = self._uc, self._uc_eff
        s.adv_w = None       # dead for this shipped config (uc_guard=None)
        s.n_b = 0             # only read by _coefficients' "pixel" branch, dead (blend="region")
        from submission import GainMapImprovedModel  # local import: avoid a
        # module-level circular import (submission.py imports THIS file)
        s._solve = GainMapImprovedModel._solve
        s._smooth = GainMapImprovedModel._smooth
        self._GainMapImprovedModel = GainMapImprovedModel

        self._gain_power = float(m.cfg["gain_power"])
        self._gain_max = float(m.cfg["gain_max"])
        self._ctr_w = float(m.cfg["ctr_w"])
        self._k_in = int(kf.cfg["k_in"])
        self._kf_scale = float(kf.scale)
        self._kf_net = kf.net
        # D2/H1 bundle config, cached as plain Python values at construction
        # time (fixed for the process lifetime) -- see module docstring.
        _w = m.cfg.get("e_blend_w")
        self._e_blend_w = None if _w is None else float(_w)
        self._mu_vp = bool(m.cfg.get("mu_vp", False))

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
        self._e_true.copy_(m.e_true)
        self._e_pred.copy_(m.e_pred)
        self._ex_prev.copy_(m.ke_flow.ex_prev)
        self._ef_prev.copy_(m.ke_flow.ef_prev)
        # NOTE: uc/uc_eff are NOT re-synced here -- forward() already copies
        # them in explicitly from its own (uc, uc_eff) arguments, as ordinary
        # fresh-per-step inputs (same role cuda_graph_tail.py gives them),
        # not as pre-stage "persistent state read from the live model".

    def _forward_body(self):
        self._body(self)

    def _body(self, b):
        """KD (kernel diet): the captured PRE + energy + TAIL math, reading and
        writing every buffer through `b` -- `b is self` for this wrapper's own
        graph / eager fallback (same buffers as before), or the whole-step
        wrapper's I/O namespace (`cuda_graph_wholestep.py`), whose attributes
        alias ITS persistent buffers so no copy-in / copy-out is needed.
        Numerics are the same op sequence as before; bit-exact changes only:
          * persistent state updated IN PLACE with `out=` (same kernel, same
            values; was: compute into a temporary, then `.copy_()` it back);
          * `1.0 - a_ctr_f64` computed once instead of six times;
          * the final outputs written with `out=` into `b._out_*`;
          * `ke_net_flow.features_kd` (fused twin of `features`)."""
        GainMapImprovedModel = self._GainMapImprovedModel
        s = b.shadow
        x, mask, prev_mask = b._x, b._mask_in, b._prev_mask
        fl_f, mf = b._fl_f, b._mf
        err, y, blend_y = b._err, b._y, b._blend_y
        prev_fl, prev_adv = b._prev_fl, b._prev_adv
        prev_const, prev_basis, prev_basis2 = b._prev_const, b._prev_basis, b._prev_basis2

        # ================= PRE half (verbatim from cuda_graph_pre.py) =================
        # KD: the 7 second-moment products are written into one stacked scratch
        # (`out=`, same kernels) and averaged over T by ONE reduction (was 7).
        P = self._P7
        d = err - b._mu
        new_mu = torch.add(b._mu, b._a_err_f32 * d, out=b._mu)
        torch.mul(d, d, out=P[0])

        a0 = y - prev_const
        b0 = prev_basis
        c0 = prev_basis2
        da = a0 - b._M0
        db = b0 - b._M1
        dc = c0 - b._M2
        new_M0 = torch.add(b._M0, b._a_ctr_f32 * da, out=b._M0)
        new_M1 = torch.add(b._M1, b._a_ctr_f32 * db, out=b._M1)
        new_M2 = torch.add(b._M2, b._a_ctr_f32 * dc, out=b._M2)
        torch.mul(da, da, out=P[1])
        torch.mul(da, db, out=P[2])
        torch.mul(db, db, out=P[3])
        torch.mul(da, dc, out=P[4])
        torch.mul(db, dc, out=P[5])
        torch.mul(dc, dc, out=P[6])
        Pm = P.mean(dim=2, keepdim=True)                     # (7,B,1,H,W,C)

        new_var = torch.mul(1.0 - b._a_err_f64, b._var + b._a_err_f64 * Pm[0], out=b._var)
        a64c = b._a_ctr_f64
        om = 1.0 - a64c
        new_Saa = torch.mul(om, b._Saa + a64c * Pm[1], out=b._Saa)
        new_Sab = torch.mul(om, b._Sab + a64c * Pm[2], out=b._Sab)
        new_Sbb = torch.mul(om, b._Sbb + a64c * Pm[3], out=b._Sbb)
        new_S02 = torch.mul(om, b._S02 + a64c * Pm[4], out=b._S02)
        new_S12 = torch.mul(om, b._S12 + a64c * Pm[5], out=b._S12)
        new_S22 = torch.mul(om, b._S22 + a64c * Pm[6], out=b._S22)

        # blend statistics, stacked: 5 products into scratch, ONE keep-mask
        # multiply, ONE EMA update of the (5,...) stack in place
        keep_prev = (~prev_mask).view(1, 1, *prev_mask.shape, 1).to(prev_fl.dtype)
        Cs = self._C5
        torch.mul(prev_fl, prev_fl, out=Cs[0])
        torch.mul(prev_adv, prev_adv, out=Cs[1])
        torch.mul(prev_fl, prev_adv, out=Cs[2])
        torch.mul(prev_fl, blend_y, out=Cs[3])
        torch.mul(prev_adv, blend_y, out=Cs[4])
        Cs.mul_(keep_prev)
        S5 = b._S5
        torch.add(S5, b._a_bld_f32 * (Cs - S5), out=S5)
        new_S = b._S

        # ================= energy / ke_net_flow half =================
        # (verbatim from _update_energy + _g2_base + _gamma_ext, specialised
        # to ke=None / er_online=False / ke_flow is not None)
        # KD: kx may come precomputed (the whole-step graph takes var over the
        # (x, prev_target) pair in one launch); e_f and the variance of the
        # demeaned bias (moved up from the TAIL: it only depends on new_mu)
        # share one var over the (fl_f, mu_p) pair.
        kx = getattr(b, "_kx", None)
        if kx is None:
            kx = _ke(x)
        if self._mu_vp:
            mu_p = torch.sub(new_mu, new_mu.mean(dim=1, keepdim=True), out=b._fm[1:2])
            v2 = b._fm.var(dim=1, keepdim=True, unbiased=False)
            e_f, vM = v2[0:1], v2[1:2]
        else:
            e_f = _ke(fl_f)
        d_et = kx - b._e_true
        new_e_true = torch.add(b._e_true, b._a_e_f32 * d_et, out=b._e_true)
        d_ep = e_f - b._e_pred
        new_e_pred = torch.add(b._e_pred, b._a_e_f32 * d_ep, out=b._e_pred)

        g2 = ((new_e_true + 1e-8) / (new_e_pred + 1e-8)).pow(self._gain_power)

        ex, ef = kx[:, 0, ..., :2], e_f[:, 0, ..., :2]
        xw, ffl = x[..., :2], fl_f[..., :2]
        rx, rf = new_e_true[:, 0, ..., :2], new_e_pred[:, 0, ..., :2]
        # KD: both 10-frame halves of a window in ONE var (view as (1,2,10,...))
        Hh, Wh, Ch = x.shape[2], x.shape[3], x.shape[4]
        xh = x.view(1, 2, 10, Hh, Wh, Ch)[..., :2].var(dim=2, unbiased=False)
        fh = fl_f.view(1, 2, 10, Hh, Wh, Ch)[..., :2].var(dim=2, unbiased=False)
        halves = (xh[:, 0], xh[:, 1], fh[:, 0], fh[:, 1])
        import ke_net_flow  # local import: avoid a module-level import-order issue
        feat = ke_net_flow.features_kd(
            xw, ffl, ex, ef, rx, rf, mask[None], b._wfrac,
            k_in=self._k_in, leads=self._leads_idx, summary=True, halves=halves,
            ex_prev=b._ex_prev, ef_prev=b._ef_prev)
        e_new = ke_net_flow.apply_correction(
            self._kf_net(feat), ke_net_flow.rule_log_energy(ef, rx, rf), mask[None], self._kf_scale)
        b._ex_prev.copy_(ex)
        b._ef_prev.copy_(ef)
        # KD: only channels :2 of the energy target are ever read (g_flow), and
        # there e_tgt == e_new, so the target is formed on (1,1,H,W,2) directly
        # (was: clone e_f, overwrite [..., :2], blend all 3 channels, slice).
        e_t2 = e_new.unsqueeze(1)
        eb = getattr(b, "_eb_ad", None)   # AX (ax_adapt.py): K blend candidates along dim 0; None = off
        br = getattr(b, "_br_ad", None)   # AXE (axe_adapt.py): + K_br ridge rows (factorised); None = off
        if eb is not None:
            e_t2 = eb.wv * e_t2 + eb.omv * b._R_run[..., :2]                  # (K,1,H,W,2)
            if br is not None:
                e_t2 = e_t2.index_select(0, b._br_eidx)                        # (Kc,1,H,W,2)
        elif self._e_blend_w is not None:
            # D2/H1 energy-target blend: e_tgt = w*Khat + (1-w)*R_run.
            e_t2 = self._e_blend_w * e_t2 + (1.0 - self._e_blend_w) * b._R_run[..., :2]
        g_flow = (e_t2 + 1e-10) / (e_f[..., :2] + 1e-10)
        if eb is not None:
            g2 = torch.cat([g_flow, g2[..., 2:].expand(eb.K if br is None else br.Kc, -1, -1, -1, -1)], dim=-1)
        else:
            g2 = torch.cat([g_flow, g2[..., 2:]], dim=-1)
        # self.ke is None for this shipped config -> _g2_base returns g2 here

        g_max = self._gain_max
        gamma = g2.clamp(1.0 / g_max ** 2, g_max ** 2).sqrt()

        # ================= TAIL half (verbatim from cuda_graph_tail.py) =================
        s.S = new_S
        s.M0, s.M1, s.M2 = new_M0, new_M1, new_M2
        s.Saa, s.Sab, s.Sbb = new_Saa, new_Sab, new_Sbb
        s.S02, s.S12, s.S22 = new_S02, new_S12, new_S22

        T = x.shape[1]
        obstacle = s.shift.last_obstacle(mask)
        mx_ = x.mean(dim=1, keepdim=True)
        fl_x = x - mx_
        adv, valid = GainMapImprovedModel._advect(s, fl_x[:, -1], obstacle, mask, T)
        if eb is not None and br is not None:
            # AXE: (a, b) for every ridge candidate at once, then one z row per candidate row (factorised)
            a_, b_ = _kd_coefficients(s, valid, mask, b._S5, lam=br.lamv)
            z = (a_ * fl_f + b_ * adv).index_select(0, b._br_zidx)              # (Kc,T,H,W,C)
        else:
            a_, b_ = _kd_coefficients(s, valid, mask, b._S5)
            z = a_ * fl_f + b_ * adv

        mz = z.mean(dim=1, keepdim=True)
        zc = zc_ls = z - mz
        if eb is not None and br is not None:
            constK = mf + mz                                                  # (Kc,1,H,W,C)
        else:
            const = torch.add(mf, mz, out=b._out_const)
        e_z = _ke(zc)
        c_zf = (zc * fl_f).mean(dim=1, keepdim=True)
        e_tgt2 = gamma * gamma * e_f
        fluc = GainMapImprovedModel._shape(s, zc, fl_f, e_z, e_f, c_zf, e_tgt2)
        if self._mu_vp:
            # D2/H1 variance-preserving mu (verbatim from step()'s eager branch).
            e_tgt_ach = _ke(fluc)        # mu_p, vM: computed with e_f above
            cFM = (fluc * mu_p).mean(dim=1, keepdim=True)
            disc = (cFM * cFM - e_tgt_ach * vM + e_tgt_ach * e_tgt_ach).clamp_min(0.0)
            c = ((-cFM + disc.sqrt()) / e_tgt_ach.clamp_min(1e-12)).clamp_min(0.0)
            fluc = c * fluc
            if eb is not None and br is not None:
                # AXE: keep every row's output for the next step's scores; emit row k_eb = (b*, lambda*)
                outK = torch.add(constK + fluc, new_mu, out=b._eb_outK)
                out = torch.index_select(outK, 0, b._eb_k, out=b._out_out)
                fluc = fluc.index_select(0, b._eb_k)
                zc_ls = zc.index_select(0, b._eb_k)
                const = torch.index_select(constK, 0, b._eb_k, out=b._out_const)
            elif eb is not None:
                # AX: keep every candidate's output for the next step's score; emit the selected one
                outK = torch.add(const + fluc, new_mu, out=b._eb_outK)
                out = torch.index_select(outK, 0, b._eb_k, out=b._out_out)
                fluc = fluc.index_select(0, b._eb_k)
            else:
                out = torch.add(const + fluc, new_mu, out=b._out_out)
        else:
            if eb is not None:
                raise RuntimeError("AX eb_adapt requires mu_vp")
            out = const + fluc
            out = torch.add(out, new_mu, out=b._out_out)

        w = self._ctr_w
        b._out_adv.copy_(adv)
        basis = torch.add((1.0 - w) * fluc, w * zc_ls, out=b._out_basis)
        b2 = torch.sub(adv, adv.mean(dim=1, keepdim=True), out=b._out_b2)
        g1, g2c = GainMapImprovedModel._ctr_gain2(s)
        ctr = const + g1 * basis + g2c * b2
        torch.sub(ctr + new_M0 - g1 * new_M1, g2c * new_M2, out=b._out_ctr_out)
        torch.clamp_min(new_Saa - 2.0 * g1 * new_Sab - 2.0 * g2c * new_S02
                        + g1 * g1 * new_Sbb + 2.0 * g1 * g2c * new_S12
                        + g2c * g2c * new_S22, 1e-12, out=b._out_ctr_var)

    def _capture(self, warmup_iters: int) -> None:
        m = self.model
        ref = m.std_tgt
        device, dtype = ref.device, ref.dtype
        if (m.S is None or m.mu is None or m.var is None or m.M0 is None or m.M2 is None
                or m.prev_const is None or m.prev_basis is None or m.prev_basis2 is None
                or m.e_true is None or m.e_pred is None or m.ke_flow is None
                or m.ke_flow.ex_prev is None or m.ke_flow.ef_prev is None
                or m.R_run is None):
            raise RuntimeError("GraphedMegaWrapper: model has no post-window-1 state yet")
        c = m.ke_flow.cfg
        if not c.get("summary"):
            raise RuntimeError("GraphedMegaWrapper: cfg['summary'] must be True")
        if c.get("lead_e") or c.get("phase"):
            raise RuntimeError("GraphedMegaWrapper: lead_e/phase must be falsy")
        if m.ke is not None:
            raise RuntimeError("GraphedMegaWrapper: ke_net must be off (ke is None) for this scope")
        if m.cfg.get("er_online"):
            raise RuntimeError("GraphedMegaWrapper: er_online must be off for this scope")

        self._alloc_buffers(device, dtype)
        for t in (self._err, self._y, self._blend_y, self._prev_fl, self._prev_adv,
                  self._prev_basis, self._prev_basis2, self._x, self._fl_f):
            t.normal_()
        self._prev_const.normal_()
        self._mf.normal_()
        self._mask_in.copy_(torch.rand(self._mask_in.shape, device=device) > 0.5)
        self._prev_mask.copy_(torch.rand(self._prev_mask.shape, device=device) > 0.5)
        self._R_run.uniform_(0.01, 1.0)  # D2/H1 bundle: dummy warm-up value, shape/dtype only
        self._a_err_f32.fill_(0.1)
        self._a_err_f64.fill_(0.1)
        self._a_ctr_f32.fill_(0.1)
        self._a_ctr_f64.fill_(0.1)
        self._a_bld_f32.fill_(0.1)
        self._a_e_f32.fill_(0.1)
        self._wfrac.fill_(0.5)
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
        would run -- the SAME gate cuda_graph_pre.py itself uses (window >= 2
        of every trajectory; see module docstring)."""
        return self.ok and self.model.S is not None

    def forward(self, x, mask, prev_mask, fl_f, mf, err, y, blend_y,
                prev_fl, prev_adv, prev_const, prev_basis, prev_basis2, uc, uc_eff,
                R_run):
        m = self.model
        self._x.copy_(x)
        self._mask_in.copy_(mask)
        self._prev_mask.copy_(prev_mask)
        self._fl_f.copy_(fl_f)
        self._mf.copy_(mf)
        self._err.copy_(err)
        self._y.copy_(y)
        self._blend_y.copy_(blend_y)
        self._prev_fl.copy_(prev_fl)
        self._prev_adv.copy_(prev_adv)
        self._prev_const.copy_(prev_const)
        self._prev_basis.copy_(prev_basis)
        self._prev_basis2.copy_(prev_basis2)
        self._uc.copy_(uc)
        self._uc_eff.copy_(uc_eff)
        self._R_run.copy_(R_run)  # D2/H1 bundle: fresh-per-step input, see module docstring
        self._sync_state_in()

        m.n += 1
        m.n_c += 1
        m.n_b += 1
        m.n_e += 1
        a_err = m._a(m.n)
        a_ctr = m._a(m.n_c)
        a_bld = m._a(m.n_b)
        a_e = m._a(m.n_e)
        self._a_err_f32.fill_(a_err)
        self._a_err_f64.fill_(a_err)
        self._a_ctr_f32.fill_(a_ctr)
        self._a_ctr_f64.fill_(a_ctr)
        self._a_bld_f32.fill_(a_bld)
        self._a_e_f32.fill_(a_e)
        self._wfrac.fill_(min(m.ke_flow.n, 20) / 20.0)

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
        m.e_true = self._e_true.clone()
        m.e_pred = self._e_pred.clone()
        m.ke_flow.ex_prev = self._ex_prev.clone()
        m.ke_flow.ef_prev = self._ef_prev.clone()
        m.ke_flow.n = m.ke_flow.n + 1

        return (
            self._out_adv.clone(),
            self._out_const.clone(),
            self._out_basis.clone(),
            self._out_b2.clone(),
            self._out_out.clone(),
            self._out_ctr_out.clone(),
            self._out_ctr_var.clone(),
        )
