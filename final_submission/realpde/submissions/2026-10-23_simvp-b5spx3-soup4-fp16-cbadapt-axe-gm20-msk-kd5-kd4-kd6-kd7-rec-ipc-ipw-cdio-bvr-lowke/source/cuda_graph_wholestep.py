"""T1 (2026-10-01): ONE CUDA graph for the whole steady-state `ttt_step`.

Diagnosis (`experiments/t1_timing_diag/notes.md`): on the organizers' box
`t_eval ~= 2.26 x CPU_local + 2.49 x GPUbusy_local`; v2 (9 graphs) spends
its CPU on 4.25 graph replays, ~82 cudaMemcpyAsync dispatches, 2 pageable
H2D + 3 pageable D2H copies (each a stream sync) and a mid-step `.item()`.
This wrapper removes all of that for every step from window 2 of a
trajectory onward, leaving per step: ONE pinned H2D copy (x and prev_target
packed), ONE graph replay, ONE pinned D2H copy (pred, lower, upper and the
adapt_loss scalar packed) and ONE stream synchronisation.

## How bit-exactness is kept

Nothing is re-derived. The graph is a COMPOSITION of the already-verified
capture bodies of the existing wrappers, executed on THEIR OWN static
buffers inside one `torch.cuda.graph` capture:
  GraphedMaskWrapper._forward_body      (mask)
  GraphedCtxNormWrapper._forward_body   (ctx-norm + backbone forward; its
                                         causal (S1, S2, n) accumulators are
                                         already persistent, in-place state)
  GraphedScanUcWrapper._forward_body    (convection-speed scan; only in the
                                         "scan" variant of the graph)
  GraphedMegaWrapper._forward_body      (PRE statistics + energy/ke_net_flow
                                         + TAIL; writes its persistent state
                                         buffers in place)
  GraphedCondWidthWrapper._update_body / _apply_body (conditional widths +
                                         bounds; persistent prev/ema buffers)
plus the few eager lines of `step()` between them (err, blend_y, loss,
_update_R, mf/fl_f, obstacle/fl_x, _update_uc's EMA) transcribed verbatim
onto the same tensors, and plain in-graph `.copy_()` glue where v2 copied
between wrappers from the host (these become graph memcpy nodes: zero CPU).

Differences from v2's per-step host work, each argued bit-exact:
  * EMA coefficients (`_a(n)` for n / n_c / n_b / n_e / n_uc, `wfrac` for
    ke_flow.n, 1/n_R for R_run) are computed IN-GRAPH from float64 device
    counters instead of Python ints + `.fill_()`:
    `max(1/(n + prior_n), alpha_min)` in IEEE double is identical on host and
    device (one correctly-rounded division, one max), and `fill_(python_float)`
    into a float32 buffer rounds exactly like `copy_` from a float64 tensor
    (round-to-nearest-even). The resulting f32/f64 scalar buffers are the
    SAME buffers mega's captured body already reads.
  * `adapt_loss = float((err ** 2).mean())` becomes an in-graph reduction
    into a scalar slot of the output pack, read from the pinned host buffer
    after the one D2H copy -- same float32 value, no mid-step sync.
  * Persistent state is seeded ONCE per trajectory (first graphed step,
    window 2) from the live model attributes that window 0/1's unchanged v2
    code produced (`mega._sync_state_in()` + this wrapper's own prev_*/
    R_run/ctr_out/uc buffers + counters) and then lives only in the graph's
    buffers: the model attributes are NOT updated during graphed steps.
    That is safe because the gate is monotonic within a trajectory (once
    live, every later step of the trajectory is graphed; only
    `reset_ttt_state` clears it) and nothing outside this wrapper reads them
    while live.
  * The host-visible convection-scan schedule (`n_win < uc_warm or n_win %
    uc_every == 0`) is kept on the host: two graphs are captured over the
    same buffers, one with the scan + `_update_uc` EMA and one without, and
    `n_win` picks which to replay.

Windows 0 and 1 of every trajectory, and everything else (CPU-only boxes,
any capture failure here or in any of the composed wrappers), run v2's code
byte-for-byte: this file is a strict additive fast path.

The backbone forward is called only through `GraphedCtxNormWrapper.
_forward_body` -> `w.ctx_unet(xs, extra)`: a swapped backbone (T2's work)
drops in unchanged.

KD (2026-09-22, experiments/kd_kernel_diet/, bit-exact kernel diet): the
composed bodies are now called through their `_body(...)` forms with this
wrapper's own buffers as arguments / through an aliasing I/O namespace
(`self._io`) for mega, so the ~45 per-step copy-in / copy-out kernels of the
composition are gone; persistent state is updated in place (`out=`); the
counters are one float64 vector; reductions over the same axis are batched
(var of the input pair, of (fl_f, mu_p), of the window halves; the 7 PRE
second moments; the 5 blend statistics); the T2 tier-2 fusions (keflow
features, cond-width features) are used by the graph bodies. Every change is
value-preserving: same ops on the same values (verified torch.equal on the
full fold0 stream and a 2,000-step mixed-reset synthetic stream). Windows 0/1
and every eager fallback run the unchanged code.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cuda_graph_megatail import _Shadow

STATIC_SHAPE = (1, 20, 32, 64, 3)
STATIC_E_SHAPE = (1, 1, 32, 64, 3)
N_ELEM = 1 * 20 * 32 * 64 * 3


def _ke(x: torch.Tensor) -> torch.Tensor:
    return x.var(dim=1, keepdim=True, unbiased=False)



def _ipc_one_storage(mode, x) -> bool:
    """IPC (experiments/ipc_outproc): return pred / lower / upper as views of ONE clone? (values identical either way)
    mode False -> no; True -> yes; "auto" -> only if `x` (this step's input_norm) is a CPU torch tensor received through
    torch's shared-memory reduction with the "file_descriptor" strategy (multiprocessing Connection / Queue harness)."""
    if mode is True:
        return True
    if mode != "auto":
        return False
    try:
        import torch.multiprocessing as _tmp
        return (isinstance(x, torch.Tensor) and x.device.type == "cpu" and bool(x.is_shared())
                and _tmp.get_sharing_strategy() == "file_descriptor")
    except Exception:  # noqa: BLE001 - any doubt -> the parent's three clones
        return False

class _IO:
    """KD: attribute bag = the I/O namespace mega's `_body(b)` reads/writes."""


class GraphedWholeStepWrapper(nn.Module):
    def __init__(self, exp, warmup_iters: int = 3):
        super().__init__()
        self.exp = exp
        self.model = exp.model
        self.ok = False
        self.live = False
        self.graph_scan = None
        self.graph_noscan = None
        self.capture_error = None
        self.n_replays = 0
        # KD2 (kd2_adapt.py): fused adaptive selectors; CONFIG "kd2_adapt". Any failure with it on -> the
        # parent code path is captured instead (kd2_error keeps the reason).
        self._kd2 = bool(self.model.cfg.get("kd2_adapt", False))
        self.kd2_error = None
        # KD3 (kd3_step.py): bit-exact diet of the backbone / mega bodies; CONFIG "kd3_step". Any failure with it
        # on -> recapture without it (kd3_error keeps the reason); then KD2's own fallback as before.
        self._kd3 = bool(self.model.cfg.get("kd3_step", False))
        # IPC: CONFIG "ipc_one_storage" (False / True / "auto") -> see _ipc_one_storage and ttt_step.
        self._one_storage = self.model.cfg.get("ipc_one_storage", False)
        # HX2: CONFIG "cuda_direct_io" -> when the CALLER's input_norm is already a CUDA tensor, skip the host staging:
        # D2D into the static input rows, return fresh CUDA clones of the static output rows (same device the old path
        # returned on, same values, bit for bit). CPU inputs (local_eval, CPU-tensor / numpy harnesses): unchanged path.
        self.cuda_direct = bool(self.model.cfg.get("cuda_direct_io", False))
        self.n_direct = 0
        self.n_one_storage = 0
        self.kd3_error = None
        # KD6 (kd6_conc.py): fork / join concurrency inside the noscan graph (same kernels on the same values, the
        # backbone-independent ~40 % on a side stream); CONFIG "kd6_concurrency". Built after the parent graphs and
        # self-tested against the parent noscan graph; any failure / mismatch -> the parent graph (kd6_error).
        self._kd6 = bool(self.model.cfg.get("kd6_concurrency", False))
        self.kd6_error = None
        self.kd6_stats = None
        self._kd6_side = None
        # KD7 (kd7_fuse.py): bit-exact jiterator fusion of the KD6 body's post-join elementwise chains; CONFIG "kd7"
        # (requires "kd6_concurrency"). Compiled + self-tested at build; any failure -> the KD6 body (kd7_error).
        self._kd7 = bool(self.model.cfg.get("kd7", False)) and self._kd6
        self._kd7j = None
        self.kd7_error = None
        self.kd7_stats = None
        # REC (rec_detect.py): causal regime-change detector + partial reset; CONFIG "rec_detect" (None = off = parent
        # bit for bit). KD3 path only; any failure -> off (rec_error keeps the reason).
        self._rec = None
        self.rec_error = None
        # TKA (experiments/tka_tke): low-ke fluctuation damping of the EMITTED pred, CONFIG "tka_lowke" (None = off = the
        # parent bit for bit). {"theta", "delta", "gmin"}: g(x) = clamp((K_p(x) / (theta * mean_x K_p))^delta, gmin, 1) on
        # u, v with K_p the physical ke map of the step's own pred; pred' = m_t + g (pred - m_t) (time mean kept).
        # Output-only: the un-damped pred is kept in _pred_raw and is what the next step's innovation reads.
        # Whole-step graph only (windows >= 2); windows 0 / 1 and every fallback path emit the un-damped pred.
        self._lowke = self.model.cfg.get("tka_lowke") or None
        try:
            self._check_preconditions()
            while True:
                try:
                    self._capture(warmup_iters)
                    break
                except Exception as e:  # noqa: BLE001
                    if self._kd3:
                        self.kd3_error = repr(e)
                        self._kd3 = False
                    elif self._kd2:
                        self.kd2_error = repr(e)
                        self._kd2 = False
                    else:
                        raise
                    self.graph_scan = self.graph_noscan = None
                    torch.cuda.synchronize()
            if self._kd6:
                self._kd6_build()
            self.ok = True
        except Exception as e:  # noqa: BLE001 - any failure -> v2 path
            self.ok = False
            self.graph_scan = self.graph_noscan = None
            self.capture_error = repr(e)

    # ------------------------------------------------------------ checks
    def _check_preconditions(self):
        m = self.model
        cfg = m.cfg
        base = m.base
        if not (type(base).__name__ == "GraphedCtxNormWrapper" and getattr(base, "ok", False)):
            raise RuntimeError("ctxnorm graph not available")
        for name in ("_mega", "_maskg", "_scang", "_cwg"):
            w = getattr(m, name, None)
            if w is None or not getattr(w, "ok", False):
                raise RuntimeError(f"{name} graph not available")
        if m.cw is None or not cfg.get("cond_width") or not cfg["bounds"] or not cfg["ctr"]:
            raise RuntimeError("scope: cond_width/bounds/ctr must be on")
        if cfg["uc_fixed"] is not None or cfg["uc_guard"] or cfg["mean_mode"] or cfg["er_online"] or m.ke_steps:
            raise RuntimeError("scope: uc_fixed/uc_guard/mean_mode/er_online/ke_steps must be off")
        if not cfg["advect"] or not cfg["gain"] or cfg.get("e_blend_w") is None:
            raise RuntimeError("scope: advect/gain/e_blend_w expected on")
        if m.prior is None:
            raise RuntimeError("scope: prior expected")
        # post-window-1 state must exist to capture against (same as mega)
        if (m.S is None or m.prev_out is None or m.prev_fl is None or m.prev_adv is None
                or m.prev_mf is None or m.prev_mask is None or m.prev_const is None
                or m.prev_basis is None or m.prev_basis2 is None or m.R_run is None
                or m.ctr_out is None or m.uc is None or m.ke_flow is None or m.ke_flow.n < 1):
            raise RuntimeError("model has no post-window-1 state yet")
        if hasattr(type(m.cw), "update_both") and getattr(m.cw, "_ema2", None) is None:
            raise RuntimeError("T2 cond-width state not seeded yet")
        # T2 (experiments/t2_gpu_compute): cond_width_net carries the batched
        # forward when `CondWidthState.update_both` exists (appended patch).
        self._t2 = hasattr(type(m.cw), "update_both")
        if not self._t2:
            raise RuntimeError("KD whole-step requires the T2 batched cond-width state")
        self._uc_warm = int(cfg["uc_warm"])
        self._uc_every = int(cfg["uc_every"])
        self._prior_n = float(cfg["prior_n"])
        self._alpha_min = float(cfg["alpha_min"])

    # ------------------------------------------------------------ buffers
    def _alloc_buffers(self, device, dtype):
        """KD (kernel diet): persistent state lives ONLY here and in mega's own
        state buffers; the composed bodies read / write it through an I/O
        namespace (`self._io`) whose attributes ALIAS these buffers, so the
        per-step copy-in / copy-out glue of the original composition is gone.
        Aliases (each argued in `_forward_body`):
          prev_out   == the device output pack's pred slot (read at the head,
                        rewritten by mega's `out=`)
          prev_mf    == this step's mf buffer (read at the head for blend_y,
                        rewritten after the backbone)
          prev_const / prev_basis / prev_basis2 / prev_adv == mega's
                        `_out_const` / `_out_basis` / `_out_b2` / `_out_adv`
                        (read in mega's PRE half, rewritten in its TAIL)
          ctr_out    == mega's `_out_ctr_out` (the cond-width update, which
                        needs the PREVIOUS centre, runs at the head)
          lower/upper == the output pack's slots (written with `out=`)."""
        B, T, H, W, C = STATIC_SHAPE
        f = lambda shp, dt=dtype: torch.zeros(*shp, device=device, dtype=dt)  # noqa: E731
        # KD2: the fused selectors' static tables (None -> parent path)
        self._kd2p = self._kd2s = None
        if self._kd2:
            try:
                import kd2_adapt as _KD2
                m = self.model
                self._kd2p = _KD2.KD2Plan(getattr(m, "cb_adapt", None), getattr(m, "kw_adapt", None),
                                          getattr(m, "eb_adapt", None), getattr(m, "cb_w", None),
                                          br=getattr(m, "br_adapt", None), T=T, H=H, W=W, device=device)
                self._kd2s = _KD2.KD2State(self._kd2p, T, H, W, device)
                _KD2.prepare_seed_tables(self._kd2p, self._kd2s, device)
            except Exception as e:  # noqa: BLE001 - not the covered configuration -> parent path
                self.kd2_error = repr(e)
                self._kd2 = False
                self._kd2p = self._kd2s = None
        # KD3: static buffers of the backbone / mega diet (None -> parent path)
        self._kd3s = None
        if self._kd3:
            try:
                import kd3_step as _KD3
                self._kd3s = _KD3.KD3State(self.model, T, H, W, C, device)
            except Exception as e:  # noqa: BLE001 - not the covered configuration -> parent path
                self.kd3_error = repr(e)
                self._kd3 = False
                self._kd3s = None
        self._rec = None
        _rc = self.model.cfg.get("rec_detect")
        if _rc:
            try:
                if not self._kd3:
                    raise RuntimeError("rec: KD3 path required")
                import rec_detect as _REC
                self._rec = _REC.RECState(_rc, self.model, T, H, W, device)
            except Exception as e:  # noqa: BLE001 - REC off, parent path
                self.rec_error = repr(e)
                self._rec = None
        # I/O packs
        self._ix, self._iy = 0, 1                                  # rows of x / prev_target in the H2D pack
        if self._kd2:
            # KD2 (AXE exact mode): ONE static buffer [candidate rows (Ke), prev_target, x, ctr_out]: the H2D pack =
            # rows Ke:Ke+2 (order [prev_target, x]), axe_adapt.ebr_scores' Z = (candidates, target) = rows 0:Ke+1 (no
            # cat; var_mean on this view == on the contiguous copy, bit for bit), the cond-width update's (target, centre)
            # = rows Ke and Ke+2
            Ke = self._kd2p.Ke
            self._big = f((3 + Ke, T, H, W, C))
            self._dev_in = self._big[Ke:Ke + 2]
            self._ix, self._iy = 1, 0
        else:
            self._dev_in = f((2, T, H, W, C))                      # [x, prev_target]
        self._host_in = torch.empty(2, T, H, W, C, dtype=dtype, pin_memory=True)
        self._dev_out = f((3 * N_ELEM + 1,))                    # [pred, lower, upper, loss]
        self._host_out = torch.empty(3 * N_ELEM + 1, dtype=dtype, pin_memory=True)
        self._dev_out_views = [self._dev_out[i * N_ELEM:(i + 1) * N_ELEM].view(B, T, H, W, C) for i in range(3)]
        self._dev_loss = self._dev_out[3 * N_ELEM]
        self._host_out_views = [self._host_out[i * N_ELEM:(i + 1) * N_ELEM].view(B, T, H, W, C) for i in range(3)]
        if self._lowke:
            # TKA: state copy of the un-damped pred + the ke weights 0.5 std^2 (u, v; 0 for p) and the u, v channel mask
            self._pred_raw = f((B, T, H, W, C))
            _std = self.model.std_tgt.to(device=device, dtype=dtype)
            _kw = torch.zeros(C, 1, device=device, dtype=dtype)
            _kw[:2, 0] = 0.5 * _std[:2] ** 2
            self._lk_kw = _kw
            _chm = torch.zeros(C, device=device, dtype=dtype)
            _chm[:2] = 1.0
            self._lk_chm = _chm
            self._lk_negchm = -_chm
            self._lk_theta = float(self._lowke["theta"])
            self._lk_delta = float(self._lowke["delta"])
            self._lk_gmin = float(self._lowke["gmin"])
        # persistent cross-step state owned here
        self._prev_fl = f((B, T, H, W, C))
        self._prev_adv = f((B, T, H, W, C))
        self._mf = f((B, 1, H, W, C))                           # this step's mf == next step's prev_mf
        self._prev_mask = torch.zeros(H, W, device=device, dtype=torch.bool)
        self._prev_const = f((B, 1, H, W, C))
        if self._kd3:
            # KD3: prev_basis / prev_basis2 are rows 1, 2 of the stacked [a0, basis, basis2] buffer
            self._prev_basis = self._kd3s.ABC[1]
            self._prev_basis2 = self._kd3s.ABC[2]
        else:
            self._prev_basis = f((B, T, H, W, C))
            self._prev_basis2 = f((B, T, H, W, C))
        self._R_run = f(STATIC_E_SHAPE)
        self._ctr_out = self._big[2 + self._kd2p.Ke:3 + self._kd2p.Ke] if self._kd2 else f((B, T, H, W, C))  # KD2: last row
        self._ctr_var = f((B, 1, H, W, C))
        self._uc = torch.zeros((), device=device, dtype=dtype)
        # scratch written with out= each step
        self._err = f((B, T, H, W, C))
        self._blend_y = f((B, T, H, W, C))
        # float64 device counters. KD: ONE vector, ONE add_ per step:
        #   [n, n_c, n_b, n_e, kf_n_pre, n_R]; element 4 holds (ke_flow.n - 1)
        #   so that after the per-step +1 it equals the PRE-increment ke_flow.n
        #   the original read for wfrac (integers are exact in float64).
        # REC: the counter vector is a view of the REC [counters | a | 1 - a] buffer (one reset where for all three)
        self._cnt = self._rec.cnt if self._rec is not None else torch.zeros(6, device=device, dtype=torch.float64)
        self._n_uc = torch.zeros((), device=device, dtype=torch.float64)
        self._prior_n_t = torch.full((), self._prior_n, device=device, dtype=torch.float64)
        self._alpha_min_t = torch.full((), self._alpha_min, device=device, dtype=torch.float64)
        if self._t2 and self._kd3:
            self._prev2 = self._kd3s.PE3[0]                 # KD3: rows of [prev2, ema2, ratio]
            self._ema2 = self._kd3s.PE3[1]
        elif self._t2:
            self._prev2 = f((2, H, W))
            self._ema2 = f((2, H, W))
        # AW (cb_adapt.py): previous window's (raw - ctr_out), pre-shift width (u, v) and the score EMA
        self._cb_ad = getattr(self.model, "cb_adapt", None)
        if self._kd2:
            # KD2: the score EMAs are views of ONE buffer; w_prev / valid live in the gather output
            ks = self._kd2s
            self._kw_ad = self.model.kw_adapt
            self._eb_ad = self.model.eb_adapt
            self._br_ad = getattr(self.model, "br_adapt", None)
            self._cb_d = f((B, T, H, W, C))
            self._cb_wd = f((B, T, H, W, 2))
            self._cb_S, self._kw_S, self._eb_S, self._eb_valid = ks.cb_S, ks.kw_S, ks.eb_S, ks.valid
            if self._br_ad is not None:
                self._br_S = ks.br_S
            Ke = self._kd2p.Ke
            self._eb_outK = self._big[0:Ke]
            self._pack12 = self._dev_out[N_ELEM:3 * N_ELEM].view(2, T, H, W, C)
            self._pc = self._big[Ke:Ke + 3:2]                           # rows [prev_target, ctr_out]
            self._zt = self._big[0:Ke + 1][..., :2]                     # (candidates, target)
        elif self._cb_ad is not None:
            self._cb_d = f((B, T, H, W, C))
            self._cb_wd = f((B, T, H, W, 2))
            self._cb_S = f((self._cb_ad.K,))
        # AX (ax_adapt.py): kappa score EMA + the CB weight emitted on the previous window; blend candidates
        if not self._kd2:
            self._kw_ad = getattr(self.model, "kw_adapt", None) if self._cb_ad is not None else None
        if self._kw_ad is not None and not self._kd2:
            self._kw_S = f((self._kw_ad.G, self._kw_ad.K))
            self._kw_wprev = f((B, T, 1, 1, C))
        if not self._kd2:
            self._eb_ad = getattr(self.model, "eb_adapt", None)
        if self._kd2:
            pass
        elif self._eb_ad is not None:
            self._eb_S = f((self._eb_ad.K,))
            self._eb_valid = torch.zeros((), device=device, dtype=dtype)
            # AXE (axe_adapt.py): with br_adapt the candidate buffer holds K_eb + K_br rows
            self._br_ad = getattr(self.model, "br_adapt", None)
            self._eb_outK = f((self._eb_ad.K if self._br_ad is None else self._br_ad.Kc, T, H, W, C))
            if self._br_ad is not None:
                self._br_S = f((self._br_ad.K,))
        else:
            self._br_ad = None
        # ---- mega's I/O namespace (static part; per-step tensors set in the body)
        mega = self.model._mega
        io = _IO()
        io._x = self._dev_in[self._ix:self._ix + 1]
        io._y = self._dev_in[self._iy:self._iy + 1]
        io._prev_mask = self._prev_mask
        io._mf = self._mf
        io._err = self._err
        io._blend_y = self._blend_y
        io._prev_fl = self._prev_fl
        io._prev_adv = self._prev_adv
        io._prev_const = self._prev_const
        io._prev_basis = self._prev_basis
        io._prev_basis2 = self._prev_basis2
        io._uc = self._uc
        io._uc_eff = self._uc
        io._R_run = self._R_run
        for k in ("_mu", "_var", "_M0", "_M1", "_M2", "_Saa", "_Sab", "_Sbb", "_S02", "_S12", "_S22",
                  "_S", "_S5", "_e_true", "_e_pred", "_ex_prev", "_ef_prev", "_fm"):
            setattr(io, k, getattr(mega, k))                  # mega's persistent state, shared
        if self._kd3:
            import kd3_step as _KD3
            _KD3.bind_io(io, self._kd3s)                      # KD3: M0..M2 / second moments live stacked
        io._out_adv = self._prev_adv
        io._out_const = self._prev_const
        io._out_basis = self._prev_basis
        io._out_b2 = self._prev_basis2
        io._out_out = self._dev_out_views[0]
        io._out_ctr_out = self._ctr_out
        io._out_ctr_var = self._ctr_var
        sh = _Shadow()
        sh.__dict__.update(mega.shadow.__dict__)
        sh.uc, sh.uc_eff = self._uc, self._uc
        io.shadow = sh
        if self._eb_ad is not None:
            io._eb_ad = self._eb_ad
            io._eb_outK = self._eb_outK
            if self._br_ad is not None:
                io._br_ad = self._br_ad
        self._io = io

    # ------------------------------------------------------------ body
    def _a64(self, n64: torch.Tensor) -> torch.Tensor:
        # == GainMapImprovedModel._a(n) = max(1/(n + prior_n), alpha_min), in IEEE double
        return torch.maximum(1.0 / (n64 + self._prior_n_t), self._alpha_min_t)

    def _forward_body(self, scan: bool):
        m = self.model
        base, mega, maskg, scang = m.base, m._mega, m._maskg, m._scang
        io = self._io
        x = self._dev_in[self._ix:self._ix + 1]
        y = self._dev_in[self._iy:self._iy + 1]
        prev_out = self._dev_out_views[0]          # still holds the previous step's pred
        rec = self._rec          # REC: detector inside kd3 ctx_body; rescale / counter reset below

        # ---- step() head (verbatim maths on the same values) ----
        # TKA: the innovation reads the un-damped previous pred (== prev_out when tka_lowke is off)
        err = torch.sub(y, self._pred_raw if self._lowke else prev_out, out=self._err)
        if self._kd3:
            torch.mean(err ** 2, dim=(0, 1, 2, 3, 4), out=self._dev_loss)   # KD3: same reduction, no copy
        else:
            self._dev_loss.copy_((err ** 2).mean())
        torch.sub(y, self._mf, out=self._blend_y)   # self._mf still holds prev_mf here
        # counters n, n_c, n_b, n_e, kf_n_pre, n_R: all +1 in one kernel
        cnt = self._cnt
        cnt.add_(1.0)
        # _update_R: plain running mean, 1/n_R. v2: `(ky - R) / self.n_R` with a
        # PYTHON INT divisor -> multiply by the float32 reciprocal. Reproduce that.
        # KD: temporal energies of x and prev_target in ONE var over the input
        # pair (kx is handed to mega through the I/O namespace)
        kxy = self._dev_in.var(dim=1, keepdim=True, unbiased=False)
        ky = kxy[self._iy:self._iy + 1]
        io._kx = kxy[self._ix:self._ix + 1]
        if rec is None:   # REC: with rec, the R_run update moves below the counter reset (nothing reads R_run between)
            if self._kd3:
                inv_nR = torch.reciprocal(cnt[5].to(ky.dtype))   # KD3: == 1.0 / t (reciprocal * 1.0, exact identity)
            else:
                inv_nR = 1.0 / cnt[5].to(ky.dtype)
            torch.add(self._R_run, (ky - self._R_run) * inv_nR, out=self._R_run)
        # cond-width UPDATE, moved up from after mega (values-preserving: it only
        # reads prev_target and the PREVIOUS centre, which mega overwrites below)
        from t2_cw_graph_body import t2_cw_apply, t2_cw_update
        if self._kd2:
            # KD2: cond-width update (mask straight into Q row 0) + AW / KW / EB (/ BR) selection, fused (kd2_adapt.py)
            import kd2_adapt as _KD2
            kp, ks = self._kd2p, self._kd2s
            _KD2.cw_update(m, m.cw, self._pc, self._prev2, self._ema2, ks.Q0cw, st=ks)
            io._eb_k = _KD2.select(kp, ks, y[0, ..., :2], self._ctr_out[0, ..., :2], self._cb_d[0, ..., :2],
                                   self._cb_wd[0], self._zt, y=y, outK=self._eb_outK)
            if kp.br:
                io._br_zidx, io._br_eidx = ks.zidx, ks.eidx
            t2_sc = None
        else:
            t2_sc = t2_cw_update(m, m.cw, y, self._ctr_out, self._prev2, self._ema2)
        # AW: this window's CB weight from the previous window's revealed truth (reads the
        # PREVIOUS ctr_out, which mega overwrites below; d / width are rewritten at the tail)
        cb_wsel = None
        if self._cb_ad is not None and not self._kd2:
            from cb_adapt import cb_adapt_select
            cb_wsel = cb_adapt_select(self._cb_ad, self._cb_S, y, self._ctr_out, self._cb_d, self._cb_wd,
                                      scored=t2_sc.permute(1, 2, 3, 0).unsqueeze(0))
        kw_f = None
        if self._kw_ad is not None and not self._kd2:
            from ax_adapt import kw_select
            kw_f = kw_select(self._kw_ad, self._kw_S, y, self._ctr_out, self._cb_d, self._cb_wd, self._kw_wprev,
                             t2_sc.permute(1, 2, 3, 0).unsqueeze(0))
        if self._eb_ad is not None and not self._kd2:
            # AX: blend weight of this window from the previous window's K candidate outputs (read before mega
            # overwrites _eb_outK); valid = 0 on the first graphed window of a trajectory
            if self._br_ad is not None:
                # AXE: factorised (blend, ridge) rows -> both choices + the per-row candidate indices
                from axe_adapt import ebr_select
                io._eb_k, io._br_zidx, io._br_eidx = ebr_select(self._eb_ad, self._br_ad, self._eb_S, self._br_S,
                                                                self._eb_valid, y, self._eb_outK)
            else:
                from ax_adapt import eb_select
                io._eb_k = eb_select(self._eb_ad, self._eb_S, self._eb_valid, y, self._eb_outK)

        # ---- mask, backbone ----
        mask = maskg._body(x)
        if self._kd3:
            import kd3_step as _KD3
            raw = _KD3.ctx_body(base, x, self._kd3s, rec=rec)
            if rec is not None:
                rec.rescale_mean([mega._mu, self._kd3s.M3])   # REC: mean-type state x r (x 1.0 when no reset)
        else:
            raw = base._body(x)
        mf = torch.mean(raw, dim=1, keepdim=True, out=self._mf)
        fl_f = torch.sub(raw, mf, out=mega._fm[0:1])   # slot 0 of mega's (fl_f, mu_p) pair

        # ---- convection speed scan (host-scheduled variant) ----
        if scan:
            obstacle = m.shift.last_obstacle(mask)
            mx = x.mean(dim=1, keepdim=True)
            fl_x = x - mx
            uc_new, _peak = scang._body(fl_x, obstacle, mask)
            self._n_uc.add_(1.0)
            a_uc = self._a64(self._n_uc).to(uc_new.dtype)
            torch.add(self._uc, a_uc * (uc_new - self._uc), out=self._uc)

        # ---- EMA coefficients of n, n_c, n_b, n_e as ONE vector op chain ----
        if self._kd3:
            if rec is None:
                a4 = torch.maximum(torch.reciprocal(cnt[0:4] + self._prior_n_t), self._alpha_min_t)   # KD3: == _a64
            else:
                # REC: the same a4 and 1 - a4 (into CA), then ONE where resets the counters, skips the EMA updates and
                # sets 1 - a = r^2 for var / second moments on a reset step; then the (moved) R_run update (same ops)
                a4, _ = rec.counters_a4(self._prior_n_t, self._alpha_min_t)
                torch.add(self._R_run, (ky - self._R_run) * torch.reciprocal(cnt[5].to(ky.dtype)), out=self._R_run)
        else:
            a4 = self._a64(cnt[0:4])
        a4f = a4.to(torch.float32)
        io._a_err_f32, io._a_err_f64 = a4f[0], a4[0]
        if self._kd3:
            io._om4 = (1.0 - a4) if rec is None else rec.om4   # KD3: the two (1 - a) of mega's compound EMAs (REC: from CA)
        io._a_ctr_f32, io._a_ctr_f64 = a4f[1], a4[1]
        io._a_bld_f32 = a4f[2]
        io._a_e_f32 = a4f[3]
        io._wfrac = (torch.clamp_max(cnt[4], 20.0) / 20.0).to(torch.float32).view(1)
        io._mask_in = mask
        io._fl_f = fl_f
        if self._kd3:
            _KD3.mega_body(mega, io, self._kd3s)
        else:
            mega._body(io)

        # ---- carry the state that is not aliased ----
        self._prev_fl.copy_(fl_f)
        if not self._kd3:
            self._prev_mask.copy_(mask)   # KD3: next step's keep_prev is written by kd3_step._coefficients

        # ---- bounds: cond-width apply, written straight into the output pack ----
        if self._kd3:
            _KD3.cw_apply(m, m.cw, m._cwg._k_scale, self._ctr_out, self._ctr_var, x, prev_out,
                          self._prev2, self._ema2, self._dev_out_views[1], self._dev_out_views[2], self._kd3s)
        else:
            t2_cw_apply(m, m.cw, m._cwg._k_scale, self._ctr_out, self._ctr_var, x, prev_out,
                        self._prev2, self._ema2, self._dev_out_views[1], self._dev_out_views[2])
        # CB (experiments/cb_centre_blend): shift the emitted interval toward the raw backbone
        # output, in the output pack, in place (3 kernels; absent when cb_w is None).
        cbw = getattr(m, "cb_w", None)
        if self._kd2:
            # KD2: same shifts, lower and upper in one launch each (per element: +w d, then +/- f wd, as before)
            ks = self._kd2s
            d = torch.sub(raw, self._ctr_out, out=self._cb_d)
            torch.sub(self._dev_out_views[2][..., :2], self._dev_out_views[1][..., :2], out=self._cb_wd)
            self._pack12.addcmul_(ks.cbw, d)
            self._pack12[..., :2].addcmul_(ks.f2, self._cb_wd)
        elif cbw is not None and cb_wsel is not None:
            # AW: same values as the eager path (cb_adapt.py); d and the pre-shift width persist
            d = torch.sub(raw, self._ctr_out, out=self._cb_d)
            torch.sub(self._dev_out_views[2][..., :2], self._dev_out_views[1][..., :2], out=self._cb_wd)
            self._dev_out_views[1].addcmul_(cb_wsel, d)
            self._dev_out_views[2].addcmul_(cb_wsel, d)
            if kw_f is not None:
                # AX: half-width x kappa (u, v); the pre-kappa width stays in _cb_wd for both scores
                self._dev_out_views[1][..., :2].addcmul_(kw_f, self._cb_wd)
                self._dev_out_views[2][..., :2].addcmul_(kw_f, self._cb_wd, value=-1.0)
                self._kw_wprev.copy_(cb_wsel)
        elif cbw is not None:
            d = raw - self._ctr_out
            self._dev_out_views[1].addcmul_(cbw, d)
            self._dev_out_views[2].addcmul_(cbw, d)
        if self._eb_ad is not None and not self._kd2:
            self._eb_valid.fill_(1.0)
        if self._lowke:
            self._lowke_apply()

    def _lowke_apply(self) -> None:
        """TKA: damp the emitted pred's temporal fluctuation where its own (physical) ke is low, in place in the output
        pack's pred slot; the un-damped pred is saved first for the next step's innovation. ~11 kernels."""
        P = self._dev_out_views[0]
        self._pred_raw.copy_(P)
        var, mean = torch.var_mean(P, dim=1, keepdim=True, unbiased=False)       # (1,1,H,W,C)
        K = torch.matmul(var, self._lk_kw)                                         # (1,1,H,W,1) physical ke
        kb = torch.mean(K).clamp_min_(1e-20).mul_(self._lk_theta)
        g = torch.div(K, kb).pow_(self._lk_delta).clamp_(self._lk_gmin, 1.0)
        G = torch.addcmul(self._lk_negchm, g, self._lk_chm)                          # (g - 1) on u, v; 0 on p
        P.addcmul_(G, torch.sub(P, mean))

    # ------------------------------------------------------------ capture
    def _capture(self, warmup_iters: int) -> None:
        m = self.model
        ref = m.std_tgt
        device, dtype = ref.device, ref.dtype
        self._alloc_buffers(device, dtype)
        # capture against the real post-window-1 state of the dummy warm-up
        # trajectory (finite, correctly shaped); every persistent buffer is
        # re-seeded for real at the first graphed step of each trajectory.
        self._seed()
        self._dev_in[0:1].normal_()
        self._dev_in[1:2].normal_()
        s_stream = torch.cuda.Stream()
        s_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_stream), torch.no_grad():
            for _ in range(warmup_iters):
                self._forward_body(scan=True)
                self._forward_body(scan=False)
        torch.cuda.current_stream().wait_stream(s_stream)
        self.graph_scan = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph_scan):
            self._forward_body(scan=True)
        self.graph_noscan = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph_noscan):
            self._forward_body(scan=False)
        torch.cuda.synchronize()

    def _kd6_build(self) -> None:
        """KD6: replace the noscan graph by the concurrent one if it builds and passes its self-test; the model
        state is restored to exactly what it was before the attempt either way."""
        try:
            import kd6_conc as _KD6
            if self._kd7:
                try:
                    import kd7_fuse as _KD7
                    self._kd7j, self.kd7_stats = _KD7.build(self)
                except Exception as e:  # noqa: BLE001 - KD6 body
                    self.kd7_error = repr(e)
                    self._kd7j, self._kd7 = None, False
            try:
                g, side, stats = _KD6.build(self)
            except Exception as e:  # noqa: BLE001
                if self._kd7j is None:
                    raise
                self.kd7_error = repr(e)          # KD7 body failed (capture / self-test) -> retry the KD6 body
                self._kd7j, self._kd7 = None, False
                g, side, stats = _KD6.build(self)
            self.graph_noscan = g
            self._kd6_side = side
            self.kd6_stats = stats
        except Exception as e:  # noqa: BLE001 - parent noscan graph
            self.kd6_error = repr(e)
            self._kd6 = False
            self._kd7j, self._kd7 = None, False
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ state
    def _seed(self) -> None:
        """Copy the live model state (as left by v2's own window 0/1 code)
        into the graph's persistent buffers. Called once per trajectory."""
        m = self.model
        mega = m._mega
        if self._kd3:
            import kd3_step as _KD3
            _KD3.sync_state_in(mega, self._kd3s)
        else:
            mega._sync_state_in()
        self._dev_out_views[0].copy_(m.prev_out)       # KD: prev_out lives in the pred slot
        if self._lowke:
            self._pred_raw.copy_(m.prev_out)           # TKA: the innovation's previous pred (un-damped)
        self._prev_fl.copy_(m.prev_fl)
        self._prev_adv.copy_(m.prev_adv)
        self._mf.copy_(m.prev_mf)                      # KD: prev_mf lives in the mf buffer
        self._prev_mask.copy_(m.prev_mask)
        if self._kd3:
            self._kd3s.KP.copy_((~self._prev_mask).view(self._kd3s.KP.shape))   # KD3: keep_prev of the first graphed step
        self._prev_const.copy_(m.prev_const)
        self._prev_basis.copy_(m.prev_basis)
        self._prev_basis2.copy_(m.prev_basis2)
        self._R_run.copy_(m.R_run)
        self._ctr_out.copy_(m.ctr_out)
        self._uc.copy_(m.uc)
        # KD-R fix: fill_ with python scalars (capturable, no host tensor / H2D copy /
        # stream sync) -- the original T1 form; values identical (exact ints in f64).
        for i, v in enumerate((float(m.n), float(m.n_c), float(m.n_b), float(m.n_e),
                               float(m.ke_flow.n) - 1.0, float(m.n_R))):
            self._cnt[i].fill_(v)
        self._n_uc.fill_(float(m.n_uc))
        if self._rec is not None:
            self._rec.seed(m.base)
        self._prev2.copy_(m.cw._prev2)
        self._ema2.copy_(m.cw._ema2)
        if self._kd2:
            # KD2: same seeds, written into the fused buffers (copy_ / fill_ only: capturable inside W01's G1)
            import kd2_adapt as _KD2
            if getattr(m, "_cb_d", None) is not None:
                self._cb_d.copy_(m._cb_d)
                self._cb_wd.copy_(m._cb_wd)
                _KD2.seed(self._kd2p, self._kd2s, m._cb_S)
            else:
                self._cb_d.zero_()
                self._cb_wd.zero_()
                _KD2.seed(self._kd2p, self._kd2s, self._cb_ad.S0)
        elif self._cb_ad is not None:
            # AW: window 1's (d, width, S) from the eager / W01 step; absent -> no evidence (ties -> fixed w)
            if getattr(m, "_cb_d", None) is not None:
                self._cb_d.copy_(m._cb_d)
                self._cb_wd.copy_(m._cb_wd)
                self._cb_S.copy_(m._cb_S)
            else:
                self._cb_d.zero_()
                self._cb_wd.zero_()
                self._cb_S.copy_(self._cb_ad.S0)
        if self._kw_ad is not None and not self._kd2:
            # AX: window 1 emitted the fixed CB weight and kappa = 1 -> its evidence is (d, width) above + cb_w
            self._kw_S.copy_(self._kw_ad.S0)
            self._kw_wprev.copy_(m.cb_w)
        if self._eb_ad is not None and not self._kd2:
            # AX: window 1's candidates do not exist -> no evidence on the first graphed window
            self._eb_S.copy_(self._eb_ad.S0)
            self._eb_valid.zero_()
            if self._br_ad is not None:
                self._br_S.copy_(self._br_ad.S0)

    def should_use(self) -> bool:
        """Read at the top of ttt_step. True from window 2 of a trajectory on
        (same gate as mega + cwg live), monotonic until reset()."""
        if not self.ok:
            return False
        if self.live:
            return True
        m = self.model
        if self._t2 and (getattr(m.cw, "_ema2", None) is None or getattr(m.cw, "_prev2", None) is None):
            return False
        return (m._mega.should_use() and m._cwg._live and m.uc is not None
                and m.prev_out is not None and m.params is not None and m.ke_flow.n >= 1)

    def reset(self) -> None:
        self.live = False

    # ------------------------------------------------------------ API
    def ttt_step(self, input_norm, prev_target_norm):
        m = self.model
        x_in = torch.as_tensor(input_norm)
        in_device = x_in.device
        if not self.live:
            self._seed()
            self.live = True
        if self.cuda_direct and in_device.type == "cuda":
            return self._ttt_step_cuda_direct(x_in, prev_target_norm)
        self._host_in[self._ix:self._ix + 1].copy_(x_in.reshape(STATIC_SHAPE))
        self._host_in[self._iy:self._iy + 1].copy_(torch.as_tensor(prev_target_norm).reshape(STATIC_SHAPE))
        self._dev_in.copy_(self._host_in, non_blocking=True)          # ONE H2D
        n_win = m.n_win
        scan = (n_win < self._uc_warm) or (n_win % self._uc_every == 0)
        (self.graph_scan if scan else self.graph_noscan).replay()      # ONE replay
        self._host_out.copy_(self._dev_out, non_blocking=True)        # ONE D2H
        torch.cuda.current_stream().synchronize()                     # ONE sync
        self.n_replays += 1
        # mirror the host-side counters v2 keeps (cheap ints; keeps the
        # schedule and any later inspection consistent)
        m.n += 1; m.n_c += 1; m.n_b += 1; m.n_e += 1; m.n_R += 1; m.ke_flow.n += 1
        if scan:
            m.n_uc += 1
        m.n_w += 1
        m.n_win += 1
        if in_device.type == "cpu" and _ipc_one_storage(self._one_storage, x_in):
            _c = self._host_out[:3 * N_ELEM].clone()
            _s = self._host_out_views[0].shape
            out, lower, upper = (_c[0:N_ELEM].view(_s), _c[N_ELEM:2 * N_ELEM].view(_s),
                                 _c[2 * N_ELEM:3 * N_ELEM].view(_s))
            self.n_one_storage += 1
        else:
            out = self._host_out_views[0].clone()
            lower = self._host_out_views[1].clone()
            upper = self._host_out_views[2].clone()
        loss = float(self._host_out[3 * N_ELEM])
        if in_device.type != "cpu":
            out, lower, upper = out.to(in_device), lower.to(in_device), upper.to(in_device)
        return out, {"adapt_loss": loss, "lower": lower, "upper": upper}

    def _ttt_step_cuda_direct(self, x_in, prev_target_norm):
        """HX2 (experiments/hx2_harness): caller's inputs already on CUDA -> D2D copies into the static input rows (was:
        two synchronous D2H into the pinned pack + one H2D), ONE replay, outputs = fresh CUDA clones of the static output
        rows (was: 1.5 MB D2H, three host clones, three pageable H2D back to the caller's device), adapt_loss via a
        4-byte D2H + the same single stream sync. Values bit-identical; outputs complete (synchronised) at return."""
        m = self.model
        self._dev_in[self._ix:self._ix + 1].copy_(x_in.reshape(STATIC_SHAPE))
        self._dev_in[self._iy:self._iy + 1].copy_(torch.as_tensor(prev_target_norm).reshape(STATIC_SHAPE))
        n_win = m.n_win
        scan = (n_win < self._uc_warm) or (n_win % self._uc_every == 0)
        (self.graph_scan if scan else self.graph_noscan).replay()      # ONE replay
        out = self._dev_out_views[0].clone()
        lower = self._dev_out_views[1].clone()
        upper = self._dev_out_views[2].clone()
        self._host_out[3 * N_ELEM:].copy_(self._dev_out[3 * N_ELEM:], non_blocking=True)
        torch.cuda.current_stream().synchronize()                     # ONE sync
        self.n_replays += 1
        m.n += 1; m.n_c += 1; m.n_b += 1; m.n_e += 1; m.n_R += 1; m.ke_flow.n += 1
        if scan:
            m.n_uc += 1
        m.n_w += 1
        m.n_win += 1
        self.n_direct += 1
        loss = float(self._host_out[3 * N_ELEM])
        return out, {"adapt_loss": loss, "lower": lower, "upper": upper}
