"""c16t_launch (launch-cost pass, phase 6), task item (c): CUDA graph capturing
`GainMapImprovedModel._scan_uc` (~45 launches/step averaged over calls/step
per `c10t_launch`'s own stage_profile, `experiments/c10t_launch/notes.md` S1
-- the single largest remaining eager block after phase 5's mega/mask
graphs, flagged but not attempted by every prior phase, see
`c10t_launch/notes.md` S7: "fires on a host-visible schedule ... as the task
brief itself suggested (capture as its own graph, replayed only on firing
steps, non-firing path stays default) -- not started yet").

`_scan_uc` is a PURE function of `(fl_x, obstacle, mask)` plus fixed
construction-time tensors (`self.uc_grid`, `self._uc_geom` -- precomputed
once in `GainMapImprovedModel.__init__`, never reassigned afterward,
confirmed by grep) and fixed CONFIG constants (`cfg["uc_taus"]`, a plain
python tuple of ints, unrolled at capture time) -- UNLIKE `_tail`/`_pre`/
`_mega` it has NO running-state dependency (no `self.n_*` counter read
anywhere inside the function body) and NO window-0 structural branch, so
this graph needs NO `should_use()` window gating beyond "capture succeeded"
(same as `cuda_graph_mask.py`'s `GraphedMaskWrapper`). No `.double()` casts,
no EMA -- verbatim transcription of `_scan_uc`'s body, nothing re-derived.

`_scan_uc` only FIRES on a host-visible schedule (`uc_warm=4`, `uc_every=4`,
`uc_fixed=None` for this shipped CONFIG -- see `submission.py`'s THREE
byte-identical
`if self.uc is None or self.n_win < uc_warm or self.n_win % uc_every == 0:
self._update_uc(self._scan_uc_dispatch(fl_x, obstacle, mask))` call sites,
one each in the `use_mega`/`self._tail`/eager-advect branches of `step()`),
decided from plain python ints / None-ness entirely on the HOST side, BEFORE
this wrapper is ever invoked -- so the graphed-vs-eager choice here composes
cleanly with that firing schedule: on a non-firing step this wrapper's
`forward()` is simply never called (0 launches, identical to the eager
code's own `elif`-skip), and `self.uc`/`self.uc_eff` are left untouched
either way, bit-identical to the eager skip. `_update_uc` itself (the EMA
merge + optional stability guard, `uc_guard=None` for this shipped CONFIG so
its own branch is dead too) is NOT captured here -- cheap (a handful of
scalar ops on 0-dim tensors), and keeping it eager avoids re-deriving a THIRD
counter's (`self.n_uc`) bit-exactness beyond what this task's `bounds()`
graph already needed none of, matching `cuda_graph_tail.py`'s own precedent
of leaving small counter-driven scalar updates eager around a graphed pure
function.

Correctness contract identical to every other graph in this package:
fixed-address static buffers, `.copy_()` in / `.clone()` out, no allocation
inside the replayed graph, try/except at construction only, never in the
timed path.
"""
from __future__ import annotations

import torch
import torch.nn as nn

STATIC_FL_SHAPE = (1, 20, 32, 64, 3)
STATIC_HW = (32, 64)


class GraphedScanUcWrapper(nn.Module):
    def __init__(self, model, warmup_iters: int = 3):
        super().__init__()
        self.model = model  # kept for eager fallback + reading fixed buffers each step
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
        self._fl = torch.zeros(*STATIC_FL_SHAPE, device=device, dtype=dtype)
        self._obstacle = torch.zeros(*STATIC_HW, device=device, dtype=dtype)
        self._mask = torch.zeros(*STATIC_HW, device=device, dtype=torch.bool)
        self._out_uc = torch.zeros((), device=device, dtype=dtype)
        self._out_peak = torch.zeros((), device=device, dtype=dtype)

    def _forward_body(self):
        uc, peak = self._body(self._fl, self._obstacle, self._mask)
        self._out_uc.copy_(uc)
        self._out_peak.copy_(peak)

    def _body(self, fl, obstacle, mask):
        """KD: the scan math on any (fl, obstacle, mask), returning 0-dim
        (uc, peak) views -- the whole-step graph reads them directly."""
        m = self.model
        f = fl[..., :2]
        grid = m.uc_grid
        u, w = grid.numel(), m.shift.w
        keep = (~mask).view(1, 1, *mask.shape, 1, 1).to(f.dtype)
        obs = obstacle.long().unsqueeze(0)
        num = den_a = den_b = torch.zeros(u, device=f.device, dtype=f.dtype)
        for tau in m.cfg["uc_taus"]:
            i0, wgt, i0c, i1c = m._uc_geom[tau]
            valid = (i0.unsqueeze(1) >= obs + 1) & (i0 >= 0).unsqueeze(1)
            a = f[:, :-tau]
            shifted = (1.0 - wgt) * a[:, :, :, i0c, :] + wgt * a[:, :, :, i1c, :]
            b = f[:, tau:].unsqueeze(3)
            k = (valid.permute(1, 0, 2).reshape(1, 1, mask.shape[0], u, w, 1).to(f.dtype)
                 * keep.view(1, 1, mask.shape[0], 1, w, 1))
            num = num + (shifted * b * k).sum(dim=(0, 1, 2, 4, 5))
            den_a = den_a + (shifted * shifted * k).sum(dim=(0, 1, 2, 4, 5))
            den_b = den_b + (b * b * k).sum(dim=(0, 1, 2, 4, 5))
        s = num / (den_a * den_b + 1e-30).sqrt()
        # NOTE (c16t): the original eager `_scan_uc` indexes with a 0-DIM
        # tensor (`k = s.argmax()`; `s[k]`, `s[km]`, `s[kp]`, `grid[k]`) --
        # under CUDA graph capture that raises "operation not permitted when
        # stream is capturing" (0-dim/scalar tensor indexing takes an
        # internal `.item()`-style host-sync path in this PyTorch version;
        # confirmed by isolating this exact line with
        # `scripts/debug_scan_uc_capture_raw.py`). Multi-element advanced
        # indexing with a >=1-element LongTensor (e.g. `a[:, :, :, i0c, :]`
        # above) is NOT affected -- already proven safe under capture by the
        # shipped `cuda_graph_tail.py`'s own `_advect` call. Fix: keep every
        # index as a 1-ELEMENT 1-D tensor and use `index_select` throughout,
        # only reshaping to 0-dim at the very end (a pure view, no sync) --
        # bit-identical arithmetic, verbatim from the eager formula, just
        # never a bare scalar-tensor `__getitem__`.
        k0 = s.argmax()
        k1 = k0.reshape(1)
        km1 = (k1 - 1).clamp(0, u - 1)
        kp1 = (k1 + 1).clamp(0, u - 1)
        s_k = torch.index_select(s, 0, k1)
        s_km = torch.index_select(s, 0, km1)
        s_kp = torch.index_select(s, 0, kp1)
        den = s_km - 2.0 * s_k + s_kp
        interior = (k1 > 0) & (k1 < u - 1) & (den < 0)
        off = torch.where(interior, 0.5 * (s_km - s_kp) / torch.where(den < 0, den, -torch.ones_like(den)),
                          torch.zeros_like(den))
        grid_k = torch.index_select(grid, 0, k1)
        uc = grid_k + off * float(m.cfg["uc_step"])
        peak = s_k
        return uc.reshape(()), peak.reshape(())

    def _capture(self, warmup_iters: int) -> None:
        m = self.model
        ref = m.std_tgt
        device, dtype = ref.device, ref.dtype
        self._alloc_buffers(device, dtype)
        # deterministic, finite dummy data for warm-up/capture -- values never
        # matter (every real replay overwrites every input buffer before
        # use), only shapes/dtypes/addresses do.
        self._fl.normal_()
        self._obstacle.copy_(torch.randint(-1, 5, STATIC_HW, device=device).to(dtype))
        self._mask.copy_(torch.rand(STATIC_HW, device=device) > 0.5)

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
        return self.ok

    def forward(self, fl: torch.Tensor, obstacle: torch.Tensor, mask: torch.Tensor):
        self._fl.copy_(fl)
        self._obstacle.copy_(obstacle)
        self._mask.copy_(mask)
        self.graph.replay()
        return self._out_uc.clone(), self._out_peak.clone()
