"""c16t_launch (launch-cost pass, phase 6), task item (b): CUDA graph capturing
`GainMapImprovedModel.bounds` (~21 launches/step per `c10t_launch`'s own
stage_profile, `experiments/c10t_launch/notes.md` S1) as called from
`ExperimentTTT.ttt_step` -- the one remaining eager stage that crosses the
`step()`/`ttt_step()` method boundary, flagged and deferred by every prior
launch-cost phase (see `cuda_graph_tail.py`'s own module docstring, "Scope
EXCLUDED from this graph and why": "`bounds()` ... called from a DIFFERENT
method ... Folding it in would require changing the `step()`/`ttt_step()`
call boundary (a second, separate risk surface)").

For this shipped CONFIG (`cfg["bounds"]=True`, `cfg["ctr"]=True`,
`cfg["prior"]=True`), `ExperimentTTT.ttt_step` always takes the
`m.bounds(m.ctr_out, x, m.ctr_var, float(cfg["ctr_k_scale"]))` branch with
BOTH `var` and `k_scale` explicitly provided (never None), from window 0 of
every trajectory onward: `_apply_prior` seeds `self.M0`/`self.Saa` (hence
`ctr_out`/`ctr_var`) BEFORE window 0's own ctr section of `step()` ever runs,
in the SAME call (confirmed by reading `_apply_prior`/`step()` directly, not
assumed) -- the exact same fact `cuda_graph_megatail.py`'s own module
docstring already established and relied on ("`bounds()` never actually reads
`self.var` for this shipped CONFIG since `ctr_out` is non-None from window 0
onward"). With `var`/`k_scale` always given, `bounds()`'s OWN
`if var is None: ...` branch (which would read `x` and
`self.var`/`self._var()`) is DEAD CODE for this shipped CONFIG -- so this
graph captures exactly (and only) the live branch: a pure function of
(`mean`, `var`) plus the two FIXED (construction-time, never-reassigned)
buffers `self.log_a`/`self.k_tab` and a constant `k_scale` float baked in at
capture time (`ctr_k_scale` is never reassigned anywhere after `__init__`,
confirmed by grep). `x` is NOT an input to this graph -- unused on the live
branch.

No counters, no EMA, no `.double()` casts anywhere in `bounds()` -- the
lowest-risk of this task's two targets (no bit-exactness machinery needed
beyond verbatim transcription).

Correctness contract identical to every other graph in this package:
fixed-address static buffers, `.copy_()` in / `.clone()` out, no allocation
inside the replayed graph, try/except at construction only, never in the
timed path. `should_use()` has NO window gating (same reasoning as
`cuda_graph_mask.py`'s `GraphedMaskWrapper`): the call site in
`ExperimentTTT.ttt_step` already gates on `cfg["ctr"] and m.ctr_out is not
None` BEFORE invoking `forward()` (see `submission.py`'s `ttt_step`), so this
wrapper only needs `self.ok`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

STATIC_MEAN_SHAPE = (1, 20, 32, 64, 3)
STATIC_VAR_SHAPE = (1, 1, 32, 64, 3)


class GraphedBoundsWrapper(nn.Module):
    def __init__(self, model, warmup_iters: int = 3):
        super().__init__()
        self.model = model  # kept for eager fallback + reading fixed buffers each step
        self.ok = False
        self.graph = None
        self.capture_error = None
        self._k_scale = float(model.cfg["ctr_k_scale"])
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
        self._mean = torch.zeros(*STATIC_MEAN_SHAPE, device=device, dtype=dtype)
        self._var = torch.zeros(*STATIC_VAR_SHAPE, device=device, dtype=dtype)
        self._out_lower = torch.zeros(*STATIC_MEAN_SHAPE, device=device, dtype=dtype)
        self._out_upper = torch.zeros(*STATIC_MEAN_SHAPE, device=device, dtype=dtype)

    def _forward_body(self):
        # local import: avoid a module-level circular import (submission.py
        # imports THIS file); _SIGMA_GLOBAL is a plain python float constant,
        # never reassigned after module load.
        from submission import _SIGMA_GLOBAL

        m = self.model
        mean, var = self._mean, self._var
        std = var.expand_as(mean).sqrt()
        a = 2.0 * std * m.std_tgt / _SIGMA_GLOBAL
        pos = (a.clamp_min(1e-12).log() - m.log_a[0]) / (m.log_a[1] - m.log_a[0])
        k = m.k_tab[pos.round().long().clamp(0, len(m.k_tab) - 1)]
        half = self._k_scale * k * std
        self._out_lower.copy_(mean - half)
        self._out_upper.copy_(mean + half)

    def _capture(self, warmup_iters: int) -> None:
        m = self.model
        ref = m.std_tgt
        device, dtype = ref.device, ref.dtype
        self._alloc_buffers(device, dtype)
        # deterministic, finite dummy data for warm-up/capture -- values never
        # matter (every real replay overwrites every input buffer before
        # use), only shapes/dtypes/addresses do. var must stay positive
        # (sqrt/log downstream).
        self._mean.normal_()
        self._var.uniform_(1e-4, 1.0)

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

    def forward(self, mean: torch.Tensor, var: torch.Tensor):
        self._mean.copy_(mean)
        self._var.copy_(var)
        self.graph.replay()
        return self._out_lower.clone(), self._out_upper.clone()
