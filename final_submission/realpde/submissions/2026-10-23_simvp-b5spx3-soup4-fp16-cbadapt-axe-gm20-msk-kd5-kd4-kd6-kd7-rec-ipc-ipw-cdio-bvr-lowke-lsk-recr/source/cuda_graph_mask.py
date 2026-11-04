"""c12t_launch (launch-cost pass, phase 5), task item (d): a tiny CUDA graph
capturing `GainMapImprovedModel._mask` (7.5 launches/step per
`experiments/c12t_launch/logs/stage_profile_base.log`, unconditional on
EVERY step including window 0 -- unlike every other graph in this
submission, this one has NO gating condition beyond "capture succeeded").

`_mask(x) = (x[0,...,:2]*std_in[:2]+mean_in[:2]).abs() < 1e-6).all(-1).all(0)`
is a pure function of `x` and the two FIXED (construction-time) buffers
`std_in`/`mean_in` -- no running state, no python-int counter, no branch
that differs between window 0 and any later window, so there is no
window-0 exclusion needed here (unlike `cuda_graph_ctxnorm.py`/
`cuda_graph_tail.py`/`cuda_graph_keflow.py`/`cuda_graph_pre.py`, all of
which skip window 0 for a structural-branch reason).

Correctness contract identical to every other graph in this package:
fixed-address static buffers, `.copy_()` in / `.clone()` out, no
allocation inside the replayed graph, try/except at construction only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

STATIC_X_SHAPE = (1, 20, 32, 64, 3)
STATIC_MASK_SHAPE = (32, 64)


class GraphedMaskWrapper(nn.Module):
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

    def _alloc_buffers(self, device, dtype):
        self._x = torch.zeros(*STATIC_X_SHAPE, device=device, dtype=dtype)
        self._out = torch.zeros(*STATIC_MASK_SHAPE, device=device, dtype=torch.bool)

    def _forward_body(self):
        self._out.copy_(self._body(self._x))

    def _body(self, x):
        """KD: the mask math on any (1,T,H,W,3) tensor, returning the (H,W)
        bool result (the whole-step graph uses it directly, no copies)."""
        m = self.model
        phys = x[0, ..., :2] * m.std_in[:2] + m.mean_in[:2]
        return (phys.abs() < 1e-6).all(-1).all(0)

    def _capture(self, warmup_iters: int) -> None:
        m = self.model
        ref = m.std_tgt
        device, dtype = ref.device, ref.dtype
        self._alloc_buffers(device, dtype)
        self._x.normal_()

        s_stream = torch.cuda.Stream()
        s_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_stream), torch.no_grad():
            for _ in range(warmup_iters):
                self._forward_body()
        torch.cuda.current_stream().wait_stream(s_stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self._forward_body()

    def should_use(self) -> bool:
        return self.ok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._x.copy_(x)
        self.graph.replay()
        return self._out.clone()
