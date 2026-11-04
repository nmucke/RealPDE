"""EN -- seed ensemble of the B1 b5 backbone (`b1_models.build_core(
"b5_h192_blk6_rf11")`, i.e. `c14a_models.SimVP1Stage(enc_dim=128, hidden=192,
n_blocks=6, k1=3, k2=5, d2=2)`), wrapped by the SAME homogeneous
`GenericCtxNormCore`/causal-stats-sharing contract as E2's `EnsembleWrapper`
(`src/e2_models.py`, unedited, read only) -- this file is that class verbatim
with ONLY the backbone module swapped (`b1_models` instead of `c11a_models`)
so `load_member_core`/`EnsembleWrapper` build b5 members instead of n1_simvp
ones. Used for the EAGER prior/ke_net_flow refit path (`scripts/en_fit_prior.py`,
`scripts/en_fit_ke_flow.py`) -- NOT the packaged/graphed accuracy-scoring path
(that uses `src/en_packed.py`'s `SeqEnsembleCore` inside the P8 stack's
`ship_ctxnorm_models.CtxNormWrapper`, which shares causal state externally).

New file per the family rule. Imports `c4r1_models` (`causal_ms`,
`standardise_input`, `unstandardise_output`, `build_extra_maps`) and
`b1_models` (`build_core`, `GenericCtxNormCore` = `c11a_models.GenericCtxNormCore`)
unedited.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

import c4r1_models as CTX      # noqa: E402  causal_ms, standardise_input, unstandardise_output, build_extra_maps
import b1_models as B1         # noqa: E402  build_core, GenericCtxNormCore


def load_member_core(model_name: str, ckpt_path: str, device: str) -> Tuple["B1.GenericCtxNormCore", dict]:
    """One member's `(GenericCtxNormCore, ctx_const)`, eval mode, frozen --
    identical loading logic to `experiments/b1_3090/pipe_fold0/submission.py`'s
    `get_ttt_model`."""
    core = B1.build_core(model_name)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    gcore = B1.GenericCtxNormCore(core)
    gcore.load_state_dict(state)
    gcore = gcore.to(device).eval()
    for p in gcore.parameters():
        p.requires_grad_(False)
    return gcore, payload["ctx_const"]


class EnsembleWrapper(nn.Module):
    """Equal-weight seed ensemble of `K` `GenericCtxNormCore` members sharing
    ONE causal-statistics accumulator and ONE `ctx_const` -- verbatim copy of
    `e2_models.EnsembleWrapper`'s design (b5 members instead of n1_simvp)."""

    def __init__(self, members: List[Tuple[str, nn.Module, Dict[str, Any]]],
                 mean_space: str = "destd"):
        super().__init__()
        assert members, "EnsembleWrapper needs at least one member"
        assert mean_space in ("destd", "std"), mean_space
        self.names = [m[0] for m in members]
        self.cores = nn.ModuleList([m[1] for m in members])
        self.mean_space = mean_space
        ref = members[0][2]
        for name, _, cc in members[1:]:
            assert set(cc.keys()) == set(ref.keys()), (name, cc.keys(), ref.keys())
            for k, v in ref.items():
                w = cc[k]
                if torch.is_tensor(v):
                    assert torch.allclose(v.float(), w.float(), atol=1e-5, rtol=1e-4), \
                        f"ctx_const[{k!r}] differs for member {name!r} vs {members[0][0]!r}"
                else:
                    assert v == w, f"ctx_const[{k!r}] differs for member {name!r}"
        self.ctx_const = ref
        self.reset()

    def reset(self) -> None:
        self.S1 = None
        self.S2 = None
        self.n = 0

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        uv = x[..., :2]
        window_sum1 = uv.sum(dim=1)
        window_sum2 = (uv * uv).sum(dim=1)
        if self.S1 is None:
            self.S1 = window_sum1.detach().clone()
            self.S2 = window_sum2.detach().clone()
        else:
            self.S1 = self.S1 + window_sum1.detach()
            self.S2 = self.S2 + window_sum2.detach()
        self.n += x.shape[1]

        m, s = CTX.causal_ms(self.S1, self.S2, self.n)
        xs = CTX.standardise_input(x, m, s)
        extra = CTX.build_extra_maps(m, s, self.ctx_const)

        y_list = [core(xs, extra) for core in self.cores]
        if self.mean_space == "std":
            y_mean = torch.stack(y_list, dim=0).mean(dim=0)
            return CTX.unstandardise_output(y_mean, m, s)
        preds = [CTX.unstandardise_output(y_s, m, s) for y_s in y_list]
        return torch.stack(preds, dim=0).mean(dim=0)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
