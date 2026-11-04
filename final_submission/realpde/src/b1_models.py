"""B1 -- speed/accuracy frontier of the SimVP backbone under the NEW eval-box
cost model (`t_eval = 2.26*CPU_local + 2.49*GPUbusy_local`, T1).

C14-A screened n1_simvp variants for ACCURACY only and never trained SMALLER
ones; this registry adds the cheap side of the frontier (narrower hidden,
fewer gSTA blocks, smaller receptive field, narrower encoder, and combinations
that roughly halve / quarter the backbone GPU time).

New file per the family rule. Imports `c11a_models` and `c14a_models`
UNEDITED -- every variant is a `c14a_models.SimVP1Stage` configuration, i.e.
exactly n1_simvp's topology with different hyperparameters (c14a's
`v0_baseline_replica` matches `c11a_models.N1SimVP` param-for-param).
"""
from __future__ import annotations

import torch.nn as nn

import c11a_models as C11A  # noqa: E402
import c14a_models as C14A  # noqa: E402

IN_CH, OUT_CH = C11A.IN_CH, C11A.OUT_CH
HH, WW = C11A.HH, C11A.WW

param_count = C11A.param_count
fp16_mb = C11A.fp16_mb
GenericCtxNormCore = C11A.GenericCtxNormCore
GenericCtxNormWrapper = C11A.GenericCtxNormWrapper

# Baseline: enc_dim 128 / hidden 256 / 8 blocks / k1 5, k2 7, d2 3 (RF 23).
VARIANTS: dict[str, dict] = {
    # reference (identical to c11a n1_simvp / c14a v0_baseline_replica)
    "b0_ref": dict(enc_dim=128, hidden=256, n_blocks=8),

    # single-factor cheap moves
    "b1_h192": dict(enc_dim=128, hidden=192, n_blocks=8),
    "b1_h128": dict(enc_dim=128, hidden=128, n_blocks=8),
    "b2_blk6": dict(enc_dim=128, hidden=256, n_blocks=6),
    "b2_blk4": dict(enc_dim=128, hidden=256, n_blocks=4),
    "b3_rf11": dict(enc_dim=128, hidden=256, n_blocks=8, k1=3, k2=5, d2=2),   # = c14a v3_kernel11
    "b4_enc96": dict(enc_dim=96, hidden=256, n_blocks=8),

    # combinations: roughly HALF the backbone GPU time
    "b5_h192_blk6": dict(enc_dim=128, hidden=192, n_blocks=6),
    "b5_h192_blk6_rf11": dict(enc_dim=128, hidden=192, n_blocks=6, k1=3, k2=5, d2=2),

    # combinations: roughly QUARTER the backbone GPU time
    "b6_h128_blk4_rf11": dict(enc_dim=128, hidden=128, n_blocks=4, k1=3, k2=5, d2=2),
    "b6_enc96_h128_blk6_rf11": dict(enc_dim=96, hidden=128, n_blocks=6, k1=3, k2=5, d2=2),
}

MODEL_NAMES = tuple(VARIANTS.keys())


def build_core(name: str, **overrides) -> nn.Module:
    if name not in VARIANTS:
        raise ValueError(f"unknown b1 variant {name!r}; choices: {sorted(VARIANTS)}")
    cfg = dict(VARIANTS[name])
    cfg.update(overrides)
    return C14A.SimVP1Stage(**cfg)
