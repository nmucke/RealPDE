#!/usr/bin/env python3
"""SP: the shipped S3 real fine-tune (`scripts/b1_finetune.py`, run UNEDITED via its `main()`)
initialised from an SP sim-pretrain snapshot instead of random init.

The only change: `b1_models.GenericCtxNormCore` is wrapped so that, right after the (random)
construction inside `b1_finetune.main()`, the pretrain snapshot's weights are loaded. The random
init is still drawn first, so the torch RNG stream and the window-sampling generator are identical
to the random-init baseline's (`data/checkpoints/b1_3090/b5_s3_<fold>.pth`, same script, seed 0):
the init is the only difference. `ctx_const` is recomputed from the fold's TRAIN files exactly as
in the shipped recipe. All other arguments are passed through to `b1_finetune.py`.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/sp_finetune.py \\
        --init data/checkpoints/sp/b5_sim_s3_it20k.pth -- \\
        --model b5_h192_blk6_rf11 --loss s3 --train data/real_lco/fold1/train --seed 0 \\
        --mem-frac 0.15 --out data/checkpoints/sp/ft_it20k_fold1.pth
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import b1_models as B1  # noqa: E402
import i1_losses as I1  # noqa: E402
import b1_finetune  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    fix_norm = None
    if argv[0] == "--fix-norm-from":          # CONTROL variant (declared after the first run)
        fix_norm = json.loads((REPO / argv[1]).read_text())["mixed_norm_consts"]
        argv = argv[2:]
    assert argv[0] == "--init" and argv[2] == "--", \
        "usage: [--fix-norm-from <baseline history json>] --init <ckpt> -- <b1_finetune args>"
    init_path = Path(argv[1]) if Path(argv[1]).is_absolute() else REPO / argv[1]
    rest = argv[3:]
    if fix_norm is not None:
        # S3 normalisers fixed to the random-init baseline's (same seed => same first batch and
        # same random init, i.e. "normalisers measured at the random init"), so the orig:std
        # effective weighting equals the baseline's and only the init differs.
        orig_loss = I1.MixedWelschLoss

        class FixedNormLoss(orig_loss):  # type: ignore[misc, valid-type]
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                assert abs(self.w_orig - fix_norm["w_orig"]) < 1e-12
                self.norm_orig = float(fix_norm["norm_orig"])
                self.norm_std = float(fix_norm["norm_std"])
                print(f"[sp_finetune] S3 normalisers FIXED to {fix_norm}", flush=True)

        I1.MixedWelschLoss = FixedNormLoss
    payload = torch.load(init_path, map_location="cpu", weights_only=False)
    init_state = payload["model_state_dict"]
    assert "train_real" not in json.dumps(payload.get("sim_files", []))

    orig_cls = B1.GenericCtxNormCore
    loaded = {"n": 0}

    class InitFromPretrain(orig_cls):  # type: ignore[misc, valid-type]
        def __init__(self, core):
            super().__init__(core)
            self.load_state_dict(init_state, strict=True)
            loaded["n"] += 1
            print(f"[sp_finetune] loaded init {init_path.name} "
                  f"(pretrain it {payload.get('iteration')})", flush=True)

    B1.GenericCtxNormCore = InitFromPretrain
    sys.argv = ["b1_finetune.py"] + rest
    rc = b1_finetune.main()
    assert loaded["n"] == 1, loaded
    # record the init in a sidecar (b1_finetune's payload note says "random init")
    out = rest[rest.index("--out") + 1]
    out = Path(out) if Path(out).is_absolute() else REPO / out
    (out.parent / (out.stem + "_init.json")).write_text(json.dumps(
        {"init": str(init_path.relative_to(REPO)), "pretrain_iteration": payload.get("iteration"),
         "fixed_norm": fix_norm,
         "b1_finetune_args": rest}, indent=2) + "\n")
    # also stamp the checkpoint itself
    ck = torch.load(out, map_location="cpu", weights_only=False)
    ck["note"] = f"sp_finetune.py: b1_finetune.py S3 recipe, init from {init_path.name}"
    ck["sp_init"] = str(init_path.relative_to(REPO))
    torch.save(ck, out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
