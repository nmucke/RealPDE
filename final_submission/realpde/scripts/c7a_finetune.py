#!/usr/bin/env python3
"""C7-A real-data fine-tune of the ctx-norm UNet with a ROBUST loss, for use
as an interval-CENTRE model only (see experiments/c7a_robust_centre/notes.md).

Identical recipe to `scripts/c4r1_finetune.py` (same data pipeline, same
causal in-context standardisation, same optimiser/schedule/seed) -- imported,
not edited -- with only the loss swapped out for `src/c7a_losses.py`'s
`make_loss(...)`. Two initialisation modes:

  * default (R1, R2): build a fresh `CtxNormUNet` from the sim-pretrained
    plain `unet_local` (`--init`, 60-wide stem; the 4 new stem columns are
    zero-initialised by `CtxNormUNet.__init__`), exactly as `c4r1_finetune`
    does -- so R1/R2 are trained from the same starting point as R0 (the
    checkpoint at `--init`), only the loss differs.
  * `--warm-ctxnorm` (R3): load a FULL ctx-norm checkpoint (already
    64-wide-stem state, e.g. the R0 `data/checkpoints/c4r1/ctxnorm_fold0.pth`
    reference) directly into the wrapped model, for training stability on the
    Welsch/correntropy surrogate (flat gradient far from the target, so a
    good init matters more than for L1/Huber).

The Welsch loss's per-channel half-width `h` is fixed BEFORE training from
TRAIN-side quantities only (never the held-out split): `h_phys = 0.30 *
SIGMA_GLOBAL` for both u and v (the family brief's own quoted pipeline
half-width, not fit to any local data), converted to the target-normalised
space this loss operates in by `h_norm_c = h_phys / std_tgt_c`. `p` reuses
the u threshold.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/c7a_finetune.py \\
        --loss l1 --train data/real_lco/fold0/train --seed 0 --mem-frac 0.12 \\
        --out data/checkpoints/c7a/r1_l1_fold0.pth
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import finetune_fno as F  # noqa: E402
import c1d_models as C1D  # noqa: E402
import c4r1_models as CTX  # noqa: E402
import c7a_losses as L  # noqa: E402
from c4r1_finetune import build_cumsum, gather_ms, pool_ctx_const  # noqa: E402
from load_baseline import load_checkpoint_state  # noqa: E402

T_WIN = F.T_WIN
SIGMA_GLOBAL = 0.0563870259
STD_TGT_UV = (0.09681041, 0.01596364)   # starting_kit/example_data/mean_std_real.pt, physical


def welsch_h_norm(h_scale: float = 1.0) -> list[float]:
    """[h_u, h_v, h_p] in target-normalised units; see module docstring.
    `h_scale` (c7a S6 sensitivity sweep only, default 1.0 = unchanged) multiplies
    the TRAIN-side `h_phys = 0.30 * SIGMA_GLOBAL` before the per-channel divide."""
    h_phys = h_scale * 0.30 * SIGMA_GLOBAL
    h_u = h_phys / STD_TGT_UV[0]
    h_v = h_phys / STD_TGT_UV[1]
    return [h_u, h_v, h_u]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--loss", required=True, choices=["mse", "l1", "huber", "welsch"])
    ap.add_argument("--mse-warmup-iters", type=int, default=0,
                     help="use MSE for the first N iters (within --iters), then switch to "
                          "--loss for the rest -- a stability fallback for loss surfaces "
                          "with vanishing gradients far from the target (e.g. Welsch from "
                          "a cold/undomain-adapted init), still inside the same iter budget "
                          "and LR schedule.")
    ap.add_argument("--init", default="data/checkpoints/c1d/unet_local_sim.pth")
    ap.add_argument("--warm-ctxnorm", default=None,
                     help="load a full ctx-norm checkpoint instead of --init (R3)")
    ap.add_argument("--h-scale", type=float, default=1.0,
                     help="c7a S6 sensitivity sweep only: multiplies the Welsch h_phys "
                          "(default 1.0 = unchanged R3 recipe); no effect on other losses")
    ap.add_argument("--train", nargs="+", default=["data/real_lco/fold0/train"])
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-frac", type=float, default=0.12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    dev = args.device
    if dev == "cuda" and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
        torch.cuda.reset_peak_memory_stats()

    train_files = F.collect(args.train)
    tw = F.Windows(*F.load_frames(train_files, dev), device=dev)
    pool_fi, pool_t0 = tw.grid(args.stride)
    print(f"train {len(train_files)} files, {pool_fi.numel()} windows (start stride {args.stride})",
          flush=True)

    t_prep = time.time()
    frames_cpu = tw.frames.detach().to("cpu")
    mean_uv = tw.mean_in[:2].cpu()
    std_uv = tw.std_in[:2].cpu()
    cs1, cs2 = build_cumsum(frames_cpu, mean_uv, std_uv)
    del frames_cpu
    ctx_const = pool_ctx_const(cs1, cs2, pool_fi.cpu(), pool_t0.cpu())
    print(f"ctx_const {ctx_const} (prefix-sum + pooling: {time.time() - t_prep:.1f}s)", flush=True)

    if args.warm_ctxnorm:
        warm_path = args.warm_ctxnorm if Path(args.warm_ctxnorm).is_absolute() \
            else str(REPO / args.warm_ctxnorm)
        payload = torch.load(warm_path, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"] if "model_state_dict" in payload else payload
        base = C1D.build("unet_local")
        model = CTX.CtxNormUNet(base)          # extends the stem to 64-wide first
        model.load_state_dict(state)           # then loads the matching full state
        model = model.to(dev)
        n_params = CTX.param_count(model)
        print(f"warm-start from {args.warm_ctxnorm}; model params {n_params:,}", flush=True)
    else:
        base = C1D.build("unet_local")
        init_path = args.init if Path(args.init).is_absolute() else str(REPO / args.init)
        state, meta = load_checkpoint_state(init_path)
        base.load_state_dict(state)
        model = CTX.CtxNormUNet(base).to(dev)
        n_params = CTX.param_count(model)
        print(f"init {args.init} ({meta.get('format')}); model params {n_params:,}", flush=True)

    welsch_h = welsch_h_norm(args.h_scale) if args.loss == "welsch" else None
    if welsch_h is not None:
        print(f"welsch h_norm (u, v, p) = {welsch_h}"
              + (f"  [h_scale={args.h_scale}]" if args.h_scale != 1.0 else ""), flush=True)
    loss_fn = L.make_loss(args.loss, welsch_h=welsch_h)
    mse_fn = L.make_loss("mse") if args.mse_warmup_iters > 0 else None
    if mse_fn is not None:
        print(f"mse warmup: iters [0, {args.mse_warmup_iters}) use mse, "
              f"[{args.mse_warmup_iters}, {args.iters}) use {args.loss}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = max(1, int(args.warmup_frac * args.iters)) if args.warmup_frac > 0 else 0

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, args.iters - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    model.train()
    run_loss = []
    history = []
    t_start = time.time()

    for it in range(args.iters):
        j = torch.randint(0, pool_fi.numel(), (args.bs,), generator=gen, device=dev)
        fi_b, t0_b = pool_fi[j], pool_t0[j]
        x, y = tw.take(fi_b, t0_b)          # (B,20,32,64,3) each, target-normalised y

        n_b = (t0_b + T_WIN).cpu()
        m, s = gather_ms(cs1, cs2, fi_b.cpu(), n_b)
        m, s = m.to(dev), s.to(dev)

        xs = CTX.standardise_input(x, m, s)
        extra = CTX.build_extra_maps(m, s, ctx_const)
        y_hat_s = model(xs, extra)
        pred = CTX.unstandardise_output(y_hat_s, m, s)
        cur_fn = mse_fn if (mse_fn is not None and it < args.mse_warmup_iters) else loss_fn
        loss = cur_fn(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run_loss.append(float(loss))

        last = it == args.iters - 1
        if (it + 1) % args.eval_every == 0 or last:
            tr = float(np.mean(run_loss[-args.eval_every:]))
            row = {"iter": it + 1, "train_loss": tr, "lr": sched.get_last_lr()[0],
                   "minutes": (time.time() - t_start) / 60}
            history.append(row)
            print(f"it {it + 1:6d}  train_{args.loss} {tr:.5f}  lr {row['lr']:.2e}  "
                  f"{row['minutes']:.1f} min", flush=True)

    payload = {"model_state_dict": model.state_dict(), "iteration": args.iters,
               "model": "c7a_robust_ctxnorm_unet_local", "loss": args.loss,
               "welsch_h": welsch_h, "n_params": n_params,
               "ctx_const": ctx_const, "history": history, "args": vars(args),
               "note": "c7a_finetune.py (last)"}
    torch.save(payload, out)
    (out.parent / (out.stem + "_history.json")).write_text(
        json.dumps({"args": vars(args), "history": history, "n_params": n_params,
                     "ctx_const": ctx_const, "welsch_h": welsch_h}, indent=2) + "\n")

    resource = {"wall_s": time.time() - t_start, "n_params": n_params}
    if dev == "cuda" and torch.cuda.is_available():
        resource["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        print(f"peak GPU: {resource['peak_mem_gib']:.2f} GiB allocated", flush=True)
    (out.parent / (out.stem + "_resource.json")).write_text(json.dumps(resource, indent=2) + "\n")
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
