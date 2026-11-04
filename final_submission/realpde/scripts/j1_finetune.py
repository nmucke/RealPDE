#!/usr/bin/env python3
"""J1: do the two small C11-A/I1 wins ADD? Train a `src/c11a_models.py`
backbone (default `n1_simvp`, the C11-A family winner) inside the SAME
`GenericCtxNormWrapper` streaming contract C11-A uses, with the I1 family's
"element-equal" mixed loss (`src/i1_losses.py`, default S3 = 0.75 orig-space
+ 0.25 standardised-space Welsch, its own family winner) instead of C11-A's
plain single-stage Welsch. Thin combination script only -- imports both
`c11a_models` (model builder, unedited) and `i1_losses` (loss classes,
unedited); no existing file is copy-edited. Random init (n1_simvp has no
sim-pretrain checkpoint, matching C11-A's own "from init" family rule), 8000
it, bs 8, lr 5e-4 warmup(3%)+cosine, stride-5 jitter, seed 0, AdamW wd 0, last
iterate, fold TRAIN only -- identical recipe to `scripts/c11a_finetune.py`
and `scripts/i1_finetune.py`.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/j1_finetune.py \\
        --model n1_simvp --loss s3 --train data/real_lco/fold0/train --seed 0 \\
        --mem-frac 0.15 --out data/checkpoints/j1/n1simvp_s3_fold0.pth
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
import c4r1_models as CTX  # noqa: E402
import c11a_models as C11A  # noqa: E402
import i1_losses as I1  # noqa: E402
from c4r1_finetune import build_cumsum, gather_ms, pool_ctx_const  # noqa: E402
from c7a_finetune import welsch_h_norm  # noqa: E402

T_WIN = F.T_WIN


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="n1_simvp", choices=list(C11A.MODEL_NAMES))
    ap.add_argument("--loss", default="s3", choices=["s1", "s2", "s3"])
    ap.add_argument("--h-std", type=float, default=I1.H_STD_DEFAULT)
    ap.add_argument("--train", nargs="+", default=["data/real_lco/fold0/train"])
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-frac", type=float, default=0.15)
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

    core = C11A.build_core(args.model)                 # RANDOM INIT, matches C11-A (no
                                                         # sim-pretrain checkpoint exists for
                                                         # this architecture)
    gcore = C11A.GenericCtxNormCore(core).to(dev)
    n_params = C11A.param_count(gcore)
    print(f"model {args.model}; random init; params {n_params:,}; loss {args.loss}", flush=True)

    h_orig = welsch_h_norm(1.0)   # shipped original-space h (u, v, p), unchanged recipe constant
    weights = {"s1": (0.0, 1.0), "s2": (0.5, 0.5), "s3": (0.75, 0.25)}[args.loss]
    mixed = None
    if args.loss in ("s2", "s3"):
        mixed = I1.MixedWelschLoss(*weights, h_orig, args.h_std)

    opt = torch.optim.AdamW(gcore.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = max(1, int(args.warmup_frac * args.iters)) if args.warmup_frac > 0 else 0

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, args.iters - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    gcore.train()
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
        y_hat_s = gcore(xs, extra)
        pred = CTX.unstandardise_output(y_hat_s, m, s)

        if args.loss == "s1":
            loss = I1.std_welsch_loss(y_hat_s, y, m, s, h_std=args.h_std)
        else:  # s2, s3
            loss = mixed(pred, y, y_hat_s, m, s)

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

    norm_consts = None
    if mixed is not None:
        norm_consts = {"norm_orig": mixed.norm_orig, "norm_std": mixed.norm_std,
                       "w_orig": mixed.w_orig, "w_std": mixed.w_std}

    payload = {"model_state_dict": gcore.state_dict(), "iteration": args.iters,
               "model": args.model, "loss": args.loss, "h_std": args.h_std,
               "h_orig": h_orig, "mixed_norm_consts": norm_consts, "n_params": n_params,
               "ctx_const": ctx_const, "history": history, "args": vars(args),
               "note": "j1_finetune.py (c11a model builder + i1 mixed loss, random init, "
                       "last iterate)"}
    torch.save(payload, out)
    (out.parent / (out.stem + "_history.json")).write_text(
        json.dumps({"args": vars(args), "history": history, "n_params": n_params,
                     "ctx_const": ctx_const, "h_orig": h_orig,
                     "mixed_norm_consts": norm_consts}, indent=2) + "\n")

    resource = {"wall_s": time.time() - t_start, "n_params": n_params}
    if dev == "cuda" and torch.cuda.is_available():
        resource["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        print(f"peak GPU: {resource['peak_mem_gib']:.2f} GiB allocated", flush=True)
    (out.parent / (out.stem + "_resource.json")).write_text(json.dumps(resource, indent=2) + "\n")
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
