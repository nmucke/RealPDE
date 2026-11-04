#!/usr/bin/env python3
"""C4-R1 real-data fine-tune: causal in-context standardisation (RevIN-style)
of the `unet_local` backbone (`src/c4r1_models.py`).

Re-parameterises `scripts/c3a_finetune.py` (same recipe: init from the sim
pretrain, 8000 it, bs 8, lr 5e-4 warmup(3%)+cosine, stride-5 window-start
jitter, seed 0, AdamW wd 0, last iterate -- no val, an LCO fold's HELDOUT
files are never touched for selection) with two changes:

  1. Model: `c4r1_models.CtxNormUNet` wrapping a freshly-built `unet_local`,
     initialised from `--init` (default the shared sim-pretrain checkpoint,
     `data/checkpoints/c1d/unet_local_sim.pth`) with the stem's 4 new input
     columns zero-initialised (`CtxNormUNet.__init__`).
  2. Each training sample's (x, y) pair is standardised/un-standardised with
     the CAUSAL statistics of frames `[0, t+20)` -- computed once as a
     per-file prefix sum (`CS1_full`/`CS2_full`, CPU, input-normalised units)
     so indexing at `t0+20` for any window start on the stride-5 grid costs
     one gather, no leakage from target frames (design doc S2). The loss is
     MSE in the *original* (un-standardised, target-normalised) space:
     `MSE(y_hat_s * s + m, y)`.

Train-set constants for the 4 broadcast conditioning channels (`m_u, m_v,
log s_u, log s_v`, each independently rescaled to ~unit scale) are pooled
over the same stride-5 training grid and saved into the checkpoint so the
streaming evaluator/pipeline wrapper can reuse them unchanged.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/c4r1_finetune.py \\
        --train data/real_lco/fold0/train --seed 0 --mem-frac 0.12 \\
        --out data/checkpoints/c4r1/ctxnorm_fold0.pth
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
from load_baseline import load_checkpoint_state  # noqa: E402

T_WIN = F.T_WIN


def build_cumsum(frames_cpu: torch.Tensor, mean_uv: torch.Tensor, std_uv: torch.Tensor):
    """Per-file prefix sums of the input-normalised u,v frames.
    frames_cpu: (nf, Tmax, 32, 64, 2) raw physical. Returns CS1, CS2, each
    (nf, Tmax+1, 32, 64, 2), CS[:, t] = sum over frames[0:t)."""
    norm = (frames_cpu - mean_uv) / std_uv
    z = torch.zeros_like(norm[:, :1])
    cs1 = torch.cat([z, torch.cumsum(norm, dim=1)], dim=1)
    cs2 = torch.cat([z, torch.cumsum(norm * norm, dim=1)], dim=1)
    del norm
    return cs1, cs2


def gather_ms(cs1: torch.Tensor, cs2: torch.Tensor, fi: torch.Tensor, n: torch.Tensor):
    """fi, n: (B,) long CPU tensors (n = t0+20, the frame count so far,
    INCLUDING the current window). Returns causal (m, s), each (B,32,64,2)."""
    s1 = cs1[fi, n]
    s2 = cs2[fi, n]
    return CTX.causal_ms(s1, s2, n.view(-1, 1, 1, 1).float())


def pool_ctx_const(cs1: torch.Tensor, cs2: torch.Tensor, pool_fi: torch.Tensor,
                    pool_t0: torch.Tensor, chunk: int = 512) -> dict:
    """Pooled (mean, std) of (m_u, m_v, log s_u, log s_v) over the training
    grid (double-precision accumulation, chunked to bound peak memory)."""
    n_pts = pool_fi.numel()
    tot = torch.zeros(4, dtype=torch.float64)
    totsq = torch.zeros(4, dtype=torch.float64)
    count = 0
    for i in range(0, n_pts, chunk):
        fi = pool_fi[i:i + chunk]
        n = pool_t0[i:i + chunk] + T_WIN
        m, s = gather_ms(cs1, cs2, fi, n)          # (b,32,64,2)
        vals = torch.cat([m, torch.log(s)], dim=-1).double()   # (b,32,64,4)
        tot += vals.sum(dim=(0, 1, 2))
        totsq += (vals * vals).sum(dim=(0, 1, 2))
        count += vals.shape[0] * vals.shape[1] * vals.shape[2]
    mean = tot / count
    var = torch.clamp_min(totsq / count - mean * mean, 1e-8)
    std = torch.sqrt(var)
    mean = mean.float()
    std = std.float()
    return {"m_mean": mean[:2].tolist(), "m_std": std[:2].tolist(),
            "logs_mean": mean[2:].tolist(), "logs_std": std[2:].tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--init", default="data/checkpoints/c1d/unet_local_sim.pth")
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

    base = C1D.build("unet_local")
    init_path = args.init if Path(args.init).is_absolute() else str(REPO / args.init)
    state, meta = load_checkpoint_state(init_path)
    base.load_state_dict(state)
    model = CTX.CtxNormUNet(base).to(dev)
    n_params = CTX.param_count(model)
    print(f"init {args.init} ({meta.get('format')}); model params {n_params:,}", flush=True)

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
    run_mse = []
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
        mse = torch.mean((pred - y) ** 2)

        opt.zero_grad(set_to_none=True)
        mse.backward()
        opt.step()
        sched.step()
        run_mse.append(float(mse))

        last = it == args.iters - 1
        if (it + 1) % args.eval_every == 0 or last:
            tr = float(np.mean(run_mse[-args.eval_every:]))
            row = {"iter": it + 1, "train_mse": tr, "lr": sched.get_last_lr()[0],
                   "minutes": (time.time() - t_start) / 60}
            history.append(row)
            print(f"it {it + 1:6d}  train_mse {tr:.5f}  lr {row['lr']:.2e}  "
                  f"{row['minutes']:.1f} min", flush=True)

    payload = {"model_state_dict": model.state_dict(), "iteration": args.iters,
               "model": "c4r1_ctxnorm_unet_local", "n_params": n_params,
               "ctx_const": ctx_const, "history": history, "args": vars(args),
               "note": "c4r1_finetune.py (last)"}
    torch.save(payload, out)
    (out.parent / (out.stem + "_history.json")).write_text(
        json.dumps({"args": vars(args), "history": history, "n_params": n_params,
                     "ctx_const": ctx_const}, indent=2) + "\n")

    resource = {"wall_s": time.time() - t_start, "n_params": n_params}
    if dev == "cuda" and torch.cuda.is_available():
        resource["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        print(f"peak GPU: {resource['peak_mem_gib']:.2f} GiB allocated", flush=True)
    (out.parent / (out.stem + "_resource.json")).write_text(json.dumps(resource, indent=2) + "\n")
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
