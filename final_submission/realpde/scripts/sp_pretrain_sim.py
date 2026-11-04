#!/usr/bin/env python3
"""SP: long sim pretrain of the b5 SimVP backbone (`b1_models` "b5_h192_blk6_rf11") inside the
SAME `GenericCtxNormCore` + ctx-norm + S3 mixed-Welsch training path as `scripts/b1_finetune.py`,
on ALL 100 `data/competition/train_sim` files. Recipe pre-declared in
`experiments/sp_sim_pretrain/notes.md`.

Sim handling (differences from real): sim is non-dimensional; u, v are rescaled by
U_inf = 1.263e-5 * re (organizers' PairDataset rule, as `scripts/c1d_pretrain_sim.py`). The sim p
field is DISCARDED and the p channel is set to 0 in physical units before normalisation, exactly
as the real loader pads it (`finetune_fno.Windows._pad`). Normalisation: official
`mean_std_real.pt`. Frames stay on CPU; each batch is gathered on CPU and moved to the GPU.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/sp_pretrain_sim.py \\
        --iters 20000 --snapshots 5000 10000 20000 --mem-frac 0.15 \\
        --out-prefix data/checkpoints/sp/b5_sim_s3
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import finetune_fno as F  # noqa: E402  (STATS path, T_WIN, SUB_S)
import c4r1_models as CTX  # noqa: E402
import b1_models as B1  # noqa: E402
import i1_losses as I1  # noqa: E402
from c4r1_finetune import build_cumsum, gather_ms, pool_ctx_const  # noqa: E402
from c7a_finetune import welsch_h_norm  # noqa: E402

T_WIN = F.T_WIN
SUB_S = 2
U_INF_COEF = 1.263e-5


def load_sim_uv(files):
    """(F, T_max, 32, 64, 2) float32 physical-unit u, v on CPU (p discarded), lengths."""
    arrs = []
    for f in files:
        with h5py.File(f, "r") as h:
            u_inf = U_INF_COEF * float(h["re"][()])
            u = np.asarray(h["u"][:, ::SUB_S, ::SUB_S], dtype=np.float32) * u_inf
            v = np.asarray(h["v"][:, ::SUB_S, ::SUB_S], dtype=np.float32) * u_inf
        a = np.stack([u, v], axis=-1)
        assert np.isfinite(a).all(), f
        arrs.append(a)
    lengths = [a.shape[0] for a in arrs]
    n = max(lengths)
    buf = np.zeros((len(arrs), n) + arrs[0].shape[1:], dtype=np.float32)
    for i, a in enumerate(arrs):
        buf[i, :a.shape[0]] = a
    if min(lengths) != n:
        print("short sim files:", [(files[i].stem, L) for i, L in enumerate(lengths) if L != n],
              flush=True)
    return torch.from_numpy(buf), lengths


class SimUVWindows:
    """CPU-resident u,v frames; `take` pads p = 0 (physical) then applies the official real
    normalisation -- the identical transform `finetune_fno.Windows.take` applies to real data."""

    def __init__(self, frames, lengths, device):
        mi, mt, si, st = torch.load(F.STATS, weights_only=False)
        one = torch.ones_like
        self.mean_in = mi.float().to(device)
        self.mean_tgt = mt.float().to(device)
        self.std_in = torch.where(si == 0, one(si), si).float().to(device)
        self.std_tgt = torch.where(st == 0, one(st), st).float().to(device)
        self.frames = frames
        self.lengths = list(lengths)
        self.device = device

    def take(self, fi, t0):
        idx = t0.view(-1, 1) + torch.arange(2 * T_WIN).view(1, -1)
        w = self.frames[fi.view(-1, 1), idx].to(self.device)          # (B,40,32,64,2)
        w = torch.cat([w, torch.zeros_like(w[..., :1])], dim=-1)      # p = 0 (masked)
        x = (w[:, :T_WIN] - self.mean_in) / self.std_in
        y = (w[:, T_WIN:] - self.mean_tgt) / self.std_tgt
        return x, y

    def grid(self, stride):
        fi, t0 = [], []
        for f, L in enumerate(self.lengths):
            starts = list(range(0, L - 2 * T_WIN + 1, stride))
            fi += [f] * len(starts)
            t0 += starts
        return torch.tensor(fi, dtype=torch.long), torch.tensor(t0, dtype=torch.long)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="b5_h192_blk6_rf11", choices=list(B1.MODEL_NAMES))
    ap.add_argument("--sim-dir", default="data/competition/train_sim")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--snapshots", nargs="*", type=int, default=[5000, 10000, 20000])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--h-std", type=float, default=I1.H_STD_DEFAULT)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-frac", type=float, default=0.15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--stop-after", type=int, default=None,
                    help="stop after this many iterations (LR schedule still spans --iters); "
                         "for the shipping chain: --iters 20000 --snapshots 10000 --stop-after 10000")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

    prefix = Path(args.out_prefix)
    prefix = prefix if prefix.is_absolute() else REPO / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    dev = args.device
    if dev == "cuda" and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
        torch.cuda.reset_peak_memory_stats()

    files = sorted((REPO / args.sim_dir).glob("*.h5"))
    assert all("train_real" not in str(f) for f in files)
    t_load = time.time()
    frames, lengths = load_sim_uv(files)
    tw = SimUVWindows(frames, lengths, dev)
    pool_fi, pool_t0 = tw.grid(args.stride)
    print(f"sim {len(files)} files, {pool_fi.numel()} windows (stride {args.stride}); "
          f"load {time.time() - t_load:.1f}s", flush=True)

    t_prep = time.time()
    mean_uv = tw.mean_in[:2].cpu()
    std_uv = tw.std_in[:2].cpu()
    cs1, cs2 = build_cumsum(frames, mean_uv, std_uv)
    ctx_const = pool_ctx_const(cs1, cs2, pool_fi, pool_t0)
    print(f"sim ctx_const {ctx_const} ({time.time() - t_prep:.1f}s)", flush=True)

    core = B1.build_core(args.model)
    gcore = B1.GenericCtxNormCore(core).to(dev)
    n_params = B1.param_count(gcore)
    print(f"model {args.model}; random init; params {n_params:,}; loss s3", flush=True)

    h_orig = welsch_h_norm(1.0)
    mixed = I1.MixedWelschLoss(0.75, 0.25, h_orig, args.h_std)

    opt = torch.optim.AdamW(gcore.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = max(1, int(args.warmup_frac * args.iters)) if args.warmup_frac > 0 else 0

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, args.iters - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gen = torch.Generator().manual_seed(args.seed)      # CPU: pool on CPU

    gcore.train()
    run_loss, history = [], []
    t_start = time.time()
    snaps = set(args.snapshots)
    for it in range(args.iters):
        j = torch.randint(0, pool_fi.numel(), (args.bs,), generator=gen)
        fi_b, t0_b = pool_fi[j], pool_t0[j]
        x, y = tw.take(fi_b, t0_b)
        m, s = gather_ms(cs1, cs2, fi_b, t0_b + T_WIN)
        m, s = m.to(dev), s.to(dev)
        xs = CTX.standardise_input(x, m, s)
        extra = CTX.build_extra_maps(m, s, ctx_const)
        y_hat_s = gcore(xs, extra)
        pred = CTX.unstandardise_output(y_hat_s, m, s)
        loss = mixed(pred, y, y_hat_s, m, s)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        lv = float(loss)
        if not math.isfinite(lv):
            raise RuntimeError(f"non-finite loss at it {it}")
        run_loss.append(lv)

        done = it + 1
        if done % args.eval_every == 0 or done == args.iters:
            tr = float(np.mean(run_loss[-args.eval_every:]))
            row = {"iter": done, "train_loss": tr, "lr": sched.get_last_lr()[0],
                   "minutes": (time.time() - t_start) / 60}
            history.append(row)
            print(f"it {done:6d}  train_s3 {tr:.5f}  lr {row['lr']:.2e}  {row['minutes']:.1f} min",
                  flush=True)
        if done in snaps:
            out = Path(f"{prefix}_it{done // 1000}k.pth")
            payload = {"model_state_dict": gcore.state_dict(), "iteration": done,
                       "model": args.model, "loss": "s3", "h_std": args.h_std, "h_orig": h_orig,
                       "mixed_norm_consts": {"norm_orig": mixed.norm_orig,
                                             "norm_std": mixed.norm_std},
                       "n_params": n_params, "ctx_const": ctx_const, "history": list(history),
                       "args": vars(args), "sim_files": [f.name for f in files],
                       "note": "sp_pretrain_sim.py: sim-only pretrain snapshot (p masked to 0)"}
            torch.save(payload, out)
            print(f"snapshot -> {out}", flush=True)
        if args.stop_after is not None and done >= args.stop_after:
            print(f"stop-after {args.stop_after} reached", flush=True)
            break

    res = {"wall_s": time.time() - t_start, "n_params": n_params, "history": history,
           "ctx_const": ctx_const, "args": vars(args)}
    if dev == "cuda" and torch.cuda.is_available():
        res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    Path(f"{prefix}_history.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
