#!/usr/bin/env python3
"""Fine-tune the baseline FNO3d on real PIV windows, with honest validation.

The organizers' `sim_real_fno.pth` was fine-tuned on *all* of `train_real`, so every
local split is memorised by it and no local ablation on it is trustworthy
(`notes/research_2026-09-15/transfer_gap.md`). This script rebuilds that fine-tune from
the sim-only checkpoint on a chosen set of files, so the files left out stay genuinely
unseen and the local subscores track the leaderboard.

    .venv/bin/python scripts/finetune_fno.py --out data/checkpoints/honest/fno_v1.pth \
        --iters 6000 --bs 8 --lr 3e-4 --stride 5

  --train / --val take directories or .h5 files (7575_0.h5 is always skipped: corrupt).
  Windows are the evaluator's: 20 input frames -> the next 20, normalised with the
  official real statistics. Validation is always on the evaluator's stride-20 grid and
  reports, besides the normalised MSE, the *raw frozen-forward* subscores
  (rel_l2 / tke / mvpe, `src/scoring.py`, pooled over windows exactly as
  `src/local_eval.py` does) -- the numbers the pipeline then builds on.

Training data is only ever the released `train_real` plus on-the-fly augmentation
(`--stride` window-start jitter, `--noise-std` input noise), per the competition rules.
Checkpoints are written in the kit's `model_state_dict` layout, so
`load_baseline.load_checkpoint_state` -- and hence `run_experiment.py --ckpt` -- reads
them directly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import scoring as official  # noqa: E402
from load_baseline import build_model, load_checkpoint_state  # noqa: E402
from training import tke_rel_l2  # noqa: E402

STATS = REPO / "starting_kit" / "example_data" / "mean_std_real.pt"
T_WIN = 20
SUB_S = 2
BAD = "7575_0"          # corrupt file, never to be used (CLAUDE.md)
# Fitted final-score weights (notes/2026-09-15_sps_tke_research.md s1), error metrics only.
SEL_W = {"rel_l2": 0.48, "tke": 0.07, "mvpe": 0.10}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def collect(paths) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p) if Path(p).is_absolute() else REPO / p
        out.extend(sorted(p.glob("*.h5")) if p.is_dir() else [p])
    files = [f for f in out if BAD not in f.name]
    if len(files) != len(out):
        print(f"skipped {len(out) - len(files)} corrupt file(s) ({BAD})", flush=True)
    seen, uniq = set(), []
    for f in files:                       # resolve symlinks: the splits are symlinked
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def load_frames(files, device):
    """((F, T_max, 32, 64, 2) float32 on `device`, per-file lengths).

    The p channel is identically zero on real data and normalises to zero, so it is
    not stored; windows are padded back to 3 channels when they are built. Three
    train_real files are short (282 / 492 / 607 frames instead of 868), so the stack
    is zero-padded and the per-file length is carried alongside -- truncating every
    trajectory to the shortest would throw away two thirds of the data.
    """
    arrs = []
    for f in files:
        with h5py.File(f, "r") as h:
            u = np.asarray(h["u"][:, ::SUB_S, ::SUB_S], dtype=np.float32)
            v = np.asarray(h["v"][:, ::SUB_S, ::SUB_S], dtype=np.float32)
        arrs.append(np.stack([u, v], axis=-1))
    lengths = [a.shape[0] for a in arrs]
    n = max(lengths)
    if min(lengths) != n:
        short = sorted(f"{files[i].stem}:{L}" for i, L in enumerate(lengths) if L != n)
        print(f"short trajectories (padded, sampled only where valid): {', '.join(short)}",
              flush=True)
    buf = np.zeros((len(arrs), n) + arrs[0].shape[1:], dtype=np.float32)
    for i, a in enumerate(arrs):
        buf[i, :a.shape[0]] = a
    return torch.from_numpy(buf).to(device), lengths


class Windows:
    """Window builder over an in-GPU frame tensor, with the official normalisation."""

    def __init__(self, frames: torch.Tensor, lengths, device: str):
        mi, mt, si, st = torch.load(STATS, weights_only=False)
        one = torch.ones_like
        self.mean_in = mi.float().to(device)
        self.mean_tgt = mt.float().to(device)
        self.std_in = torch.where(si == 0, one(si), si).float().to(device)
        self.std_tgt = torch.where(st == 0, one(st), st).float().to(device)
        self.frames = frames
        self.lengths = list(lengths)
        self.n_files = frames.shape[0]

    def _pad(self, w: torch.Tensor) -> torch.Tensor:
        return torch.cat([w, torch.zeros_like(w[..., :1])], dim=-1)

    def take(self, fi: torch.Tensor, t0: torch.Tensor):
        """Normalised (input, target) for the given (file, start) pairs."""
        idx = t0.view(-1, 1) + torch.arange(2 * T_WIN, device=t0.device).view(1, -1)
        w = self._pad(self.frames[fi.view(-1, 1), idx])        # (B, 40, 32, 64, 3)
        x = (w[:, :T_WIN] - self.mean_in) / self.std_in
        y = (w[:, T_WIN:] - self.mean_tgt) / self.std_tgt
        return x, y

    def denorm_tgt(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std_tgt + self.mean_tgt

    def grid(self, stride: int = T_WIN):
        """Every valid (file, start) on a fixed stride, as the evaluator walks them.

        Per file, because the short trajectories carry fewer windows; stride 20 is the
        evaluator's own grid, a smaller stride is the window-start jitter augmentation.
        """
        fi, t0 = [], []
        for f, L in enumerate(self.lengths):
            starts = list(range(0, L - 2 * T_WIN + 1, stride))
            fi += [f] * len(starts)
            t0 += starts
        dev = self.frames.device
        return (torch.tensor(fi, device=dev, dtype=torch.long),
                torch.tensor(t0, device=dev, dtype=torch.long))


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, win: Windows, files, bs: int = 8) -> dict:
    """Normalised MSE plus the raw frozen-forward subscores on the stride-20 grid."""
    was_training = model.training
    model.eval()
    fi, t0 = win.grid(T_WIN)
    preds, tgts, mses = [], [], []
    for i in range(0, fi.numel(), bs):
        x, y = win.take(fi[i:i + bs], t0[i:i + bs])
        p = model(x)
        mses.append(torch.mean((p - y) ** 2, dim=(1, 2, 3, 4)))
        preds.append(win.denorm_tgt(p).float().cpu().numpy())
        tgts.append(win.denorm_tgt(y).float().cpu().numpy())
    model.train(was_training)
    pred = np.concatenate(preds).astype(np.float32)
    tgt = np.concatenate(tgts).astype(np.float32)
    mse = torch.cat(mses)
    per = {
        "rel_l2": official.rel_l2_per_sample(pred, tgt, 2),
        "tke": official.tke_rel_l2_per_sample(pred, tgt, 2),
        "mvpe": official.mvpe_rel_l2_per_sample(pred, tgt),
    }
    out = {"mse": float(mse.mean())}
    for k, v in per.items():
        out[k] = official.score_error(float(np.mean(v)))
    out["sel"] = sum(SEL_W[k] * out[k] for k in SEL_W)
    fi_np = fi.cpu().numpy()
    out["per_file"] = {}
    for f in range(win.n_files):
        m = fi_np == f
        out["per_file"][files[f].stem] = {
            "mse": float(mse[torch.from_numpy(m).to(mse.device)].mean()),
            **{k: official.score_error(float(np.mean(v[m]))) for k, v in per.items()},
        }
    return out


# --------------------------------------------------------------------------- #
# weight averaging
# --------------------------------------------------------------------------- #
def average(args, out: Path, val_files) -> int:
    """Mean of several checkpoints' weights, written as one checkpoint.

    Fine-tunes from the same initialisation usually sit in one loss basin, so the
    average is a model in its own right -- and unlike an output-averaged ensemble it
    still costs exactly one forward pass, which is what the Time subscore charges for.
    """
    states = []
    for c in args.average:
        st, _ = load_checkpoint_state(c if os.path.isabs(c) else str(REPO / c))
        states.append(st)
    keys = set(states[0])
    if any(set(st) != keys for st in states):
        raise SystemExit("checkpoints have different parameter sets")
    avg = {}
    for k, v in states[0].items():
        if v.is_floating_point() or v.is_complex():
            acc = v.detach().clone().double() if v.is_floating_point() else v.detach().clone()
            for st in states[1:]:
                acc = acc + (st[k].double() if v.is_floating_point() else st[k])
            avg[k] = (acc / len(states)).to(v.dtype)
        else:
            avg[k] = v.clone()            # num_batches_tracked and friends
    torch.save({"model_state_dict": avg, "n_averaged": len(states),
                "sources": list(args.average), "args": vars(args),
                "note": "finetune_fno.py --average"}, out)
    print(f"averaged {len(states)} checkpoints -> {out}", flush=True)
    if val_files:
        model = build_model("fno", device=args.device)
        model.load_state_dict(avg)
        vw = Windows(*load_frames(val_files, args.device), device=args.device)
        ev = evaluate(model, vw, val_files, bs=args.bs)
        print(f"val: mse {ev['mse']:.5f}  rel_l2 {ev['rel_l2']:.2f}  tke {ev['tke']:.2f}  "
              f"mvpe {ev['mvpe']:.2f}  sel {ev['sel']:.3f}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--init", default="data/competition/baseline_checkpoints/sim_pretrain/sim_fno.pth",
                    help="checkpoint to start from (fp32 kit layout or fp16-packed)")
    ap.add_argument("--train", nargs="+", default=["data/real/train"],
                    help="directories and/or .h5 files to train on")
    ap.add_argument("--val", nargs="*", default=["data/real/val"],
                    help="held-out files for model selection ([] = select on train loss)")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--min-lr-frac", type=float, default=0.0,
                    help="cosine floor as a fraction of --lr")
    ap.add_argument("--tke-weight", type=float, default=0.0,
                    help="weight of the differentiable tke relative-L2 term")
    ap.add_argument("--clip", type=float, default=0.0, help="grad-norm clip (0 = off)")
    ap.add_argument("--stride", type=int, default=T_WIN,
                    help="window-start spacing for training samples: 20 = the "
                         "evaluator's grid, <20 = temporal jitter augmentation")
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="std of Gaussian noise added to the normalised input")
    ap.add_argument("--freeze-bn", action="store_true",
                    help="keep the 4 BatchNorm3d layers in eval mode (running stats "
                         "frozen at the init checkpoint's)")
    ap.add_argument("--swa-frac", type=float, default=0.0,
                    help="average the weights over the last fraction of the schedule "
                         "(written to <out>.swa); 0 = off")
    ap.add_argument("--swa-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--select", default="sel", choices=["sel", "mse"],
                    help="best-checkpoint criterion on val: weighted subscores or MSE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--average", nargs="*", default=[],
                    help="average these checkpoints' weights into --out and exit "
                         "(no training): the multi-seed / SWA merge, which must stay a "
                         "single forward pass to keep the Time subscore")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    val_files = collect(args.val) if args.val else []
    if args.average:
        return average(args, out, val_files)

    train_files = collect(args.train)
    overlap = {f.resolve() for f in train_files} & {f.resolve() for f in val_files}
    if overlap:
        print(f"WARNING: {len(overlap)} val file(s) are also in train -- validation is "
              "NOT honest for these", flush=True)
    dev = args.device
    tw = Windows(*load_frames(train_files, dev), device=dev)
    vw = Windows(*load_frames(val_files, dev), device=dev) if val_files else None
    pool_fi, pool_t0 = tw.grid(args.stride)     # every training window, uniformly sampled
    print(f"train {len(train_files)} files, {pool_fi.numel()} windows "
          f"(start stride {args.stride}); val {len(val_files)} files", flush=True)

    model, meta = None, None
    init = args.init if os.path.isabs(args.init) else str(REPO / args.init)
    model = build_model("fno", device=dev)
    state, meta = load_checkpoint_state(init)
    model.load_state_dict(state)
    print(f"init {args.init} ({meta.get('format')})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = max(1, int(args.warmup_frac * args.iters)) if args.warmup_frac > 0 else 0

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, args.iters - warm)
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    model.train()
    if args.freeze_bn:
        n_bn = 0
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
                n_bn += 1
        print(f"BatchNorm frozen in eval mode ({n_bn} layers)", flush=True)

    swa_state, swa_n = None, 0
    swa_start = int((1.0 - args.swa_frac) * args.iters) if args.swa_frac > 0 else args.iters + 1
    best = {"sel": -1e9, "mse": 1e9}
    best_it, history, run_mse = -1, [], []
    t_start = time.time()

    for it in range(args.iters):
        j = torch.randint(0, pool_fi.numel(), (args.bs,), generator=gen, device=dev)
        x, y = tw.take(pool_fi[j], pool_t0[j])
        if args.noise_std > 0:
            x = x + args.noise_std * torch.randn(x.shape, generator=gen, device=dev)
        pred = model(x)
        mse = torch.mean((pred - y) ** 2)
        loss = mse if args.tke_weight == 0 else mse + args.tke_weight * tke_rel_l2(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        run_mse.append(float(mse))

        if it >= swa_start and (it - swa_start) % args.swa_every == 0:
            # keep each tensor's own dtype: 99.8 % of an FNO is complex64 spectral
            # weights, and .float() on those would silently drop the imaginary part.
            sd = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state, swa_n = sd, 1
            else:
                swa_n += 1
                for k, v in sd.items():
                    if v.is_floating_point() or v.is_complex():
                        swa_state[k] = swa_state[k] + (v - swa_state[k]) / swa_n
                    else:
                        swa_state[k] = v

        last = it == args.iters - 1
        if (it + 1) % args.eval_every == 0 or last:
            tr = float(np.mean(run_mse[-args.eval_every:]))
            row = {"iter": it + 1, "train_mse": tr, "lr": sched.get_last_lr()[0],
                   "minutes": (time.time() - t_start) / 60}
            msg = (f"it {it + 1:6d}  train_mse {tr:.5f}  lr {row['lr']:.2e}  "
                   f"{row['minutes']:.1f} min")
            if vw is not None:
                ev = evaluate(model, vw, val_files, bs=args.bs)
                row.update({f"val_{k}": v for k, v in ev.items() if k != "per_file"})
                row["val_per_file"] = ev["per_file"]
                msg += (f" | val_mse {ev['mse']:.5f}  rel_l2 {ev['rel_l2']:.2f}  "
                        f"tke {ev['tke']:.2f}  mvpe {ev['mvpe']:.2f}  sel {ev['sel']:.3f}")
                better = (ev["sel"] > best["sel"] if args.select == "sel"
                          else ev["mse"] < best["mse"])
                if better:
                    best, best_it = ev, it + 1
                    torch.save({"model_state_dict": model.state_dict(), "iteration": it + 1,
                                "best_iteration": it + 1, "best_val_loss": ev["mse"],
                                "val": {k: v for k, v in ev.items() if k != "per_file"},
                                "val_per_file": ev["per_file"], "train_mse": tr,
                                "args": vars(args), "note": "finetune_fno.py"}, out)
                    msg += "  *"
            history.append(row)
            print(msg, flush=True)

    payload = {"model_state_dict": model.state_dict(), "iteration": args.iters,
               "history": history, "args": vars(args), "note": "finetune_fno.py (last)"}
    torch.save(payload, str(out) + ".last")
    if vw is None:      # no val: the last iterate is the deliverable
        torch.save({**payload, "note": "finetune_fno.py"}, out)
    if swa_state is not None:
        torch.save({"model_state_dict": swa_state, "iteration": args.iters,
                    "swa_points": swa_n, "args": vars(args),
                    "note": f"finetune_fno.py SWA over the last {args.swa_frac:g} of training"},
                   str(out) + ".swa")
        print(f"SWA over {swa_n} points -> {out}.swa", flush=True)
    (out.parent / (out.stem + "_history.json")).write_text(
        json.dumps({"args": vars(args), "history": history,
                    "best_iteration": best_it,
                    "best": {k: v for k, v in best.items() if k != "per_file"}}, indent=2) + "\n")
    if str(dev).startswith("cuda"):
        print(f"peak GPU: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
              f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved", flush=True)
    print(f"done in {(time.time() - t_start) / 60:.1f} min; best iter {best_it} -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
