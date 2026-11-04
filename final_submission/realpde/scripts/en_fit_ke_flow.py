#!/usr/bin/env python3
"""EN -- refit the ke_net_flow energy net (src/ke_net_flow.py) on fold0 TRAIN
for a K=2 b5 seed-ensemble backbone (`src/en_models.EnsembleWrapper`) --
copy of `scripts/e2_fit_ke_flow.py` (unedited elsewhere, read only) with ONLY
`e2_models`/`n1_simvp` swapped for `en_models`/`b5_h192_blk6_rf11`.

    .venv/bin/python scripts/en_fit_ke_flow.py --tag en_k2_fold0 \\
        --ckpts data/checkpoints/b1_3090/b5_s3_fold0.pth \\
                data/checkpoints/b1_3090/b5_s3_fold0_seed1.pth \\
        --files data/real_lco/fold0/train \\
        --cache data/cache/tke_flow_en_k2_fold0 \\
        --out experiments/en_b5_ensemble/fold0_k2/ke_net_flow.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
BAD = "7575_0.h5"
ALPHA_MIN, PRIOR_N = 0.05, 1.0     # must match src/tke_energy_fit.py


def load_tke_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tke_energy_fit_e2", os.path.join(SRC, "tke_energy_fit.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_cache_causal(cache: str, files_split, wrapper, mi, mt, si, st, device: str) -> None:
    os.makedirs(cache, exist_ok=True)
    todo = [(p, s) for p, s in files_split
            if not os.path.isfile(os.path.join(cache, os.path.basename(p)[:-3] + ".npz"))]
    if not todo:
        print(f"cache complete ({len(files_split)} files) in {cache}", flush=True)
        return
    var = lambda a, d=1: a.var(dim=d, unbiased=False)          # noqa: E731
    for n, (fp, split) in enumerate(todo):
        with h5py.File(fp, "r") as h:
            u, v = h["u"][:, ::2, ::2], h["v"][:, ::2, ::2]
            re, aoa = float(h["re"][()]), float(h["aoa"][()])
        d = np.stack([u, v, np.zeros_like(u)], -1).astype(np.float32)
        t0s = list(range(0, len(d) - 40 + 1, 20))
        Xw = torch.from_numpy(np.stack([d[t:t + 20] for t in t0s]))
        Y = torch.from_numpy(np.stack([d[t + 20:t + 40] for t in t0s]))
        xn = (Xw - mi) / si
        yn = ((Y - mt) / st)[..., :2]
        wrapper.reset()
        with torch.no_grad():
            F = torch.cat([wrapper.step(xn[i:i + 1].to(device)).cpu() for i in range(len(xn))])
        ffl = (F - F.mean(1, keepdim=True))[..., :2]
        xw = xn[..., :2]
        rec = dict(
            xw=xw.numpy().astype(np.float16), ffl=ffl.numpy().astype(np.float16),
            ex=var(xw).numpy(), ef=var(ffl).numpy(), ey=var(yn).numpy(),
            ex1=var(xw[:, :10]).numpy(), ex2=var(xw[:, 10:]).numpy(),
            ef1=var(ffl[:, :10]).numpy(), ef2=var(ffl[:, 10:]).numpy(),
            mask=(Xw[0, ..., :2].abs() < 1e-6).all(-1).all(0).numpy(),
            re=np.float32(re), aoa=np.float32(aoa), split=np.int32(split))
        for src, dst in (("ex", "runx"), ("ef", "runf")):
            e = rec[src]
            r = np.empty_like(e)
            r[0] = e[0]
            for i in range(1, len(e)):
                r[i] = r[i - 1] + max(1.0 / (i + PRIOR_N), ALPHA_MIN) * (e[i] - r[i - 1])
            rec[dst] = r
        np.savez(os.path.join(cache, os.path.basename(fp)[:-3] + ".npz"), **rec)
        print(f"[{n + 1:2d}/{len(todo)}] {os.path.basename(fp):14s} Re {re:6.0f} "
              f"AoA {aoa:4.0f}  {len(t0s)} windows (causal, batch 1)", flush=True)


def build_ensemble(ckpts, model_name: str, device: str):
    import en_models as EN  # noqa: E402
    members = []
    for i, ckpt in enumerate(ckpts):
        ckpt_path = ckpt if os.path.isabs(ckpt) else os.path.join(REPO, ckpt)
        core, ctx_const = EN.load_member_core(model_name, ckpt_path, device)
        members.append((f"seed{i}", core, ctx_const))
    return EN.EnsembleWrapper(members, mean_space="destd").to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", required=True, help="directory of TRAIN .h5 files (fold-only)")
    ap.add_argument("--val-every", type=int, default=7)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--model", default="b5_h192_blk6_rf11", choices=["b5_h192_blk6_rf11"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--mem-frac", type=float, default=0.2)
    args, extra = ap.parse_known_args()

    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)

    sys.path.insert(0, SRC)

    files = sorted(f for f in glob.glob(os.path.join(args.files, "*.h5")) if not f.endswith(BAD))
    if not files:
        raise SystemExit(f"no .h5 files under {args.files}")
    fold_split = [(f, 1 if i % args.val_every == 0 else 0) for i, f in enumerate(files)]
    n_val = sum(1 for _, s in fold_split if s == 1)
    print(f"[e2_fit_ke_flow] {len(files)} fold-train files -> internal "
          f"{len(files) - n_val} train / {n_val} val (every {args.val_every}th)", flush=True)

    wrapper = build_ensemble(args.ckpts, args.model, dev)

    mi, mt, si, st = torch.load(
        os.path.join(REPO, "starting_kit", "example_data", "mean_std_real.pt"), weights_only=False)
    si = torch.where(si == 0, torch.ones_like(si), si).float()
    st = torch.where(st == 0, torch.ones_like(st), st).float()
    mi, mt = mi.float(), mt.float()

    cache_path = args.cache if os.path.isabs(args.cache) else os.path.join(REPO, args.cache)
    build_cache_causal(cache_path, fold_split, wrapper, mi, mt, si, st, dev)
    del wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mod = load_tke_module()
    mod.split_files = lambda: fold_split
    dummy_ckpt = os.path.join(REPO, "data", "checkpoints", "c1d", "unet_local_fold0.pth")
    sys.argv = ["tke_energy_fit.py", "--cache", cache_path, "--ckpt", dummy_ckpt, *extra]
    mod.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
