#!/usr/bin/env python3
"""EN -- fit the (Re, AoA)-conditioned prior maps (prior.npz) for a K=2 b5
(`b1_models.VARIANTS["b5_h192_blk6_rf11"]`) seed ensemble backbone
(`src/en_models.EnsembleWrapper`) -- copy of `scripts/e2_fit_prior.py`
(unedited elsewhere, read only) with ONLY `e2_models`/`n1_simvp` swapped for
`en_models`/`b5_h192_blk6_rf11` (same causal, batch-1, per-file-reset walk,
same `mu`/`var`/`e_true`/`e_pred`/`uc` record format).

    .venv/bin/python scripts/en_fit_prior.py \\
        --files data/real_lco/fold0/train \\
        --ckpts data/checkpoints/b1_3090/b5_s3_fold0.pth \\
                data/checkpoints/b1_3090/b5_s3_fold0_seed1.pth \\
        --out experiments/en_b5_ensemble/fold0_k2/prior.npz --device cuda --mem-frac 0.15
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys

import h5py
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import en_models as EN  # noqa: E402

GMI_DIR = os.path.join(REPO, "experiments", "gain_map_improved")
BAD = "7575_0.h5"


def load_gmi_module():
    spec = importlib.util.spec_from_file_location("gmi", os.path.join(GMI_DIR, "submission.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect(path: str) -> list[str]:
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    files = sorted(glob.glob(os.path.join(p, "*.h5"))) if os.path.isdir(p) else [p]
    return [f for f in files if not f.endswith(BAD)]


def build_ensemble(ckpts, model_name: str, device: str) -> "EN.EnsembleWrapper":
    members = []
    for i, ckpt in enumerate(ckpts):
        ckpt_path = ckpt if os.path.isabs(ckpt) else os.path.join(REPO, ckpt)
        core, ctx_const = EN.load_member_core(model_name, ckpt_path, device)
        members.append((f"seed{i}", core, ctx_const))
    return EN.EnsembleWrapper(members, mean_space="destd").to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", required=True, help="directory of .h5 files (or symlinks)")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--model", default="b5_h192_blk6_rf11", choices=["b5_h192_blk6_rf11"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mem-frac", type=float, default=0.2)
    args = ap.parse_args()
    dev = args.device
    torch.set_num_threads(4)

    if dev == "cuda" and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)

    gmi = load_gmi_module()
    cfg = dict(gmi.CONFIG)
    mi, mt, si, st = torch.load(
        os.path.join(REPO, "starting_kit", "example_data", "mean_std_real.pt"), weights_only=False)
    si = torch.where(si == 0, torch.ones_like(si), si).float()
    st = torch.where(st == 0, torch.ones_like(st), st).float()
    mi, mt = mi.float(), mt.float()

    wrapper = build_ensemble(args.ckpts, args.model, dev)
    scanner = gmi.GainMapImprovedModel(torch.nn.Identity(), dev, cfg)   # for _scan_uc / masks only

    files = collect(args.files)
    out = {k: [] for k in ("re", "aoa", "mu", "var", "e_true", "e_pred", "uc", "name")}
    for fi, fp in enumerate(files):
        with h5py.File(fp, "r") as h:
            u = h["u"][:, ::2, ::2]
            v = h["v"][:, ::2, ::2]
            re, aoa = float(h["re"][()]), float(h["aoa"][()])
        d = np.stack([u, v, np.zeros_like(u)], -1).astype(np.float32)
        t0s = list(range(0, len(d) - 40 + 1, 20))
        X = torch.from_numpy(np.stack([d[t:t + 20] for t in t0s]))
        Y = torch.from_numpy(np.stack([d[t + 20:t + 40] for t in t0s]))
        xn = ((X - mi) / si).to(dev)

        wrapper.reset()
        with torch.no_grad():
            F = torch.cat([wrapper.step(xn[i:i + 1]) for i in range(len(xn))])
        F = (F * st.to(dev) + mt.to(dev)).cpu()
        inn = (Y - F)[..., :2]
        mu = inn.mean(0)
        var = ((inn - mu) ** 2).mean(dim=(0, 1))
        e_true = X[..., :2].var(dim=1, unbiased=False).mean(0)
        e_pred = F[..., :2].var(dim=1, unbiased=False).mean(0)
        ucs = []
        with torch.no_grad():
            for i in range(len(xn)):
                x = xn[i:i + 1]
                mask = scanner._mask(x)
                obstacle = scanner.shift.last_obstacle(mask)
                ucs.append(float(scanner._scan_uc(x - x.mean(dim=1, keepdim=True), obstacle, mask)))
        out["re"].append(re); out["aoa"].append(aoa); out["name"].append(os.path.basename(fp))
        out["mu"].append(mu.numpy().astype(np.float16)); out["var"].append(var.numpy())
        out["e_true"].append(e_true.numpy()); out["e_pred"].append(e_pred.numpy())
        out["uc"].append(float(np.mean(ucs)))
        print(f"[{fi + 1:2d}/{len(files)}] {os.path.basename(fp):14s} Re {re:6.0f} AoA {aoa:4.0f}  "
              f"windows {len(t0s)}  uc {np.mean(ucs):.2f} px/frame  |mu| {float(mu.abs().mean()):.4f}  "
              f"var {float(var.mean()):.2e}  E_true/E_pred {float(e_true.sum() / e_pred.sum()):.2f}",
              flush=True)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **{k: np.array(v) for k, v in out.items()})
    size = os.path.getsize(out_path) / 1e6
    print(f"wrote {out_path} ({size:.1f} MB, {len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
