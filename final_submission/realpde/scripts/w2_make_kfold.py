#!/usr/bin/env python3
"""W2 -- K-fold cross-fit layout INSIDE an LCO fold's TRAIN files, for
out-of-sample residual generation (conditional-width net training data).

For fold F with train files T_F (data/real_lco/<F>/train), split T_F into K
groups round-robin over the (Re, AoA)-sorted file list (N1's stratification,
experiments/n1_crossfit/splits_fold0.json), and create, per group g:

  experiments/w2_condwidth_v2/kfold/<F>/g<g>/train/    symlinks: T_F minus group g
  experiments/w2_condwidth_v2/kfold/<F>/g<g>/heldout/  symlinks: group g
  experiments/w2_condwidth_v2/kfold/<F>/g<g>/pipe/submission.py
        = experiments/d2_bundle/fold0_c/submission.py with CONFIG["unet_ckpt"]
          repointed to data/checkpoints/w2/n1simvp_s3_<F>_g<g>.pth and a
          W2 aux-map recorder (env W2_AUX_OUT) that stores per-step CAUSAL
          internal maps (energy gain gamma, forecast/input energies, running
          energies, mu stats, centre gain) for the H4 feature study.
          prior.npz / ke_net_flow.pt are fitted later next to it.

Never touches data/real_lco, experiments/d2_bundle or the fold's held-out
files (those are only ever streamed by R0's reference pipelines).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BAD = "7575_0.h5"
REF_SUB = REPO / "experiments/d2_bundle/fold0_c/submission.py"

AUX_HOOK = r'''

# ---- W2 aux-map recorder (research only; env W2_AUX_OUT) --------------------
_W2_AUX_OUT = os.environ.get("W2_AUX_OUT")
_W2_REC = []


def _w2_map(t):
    """(1,*,H,W,3) tensor or scalar -> (H,W,2) float16 numpy (lead-pooled maps)."""
    import numpy as _np
    if t is None:
        return None
    if not torch.is_tensor(t):
        return _np.full((32, 64, 2), float(t), dtype=_np.float16)
    t = t.detach().float()
    while t.dim() > 3:
        t = t.mean(dim=0) if t.shape[0] > 1 else t[0]
    if t.dim() == 2:
        t = t[..., None].expand(-1, -1, 3)
    return t[..., :2].cpu().numpy().astype(_np.float16)


def _w2_record(m):
    aux = getattr(m, "w2_aux", None) or {}
    rec = {}
    for k in ("gamma", "e_f", "kx", "e_true", "e_pred", "g", "g2b"):
        rec[k] = _w2_map(aux.get(k))
    mu = aux.get("mu")
    if mu is not None:
        rec["mu_mean"] = _w2_map(mu.mean(dim=1, keepdim=True))
        rec["mu_std"] = _w2_map(mu.std(dim=1, keepdim=True, unbiased=False))
    _W2_REC.append(rec)


def _w2_save():
    import numpy as _np
    if not _W2_REC:
        return
    keys = sorted({k for r in _W2_REC for k, v in r.items() if v is not None})
    out = {}
    for k in keys:
        ref = next(r[k] for r in _W2_REC if r.get(k) is not None)
        out[k] = _np.stack([r[k] if r.get(k) is not None else _np.zeros_like(ref) for r in _W2_REC])
        out[k + "_present"] = _np.array([r.get(k) is not None for r in _W2_REC])
    os.makedirs(os.path.dirname(_W2_AUX_OUT) or ".", exist_ok=True)
    _np.savez(_W2_AUX_OUT, **out)


if _W2_AUX_OUT:
    import atexit as _atexit
    _atexit.register(_w2_save)
'''


def patch_submission(src: str, ckpt_rel: str) -> str:
    s = src
    # 1. repoint the backbone checkpoint
    pat = r'"unet_ckpt":\s*"[^"]+"'
    assert re.search(pat, s), "unet_ckpt not found"
    s = re.sub(pat, f'"unet_ckpt": "{ckpt_rel}"', s, count=1)
    # 2. stash causal internal maps inside step()
    anchor = "                gamma = self._gamma_ext(x, fl_f, e_f, kx, mask)\n"
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor +
                  "            self.w2_aux = {\"gamma\": gamma, \"e_f\": e_f, \"kx\": kx, "
                  "\"e_true\": self.e_true, \"e_pred\": self.e_pred, \"g2b\": self.g2b}\n")
    anchor2 = "        out = const + fluc\n        if mu is not None:\n            out = out + mu\n"
    assert s.count(anchor2) == 1
    s = s.replace(anchor2, anchor2 + "        if hasattr(self, 'w2_aux'):\n            self.w2_aux['mu'] = mu\n")
    anchor3 = "self.ctr_out = ctr.detach()\n"
    assert s.count(anchor3) == 2
    s = s.replace(anchor3, anchor3.rstrip("\n") + "; self.w2_aux['g'] = g\n")
    # reset the stash at trajectory start so window 0 never carries stale maps
    anchor4 = "        self.prev_const = self.prev_basis = self.ctr_out = self.ctr_var = None\n"
    assert s.count(anchor4) == 1
    s = s.replace(anchor4, anchor4 + "        self.w2_aux = {}\n")
    # 3. record after every step (ttt_step)
    anchor5 = "        out, loss = self.model.step(x, y)\n"
    assert s.count(anchor5) == 1
    s = s.replace(anchor5, anchor5 + "        if _W2_AUX_OUT:\n            _w2_record(self.model)\n")
    return s + AUX_HOOK


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--out-root", default="experiments/w2_condwidth_v2/kfold")
    ap.add_argument("--ckpt-dir", default="data/checkpoints/w2")
    ap.add_argument("--train-dir", default=None,
                    help="dir of .h5 files to K-fold (default data/real_lco/<fold>/train); e.g. a dir of all 81 usable files for the shipping fit, with --fold all81")
    args = ap.parse_args()

    train_dir = REPO / (args.train_dir or f"data/real_lco/{args.fold}/train")
    files = sorted(f.name for f in train_dir.glob("*.h5") if f.name != BAD)
    assert BAD not in files

    def key(fn):
        re_, aoa = fn[:-3].split("_")
        return (int(re_), int(aoa))

    files.sort(key=key)
    parts = [[] for _ in range(args.K)]
    for i, fn in enumerate(files):
        parts[i % args.K].append(fn)
    root = REPO / args.out_root / args.fold
    root.mkdir(parents=True, exist_ok=True)
    split = {"fold": args.fold, "K": args.K, "n_files": len(files), "parts": parts,
             "train_dir": str(train_dir.relative_to(REPO))}
    (root / "split.json").write_text(json.dumps(split, indent=1))

    src = REF_SUB.read_text()
    for g in range(args.K):
        gd = root / f"g{g}"
        for sub in ("train", "heldout"):
            d = gd / sub
            if d.exists():
                for p in d.iterdir():
                    p.unlink()
            d.mkdir(parents=True, exist_ok=True)
        for fn in files:
            dst = gd / ("heldout" if fn in parts[g] else "train") / fn
            dst.symlink_to((train_dir / fn).resolve())
        pipe = gd / "pipe"
        pipe.mkdir(exist_ok=True)
        ckpt_rel = str(REPO / f"{args.ckpt_dir}/n1simvp_s3_{args.fold}_g{g}.pth")  # absolute: the copy sits deeper than d2_bundle, so its _REPO resolves wrongly
        (pipe / "submission.py").write_text(patch_submission(src, ckpt_rel))
        print(f"[{args.fold} g{g}] heldout {len(parts[g])} files, train {len(files) - len(parts[g])}, "
              f"ckpt {ckpt_rel}")
    print("wrote", root / "split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
