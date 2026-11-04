#!/usr/bin/env python3
"""KE Stage 0 -- seed-ensembled `ke_net_flow` energy net (WEB-2 idea 6: variance
reduction, not capacity). Brief: `runs/briefs/KE_seed_ensemble.md`.

Reuses the EXISTING fold0-TRAIN cache built by `scripts/b1_fit_ke_flow.py --model
b5_h192_blk6_rf11` (`data/cache/tke_flow_b1s_b5_fold0`, 61 fold0-TRAIN files,
internal train/val split = every 7th file, matching `experiments/b1_3090/notes.md`
section 8's single-seed reference: rule val e 0.5294, net BEST val e 0.4447).
Nothing here rebuilds the cache or retrains the backbone; only NEW files are
touched (`data/checkpoints/ke_seed_ens/`, `experiments/ke_seed_ens/`).

Trains K independent `KeFlowNet` seeds with the SAME recipe as
`src/tke_energy_fit.py`'s `main()` (same cache, same internal train/val split --
selection stays on that split, never on the fold's held-out files), then reports:
  (a) each seed's own BEST val e (picked on its own val split, as usual),
  (b) the val e of the K-seed ENSEMBLE: the K best nets' raw (pre-clamp) log-space
      corrections averaged, then ONE `apply_correction` call -- i.e. averaging in
      the log-residual space the net predicts, matching how the eventual
      grouped-conv packed net (K parallel branches feeding one average) would
      behave.

KILL (per brief): ensemble val e must be <= 0.99 * mean(single-seed val e).

Usage:
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/kes_fit_ensemble.py \\
        --cache data/cache/tke_flow_b1s_b5_fold0 --seeds 0,1,2,3 --mem-frac 0.15 \\
        --out-dir experiments/ke_seed_ens/stage0
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
BAD = "7575_0.h5"


def load_tke_module():
    """Fresh, unmodified import of `src/tke_energy_fit.py` (never edited)."""
    spec = importlib.util.spec_from_file_location(
        "tke_energy_fit_kes", os.path.join(SRC, "tke_energy_fit.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def fold_train_split(files_dir: str, val_every: int = 7):
    """EXACT reproduction of `b1_fit_ke_flow.py`'s split that built the cache:
    sorted glob of `<files_dir>/*.h5`, corrupt file excluded, every `val_every`-th
    file -> internal val. `files_dir` is e.g. `data/real_lco/fold1/train`,
    `data/real_lco/foldR/train`, or `data/real` (Stage 2's all-81 shipping note)."""
    files = sorted(f for f in glob.glob(os.path.join(REPO, files_dir, "*.h5"))
                   if not f.endswith(BAD))
    if not files:
        raise SystemExit(f"no files found under {files_dir}")
    return [(f, 1 if i % val_every == 0 else 0) for i, f in enumerate(files)]


def train_one_seed(mod, K, d, tr, va, cfg, seed: int, epochs: int, batch_size: int,
                   lr: float, wd: float, flip: bool, dev: str, tag: str):
    torch.manual_seed(seed)
    net = K.KeFlowNet(cfg["in_ch"], cfg["width"], cfg["wide"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    base = mod.evaluate(net, d, va, cfg, dev)
    print(f"[{tag}] seed {seed} zero-init rule val e {base['e_rule']:.4f} "
          f"(tke {mod.score(base['e_rule']):.2f})", flush=True)

    best = (float("inf"), None, None)
    g = torch.Generator().manual_seed(seed)
    t_start = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=g)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            fl = (torch.rand(len(idx), generator=g) < 0.5) if flip else None
            feat, log_rule, mask, ey = mod.batch_features(d, idx, cfg, dev, fl)
            e_p = K.apply_correction(net(feat), log_rule, mask)
            loss = K.ke_rel_l2(e_p, ey).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        net.eval()
        v = mod.evaluate(net, d, va, cfg, dev)
        star = ""
        if v["e_net"] < best[0]:
            best = (v["e_net"], {k: v[k] for k in v},
                    {k: t.detach().cpu().clone() for k, t in net.state_dict().items()})
            star = " *"
        if ep % 20 == 0 or star or ep == 1 or ep == epochs:
            print(f"[{tag}] seed {seed} ep {ep:3d} train {tot / nb:.4f} "
                  f"val {v['e_net']:.4f} (tke {mod.score(v['e_net']):.2f}){star}", flush=True)
    dt = time.time() - t_start
    print(f"[{tag}] seed {seed} DONE {dt:.0f}s BEST val e {best[0]:.4f} "
          f"(tke {mod.score(best[0]):.2f})", flush=True)
    return best  # (val_e, val_metrics, state_dict)


@torch.no_grad()
def evaluate_ensemble(mod, K, nets, d, idx, cfg, dev, bs: int = 128):
    """Average the K nets' RAW (pre-clamp, pre-scale) log-corrections, then apply
    ONE `apply_correction` -- log-space ensembling of the residual each net predicts."""
    errs = []
    for i in range(0, len(idx), bs):
        feat, log_rule, mask, ey = mod.batch_features(d, idx[i:i + bs], cfg, dev)
        corr_mean = torch.stack([net(feat) for net in nets], 0).mean(0)
        e_ens = K.apply_correction(corr_mean, log_rule, mask)
        errs.append(K.ke_rel_l2(e_ens, ey))
    return float(torch.cat(errs).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default="data/cache/tke_flow_b1s_b5_fold0")
    ap.add_argument("--files-dir", default="data/real_lco/fold0/train",
                    help="directory the cache was built from (fold TRAIN or all-81 dir)")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--flip", action="store_true", default=True)
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.add_argument("--val-every", type=int, default=7)
    ap.add_argument("--mem-frac", type=float, default=0.15)
    ap.add_argument("--out-dir", default="experiments/ke_seed_ens/stage0")
    ap.add_argument("--ckpt-dir", default="data/checkpoints/ke_seed_ens")
    ap.add_argument("--tag", default="kes_stage0")
    args = ap.parse_args()

    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir)
    ckpt_dir = args.ckpt_dir if os.path.isabs(args.ckpt_dir) else os.path.join(REPO, args.ckpt_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    mod = load_tke_module()
    K = mod.K
    cache_path = args.cache if os.path.isabs(args.cache) else os.path.join(REPO, args.cache)

    split = fold_train_split(args.files_dir, args.val_every)
    mod.split_files = lambda: split
    n_val = sum(1 for _, s in split if s == 1)
    print(f"[{args.tag}] {len(split)} files ({args.files_dir}) -> internal "
          f"{len(split) - n_val} train / {n_val} val (every {args.val_every}th) "
          f"cache={cache_path}", flush=True)

    d = mod.load_cache(cache_path)
    tr = (d["split"] == 0).nonzero().flatten()
    va = (d["split"] == 1).nonzero().flatten()
    print(f"[{args.tag}] {len(tr)} train / {len(va)} val windows", flush=True)

    cfg = {"k_in": 6, "leads": (0, 4, 9, 14, 19), "summary": True,
           "lead_e": False, "phase": False, "n_blocks": 4, "phase_kind": "probe",
           "width": 24, "wide": 32}
    cfg["in_ch"] = K.n_features(cfg["k_in"], len(cfg["leads"]), cfg["summary"])
    print(f"[{args.tag}] in_ch={cfg['in_ch']}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = []
    nets = []
    for s in seeds:
        val_e, val_metrics, state = train_one_seed(
            mod, K, d, tr, va, cfg, s, args.epochs, args.batch_size,
            args.lr, args.wd, args.flip, dev, args.tag)
        ckpt_path = os.path.join(ckpt_dir, f"ke_net_flow_seed{s}.pt")
        net = K.KeFlowNet(cfg["in_ch"], cfg["width"], cfg["wide"]).to(dev)
        net.load_state_dict(state)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        K.save_ke_flow_net(ckpt_path, net, cfg)
        nets.append(net)
        results.append({"seed": s, "val_e": val_e, "val_metrics": val_metrics,
                        "ckpt": ckpt_path})
        print(f"[{args.tag}] seed {s} saved -> {ckpt_path}", flush=True)

    mean_single = sum(r["val_e"] for r in results) / len(results)
    ens_val_e = evaluate_ensemble(mod, K, nets, d, va, cfg, dev)
    rel_gain = 1.0 - ens_val_e / mean_single
    kill = rel_gain < 0.01

    summary = {
        "tag": args.tag, "seeds": seeds, "epochs": args.epochs,
        "per_seed_val_e": {str(r["seed"]): r["val_e"] for r in results},
        "mean_single_seed_val_e": mean_single,
        "ensemble_val_e": ens_val_e,
        "relative_gain": rel_gain,
        "kill_threshold": 0.01,
        "verdict": "KILL" if kill else "PASS",
    }
    print(f"[{args.tag}] per-seed val e: "
          + ", ".join(f"{r['seed']}={r['val_e']:.4f}" for r in results), flush=True)
    print(f"[{args.tag}] mean single-seed val e {mean_single:.4f} (tke {mod.score(mean_single):.2f}) "
          f"| ensemble val e {ens_val_e:.4f} (tke {mod.score(ens_val_e):.2f}) "
          f"| relative gain {rel_gain * 100:.2f}% | {summary['verdict']}", flush=True)

    with open(os.path.join(out_dir, "stage0_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.tag}] wrote {os.path.join(out_dir, 'stage0_summary.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
