#!/usr/bin/env python3
"""Per-trajectory subscore breakdown + composite score for an LCO run .npz
(written by scripts/lco_run_dump.py). Calls the official scoring functions
(`starting_kit/scoring.py`) per file; does not modify or reimplement them.

Composite: final ~= 0.48*rel_l2 + 0.07*tke + 0.10*mvpe + 0.09*time + 0.28*sps
(MEMORY.md fitted leaderboard weights); --time-score fixes the time subscore
for every row since timing was not measured in this task (shared GPU).

Usage:
    .venv/bin/python scripts/lco_analyze.py experiments/lco_baseline/runs/fold0_heldout.npz \\
        --tag "integrated@fold0-heldout" --time-score 88 \\
        --json experiments/lco_baseline/runs/fold0_heldout.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "starting_kit"))
import scoring as official  # noqa: E402

FINAL_W = {"rel_l2": 0.48, "tke": 0.07, "mvpe": 0.10, "time": 0.09, "sps": 0.28}


def composite(rel_l2, tke, mvpe, sps, time_score):
    return (FINAL_W["rel_l2"] * rel_l2 + FINAL_W["tke"] * tke + FINAL_W["mvpe"] * mvpe
            + FINAL_W["time"] * time_score + FINAL_W["sps"] * sps)


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def score_block(pred, tgt, lower, upper, c):
    rel = official.rel_l2_per_sample(pred, tgt, c)
    tke = official.tke_rel_l2_per_sample(pred, tgt, c)
    mv = official.mvpe_rel_l2_per_sample(pred, tgt)
    sps, cov = official.aggregate_sps(pred, tgt, c, lower=lower, upper=upper)
    return {
        "rel_l2": official.score_error(float(rel.mean())),
        "tke": official.score_error(float(tke.mean())),
        "mvpe": official.score_error(float(mv.mean())),
        "sps": official.score_sps(sps),
        "coverage": cov,
        "n_windows": int(pred.shape[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--tag", default="")
    ap.add_argument("--time-score", type=float, default=88.0,
                     help="fixed time_score to use in the composite (not measured here)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    z = load(args.run)
    pred, tgt = z["pred"], z["target"]
    lower = z["lower"] if "lower" in z else None
    upper = z["upper"] if "upper" in z else None
    sim = [str(s) for s in z["sim_ids"]]
    c = official.measured_channels(tgt)

    tag = args.tag or Path(args.run).stem
    overall = score_block(pred, tgt, lower, upper, c)
    overall["composite"] = composite(overall["rel_l2"], overall["tke"], overall["mvpe"],
                                     overall["sps"], args.time_score)
    print(f"\n===== {tag} ===== ({overall['n_windows']} windows, {len(set(sim))} files)")
    print(f"  rel_l2={overall['rel_l2']:.2f}  tke={overall['tke']:.2f}  "
          f"mvpe={overall['mvpe']:.2f}  sps={overall['sps']:.2f}  "
          f"time={args.time_score:.2f} (fixed)  coverage={overall['coverage']:.3f}  "
          f"composite={overall['composite']:.3f}")

    per_file = {}
    print(f"\n  {'file':<12}{'nw':>4}{'rel_l2':>9}{'tke':>8}{'mvpe':>8}{'sps':>8}{'cov':>7}{'composite':>11}")
    for f in sorted(set(sim)):
        idx = np.array([i for i, s in enumerate(sim) if s == f])
        lo_f = lower[idx] if lower is not None else None
        up_f = upper[idx] if upper is not None else None
        row = score_block(pred[idx], tgt[idx], lo_f, up_f, c)
        row["composite"] = composite(row["rel_l2"], row["tke"], row["mvpe"], row["sps"],
                                     args.time_score)
        per_file[f] = row
        print(f"  {f:<12}{row['n_windows']:>4}{row['rel_l2']:>9.2f}{row['tke']:>8.2f}"
              f"{row['mvpe']:>8.2f}{row['sps']:>8.2f}{row['coverage']:>7.3f}{row['composite']:>11.3f}")

    result = {"tag": tag, "run": str(args.run), "time_score_fixed": args.time_score,
              "overall": overall, "per_file": per_file}
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=float) + "\n")
        print(f"\n[lco_analyze] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
