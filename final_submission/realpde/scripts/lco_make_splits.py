#!/usr/bin/env python3
"""Materialise a leave-conditions-out fold as symlink directories.

Reads `experiments/lco_split/split.json` and, for the chosen fold, creates
    data/real_lco/<fold>/train/*.h5     -> symlinks into data/competition/train_real/
    data/real_lco/<fold>/heldout/*.h5   -> symlinks into data/competition/train_real/
so that the fold's file lists can be handed to any script that already accepts
a data directory (e.g. `scripts/finetune_fno.py --train/--val`), without
touching `data/real/{train,val,test}` (the unseen-trajectory-only honest
split) or editing any existing script.

`train_real/7575_0.h5` (corrupt, CLAUDE.md) is never in split.json's lists;
this script also refuses to symlink it defensively.

Usage:
    .venv/bin/python scripts/lco_make_splits.py --fold fold0_aoa20_plus_re_extremes
    .venv/bin/python scripts/lco_make_splits.py --fold fold0_aoa20_plus_re_extremes \\
        --train-sample-n 20 --train-sample-seed 0   # also materialise a same-size
                                                      # random sample of TRAIN, for
                                                      # a memorised reference split
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BAD = "7575_0.h5"


def relink(dst_dir: Path, files: list[str], src: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    # clear stale symlinks so a re-run reflects the current file list exactly
    for p in dst_dir.glob("*.h5"):
        if p.is_symlink():
            p.unlink()
    for name in files:
        assert name != BAD, f"refusing to link corrupt file {BAD}"
        target = src / name
        if not target.is_file():
            raise SystemExit(f"missing source file: {target}")
        (dst_dir / name).symlink_to(target.resolve())
    print(f"{dst_dir}: {len(files)} files")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", required=True,
                     help="key into experiments/lco_split/split.json 'folds'")
    ap.add_argument("--split-json", default=str(REPO / "experiments" / "lco_split" / "split.json"))
    ap.add_argument("--src", default=str(REPO / "data" / "competition" / "train_real"))
    ap.add_argument("--out", default=str(REPO / "data" / "real_lco"))
    ap.add_argument("--train-sample-n", type=int, default=0,
                     help="also materialise <out>/<fold>/train_sample with this many "
                          "TRAIN files (for a same-size 'memorised' reference split)")
    ap.add_argument("--train-sample-seed", type=int, default=0)
    args = ap.parse_args()

    split = json.loads(Path(args.split_json).read_text())
    if args.fold not in split["folds"]:
        raise SystemExit(f"unknown fold {args.fold!r}; have {sorted(split['folds'])}")
    fold = split["folds"][args.fold]
    src = Path(args.src)
    out = Path(args.out) / args.fold

    relink(out / "train", fold["train_files"], src)
    relink(out / "heldout", fold["heldout_files"], src)

    if args.train_sample_n:
        rng = random.Random(args.train_sample_seed)
        sample = sorted(rng.sample(fold["train_files"], min(args.train_sample_n,
                                                              len(fold["train_files"]))))
        relink(out / "train_sample", sample, src)

    print(f"fold {args.fold}: train {len(fold['train_files'])}, "
          f"heldout {len(fold['heldout_files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
