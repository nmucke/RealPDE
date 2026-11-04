"""Randomly split the competition data into train/val/test.

Creates data/<real|sim>/<train|val|test>/ containing symlinks to the .h5 files in
data/competition/train_<real|sim>/ (symlinks, not copies: the source is ~16 GB).

    python scripts/split_data.py
"""

import random
from pathlib import Path

# --- config -----------------------------------------------------------------
# 0.90/0.05/0.05 with seed 0 reproduces the 73/4/4 `data/real` split that every shipped
# artefact and manifest was built against (val: 20325_5, 25425_0, 5025_0, 5025_20; test:
# 11400_0, 19050_5, 24150_15, 25425_15). Commit daddf7c3 changed the ratios to 0.85/0.10 without
# regenerating the split on disk, so a fresh clone produced a different 69/8/4 split
# (experiments/d3_repro). Do not change these without regenerating and re-recording.
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05
SEED = 0
SKIP = {"7575_0.h5"}  # corrupt, per the competition organizers
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "competition"

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "ratios must sum to 1"

for kind in ("real", "sim"):
    files = sorted(f for f in (SRC / f"train_{kind}").glob("*.h5") if f.name not in SKIP)
    random.Random(SEED).shuffle(files)

    n_train = round(TRAIN_RATIO * len(files))
    n_val = round(VAL_RATIO * len(files))
    splits = {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }

    for split, members in splits.items():
        out = ROOT / "data" / kind / split
        out.mkdir(parents=True, exist_ok=True)
        for link in out.glob("*.h5"):
            link.unlink()
        for f in members:
            (out / f.name).symlink_to(f.resolve())
        print(f"{kind}/{split}: {len(members)} files -> {out.relative_to(ROOT)}")
