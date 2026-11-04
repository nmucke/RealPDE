#!/usr/bin/env bash
# Retrain the weights a preset needs and assemble a package for run_preset.py.
#
#   bash experiments/train_preset.sh PRESET DATA_DIR WORK_DIR [--dry-run]
#
# PRESET    id of experiments/presets/<id>.json; its `training.env` holds the retrain_all.sh flags
# DATA_DIR  <dir>/train_sim (100 .h5), <dir>/train_real (82 files; 7575_0.h5 is skipped by name),
#           optional <dir>/mean_std_real.pt
#           (the layout final_submission/solution.py train() expects)
# WORK_DIR  scratch space (tens of GB of checkpoints / streams); reused on a rerun (the chain skips
#           finished steps). No spaces in the path (the chain's output is parsed).
#
# What it does:
#   1. copies the training tree final_submission/realpde to WORK_DIR/tree. A rerun reuses it;
#   2. points the copy's data/competition/{train_sim,train_real} at DATA_DIR and its
#      .venv/bin/python at $PYTHON (default python3), as solution.py does;
#   3. prints the chain's DRY_RUN resolution for the preset's flags (stops here with --dry-run);
#   4. runs scripts/retrain_all.sh with those flags: all steps if the resolved package source exists
#      in the tree, else steps 01-09 (weights only);
#   5. checks model.pth's arch key and the width-net path against the preset, and assembles
#      WORK_DIR/package_<PRESET> = the final source + the four weight files.
# Then: python experiments/run_preset.py --package WORK_DIR/package_<PRESET> --data ... PRESET
#
# Needs one CUDA GPU (>= 8 GB free) and the eval-image stack (Python 3.10, torch 2.2.2, numpy 1.26,
# h5py, scipy, einops). Times: about 2 h for the b5spx3 presets, less for n1 / b5sp (RTX 3090 figures in
# final_submission/README.md). Nothing is written outside WORK_DIR.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
usage () { sed -n '2,26p' "${BASH_SOURCE[0]}"; }
case "${1:-}" in -h|--help) usage; exit 0 ;; esac
[ $# -ge 3 ] || { usage; exit 2; }
PRESET="$1"; WORK="$3"
[ -d "$2" ] || { echo "FATAL: DATA_DIR $2 is not a directory"; exit 2; }
DATA="$(cd "$2" && pwd)"; shift 3
TREE_SRC="$REPO/final_submission/realpde"
DRY=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    *) echo "unknown argument: $a"; exit 2 ;;
  esac
done
PY_HOST="${PYTHON:-python3}"
command -v "$PY_HOST" >/dev/null 2>&1 || { echo "FATAL: python '$PY_HOST' not found (set PYTHON=/path/to/python3.10)"; exit 2; }
for p in "$WORK" "$REPO" "$DATA"; do
  case "$p" in *[[:space:]]*) echo "FATAL: path with whitespace not supported: '$p'"; exit 2 ;; esac
done
mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
TREE="$WORK/tree"
MARK="$TREE/.train_preset_tree"

# preset -> flags, expected outputs and arch (stdlib json only)
read_preset () {
  "$PY_HOST" -B - "$HERE/presets/$PRESET.json" "$1" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))["training"]
print({"env": lambda: " ".join("%s=%s" % kv for kv in t["env"].items()),
       "ship": lambda: t["outputs"]["model.pth, prior.npz, ke_net_flow.pt"],
       "width": lambda: t["outputs"]["cond_width_weights.pt"],
       "arch": lambda: t["arch"]}[sys.argv[2]]())
EOF
}
[ -f "$HERE/presets/$PRESET.json" ] || { echo "no preset $PRESET"; exit 2; }
FLAGS="$(read_preset env)"
SHIP="$(read_preset ship)"
WIDTH="$(read_preset width)"
ARCH="$(read_preset arch)"
echo "preset $PRESET: flags [${FLAGS:-<defaults>}], tree $TREE_SRC"
OUT="$WORK/package_$PRESET"
# the package location must not let submission.py import a foreign src/ (checked before any training)
"$PY_HOST" -B -c 'import sys; sys.path.insert(0, sys.argv[1]); import ablate; ablate.check_package_location(sys.argv[2])' \
  "$REPO/experiments" "$OUT" || { echo "FATAL: unusable package location $OUT; choose another WORK_DIR"; exit 2; }

if [ -d "$TREE" ]; then
  [ -f "$MARK" ] || { echo "FATAL: $TREE has no source marker (not made by this script); delete $TREE and rerun"; exit 2; }
  [ "$(cat "$MARK")" = "$TREE_SRC" ] \
    || { echo "FATAL: $TREE was copied from '$(cat "$MARK")', not $TREE_SRC; use another WORK_DIR"; exit 2; }
else
  cp -a "$TREE_SRC" "$TREE"
  rm -rf "$TREE/data/competition"
  echo "$TREE_SRC" > "$MARK"
fi
mkdir -p "$TREE/data/competition" "$TREE/.venv/bin"
for d in train_sim train_real; do
  [ -d "$DATA/$d" ] || { echo "missing $DATA/$d"; exit 2; }
  ln -sfn "$DATA/$d" "$TREE/data/competition/$d"
done
# normalisation stats: DATA_DIR's if present, else the copy in the source tree (never a stale earlier copy)
if [ -f "$DATA/mean_std_real.pt" ]; then
  cp -f "$DATA/mean_std_real.pt" "$TREE/starting_kit/example_data/mean_std_real.pt"
else
  cp -f "$TREE_SRC/starting_kit/example_data/mean_std_real.pt" "$TREE/starting_kit/example_data/mean_std_real.pt"
fi
PY_ABS="$(command -v "$PY_HOST")"
for exe in python python3; do
  printf '#!/bin/sh\nexec "%s" "$@"\n' "$PY_ABS" > "$TREE/.venv/bin/$exe"
  chmod 755 "$TREE/.venv/bin/$exe"
done
export PYTHONDONTWRITEBYTECODE=1

cd "$TREE"
RES="$(env $FLAGS DRY_RUN=1 bash scripts/retrain_all.sh)" || { echo "$RES"; echo "flags do not resolve in $TREE_SRC"; exit 1; }
echo "$RES" | sed -n '1,/^# STEPS/p' | sed '$d'
field () { echo "$RES" | sed -n "s/^.*[[:space:]]$1=\([^[:space:]]*\).*$/\1/p" | head -1; }
SRC_PKG="$(field SRC_PKG)"
SHIP_DIR="$(field SHIP_DIR)"
W2_OUT="$(field W2_OUT)"
RECIPE="$(field CONDWIDTH_RECIPE)"
[ -n "$SRC_PKG" ] && [ -n "$SHIP_DIR" ] && [ -n "$RECIPE" ] \
  || { echo "FATAL: could not read SRC_PKG / SHIP_DIR / CONDWIDTH_RECIPE from the DRY_RUN output"; exit 1; }
[ "$SHIP_DIR" = "$SHIP" ] || { echo "FATAL: chain SHIP_DIR=$SHIP_DIR, preset says $SHIP"; exit 1; }
if [ "$RECIPE" = "w2" ]; then WEXP="$W2_OUT/cond_width_weights.pt"; else WEXP="$TREE/data/checkpoints/ship_condwidth/net_all3.pt"; fi
[ "$WEXP" = "$TREE/$WIDTH" ] || { echo "FATAL: chain width net $WEXP (recipe $RECIPE), preset says $TREE/$WIDTH"; exit 1; }
[ "$DRY" = "1" ] && exit 0

if [ -f "$SRC_PKG/submission.py" ]; then
  env $FLAGS bash scripts/retrain_all.sh
else
  echo "package source $SRC_PKG is not in this repository: running the weight steps only"
  env $FLAGS bash scripts/retrain_all.sh --only 01,02,03,04,05,05a,05b,06,07,08,09
fi

rm -rf "$OUT"
"$PY_HOST" -B - "$REPO/experiments" "$OUT" "$TREE/$SHIP_DIR" "$TREE/$WIDTH" "$ARCH" <<'EOF'
import os, sys
sys.path.insert(0, sys.argv[1])
import torch
import ablate
out, ship, width, arch = sys.argv[2:6]
p = torch.load(os.path.join(ship, "model.pth"), map_location="cpu", weights_only=False)
got = p.get("arch") if isinstance(p, dict) else None
if got != arch:
    raise SystemExit("FATAL: %s/model.pth has arch %r, preset expects %r" % (ship, got, arch))
w = {n: os.path.join(ship, n) for n in ("model.pth", "prior.npz", "ke_net_flow.pt")}
w["cond_width_weights.pt"] = width
print("package:", ablate.assemble_package(w, out))
EOF
