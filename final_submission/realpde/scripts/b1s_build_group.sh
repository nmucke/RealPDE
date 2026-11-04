#!/usr/bin/env bash
# B1-3090 Stage 2: fit prior.npz + ke_net_flow.pt for one K-fold sub-pipeline group
# (b5_h192_blk6_rf11 backbone), stream its held-out group through the full bundle
# pipeline (with the W2 aux-map hook already baked in by scripts/w2_make_kfold.py),
# extract the compact feature cache, delete the dump.
#
# Adapted from scripts/w2p_build_group.sh: c11a_fit_prior/c11a_fit_ke_flow (n1_simvp)
# swapped for b1_fit_prior/b1_fit_ke_flow (b5_h192_blk6_rf11); ROOT moved to
# experiments/b1_3090; checkpoint dir to data/checkpoints/b1_3090/kfold.
#   Usage: b1s_build_group.sh <fold> <g>      e.g. b1s_build_group.sh all81 0
set -euo pipefail
FOLD=$1; G=$2
MODEL=b5_h192_blk6_rf11
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$(dirname "$0")/.."
ROOT=experiments/b1_3090
GD=$ROOT/kfold/$FOLD/g$G
CK=data/checkpoints/b1_3090/kfold/${MODEL}_${FOLD}_g$G.pth
PY=.venv/bin/python
echo "[build $FOLD g$G] start $(date +%T)"
$PY scripts/b1_fit_prior.py --model $MODEL --files $GD/train --ckpt $CK \
    --out $GD/pipe/prior.npz --device cuda --mem-frac 0.15
command rm -rf $ROOT/cache/tke_flow_${FOLD}_g$G   # R4 (experiments/rch/notes.md): never resume a killed run's partial cache
$PY scripts/b1_fit_ke_flow.py --tag b1s_${FOLD}_g$G --model $MODEL --ckpt $CK --files $GD/train \
    --cache $ROOT/cache/tke_flow_${FOLD}_g$G --mem-frac 0.15 --out $GD/pipe/ke_net_flow.pt
command rm -rf $ROOT/cache/tke_flow_${FOLD}_g$G
echo "[build $FOLD g$G] fits done $(date +%T)"
W2_AUX_OUT=$ROOT/cache/aux_${FOLD}_g$G.npz $PY scripts/lco_run_dump.py --exp-dir $GD/pipe \
    --data-dir $GD/heldout --device cuda --mem-frac 0.15 --out $ROOT/cache/dump_${FOLD}_g$G.npz
$PY scripts/lco_analyze.py $ROOT/cache/dump_${FOLD}_g$G.npz --tag "b1s $FOLD g$G" --time-score 88 \
    --json $GD/heldout_scores.json > $GD/analyze.log
head -3 $GD/analyze.log
echo "[build $FOLD g$G] stream done $(date +%T)"
$PY scripts/w2_extract.py --dump $ROOT/cache/dump_${FOLD}_g$G.npz --data-dir $GD/heldout \
    --aux $ROOT/cache/aux_${FOLD}_g$G.npz --out $ROOT/cache/feat_${FOLD}_g$G.npz
command rm -f $ROOT/cache/dump_${FOLD}_g$G.npz
echo "[build $FOLD g$G] DONE $(date +%T)"
