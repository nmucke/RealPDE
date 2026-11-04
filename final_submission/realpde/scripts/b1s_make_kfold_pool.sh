#!/usr/bin/env bash
# B1-3090 Stage 2: K=4 cross-fit pool with the b5_h192_blk6_rf11 backbone, for
# out-of-sample W2 (conditional-width net) feature generation -- generalises
# scripts/w2p_make_shipping_data.sh (which is n1_simvp/all81-only, via
# scripts/j1_finetune.py) to ANY fold/train-dir and the b1 registry's
# b1_finetune.py. `scripts/w2_make_kfold.py` is reused UNMODIFIED for the
# split + pipe-dir layout (it hardcodes the n1_simvp/c11a_models submission.py
# and an `n1simvp_s3_<fold>_g<g>.pth` ckpt name); this script re-patches each
# group's pipe/submission.py exactly as scripts/b1s_make_pipe.py does (import
# c11a_models -> b1_models, build_core("n1_simvp") -> build_core(<model>),
# unet_ckpt -> the b5 checkpoint path) before training/fitting/streaming.
#
# Deterministic (seed 0), idempotent (skips a group whose feature cache exists).
#   Usage: b1s_make_kfold_pool.sh <fold> <train-dir> [K=4] [n_concurrent=2]
#   e.g.   b1s_make_kfold_pool.sh fold0 data/real_lco/fold0/train 4 2
#          b1s_make_kfold_pool.sh all81 experiments/b1_3090/all81 4 2
set -euo pipefail
FOLD=$1; TRAIN_DIR=$2; K=${3:-4}; JOBS=${4:-2}
MODEL=b5_h192_blk6_rf11
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=experiments/b1_3090
LOGS=$ROOT/logs
KFOLD_ROOT=$ROOT/kfold
CKPT_DIR=data/checkpoints/b1_3090/kfold
mkdir -p "$LOGS" "$ROOT/cache" "$CKPT_DIR"

# R4 (2026-09-26, experiments/rch/notes.md; same check as retrain_all.sh's artefact_ok): a group's backbone / feature cache
# is reused only if it reads back (zip central directory + CRC of every member; non-zip .pth: torch.load). A truncated file
# (job killed mid-save) is renamed to <file>.corrupt and rebuilt. VERIFY_REUSE=0 -> existence only (the old behaviour).
ok() {
    [ -f "$1" ] || return 1
    [ "${VERIFY_REUSE:-1}" = "1" ] || return 0
    CUDA_VISIBLE_DEVICES= $PY - "$1" <<'PYEOF'
import os, sys, zipfile
f = sys.argv[1]
try:
    if zipfile.is_zipfile(f):
        with zipfile.ZipFile(f) as z:
            m = z.testzip()
        if m is not None:
            raise ValueError(f"CRC mismatch in member {m}")
    elif f.endswith((".pt", ".pth")):
        import torch
        torch.load(f, map_location="cpu", weights_only=False)
    else:
        raise ValueError("not a readable zip archive (truncated?)")
except Exception as e:  # noqa: BLE001
    os.replace(f, f + ".corrupt")
    print(f"REUSE CHECK FAILED: {f} ({type(e).__name__}: {str(e)[:120]}) -> moved to {f}.corrupt; rebuilding", flush=True)
    sys.exit(1)
PYEOF
}

# --- 1. K-fold layout (reuse scripts/w2_make_kfold.py UNMODIFIED) ----------
if [ ! -f $KFOLD_ROOT/$FOLD/g$((K-1))/pipe/submission.py ]; then
    $PY scripts/w2_make_kfold.py --fold "$FOLD" --K "$K" --train-dir "$TRAIN_DIR" \
        --out-root "$KFOLD_ROOT" --ckpt-dir "$CKPT_DIR" 2>&1 | tee "$LOGS/make_kfold_$FOLD.log"
fi

# --- 2. re-patch each group's pipe/submission.py for the b5 arch -----------
for g in $(seq 0 $((K-1))); do
    SUB=$KFOLD_ROOT/$FOLD/g$g/pipe/submission.py
    # ABSOLUTE path required: this pipe dir sits 4 levels under the repo root
    # (experiments/b1_3090/kfold/<fold>/g<g>/pipe/), one deeper than
    # experiments/d2_bundle/fold0_c/ (2 levels) that submission.py's own
    # `_REPO = abspath(join(_HERE, pardir, pardir, pardir))` assumes -- a
    # relative unet_ckpt resolves to the wrong directory (w2_make_kfold.py's
    # own ckpt_rel is absolute for exactly this reason, see its comment).
    CK=$(readlink -f "$CKPT_DIR")/${MODEL}_${FOLD}_g$g.pth
    if ! grep -q "import b1_models as _C11A" "$SUB"; then
        $PY - "$SUB" "$CK" "$MODEL" <<'PYEOF'
import re, sys
sub, ck, model = sys.argv[1:4]
s = open(sub).read()
n1 = s.count('import c11a_models as _C11A')
assert n1 == 1, f"expected 1 c11a_models import, found {n1} in {sub}"
s = s.replace('import c11a_models as _C11A', 'import b1_models as _C11A')
n2 = s.count('_C11A.build_core("n1_simvp")')
assert n2 == 1, f"expected 1 build_core(n1_simvp), found {n2} in {sub}"
s = s.replace('_C11A.build_core("n1_simvp")', f'_C11A.build_core("{model}")')
pat = r'"unet_ckpt":\s*"[^"]+"'
assert re.search(pat, s), f"unet_ckpt not found in {sub}"
s = re.sub(pat, f'"unet_ckpt": "{ck}"', s, count=1)
open(sub, "w").write(s)
print(f"patched {sub} -> model={model} ckpt={ck}")
PYEOF
    fi
done

# --- 3. per group: b5 backbone on the other K-1 groups, then build_group ---
one_group() {
    g=$1
    if ok $ROOT/cache/feat_${FOLD}_g$g.npz; then echo "[g$g] feature cache exists, skip"; return 0; fi
    ck=$CKPT_DIR/${MODEL}_${FOLD}_g$g.pth
    if ! ok $ck; then
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY scripts/b1_finetune.py --model $MODEL --loss s3 \
            --train $KFOLD_ROOT/$FOLD/g$g/train --seed 0 --mem-frac 0.15 \
            --out $ck > $LOGS/train_${FOLD}_g$g.log 2>&1
    fi
    scripts/b1s_build_group.sh "$FOLD" "$g" > $LOGS/build_${FOLD}_g$g.log 2>&1
}

pids=()
for g in $(seq 0 $((K-1))); do
    one_group "$g" &
    pids+=($!)
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 20; done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -eq 0 ] || { echo "a group FAILED -- see $LOGS"; exit 1; }

echo "=== $FOLD K=$K pool done $(date -u +%FT%TZ) ==="
for g in $(seq 0 $((K-1))); do
    sha256sum "$CKPT_DIR/${MODEL}_${FOLD}_g$g.pth" "$ROOT/cache/feat_${FOLD}_g$g.npz"
done
