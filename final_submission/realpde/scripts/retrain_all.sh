#!/usr/bin/env bash
# =============================================================================
# retrain_all.sh -- ONE-COMMAND retrain of the full shipping chain from a FRESH
# `git clone`, for submissions/2026-10-03_simvp-s3-bundle-condwidth-v2-wholestep-t2
# (the archive on the leaderboard since 2026-09-22, final 81.828: SimVP-gSTA +
# causal ctx-norm + S3 Welsch loss + D2 free-rider bundle + W2 conditional
# interval widths v2 (coverage-aware target, K=4 cross-fit over all 81 files),
# whole-step CUDA graph + T2 kernel cuts).
#
# LINEAGE OPTION 2026-09-23 (c2 ENRx, packed K = 2 ensemble of SP soups from TWO sim-pretrain seeds):
# BACKBONE=b5spx2 (opt-in; evidence + exact recipe: experiments/c2_soup/report.md, section "ENRx"):
#   05a  the b5sp seed-0 sim pretrain (unchanged, reused) + a SECOND one, identical args but --seed 1
#        -> data/checkpoints/sp/b5_sim_s3_seed1_it10k.pth
#   06   member 0 = uniform soup of SOUP_K (default 4) all-81 fine-tunes from the seed-0 pretrain (seeds 0..K-1;
#        seed 0 = the b5sp ship ckpt; soup path = the b5sp SOUP_K soup's); member 1 = soup of SOUP_K all-81
#        fine-tunes from the seed-1 pretrain. Same sp_finetune.py command as b5sp. Never souped ACROSS inits.
#   07 / 08  ENR's b5x2 path: scripts/ens_pack_b5x2.py, ensemble prior + single ensemble ke net
#        (en_fit_prior.py / en_fit_ke_flow.py, all 81 = the gated config; ENRX_KE=1 adds KE's 4-seed packed
#        ke net instead -- NOT gated for ENRx). All outputs in X2_DIR=data/checkpoints/sp/ship_enrx_soup<K>[_kes4].
#   05b / 09 IDENTICAL to b5 / b5sp (W2 v2 on the random-init b5 K=4 all-81 pool).
#   10   package source 2026-10-07_simvp-b5x2-... (ENR); weights from X2_DIR, W2 from the b5-family W2_OUT.
#   Needs train_sim (100 files). Cost vs b5sp: +1 pretrain + 2K-1 fine-tunes (K=4: ~10 min + ~1 h, 3090).
#
# LINEAGE OPTION 2026-09-24 (K3F, packed K = 3 ensemble + fp16 core): BACKBONE=b5spx3 = BACKBONE=b5spx2 (every flag,
# every step) + a THIRD member and FP16X's scoped fp16 packed core (evidence: experiments/k3_ensemble/notes.md,
# experiments/fp16x_core/notes.md; packaged by experiments/k3f_build/notes.md):
#   05a  + a THIRD sim pretrain, identical args, --seed 2 -> data/checkpoints/sp/b5_sim_s3_seed2_it10k.pth
#   06   + member 2 = soup of SOUP_K all-81 fine-tunes from the seed-2 pretrain (same sp_finetune.py command,
#        fine-tune seeds 0..K-1) -> data/checkpoints/sp/b5sp_pre2_soup<K>_s3_all81.pth
#   07   scripts/k3_pack_b5xk.py member0 member1 member2 -> arch "b5x3_h192_blk6_rf11"
#   08   en_fit_prior.py / en_fit_ke_flow.py with the 3 members, all 81. Outputs in data/checkpoints/sp/ship_enrx3_soup<K>.
#   10   package source 2026-10-16_simvp-b5spx3-soup4-fp16-... (the AXG+MSK source + K3's b5x3 dispatch + FP16X's
#        fp16 core, CONFIG "fp16_core": True). Only the shipped flag set exists: CB=1 CB_ADAPT=1 AX=1 GM20=1 MSK=1,
#        AXE=0 B2=0 ENRX_KE=0, FP16=1 (FP16 defaults to 1 here and is FATAL anywhere else).
#        K3E (2026-09-24, experiments/k3e_build/notes.md): AXE=1 (default 0) packages instead the source
#        2026-10-16_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-s3-bundle-condwidth-v2-wholestep-t2-w01-kd
#        = the K3F source + AXE's E4 delta (axe_adapt.py; cb_blend_w0 1.0, eb_adapt grid [0.45 .. 0.925], br_adapt;
#        no new weights / fits, same 05a-08 outputs); PKG_NAME "-cbadapt-axe-gm20-msk-".
#        KD2 (2026-09-24, experiments/kd2_adapt_diet/notes.md): KD2=1 (DEFAULT) packages the kernel-diet sources
#        2026-10-17_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-kd2-... (AXE=0) / ...-cbadapt-axe-gm20-msk-kd2-... (AXE=1)
#        = the K3F / K3E source + kd2_adapt.py, a fused cuda_graph_wholestep.py and CONFIG "kd2_adapt": True (the AW /
#        AX (/ AXE) selectors fused in the whole-step graph; same pred / bounds bit for bit, ~40 fewer kernels / step).
#        Code only: no new weights / fits, same 05a-08 outputs; PKG_NAME gets "-gm20-msk-kd2-". KD2=0 = the K3F / K3E
#        sources above, unchanged. KD2=1 is FATAL outside BACKBONE=b5spx3.
#        KD3 (2026-09-24, experiments/kd3_step_diet/notes.md; K3E port experiments/v22_verify/notes.md): KD3=1 (DEFAULT
#        when KD2=1; requires KD2=1) packages 2026-10-17_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-kd3-... (AXE=0) /
#        ...-cbadapt-axe-gm20-msk-kd3-... (AXE=1) = the KD2 source + kd3_step.py, a KD3 cuda_graph_wholestep.py and
#        CONFIG "kd3_step": True (bit-exact whole-step backbone / mega diet, ~95 fewer kernels / step). Code only: no new
#        weights / fits; PKG_NAME gets "-gm20-msk-kd3-" (instead of "-kd2-"). KD3=0 = the KD2 sources, unchanged.
#        KD2=0 defaults KD3 to 0; KD3=1 with KD2=0, or outside BACKBONE=b5spx3, is FATAL.
#        IPC (2026-09-24, experiments/ipc_outproc/notes.md; K3E port experiments/k3ei_build/notes.md): IPC=1 (default 0;
#        requires KD3=1) packages 2026-10-18_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-kd3-ipc-... (AXE=0) /
#        ...-cbadapt-axe-gm20-msk-kd3-ipc-... (AXE=1) = the KD3 source + CONFIG "ipc_one_storage": "auto" (graphed steps
#        return pred / lower / upper as three disjoint views of ONE CPU clone when the input arrived through torch's
#        shared-memory multiprocessing reduction; same values bit for bit). Code only: no new weights / fits;
#        PKG_NAME gets "-gm20-msk-kd3-ipc-". IPC=0 = the KD3 sources, unchanged. IPC=1 with KD3=0, or outside
#        BACKBONE=b5spx3, is FATAL.
#        KD5 (2026-09-24, experiments/kd5_more/notes.md, FR24 GO; IPC port experiments/k3eik_build/notes.md): KD5=1
#        (default 0; requires KD3=1) packages 2026-10-18_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-kd5-... (AXE=0) /
#        ...-cbadapt-axe-gm20-msk-kd5-... (AXE=1) / ...-cbadapt-axe-gm20-msk-kd5-ipc-... (AXE=1 IPC=1) = the KD3 (/ KD3-IPC)
#        source + a KD5 kd3_step.py and CONFIG "kd5_near": True (NEAR-EXACT steady whole-step graph diet: stem / down
#        convs as unfold + GEMM, dw2 space-to-batch, W2 c1 bias as a ones channel; fp16 rounding-order differences only,
#        |d composite| <= 0.0004 on fold0 / fold1 / foldR). Code only: no new weights / fits; PKG_NAME gets
#        "-gm20-msk-kd5-" (instead of "-kd3-"). KD5=0 = the sources above, unchanged. KD5=1 with KD3=0, AXE=0 IPC=1
#        (no K3F-KD5-IPC source), or outside BACKBONE=b5spx3, is FATAL.
#        KD4 / REC / FZ (2026-09-24; KD4 experiments/kd4_w01_diet FR26 GO, REC experiments/rec_recovery FR25 GO, the
#        freeze combination experiments/fz_freeze/notes.md): KD4=1 (default 0) = W01 window-0 / window-1 graph diet
#        (CONFIG "kd4_w01": True, kd4_w01.py; bit-exact); REC=1 (default 0) = causal regime-change detector + partial
#        state reset in the whole-step graph (CONFIG "rec_detect", rec_detect.py; bit-identical on normal streams).
#        Both require KD3=1. Code only: no new weights / fits. Only combinations with a tracked source exist
#        (AXE KD5 KD4 REC IPC): 1 1 1 1 1 -> FZ-E 2026-10-19_...-cbadapt-axe-gm20-msk-kd5-kd4-rec-ipc-... and
#        0 1 1 1 1 -> FZ-F ...-cbadapt-ax-gm20-msk-kd5-kd4-rec-ipc-... (= K3E-KD3 / K3F-KD3 + IPC + KD5 + KD4 + REC,
#        each behind its CONFIG flag; PKG_NAME "-gm20-msk-kd5-kd4-rec-ipc-"); 0 0 1 0 0 -> K3F-KD4 2026-10-17_...-ax-
#        gm20-msk-kd4-... (PKG_NAME "-gm20-msk-kd4-"); x 0 0 1 0 -> K3E-KD3-REC / K3F-KD3-REC 2026-10-18_...-msk-kd3-rec-...
#        (PKG_NAME "-gm20-msk-kd3-rec-"). Any other combination with KD4=1 or REC=1 is FATAL. KD4=0 REC=0 = the sources
#        above, unchanged (with KD4=1 or REC=1 the KD5 block's own source choice is skipped).
#   Internally BACKBONE=b5spx3 runs the b5spx2 steps with X3=1 (all b5spx2 guards apply unchanged).
#   Cost vs b5spx2: +1 pretrain + SOUP_K fine-tunes (K=4: ~3 min + ~25 min, 3090) + larger 08 fits.
#
# LINEAGE OPTION 2026-09-23 (P9, sim-pretrained b5): BACKBONE=b5sp = the b5 lineage below with the
# shipping backbone initialised from a SIM pretrain (experiments/sp_sim_pretrain/notes.md, PIPELINE STAGE):
#   05a  NEW: sim pretrain, all 100 train_sim files  scripts/sp_pretrain_sim.py --iters 20000 --snapshots 10000
#        --stop-after 10000 -> data/checkpoints/sp/b5_sim_s3_it10k.pth (20k-it cosine stopped at 10k; the
#        state dict of a re-run was bit-equal to the validated snapshot on the 3090)
#   06   scripts/sp_finetune.py --init <05a snapshot> -- <the unchanged b1_finetune.py b5 args>
#        -> data/checkpoints/sp/b5sp_s3_all81.pth (new name, so the b5 guard never reuses the random-init ckpt)
#   07 / 08  unchanged scripts fed the 06 checkpoint; outputs in data/checkpoints/sp/ship/
#   05b / 09 IDENTICAL to BACKBONE=b5 (the W2 width net is fit on the RANDOM-INIT b5 K=4 all-81 pool, exactly
#        what shipped in 2026-10-06 and in the 2026-10-08 b5sp archive; every SP gate measured that net)
#   10   same package source (2026-10-06_...); the archive differs from b5 only in model.pth / prior.npz /
#        ke_net_flow.pt.  Needs data/competition/train_sim/ (100 files) in addition to train_real.
#
# LINEAGE REFRESH 2026-09-23 (P8, B1 backbone): the SHIPPING backbone is now B1's smaller
# b5_h192_blk6_rf11 (src/b1_models.py; SimVP1Stage enc 128 / hidden 192 / 6 gSTA blocks / RF 11,
# 3.39 M params) -- BACKBONE=b5 switches steps 05b / 06 / 07 / 08 / 09 / 10 to B1's recipe:
#   05b  K=4 all-81 OOS pool with b5 backbones  scripts/b1s_make_kfold_pool.sh (+ b1s_build_group.sh)
#   06   b5 all-81 backbone                     scripts/b1_finetune.py --model b5_h192_blk6_rf11
#   07   pack (fp32 + "arch" key)               scripts/p8_pack_b5.py
#   08   prior + ke_net_flow on ALL 81 files     scripts/b1_fit_prior.py / b1_fit_ke_flow.py
#   09   W2 v2 best-of-3 fit on the b5 caches   unchanged scripts, b5 cache paths
#   10   package source 2026-10-06_simvp-b5-... (its n1 branch is bit-exact to 2026-10-05)
# Steps 01-05 (splits, the three n1 LCO fold backbones / fits / streams) are unchanged: they are
# reference checks only (the W2 recipe fits on 05b, not on 05). BACKBONE=n1 restores the
# 2026-10-05 chain byte-for-byte except for the package source (whose n1 path is bit-exact).
#
# LINEAGE REFRESH 2026-09-22: package source -> the 2026-10-03 archive's tracked
# source; step 05b (KFOLD=all81) now ON by default; step 09 fits the W2 v2 width
# net on the 05b caches (scripts/w2_fit.py + scripts/w2_convert_ckpt.py) instead
# of the M1 net on the step-05 residuals (CONDWIDTH_RECIPE=m1 restores that).
# Previous lineage (2026-09-29 archive, M1 widths): git history of this file.
#
# R1 deliverable (docs/HANDOVER.md S7 item R1). Supersedes the audit-era
# skeleton `experiments/d3_repro/retrain_all.sh`, whose fix list is now applied
# (PR #3 `repro/fresh-clone-fixes`: `scripts/pack_zip.py`, the weight-free
# package source under `submissions/2026-09-29_.../source/`, and
# `scripts/split_data.py`'s 0.90/0.05/0.05 ratios reproducing the on-disk
# 73/4/4 `data/real` split are all tracked now).
#
# PREREQUISITES (not scripted here -- environment, not the chain):
#   1. `python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt`
#      (torch 2.2.2+cu121; see requirements.txt for the CUDA index-url line).
#   2. `data/competition/` populated per `data/README.md` (the released
#      train_real / train_sim / example_data / baseline_checkpoints; symlink or
#      download). Only `train_real` is used by BACKBONE=b5 / n1; BACKBONE=b5sp / b5spx2
#      also need `train_sim` (all 100 files, step 05a sim pretrain).
#   3. One CUDA GPU with >= 8 GB free. ~45-60 min end to end on an RTX 4090.
#      GPU memory (R3, docs/REPRODUCE.md "Prerequisites"): every process caps itself with
#      torch.cuda.set_per_process_memory_fraction(0.15-0.3) of the CARD (not of free memory); by default
#      steps 03 / 04 run 3 processes and 05b 2 at once. Tested on 24 GB cards; >= 16 GB expected to work.
#   4. SERIAL=1 (R2; default 0) runs the processes that steps 03 / 04 / 05b would start concurrently on one
#      GPU one after another -- needed on a GPU in EXCLUSIVE_PROCESS compute mode (common on HPC clusters).
#      Same commands / args / seeds; only the scheduling changes (+~40 min wall time on a 3090).
#   5. Resume safety (R4): a step is skipped only if its outputs exist AND each reused .pt / .pth / .npz /
#      .zip passes a zip-CRC read (non-zip .pt / .pth: torch.load); an unreadable one (e.g. a job killed
#      mid-save) is moved to <file>.corrupt and the step is redone. VERIFY_REUSE=0 = existence-only check.
#
# USAGE
#   bash scripts/retrain_all.sh                 # run every step, skipping done ones
#   bash scripts/retrain_all.sh --force         # redo everything from scratch
#   bash scripts/retrain_all.sh --only 06,07    # run just these steps
#   bash scripts/retrain_all.sh --from 05       # run step 05 onwards
#   bash scripts/retrain_all.sh --list          # print the step table and exit
#   PKG_NAME=my-archive bash scripts/retrain_all.sh   # package dir name
#   KFOLD=all81 bash scripts/retrain_all.sh --only 05b   # K-fold OOS data (see 05b)
#   KEEP_DUMPS=0 bash scripts/retrain_all.sh    # delete the ~4 GB fold streams
#                                               # after the width-net fit
#
# STEPS (each idempotent: skipped when its outputs exist, unless --force)
#   01 split        data/real/{train,val,test}                      73/4/4, seed 0
#   02 lco          data/real_lco/{fold0,fold1,fold2,foldR}         symlink folds
#   03 foldtrain    3 LCO backbones (fold0, fold1, foldR)           concurrent
#   04 foldfit      per-fold prior.npz + ke_net_flow.pt             concurrent
#   05 foldstream   3 held-out streams + official scoring           the out-of-sample
#                                                                   residuals the width
#                                                                   net is fitted on
#   05a simpretrain (BACKBONE=b5sp only) sim pretrain of the b5 backbone, all 100
#                   train_sim files, 20k-it cosine stopped at 10k, seed 0
#                   (BACKBONE=b5spx2: the same + a second pretrain with seed 1)
#   05b kfold       K-fold out-of-sample residual generation (KFOLD=all81 by
#                   default = the SHIPPING recipe; KFOLD= "" disables; or an LCO
#                   fold's TRAIN files, see below)
#   06 shiptrain    all-81-file S3 backbone
#                   (BACKBONE=b5spx2: 2 x SOUP_K fine-tunes -> 2 soups; 07 / 08 then pack /
#                   fit the K=2 ensemble into X2_DIR, see the ENRx header block)
#   07 shippack     fp16 pack -> model.pth
#   08 shipfit      shipping prior.npz + ke_net_flow.pt             train+val only
#   09 condwidth    W2 v2 conditional-width net on the 05b caches -> cond_width_weights.pt
#                   (CONDWIDTH_RECIPE=m1: the old M1 fit on the step-05 residuals)
#   10 package      assemble + scripts/pack_zip.py -> submission.zip
#   11 localeval    starting_kit/local_eval.py on data/real/{val,test}
#
# DETERMINISM / HONESTY
#   * Every training and fit runs with a fixed seed (0) and the same recipe the
#     manifests record; `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` everywhere.
#   * `data/competition/train_real/7575_0.h5` (corrupt) is never used: it is
#     excluded by `scripts/split_data.py`, by every LCO split JSON, and
#     defensively by `scripts/lco_make_splits.py` and `c11a_fit_ke_flow.py`.
#   * Each LCO fold's backbone, prior and energy net see that fold's TRAIN files
#     only; the held-out files are streamed but never fitted on.
#   * The conditional-width net is fitted on the OUT-OF-SAMPLE residuals of
#     those held-out streams (step 05), never on the shipped backbone's own
#     training data.
#   * No evaluation file is read anywhere in the packaged submission code.
#
# OUTPUT
#   submissions/$PKG_NAME/submission.zip  + SHA256SUMS + retrain_report.txt
#   one log per step under logs/
# =============================================================================

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

BACKBONE="${BACKBONE:-b5spx3}"             # b5spx3 (DEFAULT; bare run = freeze target FZ-EIKC+BVR+LOWKE+KD6, see FREEZE below) | b5spx2 | b5sp (09-23, 74f9d98f) | b5 (P8 lineage) | n1 (2026-10-05 lineage)
B5FAM=0                                    # 1 for the b5 family (b5, b5sp): shared 05b / 09 / W2_OUT / arch
# K3F: BACKBONE=b5spx3 = the b5spx2 lineage + a third member (X3=1); every b5spx2 step / guard applies unchanged.
X3=0
if [ "$BACKBONE" = "b5spx3" ]; then X3=1; BACKBONE=b5spx2; fi
# FREEZE (2026-09-26, code freeze 12:00 UTC): under b5spx3 a bare `bash scripts/retrain_all.sh` builds the INTENDED RETRAIN
# TARGET = FZ-EIKC + BVR + LOWKE + KD6 + KD7 + LSK(ρ 0.97, REC reset) + KD8 (zip f11d2c1d…, board 09-27; docs/REPRODUCE.md). Every flag can still be overridden (e.g. BVR=0);
# FREEZE_DEFAULTS=0 restores the pre-freeze defaults (bare run = K3F-KD3, the 09-24 board row f39128a9).
if [ "$X3" = "1" ] && [ "${FREEZE_DEFAULTS:-1}" = "1" ]; then
  : "${AXE:=1}" "${IPC:=1}" "${KD5:=1}" "${KD4:=1}" "${REC:=1}" "${IPW:=1}" "${KE3:=1}" "${CDIO:=1}" "${BVR:=1}" "${LOWKE:=1}" "${KD6:=1}" "${KD7:=1}" "${LSK:=1}" "${LSKR:=1}" "${KD8:=1}" "${LSKR97:=1}"
fi
case "$BACKBONE" in
  b5) PKG_NAME="${PKG_NAME:-retrained_simvp-b5-s3-bundle-condwidth-v2-wholestep-t2-w01-kd}"
      SHIP_DIR="data/checkpoints/b1_3090/ship"
      SHIP_CKPT="$SHIP_DIR/b5_s3_all81.pth"
      B5_MODEL="b5_h192_blk6_rf11"; B5FAM=1 ;;
  b5sp) PKG_NAME="${PKG_NAME:-retrained_simvp-b5sp-s3-bundle-condwidth-v2-wholestep-t2-w01-kd}"
      SHIP_DIR="data/checkpoints/sp/ship"
      SHIP_CKPT="data/checkpoints/sp/b5sp_s3_all81.pth"
      SP_INIT="data/checkpoints/sp/b5_sim_s3_it10k.pth"
      B5_MODEL="b5_h192_blk6_rf11"; B5FAM=1 ;;
  b5spx2)   # c2 ENRx; B5FAM=1 for 05b / 09 / W2_OUT only -- steps 06 / 07 / 08 / 10 have their own b5spx2 branches
      SOUP_K="${SOUP_K:-4}"                  # this lineage's default (the gated ENRx = soup4 members)
      ENRX_KE="${ENRX_KE:-0}"                # 1 = KE 4-seed packed ke net (NOT gated for ENRx); 0 = the gated single ke net
      X2_SFX=""; [ "$ENRX_KE" = "1" ] && X2_SFX="-kes4"
      PKG_NAME="${PKG_NAME:-retrained_simvp-b5spx2-soup${SOUP_K}${X2_SFX}-s3-bundle-condwidth-v2-wholestep-t2-w01-kd}"
      SP_INIT="data/checkpoints/sp/b5_sim_s3_it10k.pth"          # pretrain seed 0 (= b5sp's SP_INIT)
      SP_INIT1="data/checkpoints/sp/b5_sim_s3_seed1_it10k.pth"   # pretrain seed 1
      B5_MODEL="b5_h192_blk6_rf11"; B5FAM=1 ;;
  n1) PKG_NAME="${PKG_NAME:-retrained_simvp-s3-bundle-condwidth-v2-wholestep-t2-w01-kd}"
      SHIP_DIR="data/checkpoints/ship_simvp_s3"
      SHIP_CKPT="$SHIP_DIR/simvp_s3_all81.pth" ;;
  *) echo "unknown BACKBONE=$BACKBONE (b5 | b5sp | b5spx2 | b5spx3 | n1)"; exit 2 ;;
esac
# c2_soup (2026-09-23, 4090): SOUP_K > 1 (b5sp only) ships a UNIFORM WEIGHT SOUP of SOUP_K all-81 fine-tunes from the
# same sim-pretrained init (seeds 0..SOUP_K-1; seed 0 = the unchanged b5sp shipping ckpt). Zero inference cost: same arch,
# one set of weights; steps 07 / 08 / 10 consume the soup like any backbone. Evidence: experiments/c2_soup/report.md.
SOUP_K="${SOUP_K:-1}"
if [ "$BACKBONE" = "b5sp" ] && [ "$SOUP_K" -gt 1 ]; then
  SHIP_MEMBER0="$SHIP_CKPT"
  SHIP_CKPT="data/checkpoints/sp/b5sp_soup${SOUP_K}_s3_all81.pth"
  SHIP_DIR="data/checkpoints/sp/ship_soup${SOUP_K}"   # own dir: steps 07 / 08 / 10 must not reuse the single-seed pack / fits
  PKG_NAME="${PKG_NAME/simvp-b5sp-/simvp-b5sp-soup${SOUP_K}-}"
fi
SRC_PKG="$REPO/submissions/2026-10-06_simvp-b5-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"
# CB centre blend (2026-09-23, experiments/cb_centre_blend/notes.md + experiments/cbp_pkg/notes.md): the soup lineage
# (BACKBONE=b5sp, SOUP_K > 1) ships the CB package source -- identical to 2026-10-06's except submission.py (CONFIG
# cb_blend_w = [0.75] * 20: emitted bounds += 0.75 * (raw backbone - interval centre), pred unchanged) and
# cuda_graph_wholestep.py (the same shift inside the whole-step graph). No new weights / fits: the constant lives in
# the tracked source. CB=0 reproduces the pre-CB soup archive (2b7dfe3c); PKG_NAME / OUT_DIR get "-cb" so step 10's
# idempotency guard never reuses a pre-CB archive.
CB="${CB:-1}"
if [ "$BACKBONE" = "b5sp" ] && [ "$SOUP_K" -gt 1 ] && [ "$CB" = "1" ]; then
  SRC_PKG="$REPO/submissions/2026-10-10_simvp-b5sp-soup4-cb-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"
  PKG_NAME="${PKG_NAME/simvp-b5sp-soup${SOUP_K}-/simvp-b5sp-soup${SOUP_K}-cb-}"
fi
ARCH_EXPECT="n1_simvp"; [ "$B5FAM" = "1" ] && ARCH_EXPECT="$B5_MODEL"   # step 10's model.pth "arch" check
# c2 ENRx (BACKBONE=b5spx2): two soups from two different SP inits, packed as ENR's K=2 ensemble. Every path that a
# step 06-10 guard tests is b5spx2-specific (X2_DIR, keyed by SOUP_K and ENRX_KE) except the soup member fine-tunes and
# member 0's soup, which are the b5sp / b5sp-SOUP_K artefacts of the IDENTICAL recipe (same init, same args, same seeds).
if [ "$BACKBONE" = "b5spx2" ]; then
  case "$SOUP_K" in ''|*[!0-9]*|0) echo "SOUP_K must be a positive integer (got '$SOUP_K')"; exit 2 ;; esac
  X2_FT0=""; X2_FT1=""                     # fine-tunes from pretrain seed 0 / seed 1 (fine-tune seeds 0..SOUP_K-1)
  for s in $(seq 0 $((SOUP_K - 1))); do
    if [ "$s" = 0 ]; then X2_FT0="data/checkpoints/sp/b5sp_s3_all81.pth"      # = the b5sp ship ckpt
    else X2_FT0="$X2_FT0 data/checkpoints/sp/b5sp_s3_all81_seed$s.pth"; fi    # = the b5sp SOUP_K members
    X2_FT1="$X2_FT1 data/checkpoints/sp/b5sp_pre1_s3_all81_seed$s.pth"
  done
  if [ "$SOUP_K" -gt 1 ]; then
    X2_M0="data/checkpoints/sp/b5sp_soup${SOUP_K}_s3_all81.pth"             # = the b5sp SOUP_K soup path
    X2_M1="data/checkpoints/sp/b5sp_pre1_soup${SOUP_K}_s3_all81.pth"
  else                                     # SOUP_K=1: ensemble of the two seed-0 fine-tunes, no soup
    X2_M0="data/checkpoints/sp/b5sp_s3_all81.pth"
    X2_M1="data/checkpoints/sp/b5sp_pre1_s3_all81_seed0.pth"
  fi
  X2_DIR="data/checkpoints/sp/ship_enrx_soup${SOUP_K}"
  [ "$ENRX_KE" = "1" ] && X2_DIR="${X2_DIR}_kes4"
  SHIP_DIR="$X2_DIR"                       # never data/checkpoints/sp/ship{,_soupK}: 07 / 08 / 10 cannot reuse b5sp outputs
  SHIP_CKPT="$X2_M0"                       # informational (DRY_RUN / report); 07 / 08 use X2_M0 + X2_M1
  SRC_PKG="$REPO/submissions/2026-10-07_simvp-b5x2-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"   # ENR source
  # CB centre blend on ENRx (2026-09-23, experiments/cbx_pkg/notes.md): the ENR source + the CB hunks, identical to the
  # ENR source except submission.py (CONFIG cb_blend_w = [0.75] * 20: emitted bounds += 0.75 * (raw ensemble-mean
  # backbone - interval centre), pred unchanged) and cuda_graph_wholestep.py (the same shift inside the whole-step graph;
  # byte-identical to the soup4-cb source's). No new weights / fits. CB=0 reproduces the pre-CB ENRx archive (894f044e);
  # PKG_NAME / OUT_DIR get "-cb" so step 10's idempotency guard never reuses a pre-CB archive.
  if [ "$CB" = "1" ]; then
    SRC_PKG="$REPO/submissions/2026-10-11_simvp-b5spx2-soup4-cb-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"
    PKG_NAME="${PKG_NAME/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cb-}"
    # B2 per-lead W2 half-width multiplier on top of CB (2026-09-23, experiments/cb2_interval/notes.md arm B2, packaged by
    # experiments/b2p_pkg/notes.md): the CB source + ONE submission.py hunk (CONFIG cw_lead_lambda = [0.775, 0.9] +
    # [0.975] * 18, folded into cw.lam at load time). The 20 constants live in the tracked source; they were fitted by
    # scripts/b2p_fit.py (exact official sps of the 3 LCO pipelines' dumps via scripts/b2p_tables.py, monotone over
    # leads; experiments/b2p_pkg/out/lambda_fit.json "ship") and are NOT re-fitted by this chain (like CB's w = 0.75).
    # No new weights / fits. B2=0 reproduces the CB archive (f725611e lineage); PKG_NAME gets "-cb-b2-".
    B2="${B2:-0}"   # default OFF: FR13 NO-GO (negative beyond the training range on foldS lo/hi)
    if [ "$B2" = "1" ]; then
      SRC_PKG="$REPO/submissions/2026-10-12_simvp-b5spx2-soup4-cb-b2-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"
      PKG_NAME="${PKG_NAME/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cb-/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cb-b2-}"
    fi
    # AW causal per-trajectory CB weight (2026-09-24, experiments/aw_adaptive_cb/notes.md, packaged by
    # experiments/awp_pkg/notes.md): the CB source + AW's patch (new cb_adapt.py; submission.py / cuda_graph_wholestep.py /
    # t2_cw_graph_body.py hunks) with CONFIG cb_adapt = {"grid": [0.30, 0.45, 0.60, 0.75, 0.90], "rho": 0.95}: from window 2
    # of a trajectory the CB weight is the grid w that scored best (EMA) on the already-revealed windows; windows 0/1 keep
    # 0.75. Grid / rho live in the tracked source (pre-declared, selected on LCO fold1 by AW) and are NOT fitted by this
    # chain. No new weights / fits. CB_ADAPT=0 reproduces the CB archive (f725611e lineage); PKG_NAME gets "-cbadapt-".
    # There is no B2 + adaptive source: B2=1 requires CB_ADAPT=0.
    CB_ADAPT="${CB_ADAPT:-1}"
    if [ "$CB_ADAPT" = "1" ]; then
      [ "$B2" = "1" ] && { echo "FATAL: B2=1 and CB_ADAPT=1 have no combined source (set CB_ADAPT=0 for B2)"; exit 2; }
      SRC_PKG="$REPO/submissions/2026-10-12_simvp-b5spx2-soup4-cbadapt-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"
      PKG_NAME="${PKG_NAME/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cb-/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-}"
      # AX causal per-trajectory W2 width multiplier + energy blend (2026-09-24, experiments/ax_adaptive/notes.md, packaged
      # by experiments/axp_pkg/notes.md): the cbadapt source + AX's patch (new ax_adapt.py; submission.py /
      # cuda_graph_megatail.py / cuda_graph_wholestep.py hunks) with CONFIG kw_adapt = {"grid": [0.85, 0.925, 1.0, 1.075,
      # 1.15], "groups": [[0, 1], list(range(2, 20))], "rho": 0.95} and eb_adapt = {"grid": [0.75, 0.85, 0.925], "rho": 0.95}.
      # Grids / rho live in the tracked source (pre-declared, selected on LCO fold1 by AX) and are NOT fitted by this chain.
      # No new weights / fits. AX=0 reproduces the cbadapt archive (e67ff687 lineage); PKG_NAME gets "-cbadapt-ax-".
      AX="${AX:-1}"
      if [ "$AX" = "1" ]; then
        SRC_PKG="$REPO/submissions/2026-10-13_simvp-b5spx2-soup4-cbadapt-ax-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"
        PKG_NAME="${PKG_NAME/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-ax-}"
        # AXE extended energy-blend grid + causal per-trajectory blend_ridge + BR window-0 CB fallback (2026-09-24,
        # experiments/axe_extended/notes.md E4, packaged by experiments/axep_pkg/notes.md): the AX source + AXE's hunks
        # (new axe_adapt.py; submission.py / cuda_graph_megatail.py / cuda_graph_wholestep.py) with CONFIG
        # cb_blend_w0 = 1.0, eb_adapt = {"grid": [0.45, 0.60, 0.75, 0.85, 0.925], "rho": 0.95} and
        # br_adapt = {"grid": [0.125, 0.25, 0.5, 1.0], "rho": 0.95}. Grids / rho live in the tracked source (pre-declared,
        # selected on LCO fold1 by AXE) and are NOT fitted by this chain. No new weights / fits. Default OFF (orchestrator
        # decides after review); AXE=0 reproduces the AX archive (43d73243 lineage); PKG_NAME gets "-cbadapt-axe-".
        AXE="${AXE:-0}"
        if [ "$AXE" = "1" ]; then
          SRC_PKG="$REPO/submissions/2026-10-14_simvp-b5spx2-soup4-cbadapt-axe-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"
          PKG_NAME="${PKG_NAME/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-ax-/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-axe-}"
        fi
        # AE by-product gain_max 20 (experiments/ae_energy_adapt/notes.md "Extra"; FR16; packaged by experiments/axg_pkg/notes.md
        # (AX) and experiments/axep_pkg/notes.md (AXE)): the single CONFIG constant gain_max 5.0 -> 20.0 (energy-stage gain clip;
        # read at graph capture, 0 kernels). Not fitted by this chain. Default OFF (orchestrator decides); GM20=0 reproduces the
        # non-gm20 archive; PKG_NAME gets "-gm20-" after "-cbadapt-ax-" / "-cbadapt-axe-".
        GM20="${GM20:-1}"
        if [ "$GM20" = "1" ]; then
          if [ "$AXE" = "1" ]; then
            SRC_PKG="$REPO/submissions/2026-10-14_simvp-b5spx2-soup4-cbadapt-axe-gm20-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"
            PKG_NAME="${PKG_NAME/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-axe-/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-axe-gm20-}"
          else
            SRC_PKG="$REPO/submissions/2026-10-14_simvp-b5spx2-soup4-cbadapt-ax-gm20-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"
            PKG_NAME="${PKG_NAME/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-ax-/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-ax-gm20-}"
          fi
        fi
        # MSK exact scored mask (2026-09-24, experiments/msk_mask/notes.md, packaged by experiments/mskp_pkg/notes.md): the gm20
        # source + MSK's 5-file change (cond_width_net._scored + CONFIG "exact_mask": True): the "scored" mask on the revealed
        # target (W2 prev-error map, AW / AX scorers) becomes |pt| > 1e-6 (the scorer's t != 0 on the physical target) instead
        # of pt != 0.0. Interval-only; 1 kernel. No constant, no fit. Default OFF (orchestrator decides); MSK=0 reproduces the
        # gm20 archive; PKG_NAME gets "-msk-" after "-gm20-". Requires GM20=1 (msk sources exist on the gm20 sources only).
        # AXE=1 MSK=1 -> the AXE-gm20 source + the same 5-file change (packaged as source only by MSKP, not LCO-validated).
        MSK="${MSK:-1}"   # default ON after FR18 GO (2026-09-24)
        if [ "$MSK" = "1" ] && [ "$GM20" = "1" ]; then
          if [ "$AXE" = "1" ]; then
            SRC_PKG="$REPO/submissions/2026-10-15_simvp-b5spx2-soup4-cbadapt-axe-gm20-msk-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"
            PKG_NAME="${PKG_NAME/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-axe-gm20-/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-axe-gm20-msk-}"
          else
            SRC_PKG="$REPO/submissions/2026-10-15_simvp-b5spx2-soup4-cbadapt-ax-gm20-msk-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"
            PKG_NAME="${PKG_NAME/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-ax-gm20-/simvp-b5spx2-soup${SOUP_K}${X2_SFX}-cbadapt-ax-gm20-msk-}"
          fi
        fi
      fi
    fi
    # AX needs AW's cb_adapt (its width score is taken at AW's emitted centre): no AX source without CB_ADAPT=1.
    [ "$CB_ADAPT" != "1" ] && [ "${AX:-0}" = "1" ] && { echo "FATAL: AX=1 requires CB_ADAPT=1 (no AX source without AW)"; exit 2; }
    # AXE's rows extend AX-B's eb_adapt (br_adapt requires it): no AXE source without CB_ADAPT=1 AND AX=1.
    [ "${AXE:-0}" = "1" ] && { [ "$CB_ADAPT" != "1" ] || [ "${AX:-0}" != "1" ]; } && { echo "FATAL: AXE=1 requires CB_ADAPT=1 and AX=1 (no AXE source without AX)"; exit 2; }
    # gain_max 20 is packaged only on the AX / AXE sources: no gm20 source without CB_ADAPT=1 AND AX=1.
    [ "${GM20:-0}" = "1" ] && { [ "$CB_ADAPT" != "1" ] || [ "${AX:-0}" != "1" ]; } && { echo "FATAL: GM20=1 requires CB_ADAPT=1 and AX=1 (gm20 archives exist on the AX / AXE sources only)"; exit 2; }
    # the exact-mask change is packaged only on the gm20 sources: no msk source without CB_ADAPT=1, AX=1 AND GM20=1.
    [ "${MSK:-0}" = "1" ] && { [ "$CB_ADAPT" != "1" ] || [ "${AX:-0}" != "1" ] || [ "${GM20:-0}" != "1" ]; } && { echo "FATAL: MSK=1 requires CB_ADAPT=1, AX=1 and GM20=1 (msk archives exist on the AX-gm20 / AXE-gm20 sources only)"; exit 2; }
  fi
  ARCH_EXPECT="b5x2_h192_blk6_rf11"
  # K3F (BACKBONE=b5spx3, X3=1): third member from pretrain seed 2 + FP16X fp16 core. The only packaged source is the
  # shipped flag set (CB / CB_ADAPT / AX / GM20 / MSK on, AXE / B2 / ENRX_KE off) with "fp16_core": True.
  # K3E: AXE=1 is also packaged (the K3F source + AXE's E4 delta); AXE=0 (default) = the K3F source, unchanged.
  FP16="${FP16:-$X3}"
  if [ "$X3" = "1" ]; then
    { [ "$CB" = "1" ] && [ "${CB_ADAPT:-0}" = "1" ] && [ "${AX:-0}" = "1" ] && [ "${GM20:-0}" = "1" ] && [ "${MSK:-0}" = "1" ] \
      && { [ "${AXE:-0}" = "0" ] || [ "${AXE:-0}" = "1" ]; } && [ "${B2:-0}" = "0" ] && [ "$ENRX_KE" = "0" ]; } \
      || { echo "FATAL: BACKBONE=b5spx3 exists only for CB=1 CB_ADAPT=1 AX=1 GM20=1 MSK=1 AXE=0|1 B2=0 ENRX_KE=0"; exit 2; }
    [ "$FP16" = "1" ] || { echo "FATAL: BACKBONE=b5spx3 requires FP16=1 (no TF32 K=3 source is packaged: +0.26 ms, experiments/k3_ensemble)"; exit 2; }
    SP_INIT2="data/checkpoints/sp/b5_sim_s3_seed2_it10k.pth"   # pretrain seed 2
    X2_FT2=""
    for s in $(seq 0 $((SOUP_K - 1))); do X2_FT2="$X2_FT2 data/checkpoints/sp/b5sp_pre2_s3_all81_seed$s.pth"; done
    if [ "$SOUP_K" -gt 1 ]; then X2_M2="data/checkpoints/sp/b5sp_pre2_soup${SOUP_K}_s3_all81.pth"
    else X2_M2="data/checkpoints/sp/b5sp_pre2_s3_all81_seed0.pth"; fi
    X2_DIR="data/checkpoints/sp/ship_enrx3_soup${SOUP_K}"    # never the K=2 X2_DIR
    SHIP_DIR="$X2_DIR"
    SRC_PKG="$REPO/submissions/2026-10-16_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"
    [ "${AXE:-0}" = "1" ] && SRC_PKG="$REPO/submissions/2026-10-16_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"   # K3E
    PKG_NAME="${PKG_NAME/simvp-b5spx2-soup${SOUP_K}-/simvp-b5spx3-soup${SOUP_K}-fp16-}"
    # KD2 kernel diet (experiments/kd2_adapt_diet): default ON; KD2=0 = the K3F / K3E sources above, unchanged.
    KD2="${KD2:-1}"
    { [ "$KD2" = "0" ] || [ "$KD2" = "1" ]; } || { echo "FATAL: KD2 must be 0 or 1 (got '$KD2')"; exit 2; }
    if [ "$KD2" = "1" ]; then
      if [ "${AXE:-0}" = "1" ]; then
        SRC_PKG="$REPO/submissions/2026-10-17_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd2-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"   # K3E-KD2
      else
        SRC_PKG="$REPO/submissions/2026-10-17_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-kd2-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"    # K3F-KD2
      fi
      PKG_NAME="${PKG_NAME/-gm20-msk-/-gm20-msk-kd2-}"
    fi
    # KD3 whole-step kernel diet (experiments/kd3_step_diet): default = KD2 (i.e. ON by default); requires KD2=1.
    KD3="${KD3:-$KD2}"
    { [ "$KD3" = "0" ] || [ "$KD3" = "1" ]; } || { echo "FATAL: KD3 must be 0 or 1 (got '$KD3')"; exit 2; }
    [ "$KD3" = "1" ] && [ "$KD2" != "1" ] && { echo "FATAL: KD3=1 requires KD2=1 (KD3 sources exist on the KD2 sources only)"; exit 2; }
    if [ "$KD3" = "1" ]; then
      if [ "${AXE:-0}" = "1" ]; then
        SRC_PKG="$REPO/submissions/2026-10-17_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd3-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"   # K3E-KD3
      else
        SRC_PKG="$REPO/submissions/2026-10-17_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-kd3-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"    # K3F-KD3
      fi
      [ -f "$SRC_PKG/kd3_step.py" ] || { echo "FATAL: KD3=1 but $SRC_PKG has no kd3_step.py"; exit 2; }
      PKG_NAME="${PKG_NAME/-gm20-msk-kd2-/-gm20-msk-kd3-}"
    fi
    # IPC one-storage return (experiments/ipc_outproc, K3E port experiments/k3ei_build): default OFF; requires KD3=1.
    IPC="${IPC:-0}"
    { [ "$IPC" = "0" ] || [ "$IPC" = "1" ]; } || { echo "FATAL: IPC must be 0 or 1 (got '$IPC')"; exit 2; }
    [ "$IPC" = "1" ] && [ "$KD3" != "1" ] && { echo "FATAL: IPC=1 requires KD3=1 (IPC sources exist on the KD3 sources only)"; exit 2; }
    if [ "$IPC" = "1" ]; then
      if [ "${AXE:-0}" = "1" ]; then
        SRC_PKG="$REPO/submissions/2026-10-18_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd3-ipc-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"   # K3E-KD3-IPC
      else
        SRC_PKG="$REPO/submissions/2026-10-18_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-kd3-ipc-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"    # K3F-KD3-IPC
      fi
      grep -q '"ipc_one_storage": "auto"' "$SRC_PKG/submission.py" 2>/dev/null || { echo "FATAL: IPC=1 but $SRC_PKG has no CONFIG ipc_one_storage auto"; exit 2; }
      PKG_NAME="${PKG_NAME/-gm20-msk-kd3-/-gm20-msk-kd3-ipc-}"
    fi
    # KD5 near-exact whole-step graph diet (experiments/kd5_more; + IPC experiments/k3eik_build): default OFF; requires
    # KD3=1. AXE=0 -> K3F-KD5, AXE=1 -> K3E-KD5, AXE=1 IPC=1 -> K3E-KD5-IPC (no AXE=0 IPC=1 source).
    KD5="${KD5:-0}"
    { [ "$KD5" = "0" ] || [ "$KD5" = "1" ]; } || { echo "FATAL: KD5 must be 0 or 1 (got '$KD5')"; exit 2; }
    # KD4 (W01 diet) / REC (regime-change recovery): default OFF; require KD3=1; with either on, the source is chosen in
    # the FZ block below (tracked combinations only) and the KD5 block's own choice is skipped.
    KD4="${KD4:-0}"; REC="${REC:-0}"
    { [ "$KD4" = "0" ] || [ "$KD4" = "1" ]; } || { echo "FATAL: KD4 must be 0 or 1 (got '$KD4')"; exit 2; }
    { [ "$REC" = "0" ] || [ "$REC" = "1" ]; } || { echo "FATAL: REC must be 0 or 1 (got '$REC')"; exit 2; }
    [ "$KD4" = "1" ] && [ "$KD3" != "1" ] && { echo "FATAL: KD4=1 requires KD3=1 (KD4 sources exist on the KD3 sources only)"; exit 2; }
    [ "$REC" = "1" ] && [ "$KD3" != "1" ] && { echo "FATAL: REC=1 requires KD3=1 (REC sources exist on the KD3 sources only)"; exit 2; }
    [ "$KD5" = "1" ] && [ "$KD3" != "1" ] && { echo "FATAL: KD5=1 requires KD3=1 (KD5 sources exist on the KD3 sources only)"; exit 2; }
    if [ "$KD5" = "1" ] && [ "$KD4$REC" = "00" ]; then
      if [ "$IPC" = "1" ]; then
        [ "${AXE:-0}" = "1" ] || { echo "FATAL: KD5=1 IPC=1 is packaged only with AXE=1 (no K3F-KD5-IPC source)"; exit 2; }
        SRC_PKG="$REPO/submissions/2026-10-18_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-ipc-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"   # K3E-KD5-IPC
      elif [ "${AXE:-0}" = "1" ]; then
        SRC_PKG="$REPO/submissions/2026-10-18_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"       # K3E-KD5
      else
        SRC_PKG="$REPO/submissions/2026-10-18_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-kd5-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"        # K3F-KD5
      fi
      grep -q '"kd5_near": True' "$SRC_PKG/submission.py" 2>/dev/null || { echo "FATAL: KD5=1 but $SRC_PKG has no CONFIG kd5_near True"; exit 2; }
      PKG_NAME="${PKG_NAME/-gm20-msk-kd3-/-gm20-msk-kd5-}"
    fi
    # FZ block (experiments/fz_freeze): KD4=1 or REC=1 -> one of the tracked sources, keyed by AXE KD5 KD4 REC IPC.
    if [ "$KD4$REC" != "00" ]; then
      case "${AXE:-0}$KD5$KD4$REC$IPC" in
        11111) SRC_PKG="$REPO/submissions/2026-10-19_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-rec-ipc-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"; FZ_TAG="kd5-kd4-rec-ipc" ;;   # FZ-E
        01111) SRC_PKG="$REPO/submissions/2026-10-19_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-kd5-kd4-rec-ipc-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"; FZ_TAG="kd5-kd4-rec-ipc" ;;    # FZ-F
        00100) SRC_PKG="$REPO/submissions/2026-10-17_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-kd4-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"; FZ_TAG="kd4" ;;       # K3F-KD4
        10010) SRC_PKG="$REPO/submissions/2026-10-18_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd3-rec-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"; FZ_TAG="kd3-rec" ;;  # K3E-KD3-REC
        00010) SRC_PKG="$REPO/submissions/2026-10-18_simvp-b5spx3-soup4-fp16-cbadapt-ax-gm20-msk-kd3-rec-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source"; FZ_TAG="kd3-rec" ;;   # K3F-KD3-REC
        *) echo "FATAL: no tracked source for AXE=${AXE:-0} KD5=$KD5 KD4=$KD4 REC=$REC IPC=$IPC (with KD4=1 or REC=1 only 1 1 1 1 1, 0 1 1 1 1, 0 0 1 0 0, x 0 0 1 0 exist)"; exit 2 ;;
      esac
      [ "$KD4" = "1" ] && { grep -q '"kd4_w01": True' "$SRC_PKG/submission.py" 2>/dev/null || { echo "FATAL: KD4=1 but $SRC_PKG has no CONFIG kd4_w01 True"; exit 2; }; }
      [ "$REC" = "1" ] && { grep -q '"rec_detect": {' "$SRC_PKG/submission.py" 2>/dev/null || { echo "FATAL: REC=1 but $SRC_PKG has no CONFIG rec_detect"; exit 2; }; }
      PKG_NAME="${PKG_NAME/-gm20-msk-kd3-ipc-/-gm20-msk-kd3-}"
      PKG_NAME="${PKG_NAME/-gm20-msk-kd3-/-gm20-msk-$FZ_TAG-}"
    fi
    # KE3 (2026-09-24, c2: experiments/c2_b2/notes.md, experiments/c2_ke3/notes.md): KE3=1 (default 0) replaces the single K=3
    # ensemble ke net by KE's 4-seed energy-net ensemble, written by scripts/c2_ke_dense.py as ONE standard KeFlowNet
    # (width 4w / wide 4W, block-diagonal = exactly the 4-seed mean): a weight-file change only -- the SAME source package,
    # the unmodified single-net load path and KD3 gate, TF32 tensor-core convs. Outputs in ${X2_DIR}_kes4 (07 re-packs
    # model.pth there from the step-06 members; no member retraining). PKG_NAME gets "-kes4-".
    # IPW (2026-09-25, c2: experiments/c2_ipw/notes.md): IPW=1 (default 0) packages the c2 twin of FZ-E / FZ-F
    # (submissions/2026-10-20_c2-…-kd5-kd4-rec-ipc-ipw-…) = the FZ source + CONFIG "ipc_prewarm": True and a 12-line untimed
    # out-of-process warm-up in get_ttt_model (outputs identical). Only on the FZ combinations (AXE=0|1 KD5=1 KD4=1 REC=1 IPC=1).
    IPW="${IPW:-0}"
    { [ "$IPW" = "0" ] || [ "$IPW" = "1" ]; } || { echo "FATAL: IPW must be 0 or 1 (got '$IPW')"; exit 2; }
    if [ "$IPW" = "1" ]; then
      case "${AXE:-0}$KD5$KD4$REC$IPC" in
        11111|01111) SRC_PKG="${SRC_PKG/\/2026-10-19_simvp-/\/2026-10-20_c2-simvp-}"; SRC_PKG="${SRC_PKG/-rec-ipc-s3-bundle-/-rec-ipc-ipw-s3-bundle-}" ;;
        *) echo "FATAL: IPW=1 is packaged only on FZ-E / FZ-F (AXE=0|1 KD5=1 KD4=1 REC=1 IPC=1)"; exit 2 ;;
      esac
      grep -q '"ipc_prewarm": True' "$SRC_PKG/submission.py" 2>/dev/null || { echo "FATAL: IPW=1 but $SRC_PKG has no CONFIG ipc_prewarm True"; exit 2; }
      PKG_NAME="${PKG_NAME/-rec-ipc-/-rec-ipc-ipw-}"
    fi
    # CDIO (2026-09-25, HX2 lever ported by CDP: experiments/hx2_harness/notes.md, experiments/cdp_port/notes.md): CDIO=1
    # (default 0; requires IPW=1, FZ-E only) packages submissions/2026-10-21_simvp-…-kd5-kd4-rec-ipc-ipw-cdio-… = the FZ-E
    # IPW source + CONFIG "cuda_direct_io": True (cuda_graph_wholestep.py: CUDA-input fast path, D2D in / CUDA clones out,
    # active ONLY when ttt_step receives CUDA tensors; CPU inputs take the unchanged path; outputs bit-identical).
    # Code only: no new weights / fits. PKG_NAME "-rec-ipc-ipw-" -> "-rec-ipc-ipw-cdio-". No FZ-F (AXE=0) twin: FATAL.
    CDIO="${CDIO:-0}"
    { [ "$CDIO" = "0" ] || [ "$CDIO" = "1" ]; } || { echo "FATAL: CDIO must be 0 or 1 (got '$CDIO')"; exit 2; }
    if [ "$CDIO" = "1" ]; then
      [ "$IPW" = "1" ] || { echo "FATAL: CDIO=1 requires IPW=1 (the CDIO source exists on the FZ-E IPW source only)"; exit 2; }
      case "${AXE:-0}$KD5$KD4$REC$IPC" in
        11111) SRC_PKG="${SRC_PKG/\/2026-10-20_c2-simvp-/\/2026-10-21_simvp-}"; SRC_PKG="${SRC_PKG/-rec-ipc-ipw-s3-bundle-/-rec-ipc-ipw-cdio-s3-bundle-}" ;;
        *) echo "FATAL: CDIO=1 is packaged only on FZ-E (AXE=1 KD5=1 KD4=1 REC=1 IPC=1 IPW=1; no FZ-F / AXE=0 CDIO source)"; exit 2 ;;
      esac
      grep -q '"cuda_direct_io": True' "$SRC_PKG/submission.py" 2>/dev/null || { echo "FATAL: CDIO=1 but $SRC_PKG has no CONFIG cuda_direct_io True"; exit 2; }
      grep -q '"ipc_prewarm": True' "$SRC_PKG/submission.py" 2>/dev/null || { echo "FATAL: CDIO=1 but $SRC_PKG has no CONFIG ipc_prewarm True"; exit 2; }
      PKG_NAME="${PKG_NAME/-rec-ipc-ipw-/-rec-ipc-ipw-cdio-}"
    fi
    # BVR (2026-09-25, c2: experiments/c2_d3/notes.md, experiments/c2_v27/notes.md, review experiments/c2_review/fr_c2_bvr.md):
    # BVR=1 (default 0; requires IPW=1 AXE=1) packages the c2 BVR twin of the selected FZ-E source = that source + CONFIG
    # "ipc_byvalue": True and a by-value output wrapper at the end of get_ttt_model (active only in a multiprocessing child, CPU
    # outputs only; values identical). With CDIO=1: submissions/2026-10-22_c2-…-ipw-cdio-bvr-…; with CDIO=0:
    # submissions/2026-10-21_c2-…-ipw-bvr-…. PKG_NAME gets "-bvr-" after "-ipw[-cdio]-". Code only: no new weights / fits.
    BVR="${BVR:-0}"
    { [ "$BVR" = "0" ] || [ "$BVR" = "1" ]; } || { echo "FATAL: BVR must be 0 or 1 (got '$BVR')"; exit 2; }
    if [ "$BVR" = "1" ]; then
      { [ "${IPW:-0}" = "1" ] && [ "${AXE:-0}" = "1" ]; } || { echo "FATAL: BVR=1 is packaged only with IPW=1 AXE=1 (FZ-E IPW twins)"; exit 2; }
      if [ "${CDIO:-0}" = "1" ]; then
        SRC_PKG="${SRC_PKG/\/2026-10-21_simvp-/\/2026-10-22_c2-simvp-}"; SRC_PKG="${SRC_PKG/-rec-ipc-ipw-cdio-s3-bundle-/-rec-ipc-ipw-cdio-bvr-s3-bundle-}"
        PKG_NAME="${PKG_NAME/-rec-ipc-ipw-cdio-/-rec-ipc-ipw-cdio-bvr-}"
      else
        SRC_PKG="${SRC_PKG/\/2026-10-20_c2-simvp-/\/2026-10-21_c2-simvp-}"; SRC_PKG="${SRC_PKG/-rec-ipc-ipw-s3-bundle-/-rec-ipc-ipw-bvr-s3-bundle-}"
        PKG_NAME="${PKG_NAME/-rec-ipc-ipw-/-rec-ipc-ipw-bvr-}"
      fi
      grep -q '"ipc_byvalue": True' "$SRC_PKG/submission.py" 2>/dev/null || { echo "FATAL: BVR=1 but $SRC_PKG has no CONFIG ipc_byvalue True"; exit 2; }
    fi
    # LOWKE (2026-09-25, TKA: experiments/tka_tke/notes.md; packaged by experiments/lkb_build/notes.md): LOWKE=1 (default 0;
    # requires CDIO=1, i.e. the FZ-E IPW CDIO combos, with or without BVR) packages the TKA low-ke fluctuation damping twin of the
    # selected FZ-E CDIO source = that source + CONFIG "tka_lowke": {"theta": 1.5, "delta": 0.25, "gmin": 0.0} + cuda_graph_wholestep.py
    # `_lowke_apply` (output-only, whole-step graph windows >= 2; bounds / internal state untouched). BVR=0:
    # submissions/2026-10-22_simvp-…-ipw-cdio-tka-lowke; BVR=1: submissions/2026-10-22_simvp-…-ipw-cdio-bvr-lowke.
    # PKG_NAME "-ipw-cdio[-bvr]-" -> "-ipw-cdio[-bvr]-lowke-". Code only: no new weights / fits. KD6 twins: KD6 block below.
    LOWKE="${LOWKE:-0}"
    { [ "$LOWKE" = "0" ] || [ "$LOWKE" = "1" ]; } || { echo "FATAL: LOWKE must be 0 or 1 (got '$LOWKE')"; exit 2; }
    if [ "$LOWKE" = "1" ]; then
      [ "${CDIO:-0}" = "1" ] || { echo "FATAL: LOWKE=1 requires CDIO=1 (the LOWKE sources exist on the FZ-E CDIO sources only)"; exit 2; }
      if [ "${BVR:-0}" = "1" ]; then
        SRC_PKG="$REPO/submissions/2026-10-22_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-rec-ipc-ipw-cdio-bvr-lowke/source"
        grep -q '"ipc_byvalue": True' "$SRC_PKG/submission.py" 2>/dev/null || { echo "FATAL: LOWKE=1 BVR=1 but $SRC_PKG has no CONFIG ipc_byvalue True"; exit 2; }
        PKG_NAME="${PKG_NAME/-rec-ipc-ipw-cdio-bvr-/-rec-ipc-ipw-cdio-bvr-lowke-}"
      else
        SRC_PKG="$REPO/submissions/2026-10-22_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-rec-ipc-ipw-cdio-tka-lowke/source"
        ! grep -q '"ipc_byvalue"' "$SRC_PKG/submission.py" 2>/dev/null || { echo "FATAL: LOWKE=1 BVR=0 but $SRC_PKG carries ipc_byvalue"; exit 2; }
        PKG_NAME="${PKG_NAME/-rec-ipc-ipw-cdio-/-rec-ipc-ipw-cdio-lowke-}"
      fi
      { grep -q '"tka_lowke": {"theta": 1.5, "delta": 0.25, "gmin": 0.0}' "$SRC_PKG/submission.py" 2>/dev/null \
        && grep -q '_lowke_apply' "$SRC_PKG/cuda_graph_wholestep.py" && grep -q '"cuda_direct_io": True' "$SRC_PKG/submission.py" \
        && grep -q '"ipc_prewarm": True' "$SRC_PKG/submission.py"; } \
        || { echo "FATAL: LOWKE=1 but $SRC_PKG has no CONFIG tka_lowke {1.5, 0.25, 0.0} / _lowke_apply / cuda_direct_io / ipc_prewarm"; exit 2; }
    fi
    # KD6 (2026-09-25, experiments/kd6_concurrency/notes.md): KD6=1 (default 0; requires CDIO=1, FZ-E only) packages the KD6
    # twin of the selected FZ-E CDIO source (with / without BVR, with / without LOWKE) = that source + CONFIG
    # "kd6_concurrency": True + kd6_conc.py + cuda_graph_wholestep.py's _kd6_build: fork / join concurrency inside the steady
    # noscan whole-step graph (the backbone-independent work on side streams), outputs bit-identical, self-tested against the
    # parent graph at build (any failure -> the parent graph). Sources (BVR LOWKE): 00 submissions/2026-10-22_simvp-…-kd5-kd4-kd6-
    # rec-ipc-ipw-cdio-s3-bundle-…, 10 …-cdio-bvr-s3-bundle-…, 11 …-kd5-kd4-kd6-rec-ipc-ipw-cdio-bvr-lowke, 01 …-cdio-tka-lowke.
    # PKG_NAME "-kd5-kd4-rec-" -> "-kd5-kd4-kd6-rec-". Code only: no new weights / fits.
    KD6="${KD6:-0}"
    { [ "$KD6" = "0" ] || [ "$KD6" = "1" ]; } || { echo "FATAL: KD6 must be 0 or 1 (got '$KD6')"; exit 2; }
    if [ "$KD6" = "1" ]; then
      [ "${CDIO:-0}" = "1" ] || { echo "FATAL: KD6=1 requires CDIO=1 (the KD6 sources exist on the FZ-E CDIO sources only)"; exit 2; }
      KD6_PFX="$REPO/submissions/2026-10-22_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-rec-ipc-ipw-cdio"
      case "${BVR:-0}${LOWKE:-0}" in
        00) SRC_PKG="$KD6_PFX-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source" ;;
        10) SRC_PKG="$KD6_PFX-bvr-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/source" ;;
        11) SRC_PKG="$KD6_PFX-bvr-lowke/source" ;;
        01) SRC_PKG="$KD6_PFX-tka-lowke/source" ;;
      esac
      { [ -f "$SRC_PKG/kd6_conc.py" ] && grep -q '"kd6_concurrency": True' "$SRC_PKG/submission.py" 2>/dev/null \
        && grep -q '"cuda_direct_io": True' "$SRC_PKG/submission.py" && grep -q '_kd6_build' "$SRC_PKG/cuda_graph_wholestep.py"; } \
        || { echo "FATAL: KD6=1 but $SRC_PKG has no kd6_conc.py / CONFIG kd6_concurrency True / cuda_direct_io True / _kd6_build"; exit 2; }
      if [ "${BVR:-0}" = "1" ]; then
        grep -q '"ipc_byvalue": True' "$SRC_PKG/submission.py" || { echo "FATAL: KD6=1 BVR=1 but $SRC_PKG has no CONFIG ipc_byvalue True"; exit 2; }
      else
        ! grep -q '"ipc_byvalue"' "$SRC_PKG/submission.py" || { echo "FATAL: KD6=1 BVR=0 but $SRC_PKG carries ipc_byvalue"; exit 2; }
      fi
      if [ "${LOWKE:-0}" = "1" ]; then
        { grep -q '"tka_lowke": {"theta": 1.5, "delta": 0.25, "gmin": 0.0}' "$SRC_PKG/submission.py" && grep -q '_lowke_apply' "$SRC_PKG/cuda_graph_wholestep.py"; } \
          || { echo "FATAL: KD6=1 LOWKE=1 but $SRC_PKG has no CONFIG tka_lowke / _lowke_apply"; exit 2; }
      else
        ! grep -q '"tka_lowke"' "$SRC_PKG/submission.py" || { echo "FATAL: KD6=1 LOWKE=0 but $SRC_PKG carries tka_lowke"; exit 2; }
      fi
      PKG_NAME="${PKG_NAME/-kd5-kd4-rec-/-kd5-kd4-kd6-rec-}"
    fi
    # KD7 (2026-09-26, experiments/kd7/notes.md): KD7=1 (default 0; requires KD6=1 BVR=1 LOWKE=1 -- the only KD7 twin)
    # packages submissions/2026-10-23_simvp-…-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke = the KD6 BVR+LOWKE source + CONFIG
    # "kd7": True + kd7_fuse.py (bit-exact jiterator / NVRTC fusion of the KD6 body's post-join elementwise chains,
    # compiled + self-tested at build; any failure -> the KD6 body) + kd6_conc.py / cuda_graph_wholestep.py hooks. Outputs
    # bit-identical to KD6's. PKG_NAME "-kd5-kd4-kd6-rec-" -> "-kd5-kd4-kd6-kd7-rec-". Code only: no new weights / fits.
    KD7="${KD7:-0}"
    { [ "$KD7" = "0" ] || [ "$KD7" = "1" ]; } || { echo "FATAL: KD7 must be 0 or 1 (got '$KD7')"; exit 2; }
    if [ "$KD7" = "1" ]; then
      [ "${KD6:-0}" = "1" ] || { echo "FATAL: KD7=1 requires KD6=1"; exit 2; }
      [ "${BVR:-0}${LOWKE:-0}" = "11" ] || { echo "FATAL: KD7=1 exists only as the BVR=1 LOWKE=1 KD6 twin (got BVR=${BVR:-0} LOWKE=${LOWKE:-0})"; exit 2; }
      SRC_PKG="$REPO/submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke/source"
      { [ -f "$SRC_PKG/kd7_fuse.py" ] && [ -f "$SRC_PKG/kd6_conc.py" ] && grep -q '"kd7": True' "$SRC_PKG/submission.py" 2>/dev/null \
        && grep -q '"kd6_concurrency": True' "$SRC_PKG/submission.py" && grep -q '"ipc_byvalue": True' "$SRC_PKG/submission.py" \
        && grep -q '"tka_lowke": {"theta": 1.5, "delta": 0.25, "gmin": 0.0}' "$SRC_PKG/submission.py" \
        && grep -q 'kd7_fuse' "$SRC_PKG/cuda_graph_wholestep.py" && grep -q '_kd7j' "$SRC_PKG/kd6_conc.py"; } \
        || { echo "FATAL: KD7=1 but $SRC_PKG has no kd7_fuse.py / CONFIG kd7 True / kd6_concurrency / ipc_byvalue / tka_lowke / KD7 hooks"; exit 2; }
      PKG_NAME="${PKG_NAME/-kd5-kd4-kd6-rec-/-kd5-kd4-kd6-kd7-rec-}"
    fi
    # LSK (2026-09-26, experiments/lsk/notes.md; KD7 twin experiments/lsb/notes.md): LSK=1 (default 0, NOT in the FREEZE
    # defaults; requires KD6=1 BVR=1 LOWKE=1) packages a frozen separable least-squares correction of the packed backbone
    # output (top-8 SVD directions of the members' concatenated head weight + a constant, per-trajectory causal fp64
    # moments, CONFIG B: rho 0.9, lam_rel 30 -- FR33's condition; A = lam 10 was LSK's pick; the moment update runs on a
    # third KD6 side stream, 2 kernels on the critical path; self-tested at build, any failure -> off). Source = the parent source + CONFIG "lsk_head" + lsk_head.py (experiments/lsk/scripts/
    # lsk_make_source.py): KD7=1 -> submissions/2026-10-23_simvp-…-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk (fa948762 + LSK);
    # KD7=0 -> submissions/2026-10-23_simvp-…-kd6-rec-ipc-ipw-cdio-bvr-lowke-lsk (40fd4907 + LSK). Weights-free: the basis
    # is computed from model.pth at load. PKG_NAME "-lowke-" -> "-lowke-lsk-". Code only: no new weights / fits.
    LSK="${LSK:-0}"
    { [ "$LSK" = "0" ] || [ "$LSK" = "1" ]; } || { echo "FATAL: LSK must be 0 or 1 (got '$LSK')"; exit 2; }
    if [ "$LSK" = "1" ]; then
      [ "${KD6:-0}${BVR:-0}${LOWKE:-0}" = "111" ] || { echo "FATAL: LSK=1 exists only as the KD6=1 BVR=1 LOWKE=1 twin (got KD6=${KD6:-0} BVR=${BVR:-0} LOWKE=${LOWKE:-0})"; exit 2; }
      if [ "${KD7:-0}" = "1" ]; then
        SRC_PKG="$REPO/submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk/source"
        { [ -f "$SRC_PKG/kd7_fuse.py" ] && grep -q '"kd7": True' "$SRC_PKG/submission.py" 2>/dev/null \
          && grep -q 'kd7_fuse' "$SRC_PKG/cuda_graph_wholestep.py" && grep -q '_kd7j' "$SRC_PKG/kd6_conc.py"; } \
          || { echo "FATAL: LSK=1 KD7=1 but $SRC_PKG has no kd7_fuse.py / CONFIG kd7 True / KD7 hooks"; exit 2; }
      else
        SRC_PKG="$REPO/submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-rec-ipc-ipw-cdio-bvr-lowke-lsk/source"
        { [ ! -f "$SRC_PKG/kd7_fuse.py" ] && ! grep -q '"kd7"' "$SRC_PKG/submission.py" 2>/dev/null; } \
          || { echo "FATAL: LSK=1 KD7=0 but $SRC_PKG carries kd7"; exit 2; }
      fi
      { [ -f "$SRC_PKG/lsk_head.py" ] && [ -f "$SRC_PKG/kd6_conc.py" ] && grep -q '"lsk_head": {"r": 8, "rho": 0.9, "lam": 30.0}' "$SRC_PKG/submission.py" 2>/dev/null \
        && grep -q '"kd6_concurrency": True' "$SRC_PKG/submission.py" && grep -q '"ipc_byvalue": True' "$SRC_PKG/submission.py" \
        && grep -q '"tka_lowke": {"theta": 1.5, "delta": 0.25, "gmin": 0.0}' "$SRC_PKG/submission.py" \
        && grep -q 'side_c' "$SRC_PKG/kd6_conc.py" && grep -q '_lsk.update(y)' "$SRC_PKG/cuda_graph_wholestep.py"; } \
        || { echo "FATAL: LSK=1 but $SRC_PKG has no lsk_head.py / CONFIG lsk_head / kd6_concurrency / ipc_byvalue / tka_lowke / LSK hooks"; exit 2; }
      PKG_NAME="${PKG_NAME/-lowke-/-lowke-lsk-}"
    fi
    # LSKR (2026-09-26, experiments/lsr/notes.md; the REC-reset hardening of experiments/lsx arm 1): LSKR=1 (default 0,
    # NOT in the FREEZE defaults; requires LSK=1 KD7=1) packages the KD7+LSK-B source + CONFIG "lsk_rec_reset": True
    # (when REC fires the LS moments / apply copy of M are multiplied by rec.keep -- exactly x 1.0 when REC does not
    # fire) + the self-test's rho fix (lsk.rho instead of a hard-coded 0.9; inert at rho 0.9) (experiments/lsr/
    # lsr_make_source.py): submissions/2026-10-23_simvp-…-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr. PKG_NAME
    # "-lowke-lsk-" -> "-lowke-lsk-recr-". Code only: no new weights / fits.
    LSKR="${LSKR:-0}"
    { [ "$LSKR" = "0" ] || [ "$LSKR" = "1" ]; } || { echo "FATAL: LSKR must be 0 or 1 (got '$LSKR')"; exit 2; }
    if [ "$LSKR" = "1" ]; then
      [ "${LSK:-0}${KD7:-0}" = "11" ] || { echo "FATAL: LSKR=1 requires LSK=1 KD7=1 (got LSK=${LSK:-0} KD7=${KD7:-0})"; exit 2; }
      SRC_PKG="$REPO/submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr/source"
      { [ -f "$SRC_PKG/lsk_head.py" ] && [ -f "$SRC_PKG/kd7_fuse.py" ] && grep -q '"kd7": True' "$SRC_PKG/submission.py" 2>/dev/null \
        && grep -q '"lsk_head": {"r": 8, "rho": 0.9, "lam": 30.0}' "$SRC_PKG/submission.py" && grep -q '"lsk_rec_reset": True' "$SRC_PKG/submission.py" \
        && grep -q 'Phi\[c\] = lsk.rho \* Phi\[c\]' "$SRC_PKG/submission.py" && grep -q 'def rec_reset_' "$SRC_PKG/lsk_head.py" \
        && grep -q '_lsk.rec_reset_(rec.keep)' "$SRC_PKG/kd6_conc.py" && grep -q '_lsk.rec_reset_(rec.keep)' "$SRC_PKG/cuda_graph_wholestep.py" \
        && grep -q 'side_c' "$SRC_PKG/kd6_conc.py" && grep -q '_kd7j' "$SRC_PKG/kd6_conc.py"; } \
        || { echo "FATAL: LSKR=1 but $SRC_PKG has no lsk_head.py rec_reset_ / kd7_fuse.py / CONFIG lsk_head B + lsk_rec_reset True / rho fix / REC-reset hooks"; exit 2; }
      PKG_NAME="${PKG_NAME/-lowke-lsk-/-lowke-lsk-recr-}"
    fi
    # KD8 (2026-09-26, experiments/kd8/notes.md; MRG experiments/mrg/notes.md): KD8=1 (default 0, NOT in the FREEZE
    # defaults; requires LSK=1 KD7=1, i.e. the a9acf984 twin) packages the KD7+LSK-B source + CONFIG "kd8": True +
    # kd8_fuse.py (6 more bit-exact jiterator kernels) + kd6_conc.py / cuda_graph_wholestep.py hooks (more of the KD6
    # body off the main stream; built + self-tested first, any failure -> the KD7 body). Code only: no new weights / fits.
    #   LSKR=0: submissions/2026-10-23_simvp-…-lowke-lsk-kd8 (outputs bit-identical to a9acf984's);
    #           PKG_NAME "-lowke-lsk-" -> "-lowke-lsk-kd8-".
    #   LSKR=1: submissions/2026-10-23_simvp-…-lowke-lsk-recr-kd8 = the LSKR source + the KD8 hunks (3-way merge) + the
    #           REC reset of the LS state on side_c under KD8 "lskc" (outputs bit-identical to b519fe06's);
    #           PKG_NAME "-lowke-lsk-recr-" -> "-lowke-lsk-recr-kd8-".
    KD8="${KD8:-0}"
    { [ "$KD8" = "0" ] || [ "$KD8" = "1" ]; } || { echo "FATAL: KD8 must be 0 or 1 (got '$KD8')"; exit 2; }
    if [ "$KD8" = "1" ]; then
      [ "${LSK:-0}${KD7:-0}" = "11" ] || { echo "FATAL: KD8=1 exists only as the LSK=1 KD7=1 twin (got LSK=${LSK:-0} KD7=${KD7:-0})"; exit 2; }
      if [ "$LSKR" = "1" ]; then
        SRC_PKG="$REPO/submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8/source"
      else
        SRC_PKG="$REPO/submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-kd8/source"
      fi
      { [ -f "$SRC_PKG/kd8_fuse.py" ] && [ -f "$SRC_PKG/kd7_fuse.py" ] && [ -f "$SRC_PKG/lsk_head.py" ] \
        && grep -q '"kd8": True' "$SRC_PKG/submission.py" 2>/dev/null && grep -q '"kd7": True' "$SRC_PKG/submission.py" \
        && grep -q '"lsk_head": {"r": 8, "rho": 0.9, "lam": 30.0}' "$SRC_PKG/submission.py" \
        && grep -q '"kd6_concurrency": True' "$SRC_PKG/submission.py" && grep -q '"ipc_byvalue": True' "$SRC_PKG/submission.py" \
        && grep -q '"tka_lowke": {"theta": 1.5, "delta": 0.25, "gmin": 0.0}' "$SRC_PKG/submission.py" \
        && grep -q 'kd8_fuse' "$SRC_PKG/cuda_graph_wholestep.py" && grep -q '_kd8j' "$SRC_PKG/kd6_conc.py" \
        && grep -q '_lsk.update(y)' "$SRC_PKG/cuda_graph_wholestep.py"; } \
        || { echo "FATAL: KD8=1 but $SRC_PKG has no kd8_fuse.py / kd7_fuse.py / lsk_head.py / CONFIG kd8 True / kd7 / lsk_head B / KD8 hooks"; exit 2; }
      if [ "$LSKR" = "1" ]; then
        { grep -q '"lsk_rec_reset": True' "$SRC_PKG/submission.py" && grep -q 'Phi\[c\] = lsk.rho \* Phi\[c\]' "$SRC_PKG/submission.py" \
          && grep -q 'def rec_reset_(self, keep' "$SRC_PKG/lsk_head.py" && grep -q '_lsk.rec_reset_(rec.keep)' "$SRC_PKG/cuda_graph_wholestep.py" \
          && [ "$(grep -c '_lsk.rec_reset_(rec.keep)' "$SRC_PKG/kd6_conc.py")" = "2" ] && grep -q 'rr8 = _lsk.rec_reset' "$SRC_PKG/kd6_conc.py"; } \
          || { echo "FATAL: KD8=1 LSKR=1 but $SRC_PKG has no CONFIG lsk_rec_reset True / rho fix / rec_reset_ / both KD6 REC-reset hooks (main + KD8 side_c) / whole-step hook"; exit 2; }
        PKG_NAME="${PKG_NAME/-lowke-lsk-recr-/-lowke-lsk-recr-kd8-}"
      else
        { ! grep -q '"lsk_rec_reset"' "$SRC_PKG/submission.py" && ! grep -q 'rec_reset_' "$SRC_PKG/kd6_conc.py"; } \
          || { echo "FATAL: KD8=1 LSKR=0 but $SRC_PKG carries lsk_rec_reset"; exit 2; }
        PKG_NAME="${PKG_NAME/-lowke-lsk-/-lowke-lsk-kd8-}"
      fi
    fi
    # LSKR97 (2026-09-26, experiments/x5v/notes.md; LSX X5R experiments/lsx/notes.md): LSKR97=1 (default 0, NOT in the
    # FREEZE defaults; requires LSK=1 LSKR=1 KD8=1) packages the -lsk-recr-kd8 source (7b788569) with ONE constant
    # changed: CONFIG "lsk_head" forgetting factor rho 0.9 -> 0.97 (X5R = rho 0.97 + the REC reset; the self-test uses
    # lsk.rho since LSKR): submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97.
    # PKG_NAME "-lowke-lsk-recr-kd8-" -> "-lowke-lsk-recr-kd8-r97-". Code only: no new weights / fits.
    LSKR97="${LSKR97:-0}"
    { [ "$LSKR97" = "0" ] || [ "$LSKR97" = "1" ]; } || { echo "FATAL: LSKR97 must be 0 or 1 (got '$LSKR97')"; exit 2; }
    if [ "$LSKR97" = "1" ]; then
      [ "${LSK:-0}${LSKR:-0}${KD8:-0}" = "111" ] || { echo "FATAL: LSKR97=1 requires LSK=1 LSKR=1 KD8=1 (got LSK=${LSK:-0} LSKR=${LSKR:-0} KD8=${KD8:-0})"; exit 2; }
      SRC_PKG="$REPO/submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97/source"
      { [ -f "$SRC_PKG/kd8_fuse.py" ] && [ -f "$SRC_PKG/kd7_fuse.py" ] && [ -f "$SRC_PKG/lsk_head.py" ] \
        && grep -q '"kd8": True' "$SRC_PKG/submission.py" 2>/dev/null && grep -q '"kd7": True' "$SRC_PKG/submission.py" \
        && grep -qF '"lsk_head": {"r": 8, "rho": 0.97, "lam": 30.0}' "$SRC_PKG/submission.py" && ! grep -q '"lsk_head": {"r": 8, "rho": 0.9,' "$SRC_PKG/submission.py" \
        && grep -q '"kd6_concurrency": True' "$SRC_PKG/submission.py" && grep -q '"ipc_byvalue": True' "$SRC_PKG/submission.py" \
        && grep -q '"tka_lowke": {"theta": 1.5, "delta": 0.25, "gmin": 0.0}' "$SRC_PKG/submission.py" \
        && grep -q 'kd8_fuse' "$SRC_PKG/cuda_graph_wholestep.py" && grep -q '_kd8j' "$SRC_PKG/kd6_conc.py" \
        && grep -q '_lsk.update(y)' "$SRC_PKG/cuda_graph_wholestep.py" \
        && grep -q '"lsk_rec_reset": True' "$SRC_PKG/submission.py" && grep -q 'Phi\[c\] = lsk.rho \* Phi\[c\]' "$SRC_PKG/submission.py" \
        && grep -q 'B\[c\] = lsk.rho \* B\[c\]' "$SRC_PKG/submission.py" \
        && grep -q 'def rec_reset_(self, keep' "$SRC_PKG/lsk_head.py" && grep -q '_lsk.rec_reset_(rec.keep)' "$SRC_PKG/cuda_graph_wholestep.py" \
        && [ "$(grep -c '_lsk.rec_reset_(rec.keep)' "$SRC_PKG/kd6_conc.py")" = "2" ] && grep -q 'rr8 = _lsk.rec_reset' "$SRC_PKG/kd6_conc.py"; } \
        || { echo "FATAL: LSKR97=1 but $SRC_PKG has no CONFIG lsk_head rho 0.97 (or still rho 0.9) / kd8 / kd7 / lsk_rec_reset True / rho fix / rec_reset_ / both KD6 REC-reset hooks / whole-step hook"; exit 2; }
      PKG_NAME="${PKG_NAME/-lowke-lsk-recr-kd8-/-lowke-lsk-recr-kd8-r97-}"
    fi
    # UV (2026-09-26, experiments/pdr/notes.md): UV=1 (default 0, NOT in the FREEZE defaults; requires KD8=1 LSKR=1) packages
    # the -lsk-recr-kd8 source (LSKR97=0: 7b788569's) or the -lsk-recr-kd8-r97 source (LSKR97=1: f11d2c1d's) + CONFIG
    # "bvr_uv_only": True: the emitted p channel (unmeasured, unscored on real data) is exactly +0.0 in pred / lower / upper
    # (u, v bit-identical: one bitwise AND kernel into a staging buffer before the existing D2H; CUDA-direct: AND in place of
    # the clones), and BVR's by-value form ships the u, v words only (the parent rebuilds a C-contiguous float32
    # torch.Tensor with numpy / builtins callables). Code only: no new weights / fits.
    #   LSKR97=0: submissions/2026-10-23_simvp-…-lowke-lsk-recr-kd8-uv;     PKG_NAME "-lowke-lsk-recr-kd8-" -> "-lowke-lsk-recr-kd8-uv-".
    #   LSKR97=1: submissions/2026-10-23_simvp-…-lowke-lsk-recr-kd8-r97-uv; PKG_NAME "-lowke-lsk-recr-kd8-r97-" -> "-lowke-lsk-recr-kd8-r97-uv-".
    UV="${UV:-0}"
    { [ "$UV" = "0" ] || [ "$UV" = "1" ]; } || { echo "FATAL: UV must be 0 or 1 (got '$UV')"; exit 2; }
    if [ "$UV" = "1" ]; then
      [ "${LSK:-0}${LSKR:-0}${KD8:-0}" = "111" ] || { echo "FATAL: UV=1 requires LSK=1 LSKR=1 KD8=1 (got LSK=${LSK:-0} LSKR=${LSKR:-0} KD8=${KD8:-0})"; exit 2; }
      if [ "$LSKR97" = "1" ]; then
        SRC_PKG="$REPO/submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97-uv/source"
        UV_RHO='"lsk_head": {"r": 8, "rho": 0.97, "lam": 30.0}'; UV_NOT='"lsk_head": {"r": 8, "rho": 0.9,'
      else
        SRC_PKG="$REPO/submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-uv/source"
        UV_RHO='"lsk_head": {"r": 8, "rho": 0.9, "lam": 30.0}'; UV_NOT='"lsk_head": {"r": 8, "rho": 0.97,'
      fi
      { [ -f "$SRC_PKG/kd8_fuse.py" ] && [ -f "$SRC_PKG/kd7_fuse.py" ] && [ -f "$SRC_PKG/lsk_head.py" ] \
        && grep -q '"kd8": True' "$SRC_PKG/submission.py" 2>/dev/null && grep -q '"kd7": True' "$SRC_PKG/submission.py" \
        && grep -qF "$UV_RHO" "$SRC_PKG/submission.py" && ! grep -qF "$UV_NOT" "$SRC_PKG/submission.py" \
        && grep -q '"kd6_concurrency": True' "$SRC_PKG/submission.py" && grep -q '"ipc_byvalue": True' "$SRC_PKG/submission.py" \
        && grep -q '"tka_lowke": {"theta": 1.5, "delta": 0.25, "gmin": 0.0}' "$SRC_PKG/submission.py" \
        && grep -q '"lsk_rec_reset": True' "$SRC_PKG/submission.py" && grep -q 'rr8 = _lsk.rec_reset' "$SRC_PKG/kd6_conc.py" \
        && [ "$(grep -c '_lsk.rec_reset_(rec.keep)' "$SRC_PKG/kd6_conc.py")" = "2" ] && grep -q '_kd8j' "$SRC_PKG/kd6_conc.py" \
        && grep -q '"bvr_uv_only": True' "$SRC_PKG/submission.py" && grep -q '^class _BVUVArray' "$SRC_PKG/submission.py" \
        && grep -q 'def _p0_zero' "$SRC_PKG/submission.py" && grep -q 'def _p0_dev' "$SRC_PKG/cuda_graph_wholestep.py" \
        && grep -q '_dev_stage_i32' "$SRC_PKG/cuda_graph_wholestep.py" && grep -q 'def _d2h' "$SRC_PKG/w01_warmup_graph.py"; } \
        || { echo "FATAL: UV=1 but $SRC_PKG has no KD8 / LSKR / BVR config + hooks, the wrong lsk_head rho for LSKR97=$LSKR97, or no bvr_uv_only True / _BVUVArray / p-zero hooks"; exit 2; }
      if [ "$LSKR97" = "1" ]; then
        PKG_NAME="${PKG_NAME/-lowke-lsk-recr-kd8-r97-/-lowke-lsk-recr-kd8-r97-uv-}"
      else
        PKG_NAME="${PKG_NAME/-lowke-lsk-recr-kd8-/-lowke-lsk-recr-kd8-uv-}"
      fi
    fi
    KE3="${KE3:-0}"
    { [ "$KE3" = "0" ] || [ "$KE3" = "1" ]; } || { echo "FATAL: KE3 must be 0 or 1 (got '$KE3')"; exit 2; }
    if [ "$KE3" = "1" ]; then
      X2_DIR="${X2_DIR}_kes4"; SHIP_DIR="$X2_DIR"
      PKG_NAME="${PKG_NAME/-s3-bundle-/-kes4-s3-bundle-}"
    fi
    ARCH_EXPECT="b5x3_h192_blk6_rf11"
  fi
fi
[ "${KE3:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: KE3=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${IPW:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: IPW=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${CDIO:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: CDIO=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${BVR:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: BVR=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${LOWKE:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: LOWKE=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${KD6:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: KD6=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${KD7:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: KD7=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${LSK:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: LSK=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${LSKR:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: LSKR=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${KD8:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: KD8=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${LSKR97:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: LSKR97=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${UV:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: UV=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${FP16:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: FP16=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${KD2:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: KD2=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${KD3:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: KD3=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${IPC:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: IPC=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${KD5:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: KD5=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${KD4:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: KD4=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
[ "${REC:-0}" = "1" ] && [ "$X3" != "1" ] && { echo "FATAL: REC=1 is packaged only with BACKBONE=b5spx3"; exit 2; }
KFOLD="${KFOLD-all81}"                     # shipping recipe; KFOLD= "" turns 05b off
CONDWIDTH_RECIPE="${CONDWIDTH_RECIPE:-w2}"  # w2 (shipped) | m1 (previous lineage)
W2_OUT="$REPO/data/checkpoints/ship_condwidth_v2"
[ "$B5FAM" = "1" ] && W2_OUT="$REPO/data/checkpoints/b1_3090/ship_condwidth_v2"
OUT_DIR="$REPO/submissions/$PKG_NAME"
# 5 levels below the repo root on purpose: submission.py computes _REPO as three
# levels up and prepends <_REPO>/src to sys.path if it exists, so a package at
# submissions/<name>/pkg/ would silently import the repo's own src/ instead of
# its vendored copies (HANDOVER.md S6, "operational lessons").
BUILD_PKG="$OUT_DIR/build/w/x/pkg"
LOGS="$REPO/logs"
DUMPS="$REPO/data/cache/p3_dumps"          # path hard-coded in scripts/ship_condwidth_fit.py
KEEP_DUMPS="${KEEP_DUMPS:-1}"
# R2 (2026-09-26, experiments/rch/notes.md): SERIAL=1 = one GPU process at a time in steps 03 / 04 / 05b (for GPUs in
# EXCLUSIVE_PROCESS compute mode). Same commands / args / seeds as the default concurrent schedule (SERIAL=0).
SERIAL="${SERIAL:-0}"
{ [ "$SERIAL" = "0" ] || [ "$SERIAL" = "1" ]; } || { echo "FATAL: SERIAL must be 0 or 1 (got '$SERIAL')"; exit 2; }
# R4 (2026-09-26, experiments/rch/notes.md): VERIFY_REUSE=1 (default) re-reads every reused artefact before a step is
# skipped (see artefact_ok below); 0 = the previous existence-only check.
VERIFY_REUSE="${VERIFY_REUSE:-1}"
{ [ "$VERIFY_REUSE" = "0" ] || [ "$VERIFY_REUSE" = "1" ]; } || { echo "FATAL: VERIFY_REUSE must be 0 or 1 (got '$VERIFY_REUSE')"; exit 2; }

FOLD_CKPT_fold0="data/checkpoints/j1/n1simvp_s3_fold0.pth"
FOLD_CKPT_fold1="data/checkpoints/d2/n1simvp_s3_fold1.pth"
FOLD_CKPT_foldR="data/checkpoints/d2/n1simvp_s3_foldR.pth"
# Recorded references (previous machine, docs/HANDOVER.md S3.1 + experiments/d2_bundle/notes.md)
REF_fold0=81.868
REF_fold1=82.353
REF_foldR=82.365

ALL_STEPS="01 02 03 04 05 05a 05b 06 07 08 09 10 11"   # 05a only for BACKBONE=b5sp; 05b only when KFOLD is non-empty
FORCE=0
ONLY=""
FROM=""

while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1 ;;
    --only) ONLY="$2"; shift ;;
    --from) FROM="$2"; shift ;;
    --list) sed -n '/^# STEPS/,/^# DETERMINISM/p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -h|--help) sed -n '2,/^# =====/p' "${BASH_SOURCE[0]}"; exit 0 ;;   # the whole header block
    *) echo "unknown argument: $1 (try --help)"; exit 2 ;;
  esac
  shift
done

# R1D FIX (bug found 2026-09-22 fresh-clone rehearsal): DRY_RUN was documented in this script's
# own header ("DRY_RUN=1 echoes commands without running them") but never implemented after the
# 09-22 refresh to the current lineage -- `DRY_RUN=1 bash scripts/retrain_all.sh` silently ran
# every step for real (confirmed: 3 concurrent fold-backbone trainings launched under it before
# anyone noticed). This minimal guard restores the safety property (nothing runs) even though it
# does not reproduce the old per-command echo UX; see experiments/r1d_rehearsal/notes.md "BUG 1".
DRY_RUN="${DRY_RUN:-0}"
if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1: not running anything. Resolved parameters:"
  echo "  BACKBONE=$BACKBONE  SHIP_DIR=$SHIP_DIR  SHIP_CKPT=$SHIP_CKPT  W2_OUT=$W2_OUT"
  echo "  PKG_NAME=$PKG_NAME  KFOLD=${KFOLD:-<empty, 05b disabled>}  CONDWIDTH_RECIPE=$CONDWIDTH_RECIPE"
  echo "  SRC_PKG=$SRC_PKG  SOUP_K=$SOUP_K  CB=${CB:-}"
  [ "$BACKBONE" = "b5spx2" ] && {
    echo "  b5spx2: SOUP_K=$SOUP_K ENRX_KE=$ENRX_KE CB=$CB B2=${B2:-} CB_ADAPT=${CB_ADAPT:-} AX=${AX:-} AXE=${AXE:-} GM20=${GM20:-} MSK=${MSK:-} X2_DIR=$X2_DIR SP_INIT=$SP_INIT SP_INIT1=$SP_INIT1"
    echo "    member 0 = $X2_M0 <-$X2_FT0"
    echo "    member 1 = $X2_M1 <-$X2_FT1"
    if [ "$X3" = "1" ]; then
      echo "    X3=1 (BACKBONE=b5spx3) FP16=$FP16 AXE=${AXE:-0} KD2=${KD2:-0} KD3=${KD3:-0} IPC=${IPC:-0} KD5=${KD5:-0} KD4=${KD4:-0} REC=${REC:-0} IPW=${IPW:-0} KE3=${KE3:-0} CDIO=${CDIO:-0} BVR=${BVR:-0} LOWKE=${LOWKE:-0} KD6=${KD6:-0} KD7=${KD7:-0} LSK=${LSK:-0} LSKR=${LSKR:-0} KD8=${KD8:-0} LSKR97=${LSKR97:-0} UV=${UV:-0} SP_INIT2=$SP_INIT2 ARCH_EXPECT=$ARCH_EXPECT"
      echo "    member 2 = $X2_M2 <-$X2_FT2"
    fi; }
  echo "  OUT_DIR=$OUT_DIR"
  echo "  ONLY=${ONLY:-<all>}  FROM=${FROM:-<start>}  FORCE=$FORCE"
  sed -n '/^# STEPS/,/^# DETERMINISM/p' "${BASH_SOURCE[0]}"
  exit 0
fi

want () {   # want <step-id>
  local s="$1"
  if [ -n "$ONLY" ]; then case ",$ONLY," in *",$s,"*) return 0 ;; *) return 1 ;; esac; fi
  if [ -n "$FROM" ]; then [ "$s" \> "$FROM" ] || [ "$s" = "$FROM" ]; return $?; fi
  return 0
}

have () {   # have <file> ... -> 0 if all exist (and every reused artefact reads back, R4) and --force was not given
  [ "$FORCE" = "1" ] && return 1
  local f
  for f in "$@"; do [ -e "$f" ] || return 1; done
  artefact_ok "$@" || return 1
  return 0
}

# R4 (2026-09-26, experiments/rch/notes.md): torch.save / np.savez / zip writes are not atomic, so a job killed mid-save
# leaves a truncated file that an existence check would silently reuse. artefact_ok <path> ... -> 0 if every path exists
# and every .pt / .pth / .npz / .zip among them reads back (zip central directory + CRC of every member via zipfile.testzip;
# a non-zip .pt / .pth is torch.load-ed on the CPU; .json is parsed); other paths (dirs, logs) need only exist. An
# unreadable file is renamed to <path>.corrupt (so no later -f / -e test can accept it) and 1 is returned -> the step is
# redone. Read-only on good files (never rewrites them); CPU only. VERIFY_REUSE=0 -> existence only.
artefact_ok () {
  local f
  for f in "$@"; do [ -e "$f" ] || return 1; done
  [ "$VERIFY_REUSE" = "1" ] || return 0
  CUDA_VISIBLE_DEVICES= "$PY" - "$@" <<'EOF'
import json, os, sys, zipfile
bad = 0
for f in sys.argv[1:]:
    ext = os.path.splitext(f)[1].lower()
    if not os.path.isfile(f) or ext not in (".pt", ".pth", ".npz", ".zip", ".json"):
        continue
    try:
        if ext == ".json":
            with open(f) as fh:
                json.load(fh)
        elif zipfile.is_zipfile(f):
            with zipfile.ZipFile(f) as z:
                m = z.testzip()
            if m is not None:
                raise ValueError(f"CRC mismatch in member {m}")
        elif ext in (".pt", ".pth"):
            import torch
            torch.load(f, map_location="cpu", weights_only=False)
        else:
            raise ValueError("not a readable zip archive (truncated?)")
    except Exception as e:  # noqa: BLE001
        os.replace(f, f + ".corrupt")
        print(f"  REUSE CHECK FAILED: {f} ({type(e).__name__}: {str(e)[:120]}) -> moved to {f}.corrupt; redoing", flush=True)
        bad += 1
sys.exit(1 if bad else 0)
EOF
}

STEP_T0=0
begin () { STEP_T0=$(date +%s); echo; echo "=== [$1] $2 ==="; }
skip ()  { echo "=== [$1] $2 -- SKIP (outputs present; --force to redo) ==="; }
done_ () {
  local dt=$(( $(date +%s) - STEP_T0 ))
  printf '=== [%s] done in %dm%02ds ===\n' "$1" $((dt/60)) $((dt%60))
  echo "$1 $dt" >> "$LOGS/retrain_timings.txt"
}

mkdir -p "$LOGS" "$DUMPS" "$REPO/data/checkpoints"

materialise_all81 () {   # data/real_all81/: symlinks to all 81 usable train_real files (7575_0.h5 excluded)
  mkdir -p "$REPO/data/real_all81"
  find "$REPO/data/real_all81" -maxdepth 1 -name '*.h5' -type l -delete
  for f in "$REPO"/data/competition/train_real/*.h5; do
    b=$(basename "$f"); [ "$b" = "7575_0.h5" ] && continue
    ln -sf "$(readlink -f "$f")" "$REPO/data/real_all81/$b"
  done
  local n; n=$(ls "$REPO"/data/real_all81/*.h5 | wc -l)
  [ "$n" = "81" ] || { echo "FATAL: data/real_all81 has $n files, expected 81"; exit 1; }
  echo "  materialised data/real_all81: $n files (7575_0.h5 excluded)"
}

sp_soup_member () {   # sp_soup_member <init> <out> <log-tag> <ft_0> ... <ft_K-1>   (b5spx2 step 06)
  # ft_s = all-81 fine-tune from <init> with fine-tune seed s (the b5sp step-06 command); K > 1: uniform soup -> <out>
  # (K = 1: <out> IS ft_0). Every ft_s shares <init>: members from different inits are never souped.
  local init="$1" out="$2" tag="$3"; shift 3
  local s=0 m members=""
  for m in "$@"; do
    if have "$m"; then echo "  have fine-tune $m"
    else
      "$PY" scripts/sp_finetune.py --init "$init" -- --model "$B5_MODEL" --loss s3 --train data/real_all81 \
          --seed "$s" --mem-frac 0.15 --out "$m" \
          2>&1 | tee "$LOGS/retrain_06_ship_train_${tag}_seed$s.log" | tail -n 4
    fi
    [ -f "$m" ] || { echo "FATAL: fine-tune $m missing"; exit 1; }
    members="$members $m"; s=$((s + 1))
  done
  if [ "$#" -gt 1 ]; then
    "$PY" scripts/c2_make_soup.py --out "$out" $members 2>&1 | tee "$LOGS/retrain_06_ship_soup_${tag}.log"
  fi
  [ -f "$out" ] || { echo "FATAL: $out not written"; exit 1; }
}

echo "== 00. sanity =="
[ -x "$PY" ] || { echo "FATAL: $PY missing -- build .venv first (see the header)."; exit 1; }
[ -d "$REPO/data/competition/train_real" ] || {
  echo "FATAL: data/competition/train_real missing -- see data/README.md."; exit 1; }
if [ "$BACKBONE" = "b5sp" ] || [ "$BACKBONE" = "b5spx2" ]; then
  n_sim=$(ls "$REPO"/data/competition/train_sim/*.h5 2>/dev/null | wc -l)
  [ "$n_sim" = "100" ] || { echo "FATAL: BACKBONE=$BACKBONE needs data/competition/train_sim/ (100 files, found $n_sim) -- see data/README.md."; exit 1; }
fi
[ -f "$SRC_PKG/submission.py" ] || {
  echo "FATAL: $SRC_PKG/submission.py missing (tracked package source)."; exit 1; }
[ -f "$REPO/scripts/pack_zip.py" ] || { echo "FATAL: scripts/pack_zip.py missing."; exit 1; }
"$PY" - <<'EOF'
import sys, torch, numpy
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | numpy {numpy.__version__} | "
      f"cuda {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
EOF
: > "$LOGS/retrain_timings.txt"

# -----------------------------------------------------------------------------
# 01. honest 73/4/4 trajectory split (seed 0, excludes 7575_0.h5)
# -----------------------------------------------------------------------------
if want 01; then
  if have data/real/train data/real/val data/real/test; then skip 01 "data split"
  else
    begin 01 "data split (scripts/split_data.py, seed 0, 0.90/0.05/0.05 -> 73/4/4)"
    "$PY" scripts/split_data.py 2>&1 | tee "$LOGS/retrain_01_split.log"
    for d in train val test; do
      printf '  data/real/%-5s %s files\n' "$d" "$(ls data/real/$d/*.h5 2>/dev/null | wc -l)"
    done
    done_ 01
  fi
fi

# -----------------------------------------------------------------------------
# 02. leave-conditions-out folds (symlinks into data/competition/train_real)
# -----------------------------------------------------------------------------
if want 02; then
  if have data/real_lco/fold0/heldout data/real_lco/fold1/heldout data/real_lco/foldR/heldout
  then skip 02 "LCO splits"
  else
    begin 02 "LCO splits (scripts/lco_make_splits.py + the documented aliases)"
    {
      "$PY" scripts/lco_make_splits.py --fold fold0_aoa20_plus_re_extremes
      "$PY" scripts/lco_make_splits.py --fold fold1_aoa15_plus_re_extremes
      "$PY" scripts/lco_make_splits.py --fold fold2_aoa5_plus_re_extremes
      "$PY" scripts/lco_make_splits.py --split-json experiments/lco_split/split_rehold.json \
            --fold foldR_leaderboard_mimic --out data/real_lco
    } 2>&1 | tee "$LOGS/retrain_02_lco_split.log"
    ( cd data/real_lco \
      && ln -sfn fold0_aoa20_plus_re_extremes fold0 \
      && ln -sfn fold1_aoa15_plus_re_extremes fold1 \
      && ln -sfn fold2_aoa5_plus_re_extremes  fold2 \
      && ln -sfn foldR_leaderboard_mimic      foldR )
    # verify the materialised folds against the split JSONs (file sets + no leak)
    "$PY" - <<'EOF' 2>&1 | tee -a "$LOGS/retrain_02_lco_split.log"
import json, pathlib, sys
inv = {d['file']: d['n_windows'] for d in json.load(open('experiments/lco_split/inventory.json'))}
folds = json.load(open('experiments/lco_split/split.json'))['folds']
folds.update(json.load(open('experiments/lco_split/split_rehold.json'))['folds'])
alias = {'fold0': 'fold0_aoa20_plus_re_extremes', 'fold1': 'fold1_aoa15_plus_re_extremes',
         'fold2': 'fold2_aoa5_plus_re_extremes', 'foldR': 'foldR_leaderboard_mimic'}
expect_w = {'fold0': 811, 'fold1': 840, 'fold2': 737, 'foldR': 653}
bad = 0
for a, k in alias.items():
    f = folds[k]
    for sub in ('train', 'heldout'):
        on_disk = sorted(p.name for p in (pathlib.Path('data/real_lco') / a / sub).glob('*.h5'))
        if on_disk != sorted(f[f'{sub}_files']):
            print(f'  FAIL {a}/{sub}: file set != split.json'); bad += 1
    w = sum(inv[n] for n in f['heldout_files'])
    ok_w = (w == expect_w[a])
    leak = set(f['train_files']) & set(f['heldout_files'])
    corrupt = '7575_0.h5' in set(f['train_files']) | set(f['heldout_files'])
    print(f'  {a}: train {len(f["train_files"])} / heldout {len(f["heldout_files"])} '
          f'({w} windows, expected {expect_w[a]}: {"ok" if ok_w else "MISMATCH"}), '
          f'train^heldout={len(leak)}, 7575_0.h5={corrupt}')
    bad += (not ok_w) + bool(leak) + bool(corrupt)
sys.exit(1 if bad else 0)
EOF
    done_ 02
  fi
fi

# -----------------------------------------------------------------------------
# 03. the three LCO fold backbones (SimVP n1_simvp, S3 loss, seed 0, fold TRAIN only)
# -----------------------------------------------------------------------------
if want 03; then
  if have "$FOLD_CKPT_fold0" "$FOLD_CKPT_fold1" "$FOLD_CKPT_foldR"; then skip 03 "fold backbones"
  else
    begin 03 "fold backbones (scripts/j1_finetune.py, $([ "$SERIAL" = "1" ] && echo "SERIAL=1: one at a time" || echo "3 concurrent"), --mem-frac 0.15)"
    mkdir -p data/checkpoints/j1 data/checkpoints/d2
    pids=""
    for fold in fold0 fold1 foldR; do
      eval "ck=\$FOLD_CKPT_$fold"
      "$PY" scripts/j1_finetune.py --model n1_simvp --loss s3 \
          --train "data/real_lco/$fold/train" --seed 0 --mem-frac 0.15 \
          --out "$ck" > "$LOGS/retrain_03_train_$fold.log" 2>&1 &
      if [ "$SERIAL" = "1" ]; then wait "$!"; else pids="$pids $!"; fi   # R2: SERIAL=1 -> one at a time
    done
    for p in $pids; do wait "$p"; done
    for fold in fold0 fold1 foldR; do
      printf '  %s: %s\n' "$fold" "$(grep -E '^it +8000' "$LOGS/retrain_03_train_$fold.log" | tail -n 1)"
    done
    done_ 03
  fi
fi

# -----------------------------------------------------------------------------
# 04. per-fold prior.npz + ke_net_flow.pt, fitted on that fold's TRAIN files only
# -----------------------------------------------------------------------------
if want 04; then
  if have experiments/d2_bundle/fold0_c/prior.npz experiments/d2_bundle/fold0_c/ke_net_flow.pt \
          experiments/d2_bundle/fold1_c/prior.npz experiments/d2_bundle/fold1_c/ke_net_flow.pt \
          experiments/d2_bundle/foldR_c/prior.npz experiments/d2_bundle/foldR_c/ke_net_flow.pt
  then skip 04 "fold prior / energy-net fits"
  else
    begin 04 "fold prior + ke_net_flow fits (c11a_fit_prior / c11a_fit_ke_flow, $([ "$SERIAL" = "1" ] && echo "SERIAL=1: one fold at a time" || echo "3 concurrent"))"
    pids=""
    for fold in fold0 fold1 foldR; do
      eval "ck=\$FOLD_CKPT_$fold"
      (
        set -e
        d="experiments/d2_bundle/${fold}_c"
        "$PY" scripts/c11a_fit_prior.py --model n1_simvp \
            --files "data/real_lco/$fold/train" --ckpt "$ck" \
            --out "$d/prior.npz" --device cuda --mem-frac 0.15 \
            > "$LOGS/retrain_04_prior_$fold.log" 2>&1
        rm -rf "data/cache/retrain_tke_flow_$fold"    # R4: never resume from a killed run's partial feature cache
        "$PY" scripts/c11a_fit_ke_flow.py --tag "retrain_n1simvp_s3_$fold" --model n1_simvp \
            --ckpt "$ck" --files "data/real_lco/$fold/train" \
            --cache "data/cache/retrain_tke_flow_$fold" --out "$d/ke_net_flow.pt" \
            --mem-frac 0.2 > "$LOGS/retrain_04_keflow_$fold.log" 2>&1
        rm -rf "data/cache/retrain_tke_flow_$fold"    # multi-GB feature cache, not needed after the fit
      ) &
      if [ "$SERIAL" = "1" ]; then wait "$!"; else pids="$pids $!"; fi   # R2: SERIAL=1 -> one fold at a time
    done
    for p in $pids; do wait "$p"; done
    for fold in fold0 fold1 foldR; do
      printf '  %s ke_net_flow: %s\n' "$fold" \
        "$(grep -E 'BEST val e' "$LOGS/retrain_04_keflow_$fold.log" | tail -n 1)"
    done
    done_ 04
  fi
fi

# -----------------------------------------------------------------------------
# 05. stream each fold's HELD-OUT files through its bundle pipeline and score
#     (these three streams ARE the out-of-sample residual pool for step 09)
# -----------------------------------------------------------------------------
if want 05; then
  if have "$DUMPS/fold0_c.npz" "$DUMPS/fold1_c.npz" "$DUMPS/foldR_c.npz"
  then skip 05 "fold held-out streams"
  else
    begin 05 "fold held-out streams (scripts/lco_run_dump.py) + scoring (scripts/lco_analyze.py)"
    for fold in fold0 fold1 foldR; do
      "$PY" scripts/lco_run_dump.py --exp-dir "experiments/d2_bundle/${fold}_c" \
          --data-dir "data/real_lco/$fold/heldout" --device cuda --mem-frac 0.3 \
          --out "$DUMPS/${fold}_c.npz" > "$LOGS/retrain_05_stream_$fold.log" 2>&1
      "$PY" scripts/lco_analyze.py "$DUMPS/${fold}_c.npz" --time-score 88 \
          --tag "bundle@${fold}-heldout" \
          --json "$LOGS/retrain_05_score_$fold.json" > "$LOGS/retrain_05_score_$fold.log" 2>&1
      eval "ref=\$REF_$fold"
      printf '  %s (recorded reference %s): %s\n' "$fold" "$ref" \
        "$(grep -m1 composite= "$LOGS/retrain_05_score_$fold.log")"
    done
    done_ 05
  fi
fi

# -----------------------------------------------------------------------------
# 05a. (BACKBONE=b5sp only) SIM PRETRAIN of the b5 backbone: all 100 train_sim files (never
#      train_real), p masked to 0, official real stats, causal ctx-norm, S3 loss, lr 5e-4 cosine over
#      a 20k-it horizon STOPPED at 10k (the validated snapshot; experiments/sp_sim_pretrain/notes.md).
# -----------------------------------------------------------------------------
if want 05a && { [ "$BACKBONE" = "b5sp" ] || [ "$BACKBONE" = "b5spx2" ]; }; then
  if have "$SP_INIT"; then skip 05a "sim pretrain snapshot (b5sp)"
  else
    begin 05a "sim pretrain (scripts/sp_pretrain_sim.py, 100 train_sim files, 20k cosine stopped at 10k, seed 0)"
    mkdir -p data/checkpoints/sp
    "$PY" scripts/sp_pretrain_sim.py --iters 20000 --snapshots 10000 --stop-after 10000 --seed 0 \
        --mem-frac 0.15 --out-prefix data/checkpoints/sp/b5_sim_s3 \
        2>&1 | tee "$LOGS/retrain_05a_simpretrain.log" | tail -n 3
    [ -f "$SP_INIT" ] || { echo "FATAL: $SP_INIT not written (see $LOGS/retrain_05a_simpretrain.log)"; exit 1; }
    echo "  $(sha256sum "$SP_INIT")   (validated 2026-09-23: f1d6594c1ba119856ade3975e793cd8032e6bba6caa2277c4e24e46adccce2fc)"
    done_ 05a
  fi
fi
# 05a (BACKBONE=b5spx2): the SECOND sim pretrain -- identical args, --seed 1 (experiments/c2_soup/report.md, "ENRx recipe")
if want 05a && [ "$BACKBONE" = "b5spx2" ]; then
  if have "$SP_INIT1"; then skip 05a "sim pretrain snapshot, pretrain seed 1 (b5spx2)"
  else
    begin 05a "sim pretrain, seed 1 (scripts/sp_pretrain_sim.py, 100 train_sim files, 20k cosine stopped at 10k)"
    mkdir -p data/checkpoints/sp
    "$PY" scripts/sp_pretrain_sim.py --iters 20000 --snapshots 10000 --stop-after 10000 --seed 1 \
        --mem-frac 0.15 --out-prefix data/checkpoints/sp/b5_sim_s3_seed1 \
        2>&1 | tee "$LOGS/retrain_05a_simpretrain_seed1.log" | tail -n 3
    [ -f "$SP_INIT1" ] || { echo "FATAL: $SP_INIT1 not written (see $LOGS/retrain_05a_simpretrain_seed1.log)"; exit 1; }
    echo "  $(sha256sum "$SP_INIT1")   (4090 build 2026-09-23: f9b0438654cbd87bc32c43bbf9fdd093d9ea3d515b369418c5b64fc4a6bb42ee; other GPUs differ, as for seed 0)"
    done_ 05a
  fi
fi

# 05a (BACKBONE=b5spx3): the THIRD sim pretrain -- identical args, --seed 2 (experiments/k3_ensemble/notes.md)
if want 05a && [ "$BACKBONE" = "b5spx2" ] && [ "$X3" = "1" ]; then
  if have "$SP_INIT2"; then skip 05a "sim pretrain snapshot, pretrain seed 2 (b5spx3)"
  else
    begin 05a "sim pretrain, seed 2 (scripts/sp_pretrain_sim.py, 100 train_sim files, 20k cosine stopped at 10k)"
    mkdir -p data/checkpoints/sp
    "$PY" scripts/sp_pretrain_sim.py --iters 20000 --snapshots 10000 --stop-after 10000 --seed 2 \
        --mem-frac 0.15 --out-prefix data/checkpoints/sp/b5_sim_s3_seed2 \
        2>&1 | tee "$LOGS/retrain_05a_simpretrain_seed2.log" | tail -n 3
    [ -f "$SP_INIT2" ] || { echo "FATAL: $SP_INIT2 not written (see $LOGS/retrain_05a_simpretrain_seed2.log)"; exit 1; }
    echo "  $(sha256sum "$SP_INIT2")   (3090 build 2026-09-24, K3: d34783b8cf2a8b25d3890b43c01a04c67393d1402e25143dc34779b8e8f31529; other GPUs differ)"
    done_ 05a
  fi
fi

# -----------------------------------------------------------------------------
# 05b. OPTIONAL K-fold out-of-sample residual generation (OFF by default).
#
# Step 05 gives three out-of-sample streams (one per LCO fold). A richer
# conditional-width recipe wants K-fold cross-fitting instead -- over an LCO
# fold's TRAIN files, or (for the SHIPPING fit) over all 81 usable files, which
# yields roughly 3x the windows. This step is the generic generator for that
# data; it does NOT fit a width net, it produces the per-group out-of-sample
# feature caches a packager then points the fit at.
#
#   KFOLD=all81 bash scripts/retrain_all.sh --only 05b        # shipping recipe
#   KFOLD=fold0 KFOLD_K=4 bash scripts/retrain_all.sh --only 05b
#   KFOLD=all81 KFOLD_TRAIN_DIR=data/my_files bash scripts/retrain_all.sh --only 05b
#
# KFOLD      tag naming the K-fold family; also the sub-directory name. With
#            "all81" and no KFOLD_TRAIN_DIR, a data/real_all81/ directory of
#            symlinks to all 81 usable train_real files is materialised first
#            (7575_0.h5 excluded). Any other tag defaults to
#            data/real_lco/<tag>/train.
# KFOLD_K    number of groups (default 4).
#
# Per group g: backbone trained on the OTHER K-1 groups, prior + energy net fit
# on the same files, then group g streamed through the bundle pipeline and
# reduced to a compact feature cache. Group g's files are therefore out-of-sample
# for everything applied to them -- the property the width net needs.
#
# For the SHIPPING recipe (KFOLD=all81, K=4) this delegates to the W2-P
# packager's own generator scripts/w2p_make_shipping_data.sh when that file
# exists, so the chain reproduces exactly the caches the packager's width fit
# consumes (experiments/w2p_packaging/cache/feat_all81_g*.npz). KFOLD_GENERIC=1
# forces the generic path instead; KFOLD_JOBS sets its group concurrency.
#
# Otherwise the generic path runs. Layout and the per-group recipe are the W2
# agent's (scripts/w2_make_kfold.py, scripts/w2_build_group.sh), used unmodified:
#   experiments/w2_condwidth_v2/kfold/<KFOLD>/g<g>/{train,heldout,pipe}/
#   data/checkpoints/w2/n1simvp_s3_<KFOLD>_g<g>.pth
#   experiments/w2_condwidth_v2/cache/feat_<KFOLD>_g<g>.npz   <- the deliverable
# -----------------------------------------------------------------------------
if want 05b && [ -n "${KFOLD:-}" ]; then
  KFOLD_K="${KFOLD_K:-4}"
  KF_ROOT="$REPO/experiments/w2_condwidth_v2"
  KF_LAST=$((KFOLD_K - 1))
  KF_JOBS="${KFOLD_JOBS:-2}"; KF_MAXJ=3             # group concurrency (b5 pool / W2-P generator) + generic-path cap
  [ "$SERIAL" = "1" ] && { KF_JOBS=1; KF_MAXJ=1; }  # R2: SERIAL=1 -> one group process at a time (overrides KFOLD_JOBS)
  [ "$B5FAM" = "1" ] && KF_ROOT="$REPO/experiments/b1_3090"
  kf_feats=""
  for g in $(seq 0 $KF_LAST); do kf_feats="$kf_feats $KF_ROOT/cache/feat_${KFOLD}_g$g.npz"; done
  w2p_feats=""
  for g in $(seq 0 $KF_LAST); do w2p_feats="$w2p_feats $REPO/experiments/w2p_packaging/cache/feat_${KFOLD}_g$g.npz"; done
  [ "$B5FAM" = "1" ] && w2p_feats="$REPO/experiments/b1_3090/cache/__never__"   # n1 caches must not satisfy b5
  if have $kf_feats || have $w2p_feats; then skip 05b "K-fold out-of-sample data ($KFOLD, K=$KFOLD_K)"
  elif [ "$B5FAM" = "1" ]; then
    begin 05b "K-fold out-of-sample data, b5 backbones ($KFOLD, K=$KFOLD_K; scripts/b1s_make_kfold_pool.sh)"
    if [ "$KFOLD" = "all81" ]; then materialise_all81; KF_TRAIN_DIR="data/real_all81"
    else KF_TRAIN_DIR="${KFOLD_TRAIN_DIR:-data/real_lco/$KFOLD/train}"; fi
    VERIFY_REUSE="$VERIFY_REUSE" bash scripts/b1s_make_kfold_pool.sh "$KFOLD" "$KF_TRAIN_DIR" "$KFOLD_K" "$KF_JOBS" \
        2>&1 | tee "$LOGS/retrain_05b_b1s_pool_$KFOLD.log" | tail -n 12
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FATAL: scripts/b1s_make_kfold_pool.sh failed (see experiments/b1_3090/logs)"; exit 1; }
    echo "  feature caches: $KF_ROOT/cache/feat_${KFOLD}_g*.npz"
    done_ 05b
  else
    begin 05b "K-fold out-of-sample data ($KFOLD, K=$KFOLD_K)"
    # Prefer the W2-P packager's own shipping-data generator when it is present and
    # this is the shipping recipe it was written for (all 81 files, K=4, its own
    # paths under experiments/w2p_packaging/ + data/checkpoints/w2p/). It is
    # deterministic and idempotent in the same way, and it is what the packager
    # will point the width fit at -- so the chain should reproduce THAT, not a
    # parallel copy of it. Set KFOLD_GENERIC=1 to force the generic path below.
    W2P_SHIP="$REPO/scripts/w2p_make_shipping_data.sh"
    if [ "$KFOLD" = "all81" ] && [ "$KFOLD_K" = "4" ] && [ -z "${KFOLD_TRAIN_DIR:-}" ] \
       && [ -z "${KFOLD_GENERIC:-}" ] && [ -f "$W2P_SHIP" ]; then
      echo "  delegating to scripts/w2p_make_shipping_data.sh (W2-P packager's generator)"
      bash "$W2P_SHIP" "$KF_JOBS" 2>&1 | tee "$LOGS/retrain_05b_w2p_shipping_data.log" | tail -n 12
      [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FATAL: scripts/w2p_make_shipping_data.sh failed (see experiments/w2p_packaging/logs)"; exit 1; }
      echo "  feature caches: $REPO/experiments/w2p_packaging/cache/feat_all81_g*.npz"
      done_ 05b
      W2P_DONE=1
    fi
    if [ -z "${W2P_DONE:-}" ]; then
    KF_TRAIN_DIR="${KFOLD_TRAIN_DIR:-}"
    if [ -z "$KF_TRAIN_DIR" ] && [ "$KFOLD" = "all81" ]; then
      KF_TRAIN_DIR="data/real_all81"
      mkdir -p "$REPO/$KF_TRAIN_DIR"
      find "$REPO/$KF_TRAIN_DIR" -maxdepth 1 -name '*.h5' -type l -delete
      for f in "$REPO"/data/competition/train_real/*.h5; do
        b=$(basename "$f"); [ "$b" = "7575_0.h5" ] && continue
        ln -sf "$(readlink -f "$f")" "$REPO/$KF_TRAIN_DIR/$b"
      done
      echo "  materialised $KF_TRAIN_DIR: $(ls "$REPO/$KF_TRAIN_DIR"/*.h5 | wc -l) files (7575_0.h5 excluded)"
    fi
    "$PY" scripts/w2_make_kfold.py --fold "$KFOLD" --K "$KFOLD_K" \
        ${KF_TRAIN_DIR:+--train-dir "$KF_TRAIN_DIR"} \
        2>&1 | tee "$LOGS/retrain_05b_kfold_layout_$KFOLD.log"
    mkdir -p data/checkpoints/w2
    # K group backbones, at most 3 concurrent (each ~2 GB GPU)
    pend=""
    for g in $(seq 0 $KF_LAST); do
      ck="data/checkpoints/w2/n1simvp_s3_${KFOLD}_g$g.pth"
      if artefact_ok "$ck" && [ "$FORCE" != "1" ]; then echo "  g$g backbone present, skipping"; continue; fi
      "$PY" scripts/j1_finetune.py --model n1_simvp --loss s3 \
          --train "$KF_ROOT/kfold/$KFOLD/g$g/train" --seed 0 --mem-frac 0.15 \
          --out "$ck" > "$LOGS/retrain_05b_train_${KFOLD}_g$g.log" 2>&1 &
      pend="$pend $!"
      while [ "$(jobs -rp | wc -l)" -ge "$KF_MAXJ" ]; do wait -n; done
    done
    for pid in $pend; do wait "$pid" || true; done
    # Per-group fits + held-out stream + feature extraction.
    #
    # These are exactly the four commands `scripts/w2_build_group.sh` runs, on the
    # same paths, with the same unmodified tracked scripts -- inlined here for ONE
    # reason: that script pipes `lco_analyze.py` into `head -3` under `set -o
    # pipefail`, so as soon as the analyzer prints more rows than `head` reads it
    # dies with BrokenPipeError and the build aborts BEFORE `w2_extract.py`, i.e.
    # before the feature cache exists. Reproduced here on group all81/g0 (21 files,
    # 882 windows): the stream and the scoring both succeeded, composite 84.067,
    # and the group build still failed. Reported to the W2 agent; not patched here
    # (another agent's file). Everything below is byte-for-byte W2's recipe.
    for g in $(seq 0 $KF_LAST); do
      log="$LOGS/retrain_05b_group_${KFOLD}_g$g.log"
      gd="$KF_ROOT/kfold/$KFOLD/g$g"
      ck="data/checkpoints/w2/n1simvp_s3_${KFOLD}_g$g.pth"
      feat="$KF_ROOT/cache/feat_${KFOLD}_g$g.npz"
      if artefact_ok "$feat" && [ "$FORCE" != "1" ]; then echo "  g$g feature cache present, skipping"; continue; fi
      {
        set -e
        artefact_ok "$gd/pipe/prior.npz" || "$PY" scripts/c11a_fit_prior.py --model n1_simvp \
            --files "$gd/train" --ckpt "$ck" --out "$gd/pipe/prior.npz" --device cuda --mem-frac 0.15
        artefact_ok "$gd/pipe/ke_net_flow.pt" || { rm -rf "$KF_ROOT/cache/tke_flow_${KFOLD}_g$g"; "$PY" scripts/c11a_fit_ke_flow.py --model n1_simvp \
            --ckpt "$ck" --files "$gd/train" --cache "$KF_ROOT/cache/tke_flow_${KFOLD}_g$g" \
            --mem-frac 0.15 --out "$gd/pipe/ke_net_flow.pt"
          rm -rf "$KF_ROOT/cache/tke_flow_${KFOLD}_g$g"; }
        W2_AUX_OUT="$KF_ROOT/cache/aux_${KFOLD}_g$g.npz" "$PY" scripts/lco_run_dump.py \
            --exp-dir "$gd/pipe" --data-dir "$gd/heldout" --device cuda --mem-frac 0.15 \
            --out "$KF_ROOT/cache/dump_${KFOLD}_g$g.npz"
        "$PY" scripts/lco_analyze.py "$KF_ROOT/cache/dump_${KFOLD}_g$g.npz" \
            --tag "kfold $KFOLD g$g" --time-score 88 --json "$gd/heldout_scores.json"
        "$PY" scripts/w2_extract.py --dump "$KF_ROOT/cache/dump_${KFOLD}_g$g.npz" \
            --data-dir "$gd/heldout" --aux "$KF_ROOT/cache/aux_${KFOLD}_g$g.npz" --out "$feat"
        rm -f "$KF_ROOT/cache/dump_${KFOLD}_g$g.npz"   # 1.5 GB per group, not needed after extraction
      } > "$log" 2>&1
      printf '  g%s: %s\n' "$g" "$(grep -m1 composite= "$log")"
    done
    echo "  feature caches: $KF_ROOT/cache/feat_${KFOLD}_g*.npz"
    done_ 05b
    fi
  fi
fi

# -----------------------------------------------------------------------------
# 06. the SHIPPING backbone: S3 loss, random init, all 81 usable real files
# -----------------------------------------------------------------------------
if want 06 && [ "$BACKBONE" = "b5" ]; then
  if have "$SHIP_DIR/b5_s3_all81.pth"; then skip 06 "shipping backbone (b5)"
  else
    begin 06 "shipping backbone (scripts/b1_finetune.py --model $B5_MODEL, all 81 files, seed 0)"
    materialise_all81
    mkdir -p "$SHIP_DIR"
    "$PY" scripts/b1_finetune.py --model "$B5_MODEL" --loss s3 --train data/real_all81 \
        --seed 0 --mem-frac 0.15 --out "$SHIP_DIR/b5_s3_all81.pth" \
        2>&1 | tee "$LOGS/retrain_06_ship_train_b5.log" | tail -n 4
    done_ 06
  fi
fi
if want 06 && [ "$BACKBONE" = "b5sp" ]; then
  if have "$SHIP_CKPT"; then skip 06 "shipping backbone (b5sp)"
  else
    begin 06 "shipping backbone, sim-pretrained init (scripts/sp_finetune.py --init $SP_INIT -- b1_finetune.py --model $B5_MODEL, all 81 files, seed 0)"
    [ -f "$SP_INIT" ] || { echo "FATAL: $SP_INIT missing -- run step 05a (BACKBONE=b5sp)"; exit 1; }
    materialise_all81
    mkdir -p "$SHIP_DIR"
    if [ "$SOUP_K" -gt 1 ]; then
      members=""
      for s in $(seq 0 $((SOUP_K - 1))); do
        if [ "$s" = 0 ]; then m="$SHIP_MEMBER0"; else m="data/checkpoints/sp/b5sp_s3_all81_seed$s.pth"; fi
        if artefact_ok "$m"; then echo "  have soup member $m"
        else
          "$PY" scripts/sp_finetune.py --init "$SP_INIT" -- --model "$B5_MODEL" --loss s3 --train data/real_all81 \
              --seed "$s" --mem-frac 0.15 --out "$m" \
              2>&1 | tee "$LOGS/retrain_06_ship_train_b5sp_seed$s.log" | tail -n 4
        fi
        [ -f "$m" ] || { echo "FATAL: soup member $m missing"; exit 1; }
        members="$members $m"
      done
      "$PY" scripts/c2_make_soup.py --out "$SHIP_CKPT" $members 2>&1 | tee "$LOGS/retrain_06_ship_soup.log"
    else
    "$PY" scripts/sp_finetune.py --init "$SP_INIT" -- --model "$B5_MODEL" --loss s3 --train data/real_all81 \
        --seed 0 --mem-frac 0.15 --out "$SHIP_CKPT" \
        2>&1 | tee "$LOGS/retrain_06_ship_train_b5sp.log" | tail -n 4
    fi
    done_ 06
  fi
fi
if want 06 && [ "$BACKBONE" = "b5spx2" ]; then
  if have "$X2_M0" "$X2_M1"; then skip 06 "ENRx members (b5spx2, SOUP_K=$SOUP_K)"
  else
    begin 06 "ENRx members (b5spx2): soup of $SOUP_K all-81 fine-tunes per SP init (scripts/sp_finetune.py + scripts/c2_make_soup.py)"
    for f in "$SP_INIT" "$SP_INIT1"; do
      [ -f "$f" ] || { echo "FATAL: $f missing -- run step 05a (BACKBONE=b5spx2)"; exit 1; }
    done
    [ "$(sha256sum < "$SP_INIT")" != "$(sha256sum < "$SP_INIT1")" ] \
      || { echo "FATAL: $SP_INIT and $SP_INIT1 are identical -- ENRx needs two different pretrain seeds"; exit 1; }
    materialise_all81
    mkdir -p data/checkpoints/sp
    if have "$X2_M0"; then echo "  have member 0 $X2_M0"
    else sp_soup_member "$SP_INIT" "$X2_M0" b5spx2_pre0 $X2_FT0; fi
    if have "$X2_M1"; then echo "  have member 1 $X2_M1"
    else sp_soup_member "$SP_INIT1" "$X2_M1" b5spx2_pre1 $X2_FT1; fi
    done_ 06
  fi
fi
if want 06 && [ "$BACKBONE" = "b5spx2" ] && [ "$X3" = "1" ]; then
  if have "$X2_M2"; then skip 06 "ENRx member 2 (b5spx3, SOUP_K=$SOUP_K)"
  else
    begin 06 "ENRx member 2 (b5spx3): soup of $SOUP_K all-81 fine-tunes from SP init seed 2"
    [ -f "$SP_INIT2" ] || { echo "FATAL: $SP_INIT2 missing -- run step 05a (BACKBONE=b5spx3)"; exit 1; }
    for f in "$SP_INIT" "$SP_INIT1"; do
      [ "$(sha256sum < "$SP_INIT2")" != "$(sha256sum < "$f")" ] \
        || { echo "FATAL: $SP_INIT2 and $f are identical -- K=3 needs three different pretrain seeds"; exit 1; }
    done
    materialise_all81
    mkdir -p data/checkpoints/sp
    sp_soup_member "$SP_INIT2" "$X2_M2" b5spx2_pre2 $X2_FT2
    done_ 06
  fi
fi
if want 06 && [ "$BACKBONE" = "n1" ]; then
  if have data/checkpoints/ship_simvp_s3/simvp_s3_all81.pth; then skip 06 "shipping backbone"
  else
    begin 06 "shipping backbone (scripts/ship_simvp_finetune_s3.py, all 81 files, seed 0)"
    "$PY" scripts/ship_simvp_finetune_s3.py \
        --out data/checkpoints/ship_simvp_s3/simvp_s3_all81.pth \
        2>&1 | tee "$LOGS/retrain_06_ship_train.log" | tail -n 4
    done_ 06
  fi
fi

# -----------------------------------------------------------------------------
# 07. fp16 pack (halves the weights and writes the "arch" key the loader dispatches on)
# -----------------------------------------------------------------------------
if want 07 && [ "$B5FAM" = "1" ] && [ "$BACKBONE" != "b5spx2" ]; then
  if have "$SHIP_DIR/model.pth"; then skip 07 "pack ($BACKBONE)"
  else
    begin 07 "pack (scripts/p8_pack_b5.py: fp32 weights + \"arch\": \"$B5_MODEL\")"
    "$PY" scripts/p8_pack_b5.py "$SHIP_CKPT" "$SHIP_DIR/model.pth" \
        2>&1 | tee "$LOGS/retrain_07_pack_b5.log"
    done_ 07
  fi
fi
if want 07 && [ "$BACKBONE" = "b5spx2" ] && [ "$X3" != "1" ]; then
  if have "$X2_DIR/model.pth"; then skip 07 "pack (b5spx2)"
  else
    begin 07 "pack (scripts/ens_pack_b5x2.py: both soups fp32 + \"arch\": \"b5x2_h192_blk6_rf11\")"
    for f in "$X2_M0" "$X2_M1"; do [ -f "$f" ] || { echo "FATAL: $f missing -- run step 06 (BACKBONE=b5spx2)"; exit 1; }; done
    mkdir -p "$X2_DIR"
    "$PY" scripts/ens_pack_b5x2.py "$X2_M0" "$X2_M1" "$X2_DIR/model.pth" \
        2>&1 | tee "$LOGS/retrain_07_pack_b5spx2.log"
    done_ 07
  fi
fi
if want 07 && [ "$BACKBONE" = "b5spx2" ] && [ "$X3" = "1" ]; then
  if have "$X2_DIR/model.pth"; then skip 07 "pack (b5spx3)"
  else
    begin 07 "pack (scripts/k3_pack_b5xk.py: the three soups fp32 + \"arch\": \"b5x3_h192_blk6_rf11\")"
    for f in "$X2_M0" "$X2_M1" "$X2_M2"; do [ -f "$f" ] || { echo "FATAL: $f missing -- run step 06 (BACKBONE=b5spx3)"; exit 1; }; done
    mkdir -p "$X2_DIR"
    "$PY" scripts/k3_pack_b5xk.py "$X2_M0" "$X2_M1" "$X2_M2" "$X2_DIR/model.pth" \
        2>&1 | tee "$LOGS/retrain_07_pack_b5spx3.log"
    done_ 07
  fi
fi
if want 07 && [ "$BACKBONE" = "n1" ]; then
  if have data/checkpoints/ship_simvp_s3/model.pth; then skip 07 "fp16 pack"
  else
    begin 07 "fp16 pack (scripts/ship_simvp_pack_fp16.py)"
    "$PY" scripts/ship_simvp_pack_fp16.py \
        data/checkpoints/ship_simvp_s3/simvp_s3_all81.pth \
        data/checkpoints/ship_simvp_s3/model.pth \
        2>&1 | tee "$LOGS/retrain_07_pack.log"
    done_ 07
  fi
fi

# -----------------------------------------------------------------------------
# 08. shipping prior + energy net, fitted on train+val (77 files) -- never test
# -----------------------------------------------------------------------------
if want 08 && [ "$B5FAM" = "1" ] && [ "$BACKBONE" != "b5spx2" ]; then
  if have "$SHIP_DIR/prior.npz" "$SHIP_DIR/ke_net_flow.pt"; then skip 08 "shipping prior / energy-net fits ($BACKBONE)"
  else
    # B1 recipe: fitted on ALL 81 usable files (the n1 lineage below uses data/real/{train,val}, 77 files).
    begin 08 "shipping prior + ke_net_flow, b5 (b1_fit_prior / b1_fit_ke_flow, all 81 files)"
    materialise_all81
    "$PY" scripts/b1_fit_prior.py --model "$B5_MODEL" --files data/real_all81 \
        --ckpt "$SHIP_CKPT" --out "$SHIP_DIR/prior.npz" --device cuda --mem-frac 0.15 \
        > "$LOGS/retrain_08_prior_b5.log" 2>&1
    tail -n 1 "$LOGS/retrain_08_prior_b5.log"
    rm -rf data/cache/retrain_b5_tke_flow_all81   # R4: never resume from a killed run's partial feature cache
    "$PY" scripts/b1_fit_ke_flow.py --tag b1s_ship_all81 --model "$B5_MODEL" \
        --ckpt "$SHIP_CKPT" --files data/real_all81 \
        --cache data/cache/retrain_b5_tke_flow_all81 --mem-frac 0.15 --out "$SHIP_DIR/ke_net_flow.pt" \
        > "$LOGS/retrain_08_keflow_b5.log" 2>&1
    rm -rf data/cache/retrain_b5_tke_flow_all81   # same cache path for b5 / b5sp (deleted after each fit)
    grep -E 'BEST val e' "$LOGS/retrain_08_keflow_b5.log" | tail -n 1
    done_ 08
  fi
fi
if want 08 && [ "$BACKBONE" = "b5spx2" ] && [ "$X3" != "1" ]; then
  if have "$X2_DIR/prior.npz" "$X2_DIR/ke_net_flow.pt"; then skip 08 "ensemble prior / energy-net fits (b5spx2)"
  else
    # ENR recipe on the two soups: the K=2 ENSEMBLE's own prior / ke net, all 81 usable files.
    begin 08 "ensemble prior + ke_net_flow, b5spx2 (en_fit_prior / en_fit_ke_flow${X2_SFX:+ + KE kes_fit_ensemble}, all 81 files)"
    for f in "$X2_M0" "$X2_M1"; do [ -f "$f" ] || { echo "FATAL: $f missing -- run step 06 (BACKBONE=b5spx2)"; exit 1; }; done
    materialise_all81
    mkdir -p "$X2_DIR"
    X2_CACHE="data/cache/retrain_b5spx2_tke_flow_all81"   # b5spx2-only cache path, deleted after the fit
    "$PY" scripts/en_fit_prior.py --files data/real_all81 --ckpts "$X2_M0" "$X2_M1" \
        --out "$X2_DIR/prior.npz" --device cuda --mem-frac 0.15 > "$LOGS/retrain_08_prior_b5spx2.log" 2>&1
    tail -n 1 "$LOGS/retrain_08_prior_b5spx2.log"
    # single ensemble ke net = the GATED config (shipped when ENRX_KE=0); with ENRX_KE=1 it only builds the energy cache
    x2_ke_single="$X2_DIR/ke_net_flow.pt"; [ "$ENRX_KE" = "1" ] && x2_ke_single="$X2_DIR/ke_net_flow_single.pt"
    rm -rf "$X2_CACHE"
    "$PY" scripts/en_fit_ke_flow.py --tag b5spx2_ship_all81 --ckpts "$X2_M0" "$X2_M1" --files data/real_all81 \
        --cache "$X2_CACHE" --mem-frac 0.15 --out "$x2_ke_single" \
        > "$LOGS/retrain_08_keflow_b5spx2.log" 2>&1
    grep -E 'BEST val e' "$LOGS/retrain_08_keflow_b5spx2.log" | tail -n 1
    if [ "$ENRX_KE" = "1" ]; then
      # KE: 4-seed energy nets on the SAME cache / internal split, packed into one grouped-conv net (ENR's recipe)
      "$PY" scripts/kes_fit_ensemble.py --cache "$X2_CACHE" --files-dir data/real_all81 \
          --seeds 0,1,2,3 --epochs 120 --mem-frac 0.15 --out-dir "$X2_DIR/kes" --ckpt-dir "$X2_DIR/kes" \
          --tag kes_ship_all81_b5spx2 > "$LOGS/retrain_08_kes_b5spx2.log" 2>&1
      grep -E 'relative gain' "$LOGS/retrain_08_kes_b5spx2.log" | tail -n 1
      "$PY" scripts/ens_kes_pack.py "$X2_DIR/kes" 2>&1 | tee "$LOGS/retrain_08_kes_pack_b5spx2.log"
      cp -f "$X2_DIR/kes/ke_net_flow_packed_k4.pt" "$X2_DIR/ke_net_flow.pt"
    fi
    rm -rf "$X2_CACHE"
    done_ 08
  fi
fi
if want 08 && [ "$BACKBONE" = "b5spx2" ] && [ "$X3" = "1" ]; then
  if have "$X2_DIR/prior.npz" "$X2_DIR/ke_net_flow.pt"; then skip 08 "ensemble prior / energy-net fits (b5spx3)"
  else
    # ENR recipe on the three soups: the K=3 ENSEMBLE's own prior / single ke net, all 81 usable files.
    begin 08 "ensemble prior + ke_net_flow, b5spx3 (en_fit_prior / en_fit_ke_flow, 3 members, all 81 files)"
    for f in "$X2_M0" "$X2_M1" "$X2_M2"; do [ -f "$f" ] || { echo "FATAL: $f missing -- run step 06 (BACKBONE=b5spx3)"; exit 1; }; done
    materialise_all81
    mkdir -p "$X2_DIR"
    X3_CACHE="data/cache/retrain_b5spx3_tke_flow_all81"   # b5spx3-only cache path, deleted after the fit
    "$PY" scripts/en_fit_prior.py --files data/real_all81 --ckpts "$X2_M0" "$X2_M1" "$X2_M2" \
        --out "$X2_DIR/prior.npz" --device cuda --mem-frac 0.15 > "$LOGS/retrain_08_prior_b5spx3.log" 2>&1
    tail -n 1 "$LOGS/retrain_08_prior_b5spx3.log"
    rm -rf "$X3_CACHE"
    "$PY" scripts/en_fit_ke_flow.py --tag b5spx3_ship_all81 --ckpts "$X2_M0" "$X2_M1" "$X2_M2" --files data/real_all81 \
        --cache "$X3_CACHE" --mem-frac 0.15 --out "$X2_DIR/ke_net_flow$([ "${KE3:-0}" = "1" ] && echo _single).pt" \
        > "$LOGS/retrain_08_keflow_b5spx3.log" 2>&1
    grep -E 'BEST val e' "$LOGS/retrain_08_keflow_b5spx3.log" | tail -n 1
    if [ "${KE3:-0}" = "1" ]; then
      # KE3: 4-seed energy nets on the SAME cache / internal split (KE's recipe), written as one dense KeFlowNet
      "$PY" scripts/kes_fit_ensemble.py --cache "$X3_CACHE" --files-dir data/real_all81 \
          --seeds 0,1,2,3 --epochs 120 --mem-frac 0.15 --out-dir "$X2_DIR/kes" --ckpt-dir "$X2_DIR/kes" \
          --tag kes_ship_all81_b5spx3 > "$LOGS/retrain_08_kes_b5spx3.log" 2>&1
      grep -E 'relative gain' "$LOGS/retrain_08_kes_b5spx3.log" | tail -n 1
      "$PY" scripts/c2_ke_dense.py --check --out "$X2_DIR/ke_net_flow.pt" "$X2_DIR"/kes/ke_net_flow_seed{0,1,2,3}.pt \
          2>&1 | tee "$LOGS/retrain_08_kes_dense_b5spx3.log"
      [ -f "$X2_DIR/ke_net_flow.pt" ] || { echo "FATAL: KE3 dense ke net not written"; exit 1; }
    fi
    rm -rf "$X3_CACHE"
    done_ 08
  fi
fi
if want 08 && [ "$BACKBONE" = "n1" ]; then
  if have data/checkpoints/ship_simvp_s3/prior.npz data/checkpoints/ship_simvp_s3/ke_net_flow.pt
  then skip 08 "shipping prior / energy-net fits"
  else
    begin 08 "shipping prior + ke_net_flow (ship_simvp_fit_prior / ship_simvp_fit_ke_flow)"
    "$PY" scripts/ship_simvp_fit_prior.py \
        --files data/real/train data/real/val \
        --ckpt data/checkpoints/ship_simvp_s3/simvp_s3_all81.pth \
        --out data/checkpoints/ship_simvp_s3/prior.npz --device cuda --mem-frac 0.15 \
        > "$LOGS/retrain_08_prior.log" 2>&1
    tail -n 1 "$LOGS/retrain_08_prior.log"
    rm -rf data/cache/retrain_ship_tke_flow_all81   # R4: never resume from a killed run's partial feature cache
    "$PY" scripts/ship_simvp_fit_ke_flow.py \
        --tag ship_simvp_s3_all81 --ckpt data/checkpoints/ship_simvp_s3/simvp_s3_all81.pth \
        --files data/real/train data/real/val \
        --cache data/cache/retrain_ship_tke_flow_all81 \
        --out data/checkpoints/ship_simvp_s3/ke_net_flow.pt --mem-frac 0.2 --device cuda \
        > "$LOGS/retrain_08_keflow.log" 2>&1
    rm -rf data/cache/retrain_ship_tke_flow_all81
    grep -E 'BEST val e' "$LOGS/retrain_08_keflow.log" | tail -n 1
    done_ 08
  fi
fi

# -----------------------------------------------------------------------------
# 09. conditional-width net. Default (CONDWIDTH_RECIPE=w2): the W2 v2 shipping
#     fit on the step-05b K=4 all-81 OUT-OF-SAMPLE feature caches (coverage-aware
#     5x5 oracle-width target, m1 features / arch, seed 0; alpha / lambda chosen on
#     the fit's own 20 % val split -- expected (1.0, 1.05)), then converted to the
#     packaged cond_width_net format. `--test` is REPORT-ONLY in w2_fit.py (never
#     used for selection): the first training cache is passed so a fresh clone
#     needs no extra reference cache.
# -----------------------------------------------------------------------------
if want 09 && [ "$CONDWIDTH_RECIPE" = "w2" ]; then
  if have "$W2_OUT/cond_width_weights.pt"; then skip 09 "W2 v2 conditional-width fit"
  else
    begin 09 "W2 v2 conditional-width fit (scripts/w2_fit.py + scripts/w2_convert_ckpt.py, K=4 all-81 caches, best-of-N seeds)"
    w2_feats=""
    for g in 0 1 2 3; do
      if [ "$B5FAM" = "1" ]; then
        if [ -f "$REPO/experiments/b1_3090/cache/feat_all81_g$g.npz" ]; then w2_feats="$w2_feats experiments/b1_3090/cache/feat_all81_g$g.npz"
        else echo "FATAL: b5 feature cache for all81 group $g missing -- run step 05b (BACKBONE=b5, KFOLD=all81)"; exit 1; fi
      elif [ -f "$REPO/experiments/w2p_packaging/cache/feat_all81_g$g.npz" ]; then w2_feats="$w2_feats experiments/w2p_packaging/cache/feat_all81_g$g.npz"
      elif [ -f "$REPO/experiments/w2_condwidth_v2/cache/feat_all81_g$g.npz" ]; then w2_feats="$w2_feats experiments/w2_condwidth_v2/cache/feat_all81_g$g.npz"
      else echo "FATAL: feature cache for all81 group $g missing -- run step 05b with KFOLD=all81"; exit 1; fi
    done
    mkdir -p "$W2_OUT"
    set -- $w2_feats; w2_test="$1"
    # R1E best-of-N seed selection (found 2026-09-23, `experiments/r1e_rehearsal/`):
    # fit W2_SEEDS (default "0 1 2") independent seeds on the SAME K=4 caches and
    # pick by each fit's OWN val-grid line only (never by shim/held-out numbers,
    # per COMMON_RULES honesty rule) -- the previous single-seed (seed 0) behaviour
    # is W2_SEEDS="0". Reports how many were tried.
    W2_SEEDS="${W2_SEEDS:-0 1 2}"
    for s in $W2_SEEDS; do
      "$PY" scripts/w2_fit.py --train $w2_feats --test "$w2_test" \
          --feats m1 --target cov5x5 --arch m1 --seed "$s" --mem-frac 0.25 --tag "ship_all81_cov5x5_seed$s" \
          --out "$W2_OUT/ship_all81_cov5x5_seed$s.json" --ckpt "$W2_OUT/ship_all81_cov5x5_seed$s.pt" \
          > "$LOGS/retrain_09_w2_fit_seed$s.log" 2>&1
      grep -E '\[val grid\]' "$LOGS/retrain_09_w2_fit_seed$s.log" | tail -n 1
    done
    best_seed=$("$PY" -c "
import json
seeds = '$W2_SEEDS'.split()
best_s, best_d = None, None
for s in seeds:
    d = json.load(open('$W2_OUT/ship_all81_cov5x5_seed' + s + '.json'))['val_delta']
    print(f'  seed {s}: val_delta={d:.4f}')
    if best_d is None or d > best_d:
        best_s, best_d = s, d
print(f'best-of-{len(seeds)}: seed {best_s} (val_delta={best_d:.4f}), picked on its OWN val line')
print(best_s)
" | tee -a "$LOGS/retrain_09_w2_seed_select.log" | tail -n 1)
    cp -f "$W2_OUT/ship_all81_cov5x5_seed$best_seed.pt" "$W2_OUT/ship_all81_cov5x5.pt"
    cp -f "$W2_OUT/ship_all81_cov5x5_seed$best_seed.json" "$W2_OUT/ship_all81_cov5x5.json"
    "$PY" scripts/w2_convert_ckpt.py --inp "$W2_OUT/ship_all81_cov5x5.pt" --out "$W2_OUT/cond_width_weights.pt" \
        2>&1 | tee "$LOGS/retrain_09_w2_convert.log" | tail -n 2
    echo "  $(sha256sum "$W2_OUT/cond_width_weights.pt")   (shipped 2026-09-22, n1: 16a65296dc561e04143f3e8a5bf056bb8e5a0cb4ec314ae59a37f2c576e4d089)"
    done_ 09
  fi
fi

# 09 (previous lineage, CONDWIDTH_RECIPE=m1): M1 nets fitted on the step-05 residuals
# -----------------------------------------------------------------------------
if want 09 && [ "$CONDWIDTH_RECIPE" = "m1" ]; then
  if have data/checkpoints/ship_condwidth/net_all3.pt \
          data/checkpoints/ship_condwidth/net_without_fold0.pt
  then skip 09 "conditional-width fit"
  else
    begin 09 "conditional-width fit (scripts/ship_condwidth_fit.py; all3 = SHIPPED)"
    "$PY" scripts/ship_condwidth_fit.py --device cuda --epochs 80 --patience 6 \
        --mem-frac 0.35 --out-dir data/checkpoints/ship_condwidth \
        --report "$LOGS/retrain_09_condwidth_fit.json" \
        > "$LOGS/retrain_09_condwidth.log" 2>&1
    grep -E 'alpha/lambda grid best on val' "$LOGS/retrain_09_condwidth.log"
    done_ 09
  fi
fi

# R1E FIX (bug found 2026-09-23 fresh-clone rehearsal, `experiments/r1e_rehearsal/notes.md`):
# KEEP_DUMPS=0's cleanup lived ONLY inside the CONDWIDTH_RECIPE=m1 branch above, so it silently
# never fired for the current default lineage (CONDWIDTH_RECIPE=w2) -- confirmed: after a full
# w2-recipe run, data/cache/p3_dumps/{fold0,fold1,foldR}_c.npz (~3.6 GB) were still on disk
# despite KEEP_DUMPS=0 and despite their step-05 score JSONs having existed since step 05
# finished. Moved out of the recipe-specific branch so it runs after step 09 either way.
if [ "$KEEP_DUMPS" = "0" ]; then
  echo "  KEEP_DUMPS=0: removing $DUMPS (~4 GB)"
  rm -rf "$DUMPS"
fi

# -----------------------------------------------------------------------------
# 10. assemble the archive: tracked weight-free source + the four new weight files
# -----------------------------------------------------------------------------
if want 10; then
  if have "$OUT_DIR/submission.zip"; then skip 10 "package assembly"
  else
    begin 10 "package assembly (scripts/pack_zip.py)"
    rm -rf "$BUILD_PKG"
    mkdir -p "$BUILD_PKG"
    # everything the tracked source holds EXCEPT its own README (not part of the
    # archive) and any __pycache__; pack_zip.py drops __pycache__ again anyway.
    ( cd "$SRC_PKG" && find . -type f ! -name README.md ! -path '*/__pycache__/*' -print0 ) \
      | ( cd "$SRC_PKG" && xargs -0 -I{} cp --parents {} "$BUILD_PKG/" )
    cp -f "$SHIP_DIR/model.pth"                           "$BUILD_PKG/model.pth"
    cp -f "$SHIP_DIR/prior.npz"                           "$BUILD_PKG/prior.npz"
    cp -f "$SHIP_DIR/ke_net_flow.pt"                      "$BUILD_PKG/ke_net_flow.pt"
    "$PY" -c "import torch,sys; p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); a=p.get('arch'); print('  model.pth arch:', a); sys.exit(0 if a==sys.argv[2] else 1)" \
        "$BUILD_PKG/model.pth" "$ARCH_EXPECT" \
      || { echo "FATAL: model.pth arch key does not match BACKBONE=$BACKBONE"; exit 1; }
    if [ "$CONDWIDTH_RECIPE" = "w2" ]; then
      cp -f "$W2_OUT/cond_width_weights.pt"               "$BUILD_PKG/cond_width_weights.pt"
    else
      cp -f data/checkpoints/ship_condwidth/net_all3.pt   "$BUILD_PKG/cond_width_weights.pt"
    fi
    # the shipped archive's submission.py must come through byte-identical
    a=$(sha256sum "$SRC_PKG/submission.py" | cut -d' ' -f1)
    b=$(sha256sum "$BUILD_PKG/submission.py" | cut -d' ' -f1)
    [ "$a" = "$b" ] || { echo "FATAL: submission.py differs from the tracked source"; exit 1; }
    grep -q '"cond_width": True' "$BUILD_PKG/submission.py" \
      || { echo "FATAL: CONFIG[\"cond_width\"] is not True in the assembled package"; exit 1; }
    if [ "${FP16:-0}" = "1" ]; then
      grep -q '"fp16_core": True' "$BUILD_PKG/submission.py" \
        || { echo "FATAL: FP16=1 but CONFIG[\"fp16_core\"] is not True in the assembled package"; exit 1; }
      if [ "${AXE:-0}" = "1" ]; then   # K3E
        { [ -f "$BUILD_PKG/axe_adapt.py" ] && grep -q '"br_adapt": {"grid": \[0.125, 0.25, 0.5, 1.0\]' "$BUILD_PKG/submission.py"; } \
          || { echo "FATAL: AXE=1 but the assembled b5spx3 package has no axe_adapt.py / br_adapt CONFIG"; exit 1; }
      fi
      if [ "${KD2:-0}" = "1" ]; then   # KD2 kernel diet
        { [ -f "$BUILD_PKG/kd2_adapt.py" ] && grep -q '"kd2_adapt": True' "$BUILD_PKG/submission.py"; } \
          || { echo "FATAL: KD2=1 but the assembled b5spx3 package has no kd2_adapt.py / CONFIG kd2_adapt True"; exit 1; }
      else
        [ ! -f "$BUILD_PKG/kd2_adapt.py" ] && ! grep -q '"kd2_adapt"' "$BUILD_PKG/submission.py" \
          || { echo "FATAL: KD2=0 but the assembled b5spx3 package carries kd2_adapt"; exit 1; }
      fi
      if [ "${KD3:-0}" = "1" ]; then   # KD3 whole-step kernel diet
        { [ -f "$BUILD_PKG/kd3_step.py" ] && grep -q '"kd3_step": True' "$BUILD_PKG/submission.py"; } \
          || { echo "FATAL: KD3=1 but the assembled b5spx3 package has no kd3_step.py / CONFIG kd3_step True"; exit 1; }
      else
        [ ! -f "$BUILD_PKG/kd3_step.py" ] && ! grep -q '"kd3_step"' "$BUILD_PKG/submission.py" \
          || { echo "FATAL: KD3=0 but the assembled b5spx3 package carries kd3_step"; exit 1; }
      fi
      if [ "${IPC:-0}" = "1" ]; then   # IPC one-storage return
        { grep -q '"ipc_one_storage": "auto"' "$BUILD_PKG/submission.py" && grep -q '_ipc_one_storage' "$BUILD_PKG/cuda_graph_wholestep.py" \
          && grep -q '_ipc_one_storage' "$BUILD_PKG/w01_warmup_graph.py"; } \
          || { echo "FATAL: IPC=1 but the assembled b5spx3 package has no CONFIG ipc_one_storage auto / one-storage return"; exit 1; }
      else
        ! grep -q '"ipc_one_storage"' "$BUILD_PKG/submission.py" && ! grep -q '_ipc_one_storage' "$BUILD_PKG/cuda_graph_wholestep.py" \
          || { echo "FATAL: IPC=0 but the assembled b5spx3 package carries ipc_one_storage"; exit 1; }
      fi
      if [ "${KD5:-0}" = "1" ]; then   # KD5 near-exact whole-step graph diet
        { grep -q '"kd5_near": True' "$BUILD_PKG/submission.py" && grep -q 'kd5_near' "$BUILD_PKG/kd3_step.py"; } \
          || { echo "FATAL: KD5=1 but the assembled b5spx3 package has no CONFIG kd5_near True / KD5 kd3_step.py"; exit 1; }
      else
        ! grep -q '"kd5_near"' "$BUILD_PKG/submission.py" && ! grep -qs 'kd5_near' "$BUILD_PKG/kd3_step.py" \
          || { echo "FATAL: KD5=0 but the assembled b5spx3 package carries kd5_near"; exit 1; }
      fi
      if [ "${KD4:-0}" = "1" ]; then   # KD4 W01 diet
        { [ -f "$BUILD_PKG/kd4_w01.py" ] && grep -q '"kd4_w01": True' "$BUILD_PKG/submission.py"; } \
          || { echo "FATAL: KD4=1 but the assembled b5spx3 package has no kd4_w01.py / CONFIG kd4_w01 True"; exit 1; }
        if [ "${KD5:-0}" = "1" ]; then   # FZ: KD4's W01 capture must trace KD3's exact branches when KD5 is on
          grep -q 'st.near = False' "$BUILD_PKG/kd4_w01.py" \
            || { echo "FATAL: KD4=1 KD5=1 but kd4_w01.py lacks the FZ near-off capture hunk"; exit 1; }
        fi
      else
        [ ! -f "$BUILD_PKG/kd4_w01.py" ] && ! grep -q '"kd4_w01"' "$BUILD_PKG/submission.py" \
          || { echo "FATAL: KD4=0 but the assembled b5spx3 package carries kd4_w01"; exit 1; }
      fi
      if [ "${REC:-0}" = "1" ]; then   # REC regime-change recovery
        { [ -f "$BUILD_PKG/rec_detect.py" ] && grep -q '"rec_detect": {"thr": 0.405465' "$BUILD_PKG/submission.py"; } \
          || { echo "FATAL: REC=1 but the assembled b5spx3 package has no rec_detect.py / CONFIG rec_detect (thr log 1.5)"; exit 1; }
      else
        [ ! -f "$BUILD_PKG/rec_detect.py" ] && ! grep -q '"rec_detect"' "$BUILD_PKG/submission.py" \
          || { echo "FATAL: REC=0 but the assembled b5spx3 package carries rec_detect"; exit 1; }
      fi
      if [ "${CDIO:-0}" = "1" ]; then   # CDIO CUDA-input fast path (HX2)
        { grep -q '"cuda_direct_io": True' "$BUILD_PKG/submission.py" && grep -q '_ttt_step_cuda_direct' "$BUILD_PKG/cuda_graph_wholestep.py" \
          && grep -q '"ipc_prewarm": True' "$BUILD_PKG/submission.py"; } \
          || { echo "FATAL: CDIO=1 but the assembled b5spx3 package has no CONFIG cuda_direct_io True / CUDA-direct path / ipc_prewarm"; exit 1; }
      else
        ! grep -q '"cuda_direct_io"' "$BUILD_PKG/submission.py" && ! grep -q '_ttt_step_cuda_direct' "$BUILD_PKG/cuda_graph_wholestep.py" \
          || { echo "FATAL: CDIO=0 but the assembled b5spx3 package carries cuda_direct_io"; exit 1; }
      fi
      if [ "${LOWKE:-0}" = "1" ]; then   # TKA low-ke fluctuation damping of the emitted pred
        { grep -q '"tka_lowke": {"theta": 1.5, "delta": 0.25, "gmin": 0.0}' "$BUILD_PKG/submission.py" && grep -q '_lowke_apply' "$BUILD_PKG/cuda_graph_wholestep.py"; } \
          || { echo "FATAL: LOWKE=1 but the assembled b5spx3 package has no CONFIG tka_lowke {1.5, 0.25, 0.0} / _lowke_apply"; exit 1; }
      else
        ! grep -q '"tka_lowke"' "$BUILD_PKG/submission.py" && ! grep -q '_lowke_apply' "$BUILD_PKG/cuda_graph_wholestep.py" \
          || { echo "FATAL: LOWKE=0 but the assembled b5spx3 package carries tka_lowke"; exit 1; }
      fi
      if [ "${KD6:-0}" = "1" ]; then   # KD6 fork / join concurrency in the steady whole-step graph
        { [ -f "$BUILD_PKG/kd6_conc.py" ] && grep -q '"kd6_concurrency": True' "$BUILD_PKG/submission.py" \
          && grep -q '_kd6_build' "$BUILD_PKG/cuda_graph_wholestep.py"; } \
          || { echo "FATAL: KD6=1 but the assembled b5spx3 package has no kd6_conc.py / CONFIG kd6_concurrency True / _kd6_build"; exit 1; }
      else
        [ ! -f "$BUILD_PKG/kd6_conc.py" ] && ! grep -q '"kd6_concurrency"' "$BUILD_PKG/submission.py" \
          || { echo "FATAL: KD6=0 but the assembled b5spx3 package carries kd6_concurrency"; exit 1; }
      fi
      if [ "${KD7:-0}" = "1" ]; then   # KD7 bit-exact jiterator fusion of the KD6 body's post-join chains
        { [ -f "$BUILD_PKG/kd7_fuse.py" ] && grep -q '"kd7": True' "$BUILD_PKG/submission.py" \
          && grep -q 'kd7_fuse' "$BUILD_PKG/cuda_graph_wholestep.py" && grep -q '_kd7j' "$BUILD_PKG/kd6_conc.py"; } \
          || { echo "FATAL: KD7=1 but the assembled b5spx3 package has no kd7_fuse.py / CONFIG kd7 True / KD7 hooks"; exit 1; }
      else
        [ ! -f "$BUILD_PKG/kd7_fuse.py" ] && ! grep -q '"kd7"' "$BUILD_PKG/submission.py" \
          || { echo "FATAL: KD7=0 but the assembled b5spx3 package carries kd7"; exit 1; }
      fi
      if [ "${LSK:-0}" = "1" ]; then   # LSK frozen separable LS head (update on a KD6 side stream)
        LSK_CFG_CHK='"lsk_head": {"r": 8, "rho": 0.9, "lam": 30.0}'; [ "${LSKR97:-0}" = "1" ] && LSK_CFG_CHK='"lsk_head": {"r": 8, "rho": 0.97, "lam": 30.0}'
        { [ -f "$BUILD_PKG/lsk_head.py" ] && grep -qF "$LSK_CFG_CHK" "$BUILD_PKG/submission.py" \
          && grep -q 'side_c' "$BUILD_PKG/kd6_conc.py" && grep -q '_lsk.update(y)' "$BUILD_PKG/cuda_graph_wholestep.py"; } \
          || { echo "FATAL: LSK=1 but the assembled b5spx3 package has no lsk_head.py / CONFIG lsk_head / LSK hooks"; exit 1; }
      else
        [ ! -f "$BUILD_PKG/lsk_head.py" ] && ! grep -q '"lsk_head"' "$BUILD_PKG/submission.py" \
          || { echo "FATAL: LSK=0 but the assembled b5spx3 package carries lsk_head"; exit 1; }
      fi
      if [ "${LSKR:-0}" = "1" ]; then   # LSKR: REC reset of the LS state + self-test rho fix
        { grep -q '"lsk_rec_reset": True' "$BUILD_PKG/submission.py" && grep -q 'def rec_reset_' "$BUILD_PKG/lsk_head.py" \
          && grep -q '_lsk.rec_reset_(rec.keep)' "$BUILD_PKG/kd6_conc.py" && grep -q '_lsk.rec_reset_(rec.keep)' "$BUILD_PKG/cuda_graph_wholestep.py"; } \
          || { echo "FATAL: LSKR=1 but the assembled b5spx3 package has no CONFIG lsk_rec_reset True / rec_reset_ hooks"; exit 1; }
      else
        ! grep -q '"lsk_rec_reset"' "$BUILD_PKG/submission.py" && ! grep -q 'rec_reset_' "$BUILD_PKG/kd6_conc.py" \
          || { echo "FATAL: LSKR=0 but the assembled b5spx3 package carries lsk_rec_reset"; exit 1; }
      fi
      if [ "${KD8:-0}" = "1" ]; then   # KD8 more of the KD6 body off the main stream + 6 bit-exact jiterator kernels
        { [ -f "$BUILD_PKG/kd8_fuse.py" ] && grep -q '"kd8": True' "$BUILD_PKG/submission.py" \
          && grep -q 'kd8_fuse' "$BUILD_PKG/cuda_graph_wholestep.py" && grep -q '_kd8j' "$BUILD_PKG/kd6_conc.py"; } \
          || { echo "FATAL: KD8=1 but the assembled b5spx3 package has no kd8_fuse.py / CONFIG kd8 True / KD8 hooks"; exit 1; }
        if [ "${LSKR:-0}" = "1" ]; then   # the REC reset must also sit on KD8's side_c path ("lskc")
          [ "$(grep -c '_lsk.rec_reset_(rec.keep)' "$BUILD_PKG/kd6_conc.py")" = "2" ] && grep -q 'rr8 = _lsk.rec_reset' "$BUILD_PKG/kd6_conc.py" \
            || { echo "FATAL: KD8=1 LSKR=1 but the assembled b5spx3 package lacks the KD8 side_c REC-reset hook"; exit 1; }
        fi
      else
        [ ! -f "$BUILD_PKG/kd8_fuse.py" ] && ! grep -q '"kd8"' "$BUILD_PKG/submission.py" \
          || { echo "FATAL: KD8=0 but the assembled b5spx3 package carries kd8"; exit 1; }
      fi
      if [ "${LSKR97:-0}" = "1" ]; then   # LSKR97: the only change vs KD8 LSKR -- lsk_head rho 0.97
        grep -q '"lsk_head": {"r": 8, "rho": 0.97, "lam": 30.0}' "$BUILD_PKG/submission.py" && ! grep -q '"lsk_head": {"r": 8, "rho": 0.9,' "$BUILD_PKG/submission.py" \
          || { echo "FATAL: LSKR97=1 but the assembled b5spx3 package has no CONFIG lsk_head rho 0.97"; exit 1; }
      fi
      if [ "${UV:-0}" = "1" ]; then   # UV: p = +0.0 on every output path + the u, v-only by-value form
        { grep -q '"bvr_uv_only": True' "$BUILD_PKG/submission.py" && grep -q '^class _BVUVArray' "$BUILD_PKG/submission.py" \
          && grep -q '_dev_stage_i32' "$BUILD_PKG/cuda_graph_wholestep.py" && grep -q 'def _d2h' "$BUILD_PKG/w01_warmup_graph.py"; } \
          || { echo "FATAL: UV=1 but the assembled b5spx3 package has no CONFIG bvr_uv_only True / _BVUVArray / p-zero hooks"; exit 1; }
      else
        ! grep -q '"bvr_uv_only"' "$BUILD_PKG/submission.py" && ! grep -q '_dev_stage_i32' "$BUILD_PKG/cuda_graph_wholestep.py" \
          || { echo "FATAL: UV=0 but the assembled b5spx3 package carries bvr_uv_only"; exit 1; }
      fi
    fi
    "$PY" scripts/pack_zip.py --pkg "$BUILD_PKG" --out "$OUT_DIR/submission.zip" \
        2>&1 | tee "$LOGS/retrain_10_package.log"
    ( cd "$BUILD_PKG" && find . -type f | sort | xargs sha256sum ) > "$OUT_DIR/SHA256SUMS"
    sha256sum "$OUT_DIR/submission.zip" >> "$OUT_DIR/SHA256SUMS"
    echo "  $(sha256sum "$OUT_DIR/submission.zip")"
    done_ 10
  fi
fi

# -----------------------------------------------------------------------------
# 11. official local scorer on the held-out local splits (SANITY ONLY -- those
#     files are memorised by the all-81 backbone; see CLAUDE.md / HANDOVER S3.1)
# -----------------------------------------------------------------------------
if want 11; then
  if have "$LOGS/retrain_11_local_eval_val.log" "$LOGS/retrain_11_local_eval_test.log"
  then skip 11 "local_eval sanity run"
  elif [ ! -f "$BUILD_PKG/submission.py" ]; then
    echo "=== [11] local_eval -- SKIP (no assembled package at $BUILD_PKG; run step 10 first) ==="
  else
  begin 11 "local_eval.py sanity run on data/real/{val,test}"
  for split in val test; do
    shim="$(mktemp -d)"
    mkdir -p "$shim/test_real"
    for f in data/real/$split/*.h5; do ln -s "$(readlink -f "$f")" "$shim/test_real/$(basename "$f")"; done
    ln -s "$REPO/starting_kit/example_data/mean_std_real.pt" "$shim/mean_std_real.pt"
    "$PY" starting_kit/local_eval.py --submission "$BUILD_PKG" --data "$shim" \
        > "$LOGS/retrain_11_local_eval_$split.log" 2>&1 || true
    rm -rf "$shim"
    echo "  --- $split ---"
    grep -E '_score|OK:|ERROR' "$LOGS/retrain_11_local_eval_$split.log" | sed 's/^/  /'
  done
  done_ 11
  fi
fi

# -----------------------------------------------------------------------------
# report
# -----------------------------------------------------------------------------
mkdir -p "$OUT_DIR"
{
  echo "retrain_all.sh report -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo HEAD: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '(not a git checkout)')"
  echo "BACKBONE=$BACKBONE  SOUP_K=$SOUP_K  SHIP_DIR=$SHIP_DIR  SHIP_CKPT=$SHIP_CKPT  W2_OUT=$W2_OUT"
  [ "$BACKBONE" = "b5sp" ] && [ "$SOUP_K" -gt 1 ] && cat "$LOGS/retrain_06_ship_soup.log" 2>/dev/null
  if [ "$BACKBONE" = "b5spx2" ]; then
    echo "b5spx2 (c2 ENRx): SOUP_K=$SOUP_K  ENRX_KE=$ENRX_KE  X2_DIR=$X2_DIR  SRC_PKG=$SRC_PKG"
    echo "  member 0 = $X2_M0   member 1 = $X2_M1"
    echo "  -- SP inits --"
    for f in "$SP_INIT" "$SP_INIT1"; do [ -f "$f" ] && sha256sum "$f" | sed 's/^/  /'; done
    for t in pre0 pre1; do
      echo "  -- soup $t (scripts/c2_make_soup.py; absent = soup reused from an earlier run, or SOUP_K=1) --"
      cat "$LOGS/retrain_06_ship_soup_b5spx2_$t.log" 2>/dev/null | sed 's/^/  /'
    done
    echo "  -- fine-tunes / members / X2_DIR artefacts --"
    if [ "$X3" = "1" ]; then
      echo "  b5spx3 (K3F): FP16=$FP16  AXE=${AXE:-0}  KD2=${KD2:-0}  KD3=${KD3:-0}  IPC=${IPC:-0}  KD5=${KD5:-0}  KD4=${KD4:-0}  REC=${REC:-0}  member 2 = $X2_M2"
      [ -f "$SP_INIT2" ] && sha256sum "$SP_INIT2" | sed 's/^/  /'
      cat "$LOGS/retrain_06_ship_soup_b5spx2_pre2.log" 2>/dev/null | sed 's/^/  /'
    fi
    for f in $X2_FT0 $X2_FT1 ${X2_FT2:-} "$X2_M0" "$X2_M1" ${X2_M2:-} "$X2_DIR/model.pth" "$X2_DIR/prior.npz" "$X2_DIR/ke_net_flow.pt"; do
      [ -f "$f" ] && sha256sum "$f" | sed 's/^/  /'
    done
  fi
  echo
  echo "== wall time per step (seconds) =="
  cat "$LOGS/retrain_timings.txt" 2>/dev/null || true
  echo
  echo "== artefact sha256 =="
  for f in data/real/train data/real/val data/real/test; do
    echo "  (split) $f: $(ls $f/*.h5 2>/dev/null | wc -l) files"
  done
  for f in "$FOLD_CKPT_fold0" "$FOLD_CKPT_fold1" "$FOLD_CKPT_foldR" \
           experiments/d2_bundle/fold0_c/prior.npz experiments/d2_bundle/fold0_c/ke_net_flow.pt \
           experiments/d2_bundle/fold1_c/prior.npz experiments/d2_bundle/fold1_c/ke_net_flow.pt \
           experiments/d2_bundle/foldR_c/prior.npz experiments/d2_bundle/foldR_c/ke_net_flow.pt \
           data/checkpoints/ship_simvp_s3/simvp_s3_all81.pth \
           data/checkpoints/ship_simvp_s3/model.pth \
           data/checkpoints/ship_simvp_s3/prior.npz \
           data/checkpoints/ship_simvp_s3/ke_net_flow.pt \
           data/checkpoints/b1_3090/ship/b5_s3_all81.pth \
           data/checkpoints/b1_3090/ship/model.pth \
           data/checkpoints/b1_3090/ship/prior.npz \
           data/checkpoints/b1_3090/ship/ke_net_flow.pt \
           data/checkpoints/sp/b5_sim_s3_it10k.pth \
           data/checkpoints/sp/b5sp_s3_all81.pth \
           data/checkpoints/sp/ship/model.pth \
           data/checkpoints/sp/ship/prior.npz \
           data/checkpoints/sp/ship/ke_net_flow.pt \
           "$W2_OUT/cond_width_weights.pt" \
           data/checkpoints/ship_condwidth/net_all3.pt \
           data/checkpoints/ship_condwidth/net_without_fold0.pt \
           "$OUT_DIR/submission.zip"; do
    [ -f "$f" ] && sha256sum "$f"
  done
  echo
  echo "== fold held-out composites (time subscore fixed at 88) =="
  for fold in fold0 fold1 foldR; do
    eval "ref=\$REF_$fold"
    if [ -f "$LOGS/retrain_05_score_$fold.log" ]; then
      printf '  %s (reference %s): %s\n' "$fold" "$ref" \
        "$(grep -m1 composite= "$LOGS/retrain_05_score_$fold.log")"
    fi
  done
  echo
  echo "== local_eval (memorised splits -- sanity only) =="
  for split in val test; do
    [ -f "$LOGS/retrain_11_local_eval_$split.log" ] && {
      echo "  --- $split ---"
      grep -E '_score' "$LOGS/retrain_11_local_eval_$split.log" | sed 's/^/  /'; }
  done
} | tee "$OUT_DIR/retrain_report.txt" 2>/dev/null || true

echo
echo "== retrain_all.sh finished =="
[ -f "$OUT_DIR/submission.zip" ] && echo "package: $OUT_DIR/submission.zip"
