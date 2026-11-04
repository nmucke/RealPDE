# Reproducing the leaderboard submission — `scripts/retrain_all.sh`

**INTENDED RETRAIN TARGET: `FZ-EIKC + BVR + LOWKE + KD6 + KD7 + LSK (ρ 0.97, REC reset) + KD8`** ← *board-proven: 2026-09-27 row 82.452
(1st on the development leaderboard at 11:28 UTC 09-27; time 92.46 ⇒ every component active in the eval environment). A bare
`bash scripts/retrain_all.sh` builds it (reference zip `f11d2c1d0e00b6270ad71316117eb44e0de303b747bf9302ddcbfcdc7804504e`,
`submissions/2026-10-23_…-lsk-recr-kd8-r97/`). The only line to edit if the target changes.*
**Previous board rows as fallbacks:** 09-26 `fa948762` (82.419) = `LSK=0 LSKR=0 KD8=0 LSKR97=0 bash scripts/retrain_all.sh`.

Every command below was checked on 2026-09-27 with `DRY_RUN=1` (exit 0; the `SRC_PKG` column is the resolved package
source, `submissions/<dir>/source`, `…` = `simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4`). The four LSK-family flags
(`LSK LSKR KD8 LSKR97`) now default to 1, so **every row below the target must switch all four off**: e.g. `KD7=0` alone
now `FATAL`s (exit 2, "LSKR=1 requires LSK=1 KD7=1") instead of building 40fd4907.

| target | exact retrain command (fresh clone, repo root) | `SRC_PKG` (DRY_RUN) | reference archive |
|---|---|---|---|
| **FZ-EIKC + BVR + LOWKE + KD6 + KD7 + LSK (ρ 0.97, REC reset) + KD8** (final; board 09-27 82.452, 1st) | `bash scripts/retrain_all.sh` (bare), equivalently `BACKBONE=b5spx3 AXE=1 IPC=1 KD5=1 KD4=1 REC=1 IPW=1 KE3=1 CDIO=1 BVR=1 LOWKE=1 KD6=1 KD7=1 LSK=1 LSKR=1 KD8=1 LSKR97=1 bash scripts/retrain_all.sh` | `2026-10-23_…-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97` | `submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97/submission.zip`, sha256 `f11d2c1d0e00b6270ad71316117eb44e0de303b747bf9302ddcbfcdc7804504e` (X5V GO; 83 entries; its 4 weights are the 3090 files of fa948762 / eef3cfe4) |
| FZ-EIKC + BVR + LOWKE + KD6 + KD7 (board 09-26 82.419; **fallback**) | `LSK=0 LSKR=0 KD8=0 LSKR97=0 bash scripts/retrain_all.sh` | `2026-10-23_…-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke` | `submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke/submission.zip`, sha256 `fa94876280c2b8f82b93d0a6b66a66f9b1da3345db6a43369614ad9d345c08ce` (FR32, V32m GO) |
| FZ-EIKC + BVR + LOWKE + KD6 (no KD7) | `KD7=0 LSK=0 LSKR=0 KD8=0 LSKR97=0 bash scripts/retrain_all.sh` | `2026-10-22_…-kd6-rec-ipc-ipw-cdio-bvr-lowke` | `submissions/2026-10-22_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-rec-ipc-ipw-cdio-bvr-lowke/submission.zip`, sha256 `40fd4907d358a3d46df814ddad55a0cb831ee8e281112e3ff68d5fb8ad767784` (3090 chain build; FR31 GO, V31m GO: the chain rebuild reproduces all 80 entries) |
| same without BVR (and without KD7, which exists only on the BVR twin) | `BVR=0 KD7=0 LSK=0 LSKR=0 KD8=0 LSKR97=0 bash scripts/retrain_all.sh` | `2026-10-22_…-kd6-rec-ipc-ipw-cdio-tka-lowke` | KD6 + LOWKE twin `a6a98e7d…` |
| FZ-EIKC + BVR + LOWKE (no KD6) | `KD6=0 KD7=0 LSK=0 LSKR=0 KD8=0 LSKR97=0 bash scripts/retrain_all.sh` | `2026-10-22_…-rec-ipc-ipw-cdio-bvr-lowke` | `75a2a371cc97be5eeea6aafce27ef30886575763373e72d20799b08def72bb13` |
| FZ-EIKC (09-26 morning reference) | `BVR=0 LOWKE=0 KD6=0 KD7=0 LSK=0 LSKR=0 KD8=0 LSKR97=0 bash scripts/retrain_all.sh` | `2026-10-21_…-rec-ipc-ipw-cdio-s3-bundle-condwidth-v2-wholestep-t2-w01-kd` | `eef3cfe43cc7cb3c1c783a1220c0e4bfbb0018d40b9a0091dad9c6a644575276` (V26m GO) |

Intermediate LSK-family archives (never uploaded; each also DRY_RUN-checked): LSK-B at ρ 0.9 `a9acf984` =
`LSKR=0 KD8=0 LSKR97=0` (`…-bvr-lowke-lsk`); + REC reset `b519fe06` = `KD8=0 LSKR97=0` (`…-lsk-recr`); + KD8 `7b788569` =
`LSKR97=0` (`…-lsk-recr-kd8`); LSK-B + KD8 without the reset `259d0871` = `LSKR=0 LSKR97=0` (`…-lsk-kd8`); the KD7-less LSK
twin (no archive, 40fd4907 + LSK) = `KD7=0 LSKR=0 KD8=0 LSKR97=0` (`2026-10-23_…-kd6-rec-ipc-ipw-cdio-bvr-lowke-lsk`).
The optional `UV=1` (p := 0 in the outputs, `360a060b`) is **not** in the target: slower on the A100 (FR35 condition a).

**Defaults (since the 2026-09-26 freeze; LSK family added 2026-09-27):** under the default `BACKBONE=b5spx3` a bare
`bash scripts/retrain_all.sh` builds the final target above (verified: its DRY_RUN resolution equals the explicit-flag
command, line for line). `FREEZE_DEFAULTS=0` restores the pre-freeze defaults (bare = K3F-KD3, the 2026-09-24 board row
`f39128a9`). BVR, LOWKE, KD6, KD7, LSK, LSKR, KD8 and LSKR97 change only code (`submission.py`, `cuda_graph_wholestep.py`,
`kd6_conc.py`, `kd7_fuse.py`, `kd8_fuse.py`, `lsk_head.py`); they add no weights and no fit step (LSK computes its basis
from `model.pth` at load), so every row above shares every trained artefact.

## What the target is

The final target = **FZ-EIKC + BVR + LOWKE + KD6 + KD7 + LSK (config B, ρ 0.97) + REC reset of the LS moments + KD8**.
FZ-EIKC = **FZ-E** (the 2026-09-25 board lineage) **+ IPW + KE3 + CDIO**; BVR = by-value outputs in an out-of-process
worker, LOWKE = ke-gated fluctuation damping of the emitted prediction, KD6 = bit-exact multi-stream schedule of the
whole-step graph, KD7 / KD8 = bit-exact NVRTC-fused elementwise kernels and more side-stream work, LSK = a frozen separable
causal least-squares correction of the backbone output (per trajectory), LSKR = its moments reset when REC fires, LSKR97 =
its forgetting factor 0.97 (docs/METHOD.md §5.3, §6.3–§6.5). FZ-E = K3E-KD3 + IPC + KD5 + KD4 + REC, where
K3E-KD3 = a packed K=3 ensemble of three sim-pretrained, real-fine-tuned 4-member weight soups (pretrain seeds 0/1/2)
with a scoped fp16 core, plus the causal output-side stack CB + AW + AX + AXE(E4) + gain_max 20 + exact mask, and the
KD2/KD3 bit-exact kernel diets. Method sections in `docs/METHOD.md`:

| component | flag(s) | METHOD.md | kind |
|---|---|---|---|
| sim pretrain ×3, soups of 4, K=3 pack, fp16 core | `BACKBONE=b5spx3` (+ defaults `SOUP_K=4 FP16=1`) | §3.1–§3.4.2 | **weights** (steps 05a/06/07) |
| CB, AW, AX, gain_max 20, MSK | defaults under `b5spx3` (`CB CB_ADAPT AX GM20 MSK` = 1) | §5 items 5–9 | source constants |
| AXE (E4: window-0 CB fallback, extended energy-blend grid, adaptive blend ridge) | `AXE=1` | §5 item 10 | source constants |
| REC regime-change recovery | `REC=1` | §5 item 11 | source |
| KE3 dense 4-seed energy net | `KE3=1` | §5 item 1 / §5.1 | **weights** (step 08 refit) |
| KD2 / KD3 / KD5 / KD4 kernel diets | defaults `KD2 KD3` = 1; `KD5=1 KD4=1` | §6.1 | source (bit-exact; KD5 near-exact) |
| IPC one-storage return, IPW warm-up, CDIO CUDA-direct I/O, BVR by-value return | `IPC=1 IPW=1 CDIO=1 BVR=1` | §6.2 | source (value-identical transport levers) |
| LOWKE low-energy fluctuation damping | `LOWKE=1` | §5.2 | source |
| KD6 multi-stream whole-step graph, KD7 fused critical-path kernels | `KD6=1 KD7=1` | §6.3, §6.4 | source (bit-exact) |
| LSK separable LS head, config B (r 8, λ_rel 30), forgetting factor ρ 0.97 | `LSK=1 LSKR97=1` | §5.3 | source (basis from `model.pth` at load; no fit) |
| REC reset of the LS moments | `LSKR=1` | §5.3 | source (bit-identical unless REC fires) |
| KD8 more side-stream work + 6 fused kernels | `KD8=1` | §6.5 | source (bit-exact) |

All flags except `BACKBONE`/`SOUP_K` and `KE3` only select which **tracked, weight-free package source** step 10
packages (`SRC_PKG`); their grids / thresholds / constants are literals in that source's `submission.py` `CONFIG`,
chosen offline on LCO folds and never refit by the chain.

## Verify the resolved configuration first (`DRY_RUN=1`)

```bash
DRY_RUN=1 bash scripts/retrain_all.sh          # bare = the final target
```

prints every resolved parameter and the step table and runs nothing (it only greps the tracked sources' CONFIG keys,
then exits before the step-00 sanity checks; nothing is trained or written). Expected (final target, 2026-09-27):

- `SRC_PKG=…/submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97/source`
- `PKG_NAME=retrained_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97-kes4-s3-bundle-condwidth-v2-wholestep-t2-w01-kd`
- `SHIP_DIR = X2_DIR = data/checkpoints/sp/ship_enrx3_soup4_kes4`, `W2_OUT=…/data/checkpoints/b1_3090/ship_condwidth_v2`
- `X3=1 (BACKBONE=b5spx3) FP16=1 AXE=1 KD2=1 KD3=1 IPC=1 KD5=1 KD4=1 REC=1 IPW=1 KE3=1 CDIO=1 BVR=1 LOWKE=1 KD6=1 KD7=1 LSK=1 LSKR=1 KD8=1 LSKR97=1 UV=0 … ARCH_EXPECT=b5x3_h192_blk6_rf11`
- (the `BACKBONE=b5spx2` in the first line is expected: `b5spx3` runs the `b5spx2` steps with `X3=1`.)

The same command with the explicit flags of the table above prints the same lines. For the fallbacks, only `SRC_PKG` /
`PKG_NAME` and the flag echo change (table above); `SHIP_DIR` / `W2_OUT` never do.

## Prerequisites

- **Environment**: `python3.10 -m venv .venv` (or `uv venv --python 3.10 .venv`) and
  `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121` → Python 3.10, `torch==2.2.2`
  (+cu121), `numpy==1.26.4` — the eval image's versions (Python 3.10, PyTorch 2.2.2, numpy 1.26). The chain uses
  `.venv/bin/python` and sets `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` itself.
- **Data** (`data/README.md`): `data/competition/train_real/` (82 released files) and `data/competition/train_sim/`
  (all **100** files — step 00 `FATAL`s otherwise; used by step 05a only). **`train_real/7575_0.h5` is corrupt and is
  excluded at every stage** (`scripts/split_data.py`, every LCO split JSON, `lco_make_splits.py`, `c11a_fit_ke_flow.py`,
  and `materialise_all81`, which `FATAL`s unless exactly 81 files remain). No other input is read; nothing from
  evaluation data. `example_data/` is used only by optional `local_eval` checks.
- **GPU**: one CUDA GPU with **≥ 8 GB free** (R1p's two OOM aborts were caused by another user's job, not by the chain).
  Disk: the fold stream dumps take ≈ 1.3–4 GB (`KEEP_DUMPS=0` deletes them after scoring).
- **GPU memory assumptions (R3; documented, not changed)**: every process caps itself with
  `torch.cuda.set_per_process_memory_fraction(f)`, and `f` is a fraction of the **card's total memory** (of the MIG slice on
  a MIG instance), not of free memory. The caps shrink on smaller cards and an allocation above the cap is an OOM, even
  when the card has room.

  | step | processes at once (default) | `--mem-frac` each |
  |---|---|---|
  | 03 fold backbones | 3 | 0.15 |
  | 04 fold prior, then ke net | 3 | 0.15, then 0.2 |
  | 05 fold streams | 1 | 0.3 |
  | 05a sim pretrains, 06 fine-tunes, 08 fits | 1 | 0.15 |
  | 05b K=4 pool (train / fits / stream per group) | 2 (`KFOLD_JOBS`) | 0.15 |
  | 09 W2 fit | 1 | 0.25 |

  Measured peaks are **1.4–1.7 GiB allocated** per training or fit process (`*_resource.json` `peak_mem_gib`: 1.71 for a
  fold backbone, 1.67 for a b5 fine-tune; 1.39–1.40 GiB for the step 05b / 08 fits). At the 0.15 cap that needs a card of
  at least ≈ 12 GB in principle. **Minimum card: 16 GB** (cap 2.4 GB; expected to work, not rehearsed). **Tested: 24 GB**
  (RTX 3090 / 4090, cap 3.6 GB) and **A100-SXM4-40GB** (R1a100 / R1img, below). 80 GB A100 / H100 cards have more headroom. A MIG slice needs ≥ 16 GB for the same
  reason. At most 3 chain processes share the card, and their caps sum to ≤ 0.6 of it; in use that is ≈ 3 × 2 GB.
- **`SERIAL=1` (R2, opt-in; default 0)**: steps 03 / 04 / 05b normally start several processes on the one GPU (table
  above). A GPU in **EXCLUSIVE_PROCESS** compute mode, common on HPC clusters, admits one CUDA context at a time, so the
  second process fails. `SERIAL=1 bash scripts/retrain_all.sh` runs those processes one after another. The commands, args
  and seeds are the same; only the scheduling changes. It adds ≈ 40 min on a 3090 (step 04 measured: 12 m 46 s serial vs 5 m 55 s concurrent; 03 ≈ +17 min and 05b ≈ +15 min estimated from their per-process times). It also overrides `KFOLD_JOBS` to 1.
  Every other step already runs one GPU process at a time. `DRY_RUN` output is identical with and without it.
  Evidence: `experiments/rch/notes.md`.
- **Resume safety (R4)**: a step is skipped only if its outputs exist **and** every reused `.pt` / `.pth` / `.npz` /
  `.zip` reads back. The check reads the zip central directory and the CRC of every member; a non-zip `.pt` / `.pth` is
  `torch.load`-ed on the CPU. A file truncated by a job killed mid-save fails the check. It is renamed to `<file>.corrupt`
  and the step is redone. The K=4 pool helper does the same for its group checkpoints and feature caches. The ke-flow
  feature caches are also wiped before each fit, so a fit never resumes from a killed run's partial cache.
  `VERIFY_REUSE=0` restores the existence-only check. A fresh run is unaffected: the check only runs on files that
  already exist, and it never rewrites them. Do not run two chains on one tree at the same time.
- **Wall time** (zero reuse, fresh clone): **≈ 3 h on an RTX 3090** (R1p, K3F-KD3 flags: 2 h 47 m of step time on a
  shared 3090; KE3 adds ≈ 5–10 min to step 08) and **≈ 1 h on an RTX 4090** (R1r, FZ-E flags: 1 h 02 m; KE3's step 07/08/10
  pass ≈ 6 min) and **≈ 2 h 15 min on an A100-SXM4-40GB** (R1a100, bare command = the fa948762 lineage, zero reuse:
  steps ≈ 1.7–2× slower than on the 4090; the same inside the official eval image, R1img: ≈ 2 h 16 min). The LSK-family
  flags add nothing to this (code only; step 10 takes seconds). Longest steps: 06 (12 fine-tunes + 3 soups), 09 (W2 fit),
  05b (K=4 pool), 05a (3 pretrains).

## Steps (13, idempotent — each skipped if its outputs exist, unless `--force`; seed 0 throughout)

| step | what | detail for the target |
|---|---|---|
| 01 split | `data/real/{train,val,test}` 73/4/4 | seed 0, 7575_0 excluded |
| 02 lco | `data/real_lco/{fold0,fold1,fold2,foldR}` | symlink folds, verified vs split JSONs (window counts, no leak, no 7575_0) |
| 03 foldtrain | 3 LCO `n1_simvp` backbones | **reference check only** (not in the archive) |
| 04 foldfit | per-fold `prior.npz` + `ke_net_flow.pt` | reference check only |
| 05 foldstream | 3 held-out streams + official scoring | **parity check**: composites vs 81.868 / 82.353 / 82.365 (time fixed 88), band ± 0.02 |
| 05a simpretrain | 3 sim pretrains (seeds 0/1/2), 100 `train_sim` files | `sp_pretrain_sim.py --iters 20000 --snapshots 10000 --stop-after 10000 --seed s` |
| 05b kfold | K=4 all-81 out-of-sample pool (random-init b5) | feeds step 09 only; shared by the whole b5 family |
| 06 shiptrain | 3 × (4 all-81 fine-tunes, seeds 0–3 → uniform soup) | `sp_finetune.py --init <05a snapshot> -- --model b5_h192_blk6_rf11 --loss s3 --train data/real_all81 --seed s`; `c2_make_soup.py` |
| 07 shippack | `k3_pack_b5xk.py` → `ship_enrx3_soup4_kes4/model.pth` | arch `b5x3_h192_blk6_rf11` |
| 08 shipfit | `en_fit_prior.py` → `prior.npz`; `en_fit_ke_flow.py` → `ke_net_flow_single.pt`; **KE3**: `kes_fit_ensemble.py --seeds 0,1,2,3 --epochs 120` on the same cache → `c2_ke_dense.py --check` → `ke_net_flow.pt` (one dense KeFlowNet, width 96 / wide 128, = the exact 4-seed mean; fp64 check ≤ 1e-8) | all 81 files |
| 09 condwidth | W2 v2 fit, best of `W2_SEEDS="0 1 2"` → `cond_width_weights.pt` | alpha/lambda expected (1.0, 1.05) |
| 10 package | tracked source + 4 weight files → `pack_zip.py` | asserts: `submission.py` byte-identical to `SRC_PKG`, arch key, and per flag the CONFIG key + module (fp16_core, br_adapt, kd2/kd3/kd5/kd4, rec_detect thr 0.405465, ipc_one_storage auto, cuda_direct_io + ipc_prewarm, ipc_byvalue, tka_lowke, kd6_conc.py, kd7_fuse.py, `lsk_head` r 8 / λ 30 with ρ 0.97 under `LSKR97=1` (0.9 otherwise) + lsk_head.py, lsk_rec_reset + both REC-reset hooks, kd8_fuse.py + the KD8 side_c reset hook; each flag-off also asserts absence) |
| 11 localeval | `local_eval.py` on `data/real/{val,test}` | sanity only (memorised splits, not leaderboard-comparable) |

Useful switches: `--only 07,08,10`, `--from 05`, `--force`, `--list`, `PKG_NAME=…`, `KEEP_DUMPS=0`. Step 10 skips when
`submissions/$PKG_NAME/submission.zip` exists; every flag above (BVR, CDIO, IPW, KE3, LOWKE, KD6, KD7, LSK, LSKR, KD8,
LSKR97) changes `PKG_NAME`, so switching the target never reuses the other target's archive. **Switching an existing build
between the final target and any fallback row** (e.g. to fa948762) is `<that row's flags> bash scripts/retrain_all.sh --only 10`
— seconds, all weights reused. (KE3 note: a clone that holds an `…_kes4` dir from an older KE3 recipe must delete it
first; a fresh clone is unaffected.)

### Flag constraints (violations `FATAL exit 2`, never a silent mis-package)

| rule | source of truth |
|---|---|
| under `b5spx3`: `CB=1 CB_ADAPT=1 AX=1 GM20=1 MSK=1 FP16=1 B2=0 ENRX_KE=0` are fixed (`AXE` free: 0 or 1) | only these sources are tracked |
| `KD3=1` requires `KD2=1`; `IPC`, `KD5`, `KD4`, `REC` each require `KD3=1` | |
| with `KD4=1` or `REC=1`, (AXE KD5 KD4 REC IPC) must be one of `11111` (FZ-E), `01111` (FZ-F), `00100`, `x0010` | FZ block |
| `IPW=1` only on `11111` / `01111`; `CDIO=1` requires `IPW=1` and `11111` (no FZ-F CDIO source) | |
| `BVR=1` requires `IPW=1 AXE=1` (with `CDIO=1` → the `2026-10-22_c2-…-cdio-bvr-…` source; with `CDIO=0` → `2026-10-21_c2-…-ipw-bvr-…`) | |
| `LOWKE=1` and `KD6=1` each require `CDIO=1`; `KD7=1` requires `KD6=1 BVR=1 LOWKE=1` | |
| `LSK=1` requires `KD6=1 BVR=1 LOWKE=1`; `LSKR=1` and `KD8=1` each require `LSK=1 KD7=1`; `LSKR97=1` requires `LSK=1 LSKR=1 KD8=1` (hence: switching KD7 / KD6 / BVR / LOWKE off needs `LSK=0 LSKR=0 KD8=0 LSKR97=0`) | FREEZE block |
| `UV=1` (optional, not in the target) requires `LSK=1 LSKR=1 KD8=1` | |
| `KE3`, `IPW`, `CDIO`, `BVR`, `LOWKE`, `FP16`, `KD*`, `IPC`, `REC`, `LSK*` outside `BACKBONE=b5spx3` | FATAL |
| any flag value other than `0`/`1` | FATAL |

Each source switch also greps the selected `SRC_PKG/submission.py` for its CONFIG key (`"ipc_prewarm": True`,
`"cuda_direct_io": True`, `"ipc_byvalue": True`, …) before anything runs.

## Expected outputs and parity band

`submissions/$PKG_NAME/{submission.zip, SHA256SUMS, retrain_report.txt}`; logs under `logs/`
(`retrain_timings.txt` per step). The final target's zip has **83 entries: 79 code entries + 4 weight files** (fa948762:
81 = 77 + 4; eef3cfe4: 79 = 75 + 4).

| entry | same GPU architecture as the reference | different GPU |
|---|---|---|
| 79 code entries | **byte-identical** to the tracked source (step 10 asserts `submission.py`; every rehearsal found every code entry identical) | identical |
| `model.pth` (3 soups packed) | **byte-identical** (3090: `2716f119…`, the same file in f39128a9, e71e0cbf, eef3cfe4, fa948762, f11d2c1d) | differs (weights of a different GPU; same quality) |
| `prior.npz` | **byte-identical** (3090: `5bbbf246…`) | differs |
| `ke_net_flow.pt` (KE3 dense) | refit — **not bit-deterministic even on the same GPU** (R1q); FZ-EIKC / f11d2c1d 3090: `c12317e5…` | differs |
| `cond_width_weights.pt` (W2) | refit — not bit-deterministic even on the same GPU; FZ-EIKC / f11d2c1d 3090: `3559adac…` | differs |

The pipeline (sim pretrain → fine-tunes → soups → pack → prior) is weight-bit-deterministic on an uncontended GPU of
one architecture (R1o: rebuilt seed-2 pretrain tensors bit-identical, `cmp`-identical `model.pth`); the two refit files
vary slightly run to run. **Score parity, not byte parity, is the correctness bar:**

| check | band | evidence |
|---|---|---|
| step 05 LCO composites (fold0 / fold1 / foldR) | **± 0.02** vs 81.868 / 82.353 / 82.365 | FZT 3090: 81.872 / 82.354 / 82.356; R1p 3090: 81.873 / 82.359 / 82.355 |
| step 09 W2 selection | alpha 1.0, lambda 1.05 | every rehearsal |
| `local_eval.py` (unedited) subscores, rebuilt vs reference zip, same device | ± 0.3 each (tke is the noisiest: ke refit) | R1o 8/8 PASS; **f11d2c1d reference** (X5V, 3090, GPU): example_data 82.412 / 79.694 / 90.880 / sps 13.823, shim_test 96.173 / 82.173 / 97.501 / 52.233; FZ-EIKC reference: example_data 82.246 / 79.756 / 90.843 / 13.656, shim_test 96.101 / 82.358 / 97.482 / 52.251 |
| leaderboard, rebuilt weights | ± 0.02 per accuracy subscore | 09-25: the 4090 zero-reuse retrain scored within ± 0.02 of the 3090 lineage (below); A100 zero-reuse retrain within ± 0.03 of the 4090 build on local_eval (R1a100) |

`local_eval` compares like device to like device: the fp16 core is GPU-only (CPU falls back to fp32), so GPU-vs-CPU
subscores differ by up to ≈ 0.3 tke by design.

## Board rows ↔ chain flags

| board row | archive (sha256 prefix) | weights | final | chain command |
|---|---|---|---|---|
| 2026-09-23 SP-b5 | `74f9d98f` | 3090 | 81.999 (3rd) | `BACKBONE=b5sp` (`SOUP_K=1`; source `2026-10-06_…`) |
| 2026-09-24 K3F-KD3 | `f39128a9` | 3090 | 82.308 (1st) | `BACKBONE=b5spx3` (all defaults: `SOUP_K=4 FP16=1 CB CB_ADAPT AX GM20 MSK KD2 KD3`=1, `AXE=0`) |
| 2026-09-25 FZ-E | `53f7865c` | **4090**, R1r zero-reuse retrain | 82.311 (2nd) | `BACKBONE=b5spx3 AXE=1 IPC=1 KD5=1 KD4=1 REC=1` (3090 twins: `e71e0cbf`, FZT rebuild `5482b250`) |
| 2026-09-26 FZ-EIKC + BVR + LOWKE + KD6 + KD7 | `fa948762` | 3090 (FZT / V32m chain build) | 82.419 (2nd; time 92.42) | `LSK=0 LSKR=0 KD8=0 LSKR97=0` (= the FZ-EIKC flags + `BVR=1 LOWKE=1 KD6=1 KD7=1`) |
| **2026-09-27 final: + LSK (ρ 0.97, REC reset) + KD8** | **`f11d2c1d`** | 3090 (the same 4 weight files as fa948762) | **82.452 (1st at 11:28 UTC; time 92.46)** | **bare** `bash scripts/retrain_all.sh` (= the 09-26 flags + `LSK=1 LSKR=1 KD8=1 LSKR97=1`) |

The 09-27 row is the development phase's last upload. Δ vs 09-26: rel_l2 +0.030, tke +0.008, mvpe +0.105, time +0.047,
sps +0.014, final +0.033 (pre-registered 82.455, band 82.40–82.51). A non-zero row with time 92.46 shows every path
(KD6/KD7/KD8 NVRTC kernels, LSK at ρ 0.97, BVR, LOWKE, CDIO/IPW) active in the eval environment; by the pre-registered
FR33/FR35 rule (time within ± 0.5 of 92.42, final ≥ 82.37) the f11d2c1d lineage is the final package (SUBMISSIONS.md
"RESULT 2026-09-27").

Intermediate archives map the same way: `IPW=1` alone on FZ-E → `2026-10-20_c2-…-ipw-…` (4090 `d3f5e0ec`), `+ KE3=1` →
`19f2f1a0` (4090) / `2a2bbb09` (3090 FZK); `KE3=1` without IPW → `1aedab25` (4090); FZ-EIKC `eef3cfe4` (3090) /
`3541e426` (4090) and FZ-EIKC + BVR `c7671458` (3090) were 09-26 candidates; the LSK-family intermediates are listed
under the target table. FZ-F = the same flags with `AXE=0`.

## Cross-GPU reproducibility evidence

- **FZT (3090, zero reuse, fresh clone, `experiments/fzt_twin/summary.txt`)**: full 01→11 rebuild of FZ-E / FZ-F
  reproduces `e71e0cbf` / `31e1891d` entry for entry except `cond_width_weights.pt` and `ke_net_flow.pt`; `model.pth` and
  `prior.npz` byte-identical; LCO 81.872 / 82.354 / 82.356. FZK (`fzk_summary.txt`) then ran `IPW=1 KE3=1 --only 07,08,10`
  on those artefacts → `2a2bbb09` (differs from the FZ-E twin in `ke_net_flow.pt` + `submission.py` only), and CDP ran
  `CDIO=1 --only 10` → **`eef3cfe4`** (differs from `2a2bbb09` in `submission.py` + `cuda_graph_wholestep.py` only).
- **R1p (3090, zero reuse, K3F-KD3 flags, `experiments/r1p_rehearsal/notes.md`)**: 2 h 47 m; `model.pth` + `prior.npz`
  byte-identical to `f39128a9`; all code entries identical; LCO within ± 0.01.
- **R1r (4090, zero reuse, FZ-E flags, `experiments/c2_soup/r1r/notes.md`)**: EXIT 0 in 1 h 02 m → `53f7865c`; 75/75
  code entries = the tracked FZ-E source; `model.pth` + `prior.npz` byte-identical across three 4090 clones (R1k/R1q/R1r);
  KE3 via `--only 07,08,10` EXIT 0. 4090 vs 3090 K=3 fold pipes: fold0 −0.010, fold1 −0.004 (`experiments/c2_k3var`).
- **Board 2026-09-25 = a 4090 retrain scored on prediction**: `53f7865c` (4090 weights, FZ-E code) scored 82.311 vs the
  pre-registered 82.31 (FR27 outcome B); Δ vs the 3090 09-24 row: rel_l2 +0.012, tke −0.019, mvpe +0.001, sps +0.021 —
  the retrain variance the organizers' rebuild should expect is ≈ ± 0.02 composite.
- **KE3 reduces retrain variance**: the single ke fit alone gives sd ≈ 0.011 on fold0 (three seeds on identical members);
  the 4-seed dense mean roughly halves that component (`experiments/c2_ke3/notes.md`).
- **V26m** (`experiments/v26_verify/notes.md`, exact zip `eef3cfe4`): integrity, flags-off ≡ parents, CPU- and CUDA-input
  identity, KD3/KD4/KD5/REC live, IPW fires in a spawned worker, local_eval / fuzz / reset / RNG / file-open audit — GO.
- **R1a100 (A100-SXM4-40GB, zero reuse, fresh clone, bare command at 278ee73 = the fa948762 lineage,
  `experiments/c2_r1a100/notes.md`)**: EXIT 0 in ≈ 2 h 15 min; 77/77 code entries == the tracked fa948762 source; weights
  differ from the 4090 build (cross-GPU, as expected); acceptance rc 0; unedited local_eval (shim_test / shim_val) within
  **± 0.03 per subscore** of the 4090 build.
- **R1img (the same retrain INSIDE the official eval image, apptainer on the A100, from the release tarball, bare
  command, the image's own Python 3.10.14 / conda torch 2.2.2 / numpy 1.26.4, no venv)**: EXIT 0 in ≈ 2 h 16 min; 77/77
  code == the fa948762 source; **`model.pth` and `prior.npz` byte-identical to R1a100** (same GPU type, different torch
  builds); local_eval within ± 0.02 of R1a100.
- **Final flags inside the image** (on the R1img weights, `--only 10`): `LSK=1 LSKR=1` → b519fe06 code (38 .py identical),
  `LSK=1 LSKR=1 KD8=1` → 7b788569 code, and `LSK=1 LSKR=1 KD8=1 LSKR97=1` → **f11d2c1d code: EXIT 0, all 39 .py == the
  `-r97` source, weights == the in-image 7b788569 build; KD6 self-test bit_exact, W01 bit_exact, KD7 / KD8 on (NVRTC in
  the image), LSK active at ρ 0.97 (`lsk_error` None)**; local_eval ms/step ≡ 7b788569 (c2, `docs/COLLAB_LOG.md`
  2026-09-26 20:44 UTC; `experiments/c2_r97/notes.md`, `experiments/c2_a100/kd8r/notes.md`).
- **X5V** (`experiments/x5v/notes.md`, exact zip `f11d2c1d`): the chain step 10 (`LSKR97=1`) reproduces every code entry;
  status, kdr identity (windows 0/1 ≡ 7b788569; KD8/KD6 failure fallbacks ≡ new), CUDA-input, spawn/fork children, 11
  out-of-process harness modes, fuzz, reset, RNG, memory (2,000 steps), degenerate inputs, race checkers, 5-run
  determinism — GO. **c2 R97** replicated ρ 0.97 vs 0.9 on independently refit ke / W2 fold weights (5/5 sets ≥ 0).

## Known nondeterminism

Seeds are fixed everywhere, but (i) the step-08 energy-net fit and the step-09 W2 fit are not bit-deterministic even on
one GPU (R1q), (ii) any GPU architecture change alters every trained weight, and (iii) heavy contention can change cuDNN
algorithm choice (R1g's differing seed-0 pretrain on a contended 3090). None of this has ever moved a parity check out
of its band. The organizers should compare scores, not hashes; the hashes above tell them which files are expected to
match when they rebuild on the same GPU model we used (3090 for `eef3cfe4`, `fa948762` and `f11d2c1d`, which share
their 4 weight files). LSK adds no weight file and no fit: its SVD basis is computed from `model.pth` at load, in fp64.

## Historical lineages

Every earlier lineage stays buildable by the same script (`docs/METHOD.md` §3.5): `BACKBONE=b5spx2` (K=2; `CB`/`CB_ADAPT`/
`AX`/`GM20`/`MSK` = 0 reproduce `894f044e` → `f725611e` → `e67ff687` → `43d73243` → `51018124` → `8ef2112a`), `BACKBONE=b5sp
SOUP_K=4` (`2b7dfe3c`, CB → `…-cb-…`), `BACKBONE=b5sp` (`74f9d98f`), `BACKBONE=b5`, `BACKBONE=n1`. Their fresh-clone
rehearsals (R1/R1e `n1`, R1f `b5`, R1g/R1h `b5sp`, R1j/R1k/V15 `b5spx2`, R1o `b5spx3` seed-2 steps) all passed the same
bands (LCO ± 0.02; local_eval ± 0.3; code entries byte-identical); details in their `experiments/r1*_rehearsal/` notes and
in this file's git history (`git log -p docs/REPRODUCE.md`). `docs/REPRODUCE_v8v9_2026-09-21.md` keeps the v8/v9 document.
