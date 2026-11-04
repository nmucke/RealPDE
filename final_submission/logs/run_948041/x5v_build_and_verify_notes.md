X5V GO: candidate submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97/submission.zip
sha256 f11d2c1d0e00b6270ad71316117eb44e0de303b747bf9302ddcbfcdc7804504e (= 7b788569 with lsk_head rho 0.9 -> 0.97, i.e. LSX's X5R; weights byte-identical).
Through LSX's pipes it reproduces X5/X5R exactly (fold1 83.1228, fold0 82.8420 = LSK-B + 0.0099). The full battery passes. Timing vs 7b788569 is 0.000 ms/step (median).
NOT uploaded.

# X5V: 7b788569 + LSK rho 0.97 (X5R) (Opus X5V, 2026-09-26, times UTC)

## 1. Source: submissions/…-lowke-lsk-recr-kd8-r97/source
- A copy of 7b788569's source (`…-lowke-lsk-recr-kd8/source`). `diff -r` is exactly one line, submission.py:395:
  `"lsk_head": {"r": 8, "rho": 0.97, "lam": 30.0},   # X5R (LSX): config B with rho 0.9 -> 0.97 (+ lsk_rec_reset below)`.
  The comment on the same line was updated too; no other byte changed.
- Other places that hold rho were checked:
  - lsk_head.py uses `self._rho_t` (a tensor built from `rho`);
  - kd6_conc / kd7_fuse / kd8_fuse / cuda_graph_wholestep have no 0.9 constant and no rho;
  - the self-test's fp64 reference already uses `lsk.rho` (LSR's fix, which is in 7b788569).

## 2. Chain: experiments/x5v/0001-retrain_all-R97.patch
- Written against HEAD dd6ad3d, which already has KD8R. `patch --dry-run -p0` is OK and `bash -n` is OK.
- New LSKR97 block, placed after the KD8 block:
  - LSKR97 ∈ {0, 1}, default 0, not in FREEZE.
  - It requires LSK=1 LSKR=1 KD8=1 and selects the `-r97` source.
  - It checks the source files and config:
    - kd8/kd7/lsk_head files;
    - CONFIG kd8 / kd7 / kd6_concurrency / ipc_byvalue / tka_lowke;
    - `lsk_head rho 0.97` present and no `rho 0.9` line;
    - lsk_rec_reset True;
    - the rho fix (Phi and B lines);
    - `def rec_reset_(self, keep`, the whole-step hook, exactly 2 kd6_conc reset sites, and `rr8`.
  - PKG_NAME `-lowke-lsk-recr-kd8-` → `-lowke-lsk-recr-kd8-r97-`.
- Other changes:
  - a b5spx3-only FATAL;
  - `LSKR97=` in the echo line;
  - step 10: the assembled package's LSK check greps the selected lsk_head line (0.97 under LSKR97=1, else 0.9), plus an
    explicit LSKR97=1 check.
- **DRY_RUN on a mirror** (/export/scratch1/ntm/tmp/x5v/mirror: real copies of the 2026-10-2[23] source dirs, the rest symlinked):
  - These 16 flag sets are identical to HEAD's script except the echo line: bare, LSK=1, LSK=1 LSKR=0/1, LSK=1 LSKR=1 KD8=0,
    KD8=0, KD7=0, FREEZE_DEFAULTS=0, KD6=0 KD7=0, LSK=1 KD7=0, SERIAL=1 LSK=1, LOWKE=0 KD7=0, LSK=1 LSKR=1 KD8=1, LSK=1 KD8=1,
    LSKR97=0, LSK=1 LSKR=1 KD8=1 LSKR97=0.
  - `LSK=1 LSKR=1 KD8=1 LSKR97=1` → the `-r97` source / `…-lowke-lsk-recr-kd8-r97-kes4-…`, rc 0.
- **FATAL (rc 2) checks:**
  - bad flag values: LSKR97=1 alone, +LSK only, +LSK+LSKR, +LSK+KD8, LSKR97=2, LSKR97=yes, +KD7=0, +FREEZE_DEFAULTS=0, +KD6=0,
    +LOWKE=0, b5spx2 (with and without LSK);
  - broken source copies: rho 0.9, rho 0.95, both lsk_head lines, kd8 False, kd7 False, lsk_rec_reset False, Phi rho-fix
    reverted, B rho-fix reverted, rr8 disabled, one kd6 reset site removed, the whole-step hook removed, no kd8_fuse.py,
    no kd7_fuse.py, no lsk_head.py, `rec_reset_` renamed, tka_lowke removed, no `_kd8j`.
- The good source gives rc 0. The mirror source was restored (`diff -r` == repo). The repo's scripts/retrain_all.sh is untouched.

## 3. Build (FZT clone /export/scratch1/ntm/tmp/fzt/clone)
- `git pull --ff-only` 346b794 → dd6ad3d. The 4 stashes are kept.
- The `-r97` source was copied into the clone, untracked. `scripts/.x5v_retrain_all.sh` = HEAD + the patch (== the mirror's).
- `LSK=1 LSKR=1 KD8=1 LSKR97=1 bash scripts/.x5v_retrain_all.sh --only 10 --force` gave rc 0 in 7 s, arch
  b5x3_h192_blk6_rf11, 83 files.
  - vs 7b788569's zip, only 2 entries differ: submission.py (the rho line) and ke_net_flow.pt (efdc53cb, the known FZB_DIFF).
- **Candidate** = that build tree + 7b788569's ke_net_flow.pt bytes (c12317e5), packed by scripts/pack_zip.py → **f11d2c1d…**.
  - 83 entries, 60 130 938 B extracted; testzip and `unzip -t` are OK; no pycache.
  - Every .py entry == the source dir.
  - vs 7b788569: only submission.py differs, and all 4 weights are identical (2716f119 / 5bbbf246 / 3559adac / c12317e5).
- The clone-local files (script, source copy, output dir) were removed and the clone's logs/retrain_10_package.log was
  restored. `git status` of the clone is as before, apart from the pull.

## 4. LSX pipes (scripts/x5v_make_pipe.py + x5v_pipes.sh; tka_run.py, the same fold weights and data as LSX)
- Pipe = the `-r97` source, code unchanged, + LSX's 4 fold-weight symlinks + the same 2 pipe-only hooks lsx_make_source.py
  adds (PC_CFG, the [lsx] status line).

| run | LS status | rel / tke / mvpe / sps | comp | LSX reference |
|---|---|---|---|---|
| fold1, shipping config | built, error None, kd8 on, rec_reset on | 95.478 / 81.607 / 96.236 / 50.1327 | **83.1228** | X5 83.1228 (every subscore identical) |
| fold0, shipping config | — | 95.173 / 81.170 / 95.704 / 49.9521 | **82.8420** | X5R 82.8420 = LSK-B 82.8321 + 0.0099 (every subscore identical) |
| fold1, PC_CFG rho 0.9 (control) | — | — | **83.1184** | = LSK-B / 7b788569 83.1184 |

- Both composites match to 1e-4, and so do the subscores. The KD8 side_c path therefore gives LSX's X5R maths exactly.

## 5. Verification battery (clean room /export/scratch1/ntm/tmp/x5v/a/b/c/d/e; scripts/x5v_lanes.sh A/B/C/D/SW/RACE, x5v_lanes_ic.sh CH/B2)
**Arms:**

| arm | what it is |
|---|---|
| new | f11d2c1d |
| ref | 7b788569 |
| off | CONFIG kd8 False |
| kd8fail | kd8_fuse.build raises |
| kd6fail | kd6_conc.build raises (serial whole-step body) |
| nors | NEGCTRL: side_c REC reset disabled |
| nrace | NEGCTRL: reset on side_b, unordered |
| rhobug | NEGCTRL: self-test reference rho hard-coded 0.9 |

**Status** (mrg_status.py):
- new:
  - **LSK active, lsk_error None, rho 0.97**, cfg {r 8, rho 0.97, lam 30}, rec_reset True;
  - KD8 True, KD7 True, KD6 bit_exact;
  - the captured rec_reset_ is on side_c (2 serial + 1 KD6 capture site, as 7b788569);
  - **714 kernels/step (== ref 714)**; W01 bit_exact; REC on; LOWKE live.
- off / kd8fail: 717 k/step, LSK active at 0.97.
- kd6fail: 771 k/step, LSK active.
- nors: 712 k/step.
- **rhobug: LSK OFF**, "lsk self-test: maths mismatch 0.000102 / 0.00189", 659 k/step. The self-test does see rho, and the
  rho fix is what keeps LS on at 0.97.

**kdr in-process, CPU inputs** (v25_cmpwin; windows 0 / 1 / ≥2):
- off, kd8fail and kd6fail vs new: **0 mismatches** on fold0 811 / short 117 / adv 378, including the adv REC fires.
- **new vs ref: window 0 and window 1 mismatches = 0 on all three streams.**
  - Windows ≥ 2 differ: fold0 771/771, short 20/20, adv 305/307.
  - The 2 adv windows ≥ 2 that are equal are the non-finite NaN/Inf-pixel INFORMATIONAL probes.
  - Non-finite and unordered sets are equal (adv 5/5); 0 exceptions.
- nors vs new differs at exactly adv steps 319–322 (the REC fire and its trajectory), so the REC reset is live at rho 0.97.
- ref here vs MRG's recorded 7b788569 run: 0 / 0 / 0.

**Other inputs and runs:**
- CUDA inputs: new-cuda vs new-cpu 0 on all 3 streams; off-cuda vs new-cuda 0.
- cpureq vs new: 0. alloc-stress vs new: 0.
- REC fire counts (new == ref == nors): fold0 0, short 0, adv 2 (steps 319, 320).

**Children and out of process:**
- kdr spawn/fork × fold0/short/adv == in-process new, 0 mismatches (BVR wraps 21083 / 3039 / 9805); CUDA-input children 0/811.
  in_child.py was run from /export/scratch1/ntm/tmp/x5v/tools, because kdr's assert_modules rejects a repo-resident
  in_child (the first attempt, rc 99, was a harness artefact and was rerun).
- v29_oop {spawn, fork} × {np, pickle, tmp, cuda, np_npout} + fork Queue:
  - **one sha 3e5d793d098b for new** in all 11 runs, 168/168, 0 exc;
  - ref is one sha 280345073ebc, == MRG's.

**REC switch surrogates** (rec_eval switch, fold0_e_rec weights, 3 files):
- off, kd8fail and kd6fail == new in every field except timings;
- comp_post: new 84.01 / 77.96 / 83.47, ref 83.96 / 77.94 / 83.46 (== MRG), nors 83.61 / 77.93 / 83.34.
- At rho 0.97 the REC reset is worth more than at 0.9 (−0.40 without it on file 1).

**Tool suite:**
- unedited local_eval, accuracy subscores:
  - example GPU new 82.412 / 79.694 / 90.880 / 13.823 (== ref);
  - shim GPU new 96.173 / 82.173 / 97.501 / 52.233 vs ref 96.170 / 82.183 / 97.495 / 52.245;
  - the CPU-only runs complete.
- fuzz 396 steps, 0 violations (in-process, spawn, fork);
- v16 reset PASS (in-process, spawn, fork);
- p7 RNG ok at build and after the run (in-process + spawn child);
- rec_wbreset all_state_restored;
- modpath cuda / cpu PASS;
- open audit: 0 files outside the package (cuda / cpu); the child extras are the audit script + the $TMPDIR probe (known).
- **2,000-step memory**: allocated 301.9 MB flat, reserved 2206.0 MB flat (== ref), 0 non-finite, 49 resets.
- **Degenerate inputs** (lsr_degen.py, 184 steps):
  - new == off bit for bit;
  - new vs ref differs only at k ≥ 2, with identical finite flags (53 non-finite, as ref) and identical REC fires;
  - reset restores.
- **Determinism**, 5 fresh-process runs each: short 9fe5c9ab7d5995f2 ×5, adv 4e051646335457a0 ×5.

**Race checks:**
- experiments/kd8/scripts/kd8_race.py, the fixed transitive checker:
  - new keep / no-keep: **0 unordered conflicts**, 0/677 storages differ;
  - nrace: **25 conflicts**, as in MRG, so the checker sees a misplaced reset.
- FR34 capture-mode checker (mrg/scripts/fr34_race_cap_copy.py):
  - new: 0 conflicts, replay 0/677, n_hook_alias 0;
  - ref: 0 / 0 / 0.

## 6. Timing (3090; QUIET_GPU window 19:33–19:44; scripts/x5v_timing_window.sh both 7; out/timing.tsv)
7 interleaved rounds (N = f11d2c1d, O = 7b788569; order alternates):

| harness | N − O per round | summary |
|---|---|---|
| in-process unedited local_eval shim_test, ms/step (both ≈ 2.48) | −0.012 −0.006 −0.072 0 +0.024 0 0 | **median 0.000** |
| time_score | — | median N 94.487, O 94.485 |
| c2_harness np, in-worker w_mean | — | median +0.002 |
| c2_harness np, round trip | — | median −0.18, IPC noise |

- As expected, rho is a tensor constant, so the kernels are unchanged (714 k/step for both).
- The accuracy subscores and np shas are constant across rounds (N 3e5d793d098b, O 280345073ebc).

## 7. Residual risks
- The expected gain is LSX's held-out measurement, looked at once, vs LSK-B:
  - fold0 +0.0099, foldR +0.0046, foldS-lo +0.0075, foldS-hi +0.033, Rx +0.007, high-Re A +0.003 / +0.008 / +0.009,
    J +0.024, sw0 +0.045;
  - it is positive on every set but below LSX's own +0.015 fold0 gate;
  - the weak spot is real AoA-0 Re ≥ 26.7k files, −0.010…−0.016 each.
- The self-test now guards rho: if the reference ever reverted to 0.9, LS would silently turn off. The rhobug arm shows this
  happens, and the chain's grep of the rho-fix lines prevents it.
- The A100 is unmeasured, but the kernels are identical to 7b788569's (0 expected delta).
- No commits, uploads or pushes. starting_kit/, scripts/retrain_all.sh and the existing submission dirs were not edited.
- Scratch /export/scratch1/ntm/tmp/x5v was deleted.
