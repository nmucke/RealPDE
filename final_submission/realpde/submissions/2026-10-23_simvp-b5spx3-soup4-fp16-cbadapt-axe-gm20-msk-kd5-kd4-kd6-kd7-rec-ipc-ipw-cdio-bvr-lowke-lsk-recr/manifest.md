# 2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr

**FZ-EIKC + BVR + LOWKE + KD6 + KD7 + LSK-B + REC-reset (LSKR)** = a9acf984 (LSK config B, ρ 0.9, λ_rel 30) with exactly the
two changes from LSX (experiments/lsx/notes.md):
1. When REC fires (regime-change reset, `rec.keep` = 0), the LS state is reset too: `_state.mul_(rec.keep)` and `Mh.mul_(rec.keep)`
   (LSX's implementation, `LSKHead.rec_reset_`). The multiplier is exactly 1.0 when REC does not fire, so those steps are
   bit-identical to a9acf984. It is wired at LSX's two sites: `kd6_conc.py` after `join_c` and before `apply_`, and the serial
   whole-step body in `cuda_graph_wholestep.py` before `step_serial`. It is controlled by CONFIG `"lsk_rec_reset": True`
   (False gives a9acf984 bit for bit).
2. The latent bug fix: `_lsk_build`'s fp64 maths self-test uses `lsk.rho` instead of a hard-coded 0.9. This is inert at ρ 0.9.
   Before the fix, any other ρ failed the self-test and silently switched LS off.

ρ 0.9 and λ_rel 30 are unchanged. Code only: no new weights or fits. Built and verified by LSR (experiments/lsr/notes.md).

## submission.zip
- sha256 `b519fe06d39e2f3d12d1e644aa331411f06b44a0906d7c2faa8f6f6e763fff8d`, 82 entries, 60 100 780 B extracted.
- Entries vs a9acf984: only these 4 .py files change:
  - `submission.py` 5777936c → 51f9f361;
  - `lsk_head.py` d03a8ccb → b1b340dd;
  - `kd6_conc.py` 9c4ed54f → 0710f5d2;
  - `cuda_graph_wholestep.py` c6bf7a65 → 4ddcd8ab.
  All 78 other entries are byte-identical, including the 4 weights (model.pth 2716f119, prior.npz 5bbbf246, cond_width_weights.pt
  3559adac, ke_net_flow.pt c12317e5).
- source/: `experiments/lsr/lsr_make_source.py` applied to a9acf984's source. The diff is 9 hunks over the 4 files above (lsk_head.py 4, submission.py 3, kd6_conc.py 1, cuda_graph_wholestep.py 1).
- Chain: `LSK=1 LSKR=1` on the FREEZE defaults (KD7=1), with experiments/lsr/0001-retrain_all-LSKR.patch.
  - Step 10 in the FZT clone reproduces every code entry, plus model.pth, prior.npz and cond_width_weights.pt.
  - ke_net_flow.pt differs there (efdc53cb). This is the same pre-existing FZB_DIFF seen for a9acf984.
  - This zip is that step-10 build tree with a9acf984's ke_net_flow.pt bytes, packed by scripts/pack_zip.py.

## Verification (experiments/lsr/notes.md)
- **Bit-identity to a9acf984 where REC does not fire:**
  - kdr fold0 (811 windows) and short (117) are bit-identical, with CPU and CUDA inputs.
  - On adv, only the 4 windows from the first REC fire to the end of `syn:file_switch_mid_traj` differ.
  - REC fire counts: fold0 0, short 0, adv 2.
  - Flag off (`lsk_rec_reset` False) is bit-identical to a9acf984 on all 3 streams.
  - An injected LSK self-test failure (early or late) is bit-identical to fa948762 (LSK off).
  - An injected KD6 failure (the serial whole-step path) is bit-identical to this zip, including on the REC switch surrogates.
- **REC switch set sw0 (21 files, fold0 weights, LSX's set):**
  - composite 80.8798 vs a9acf984's 80.8440, a gain of **+0.0358**;
  - equal to LSX's X1 bit for bit (every per-window array);
  - in each file, outputs differ only from the REC fire window to the end, and all are finite.
- Harness and robustness checks: the out-of-process harness gives one sha per mode; fuzz, reset, RNG, REC write-back reset, open
  audit, modpath, 2,000-step memory, degenerate inputs and 5-run determinism all pass.
- 3090 timing vs a9acf984 (6 interleaved rounds, QUIET_GPU): in-process ≈ +0.015 ms/step (+2 kernels, 717 vs 715), time_score ≈ −0.014.

Status: NOT uploaded.
