# 2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97

**7b788569 (MRG GO: LSK-B + LSKR REC reset + KD8) with ONE constant changed: the LSK forgetting factor rho 0.9 -> 0.97**
(= LSX's X5R, experiments/lsx/notes.md). Built and verified by X5V (experiments/x5v/notes.md). Code only: no new weights or fits.

## What it is
- source/ = 7b788569's source; `diff -r` vs it is exactly line 395 of submission.py:
  `"lsk_head": {"r": 8, "rho": 0.97, "lam": 30.0},` (was 0.9; the comment on that line was updated too).
- The self-test's rho fix (fp64 reference uses `lsk.rho`) and the REC reset (main + KD8 side_c sites) are already in 7b788569;
  at rho 0.97 the self-test passes (lsk_error None) and the LS head is active.

## submission.zip
- sha256 **f11d2c1d0e00b6270ad71316117eb44e0de303b747bf9302ddcbfcdc7804504e**, 83 entries, 60 130 938 B extracted.
- vs 7b788569: only submission.py differs. Weights byte-identical: model.pth 2716f119, prior.npz 5bbbf246,
  cond_width_weights.pt 3559adac, ke_net_flow.pt c12317e5.
- Chain: `LSK=1 LSKR=1 KD8=1 LSKR97=1` with experiments/mrg/0001-retrain_all-KD8R.patch (in HEAD) +
  experiments/x5v/0001-retrain_all-R97.patch.
  - Step 10 in the FZT clone reproduces every code entry; ke_net_flow.pt differs (efdc53cb), the known FZB_DIFF.
  - This zip is that build tree with 7b788569's ke_net_flow.pt bytes, packed by scripts/pack_zip.py.

Status: NOT uploaded.
