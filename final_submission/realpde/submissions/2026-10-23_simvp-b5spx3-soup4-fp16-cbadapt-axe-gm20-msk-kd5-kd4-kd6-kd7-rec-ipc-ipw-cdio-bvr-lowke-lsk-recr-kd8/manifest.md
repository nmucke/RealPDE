# 2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8

**b519fe06 (LSK-B + LSKR REC-reset hardening) + KD8 (bit-exact critical-path cuts)**, built and verified by MRG
(experiments/mrg/notes.md). Code only: no new weights or fits.

## What it is
- source/ = a 3-way merge (`git merge-file`, base = a9acf984's source, ours = the `-lsk-kd8` source, theirs = the
  `-lsk-recr` source). The merge was clean, and the hunks were diff-checked both ways: vs `-lsk-kd8` they are exactly LSR's 9
  hunks, and vs `-lsk-recr` they are exactly KD8's hunks. Two hand edits in `kd6_conc.py` then address FR34's findings
  (experiments/fr34_review/report.md):
  1. **REC reset on KD8's side_c path.** Under KD8 `"lskc"`, the LS apply moves into `_ctx_body` (the corr baddbmm runs on
     side_c). That skips the block holding LSR's main-stream reset, so a textual merge would silently fall back to
     a9acf984's behaviour whenever REC fires.
     - Fix: `_lsk_issue` issues `_lsk.rec_reset_(rec.keep)` on side_c right after `_lsk.update(y)`, guarded by `rr8`
       (rec_reset and KD8 and "lskc"). It is on the same stream as the update and before the hook GEMM and the corr
       baddbmm.
     - `rec.keep` is written on main by `rec.detect()` before the `fork_c` that precedes it.
     - Without "lskc" (KD8 off or failed), LSR's main-stream site is used unchanged.
  2. **`S.keep.append(h)` after the side_c hook.** side_c reads the backbone's `h` after `_packed_forward` returns.
     Keeping a reference stops the private pool from reusing that memory before `join_c`.
     - FR34's capture-mode checker reports hook_alias 3 on the KD8 zip and **0** here.
- vs b519fe06: 3 .py files change (submission.py, kd6_conc.py, cuda_graph_wholestep.py) and kd8_fuse.py is added. The
  other 79 entries are byte-identical, including all 4 weights.

## submission.zip
- sha256 **7b788569bb7153cfa9b36d4ac0f9ba7596a980d1c75ecdb1faf6645f1224a3d6**, 83 entries, 60 130 940 B extracted.
- Weights: model.pth 2716f119, prior.npz 5bbbf246, cond_width_weights.pt 3559adac, ke_net_flow.pt c12317e5.
- Chain: `LSK=1 LSKR=1 KD8=1` with experiments/mrg/0001-retrain_all-KD8R.patch.
  - Step 10 in the FZT clone reproduces every code entry. ke_net_flow.pt differs (efdc53cb), the known FZB_DIFF.
  - This zip is that build tree with a9acf984/b519fe06's ke_net_flow.pt bytes, packed by scripts/pack_zip.py.
- Outputs are bit-identical to b519fe06 on every stream checked, including REC fires (adv steps 319–322 and the REC switch
  surrogates).

Status: NOT uploaded.
