# 2026-10-22_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-rec-ipc-ipw-cdio-bvr-lowke  — (B) FZ-EIKC + BVR + LOWKE

LKB merge (experiments/lkb_build/notes.md): source = the BVR source
(submissions/2026-10-22_c2-…-ipw-cdio-bvr-…/source, incl. the FR29 F1 `__format__` fix of e316dea) + TKA's lowke CONFIG hunk
(`"tka_lowke": {"theta": 1.5, "delta": 0.25, "gmin": 0.0}`) + TKA's cuda_graph_wholestep.py (`_lowke_apply`).
diff -rq vs (A) (TKA's …-cdio-tka-lowke source): submission.py only, and those hunks are line-identical to FZ-EIKC -> BVR.
POST-HOC lever, pending FR30 + c2 KE3 check.

## submission.zip
- sha256 `75a2a371cc97be5eeea6aafce27ef30886575763373e72d20799b08def72bb13`, 79 entries.
- Provenance: FZT clone /export/scratch1/ntm/tmp/fzt/clone at b11c3b7 + this source dir copied clone-locally +
  experiments/lkb_build/0001-retrain_all-LOWKE.patch, then
  `BACKBONE=b5spx3 AXE=1 IPC=1 KD5=1 KD4=1 REC=1 IPW=1 KE3=1 CDIO=1 BVR=1 LOWKE=1 bash scripts/retrain_all.sh --only 10`
  -> submissions/retrained_simvp-…-ipw-cdio-bvr-lowke-kes4-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/.
- Content vs c7671458 (V29m's FZ-EIKC+BVR, same clone): identical entry list; bytes differ ONLY in cuda_graph_wholestep.py and
  submission.py (= exactly the 5-line lowke CONFIG hunk); weights byte-identical. vs (A) c63400be: submission.py only.
- Chain flags: `BACKBONE=b5spx3 AXE=1 IPC=1 KD5=1 KD4=1 REC=1 IPW=1 KE3=1 CDIO=1 BVR=1 LOWKE=1` (needs the LOWKE patch).
Status: NOT uploaded. Verification: experiments/lkb_build/notes.md §4.
