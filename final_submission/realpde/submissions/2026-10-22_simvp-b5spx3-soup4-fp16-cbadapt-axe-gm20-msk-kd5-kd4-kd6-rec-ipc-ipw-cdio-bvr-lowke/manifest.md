# 2026-10-22_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-rec-ipc-ipw-cdio-bvr-lowke

**FZ-EIKC + BVR + LOWKE + KD6** (KD6 v6): fork / join concurrency inside the steady whole-step CUDA graph (experiments/kd6_concurrency/notes.md): the
backbone-independent ~45 % of the noscan graph's work (step head, KD2 selectors, mask, REC counters, mega PRE, advect /
coefficients / centre gains, k-table, x-only cond-width features) on a side stream concurrently with the backbone, two
smaller forks (‖ ke net, ‖ W2 net) and, on a second side stream, the gate branch of every backbone block ‖ its depthwise
convs. Outputs bit-identical to the base; build-time self-test vs the parent graph (any failure / mismatch → the parent
graph). Code only: no new weights / fits.

## submission.zip
- sha256 `40fd4907d358a3d46df814ddad55a0cb831ee8e281112e3ff68d5fb8ad767784`, 80 entries.
- Built by experiments/kd6_concurrency/scripts/kd6_build_pkg.py from 75a2a371… (the 09-26 candidate, 2026-10-22_simvp-…-cdio-bvr-lowke): every entry copied byte for byte except
  `submission.py` 890642be → 2b0a4b93 (+ CONFIG `"kd6_concurrency": True` with a 4-line comment), `cuda_graph_wholestep.py` 10fa709c → 297cb304 (LOWKE's file + the same KD6 hunks)
  (flag read, `_kd6_build()` after the parent captures), NEW `kd6_conc.py` 587d241f (the same file in every KD6 package).
- Chain: `BACKBONE=b5spx3 AXE=1 IPC=1 KD5=1 KD4=1 REC=1 IPW=1 KE3=1 CDIO=1 BVR=1 LOWKE=1 KD6=1` with
  experiments/kd6_concurrency/0001-retrain_all-KD6.patch (on top of HEAD incl. the LOWKE patch); step 10 on the FZT clone's
  outputs reproduces all 80 entries.

Status: NOT uploaded. Acceptance: experiments/kd6_concurrency/notes.md (acc_v6c). A100 timing: requested from c2.
