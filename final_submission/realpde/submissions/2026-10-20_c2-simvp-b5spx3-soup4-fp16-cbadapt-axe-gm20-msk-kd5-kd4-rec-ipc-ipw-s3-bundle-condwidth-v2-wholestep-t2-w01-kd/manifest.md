# 2026-10-20_c2-simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-rec-ipc-ipw-s3-bundle-condwidth-v2-wholestep-t2-w01-kd

c2 twin of **FZ-E e71e0cbf** + **IPW** (experiments/c2_ipw/notes.md): source = the FZ source + CONFIG `"ipc_prewarm": True` + a 12-line untimed
out-of-process warm-up at the end of get_ttt_model (`experiments/c2_ipw/ipw_submission.diff`; 16 changed lines in submission.py, nothing else).
Outputs identical in-process and out-of-process; mean round trip −0.66 ms/step on the 168-step shim stream (one-time ≈ 110 ms removed).
Chain: `experiments/c2_ipw/0001-retrain_all-b5spx3-KE3-dense-IPW.patch` (supersedes c2_ke3/0002; adds IPW=1 and the KE3 dense flag), flags
`AXE=1 IPC=1 KD5=1 KD4=1 REC=1 IPW=1 [KE3=1]`. Weights: no new fits (IPW) / dense ke (KE3).
Status: NOT READY — needs main's review + V; c2 chain proof + acceptance below when done.
