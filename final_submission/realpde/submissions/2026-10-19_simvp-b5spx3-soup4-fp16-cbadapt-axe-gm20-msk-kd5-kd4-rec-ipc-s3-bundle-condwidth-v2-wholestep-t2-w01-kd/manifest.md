# 2026-10-19_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-rec-ipc-s3-bundle-condwidth-v2-wholestep-t2-w01-kd

**FZ-E** (freeze candidate) = K3E-KD3 165826e9 (`2026-10-17_…-axe-gm20-msk-kd3-…`) + every approved add-on, each behind its own CONFIG flag
(experiments/fz_freeze/notes.md). Weights byte-identical to the base (model.pth, prior.npz, ke_net_flow.pt,
cond_width_weights.pt); no new fits; no new data.

- **submission.zip sha256 `e71e0cbfcabea6ad3aebaae38033edde8952918bcef9850cff46a1fa43771410`**, 79 entries (base + `kd4_w01.py`, `rec_detect.py`); content differs in exactly
  `submission.py` (4 CONFIG entries), `cuda_graph_wholestep.py` (IPC + REC), `kd3_step.py` (KD5 + REC),
  `w01_warmup_graph.py` (IPC + KD4) (experiments/fz_freeze/out/entry_diff_E.txt, fz_E_vs_base.diff).
- CONFIG (off value restores the base bit for bit — verified all-off 811/811 vs base):
  `"ipc_one_storage": "auto"` (False) · `"kd5_near": True` (False) · `"kd4_w01": True` (False) ·
  `"rec_detect": {"thr": 0.405465, "n0": 6.0, "rescale": True, "u_eps": 1e-4}` (None).
- Built by experiments/fz_freeze/scripts/fz_build.py (3-way merges of the verified add-on packages IPC 2a75cbfa, KD5 661fac2d, KD4 7d5ce586 (base-independent files), REC 2a38d03c; all clean)
  + ONE composition hunk in kd4_w01.py (KD4's G0 / G1 capture traces KD3's exact branches when KD5 is on; without it
  the W01 self-test fails and KD4 silently disables itself). Packaged by fz_pack.sh.
- Chain: scripts/retrain_all.sh + experiments/fz_freeze/0001-retrain_all-kd4-rec-fz-flags.patch, BACKBONE=b5spx3
  AXE=1 IPC=1 KD5=1 KD4=1 REC=1 → step 10 reproduces this archive entry-for-entry.
