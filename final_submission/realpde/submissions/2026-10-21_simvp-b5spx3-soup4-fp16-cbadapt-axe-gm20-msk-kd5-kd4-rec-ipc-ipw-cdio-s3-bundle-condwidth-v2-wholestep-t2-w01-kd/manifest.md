# 2026-10-21_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-rec-ipc-ipw-cdio-s3-bundle-condwidth-v2-wholestep-t2-w01-kd

CDP port (experiments/cdp_port/notes.md) of HX2's **cuda_direct_io** lever (experiments/hx2_harness/notes.md) onto
**FZ-E + IPW** (c2 twin 2026-10-20_c2-…-ipc-ipw-…): source = the IPW source + CONFIG `"cuda_direct_io": True` (3 lines in
submission.py) + HX2's cuda_graph_wholestep.py (flag read + `_ttt_step_cuda_direct`). Nothing else differs.
Active ONLY when ttt_step receives CUDA tensors (D2D in, CUDA clones out, 4-byte loss D2H + one sync); CPU / numpy inputs
take FZ-E's unchanged path. Code only: no new weights / fits.

## submission.zip (FZ-E + IPW + KE3 + CDIO, 3090 chain weights)
- sha256 `eef3cfe43cc7cb3c1c783a1220c0e4bfbb0018d40b9a0091dad9c6a644575276`, 79 entries.
- Provenance: FZT clone /export/scratch1/ntm/tmp/fzt/clone (steps 01-08 from the 2a2bbb09 FZ-E+IPW+KE3 3090 build) with
  this source dir + experiments/cdp_port/0001-retrain_all-CDIO.patch applied clone-locally, then
  `BACKBONE=b5spx3 AXE=1 IPC=1 KD5=1 KD4=1 REC=1 IPW=1 KE3=1 CDIO=1 bash scripts/retrain_all.sh --only 10`
  -> submissions/retrained_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-rec-ipc-ipw-cdio-kes4-s3-bundle-condwidth-v2-wholestep-t2-w01-kd/.
- Content vs 2a2bbb09 (…-ipc-ipw-kes4-…, same clone): identical entry list; bytes differ ONLY in submission.py and
  cuda_graph_wholestep.py (other entries differ in zip timestamps only; weights model.pth / prior.npz / ke_net_flow.pt /
  cond_width_weights.pt byte-identical).
- Chain flags: `BACKBONE=b5spx3 AXE=1 IPC=1 KD5=1 KD4=1 REC=1 IPW=1 KE3=1 CDIO=1` (CDIO needs the patch above).
Status: NOT uploaded; acceptance in experiments/cdp_port/notes.md.

## Acceptance (experiments/cdp_port/notes.md §4) on the exact zip vs 2a2bbb09
- CPU inputs: kdr fold0 811 / short 117 0 mismatches; unedited local_eval example + shim_test subscores identical.
- CUDA inputs (lever active): kdr --no-mutate finite (0 nonfinite) and value-identical to the CPU-input 2a2bbb09 reference,
  fold0 0/811, short 0/117; hx2_oop out-of-process (torch-mp and by-value, CUDA and CPU inputs) one sha in all arms.
- IPW fires in a spawned torch-mp worker (resource_sharer listener started in get_ttt_model; FZ-E control does not).
- Timing (calm 3090, 8 rounds): −0.44 to −0.55 ms/step with CUDA inputs (8/8); CPU-input controls ±0.02 ms call.
