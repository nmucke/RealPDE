# 2026-10-17_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd3-s3-bundle-condwidth-v2-wholestep-t2-w01-kd

This archive is K3E-KD2 3b910139 (`2026-10-17_…-axe-gm20-msk-kd2-…`) with ONE change: **KD3** (experiments/kd3_step_diet),
ported to the AXE layout by V22 (experiments/v22_verify/notes.md). The steady whole-step CUDA graph runs the backbone, the
mega (PRE + energy + TAIL) body, the ke_net_flow features and the cond-width apply through KD3's bit-exact, fewer-kernel
twin (new `kd3_step.py`, CONFIG `"kd3_step": True`). Outputs are bit-identical to 3b910139 (and so to K3E 51016ad3).
No new weights and no new fits: model.pth, prior.npz, ke_net_flow.pt and cond_width_weights.pt are byte-identical to 3b910139.

- **submission.zip sha256 `165826e9ddb21fb72bc8bfa926aebed1ab34c3cf9250f32c32b05bc210a99640`**, 54.3 MB zipped, 77 entries
  (3b910139's 76 + kd3_step.py). Entry diff: only kd3_step.py is new; cuda_graph_wholestep.py and submission.py differ;
  74 entries byte-identical (`experiments/v22_verify/out/k3ekd3_entry_diff_vs_3b910139.txt`).
- **Code**: `source/` = the 3b910139 source + `experiments/v22_verify/out/k3e_kd3_changes.diff` (3 files):
  - submission.py: 1 CONFIG line `"kd3_step": True` (same as K3F-KD3 f39128a9);
  - cuda_graph_wholestep.py: KD3's 10 hunks; 9 applied by `patch` (with offsets), and 1 re-applied by hand
    (`inv_nR = reciprocal(...)`: its context line is `ky = kxy[self._iy:...]` in the AXE layout). Same code as f39128a9's hunks;
  - kd3_step.py: f39128a9's file + AXE's br_adapt branches in `mega_body` / `_coefficients`, copied op for op from
    cuda_graph_megatail._body's AXE branches: e_t2 row gather by `_br_eidx`, g2 expanded to Kc rows, `_coefficients(lam=br.lamv)`,
    z row gather by `_br_zidx`, constK, and the emitted row k_eb (out / fluc / zc_ls / const gathers). Without these branches,
    KD3's mega_body would ignore the ridge-candidate rows.
- Build: `experiments/v22_verify/v22_build_k3e_kd3.sh`. Chain: `experiments/v22_verify/0001-retrain_all-kd3-source.patch`
  (BACKBONE=b5spx3 AXE=1, KD3 default) rebuilds these 77 entries byte-identical in a fresh clone.

## Evidence (V22, own clean rooms `/export/scratch1/ntm/tmp/v22/a/b/c/d/e/`, own drivers over unedited kdr / p7 / v10 / v16 tools)
- The KD3 path is live after build and after every stream (`_kd3=True`, `kd3_error=None`, KD2 live). The AXE branch is wired
  into the KD3 graph: io._br_ad = BRAdapt, K_br 4, Kc 9, eb_outK has 9 rows.
- kdr fold0 (811) / short (117) / adv (378): **0 pred / lower / upper / loss mismatches vs 3b910139**. adv non-finite and
  unordered sets are the same as the parent's (5 / 5). whole replays are 773 / 22 / 309 (= parent). W01 self-test bit_exact.
- p7 G0 / G1 runtime replay failure: 0 mismatches vs the uninjected run, and the RNG probe is OK. v16 reset probe PASS. Own
  white-box B→A(NaN/Inf)→B probe: B1 == B2, all KD2 and KD3 state tensors are restored, and all 36 window hashes equal 3b910139.
- Contract fuzz cuda + cpu: 396 / 396, 0 violations. modpath: 0 modules resolve into the repo. local_eval shim_test GPU and
  CPU: rel_l2 / tke / mvpe / sps = 3b910139's.
- Kernels per steady step (CUPTI, fold0 traj 0 steps 10–29): **777 → 681 (−96)**; memcpy 9 → 5; kernel time 2568 → 2486 µs.
- Max allocated GPU memory: 416.8 MB (3b910139: 380.5 MB).
- Timing: see experiments/v22_verify/notes.md §C.
- Accuracy: the outputs are identical to 3b910139 / 51016ad3, so K3E's LCO numbers apply.
