# 2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke

**FZ-EIKC + BVR + LOWKE + KD6 + KD7**: KD6's steady whole-step body with its main-stream (critical-path) post-join
elementwise chains fused into 11 bit-exact jiterator (NVRTC) kernels (experiments/kd7/notes.md): e_new → gamma → _shape →
fluc (A, B, C), disc → c → outK (D, E), pred_phys (P), the W2 grad-magnitude (G), dp / f5 / CT normalisation (F, N), the
width multiplier (M), LOWKE's gain (L). Written with IEEE _rn intrinsics (no FMA contraction), native NaN-propagating
clamp, expf / logf / powf (bit-identical to torch's over all 2^32 inputs, NVRTC 12.1 and 12.5). Compiled + self-tested
(bitwise, vs the verbatim native chains, NaN / inf / denormal inputs) at build, then covered by KD6's graph self-test;
any failure → the KD6 body (tier 2 failure → tier 1 only). Outputs bit-identical to 40fd4907. Code only: no new
weights / fits.

## submission.zip
- sha256 `fa94876280c2b8f82b93d0a6b66a66f9b1da3345db6a43369614ad9d345c08ce`, 81 entries, 60 080 443 B extracted.
- Built by experiments/kd7/scripts/kd7_build_pkg.py from 40fd4907…: every entry copied byte for byte except
  `submission.py` 2b0a4b93 → d568c956 (+ CONFIG `"kd7": True` with a 4-line comment), `cuda_graph_wholestep.py`
  297cb304 → a6315dba (flag `_kd7`, kd7_fuse.build before KD6's build, KD6-without-KD7 retry), `kd6_conc.py`
  587d241f → d8617e30 (`w._kd7j` branches; native lines kept verbatim in the else-branches), NEW `kd7_fuse.py` 7a12d078.
- Chain: the freeze flags (bare b5spx3) + `KD7=1` with experiments/kd7/0001-retrain_all-KD7.patch; step 10 on the FZT
  clone's outputs reproduces every code entry (ke_net_flow.pt differs there for the unpatched freeze target as well —
  FZB_DIFF, pre-existing).

3090: steady replay −0.103 ms (8/8), local_eval −0.119 ms/step (8/8), time_score +0.12.
Status: NOT uploaded. Acceptance: experiments/kd7/notes.md (out/acc). A100 timing: not measured.
