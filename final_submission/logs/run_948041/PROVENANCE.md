# Provenance of submission 948041 (archive sha256 f11d2c1d0e00b6270ad71316117eb44e0de303b747bf9302ddcbfcdc7804504e)

All artefacts were produced by `scripts/retrain_all.sh` (this package's `realpde/`), running in one fresh clone
(nothing reused) on 1 × RTX 3090. Timestamps in `chain_step_logs/` are CEST (UTC+2).

| archive entry | produced by | log |
|---|---|---|
| `model.pth` (packed K=3 ensemble) | steps 05a (3 sim pretrains), 06 (12 real fine-tunes + 3 soups), 07 (pack) — full zero-reuse chain, 2026-09-24 19:22–21:26 UTC | `fzt_chain.log`, `chain_step_logs/retrain_05a_*`, `retrain_06_*`, `retrain_07_*` |
| `prior.npz` | step 08 (prior), same run | `fzt_chain.log` |
| `cond_width_weights.pt` (W2) | step 05b (cross-fitted pool) + step 09 (3 seeds, best-of on own validation), same run | `chain_step_logs/retrain_05b_*`, `retrain_09_*` |
| `ke_net_flow.pt` (KE3, 4-seed dense energy net) | step 08 with `KE3=1` in the same clone, 2026-09-25 08:51–09:15 UTC | `fzk_chain.log` |
| code (`*.py`) | step 10 package assembly with the final flags; tracked source `submissions/2026-10-23_…-lsk-recr-kd8-r97/source` | `x5v_build_and_verify_notes.md` |

Notes:
- `chain_step_logs/retrain_07_pack_b5spx3.log` comes from a later re-run of the deterministic pack step, which
  produced byte-identical `model.pth`. The step-08 logs of the original runs were overwritten by later re-runs, so
  `fzk_chain.log` is their record.
- The same chain, run from our release inside the official image on an A100 with the final flags, reproduced every
  code file. It also reproduced `model.pth` and `prior.npz` byte for byte against another in-image run, and the local
  evaluation matched within ±0.03 per subscore (2 h 27 min).
