# CLAUDE.md

Team entry for the NeurIPS 2026 RealPDE Competition, **Track 2: Long-Term Test-Time
Adaptation** — streaming prediction of real PIV airfoil flow, with data assimilation as our
adaptation/uncertainty machinery. Development Phase ends **2026-09-27 UTC**; 1 submission/day.

## Read first

- `docs/track2.md` — the working spec: protocol, `ttt_step` interface, metric formulas, limits.
  Treat it as the contract every submission must satisfy.
- `docs/plan.md` — current strategy and milestones.
- `docs/reference/` — raw Codabench page dump. Re-fetch for announcements:
  `curl -s https://www.codabench.org/api/competitions/17385/` (JSON API works without login).

## Hard constraints (violations waste a daily submission or disqualify us)

- Submission code targets the eval image: **Python 3.10, PyTorch 2.2.2, numpy 1.26**, offline.
  Only pure-Python extras, vendored inside the archive. ≤ 256 MB extracted. 10-min run limit.
- Everything inside `ttt_step` and `reset_ttt_state` is timed (Time subscore,
  `t_numerical = 0.72896 s` reference) — keep per-step cost low by design.
- Interval bounds (`info["lower"]/["upper"]`) are all-or-nothing for an entire run; both keys or
  neither, every step. NaN/Inf or wrong shapes zero out *all* subscores.
- Training may use **only** the released `train_sim`/`train_real` (+ on-the-fly augmentation).
  No external datasets or pretrained weights. **Never use `train_real/7575_0.h5`** (corrupt).
- Everything must be re-trainable from scratch by the organizers: deterministic seeds, pinned
  deps, no local-only inputs. Never read evaluation files from disk in submission code.

## Layout & conventions

- `starting_kit/` — official kit v6, drop-in from the Codabench Files tab; don't edit in place,
  build on it from `src/`. Its `scoring.py`/`local_eval.py` are the local scoring reference.
- `src/` — our code. `experiments/` — configs + recorded results (subscores per run).
  `submissions/` — one dir per packaged submission, named `YYYY-MM-DD_short-desc/`.
- `data/` is gitignored; layout in `data/README.md`.
- Validate any submission with `local_eval.py` (streaming loop + the five subscores) before
  uploading — the daily quota is precious.
- Commit as you go, small logical commits.
