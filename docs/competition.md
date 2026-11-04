# RealPDE Competition: overview

This page summarises the competition pages as of 2026-08-20: the [competition site](https://realpdecompetition.github.io/),
the [Codabench Track 2 pages](https://www.codabench.org/competitions/17385/) (read through the JSON API,
`https://www.codabench.org/api/competitions/17385/`) and the [RealPDEBench](https://realpdebench.github.io)
benchmark. The Codabench pages are the authoritative source; later announcements may have changed details.

## What it is

The RealPDE Competition (NeurIPS 2026) is a scientific machine-learning competition built on real-world measurements
paired with simulations: time-resolved PIV of water-tunnel flow around a NACA4418 airfoil, matched with 3D CFD of the
same geometry and conditions. It is based on the ICLR 2026 paper [RealPDEBench](https://arxiv.org/abs/2601.01829)
(Hu, Feng, … Tailin Wu; Westlake University AI4Science).

There are two tracks with a shared data release and separate leaderboards:

- Track 1, Sim-to-Real Transfer: fuse simulated data with noisy, partially observed real measurements across
  distribution shift ([Codabench 17363](https://www.codabench.org/competitions/17363/)).
- Track 2, Long-Term Test-Time Adaptation (LTTTA): window-by-window streaming prediction on real PIV trajectories,
  with adaptation to the ground truth revealed after each window. Agent-augmented methods are allowed
  ([Codabench 17385](https://www.codabench.org/competitions/17385/)). This repository is a Track 2 entry; the
  technical specification is in [track2.md](track2.md).

## Timeline

The schedule as published on 2026-08-20. All boundaries are UTC (not AoE); each phase starts at 00:00:00 and ends at
23:59:59 on its boundary dates.

| Date | Milestone |
|------|-----------|
| 2026-07-05 | Launch; Warm-up Phase opens |
| 2026-07-19 | Warm-up Phase ends |
| 2026-08-05 | Development Phase restarted (scoring changed; starting kit v6; old leaderboard frozen) |
| 2026-08-20 | Registration deadline (a form is required; joining on Codabench alone is not registration) |
| 2026-09-27 | Main Development Phase ends; the hidden validation leaderboard decides the top-10 shortlist |
| 2026-10-25 | Final Decision Phase ends (organizers retrain the shortlisted methods and evaluate them on a private test set) |
| 2026-11-10 | Final results announced |
| 2026-11-25 | Code and fact-sheet deadline |
| 2026-12-06 | NeurIPS 2026 workshop presentation |

Outcome for this entry: it was first on the Track 2 development leaderboard at the end of the Development Phase
(final score 82.452, submission 948041), and after the organizers retrained the shortlisted methods and evaluated them
on the private test set it won Track 2. The development-phase uploads are listed in [submissions.md](submissions.md).

## Participation rules

- Teams of at most 3 people sharing one Codabench account; multiple accounts lead to disqualification.
- Registration forms: [Track 2](https://forms.gle/qmMYaK5u9r86rPGKA), [Track 1](https://forms.gle/LYeTgUTfr4ygntNu6).
- Submissions: 1 per day in the Development Phase (3 per day during warm-up), at most 100 per phase. Submissions that
  fail on the platform side (storage errors) do not count and can be resubmitted about an hour later.
- The starting kit is in the Codabench Files tab, visible after participation approval. v6 is the current version;
  older `scoring.py` / `local_eval.py` files are out of date.

## Rules on data, verification and evaluation

- Data provenance: training may use only the released `train_sim` / `train_real`. Self-generated simulation data,
  external datasets and pretrained models from other sources are not allowed. The official baseline checkpoints
  (trained on the release) are allowed. Standard on-the-fly augmentation of released samples is allowed; every
  training sample must trace back to the release through an identifiable transform.
- Final verification: the top 10 on the development leaderboard are shortlisted. The organizers retrain each method
  from scratch on their cluster (they hold only the released data and the team's training code; no weights are
  accepted) and evaluate it on a private test set with unseen parameter regimes. Private-test scores decide the top 3.
  A method that cannot be reproduced is disqualified. The exact final package format was to be announced about
  5 days before the end of the Development Phase.
- Evaluation integrity: the test-time protocol reveals ground truth step by step through the interface. Reading
  evaluation files from disk, using the current step's ground truth, or exfiltrating evaluation data leads to
  disqualification. Adaptation logs are audited on the server.
- LLM use: only through the organizers' gateway (OpenAI-compatible, `gemini-3.5-flash` at the time); its latency
  counts toward the Time metric. No participant API keys and no other network access.
- Winners must open-source the complete solution before the NeurIPS presentation.

## Prizes

Per track: $6,000 / $3,000 / $1,500 (funded by Uniforce AI Ltd.). The top 3 give oral presentations at the workshop;
the top 5 co-author a joint results paper (each team may add one advisor) and receive certificates.

## Data issues and announcements up to 2026-08-20

- 2026-08-12: `train_real/7575_0.h5` duplicates `6300_0.h5` and should be excluded from training. Other Re 7575
  angles are unaffected.
- 2026-08-09: the augmentation boundary was clarified (FAQ; see the provenance rule above).
- 2026-08-05: the Development Phase was restarted with new scoring: `sps_score` maps linearly to 0–100, targets
  outside the field of view or inside the airfoil are excluded from SPS, and `reset_ttt_state` is timed. The interface
  was unchanged; all teams had to resubmit.
- 2026-07-22: SPS interval bounds (`lower` / `upper` in `info`) became supported.

## Contact

`realpde-competition@googlegroups.com`, or the Codabench forum. Lead organizer: Tailin Wu (Westlake University);
15 organizers from 11 institutions.
