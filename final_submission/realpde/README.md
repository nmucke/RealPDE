# RealPDE Competition — Track 2 (LTTTA)

Working repo for our team's entry in the [NeurIPS 2026 RealPDE Competition](https://realpdecompetition.github.io/),
**Track 2: Long-Term Test-Time Adaptation** — long-horizon prediction of real PIV
airfoil flow with continuous adaptation to streaming observations. Our angle:
**data assimilation** as the adaptation and uncertainty machinery.

## Status / next steps

- [x] Team registered (2026-08-20)
- [ ] Participation approved on [Codabench Track 2](https://www.codabench.org/competitions/17385/);
      download **starting kit v6** from the Files tab (visible after approval) into `starting_kit/`
- [ ] Download training data (~16 GB compressed): see [docs/data.md](docs/data.md)
- [ ] First smoke-test submission early — only **1 submission/day** in the Development Phase

## Key facts

| | |
|---|---|
| Task | Streaming windows of real PIV flow (NACA4418 airfoil): predict next 20 frames, adapt when the previous window's ground truth is revealed |
| Tensor per step | `(1, 20, 32, 64, 3)` normalized `[u, v, p]` (p zero-filled, not scored) |
| Ranking | `final_score` from 5 subscores: Rel-L2, TKE, MVPE, Time, SPS (weights unpublished) |
| Development Phase ends | **2026-09-27** (UTC) — hidden validation leaderboard |
| Final | Top 10 shortlisted → organizers **re-train from scratch** and evaluate on private test → top 3 |
| Environment | PyTorch 2.2.2 / CUDA 12.1 / Python 3.10, offline, 256 MB submission, 10 min run limit |

## Docs

- [docs/track2.md](docs/track2.md) — Track 2 technical reference: protocol, interface, metrics (the working spec)
- [docs/data.md](docs/data.md) — data description and download
- [docs/METHOD.md](docs/METHOD.md) — method description of the final solution
- [docs/REPRODUCE.md](docs/REPRODUCE.md) — reproduction notes for the training chain

## Repo layout

```
docs/         Track 2 reference, data, method and reproduction notes
data/         datasets (gitignored; see data/README.md)
starting_kit/ official starting kit v6 (drop-in from Codabench Files tab)
src/          our code (models, DA layer, training, local eval harness)
experiments/  configs, run logs, results
submissions/  packaged submission archives, one dir per submission
```

## Ground rules (from the competition rules — these bite at the final stage)

- Training may use **only the released data** (`train_sim`, `train_real`). No external
  data, no external pretrained weights. Official baseline checkpoints are allowed.
- **Exclude `train_real/7575_0.h5`** (corrupt: duplicates case 6300).
- Top-10 methods are re-trained from scratch by the organizers — keep the whole
  training pipeline reproducible from the release from day one.
- Never read evaluation files from disk in a submission; only use what `ttt_step` provides.
