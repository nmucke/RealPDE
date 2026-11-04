# data/

Datasets live here and are **gitignored**. See [../docs/data.md](../docs/data.md) for download
instructions. Expected layout:

```
data/
└── competition/
    ├── train_sim/       # extracted from train_sim.tar.gz
    ├── train_real/      # extracted from train_real.tar.gz  (never use 7575_0.h5)
    ├── example_data/
    └── baseline_checkpoints/
```

## Fresh clone: what is regenerable and what must be copied

Everything under `data/`, all `*.pth`/`*.pt`/`*.npz` and all `submissions/*/submission.zip`
are gitignored. From a bare clone:

- `data/competition/` — download as in `docs/data.md`.
- `data/real/`, `data/sim/` — `python scripts/split_data.py` (ratios pinned to reproduce the
  73/4/4 real split used by every manifest).
- `data/real_lco/` — `python scripts/lco_make_splits.py`.
- Backbone, `prior.npz`, `ke_net_flow.pt`, `cond_width_weights.pt` — retrain chains in each
  `submissions/*/manifest.md`; the vendored shipping modules are resolved from the tracked
  `submissions/2026-09-29_simvp-s3-bundle-condwidth-graphed/source/` when the untracked
  `experiments/c13p_packaging/pkgs/` tree is absent.
- The leaderboard archive itself (sha256 `b7c6004b…`) cannot be rebuilt byte-identically; copy
  `submissions/2026-09-29_simvp-s3-bundle-condwidth-graphed/submission.zip` from the machine
  that uploaded it if a safety re-upload may be needed.
