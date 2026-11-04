# data/

Nothing in this directory except this README is tracked. The competition data are released by the organizers under
CC BY-NC 4.0 and are not redistributed here. [../docs/data.md](../docs/data.md) describes the release and gives the
full download commands; in short:

```bash
pip install -U "huggingface_hub[cli]"
DATA=$PWD/data/competition
hf download AI4Science-WestlakeU/RealPDE-Competition-Data --repo-type dataset --local-dir "$DATA"
tar -C "$DATA" -xzf "$DATA/train_sim.tar.gz"     # -> data/competition/train_sim/*.h5   (100 files)
tar -C "$DATA" -xzf "$DATA/train_real.tar.gz"    # -> data/competition/train_real/*.h5  (82 files, 81 usable)
cp "$DATA/example_data/mean_std_real.pt" "$DATA/"   # optional
```

Only `train_sim/` (100 files) and `train_real/` are used for training. `train_real/7575_0.h5` duplicates
`train_real/6300_0.h5` (organizer announcement of 2026-08-12); it can stay in the directory, because the training
chain skips it by name, leaving 81 usable real files. The release's baseline checkpoints are not used.

## Retraining the final submission

The final package is [`../final_submission/`](../final_submission/) (see its `README.md`). There are two ways to run
its training chain.

1. As the organizers did (official image, `solution.train`). `train(data_dir, output_dir, device)` expects

   ```
   <data_dir>/
   ├── train_sim/          # 100 .h5 files
   ├── train_real/         # the released .h5 files (7575_0.h5 is skipped)
   └── mean_std_real.pt    # optional; without it the identical copy in the starting kit is used
   ```

   With the commands above, `<data_dir>` is `data/competition/`. Run it from `final_submission/` (or `/workspace`
   inside the image built from its `Dockerfile`, with the data mounted at `/data`):

   ```bash
   python -c "import solution; solution.train('/data', '/output', 'cuda')"
   ```

   `train()` copies `final_submission/realpde/` into `<output_dir>/_work/`, symlinks
   `data/competition/{train_sim,train_real}` there to `<data_dir>`, and runs `scripts/retrain_all.sh`. Nothing is
   written into the repository.

2. Without Docker, from a copy of `final_submission/realpde/`. Copy the tree first, because the chain writes data
   links, checkpoints and the package into the tree it runs in. With `DATA` set as above:

   ```bash
   cp -r final_submission/realpde ~/realpde-work && cd ~/realpde-work
   python3.10 -m venv .venv
   .venv/bin/pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
   .venv/bin/pip install -r requirements-train.txt
   mkdir -p data/competition && ln -s "$DATA"/train_sim "$DATA"/train_real data/competition/
   DRY_RUN=1 bash scripts/retrain_all.sh   # print the resolved configuration, run nothing
   KEEP_DUMPS=0 bash scripts/retrain_all.sh 2>&1 | tee retrain.log
   ```

   The chain writes checkpoints, splits and caches under the copy's `data/`, `experiments/` and
   `submissions/retrained_*/`.

