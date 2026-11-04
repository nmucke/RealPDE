# Trained weights of the final solution (retrain by the published final image)

The four weight files of the final Track 2 model, as produced by `final_submission/`'s `train()`. Together with the final
source dir they form a complete Codabench package:

| file | size | sha256 |
|---|---|---|
| `model.pth` | 40.8 MB | `3d7d13009805fbef509bc804bdaeb6b9ad2f32a8d02db457431149fb6765ef6d` |
| `prior.npz` | 16.1 MB | `f2b254d6596bdd0f8813f093651f0f5eec7741ca10ad0bf9cdb7b9341e8779a0` |
| `ke_net_flow.pt` | 2.1 MB | `3176a26a1c278cde953b68c0e2109a7ea0f6a5c56d397753e25499f49bc342cc` |
| `cond_width_weights.pt` | 95 kB | `96356f99bdcea2704e8802701e74b8f60257c26768c19491ef7312bcffaa1c6e` |

`SHA256SUMS` holds the same hashes (`sha256sum -c SHA256SUMS`). `package_info.json` is the record `train()` wrote, with the
sha256 of every file of the package it built.

## Where they come from
- Image: the published final image `docker.io/nmucke/realpde-t2-nmucke@sha256:20140bd5f8c86875cf9e78bd576559d4017d047f5c57a3a6c1515c69473e5dbf`,
  pulled by digest.
- Run: `python -c "import solution; solution.train('/data', '/output', 'cuda')"` on 2026-10-06, under the organizers' conditions:
  - no network;
  - `/data` read-only, holding only the released `train_sim/` (100 files), `train_real/` (81 files, without `7575_0.h5`) and
    `mean_std_real.pt`;
  - a fresh `/output`;
  - 1 × NVIDIA A100-SXM4 40 GB and 16 CPU cores.
- Result: exit 0 after 2 h 24 min. The package's 79 code files are identical to those of Codabench submission 948041
  (`f11d2c1d`).

## How they relate to submission 948041
These are **not** the byte-identical weights of the uploaded archive `f11d2c1d`. Those were trained on an RTX 3090 and are not
in this repository. These weights are a fresh retrain of the same recipe and seeds, which is what the organizers' own retrain
produces:
- `model.pth` and `prior.npz` are byte-identical across all four of our independent A100 retrains, so the backbone is
  bit-reproducible on that GPU model.
- `ke_net_flow.pt` and `cond_width_weights.pt` come from small refits that are not bit-deterministic. They vary slightly between
  runs.

Local subscores on the same 168-step evaluation stream (starting-kit `local_eval.py` / `experiments/run_preset.py`):

| weights | rel_l2 | tke | mvpe | sps |
|---|---|---|---|---|
| these, in the image on an A100 | 96.177 | 82.172 | 97.503 | 52.239 |
| these, on an RTX 4090 | 96.177 | 82.172 | 97.503 | 52.240 |
| submission 948041 (`f11d2c1d`), RTX 3090 | 96.173 | 82.173 | 97.501 | 52.233 |

The weights are trained on all of `train_real`, so scores on those files are in-sample.

## License
The weights are trained on the RealPDE competition data (RealPDEBench, CC BY-NC 4.0) and are distributed under the
same terms: CC BY-NC 4.0, attribution to the RealPDEBench authors and to this repository. They are not covered by the
MIT license of the code (see `THIRD_PARTY_NOTICES.md`).

## Use
Copy the final source dir and add the weights:
```bash
cp -r final_submission/realpde/submissions/2026-10-23_simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-kd6-kd7-rec-ipc-ipw-cdio-bvr-lowke-lsk-recr-kd8-r97/source PKG
cp weights/final_retrain_a100/{model.pth,prior.npz,ke_net_flow.pt,cond_width_weights.pt} PKG/
python -B experiments/run_preset.py --package PKG --data EVAL lb0927_final
```
`PKG` is then the same as `/output/package` after `train()`, and is usable as a Codabench submission or with any preset in
`experiments/`.
