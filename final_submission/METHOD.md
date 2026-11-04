# nmucke: Track 2 method

## 1. Summary
The forecaster is an ensemble of three SimVP-style convolutional networks. Each member is pretrained on the simulated
data, fine-tuned on the real data, and is the weight average ("soup") of four fine-tunes. The three members run
packed together as one network.

Around the networks sits a test-time pipeline. It is causal (it only uses frames that have already been revealed) and
gradient-free (no network weights change at test time). Per trajectory, it:
- normalises the inputs with running statistics of the frames revealed so far;
- corrects the ensemble output with a small online least-squares model fitted to the errors seen so far (called LSK
  below);
- calibrates the energy of the predicted fluctuations;
- damps fluctuations where the predicted energy is low (LOWKE);
- returns prediction intervals whose width and centre are tuned on the previous window's interval score;
- restarts its running statistics if the flow regime changes abruptly (REC).

For speed, each step runs as pre-recorded CUDA graphs spread over several GPU streams, with fused kernels that give
bit-identical results. Sim pretraining and the souped three-member ensemble gave the largest accuracy gains. The
per-trajectory interval and energy rules came next, then the latency work.

**Names used below**
- *Window*: one Codabench step, i.e. 20 input frames and the 20 frames to predict. *Lead*: the position (1–20) of a
  predicted frame within the window.
- *SPS*: the competition's interval score (sharpness and coverage of `lower`/`upper`).
- *Energy net*: a small network that predicts the fluctuation (turbulent kinetic) energy of the coming window. *KE3*
  is the 4-seed version of it that the final model uses.
- *W2*: the small network that predicts the interval half-width.
- *LSK*, *LOWKE*, *REC*: the least-squares correction, the low-energy damping and the regime-change reset (§6).
- *LCO folds*: leave-conditions-out validation folds used during development. Each holds out whole
  (Reynolds number, angle of attack) conditions. fold0 and fold1 hold out real conditions. foldR holds out real files
  that mimic the 18 (Re, angle) conditions present only in the simulated data, a proxy for the unseen test conditions.
  Rx holds out the highest Reynolds numbers (extrapolation). The "beyond-range" sets are simulation-derived and lie
  outside the real training range.
- *b5*: our name for the network architecture of §2.

## 2. Model
- Architecture and number of parameters:
  - SimVP-style CNN with gated spatiotemporal-attention blocks (`b5_h192_blk6_rf11`): encoder width 128, hidden
    width 192, 6 gSTA blocks, receptive field 11; 3.39 M parameters per member.
  - Three members are packed into one batched network (stem concatenation, grouped convolutions, batched 1×1
    convolutions; 10.17 M parameters), whose output is the members' mean.
  - The packed core runs in fp16; everything around it runs in fp32.
  - Small auxiliaries:
    - a condition prior: statistic maps fitted on the training files, used only to stabilise the first few windows
      of a trajectory, before the running statistics have enough data;
    - an energy net: a CNN of about 35k parameters per seed, 4 seeds fused exactly into one dense net;
    - an interval-width net (W2), a small CNN.
- Inputs and outputs:
  - Codabench's normalised `(1,20,32,64,3)` windows (official `mean_std_real.pt`); p is zero-filled and unscored.
  - Inside the model, *causal context normalisation* keeps a per-pixel running mean and standard deviation of every
    frame revealed so far in the trajectory: inputs and previously revealed targets.
  - Those two maps are fed as extra input channels, and the network output is de-standardised with them.
  - The model returns the next 20 frames plus per-element `lower`/`upper` on every step.
- Starting point: random initialisation, then sim pretraining. No official checkpoint is used.

## 3. Data
- Training data used, and how they are combined: sequential, not mixed.
  - `train_sim`: all 100 files. They are subsampled to the real 32×64 grid, rescaled to physical units
    (U∞ = 1.263·10⁻⁵·Re), and p is masked to 0 so the layout matches real data. They are used only for pretraining.
  - `train_real`: the 81 released files, used for fine-tuning and every fitted auxiliary.
- Validation split carved out of the training data:
  - None for the backbone, which trains on all 81 real files.
  - The energy nets and W2 use internal validation splits carved from the training files, for best-epoch / best-seed
    selection.
  - W2 is fitted on out-of-sample residuals from 4-group cross-fitting over the 81 files: each group's backbone,
    prior and energy net are trained on the other three groups.
  - Development decisions used the LCO folds (§1). The chain also retrains the fold models, only to check that it
    reproduces the recorded validation numbers; they are not part of the final model.
- Augmentation: none for the backbone; a random flip in y when fitting the energy nets.
- Data left out, and why: `train_real/7575_0.h5`, which duplicates `6300_0.h5` (organizer announcement of
  2026-08-12).

## 4. Training
One row per stage, in the order `train()` runs them (`scripts/retrain_all.sh`).

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps or epochs | Batch size |
|---|---|---|---|---|---|
| 1. Data splits, LCO folds and fold models (reproduction check only; not used by the final model) | train_real | as in rows 2–3 | as in rows 2–3 | as in rows 2–3 | 8 |
| 2. Sim pretraining, seeds 0/1/2 | 100 train_sim | Welsch (correntropy), 75/25 physical / standardised space | AdamW lr 5e-4, wd 0, 3 % warmup + cosine over a 20k schedule, stopped at 10k | 10,000 it each | 8 |
| 3. Cross-fitted pool for W2 (4 groups; backbones of the same architecture trained from random initialisation) | train_real, 3 of 4 groups each | Welsch | AdamW lr 5e-4, cosine (same recipe as stage 4) | 8,000 it per backbone, + prior / energy fits | 8 |
| 4. Real fine-tunes: 4 seeds per pretrain (12 in total) | 81 train_real | Welsch | AdamW lr 5e-4, wd 0, 3 % warmup + cosine | 8,000 it each | 8 |
| 5. Weight soup per pretrain; pack the 3 soups into K=3 | — | — | uniform fp64 weight average | — | — |
| 6. Condition prior (statistics, no training) and a single energy net | 81 train_real, ensemble outputs | per-window rel-L2 of the energy map | as stage 7, single seed | as stage 7 | as stage 7 |
| 7. 4-seed energy net (KE3), written exactly as one wider network | same | per-window rel-L2 of the energy map | AdamW lr 2e-3, cosine | 120 epochs per seed (best-val epoch kept) | 48 |
| 8. Interval-width net W2 | cross-fitted residuals from stage 3 | regression to a coverage-aware target half-width computed over 5×5-pixel neighbourhoods | Adam, lr 1e-3 | 40 epochs per seed; 3 seeds, best by own validation | 12 (window, channel) pairs × 20 leads |
| 9. Package assembly and local sanity evaluation | — | — | — | — | — |

- How runs or checkpoints are selected: backbones and soups use the last iterate, with no selection. The energy nets
  keep their best internal-validation epoch. W2 is best-of-3 seeds on its own validation split. All selection runs
  inside `train()` on training data.
- Seeds, and the run-to-run spread seen: seeds are fixed. Independent retrains, on another GPU model or inside the
  official image, landed within about ±0.03 per local subscore and within about 0.01 of the local composite score
  (our local re-implementation of the final score, with the time subscore held fixed).
  - The backbone is bit-reproducible on the same GPU model.
  - The energy and W2 refits vary slightly between runs.

## 5. Uncertainty bounds
The bounds are `lower/upper = c′ ∓ h`.
- **Half-width h:** the W2 net's output (a log-ratio of local to running error scale, from causal features, the lead
  index and the previous window's error map), multiplied by the pipeline's running error scale and by an adaptive
  factor κ ∈ {0.85, 0.925, 1, 1.075, 1.15}. κ is chosen separately for leads 1–2 and 3–20.
- **Centre c′ = c + w·(raw − c):** c is the interval centre produced by the energy / advection stage (§6, step 4)
  and raw is the (LSK-corrected) ensemble output. w ∈ {0.30, 0.45, 0.60, 0.75, 0.90}; the first window uses w = 1.
- **How κ and w are chosen:** once the previous window's truth is revealed, every candidate value is scored with the
  exact SPS formula on that window. Per trajectory, an exponential moving average (EMA, ρ = 0.95) of these scores is
  kept, and the best candidate is used for the next window.
- **Scored mask:** the per-lead error maps use the scorer's mask, |y| > 1e-6.
- Bounds are returned on every step.

## 6. Inference (Track 2)
No parameters are updated by gradient; adaptation consists of causal statistics and closed-form least squares.
Per step, in this order:
1. **Regime-change detector (REC):** d = |log(Ū_window / Ū_running)|, where Ū is the mean flow speed of the current
   input window and of the trajectory so far. If d > log 1.5, the running statistics restart from the current
   window, and the stored mean and energy levels are rescaled by the speed ratio. It never fires on the released real
   files; it guards against an abrupt change of flow conditions within a trajectory.
2. **Context normalisation:** update the running statistics with the revealed frames, then run the packed backbone.
3. **LSK (online least-squares correction):** H are the members' features just before their final output layer.
   φ = [Vᵀ H; 1], where V holds the top-8 right singular vectors of the stacked member output-layer weights
   (computed from the trained weights at load time). Per channel, with the revealed error e of the previous window
   and the context standard deviation s:
   - Φ ← ρΦ + Σ s²φφᵀ
   - B ← ρB + e(sφ)ᵀ
   - M = B(Φ + λI)⁻¹, with forgetting factor ρ = 0.97 and λ = 30·tr(Φ₁)/9, where Φ₁ is the first update's Σ s²φφᵀ
     and 9 is the number of features (8 directions + 1).

   The raw u, v of all 20 leads get the correction (Mφ)s. The moments are kept in fp64 and are also reset by REC.
4. **Energy calibration:** the predicted fluctuation is rescaled to match a blend of the energy net and the running
   mean of revealed energy, with the gain clamped to [1/20, 20].
   - The blend weight b ∈ {0.45, …, 0.925} is chosen per trajectory by each candidate's composite score on the
     previous window.
   - The prediction is then blended with a simple advected copy of the last revealed frame (a small least-squares
     fit whose ridge strength ∈ {0.125, 0.25, 0.5, 1} is chosen the same way). This stage also produces the interval
     centre c of §5.
5. **LOWKE (low-energy damping):** let m_t be the prediction's 20-frame time mean and K_p its fluctuation-energy map.
   With g = clamp((K_p / (1.5·mean K_p))^0.25, 0, 1), pred ← m_t + g·(pred − m_t): fluctuations are shrunk where
   the predicted energy is low. Applied from the third window of a trajectory on (windows are counted from 0).
6. **Intervals:** as in §5.

`reset_ttt_state` restores the state at the start of a trajectory: it clears every running statistic, the LSK
moments, the selector EMAs and the energy history. Nothing carries across trajectories, and the current target is
never used.

## 7. Speed
Measured on the Codabench harness: 4.84 ms/step (time score 92.46).
- **CUDA graphs:** from the third window of a trajectory on, the whole step is one replay of a pre-recorded CUDA
  graph; the first two windows have their own graphs. At load, every graph is checked against normal (eager) execution, and the model
  falls back to eager execution if a check fails.
- **Packed fp16 ensemble:** one batched forward pass instead of three separate ones.
- **Fewer GPU kernels per step:** fused scoring of the per-trajectory candidates, a leaner step body, and convolution
  rewrites that differ only in floating-point rounding order (|Δ local composite| ≤ 0.0004).
- **Several GPU streams:** work that doesn't depend on the backbone runs on side streams, in parallel with it.
  Checked with a race checker and with replays under perturbed timing. Saves 0.42 ms on a 3090 and 0.50 ms on an A100.
- **Fused elementwise kernels:** `torch.cuda.jiterator` kernels written with round-to-nearest intrinsics, so they are
  bit-identical to the unfused graph. They are compiled at load, outside the timed region.
- **Output transport by value:** the evaluation harness receives our outputs from another process. The outputs are a
  tensor subclass that is pickled as plain values, not through PyTorch's shared-memory file descriptors. This saved
  about 0.7 ms/step on the leaderboard.
- **Other transport paths**, inactive where they don't apply: a direct path for CUDA inputs, and a warm-up of
  PyTorch's inter-process machinery while the model is built.

## 8. What mattered
Changes in the local composite score on the held-out LCO folds (§1), measured when each change was added
(increments, not a clean ablation):

| Change | fold0 | Proxies for unseen conditions (foldR, Rx, beyond-range; §1) |
|---|---|---|
| Sim pretraining | +0.18 | foldR +0.06, Rx +0.18 |
| Soup of 4 | +0.075 | |
| 2nd ensemble member | +0.09 | |
| 3rd member + fp16 core | +0.04 | |
| Interval centre blend toward the raw ensemble output | +0.09 | foldR +0.11 |
| Adaptive width / energy blend | +0.03 | beyond-range +0.05 |
| 4-seed energy net | +0.04 | |
| LOWKE | +0.03 | |
| LSK | +0.04 | foldR +0.03, Rx +0.03 |

On the leaderboard, the latency work took the step from 7.8 to 4.8 ms. Every change was selected on one fold,
evaluated once on the other held-out folds, independently reviewed, and verified on the exact uploaded archive.

## 9. Compute
- Run behind submission 948041: 1 × RTX 3090 24 GB, AMD Ryzen 9 3950X (16 cores), 128 GB RAM.
  - Full chain from scratch: 2 h 04 min, plus 24 min for the energy-net refit step.
- The same chain inside the official image on 1 × A100-SXM4 40 GB: 2 h 27 min. `train()` of the submitted image on
  the same GPU type with 16 cores: 2 h 17 min.
- Peak GPU memory: 7.1 GiB on the A100 (README "Notes"). Peak host RAM: 13.7 GB. Each process is capped at a
  fraction of the card, with at most three concurrent processes.

## 10. References
- SimVP / gSTA (Gao et al., 2022; Tan et al., 2022, OpenSTL)
- model soups (Wortsman et al., 2022)
- RevIN-style instance normalisation (Kim et al., 2022)
- recursive least squares with forgetting (Haykin, *Adaptive Filter Theory*)
- Packed-Ensembles (Laurent et al., 2023)
- RealPDEBench (arXiv:2601.01829)
- PyTorch 2.2 (CUDA graphs, jiterator)
- The official RealPDE starting kit.
- Full write-up with evidence tables: `realpde/docs/METHOD.md`.
