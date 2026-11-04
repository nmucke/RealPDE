"""D2/fold0_a -- "free-rider bundle" ladder, rung (a): S3 loss only.

Copied from experiments/j1_simvp_s3/fold0/submission.py (S3-trained n1_simvp,
checkpoint + refit prior.npz/ke_net_flow.pt UNCHANGED -- those artifacts are
byte-identical copies, no refit needed since neither the backbone nor what
ke_net_flow is trained to predict changes anywhere in the D2 ladder).

Three NEW, config-gated, default-OFF code blocks are added below (present in
every rung of experiments/d2_bundle/fold0_{a,b,c,d}/submission.py identically;
only CONFIG values differ between rungs, so rung (a) with all three configs at
their off defaults is bit-identical in behaviour to the J1 S3 pipeline this
file was copied from):

  * `CONFIG["e_blend_w"]` (H1 energy-target blend, experiments/h1_tke_map/
    notes.txt STAGE 0a/0d, scripts/h1_stage0.py/h1_stage0d.py): None = off
    (the ke_flow net's raw Khat is used, as shipped). A float `w` blends the
    energy target used for the output's temporal-variance imposition,
    `e_tgt = w*Khat + (1-w)*R`, with `R` a NEW causal running (plain 1/n,
    not EMA) mean of REVEALED target ke maps, seeded at window 0 by the
    input window's own energy `kx` -- exactly `scripts/h1_stage0.py`'s
    causal `R_k` candidate and `h1_stage0d.py`'s blend, now computed ONLINE
    inside the wrapper instead of via the offline closed-form rescore those
    scripts used on an already-dumped prediction.
  * `CONFIG["mu_vp"]` (H1 variance-preserving mu, same notes.txt STAGE
    0c(prompt)): False = off (mu added as-is, as shipped -- the bias map's
    own zero-mean-over-t fluctuation silently adds uncounted variance on top
    of the imposed energy target). True demeans mu (`mu' = mu - mean_t mu`)
    and rescales `fluc` by the closed-form `c` that restores
    `var_t(c*fluc + mu') == var_t(fluc)` (the already-achieved, imposition-
    exact target), before adding `mu' + mean_t(mu)` back (algebraically the
    same `mu`).
  * `CONFIG["sig_sched"]` (E1 window-0 width, experiments/e1_pipeline_cold_start/
    e1_minimal.diff, ported verbatim -- same tuple semantics, same `bounds()`
    call site pattern in this file's `GenericCtxNormWrapper`): None = off
    (shipped). A tuple widens the interval sigma by `sig_sched[min(k, len-1)]`
    at window index `k`; the diff's fold0-TRAIN optimum `(1.15, 1.0)` is used
    here unchanged.

Everything else in this file (docstring below included) is
experiments/j1_simvp_s3/fold0/submission.py's own copied text, reproduced
verbatim for context.

---

J1/fold0 -- does C11-A's n1_simvp backbone ADD to I1's S3 mixed loss?
Copied from experiments/c11a_backbones2/fold0_n1simvp/submission.py (family
rule: copy, swap ONLY the checkpoint) with exactly one change: `CONFIG
["unet_ckpt"]` repointed to `data/checkpoints/j1/n1simvp_s3_fold0.pth`, the
n1_simvp checkpoint trained with I1's S3 loss instead of plain Welsch
(`scripts/j1_finetune.py`, see experiments/j1_simvp_s3/notes.md). `prior.npz`/
`ke_net_flow.pt` next to this file are refit for THIS checkpoint on fold0
TRAIN only, using `scripts/c11a_fit_prior.py`/`scripts/c11a_fit_ke_flow.py`
UNEDITED (both already generic over `--model`/`--ckpt`).

Everything below (docstring included) is C11-A's own copied text, reproduced
verbatim for context -- see experiments/c11a_backbones2/fold0_n1simvp/
submission.py for that version's own history.

---

C11-A/fold0_n1simvp -- the c7a/c4r1 ctx-norm pipeline with the N1 (SimVP-style)
backbone (`src/c11a_models.py`) swapped in for `unet_local` inside the SAME
`GenericCtxNormWrapper` streaming contract, on LCO fold0.

Copied from experiments/c7a_robust_centre/fold0_r3c_single/submission.py
(family rule: copy, swap ONLY the backbone construction in `get_ttt_model`)
with exactly this change:

  * `get_ttt_model` builds `c11a_models.GenericCtxNormWrapper` around a
    `GenericCtxNormCore(build_core("n1_simvp"))` instead of
    `c4r1_models.CtxNormWrapper`/`CtxNormUNet`/`unet_local` -- loading the
    `scripts/c11a_finetune.py` checkpoint named by `CONFIG["unet_ckpt"]`
    (env `C3B_UNET_CKPT` still wins, `_resolve_unet_ckpt` reused unchanged).
    `GenericCtxNormWrapper` has the identical `reset()`/`step()`/`forward()`
    contract as `CtxNormWrapper` (see `src/c11a_models.py` docstring), so the
    single `raw = self.base(x)` call site inside `GainMapImprovedModel.step`
    and `ExperimentTTT.reset_ttt_state`'s `self.model.base.reset()` call need
    no change at all.
  * `prior.npz` / `ke_net_flow.pt` next to this file are refit for the N1
    backbone on fold0 TRAIN only (`scripts/c11a_fit_prior.py`,
    `scripts/c11a_fit_ke_flow.py` -- thin copies of the c4r1 fit scripts with
    only their wrapper-construction helper swapped).

Everything below this point (docstring included) is the copied file's own
unmodified text describing the rest of the pipeline; reproduced verbatim for
context -- see experiments/c7a_robust_centre/fold0_r3c_single/submission.py
for the original.

---

c4r1_ctxnorm/fold0 -- the c3b UNet `integrated` pipeline with a CAUSAL
in-context-standardised (RevIN-style) backbone (`src/c4r1_models.py`) instead
of the plain `unet_local`, on LCO fold0.

Copied from experiments/c3b_unet_pipeline/fold0/submission.py (family rule:
"copy ... with only the backbone construction swapped and reset_ttt_state
also clearing the wrapper state") with exactly those two changes:

  * `get_ttt_model` builds `c4r1_models.CtxNormWrapper` (a stateful wrapper
    around `c4r1_models.CtxNormUNet`, itself a `unet_local` with a 4-channel-
    extended stem) instead of the plain `unet_local`, loading the
    `scripts/c4r1_finetune.py` checkpoint named by `C3B_UNET_CKPT` (env,
    `_resolve_unet_ckpt` reused unchanged -- it is just a path resolver) /
    `CONFIG["unet_ckpt"]`. The wrapper's `forward(x)` already does
    accumulate-then-predict internally (see its docstring), so the single
    `raw = self.base(x)` call site inside `GainMapImprovedModel.step` (S1 of
    the original c3b docstring below) needs no change at all.
  * `ExperimentTTT.reset_ttt_state` additionally calls the wrapper's own
    `reset()` (clears its causal S1/S2/n) next to `self.model.reset()` (the
    pipeline's own online-adaptation state) -- both are trajectory-scoped.
  * `prior.npz` / `ke_net_flow.pt` next to this file are refit for the
    ctx-norm backbone on fold0 TRAIN only (`scripts/c4r1_fit_prior.py`,
    `scripts/c4r1_fit_ke_flow.py` -- STATEFUL adaptations of the c3b fit
    scripts, since the frozen forecast is now causal/sequential, not a
    chunked one-shot forward; see experiments/c4r1_ctxnorm/notes.md).

Everything below this point (docstring included) is c3b's own unmodified
text describing the rest of the pipeline (advect/gain/bias/spread/ke-net/
bounds), which this task did not touch -- reproduced verbatim for context:

Copied from experiments/lco_baseline/fold0/submission.py (itself an unmodified
copy of experiments/integrated/submission.py except the path-depth fix) with
ONLY the backbone construction changed:

  * `get_ttt_model` builds `c1d_models.build("unet_local")` instead of
    `load_baseline.build_model("fno")`, and loads the checkpoint named by
    `C3B_UNET_CKPT` (env) / `CONFIG["unet_ckpt"]` (single config point, per
    task instruction -- a better UNet checkpoint drops in by changing one
    string). Default: `data/checkpoints/c1d/unet_local_fold0.pth` (recipe B on
    fold0 TRAIN, but sim-pretrained for only 1500 it -- see
    experiments/c1d_capacity/notes.md and experiments/c3b_unet_pipeline/notes.md).
  * `prior.npz` / `ke_net_flow.pt` next to this file are refit for this UNet
    checkpoint on fold0 TRAIN only (`scripts/c3b_fit_prior.py`,
    `scripts/c3b_fit_ke_flow.py` -- thin adaptations of the `lco_fit_*`
    scripts that swap the FNO builder for the UNet one; see notes.md).
  * The ke_net_flow features (`src/ke_net_flow.py:features`) are computed
    purely from the input window, the backbone's OWN predicted fluctuation and
    energies -- no FNO-internal activations are consumed anywhere in the
    downstream pipeline -- so no feature-source change was needed beyond
    swapping which backbone produces `fl_f` / `e_f`.

Nothing else in the pipeline logic (advect/blend, gain rule, online mean
correction, decoupled interval centre, bounds) was touched. See
`experiments/c3b_unet_pipeline/fold0_v1/submission.py` for V1 (energy net off,
== `experiments/c2_diagnostics/stage_b_meanonly`'s CONFIG diff applied here).

Original header (experiments/integrated/submission.py) follows, unmodified.

---

The merge of the two 2026-09-16 tracks, on top of experiments/ke_net:

  * experiments/knobs_honest: the online window-mean correction
    (`mean_mode = "both"`, `mean_ridge = 0.25`) and the two-vector interval
    centre basis (`ctr_basis2`). Every other knob of that fork keeps the
    default that reproduces ke_net bit-for-bit.
  * experiments/tke_gate: the learned flow-field energy net
    (`ke_flow = "ke_net_flow.pt"`) REPLACES `ke_net`'s energy correction, so
    `ke_net = None` and `ke_net_flow.pt` ships instead of `ke_net.pt`.

Both branches are residual: with `ke_flow = None`, `mean_mode = None` and
`ctr_basis2 = False` this file is experiments/ke_net bit-for-bit (verified in
notes.md), so the ke_net numbers remain the reference for every row here.

The energy net is fitted on the *features of the backbone it runs on*
(`src/tke_energy_fit.py --ckpt <backbone>`, ~20 min): refitting it is worth
+1.5 tke / +0.4 sps on clean test, so `ke_net_flow.pt` must be re-fitted
whenever `model.pth` changes (notes.md, "Re-packaging with a new backbone").

Original header follows.

gain_map_improved -- RealPDE Track 2 LTTTA submission.

Scaffolded by scripts/create_experiment.py (forked from experiments/gain_map). This
file is the experiment: CONFIG below is the whole configuration, and it is what
ships in the zip.

Runs in two places unchanged:
  * in place, `python src/local_eval.py --submission experiments/gain_map_improved \
        --data starting_kit/example_data` (kit modules resolved from src/,
    checkpoint resolved from data/competition/baseline_checkpoints/);
  * from the packaged zip, where scripts/create_submission.py has placed the
    imported kit modules and `model.pth` next to this file.

Contract (docs/track2.md): every tensor is (1, 20, 32, 64, 3) in normalized
space, `ttt_step` and `reset_ttt_state` are timed, and `info["lower"]` /
`info["upper"]` are all-or-nothing for the entire run.

Idea: gain_map's three online per-pixel maps (fluctuation gain, bias, spread),
plus two things the stream itself can tell us -- the wake's propagation speed
and the flow parameters -- used where each provably helps:

  * propagation speed -> the *shape* of the fluctuation. The vortex street
    convects downstream at Uc px/frame, read off every input window as the
    shift that best maps frame t onto frame t+tau (grid scan, running average
    over the trajectory; 0.80 px/frame = 0.74 U_inf at Re 11k / AoA 0, 1.36 =
    0.60 U_inf at Re 24k / AoA 15). Taylor's hypothesis makes the last input
    frame, shifted by Uc (k+1), a forecast of lead k with the right *phase*:
    its correlation with the truth is 0.94 at lead 0 against 0.75 for the FNO,
    and it decays with lead time -- slowly at low Re (0.4 at lead 19), within
    two frames at high Re. It exists only where the source column lies
    downstream of the airfoil / laser shadow. The fluctuation shape is
    a * (F(x) - mean_t F) + b * advected with (a, b) per lead and channel the
    ridge least-squares fit to the revealed targets (per-element sufficient
    statistics, pooled separately over pixels with and without an advected
    value), then rescaled per pixel and window to the energy below.
  * energy -> gain_map's rule, untouched: per (pixel, channel) the output's
    temporal variance this window is var_t(F') times the running ratio of the
    input's energy to the forecast's, to the power 0.75. Measured on the two test
    trajectories, the FNO's window-to-window energy modulation tracks the true
    TKE better than anything the propagation speed offers (the advected
    pattern's energy adds nothing at high Re), and the per-window TKE varies by
    ~50 %, so the energy is not the blend's to set: the least-squares blend is
    rel_l2-optimal but zeroes the fluctuation where nothing predicts the phase,
    which is exactly where TKE still needs the energy. Splitting shape and
    energy keeps the TKE map identical to gain_map's and improves the phase
    where it can.
  * parameters -> the starting point of the running statistics. (Re, AoA) are
    read off the first input window (src/parameter_estimation.py) and select,
    by a Gaussian kernel in (AoA, log Re), the training files whose converged
    maps (mean innovation, residual variance, input and forecast energies;
    scripts/fit_gain_prior.py) seed the bias, spread and gain with the weight
    of a few windows, so the first windows of every trajectory -- where
    gain_map's statistics start from nothing -- are no longer the worst.
  * bias / spread: gain_map's running mean and pooled variance of the
    innovation; SPS interval at the closed-form score-optimal width.

This fork (ke_net) adds two ways of improving that one number -- the per-(pixel,
channel) energy the fluctuation is rescaled to -- and nothing else:

  * `ke_net`: an offline-trained residual correction to the energy rule
    (src/ke_net.py). The net predicts a log-space correction on top of
    log[var_t(F') (runx/runf)^0.75], with its last layer zero-initialised, and is
    fitted on data/real/train (selected on data/real/val, test never touched)
    with the tke metric itself as the loss. `ke_net_scale = 0` is the shipped
    rule bit-for-bit. `ke_net_online` additionally takes a few Adam steps per
    revealed window on the just-observed (features -> true ke map) pair.
  * `er_online`: the online-tuned energy rule. A small candidate grid of
    multiplicative corrections (amplitude A, modulation exponent q) is scored on
    every revealed window by the *exact* realised TKE the candidate would have
    produced, and the running argmin is used for the next window. With `ke_net`
    on, the candidates multiply the net's energy target rather than the rule's.

It also decouples the interval's centre from the returned prediction (`ctr`, from
sps_ideas/sps_3). Only `pred` enters the accuracy factors, so the bounds may be
centred anywhere -- and the shipped `pred` is deliberately energy-inflated for TKE,
which makes the innovation the intervals are built from louder than it needs to be.
So carry a second centre `const + g * basis`, with `const` the FNO/blend mean and
`basis` the un-inflated least-squares fluctuation (the blend shape *before* the
energy rescaling), and emit `centre +- k*(sigma2) sigma2` while returning `pred`
untouched. Writing `a = y - const`, every member's innovation is `a - g basis`, so
one pair of running means (M0, M1) and three pooled second moments (Saa, Sab, Sbb)
give the bias `M0 - g M1` and the variance `Saa - 2 g Sab + g^2 Sbb` of *every* `g`
in closed form -- including the per-(pixel, channel) least-squares `g* = Sab/Sbb`,
which minimises sigma2 and hence (the interval factor being monotone decreasing in
sigma) maximises SPS. `ctr = False` is the pred-centred baseline.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

CONFIG = {
    "base": "unet_local",      # c3b: src/c1d_models.py MODEL_NAMES entry (was "fno")
    # c3b: SINGLE config point for the backbone checkpoint -- swap in a better
    # UNet checkpoint by changing this string (or setting env C3B_UNET_CKPT,
    # which wins). Relative paths resolve against the repo root.
    # c7a S-B1: single backbone = R3 (Welsch-trained), used for BOTH pred and ctr.
    # Everything else in this file is experiments/c4r1_ctxnorm/fold0_meanoff/
    # submission.py verbatim (this is a pure checkpoint swap; prior.npz /
    # ke_net_flow.pt refit for this checkpoint, see experiments/c7a_robust_centre/
    # notes.md S4b). env C4R1_CTX_CKPT still wins if set.
    "unet_ckpt": "data/checkpoints/d2/n1simvp_s3_foldR.pth",
    "ckpt": "sim_real_fno_fp16.pth",            # unused by c3b's get_ttt_model; kept for header parity
    "backbone_ckpt": None,         # dev-only override (see _resolve_ckpt); env RPDE_BACKBONE_CKPT wins
    "adapt": False,                # kept for the notes.md summary: no gradient TTT here
    # --- propagation-speed fluctuation source ---
    "advect": True,                # blend in the Taylor-advected last input frame
    "uc_max": 4.0,                 # px/frame: top of the convection-speed scan grid
    "uc_step": 0.1,                # scan resolution (parabolic refinement on top)
    "uc_taus": (2, 3),             # frame lags the scan correlates over
    "uc_fixed": None,              # px/frame: bypass the scan with a constant (diagnostics)
    "uc_warm": 4,                  # scan every window until this many are seen ...
    "uc_every": 4,                 # ... then every uc_every-th window (the speed is a stable running average)
    "blend": "region",             # "region": (a, b) per lead & channel, split valid/invalid | "pixel": per element on top
    "blend_ridge": 1.0,            # ridge toward (a, b) = (1, 0) [gain_map], relative to the local energies; 0 = pure LS
    "blend_pixel_n0": 8.0,         # "pixel" only: pseudo-windows of ridge toward the region solution (fades as 1/n)
    "blend_smooth": True,          # "pixel" only: 3x3 box-smooth the sufficient statistics before solving
    # --- gain / bias / spread ---
    "gain": True,                  # gain_map's rule: per-window energy = var_t(F') * (E_input / E_F)^power, running per pixel
    "gain_power": 0.75,            # gamma = ratio^(power/2); < 1 shrinks the gain toward 1
    "gain_max": 5.0,               # clip gamma to [1/gain_max, gain_max]
    "shape_cap": 2.0,              # blend shape scaled up to this factor to reach the window energy; F' fills the rest
    # --- learned TKE-map predictor (src/ke_net.py, scripts/fit_ke_net.py) ---
    #   log E_out = log[ ef (runx/runf)^gain_power ] + ke_net_scale * net(features)
    # The net's last layer is zero-initialised and it is trained as a residual on the
    # rule, so ke_net = None (or ke_net_scale = 0) is gain_map_improved bit-for-bit.
    "ke_net": None,            # integrated: off -- the flow net below replaces it
    "ke_net_scale": 1.0,           # shrink knob on the log correction; 0 = the rule, 1 = the fit
    "ke_net_clip": 1.5,            # cap on |log correction| (a factor 4.5 in energy)
    "ke_net_online": False,        # False | int: Adam steps per revealed window on the
                                   # previous window's features vs the revealed ke map.
                                   # Timed; reset_ttt_state restores the offline weights.
    "ke_net_online_lr": 1e-4,      # lr of those steps
    # --- learned flow-field correction to the energy (src/ke_net_flow.py) ---
    # When set, the flow net's energy map REPLACES ke_net's for the u, v channels
    # (both are residuals on the same rule, so applying both would double-correct);
    # `ke_flow_blend` < 1 takes a geometric blend in log-energy space,
    #   log E_out = (1 - blend) * log E_ke_net + blend * log E_ke_flow.
    "ke_flow": "ke_net_flow.pt",   # src/ke_net_flow.py weights next to this file
    "ke_flow_scale": 1.0,          # shrink knob on the flow net's log correction
    "ke_flow_blend": 1.0,          # 1 = the flow net alone, 0 = ke_net alone
    # --- online-tuned energy rule (idea_1), applied on top of the above ---
    #   E_out = E_base * A * (kf/runf)^(q-1) * (kx/runx)^w
    # with E_base the rule's energy, or the net's when ke_net is on -- so the candidate
    # family is always "amplitude x modulation-shrinkage around the current energy
    # target", and (q, w, A) = (1, 0, 1) leaves that target untouched. (q, w, A) is
    # chosen online per trajectory by the realised score on the revealed windows.
    "er_online": False,            # False = whatever the energy target already is
    "er_q": (0.7, 0.85, 1.0),      # exponent on the FNO's window-to-window energy modulation
    "er_w": (0.0,),                # weight on the input window's own modulation (persistence)
    "er_a": (1.0, 1.05, 1.1),      # global amplitude. Capped at 1.1 on purpose: letting the
                                   # search reach 1.15-1.2 buys ~0 extra TKE and costs
                                   # 0.2-0.6 SPS, because a louder prediction means a louder
                                   # innovation and hence a wider optimal interval.
    "er_warm": (1.0, 0.0, 1.0),    # used until the first target is revealed
    "er_obj": "tke",               # "tke" = ke-map rel-L2 only | "both" = + er_lam * field rel-L2
    "er_lam": 1.4,                 # ratio of the two subscores' d(score)/d(error) at their operating points
    "er_exact": True,              # score candidates on the reconstructed output ke (incl. the bias
                                   # term's own temporal variance); False = the bare energy rule
    # --- (Re, AoA)-conditioned prior for the running statistics ---
    "prior": True,                 # start mu / var / energies from prior.npz maps if the file exists
    "prior_windows": 6.0,          # pseudo-windows the prior counts for before the stream takes over
    "prior_sigma_aoa": 2.5,        # deg: kernel width over the training files' AoA
    "prior_sigma_re": 0.15,        # log-Re kernel width (0.15 ~ 16 % in Re, one grid step is ~10 %)
    "bias": True,                  # add the running mean innovation
    "bounds": True,                # SPS intervals from the running innovation variance
    "pool_t": True,                # innovation variance pooled over the 20 frames: True | False | "sep" (pooled x lead profile)
    "alpha_min": 0.05,             # floor on the running-average gain max(1/(n+prior_n), alpha_min)
    "prior_n": 1.0,                # pseudo-count of the initial statistics
    "k_scale": 1.0,                # multiplier on the optimal interval half-width

    # --- decoupled interval centre (sps_ideas/sps_3) ---
    # Only `pred` enters the SPS accuracy factors; the bounds may be centred anywhere.
    # The shipped `pred` is energy-inflated for TKE, which makes its innovation louder
    # than necessary. When `ctr` is on we carry a SECOND, un-inflated centre with its
    # own running bias and innovation variance, and put the interval around that while
    # returning `pred` unchanged -- so rel_l2 / tke / mvpe are bit-identical either way.
    "ctr": True,                   # False = baseline (bounds centred on pred)
    "ctr_w": 1.0,                  # weight of the un-inflated LS blend shape zc vs the shipped fluc
    "ctr_g": "auto_px",            # float shrink on the centre's fluctuation (0 = FNO mean only),
                                   # or "auto" / "auto_px": the running least-squares gain,
                                   # pooled over the field / per (pixel, channel)
    "ctr_g_ridge": 0.1,            # pseudo-weight pulling the online gain toward 1 (val-selected)
    "ctr_g_clip": (0.0, 1.25),     # clamp on the online gain
    "ctr_bias": True,              # give the centre its own running bias map
    "ctr_k_scale": 1.0,            # multiplier on the centre's optimal half-width

    # --- knobs_honest: every default below reproduces ke_net bit-for-bit -------
    # (1) prior life: how long the (Re, AoA) prior outweighs the stream
    "prior_windows_e": None,       # separate pseudo-count for the energy statistics (None = prior_windows)
    "prior_windows_c": None,       # separate pseudo-count for the centre moments (None = prior_windows)
    # (2) convection-scan instability guard
    "uc_guard": None,              # None | "median" | "shrink" | "both"
    "uc_guard_k": 4,               # most recent scans the stability test looks at
    "uc_guard_cv": 0.25,           # median-absolute-deviation / median above which the scan is unstable
    "uc_guard_peak": 0.0,          # scan peak correlation below which the scan is "flat" (0 = off)
    "uc_guard_w": 0.0,             # weight kept on the Taylor branch while unstable
    # (3) mean / bias predictor
    "mu_alpha_min": None,          # gain floor of the bias map only (float or per-channel); None = alpha_min
    "mean_mode": None,             # c4r1 meanoff variant: online mean/bias correction OFF (was "both")
    "mean_ridge": 0.25,            # pseudo-weight toward "no correction" (slope) / toward the FNO mean (ls)
    "mean_clip": 1.0,              # clamp on the slope correction coefficient
    # (4) interval centre basis
    # (5) incoherent-band energy floor (experiments/gain_map_improved/tke_ideas/idea_4.md)
    "band_split": None,            # None = off; int kc = cut the fluctuation at DFT mode kc
    "hi_scale": 2.0,               # the injected high-band energy, in units of the input window's own
    "ctr_lead": False,             # per-lead centre gain (second moments not pooled over t)
    "ctr_basis2": True,            # two-vector basis (blend shape, advected fluctuation), 2x2 per-pixel LS
    "seed": 0,

    # --- D2 free-rider bundle: H1 energy-target blend / variance-preserving mu,
    # E1 window-0 width (experiments/d2_bundle/notes.md). All default OFF; see
    # the module docstring for the exact semantics of each knob.
    "e_blend_w": 0.85,              # D2 rung (c): S3 + H1 blend w=0.85 (fold0-TRAIN optimum, h1_stage0d.py)
    "mu_vp": True,                  # D2 rung (c): variance-preserving mu (H1, h1_tke_map notes STAGE 0c(prompt))
    "sig_sched": None,             # None = off | tuple of per-window-index sigma multipliers, last held
}

# Official train_real per-channel statistics (mean_std_real.pt). The evaluator
# normalises the input with the *input* stats and the target with the *target*
# stats; SPS measures interval width against SIGMA_GLOBAL in physical units.
_MEAN_IN = (0.15496086, -0.00051399, 0.0)
_STD_IN = (0.09680565, 0.01596068, 1.0)
_MEAN_TGT = (0.15496257, -0.00051779, 0.0)
_STD_TGT = (0.09681041, 0.01596364, 1.0)
_SIGMA_GLOBAL = 0.0563870259

_HERE = os.path.dirname(os.path.abspath(__file__))
# NOTE (lco_baseline): this file is an unmodified copy of experiments/integrated/submission.py
# (commit ca20bb5), except for this one path-depth fix. The original assumes it lives at
# <repo>/experiments/<name>/ (two levels under the repo root) and walks up two pardirs; this
# copy lives at <repo>/experiments/lco_baseline/fold0/ (three levels), so it needs a third.
# No pipeline/model logic changed -- see experiments/lco_baseline/notes.md.
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))

# Packaged: load_baseline.py / rpde_baselines/ sit next to this file. Dev: they
# live in the repo's src/. _HERE is inserted last so the packaged copy wins.
for _p in (os.path.join(_REPO, "src"), _HERE):
    if os.path.isfile(os.path.join(_p, "load_baseline.py")) and _p not in sys.path:
        sys.path.insert(0, _p)

import ke_net as K  # noqa: E402
import ke_net_flow  # noqa: E402
import c11a_models as _C11A  # noqa: E402  (C11-A: generic backbone + ctx-norm wrapper)
from load_baseline import load_checkpoint_state  # noqa: E402
from parameter_estimation import estimate_aoa, estimate_re  # noqa: E402


def _resolve_unet_ckpt(submission_dir: str) -> str:
    """c3b: the single config/env point for the UNet backbone checkpoint.

    ``C3B_UNET_CKPT`` (env) wins over ``CONFIG["unet_ckpt"]``; both formats
    ``load_checkpoint_state`` knows work (this repo's ``{"model_state_dict":
    ...}`` training-checkpoint layout, or a raw/fp16-packed state dict).
    Relative paths resolve against the repo root; a named-but-missing path is
    an error, matching ``_resolve_ckpt``'s contract for the FNO checkpoint.
    ``submission_dir`` is accepted for interface parity with ``_resolve_ckpt``
    but unused: unlike the FNO pipeline, no packaged ``model.pth`` convention
    exists for this experiment yet (dev-only, never packaged for submission).
    """
    override = os.environ.get("C3B_UNET_CKPT") or CONFIG.get("unet_ckpt")
    if not override:
        raise FileNotFoundError("no UNet checkpoint configured: set C3B_UNET_CKPT "
                                "or CONFIG['unet_ckpt']")
    path = override if os.path.isabs(override) else os.path.join(_REPO, override)
    if not os.path.isfile(path):
        raise FileNotFoundError("UNet backbone checkpoint not found: %s" % path)
    return path


def _resolve_ckpt(submission_dir: str) -> str:
    """model.pth from the archive root, else the dev checkpoint tree.

    Development override: ``RPDE_BACKBONE_CKPT`` (or ``CONFIG["backbone_ckpt"]``)
    names a checkpoint to load instead, absolute or relative to the repo root.
    Both formats that ``load_checkpoint_state`` knows work (fp16-packed and the
    kit's fp32 ``model_state_dict`` layout), so a locally fine-tuned backbone can
    be streamed through the unmodified pipeline. Unset in a packaged submission,
    so shipping behaviour is untouched; a named-but-missing path is an error.
    """
    override = os.environ.get("RPDE_BACKBONE_CKPT") or CONFIG.get("backbone_ckpt")
    if override:
        path = override if os.path.isabs(override) else os.path.join(_REPO, override)
        if not os.path.isfile(path):
            raise FileNotFoundError("backbone override not found: %s" % path)
        return path
    packaged = os.path.join(submission_dir, "model.pth")
    if os.path.isfile(packaged):
        return packaged
    for sub in ("sim_real_ft", "sim_pretrain"):
        dev = os.path.join(_REPO, "data", "competition", "baseline_checkpoints",
                           sub, CONFIG["ckpt"])
        if os.path.isfile(dev):
            return dev
    raise FileNotFoundError(
        "no model.pth in %s and no %s under data/competition/baseline_checkpoints/"
        % (submission_dir, CONFIG["ckpt"]))


def _load_prior(device: str):
    """prior.npz next to this file (scripts/fit_gain_prior.py), as device tensors, or None.

    Per training file: mean innovation (physical, fp16), pooled residual variance and
    the input / forecast energies, plus its (Re, AoA). Converted here to the
    normalised spaces the online statistics live in (target stats for the
    innovation and the forecast energy, input stats for the input energy).
    """
    path = os.path.join(_HERE, "prior.npz")
    if not os.path.isfile(path):
        return None
    import numpy as np
    z = np.load(path)
    std_t = torch.tensor(_STD_TGT[:2]); std_i = torch.tensor(_STD_IN[:2])
    pad = lambda t: torch.cat([t, torch.zeros_like(t[..., :1])], dim=-1)   # p channel
    return {
        "re": torch.tensor(z["re"], dtype=torch.float32, device=device),
        "aoa": torch.tensor(z["aoa"], dtype=torch.float32, device=device),
        "mu": pad(torch.tensor(z["mu"].astype(np.float32)) / std_t).to(device),            # (F, T, H, W, 3)
        "var": pad(torch.tensor(z["var"]) / std_t ** 2).unsqueeze(1).to(device),         # (F, 1, H, W, 3)
        "e_true": pad(torch.tensor(z["e_true"]) / std_i ** 2).unsqueeze(1).to(device),
        "e_pred": pad(torch.tensor(z["e_pred"]) / std_t ** 2).unsqueeze(1).to(device),
    }


class _KeFlow:
    """The learned flow-field correction to the energy rule (src/ke_net_flow.py).

    Lifted unchanged from experiments/ke_net_flow/submission.py: the net sees the last
    input frames and the FNO fluctuation as scale-free fields, so it can resolve *where*
    the wake is and what state it is in, which no summary of the energy can. The head is
    zero-initialised and the parameterisation is residual, so `scale = 0` is the rule.

    Holds the per-trajectory state its features need (the previous window's energy maps
    and the window counter), cleared by `reset`.
    """

    def __init__(self, net, cfg: Dict[str, Any], scale: float):
        self.net, self.cfg, self.scale = net, cfg, float(scale)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.ex_prev = self.ef_prev = None

    def __call__(self, x, fl_f, e_x, e_f, runx, runf, mask):
        """Per-(pixel, channel) target energy of the output fluctuation, (1, 1, H, W, 3)."""
        c = self.cfg
        ex, ef = e_x[:, 0, ..., :2], e_f[:, 0, ..., :2]
        xw, ffl = x[..., :2], fl_f[..., :2]
        rx, rf = runx[:, 0, ..., :2], runf[:, 0, ..., :2]
        halves = None
        if c["summary"]:
            halves = (xw[:, :10].var(1, unbiased=False), xw[:, 10:].var(1, unbiased=False),
                      ffl[:, :10].var(1, unbiased=False), ffl[:, 10:].var(1, unbiased=False))
        wfrac = torch.full((1,), min(self.n, 20) / 20.0, device=x.device, dtype=x.dtype)
        feat = ke_net_flow.features(
            xw, ffl, ex, ef, rx, rf, mask[None], wfrac,
            k_in=c["k_in"], leads=c["leads"], summary=c["summary"], halves=halves,
            ex_prev=self.ex_prev, ef_prev=self.ef_prev)
        if c.get("lead_e") or c.get("phase"):
            # nets fitted by src/tke_energy_fit.py carry the extra state channels
            import tke_energy_feats  # noqa: PLC0415  (diagnostic nets only)
            feat = torch.cat([feat, tke_energy_feats.extra_features(
                xw, ffl, ef, mask[None], bool(c.get("lead_e")), bool(c.get("phase")),
                int(c.get("n_blocks", 4)), c.get("phase_kind", "probe"))], 1)
        e_new = ke_net_flow.apply_correction(
            self.net(feat), ke_net_flow.rule_log_energy(ef, rx, rf), mask[None], self.scale)
        self.ex_prev, self.ef_prev, self.n = ex, ef, self.n + 1
        out = e_f.clone()                       # p channel keeps the rule's value
        out[:, 0, ..., :2] = e_new
        return out


def _load_ke_flow(cfg: Dict[str, Any], device: str):
    """The trained flow net named by CONFIG['ke_flow'], next to this file, or None."""
    name = cfg.get("ke_flow")
    if not name or float(cfg.get("ke_flow_scale", 1.0)) == 0.0:
        return None
    path = name if os.path.isabs(name) else os.path.join(_HERE, name)
    if not os.path.isfile(path):
        raise FileNotFoundError("ke_flow weights not found: %s" % path)
    net, ncfg = ke_net_flow.load_ke_flow_net(path, device)
    return _KeFlow(net, ncfg, cfg["ke_flow_scale"])


def _load_ke_net(device: str, cfg: Dict[str, Any]):
    """The trained TKE-map predictor next to this file (scripts/fit_ke_net.py), or None.

    Resolved exactly like prior.npz, so scripts/create_submission.py packages it with the
    rest of the experiment. A named-but-missing file is an error rather than a silent
    fallback to the rule: a submission that quietly loses its correction is worse than one
    that fails the preflight.
    """
    name = cfg.get("ke_net")
    if not name or not float(cfg["ke_net_scale"]):
        return None
    return K.load_ke_net(os.path.join(_HERE, name), device)


# --------------------------------------------------------------------------- #
# SPS-optimal interval width for a Gaussian error.
# --------------------------------------------------------------------------- #
def _optimal_k_table(n: int = 512, a_min: float = 1e-3, a_max: float = 1e2):
    """k*(a) maximising exp(-a k) * erf(k / sqrt 2), a = 2 sigma_phys / SIGMA_GLOBAL.

    For an element with Gaussian error of std sigma and a symmetric interval of
    half-width k sigma, the expected SPS element score is
    P(inside) * exp(-width / SIGMA_GLOBAL) = erf(k/sqrt2) * exp(-a k). The
    maximiser solves a erf(k/sqrt2) = sqrt(2/pi) exp(-k^2/2); solved once by
    bisection on a log grid of `a` (untimed, at construction).
    """
    log_a = torch.linspace(math.log(a_min), math.log(a_max), n, dtype=torch.float64)
    ks = []
    for a in log_a.exp().tolist():
        lo, hi = 0.0, 10.0
        for _ in range(60):
            k = 0.5 * (lo + hi)
            g = a * math.erf(k / math.sqrt(2.0)) - math.sqrt(2.0 / math.pi) * math.exp(-0.5 * k * k)
            if g > 0:
                hi = k
            else:
                lo = k
        ks.append(0.5 * (lo + hi))
    return log_a.float(), torch.tensor(ks, dtype=torch.float32)


def _ke(x: torch.Tensor) -> torch.Tensor:
    """Per-pixel, per-channel temporal variance of a (1, T, H, W, C) window: (1, 1, H, W, C)."""
    return x.var(dim=1, keepdim=True, unbiased=False)


# --------------------------------------------------------------------------- #
# Streamwise shift with linear interpolation (Taylor's hypothesis).
# --------------------------------------------------------------------------- #
class _Shifter:
    """Shift fields along W by a fractional number of columns, source-validity aware.

    A pixel at column c shifted by d takes its value from column c - d (linear
    interpolation). The value is valid when the whole path c - d .. c lies in
    the domain and downstream of the last masked pixel (airfoil body / laser
    shadow) upstream of c on that row, so nothing is advected through the
    airfoil or from outside the field of view.
    """

    def __init__(self, w: int, device: str):
        self.w = w
        self.cols = torch.arange(w, device=device, dtype=torch.float32)

    def last_obstacle(self, mask: torch.Tensor) -> torch.Tensor:
        """(H, W) column index of the last masked pixel at or upstream of each column, -1 if none."""
        idx = torch.where(mask, self.cols.expand_as(mask), torch.full_like(self.cols, -1.0).expand_as(mask))
        return torch.cummax(idx, dim=-1).values

    def __call__(self, f: torch.Tensor, d: float, obstacle: torch.Tensor):
        """f (..., H, W, C) shifted by d columns; returns (shifted, valid (H, W))."""
        src = self.cols - d
        i0 = torch.floor(src)
        wgt = (src - i0).view(-1, 1)
        i0 = i0.long()
        valid = (i0.view(1, -1) >= obstacle.long() + 1) & (i0 >= 0).view(1, -1)
        i0c = i0.clamp(0, self.w - 1)
        i1c = (i0 + 1).clamp(0, self.w - 1)
        out = (1.0 - wgt) * f[..., i0c, :] + wgt * f[..., i1c, :]
        return out, valid


class GainMapImprovedModel:
    """F(x) blended with the advected input, then online gain, bias and spread."""

    def __init__(self, base: torch.nn.Module, device: str, cfg: Dict[str, Any]):
        self.base = base.to(device).eval()
        self.device = device
        self.cfg = cfg
        self.log_a, self.k_tab = (t.to(device) for t in _optimal_k_table())
        self.std_tgt = torch.tensor(_STD_TGT, device=device)
        self.mean_in = torch.tensor(_MEAN_IN, device=device)
        self.std_in = torch.tensor(_STD_IN, device=device)
        self.mean_tgt = torch.tensor(_MEAN_TGT, device=device)
        # gain floor of the bias map only: scalar or per-channel, as a broadcastable map
        f = cfg.get("mu_alpha_min")
        self.mu_floor = None if f is None else torch.tensor(
            [float(f)] * 3 if not isinstance(f, (tuple, list)) else list(f) + [0.0] * (3 - len(f)),
            device=device).view(1, 1, 1, 1, 3)
        self.shift = _Shifter(64, device)
        self.uc_grid = torch.arange(0.0, float(cfg["uc_max"]) + 1e-6, float(cfg["uc_step"]), device=device)
        self.prior = _load_prior(device) if cfg["prior"] else None
        # the learned energy correction, plus the weights reset_ttt_state restores
        self.ke = _load_ke_net(device, cfg)
        self.ke_flow = _load_ke_flow(cfg, device)
        self.ke_steps = int(cfg["ke_net_online"] or 0) if self.ke is not None else 0
        self.ke_offline = ({k: v.detach().clone() for k, v in self.ke.state_dict().items()}
                           if self.ke_steps else None)
        self.ke_opt = None
        # online energy-rule candidates, as (K, 1, 1, 1, 1, 1) tensors ready to broadcast
        self.er_cands = [(float(q), float(w), float(a))
                         for q in cfg["er_q"] for w in cfg["er_w"] for a in cfg["er_a"]]
        v = lambda j: torch.tensor([c[j] for c in self.er_cands], device=device).view(-1, 1, 1, 1, 1, 1)
        self.er_q, self.er_w, self.er_a = v(0), v(1), v(2)
        self.reset()

    def reset(self) -> None:
        self.n = 0                  # innovations seen
        self.n_e = 0                # energy pairs seen
        self.n_uc = 0               # convection-speed estimates seen
        self.n_win = 0              # windows seen in this trajectory
        self.k_win = 0              # index of the window currently being emitted (D2/E1)
        self.R_run = None           # D2/H1: causal plain running mean of revealed target ke maps
        self.n_R = 0                # ... its pseudo-count (window 0 seeds with the input window's own kx)
        self.n_b = 0                # blend pairs seen
        self.uc = None              # convection speed, px/frame (0-d tensor on the device)
        self.params: Optional[Tuple[float, float]] = None   # (Re, AoA) read off the first window
        self.prev_out = self.prev_fl = self.prev_adv = self.prev_mf = self.prev_mask = None
        self.n_w = 0                # windows seen in this trajectory (the net's wfrac feature)
        self.mu = self.var = self.prof = self.e_true = self.e_pred = None
        # the alternative centre, carried as the two components y - const and the
        # fluctuation basis, so any shrink g has a closed-form bias and variance
        self.M0 = self.M1 = self.Saa = self.Sab = self.Sbb = None
        self.n_c = 0
        self.prev_const = self.prev_basis = self.ctr_out = self.ctr_var = None
        self.uc_hist = []           # scanned convection speeds (0-d tensors), for the stability guard
        self.uc_eff = None          # the speed actually advected with (guard may freeze it)
        self.adv_w = None           # weight on the Taylor branch (guard may shrink it)
        self.M2 = self.S02 = self.S12 = self.S22 = None   # second centre basis
        self.prev_basis2 = None
        self.Ec = self.Ey = self.Sdd = self.Sdy = None    # window-mean regression
        self.prev_cdev = self.prev_c_raw = self.prev_cx = None
        self.mx_bias = None         # running (target mean - input mean)
        self.run_xhi = None         # running high-band energy of the input window
        self.Mcc = self.Mxx = self.Mcx = self.Mcy = self.Mxy = None   # mean_mode "ls"
        self.n_m = 0
        self.S = None               # blend sufficient statistics: dict of (1, T, H, W, C) maps
        self.ke_ex_prev = self.ke_ef_prev = None    # the net's lagged-energy features
        if self.ke_flow is not None:
            self.ke_flow.reset()
        self.ke_prev = None         # (features, log rule energy, mask) of the emitted window
        self.g2b = None             # energy gain before the online candidate correction
        self.er_acc = None          # (K,) accumulated realised error of every candidate
        self.er_cur = tuple(float(t) for t in self.cfg["er_warm"])
        self.er_prev = None         # what the candidate scorer needs about the last window
        if self.ke_steps:           # undo whatever the previous trajectory adapted
            self.ke.load_state_dict(self.ke_offline)
            self.ke_opt = torch.optim.Adam(self.ke.parameters(),
                                           lr=float(self.cfg["ke_net_online_lr"]))

    def _a(self, n: int) -> float:
        return max(1.0 / (n + float(self.cfg["prior_n"])), float(self.cfg["alpha_min"]))

    # ------------------------------------------------------------------ masks
    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        """(H, W) pixels that are exactly zero (physical u and v) in every frame."""
        phys = x[0, ..., :2] * self.std_in[:2] + self.mean_in[:2]
        return (phys.abs() < 1e-6).all(-1).all(0)

    # ------------------------------------------------------- convection speed
    def _scan_uc(self, fl: torch.Tensor, obstacle: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Shift (px/frame) that best maps frame t onto frame t+tau inside the window.

        Correlation over valid, unmasked pixels of the shifted earlier frames with
        the later frames, summed over the lags in `uc_taus`, maximised on the grid
        and refined by a parabola through the three best points. All candidates are
        gathered in one batched indexing op and the result stays on the device: no
        host sync per candidate (the per-candidate loop this replaces cost ~30 ms per
        step on a GPU, four times the FNO forward).
        """
        f = fl[..., :2]
        grid = self.uc_grid
        u, w = grid.numel(), self.shift.w
        keep = (~mask).view(1, 1, *mask.shape, 1, 1).to(f.dtype)          # (1, 1, H, W, 1, 1)
        obs = obstacle.long().unsqueeze(0)                                  # (1, H, W)
        num = den_a = den_b = torch.zeros(u, device=f.device, dtype=f.dtype)
        for tau in self.cfg["uc_taus"]:
            src = self.shift.cols.view(1, -1) - grid.view(-1, 1) * tau      # (U, W)
            i0 = torch.floor(src)
            wgt = (src - i0).view(1, 1, 1, u, w, 1)
            i0 = i0.long()
            valid = (i0.unsqueeze(1) >= obs + 1) & (i0 >= 0).unsqueeze(1)   # (U, H, W)
            i0c, i1c = i0.clamp(0, w - 1), (i0 + 1).clamp(0, w - 1)
            a = f[:, :-tau]                                                 # (1, T', H, W, 2)
            shifted = (1.0 - wgt) * a[:, :, :, i0c, :] + wgt * a[:, :, :, i1c, :]   # (1, T', H, U, W, 2)
            b = f[:, tau:].unsqueeze(3)                                     # (1, T', H, 1, W, 2)
            k = valid.permute(1, 0, 2).reshape(1, 1, mask.shape[0], u, w, 1).to(f.dtype) * keep.view(1, 1, mask.shape[0], 1, w, 1)
            num = num + (shifted * b * k).sum(dim=(0, 1, 2, 4, 5))
            den_a = den_a + (shifted * shifted * k).sum(dim=(0, 1, 2, 4, 5))
            den_b = den_b + (b * b * k).sum(dim=(0, 1, 2, 4, 5))
        s = num / (den_a * den_b + 1e-30).sqrt()
        k = s.argmax()
        km, kp = (k - 1).clamp(0, u - 1), (k + 1).clamp(0, u - 1)
        den = s[km] - 2.0 * s[k] + s[kp]
        interior = (k > 0) & (k < u - 1) & (den < 0)
        off = torch.where(interior, 0.5 * (s[km] - s[kp]) / torch.where(den < 0, den, -torch.ones_like(den)),
                          torch.zeros_like(den))
        return grid[k] + off * float(self.cfg["uc_step"]), s[k]

    def _update_uc(self, scan) -> None:
        """Merge a new scan into the running speed, then apply the stability guard.

        The scan is noisy on a few trajectories (25425_0: uc 0.84 +- 0.67 over windows,
        `notes/research_2026-09-15/probe_dmd.md`), and a wrong speed advects the last
        input frame into the wrong place -- worse than not advecting at all. The guard
        calls a scan unstable when the recent scans disagree (MAD / median above
        `uc_guard_cv`) or the objective is flat (peak correlation below
        `uc_guard_peak`), and then freezes the speed at the running median and/or
        shrinks the Taylor branch to `uc_guard_w`. Everything stays on the device: no
        host sync, and the cost is a median of <= `uc_guard_k` scalars.
        """
        uc, peak = scan
        if self.uc is None:
            self.uc = uc
        else:
            self.n_uc += 1
            self.uc = self.uc + self._a(self.n_uc) * (uc - self.uc)
        mode = self.cfg["uc_guard"]
        if not mode:
            self.uc_eff, self.adv_w = self.uc, None
            return
        self.uc_hist.append(uc.detach())
        hist = torch.stack(self.uc_hist[-int(self.cfg["uc_guard_k"]):])
        med = hist.median()
        bad = torch.zeros((), device=uc.device, dtype=torch.bool)
        if hist.numel() >= 2:
            mad = (hist - med).abs().median()
            bad = mad / med.abs().clamp_min(1e-2) > float(self.cfg["uc_guard_cv"])
        if float(self.cfg["uc_guard_peak"]) > 0.0:
            bad = bad | (peak < float(self.cfg["uc_guard_peak"]))
        self.uc_eff = torch.where(bad, med, self.uc) if mode in ("median", "both") else self.uc
        self.adv_w = (torch.where(bad, torch.full_like(med, float(self.cfg["uc_guard_w"])),
                                  torch.ones_like(med))
                      if mode in ("shrink", "both") else None)

    def _advect(self, fl_last: torch.Tensor, obstacle: torch.Tensor, mask: torch.Tensor, t: int):
        """Last input frame's fluctuation shifted by uc*(k+1) for k < t, in one gather:
        (1, T, H, W, C), and the validity (1, T, H, W, 1)."""
        w = self.shift.w
        uc = self.uc if self.uc_eff is None else self.uc_eff
        d = uc * torch.arange(1, t + 1, device=fl_last.device, dtype=fl_last.dtype)   # (T,)
        src = self.shift.cols.view(1, -1) - d.view(-1, 1)                                   # (T, W)
        i0 = torch.floor(src)
        wgt = (src - i0).view(1, 1, t, w, 1)
        i0 = i0.long()
        valid = (i0.unsqueeze(1) >= obstacle.long().unsqueeze(0) + 1) & (i0 >= 0).unsqueeze(1) & (~mask).unsqueeze(0)  # (T, H, W)
        i0c, i1c = i0.clamp(0, w - 1), (i0 + 1).clamp(0, w - 1)
        adv = (1.0 - wgt) * fl_last[:, :, i0c, :] + wgt * fl_last[:, :, i1c, :]              # (1, H, T, W, C)
        adv = adv.permute(0, 2, 1, 3, 4) * valid.view(1, t, *mask.shape, 1).to(fl_last.dtype)
        adv[..., 2] = 0.0
        return adv, valid.view(1, t, *mask.shape, 1)

    # ------------------------------------------------------------------ blend
    def _update_blend(self, fl: torch.Tensor, adv: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
        keep = (~mask).view(1, 1, *mask.shape, 1).to(fl.dtype)
        cur = {"ff": fl * fl, "aa": adv * adv, "fa": fl * adv, "fy": fl * y, "ay": adv * y}
        cur = {k: v * keep for k, v in cur.items()}
        if self.S is None:
            self.S = cur
            return
        self.n_b += 1
        a = self._a(self.n_b)
        self.S = {k: self.S[k] + a * (cur[k] - self.S[k]) for k in cur}

    @staticmethod
    def _smooth(m: torch.Tensor) -> torch.Tensor:
        """3x3 box average over (H, W) of a (1, T, H, W, C) map, edge-aware."""
        t, h, w, c = m.shape[1:]
        flat = m.permute(0, 1, 4, 2, 3).reshape(t * c, 1, h, w)
        out = torch.nn.functional.avg_pool2d(flat, 3, stride=1, padding=1, count_include_pad=False)
        return out.reshape(1, t, c, h, w).permute(0, 1, 3, 4, 2)

    @staticmethod
    def _solve(ff, aa, fa, fy, ay, lam: float, with_b: bool, a0=1.0, b0=0.0):
        """Ridge least squares y ~ a f + b adv toward (a0, b0), weight lam x own energies.

        Cauchy-Schwarz gives det >= ff aa (1 - (1+lam)^-2) > 0 whenever lam > 0; the
        clamp guards float32 and lam = 0.
        """
        p11 = ff * (1.0 + lam)
        r1 = fy + lam * ff * a0
        if not with_b:
            return r1 / (p11 + 1e-30), torch.zeros_like(r1)
        p22 = aa * (1.0 + lam)
        r2 = ay + lam * aa * b0
        det = (p11 * p22 - fa * fa).clamp_min(1e-3 * p11 * p22 + 1e-30)
        return (p22 * r1 - fa * r2) / det, (p11 * r2 - fa * r1) / det

    def _coefficients(self, valid: torch.Tensor, mask: torch.Tensor):
        """(a, b) maps of shape (1, T, H, W, C) from the running statistics.

        Ridge least squares toward gain_map's (a, b) = (1, 0), weight `blend_ridge`
        relative to the pooled energies, so that where the revealed targets say the
        blend explains nothing the FNO fluctuation is kept rather than zeroed:
            [ff (1+l), fa; fa, aa (1+l)] [a; b] = [fy + l ff; ay].
        """
        T = valid.shape[1]
        shape = (1, T, *mask.shape, 3)
        if self.S is None:
            return torch.ones(shape, device=self.device), torch.zeros(shape, device=self.device)
        lam = float(self.cfg["blend_ridge"])
        keep = (~mask).view(1, 1, *mask.shape, 1)
        v = (valid & keep).to(self.S["ff"].dtype)
        i = (~valid & keep).to(self.S["ff"].dtype)
        pooled = {}
        for region, w, with_b in (("v", v, True), ("i", i, False)):
            sums = [(self.S[k] * w).sum(dim=(2, 3), keepdim=True) for k in ("ff", "aa", "fa", "fy", "ay")]
            pooled[region] = self._solve(*sums, lam=lam, with_b=with_b)
        a = torch.where(valid, pooled["v"][0], pooled["i"][0])
        b = torch.where(valid, pooled["v"][1], torch.zeros_like(pooled["v"][1]))
        if self.cfg["blend"] == "pixel":
            # per element, ridge toward the region solution with n0 pseudo-windows
            S = self.S
            if self.cfg["blend_smooth"]:
                S = {k: self._smooth(v_) for k, v_ in S.items()}
            lam_p = float(self.cfg["blend_pixel_n0"]) / float(self.n_b + 1)
            ap, bp = self._solve(*(S[k] for k in ("ff", "aa", "fa", "fy", "ay")), lam=lam_p, with_b=True, a0=a, b0=b)
            a = ap
            b = torch.where(valid, bp, torch.zeros_like(bp))
        a[..., 2] = 1.0
        b[..., 2] = 0.0
        return a, b

    # ------------------------------------------------------------------- gain
    def _update_energy(self, e_true: torch.Tensor, e_pred: torch.Tensor) -> None:
        if self.e_true is None:
            self.e_true, self.e_pred = e_true.clone(), e_pred.clone()
            return
        self.n_e += 1
        a = self._a(self.n_e)
        self.e_true = self.e_true + a * (e_true - self.e_true)
        self.e_pred = self.e_pred + a * (e_pred - self.e_pred)

    def _update_R(self, y: torch.Tensor) -> None:
        """D2/H1: causal PLAIN running mean (1/n, not EMA -- matches
        `scripts/h1_stage0.py`'s `R_k` candidate exactly) of REVEALED target ke
        maps. Called on the just-revealed `prev_target` before this window's own
        `kx` is available; `self.R_run` is seeded from `kx` itself at window 0
        (see `step()`), matching `h1_stage0.py`'s `M[0] = ke of the raw input
        window` causal seed."""
        ky = _ke(y)
        self.n_R += 1
        self.R_run = self.R_run + (ky - self.R_run) / self.n_R

    def _gamma(self) -> torch.Tensor:
        g_max = float(self.cfg["gain_max"])
        ratio = (self.e_true + 1e-8) / (self.e_pred + 1e-8)
        return ratio.pow(0.5 * float(self.cfg["gain_power"])).clamp(1.0 / g_max, g_max)

    # ----------------------------------------------- learned / online energy target
    def _g2_base(self, x, fl_f, e_f, kx, mask) -> torch.Tensor:
        """Energy gain E_out / e_f before the online candidate correction, (1, 1, H, W, C).

        gain_map's calibration `(runx/runf)^power` in energy units, times the learned
        log-space correction on the u, v channels. The net's last layer is
        zero-initialised and it is fitted as a residual on the rule, so with the net off
        (or `ke_net_scale = 0`) this is the shipped rule.
        """
        g2 = ((self.e_true + 1e-8) / (self.e_pred + 1e-8)).pow(float(self.cfg["gain_power"]))
        if self.ke_flow is not None:
            # the flow net predicts the energy target directly (rule + residual); express
            # it as a gain on e_f so the clip and both branches downstream are untouched
            e_tgt = self.ke_flow(x, fl_f, kx, e_f, self.e_true, self.e_pred, mask)
            w = self.cfg.get("e_blend_w")
            if w is not None and self.R_run is not None:
                # D2/H1 energy-target blend (experiments/h1_tke_map/notes.txt
                # STAGE 0a/0d): e_tgt = w*Khat + (1-w)*R, R the causal running
                # mean of revealed target ke (self.R_run, updated in step()).
                e_tgt = float(w) * e_tgt + (1.0 - float(w)) * self.R_run
            g_flow = (e_tgt[..., :2] + 1e-10) / (e_f[..., :2] + 1e-10)
            if self.ke is None:
                g2 = g2.clone()
                g2[..., :2] = g_flow
                return g2
        if self.ke is None:
            return g2
        wfrac = torch.full((1,), min(self.n_w, 20) / 20.0, device=x.device, dtype=x.dtype)
        feat, ex, ef = K.window_features(x, fl_f, kx, e_f, self.e_true, self.e_pred,
                                         mask.unsqueeze(0), wfrac,
                                         self.ke_ex_prev, self.ke_ef_prev)
        corr = K.predict_log_energy(self.ke, feat, mask.unsqueeze(0),
                                    float(self.cfg["ke_net_scale"]), float(self.cfg["ke_net_clip"]))
        self.ke_ex_prev, self.ke_ef_prev = ex, ef
        if self.ke_steps:
            self.ke_prev = (feat, K.rule_log_energy(ef, self.e_true[:, 0, ..., :2],
                                                    self.e_pred[:, 0, ..., :2],
                                                    float(self.cfg["gain_power"])),
                            mask.unsqueeze(0))
        g2 = g2.clone()
        g2[..., :2] = g2[..., :2] * corr.unsqueeze(1).exp()
        if self.ke_flow is not None:
            # both nets are residuals on the same rule: blend them in log-energy space
            b = float(self.cfg.get("ke_flow_blend", 1.0))
            g2[..., :2] = g2[..., :2].clamp_min(1e-12).log().mul(1.0 - b).add(
                g_flow.clamp_min(1e-12).log().mul(b)).exp()
        return g2

    def _gamma_ext(self, x, fl_f, e_f, kx, mask) -> torch.Tensor:
        """gamma with the learned correction and the online candidate applied.

        E_out = e_f * g2_base * A * (e_f/runf)^(q-1) * (kx/runx)^w; the candidate factor is
        1 for (q, w, A) = (1, 0, 1), which is what `er_warm` and `er_online = False` use.
        """
        g_max = float(self.cfg["gain_max"])
        g2 = self.g2b = self._g2_base(x, fl_f, e_f, kx, mask)
        if self.cfg["er_online"]:
            q, w, a = self.er_cur
            if (q, w, a) != (1.0, 0.0, 1.0):
                g2 = g2 * a * ((e_f + 1e-10) / (self.e_pred + 1e-10)).pow(q - 1.0) \
                            * ((kx + 1e-10) / (self.e_true + 1e-10)).pow(w)
        return g2.clamp(1.0 / g_max ** 2, g_max ** 2).sqrt()

    def _ke_online_update(self, y: torch.Tensor) -> None:
        """`ke_net_online` Adam steps on the window whose target has just been revealed.

        The features of that window are cached, so this is a few forward/backward passes
        of a 5.5k-parameter conv net on one sample -- the same metric loss the offline fit
        minimises, on the single pair the stream just handed us.
        """
        if self.ke_prev is None:
            return
        feat, log_rule, mask = self.ke_prev
        ey = _ke(y)[:, 0, ..., :2]
        with torch.enable_grad():
            for _ in range(self.ke_steps):
                loss = K.metric_loss(self.ke, feat, log_rule, ey, mask,
                                     float(self.cfg["ke_net_scale"]), float(self.cfg["ke_net_clip"]))
                self.ke_opt.zero_grad()
                loss.backward()
                self.ke_opt.step()

    # ------------------------------------------------- online energy-rule selection
    def _e_cands(self, P) -> torch.Tensor:
        """(K, 1, 1, H, W, C) output energy of every candidate for the cached window."""
        g_max = float(self.cfg["gain_max"])
        mod = ((P["kf"] + 1e-10) / (P["rf"] + 1e-10)).pow(self.er_q - 1.0)
        per = ((P["kx"] + 1e-10) / (P["rx"] + 1e-10)).pow(self.er_w)
        g2 = (P["g2b"] * self.er_a * mod * per).clamp(1.0 / g_max ** 2, g_max ** 2)
        return P["kf"] * g2

    def _update_er(self, y: torch.Tensor) -> None:
        """Score every candidate on the window whose target has just been revealed.

        The emitted output was `const + fluc + mu` with `const` constant in t, so a
        candidate that wants energy E instead of the realised var_t(fluc) simply scales
        `fluc` by c = sqrt(E / var_t(fluc)); its temporal variance is then
        c^2 vF + 2 c cFM + vM in closed form, and its field residual is R + (c-1) fluc.
        No extra forward pass and no per-candidate copy of the field.
        """
        P = self.er_prev
        if P is None:
            return
        std2 = (self.std_tgt[:2] ** 2).view(1, 1, 1, 1, 1, 2)
        c = (self._e_cands(P) / P["vF"].clamp_min(1e-12)).clamp_min(0.0).sqrt()   # (K,1,1,H,W,C)
        var = c * c * P["vF"] + 2.0 * c * P["cFM"] + P["vM"]                      # (K,1,1,H,W,C)
        ke_c = 0.5 * (var[..., :2] * std2).sum(-1)                                # (K,1,1,H,W)
        ke_y = 0.5 * (_ke(y)[..., :2] * std2[0]).sum(-1)                          # (1,1,H,W)
        num = (ke_c - ke_y).flatten(1).pow(2).sum(1)
        err = (num / ke_y.pow(2).sum().clamp_min(1e-16)).sqrt()                   # (K,)
        if self.cfg["er_obj"] == "both":
            R = P["out"] - y
            SRR = (R * R).sum(dim=1, keepdim=True)
            SRF = (R * P["fluc"]).sum(dim=1, keepdim=True)
            SFF = (P["fluc"] * P["fluc"]).sum(dim=1, keepdim=True)
            d = c - 1.0
            q = ((SRR + 2.0 * d * SRF + d * d * SFF)[..., :2] * std2).clamp_min(0.0)
            den = ((y[..., :2] * self.std_tgt[:2]) ** 2).sum().clamp_min(1e-16)
            err = err + float(self.cfg["er_lam"]) * (q.flatten(1).sum(1) / den).sqrt()
        self.er_acc = err if self.er_acc is None else self.er_acc + err
        self.er_cur = self.er_cands[int(self.er_acc.argmin())]

    def _shape(self, zc, fl_f, e_z, e_f, c_zf, e_tgt):
        """Fluctuation with per-(pixel, channel) temporal energy e_tgt from the blend's shape zc.

        zc is scaled by up to `shape_cap`; if that does not reach e_tgt (the blend has
        (nearly) zeroed the fluctuation because nothing predicts the phase there) the
        remainder comes from the FNO's own fluctuation fl_f, whose energy is the right
        one: s solves s^2 e_f + 2 s cov(gamma zc, fl_f) = e_tgt - gamma^2 e_z. Where the
        blend is (1, 0) this reduces exactly to gain_map's gamma * fl_f.
        """
        gamma = ((e_tgt + 1e-10) / (e_z + 1e-10)).sqrt().clamp(0.0, float(self.cfg["shape_cap"]))
        d_e = (e_tgt - gamma * gamma * e_z).clamp_min(0.0)
        c = gamma * c_zf
        s = ((c * c + e_f * d_e).clamp_min(0.0).sqrt() - c) / (e_f + 1e-10)
        return gamma * zc + s.clamp(0.0, float(self.cfg["gain_max"])) * fl_f

    # ------------------------------------------------- incoherent-band energy floor
    def _band_floor(self, fluc: torch.Tensor, fl_x: torch.Tensor) -> torch.Tensor:
        """Re-energise the fluctuation's incoherent band to a floor read off the input.

        `idea_4` measured that the whole TKE error lives in DFT modes <= 4 and that the
        prediction is amplitude-deficient there; injecting the missing energy into the
        *incoherent* band instead raises the amplitude at a fraction of the rel_l2 cost,
        because that band's phase is unpredictable anyway and contributes almost nothing
        to the pointwise error. The two bands are exactly orthogonal over the 20 frames,
        so the per-pixel TKE map is the sum of their energies and nothing else changes.
        Deterministic rescaling of the predictor's own high band (the `noise_hi`
        synthesis of idea_4 is worth another +0.1 TKE and is not implemented here).
        """
        kc = int(self.cfg["band_split"])
        F = torch.fft.rfft(fluc, dim=1)
        F[:, :kc + 1] = 0.0                                  # keep only modes > kc
        hi = torch.fft.irfft(F, n=fluc.shape[1], dim=1)
        lo = fluc - hi
        Fx = torch.fft.rfft(fl_x, dim=1)
        Fx[:, :kc + 1] = 0.0
        x_hi = _ke(torch.fft.irfft(Fx, n=fl_x.shape[1], dim=1)) * (self.std_in / self.std_tgt) ** 2
        if self.run_xhi is None:
            self.run_xhi = x_hi.clone()
        else:
            self.run_xhi = self.run_xhi + self._a(self.n_e) * (x_hi - self.run_xhi)
        tgt = self.run_xhi * float(self.cfg["hi_scale"])
        c = ((tgt + 1e-12) / (_ke(hi) + 1e-12)).sqrt().clamp(0.0, float(self.cfg["gain_max"]))
        return lo + c * hi

    # ------------------------------------------------------------ bias/spread
    def _update_error(self, err: torch.Tensor) -> None:
        pool = self.cfg["pool_t"]
        if self.mu is None:
            self.mu = torch.zeros_like(err)
            self.var = _ke(err).clone() if pool else _ke(err).expand_as(err).clone()
            self.prof = torch.ones_like(err.mean(dim=(2, 3), keepdim=True))
        self.n += 1
        a = self._a(self.n)
        d = err - self.mu
        am = a if self.mu_floor is None else torch.clamp(self.mu_floor, min=a)
        self.mu = self.mu + am * d
        d2 = d * d
        self.var = (1.0 - a) * (self.var + a * (d2.mean(dim=1, keepdim=True) if pool else d2))
        if pool == "sep":  # lead-time profile pooled over the field, (1, T, 1, 1, C)
            self.prof = self.prof + a * (d2.mean(dim=(2, 3), keepdim=True) - self.prof)

    def _var(self) -> torch.Tensor:
        if self.cfg["pool_t"] == "sep":
            return self.var * self.prof / self.prof.mean(dim=1, keepdim=True)
        return self.var

    # --------------------------------------------------- window-mean correction
    def _to_tgt(self, x: torch.Tensor) -> torch.Tensor:
        """A window normalised with the *input* statistics, in *target*-normalised units."""
        return (x * self.std_in + self.mean_in - self.mean_tgt) / self.std_tgt

    def _update_mean(self, y: torch.Tensor) -> None:
        """Running statistics of the window mean, for `mean_mode`.

        `rel_l2` on unseen trajectories is set by the window mean plus the bias map
        (`notes/research_2026-09-15/transfer_gap.md` 6.3), so the mean deserves its own
        online model. Two families, both per (pixel, channel):

        * "slope": ridge regression of the revealed window mean on the deviation of the
          FNO/blend window mean from its own running mean. Only the *slope* is applied
          (`(slope - 1) * dev`, zero-mean over windows), so it composes with the bias map
          `mu` instead of fighting it for the level.
        * "persist" / "ls": the input window's own mean is a second candidate predictor
          (the target frames follow the input frames, so in a statistically stationary
          wake it is a strong one). "persist" uses it plus a running offset; "ls" solves
          the per-pixel 2-term ridge least squares between it and the FNO mean, ridged
          toward the FNO mean, so `mean_ridge -> inf` is the shipped behaviour.
        """
        mode = self.cfg["mean_mode"]
        ym = y.mean(dim=1, keepdim=True)
        c = self.prev_c_raw
        cx = self.prev_cx
        if c is None:
            return
        self.n_m += 1
        al = self._a(self.n_m)
        if mode == "both":
            for m in ("slope", "ls"):
                self.cfg["mean_mode"] = m
                self.n_m -= 1
                self._update_mean(y)
            self.cfg["mean_mode"] = "both"
            return
        if mode == "slope":
            if self.Ec is None:
                self.Ec, self.Ey = c.clone(), ym.clone()
                self.Sdd = torch.zeros_like(c)
                self.Sdy = torch.zeros_like(c)
                return
            dc, dy = c - self.Ec, ym - self.Ey
            self.Ec = self.Ec + al * dc
            self.Ey = self.Ey + al * dy
            self.Sdd = (1.0 - al) * (self.Sdd + al * dc * dc)
            self.Sdy = (1.0 - al) * (self.Sdy + al * dc * dy)
            return
        if mode == "persist":
            d = ym - cx
            self.mx_bias = d.clone() if self.mx_bias is None else self.mx_bias + al * (d - self.mx_bias)
            return
        if mode == "ls":
            cur = (c * c, cx * cx, c * cx, c * ym, cx * ym)
            if self.Mcc is None:
                self.Mcc, self.Mxx, self.Mcx, self.Mcy, self.Mxy = (t.clone() for t in cur)
                return
            self.Mcc = self.Mcc + al * (cur[0] - self.Mcc)
            self.Mxx = self.Mxx + al * (cur[1] - self.Mxx)
            self.Mcx = self.Mcx + al * (cur[2] - self.Mcx)
            self.Mcy = self.Mcy + al * (cur[3] - self.Mcy)
            self.Mxy = self.Mxy + al * (cur[4] - self.Mxy)

    def _mean_correct(self, const: torch.Tensor, cx: torch.Tensor) -> torch.Tensor:
        """`const` (constant in t) after the `mean_mode` correction."""
        mode = self.cfg["mean_mode"]
        if mode == "both":   # the two corrections are additive deltas on the same base
            self.cfg["mean_mode"] = "slope"
            out = self._mean_correct(const, cx)
            self.cfg["mean_mode"] = "ls"
            out = out + self._mean_correct(const, cx) - const
            self.cfg["mean_mode"] = "both"
            return out
        r = float(self.cfg["mean_ridge"])
        cl = float(self.cfg["mean_clip"])
        if mode == "slope":
            if self.Sdd is None:
                return const
            slope = (self.Sdy + r * self.Sdd) / (self.Sdd * (1.0 + r) + 1e-20)
            return const + (slope - 1.0).clamp(-cl, cl) * (const - self.Ec)
        if mode == "persist":
            if self.mx_bias is None:
                return const
            return cx + self.mx_bias
        if mode == "ls":
            if self.Mcc is None:
                return const
            p11 = self.Mcc * (1.0 + r)
            p22 = self.Mxx * (1.0 + r)
            r1 = self.Mcy + r * self.Mcc
            r2 = self.Mxy
            det = (p11 * p22 - self.Mcx * self.Mcx).clamp_min(1e-3 * p11 * p22 + 1e-30)
            w1 = (p22 * r1 - self.Mcx * r2) / det
            w2 = (p11 * r2 - self.Mcx * r1) / det
            return w1.clamp(0.0, 1.0 + cl) * const + w2.clamp(-cl, cl) * cx
        return const

    # ------------------------------------------- the alternative centre's statistics
    def _update_ctr(self, y: torch.Tensor) -> None:
        """Running statistics of the centre family `const + g * basis`.

        Writing `a = y - const` and `b = basis`, every member's innovation is
        `a - g b`, so one pair of running means (M0, M1) and the three pooled second
        moments (Saa, Sab, Sbb) give the bias `M0 - g M1` and the pooled variance
        `Saa - 2 g Sab + g^2 Sbb` of *every* g in closed form -- including the online
        least-squares choice `g = Sab / Sbb`.  Three elementwise products and three
        temporal means: ~1 ms.
        """
        a0 = y - self.prev_const
        b0 = self.prev_basis
        # pooled over the 20 leads by default; `ctr_lead` keeps the moments per lead,
        # so the shrink (and the interval width) may differ between lead 1 and lead 20.
        lead = bool(self.cfg["ctr_lead"])
        m = (lambda t: t) if lead else (lambda t: t.mean(dim=1, keepdim=True))
        two = bool(self.cfg["ctr_basis2"])
        c0 = self.prev_basis2 if two else None
        if self.M0 is None:
            self.M0 = torch.zeros_like(a0)
            self.M1 = torch.zeros_like(b0)
            self.Saa = m(_ke(a0).expand_as(a0) if lead else _ke(a0)).clone()
            self.Sab = torch.zeros_like(self.Saa)
            self.Sbb = torch.zeros_like(self.Saa)
        if two and self.M2 is None:
            self.M2 = torch.zeros_like(b0)
            self.S02 = torch.zeros_like(self.Saa)
            self.S12 = torch.zeros_like(self.Saa)
            self.S22 = torch.zeros_like(self.Saa)
        self.n_c += 1
        al = self._a(self.n_c)
        da, db = a0 - self.M0, b0 - self.M1
        self.M0 = self.M0 + al * da
        self.M1 = self.M1 + al * db
        self.Saa = (1.0 - al) * (self.Saa + al * m(da * da))
        self.Sab = (1.0 - al) * (self.Sab + al * m(da * db))
        self.Sbb = (1.0 - al) * (self.Sbb + al * m(db * db))
        if two:
            dc = c0 - self.M2
            self.M2 = self.M2 + al * dc
            self.S02 = (1.0 - al) * (self.S02 + al * m(da * dc))
            self.S12 = (1.0 - al) * (self.S12 + al * m(db * dc))
            self.S22 = (1.0 - al) * (self.S22 + al * m(dc * dc))

    def _ctr_gain(self) -> torch.Tensor:
        """The shrink g applied to the centre's fluctuation.

        A constant, or the running least-squares gain `Sab / Sbb` -- the g that minimises
        the centre's innovation variance, and so (the interval factor being monotone
        decreasing in sigma) the g that maximises SPS.  Ridged toward 1 with the g = 0
        variance Saa as the reference scale, which is exactly 1 before any basis has
        been seen.
        """
        g = self.cfg["ctr_g"]
        if not isinstance(g, str):
            return torch.full((1, 1, 1, 1, 1), float(g),
                              device=self.Saa.device, dtype=self.Saa.dtype)
        r = float(self.cfg["ctr_g_ridge"])
        lo, hi = self.cfg["ctr_g_clip"]
        if g == "auto":                                    # one gain for the whole field
            num = self.Sab.sum() + r * self.Saa.sum()
            den = self.Sbb.sum() + r * self.Saa.sum()
            return (num / den.clamp_min(1e-20)).clamp(float(lo), float(hi)) \
                .reshape(1, 1, 1, 1, 1)
        num = self.Sab + r * self.Saa                      # per (pixel, channel)
        den = self.Sbb + r * self.Saa
        return (num / den.clamp_min(1e-20)).clamp(float(lo), float(hi))

    def _ctr_gain2(self):
        """(g1, g2) of the two-vector centre `const + g1 b1 + g2 b2`, per pixel/channel.

        Same closed form one dimension up: minimising `Saa - 2 g.S0 + g' S g` plus the
        ridge `r Saa (|g1 - 1|^2 + |g2|^2)` -- so `mean_ridge -> inf` gives the shipped
        (g1, g2) = (1, 0) and a 2x2 symmetric solve gives the optimum. b1 is the shipped
        basis (the least-squares blend shape), b2 the advected fluctuation alone, so the
        pair spans the FNO and Taylor directions separately.
        """
        r = float(self.cfg["ctr_g_ridge"])
        lo, hi = self.cfg["ctr_g_clip"]
        p11 = self.Sbb + r * self.Saa
        p22 = self.S22 + r * self.Saa
        r1 = self.Sab + r * self.Saa
        r2 = self.S02
        det = (p11 * p22 - self.S12 * self.S12).clamp_min(1e-3 * p11 * p22 + 1e-30)
        g1 = ((p22 * r1 - self.S12 * r2) / det).clamp(float(lo), float(hi))
        g2 = ((p11 * r2 - self.S12 * r1) / det).clamp(-float(hi), float(hi))
        return g1, g2

    # ------------------------------------------------------------------ prior
    def _apply_prior(self, x: torch.Tensor) -> None:
        """Start the running statistics from the kernel-weighted prior maps of the
        training files nearest in (AoA, log Re) to what the first window says."""
        state = x[0].cpu().numpy()
        re, aoa = float(estimate_re(state)), float(estimate_aoa(state))
        self.params = (re, aoa)
        p = self.prior
        d_aoa = (p["aoa"] - aoa) / float(self.cfg["prior_sigma_aoa"])
        d_re = (p["re"].clamp_min(1.0).log() - math.log(max(re, 1.0))) / float(self.cfg["prior_sigma_re"])
        w = torch.exp(-0.5 * (d_aoa * d_aoa + d_re * d_re))
        w = w / w.sum().clamp_min(1e-12)
        mix = lambda m: torch.einsum("f,f...->...", w, m).unsqueeze(0)
        self.mu, self.var = mix(p["mu"]), mix(p["var"])
        self.prof = torch.ones_like(self.mu.mean(dim=(2, 3), keepdim=True))
        self.e_true, self.e_pred = mix(p["e_true"]), mix(p["e_pred"])
        n0 = float(self.cfg["prior_windows"])
        n0e = self.cfg["prior_windows_e"]
        n0c = self.cfg["prior_windows_c"]
        self.n = n0
        self.n_e = n0 if n0e is None else float(n0e)
        n0c = n0 if n0c is None else float(n0c)
        if self.cfg["ctr"]:
            # the prior maps are of the *raw FNO* innovation, which is what the
            # un-inflated centre is closer to; seed (M0, Saa) from them and leave the
            # basis moments at zero, so g starts at the ridge's 1.
            var0 = self.var.expand_as(self.mu).clone() if self.cfg["ctr_lead"] else self.var.clone()
            self.M0, self.Saa, self.n_c = self.mu.clone(), var0, n0c
            self.M1 = torch.zeros_like(self.M0)
            self.Sab = torch.zeros_like(self.Saa)
            self.Sbb = torch.zeros_like(self.Saa)
            if self.cfg["ctr_basis2"]:
                self.M2 = torch.zeros_like(self.M0)
                self.S02 = torch.zeros_like(self.Saa)
                self.S12 = torch.zeros_like(self.Saa)
                self.S22 = torch.zeros_like(self.Saa)

    # ------------------------------------------------------------------- step
    def step(self, x: torch.Tensor, prev_target: Optional[torch.Tensor]):
        loss = None
        self.k_win = self.n_win     # D2/E1: window index of the window about to be emitted
        if prev_target is not None and self.prev_out is not None:
            err = prev_target - self.prev_out
            loss = float((err ** 2).mean())
            self._update_error(err)
            if self.cfg["ctr"] and self.prev_const is not None:
                self._update_ctr(prev_target)
            if self.cfg["advect"]:
                self._update_blend(self.prev_fl, self.prev_adv, prev_target - self.prev_mf, self.prev_mask)
            if self.cfg["mean_mode"]:
                self._update_mean(prev_target)
            if self.cfg.get("e_blend_w") is not None:
                self._update_R(prev_target)   # D2/H1: causal running mean of revealed target ke

        if prev_target is not None:
            if self.ke_steps:
                self._ke_online_update(prev_target)
            if self.cfg["er_online"]:
                self._update_er(prev_target)

        if self.prior is not None and self.params is None:
            self._apply_prior(x)

        T = x.shape[1]
        mask = self._mask(x)
        raw = self.base(x)
        mf = raw.mean(dim=1, keepdim=True)
        fl_f = raw - mf

        if self.cfg["advect"]:
            obstacle = self.shift.last_obstacle(mask)
            mx = x.mean(dim=1, keepdim=True)
            fl_x = x - mx
            if self.cfg["uc_fixed"] is not None:
                self.uc = float(self.cfg["uc_fixed"])
            elif (self.uc is None or self.n_win < int(self.cfg["uc_warm"])
                  or self.n_win % int(self.cfg["uc_every"]) == 0):
                self._update_uc(self._scan_uc(fl_x, obstacle, mask))
            adv, valid = self._advect(fl_x[:, -1], obstacle, mask, T)
            a, b = self._coefficients(valid, mask)
            if self.adv_w is not None:
                b = b * self.adv_w
            z = a * fl_f + b * adv
        else:
            adv = None
            z = fl_f

        # out = const (constant in t) + fluc (zero temporal mean) + mu; the online energy
        # rule only ever rescales `fluc`, so keeping the three apart makes the candidate
        # scorer exact and free.
        const, fluc = mf, torch.zeros_like(fl_f)
        zc_ls = None                 # the un-inflated least-squares fluctuation, if any
        kx = _ke(x)
        if self.cfg.get("e_blend_w") is not None and self.R_run is None:
            # D2/H1: seed the causal running mean R from the input window's own
            # energy at window 0 -- matches h1_stage0.py's causal M[0] exactly
            # (no revealed target exists yet, so the input window IS the history).
            self.R_run = kx.clone()
            self.n_R = 1
        if self.cfg["gain"]:
            # gain_map's energy rule: the output's per-(pixel, channel) temporal energy this
            # window is var_t(F') * gamma^2 with gamma from the running ratio of the input
            # window's energy to the forecast's (the current window included) -- corrected by
            # the learned map and/or the online candidate when either is switched on.
            e_f = _ke(fl_f)
            self._update_energy(kx, e_f)
            self.g2b = None
            if self.ke is None and self.ke_flow is None and not self.cfg["er_online"]:
                gamma = self._gamma()          # untouched: gain_map_improved bit-for-bit
            else:
                gamma = self._gamma_ext(x, fl_f, e_f, kx, mask)
            if self.cfg["advect"] and self.S is not None:
                # the blend supplies the *shape*: rescale it to that energy, per window
                mz = z.mean(dim=1, keepdim=True)
                zc = zc_ls = z - mz
                const = mf + mz
                fluc = self._shape(zc, fl_f, _ke(zc), e_f, (zc * fl_f).mean(dim=1, keepdim=True), gamma * gamma * e_f)
            else:
                fluc = gamma * fl_f
        elif self.cfg["advect"]:
            mz = z.mean(dim=1, keepdim=True)
            const, fluc = mf + mz, z - mz
        else:
            const, fluc = mf, raw - mf
        if self.cfg["band_split"] and self.cfg["gain"] and self.cfg["advect"]:
            fluc = self._band_floor(fluc, fl_x)
        if self.cfg["mean_mode"]:
            # the window mean gets its own online model; the raw mean and the input
            # window's mean are cached for the regression the next revealed target feeds
            cx = self._to_tgt(x.mean(dim=1, keepdim=True))
            self.prev_c_raw, self.prev_cx = const.detach(), cx.detach()
            const = self._mean_correct(const, cx)
        mu = None
        if self.cfg["bias"] and self.mu is not None and self.cfg.get("mu_vp", False):
            # D2/H1 variance-preserving mu (experiments/h1_tke_map/notes.txt STAGE
            # 0c(prompt)): mu's own zero-mean-over-t fluctuation otherwise adds
            # UNCOUNTED variance on top of fluc's already-imposed energy target
            # (measured -0.11 tke on the shipped config). Demean mu (mu' = mu -
            # mean_t mu) and rescale fluc by the closed-form c solving
            # var_t(c*fluc + mu') == var_t(fluc) (the already-achieved,
            # imposition-exact target -- no extra plumbing needed to know it).
            mu = self.mu
            mu_p = mu - mu.mean(dim=1, keepdim=True)
            e_tgt_ach = _ke(fluc)
            vM = _ke(mu_p)
            cFM = (fluc * mu_p).mean(dim=1, keepdim=True)
            disc = (cFM * cFM - e_tgt_ach * vM + e_tgt_ach * e_tgt_ach).clamp_min(0.0)
            c = ((-cFM + disc.sqrt()) / e_tgt_ach.clamp_min(1e-12)).clamp_min(0.0)
            fluc = c * fluc
        elif self.cfg["bias"] and self.mu is not None:
            mu = self.mu
        out = const + fluc
        if mu is not None:
            out = out + mu

        # ---- the decoupled interval centre ------------------------------------
        # `out` is what we return and what the accuracy factors see; `ctr_out` is only
        # ever the middle of the interval.  Its fluctuation is a blend of the shipped
        # (energy-inflated) one and the un-inflated least-squares one, shrunk by g, and
        # its bias and spread come from the centre family's own running statistics.
        self.ctr_out = self.ctr_var = None
        if self.cfg["ctr"]:
            w = float(self.cfg["ctr_w"])
            basis = fluc if (zc_ls is None or w == 0.0) else (1.0 - w) * fluc + w * zc_ls
            self.prev_const, self.prev_basis = const.detach(), basis.detach()
            two = bool(self.cfg["ctr_basis2"])
            b2 = None
            if two:
                b2 = (torch.zeros_like(basis) if adv is None
                      else adv - adv.mean(dim=1, keepdim=True))
                self.prev_basis2 = b2.detach()
            if self.M0 is not None:
                if two and self.M2 is not None:
                    g, g2 = self._ctr_gain2()
                    ctr = const + g * basis + g2 * b2
                    if self.cfg["ctr_bias"]:
                        ctr = ctr + self.M0 - g * self.M1 - g2 * self.M2
                    elif mu is not None:
                        ctr = ctr + mu
                    self.ctr_var = (self.Saa - 2.0 * g * self.Sab - 2.0 * g2 * self.S02
                                    + g * g * self.Sbb + 2.0 * g * g2 * self.S12
                                    + g2 * g2 * self.S22).clamp_min(1e-12)
                    self.ctr_out = ctr.detach()
                else:
                    g = self._ctr_gain()
                    ctr = const + g * basis
                    if self.cfg["ctr_bias"]:
                        ctr = ctr + self.M0 - g * self.M1
                    elif mu is not None:
                        ctr = ctr + mu
                    self.ctr_var = (self.Saa - 2.0 * g * self.Sab
                                 + g * g * self.Sbb).clamp_min(1e-12)
                    self.ctr_out = ctr.detach()

        if self.cfg["er_online"] and self.cfg["gain"]:
            mu_c = torch.zeros_like(fluc) if mu is None else mu - mu.mean(dim=1, keepdim=True)
            self.er_prev = {
                "kf": e_f, "kx": kx, "rx": self.e_true.clone(), "rf": self.e_pred.clone(),
                "g2b": self.g2b, "vF": _ke(fluc), "vM": _ke(mu_c),
                "cFM": (fluc * mu_c).mean(dim=1, keepdim=True) if mu is not None else torch.zeros_like(e_f),
            }
            if not self.cfg["er_exact"]:
                self.er_prev["vM"] = torch.zeros_like(e_f)
                self.er_prev["cFM"] = torch.zeros_like(e_f)
            if self.cfg["er_obj"] == "both":
                self.er_prev["out"] = out.detach()
                self.er_prev["fluc"] = fluc.detach()

        self.n_w += 1
        self.prev_out = out.detach()
        self.prev_fl, self.prev_adv, self.prev_mf, self.prev_mask = fl_f.detach(), adv, mf.detach(), mask
        self.n_win += 1
        return out.detach(), loss

    def bounds(self, mean: torch.Tensor, x: torch.Tensor,
               var: Optional[torch.Tensor] = None,
               k_scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score-optimal interval around `mean`. `var` / `k_scale` override the
        prediction's own spread, so the decoupled centre can bring its own."""
        if var is None:
            var = self._var() if self.var is not None else _ke(x)   # first window: input variance
        sch = self.cfg.get("sig_sched")
        if sch is not None:
            s = float(sch[self.k_win if self.k_win < len(sch) else len(sch) - 1])
            if s != 1.0:            # python-level branch: no kernel launch when s == 1
                var = var * (s * s)
        std = var.expand_as(mean).sqrt()
        a = 2.0 * std * self.std_tgt / _SIGMA_GLOBAL
        pos = (a.clamp_min(1e-12).log() - self.log_a[0]) / (self.log_a[1] - self.log_a[0])
        k = self.k_tab[pos.round().long().clamp(0, len(self.k_tab) - 1)]
        half = float(self.cfg["k_scale"] if k_scale is None else k_scale) * k * std
        return mean - half, mean + half


class ExperimentTTT:
    """c3b: internal compute device may differ from the harness's `device`.

    `local_eval_with_analytics.Normalizer` (shared infra, not ours to edit) is
    never moved off CPU, so a CUDA-returning `ttt_step` crashes its
    `postprocess_pred` (device-mismatch bug documented in
    experiments/lco_baseline/notes.md). Rather than force the whole pipeline
    onto CPU threads -- expensive on this shared 32-core box under heavy
    contention (orchestrator instruction 2026-09-18) -- compute always runs on
    CUDA when available (`compute_device`, independent of whatever `device`
    string the harness constructed us with), and `ttt_step` converts back to
    the caller's own input device (`in_device`, read off the actual tensor it
    is handed) before returning -- so CPU-harness callers (e.g.
    `scripts/lco_run_dump.py`'s default `--device cpu`) still get CPU tensors
    back, satisfying the Normalizer, while none of the backbone forward or the
    online-statistics tensor ops run on the CPU.
    """

    def __init__(self, base: torch.nn.Module, device: str, cfg: Dict[str, Any]):
        compute_device = "cuda" if torch.cuda.is_available() else device
        self.model = GainMapImprovedModel(base, compute_device, cfg)
        self.device = compute_device
        self.cfg = cfg

    def reset_ttt_state(self) -> None:
        self.model.reset()
        # c4r1: also clear the backbone wrapper's own causal (S1, S2, n) state --
        # trajectory-scoped exactly like self.model.reset() above, but a
        # separate object (GainMapImprovedModel.base is the CtxNormWrapper).
        reset_fn = getattr(self.model.base, "reset", None)
        if callable(reset_fn):
            reset_fn()

    @torch.no_grad()
    def ttt_step(self, input_norm, prev_target_norm=None):
        x_in = torch.as_tensor(input_norm)
        in_device = x_in.device
        x = x_in.to(self.device)
        y = torch.as_tensor(prev_target_norm).to(self.device) if prev_target_norm is not None else None
        out, loss = self.model.step(x, y)
        info: Dict[str, Any] = {"adapt_loss": loss}
        if self.cfg["bounds"]:
            m = self.model
            if self.cfg["ctr"] and m.ctr_out is not None:
                info["lower"], info["upper"] = m.bounds(
                    m.ctr_out, x, m.ctr_var, float(self.cfg["ctr_k_scale"]))
            else:
                info["lower"], info["upper"] = m.bounds(out, x)
        out = out.to(in_device)
        if "lower" in info:
            info["lower"] = info["lower"].to(in_device)
            info["upper"] = info["upper"].to(in_device)
        return out, info


def get_ttt_model(submission_dir: str, device: str):
    """Entry point. Called once before the stream; not timed.

    c3b: builds the ConvNeXt U-Net (`unet_local`) instead of the FNO. Same
    (B, 20, 32, 64, 3) normalised in / out contract (src/unet_convnext.py
    docstring; verified against `experiments/lco_baseline/fold0/submission.py`'s
    `raw = self.base(x)` call site, which this pipeline is otherwise identical
    to), `cond=False` (built unconditioned, matching how
    `data/checkpoints/c1d/unet_local_fold0.pth` was trained -- see
    experiments/c1d_capacity/notes.md) so no (Re, AoA) params argument needed.

    Compute device: always CUDA when available regardless of `device` (see
    `ExperimentTTT` docstring); the CPU-thread cap below (orchestrator
    instruction) covers whatever CPU-side glue (masking, host<->device
    transfer) remains either way.
    """
    torch.set_num_threads(4)
    torch.manual_seed(int(CONFIG["seed"]))
    compute_device = "cuda" if torch.cuda.is_available() else device
    # C11-A: base = a fresh N1 (SimVP-style) core, wrapped by GenericCtxNormCore
    # (fold/concat/unfold around the (B,64,H,W)->(B,60,H,W) contract), then given
    # causal (S1, S2, n) state by GenericCtxNormWrapper -- ckpt is the
    # c11a_finetune.py payload ({"model_state_dict": ..., "ctx_const": ...}), not
    # a plain state dict, so loaded directly rather than through
    # load_checkpoint_state (which only knows the plain/fp16-packed formats).
    raw_core = _C11A.build_core("n1_simvp")
    ckpt_path = _resolve_unet_ckpt(submission_dir)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    gcore = _C11A.GenericCtxNormCore(raw_core)     # matches the checkpoint's own wrapping
    gcore.load_state_dict(state)
    base = _C11A.GenericCtxNormWrapper(gcore.to(compute_device), payload["ctx_const"])
    base = base.to(compute_device)
    return ExperimentTTT(base, device, CONFIG)
