"""Minimal supervised training loop for the Track 2 models.

    trainer = Trainer(model, train_loader, val_loader, lr=1e-4, epochs=10, device="cuda",
                      ckpt_path="data/ckpt/film_sim.pth")
    trainer.train()

Batches come from `datasets.PairDataset`: `(state_input, state_target, params)` with the
window pair normalised by the official stats and `params = [re, aoa]`. The loss is

    MSE(pred, target) + tke_weight * TKE(pred, target)

where `TKE` is the differentiable twin of `scoring.tke_rel_l2_per_sample`: per window,
the relative L2 between the turbulent-kinetic-energy maps of prediction and target,
`ke = 0.5 * (var_t(u) + var_t(v))` per pixel. The scorer works in physical units, so the
normalised variances are rescaled by the channel stds before summing (u fluctuates ~6x
more than v in m/s). `tke_weight=0` (the default) is plain MSE.

`estimate_params=True` ignores the loader's true `[re, aoa]` and conditions the model on
the model-free estimates read off each input window with `parameter_estimation.py`
(`estimate_re`, continuous `estimate_aoa`) -- what the submission has to do at
evaluation time. Meant for the real-data stage: on sim the AoA estimator has no
occlusion mask to read and returns 0.

Checkpoints use the same layout as the official ones (`model_state_dict` plus loss
history), so `load_baseline.load_checkpoint_state` reads them. The best-validation
weights are written to `ckpt_path`; the latest epoch to `<ckpt_path>.last`.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import parameter_estimation as pe
from transolver_film import _expand_keep as _expand_keep_t

# Official train_real [u, v] stds (mean_std_real.pt; input and target stats coincide).
TARGET_STD = (0.09680565, 0.01596068)


def kinetic_energy(x: torch.Tensor, std=TARGET_STD) -> torch.Tensor:
    """`scoring.kinetic_energy` in torch, on normalised (B, T, H, W, C) windows.

    Returns (B, H, W): 0.5 * (var_t(u) + var_t(v)) in physical units, the variance of
    each normalised channel being scaled by its std squared (means cancel)."""
    u_var = x[..., 0].var(dim=1, unbiased=False) * std[0] ** 2
    v_var = x[..., 1].var(dim=1, unbiased=False) * std[1] ** 2
    return 0.5 * (u_var + v_var)


def tke_rel_l2(pred: torch.Tensor, target: torch.Tensor, std=TARGET_STD,
               eps: float = 1e-8) -> torch.Tensor:
    """Per-window relative L2 between TKE maps, averaged over the batch -- the error
    behind `tke_score`, differentiable in `pred`."""
    p = kinetic_energy(pred, std).flatten(1)
    t = kinetic_energy(target, std).flatten(1)
    return ((p - t).norm(dim=1) / t.norm(dim=1).clamp_min(eps)).mean()


# --- spectral loss -----------------------------------------------------------------
# Cache of radial-bin indices per (H, W, device); the grid is fixed at 32x64 in practice.
_BIN_CACHE: dict = {}


def _radial_bins(h: int, w: int, device):
    """(bin index per rfft2 coefficient, per-bin counts, n_bins) for an h x w field.

    Wavenumbers are in cycles-per-domain, so `k = 0` is the mean and `k = 1` is one
    wavelength across the box. Coefficients are binned by `round(|k|)`.
    """
    key = (h, w, str(device))
    if key not in _BIN_CACHE:
        ky = torch.fft.fftfreq(h, device=device) * h
        kx = torch.fft.rfftfreq(w, device=device) * w
        k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        idx = k.round().long().flatten()
        n_bins = int(idx.max().item()) + 1
        counts = torch.bincount(idx, minlength=n_bins).clamp_min(1).to(torch.float32)
        _BIN_CACHE[key] = (idx, counts, n_bins)
    return _BIN_CACHE[key]


def energy_spectrum(x: torch.Tensor, std=TARGET_STD) -> torch.Tensor:
    """Radially averaged energy spectrum of the fluctuations, (B, n_bins).

    `x` is a normalised (B, T, H, W, C) window. Each channel is put back on its
    physical scale, its temporal mean removed -- so this is the spectrum of the
    turbulent part, the same decomposition `kinetic_energy` uses -- and the 2D
    spatial power averaged over time and over each radial wavenumber shell.
    """
    u = (x[..., 0] * std[0])
    v = (x[..., 1] * std[1])
    u = u - u.mean(dim=1, keepdim=True)
    v = v - v.mean(dim=1, keepdim=True)
    fu = torch.fft.rfft2(u, norm="ortho")
    fv = torch.fft.rfft2(v, norm="ortho")
    e = 0.5 * (fu.real ** 2 + fu.imag ** 2 + fv.real ** 2 + fv.imag ** 2)
    e = e.mean(dim=1)                                   # (B, H, W//2+1), time-averaged
    idx, counts, n_bins = _radial_bins(x.shape[2], x.shape[3], x.device)
    flat = e.reshape(e.shape[0], -1)
    out = torch.zeros(e.shape[0], n_bins, device=x.device, dtype=flat.dtype)
    out = out.index_add(1, idx, flat)
    return out / counts


def spectrum_log_mse(pred: torch.Tensor, target: torch.Tensor, std=TARGET_STD,
                     eps_frac: float = 1e-8) -> torch.Tensor:
    """Mean squared error between log energy spectra, averaged over the batch.

    Working in the log makes every wavenumber shell count equally. That is the whole
    point: the raw spectrum is dominated by the largest scales, so a plain L2 on it
    would say nothing about the small scales, which is exactly where an over-smooth
    prediction loses energy -- and losing that energy is what shows up as a poor
    `tke_score`. Scale-free, so it does not compete with the MSE for magnitude.

    `eps` is set from the target's own mean power, keeping the floor meaningful for
    both channels without a hand-tuned absolute constant.
    """
    sp = energy_spectrum(pred, std)
    st = energy_spectrum(target, std)
    eps = eps_frac * st.mean(dim=1, keepdim=True).clamp_min(1e-30)
    return ((torch.log(sp + eps) - torch.log(st + eps)) ** 2).mean()


# --- CRPS ---------------------------------------------------------------------------

def crps_loss(samples: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor | None = None, alpha: float = 0.95) -> torch.Tensor:
    """Per-element almost-fair CRPS. `samples` (M, B, ...), `target` (B, ...).

        CRPS_af = (1/M) Σ_i |x_i - y|
                  - [ alpha/(2M(M-1)) + (1-alpha)/(2M²) ] Σ_{i,j} |x_i - x_j|

    `alpha=1` is the fair (unbiased) estimator, `alpha=0` the biased one whose
    minimiser is under-dispersed at finite M; 0.95 is AIFS-CRPS's published value,
    close to fair while avoiding the degeneracy of the fully fair loss early on.

    The pairwise sum uses the sorted identity Σ_{i<j}|x_i - x_j| = Σ_i (2i-M+1)·x_(i),
    which is O(M log M) rather than materialising an M×M tensor over 122880 elements
    per member. Computed in float32 even under autocast: it is a difference of two
    comparable positive quantities and cancels badly in low precision.
    """
    m = samples.shape[0]
    if m < 2:
        raise ValueError("CRPS needs at least 2 members")
    samples = samples.float()
    target = target.float()
    ae = (samples - target.unsqueeze(0)).abs().mean(0)
    xs, _ = torch.sort(samples, dim=0)
    w = (2 * torch.arange(m, device=samples.device, dtype=xs.dtype) - (m - 1))
    pw = (w.view(m, *([1] * (xs.dim() - 1))) * xs).sum(0)
    coef = alpha / (m * (m - 1)) + (1.0 - alpha) / m ** 2
    crps = ae - coef * pw
    if mask is None:
        return crps.mean()
    mask = mask.float()
    return (crps * mask).sum() / mask.sum().clamp_min(1.0)


def fair_mse(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE of the ensemble mean, debiased by the sampling variance of that mean.

    E‖x̄_M - y‖² = ‖mu - y‖² + sigma²/M, so a plain MSE on the empirical mean pays a
    bonus for shrinking the spread -- an anti-dispersion force pulling straight
    against the CRPS. Subtracting s²/M with the unbiased s² removes it.
    """
    m = samples.shape[0]
    mean = samples.mean(0)
    var = samples.var(0, unbiased=True)
    return F.mse_loss(mean, target) - var.mean() / m


def spread_skill(samples: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor | None = None) -> tuple[float, float]:
    """(SSR, mean spread) -- the standard ensemble calibration diagnostic.

    SSR = sqrt((M+1)/M) · spread / RMSE(ensemble mean); 1.0 is calibrated, below ~0.8
    under-dispersed, and a value collapsing toward 0 means the model is ignoring the
    noise. CRPS alone cannot distinguish a calibrated ensemble from an over-dispersed
    one, so this is logged every epoch.
    """
    m = samples.shape[0]
    with torch.no_grad():
        var = samples.var(0, unbiased=True)
        se = (samples.mean(0) - target) ** 2
        if mask is not None:
            w = mask.float()
            denom = w.sum().clamp_min(1.0)
            spread = ((var * w).sum() / denom).sqrt()
            rmse = ((se * w).sum() / denom).sqrt()
        else:
            spread, rmse = var.mean().sqrt(), se.mean().sqrt()
        ssr = math.sqrt((m + 1) / m) * spread / rmse.clamp_min(1e-12)
    return float(ssr), float(spread)


def estimate_params(x: torch.Tensor) -> torch.Tensor:
    """Model-free [re, aoa] per window of a normalised (B, T, H, W, C) batch, (B, 2)
    float32 on `x.device`, no gradient."""
    arr = x.detach().cpu().numpy()
    est = np.array([[pe.estimate_re(w), pe.estimate_aoa(w)] for w in arr], dtype=np.float32)
    return torch.from_numpy(est).to(x.device)


class Trainer:
    def __init__(self, model, train_loader, val_loader=None, lr=1e-4, epochs=10,
                 weight_decay=0.0, tke_weight=0.0, spectral_weight=0.0,
                 warmup_frac=0.03, min_lr_frac=0.0, estimate_params=False, device="cpu",
                 ckpt_path=None, log_every=50, loss="mse", n_members=4, crps_alpha=0.95,
                 mse_weight=0.0, amp=False, amp_dtype="bf16", freeze_noise_steps=0,
                 freeze_bn=False):
        self.model = model.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.epochs, self.device = epochs, device
        self.tke_weight, self.spectral_weight = tke_weight, spectral_weight
        self.estimate_params = estimate_params
        self.ckpt_path, self.log_every = ckpt_path, log_every
        assert loss in ("mse", "crps"), loss
        self.loss_kind, self.n_members = loss, n_members
        self.crps_alpha, self.mse_weight = crps_alpha, mse_weight
        if loss == "crps" and not getattr(model, "stochastic", False):
            raise ValueError("loss='crps' needs a stochastic model: build it with "
                             "z_dim>0 and/or noise_channels>0, else every member is "
                             "identical and the spread term is exactly zero.")

        # Mixed precision. bf16 needs no GradScaler (same exponent range as fp32), so
        # it is the default; fp16 gets one. Losses are computed in float32 regardless
        # -- see `crps_loss` -- so only the forward/backward matmuls are reduced.
        self.amp = bool(amp) and str(device).startswith("cuda")
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp and self.amp_dtype is torch.float16)
        if self.amp:
            print(f"mixed precision: {amp_dtype}"
                  + ("" if self.amp_dtype is torch.bfloat16 else " + GradScaler"),
                  flush=True)
        trainable = [p for p in model.parameters() if p.requires_grad]  # LoRA-aware
        n_all = sum(p.numel() for p in model.parameters())
        n_tr = sum(p.numel() for p in trainable)
        size_mib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
        print(f"model: {n_all:,} parameters ({n_tr:,} trainable), {size_mib:.1f} MiB", flush=True)
        self.opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        # Linear warmup then cosine decay, stepped per batch. Warmup matters here
        # because the modulation layers are zero-initialised: the conditioning path
        # carries no signal at step 0, and a full-size first step on a cold Adam
        # state is what tends to knock it off course.
        total = max(1, epochs * len(train_loader))
        warmup = min(int(warmup_frac * total), total - 1) if warmup_frac > 0 else 0

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            p = (step - warmup) / max(1, total - warmup)
            return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        print(f"schedule: {total} steps, {warmup} warmup ({100 * warmup / total:.1f} %), "
              f"cosine to {min_lr_frac:g} x lr", flush=True)
        # Phase A of the CRPS warm start: grow the spread with only the noise heads
        # trainable, so it reaches roughly the right magnitude before the mean is
        # allowed to move. Without it the spread chases a mean that is still shifting.
        # Fine-tuning a pretrained model on batches of 8, with noise injected into
        # the activations, drags the BatchNorm running statistics away from the ones
        # the checkpoint was trained with -- and they are then used at eval, so the
        # train/eval gap grows as the noise does. Holding BN in eval mode keeps the
        # pretrained statistics and removes that gap entirely.
        self.freeze_bn = freeze_bn
        if freeze_bn:
            n_bn = sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                       for m in model.modules())
            print(f"freezing {n_bn} BatchNorm layers (running stats kept, eval mode)",
                  flush=True)

        self.freeze_noise_steps, self.global_step = freeze_noise_steps, 0
        self._noise_names = tuple(n for n, _ in model.named_parameters()
                                  if n.startswith(("z_proj", "tok_noise")))
        if freeze_noise_steps > 0:
            if not self._noise_names:
                raise ValueError("--freeze-noise-steps needs a stochastic model")
            self._set_base_trainable(False)
            print(f"phase A: {freeze_noise_steps} steps with only "
                  f"{len(self._noise_names)} noise tensors trainable", flush=True)

        self.train_losses, self.val_losses = [], []
        self.best_val, self.best_epoch = float("inf"), -1

    def _set_base_trainable(self, flag: bool):
        """Freeze/unfreeze everything except the noise heads."""
        for name, p in self.model.named_parameters():
            if name not in self._noise_names:
                p.requires_grad_(flag)

    def _keep_for(self, x):
        """The keep mask the model would compute, materialised once so the M members
        share it (and so the CRPS can be averaged over valid pixels only)."""
        m = self.model
        keep = m._default_keep(x) if hasattr(m, "_default_keep") else None
        return keep

    def _loss(self, batch):
        """Total loss, plus (mse, tke, spectral, crps, ssr) floats for logging."""
        x, y, params = (t.to(self.device) for t in batch)
        if self.estimate_params:
            params = estimate_params(x)

        if self.loss_kind == "mse":
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp):
                pred = self.model(x, params)
            pred = pred.float()
            mse = F.mse_loss(pred, y)
            total, tke_v, sp_v = mse, 0.0, 0.0
            if self.tke_weight != 0.0:
                tke = tke_rel_l2(pred, y)
                total, tke_v = total + self.tke_weight * tke, tke.item()
            if self.spectral_weight != 0.0:
                sp = spectrum_log_mse(pred, y)
                total, sp_v = total + self.spectral_weight * sp, sp.item()
            return total, (mse.item(), tke_v, sp_v, 0.0, 0.0)

        # --- CRPS: M members in ONE forward, repeated along the batch axis. ---------
        # Members must not see each other -- any cross-member information flow breaks
        # the fairness of the estimator and over-rewards spread (Roberts 2026) -- and
        # stacking on the batch axis guarantees that, since the physics attention
        # never crosses batch elements.
        m, b = self.n_members, x.shape[0]
        keep = self._keep_for(x)
        xr = x.repeat(m, *([1] * (x.dim() - 1)))
        pr = params.repeat(m, 1)
        kr = keep.repeat(m, *([1] * (keep.dim() - 1))) if keep is not None else None
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp):
            out = self.model(xr, pr, keep=kr)
        samples = out.float().view(m, b, *out.shape[1:])

        wmask = None
        if keep is not None:
            wmask = _expand_keep_t(keep, x.shape[1]).unsqueeze(-1)  # (B,T,H,W,1)
        crps = crps_loss(samples, y, mask=wmask, alpha=self.crps_alpha)
        total = crps
        mean = samples.mean(0)
        mse_v = F.mse_loss(mean, y).item()
        tke_v, sp_v = 0.0, 0.0
        if self.mse_weight != 0.0:
            total = total + self.mse_weight * fair_mse(samples, y)
        if self.tke_weight != 0.0:
            tke = tke_rel_l2(mean, y)
            total, tke_v = total + self.tke_weight * tke, tke.item()
        if self.spectral_weight != 0.0:
            sp = spectrum_log_mse(mean, y)
            total, sp_v = total + self.spectral_weight * sp, sp.item()
        ssr, _ = spread_skill(samples, y, wmask)
        return total, (mse_v, tke_v, sp_v, crps.item(), ssr)

    def _set_train_mode(self):
        self.model.train()
        if self.freeze_bn:
            for m in self.model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()

    def train_epoch(self, epoch):
        self._set_train_mode()
        total, mse_sum, tke_sum, sp_sum = 0.0, 0.0, 0.0, 0.0
        crps_sum, ssr_sum, t0 = 0.0, 0.0, time.time()
        for i, batch in enumerate(self.train_loader, 1):
            loss, (mse, tke, sp, crps, ssr) = self._loss(batch)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            self.global_step += 1
            if self.freeze_noise_steps and self.global_step == self.freeze_noise_steps:
                self._set_base_trainable(True)
                print(f"  phase B: unfroze the base model at step "
                      f"{self.global_step}", flush=True)
            total, mse_sum = total + loss.item(), mse_sum + mse
            tke_sum, sp_sum = tke_sum + tke, sp_sum + sp
            crps_sum, ssr_sum = crps_sum + crps, ssr_sum + ssr
            if self.log_every and i % self.log_every == 0:
                extra = (f"crps {crps_sum / i:.5f} ssr {ssr_sum / i:.3f} "
                         if self.loss_kind == "crps" else "")
                print(f"  epoch {epoch} step {i}/{len(self.train_loader)} "
                      f"loss {total / i:.5f} ({extra}mse {mse_sum / i:.5f} "
                      f"tke {tke_sum / i:.4f} spec {sp_sum / i:.4f}) "
                      f"lr {self.sched.get_last_lr()[0]:.2e} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        return total / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total = sum(self._loss(b)[0].item() for b in self.val_loader)
        return total / len(self.val_loader)

    def save(self, path, epoch):
        torch.save({"model_state_dict": self.model.state_dict(),
                    "train_losses": self.train_losses, "val_losses": self.val_losses,
                    "iteration": epoch, "best_iteration": self.best_epoch,
                    "best_val_loss": self.best_val}, path)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            tr = self.train_epoch(epoch)
            self.train_losses.append(tr)
            msg = f"epoch {epoch}/{self.epochs} train {tr:.5f}"
            if self.val_loader is not None:
                va = self.evaluate()
                self.val_losses.append(va)
                msg += f" val {va:.5f}"
                if va < self.best_val:
                    self.best_val, self.best_epoch = va, epoch
                    if self.ckpt_path:
                        self.save(self.ckpt_path, epoch)
                        msg += " *"
            elif self.ckpt_path:
                self.save(self.ckpt_path, epoch)
            if self.ckpt_path:
                self.save(self.ckpt_path + ".last", epoch)
            print(msg + f" ({time.time() - t0:.0f}s)", flush=True)
        return self.train_losses, self.val_losses
