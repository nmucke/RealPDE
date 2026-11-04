"""Transolver conditioned on (Re, AoA) through FiLM, with occlusion-aware processing.

A variant of the vendored `Transolver_Structured_Mesh_3D` for Track 2 with two additions:

* **Parameter dependence.** Re and AoA go through a small MLP into a conditioning vector,
  and every block modulates its two LayerNorm outputs (scale, shift) and gates its two
  residual branches from that vector -- adaLN, as in DiT. The modulation layers are
  zero-initialised, so at initialisation the network is a plain Transolver and the
  conditioning is learned on top. The caller normally owns the parameters: true values
  on sim, or whatever an assimilation loop around the model settles on at evaluation
  time. Omitting them (`forward(x)` with `params=None`) falls back to `estimate_params`,
  the model-free read-off of `parameter_estimation.py` -- what the submission does when
  it has nothing better. Note the AoA estimator needs the occlusion mask, so on sim
  (which has none) the fallback returns AoA 0; pass the true values there.

* **Occlusion awareness.** The PIV window carries an exact-zero footprint (airfoil body +
  laser shadow). Tokens in that footprint are not zeroed or altered; they are excluded
  from every cross-token operation, so a valid token's output is provably independent
  of what the occluded pixels contain:

    - the two 3x3x3 convolutions inside physics attention become partial convolutions
      (masked taps dropped, remaining taps renormalised);
    - occluded tokens get zero slice weight, so they never enter a slice token.

  LayerNorm, the token MLP and the output head are per-token and need nothing. The keep
  mask is also appended to the input as a fourth channel so valid tokens know where the
  body and shadow are. The mask may be omitted too: `forward(x, keep=None)` reads the
  persistent footprint off the window with `masking.occlusion_mask`.

Layout: unlike `load_baseline.build_model`, which hands the vendored model
`(H, W, D) = (W_s, H_s, T)` and thereby scrambles the flat token index relative to the
window's own `(T, H_s, W_s)` order (the 3x3x3 kernel then mixes non-neighbouring points),
this model reshapes tokens as `(T, H_s, W_s)`, so the kernel is a true space-time
neighbourhood. Consequently the official checkpoints' convolution weights are not
meaningfully transferable; this model is trained from scratch.

Interface:

    model = TransolverFiLM()                              # baseline sizes by default
    params = estimate_params(x)                           # or the true (B, 2) [re, aoa]
    pred = model(x, params)                               # x: (B, T, 32, 64, 3) normalised
    pred = model(x)                                       # same, params estimated off x
    pred = model(x, params, keep=keep)                    # (B, 32, 64) or (B, T, 32, 64)

Pure torch/numpy/einops; imports `masking` and `parameter_estimation` from the flat
submission layout and the vendored `rpde_baselines`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import masking
import parameter_estimation as pe
from rpde_baselines.model.TRANSOLVER_libs.Physics_Attention import (
    Physics_Attention_Structured_Mesh_3D,
)
from rpde_baselines.model.TRANSOLVER_libs.Transolver_Structured_Mesh_3D import MLP

# Parameter features: log Re centred on the release range (3750-27975 -> roughly [-1, 1]),
# AoA centred on the 0-20 deg range.
LOG_RE_CENTER, LOG_RE_SCALE = 9.2, 1.0
AOA_CENTER, AOA_SCALE = 10.0, 10.0

# Normalised value of a physical zero for [u, v, p] (official train_real stats; p is
# zero-filled and its std is replaced by 1 in the evaluator, so it maps to 0).
ZERO_NORM = np.array([-masking.MEAN[0] / masking.STD[0],
                      -masking.MEAN[1] / masking.STD[1], 0.0], dtype=np.float32)


# --- helpers on the window ---------------------------------------------------------------

def param_features(params: torch.Tensor) -> torch.Tensor:
    """(B, 2) physical [re, aoa] -> (B, 2) features of order one."""
    re = params[:, 0].clamp_min(1.0)
    aoa = params[:, 1]
    return torch.stack([(torch.log(re) - LOG_RE_CENTER) / LOG_RE_SCALE,
                        (aoa - AOA_CENTER) / AOA_SCALE], dim=-1)


def estimate_params(x: torch.Tensor, discrete_aoa: bool = False) -> torch.Tensor:
    """Model-free [re, aoa] per batch element, (B, 2) float32 on `x.device`.

    `x` is a normalised window (B, T, 32, 64, 3). The estimators are numpy; the result
    carries no gradient. `discrete_aoa=True` snaps to the five trained angles.
    """
    est_aoa = pe.estimate_aoa_discrete if discrete_aoa else pe.estimate_aoa
    arr = x.detach().cpu().numpy()
    out = np.array([[pe.estimate_re(w), est_aoa(w)] for w in arr], dtype=np.float32)
    return torch.from_numpy(out).to(x.device)


def keep_from_state(x: torch.Tensor, threshold: float = 0.5,
                    robust: bool = False) -> torch.Tensor:
    """Persistent keep mask per batch element, (B, 32, 64) float32 on `x.device`,
    1 = valid, 0 = occluded. Majority vote over the window (see `masking.py`)."""
    fn = masking.occlusion_mask_robust if robust else masking.occlusion_mask
    arr = x.detach().cpu().numpy()
    occ = np.stack([fn(w) if robust else fn(w, threshold=threshold) for w in arr])
    return torch.from_numpy((~occ).astype(np.float32)).to(x.device)


def jitter_params(params: torch.Tensor, re_rel: float = 0.02,
                  aoa_deg: float = 1.2) -> torch.Tensor:
    """Training-time noise matched to the estimators' leave-one-out error: lognormal on
    Re (relative std `re_rel`), Gaussian on AoA (`aoa_deg` degrees)."""
    re = params[:, 0] * torch.exp(re_rel * torch.randn_like(params[:, 0]))
    aoa = params[:, 1] + aoa_deg * torch.randn_like(params[:, 1])
    return torch.stack([re, aoa], dim=-1)


def fill_occluded(pred: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Write the normalised physical zero into occluded pixels of a prediction.

    The error metrics are computed over the whole field and the reference is exactly
    zero there, so this is the known answer, not a model output. `pred` is
    (B, T, 32, 64, C); `keep` is (B, 32, 64) or (B, T, 32, 64)."""
    keep = _expand_keep(keep, pred.shape[1]).unsqueeze(-1)  # (B, T, H, W, 1)
    fill = torch.as_tensor(ZERO_NORM[: pred.shape[-1]], device=pred.device, dtype=pred.dtype)
    return pred * keep + fill * (1.0 - keep)


def _expand_keep(keep: torch.Tensor, n_steps: int) -> torch.Tensor:
    """(B, H, W) or (B, T, H, W) -> (B, T, H, W) float."""
    keep = keep.float()
    if keep.dim() == 3:
        keep = keep.unsqueeze(1).expand(-1, n_steps, -1, -1)
    return keep


# --- masked physics attention ------------------------------------------------------------

class MaskedPhysicsAttention(Physics_Attention_Structured_Mesh_3D):
    """Vendored physics attention with the mask applied at both cross-token steps.

    Same parameters as the parent (state_dict-compatible). `forward` takes the mask in
    three precomputed forms, shared by every block of the model:
      keep_grid  (B, 1, H, W, D)  the token grid layout of the convolutions
      keep_flat  (B, N)           the flat token layout of the slicing
      ratio      (B, 1, H, W, D)  partial-convolution renormalisation
    """

    def _partial_conv(self, conv: nn.Conv3d, x: torch.Tensor, ratio: torch.Tensor):
        # x is already multiplied by keep_grid; ratio = (#taps) / (#valid taps).
        out = F.conv3d(x, conv.weight, None, conv.stride, conv.padding) * ratio
        return out + conv.bias.view(1, -1, 1, 1, 1)

    def forward(self, x, keep_grid, keep_flat, ratio):  # noqa: D102
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).permute(0, 4, 1, 2, 3).contiguous()
        x = x * keep_grid

        # (1) Slice -- occluded tokens carry zero slice weight.
        fx_mid = self._partial_conv(self.in_project_fx, x, ratio)
        fx_mid = fx_mid.permute(0, 2, 3, 4, 1).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self._partial_conv(self.in_project_x, x, ratio)
        x_mid = x_mid.permute(0, 2, 3, 4, 1).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))
        slice_weights = slice_weights * keep_flat[:, None, :, None]  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-5)[:, :, :, None]

        # (2) Attention among slice tokens -- unchanged.
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.softmax(dots))
        out_slice_token = torch.matmul(attn, v)  # B H G D

        # (3) Deslice -- occluded tokens receive something too; nothing reads it across
        # tokens, and the loss / post-processing ignore it.
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


# --- FiLM block --------------------------------------------------------------------------

def _modulate(h: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FiLMTransolverBlock(nn.Module):
    """`Transolver_block` with adaLN modulation from a conditioning vector.

    Per block, `cond -> SiLU -> Linear` yields six vectors: (scale, shift, gate) for the
    attention branch and for the MLP branch. Zero-initialised, so the block starts as
    the unconditioned one (gates enter as `1 + gate`).
    """

    def __init__(self, num_heads, hidden_dim, cond_dim, dropout, act="gelu", mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32, H=32, W=32, D=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = MaskedPhysicsAttention(hidden_dim, heads=num_heads,
                                           dim_head=hidden_dim // num_heads,
                                           dropout=dropout, slice_num=slice_num,
                                           H=H, W=W, D=D)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0,
                       res=False, act=act)
        n_mod = 6
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
            n_mod = 8
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, n_mod * hidden_dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, fx, cond, keep_grid, keep_flat, ratio):
        mod = self.modulation(cond).chunk(8 if self.last_layer else 6, dim=-1)
        s1, b1, g1, s2, b2, g2 = mod[:6]
        h = _modulate(self.ln_1(fx), s1, b1)
        fx = fx + (1.0 + g1.unsqueeze(1)) * self.Attn(h, keep_grid, keep_flat, ratio)
        h = _modulate(self.ln_2(fx), s2, b2)
        fx = fx + (1.0 + g2.unsqueeze(1)) * self.mlp(h)
        if self.last_layer:
            s3, b3 = mod[6:]
            return self.mlp2(_modulate(self.ln_3(fx), s3, b3))
        return fx


# --- model -------------------------------------------------------------------------------

class TransolverFiLM(nn.Module):
    """Transolver on (T, H_s, W_s) space-time tokens, FiLM-conditioned on (Re, AoA) and
    blind to occluded pixels. Defaults reproduce the official baseline's sizes."""

    def __init__(self, n_steps=20, height=32, width=64, in_channels=3, out_dim=3,
                 n_layers=3, n_hidden=256, n_head=8, slice_num=16, mlp_ratio=4,
                 dropout=0.1, act="gelu", cond_hidden=None,
                 mask_threshold=0.5, robust_mask=False, discrete_aoa=False,
                 use_mask=True, z_dim=0, noise_channels=0, noise_coarse=(5, 8, 16),
                 si_state=False):
        super().__init__()
        self.__name__ = "Transolver_FiLM"
        self.T, self.Hs, self.Ws = n_steps, height, width
        self.in_channels, self.out_dim = in_channels, out_dim
        self.n_hidden = n_hidden
        self.mask_threshold, self.robust_mask = mask_threshold, robust_mask
        # `use_mask=False` treats every pixel as valid: the partial convolutions
        # become ordinary ones, no token is dropped from a slice, and the mask
        # channel is a constant. The parameter shapes are unchanged either way, so
        # the two variants' checkpoints stay interchangeable.
        self.use_mask = use_mask
        # Used only when `forward` is called without `params`; see `estimate_params`.
        self.discrete_aoa = discrete_aoa

        cond_hidden = cond_hidden or n_hidden
        self.cond_mlp = nn.Sequential(nn.Linear(2, cond_hidden), nn.SiLU(),
                                      nn.Linear(cond_hidden, cond_hidden), nn.SiLU())
        # +1 input channel: the keep mask.
        self.preprocess = MLP(in_channels + 1, n_hidden * 2, n_hidden, n_layers=0,
                              res=False, act=act)
        self.blocks = nn.ModuleList([
            FiLMTransolverBlock(num_heads=n_head, hidden_dim=n_hidden, cond_dim=cond_hidden,
                                dropout=dropout, act=act, mlp_ratio=mlp_ratio,
                                out_dim=out_dim, slice_num=slice_num,
                                H=n_steps, W=height, D=width,
                                last_layer=(i == n_layers - 1))
            for i in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))
        self.kernel = self.blocks[0].Attn.in_project_x.kernel_size[0]

        # --- stochastic pathway (CRPS ensembles); both default to off ---------------
        # `z_dim=0, noise_channels=0` leaves the model bit-for-bit deterministic and
        # adds no state-dict keys, so existing checkpoints stay exactly valid. Both
        # output layers are zero-initialised, so even when enabled the model starts
        # as the deterministic one and the spread is learned from there.
        self.z_dim, self.noise_channels = z_dim, noise_channels
        self.noise_coarse = tuple(noise_coarse)
        if z_dim:
            # AIFS-CRPS's recipe: noise -> 2-layer MLP -> LayerNorm -> conditioning.
            self.z_proj = nn.Sequential(
                nn.Linear(z_dim, cond_hidden), nn.SiLU(),
                nn.Linear(cond_hidden, cond_hidden), nn.LayerNorm(cond_hidden),
                nn.Linear(cond_hidden, cond_hidden))
            nn.init.zeros_(self.z_proj[-1].weight)
            nn.init.zeros_(self.z_proj[-1].bias)
        if noise_channels:
            # NeuralGCM-style correlated random field: white noise on a coarse grid,
            # trilinearly upsampled, so the perturbation has spatial structure.
            self.tok_noise = nn.Linear(noise_channels, n_hidden)
            nn.init.zeros_(self.tok_noise.weight)
            nn.init.zeros_(self.tok_noise.bias)

        # --- pseudo-state pathway (denoising stochastic interpolant) ---------------
        # A second, dedicated token embedding for the SDE pseudo-state X_tau, summed
        # with the embedding of the conditioning window -- Xaurora's treatment of the
        # extra denoiser input (Sec. 5.2). Zero-initialised at the output, so at
        # initialisation the model ignores X_tau and reproduces the deterministic
        # prior exactly; that is the warm start the DSI parameterisation is chosen for.
        # `si_state=False` adds no keys, so existing checkpoints stay valid.
        self.state_preprocess = None
        if si_state:
            self.state_preprocess = MLP(out_dim, n_hidden * 2, n_hidden, n_layers=0,
                                        res=False, act=act)
            self.state_preprocess.apply(self._init_weights)
            nn.init.zeros_(self.state_preprocess.linear_post.weight)
            nn.init.zeros_(self.state_preprocess.linear_post.bias)

    @property
    def stochastic(self) -> bool:
        return bool(self.z_dim or self.noise_channels)

    def draw_z(self, batch: int, device, dtype=torch.float32) -> Optional[torch.Tensor]:
        """(B, z_dim) latent, or None when the conditioning noise is off."""
        return torch.randn(batch, self.z_dim, device=device, dtype=dtype) if self.z_dim else None

    def draw_tok_noise(self, batch: int, device, dtype=torch.float32) -> Optional[torch.Tensor]:
        """(B, *noise_coarse, noise_channels) coarse field, or None when off."""
        if not self.noise_channels:
            return None
        return torch.randn(batch, *self.noise_coarse, self.noise_channels,
                           device=device, dtype=dtype)

    def _token_noise(self, tok_n: torch.Tensor, T: int) -> torch.Tensor:
        """(B, *coarse, k) -> (B, T*Hs*Ws, n_hidden), trilinearly upsampled."""
        b = tok_n.shape[0]
        n = tok_n.permute(0, 4, 1, 2, 3)                       # (B, k, t, h, w)
        n = F.interpolate(n, size=(T, self.Hs, self.Ws), mode="trilinear",
                          align_corners=False)
        n = n.permute(0, 2, 3, 4, 1).reshape(b, -1, self.noise_channels)
        return self.tok_noise(n)

    # Same scheme as the vendored model; the zero-init of the modulation layers is
    # applied afterwards inside each block's __init__ ... except that `apply` runs
    # here, after the blocks exist, so re-zero them.
    def initialize_weights(self):
        self.apply(self._init_weights)
        for blk in self.blocks:
            nn.init.zeros_(blk.modulation[1].weight)
            nn.init.zeros_(blk.modulation[1].bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # -- mask plumbing --
    def _mask_forms(self, keep: torch.Tensor):
        """(B, T, H, W) -> (keep_grid, keep_flat, ratio) as the attention wants them."""
        B = keep.shape[0]
        keep_grid = keep.reshape(B, 1, self.T, self.Hs, self.Ws)
        keep_flat = keep.reshape(B, -1)
        k = self.kernel
        ones = torch.ones(1, 1, k, k, k, device=keep.device, dtype=keep.dtype)
        n_valid = F.conv3d(keep_grid, ones, padding=k // 2)
        n_full = F.conv3d(torch.ones_like(keep_grid), ones, padding=k // 2)
        ratio = torch.where(n_valid > 0, n_full / n_valid.clamp_min(1.0),
                            torch.zeros_like(n_valid))
        return keep_grid, keep_flat, ratio

    def forward(self, x: torch.Tensor, params: Optional[torch.Tensor] = None,
                keep: Optional[torch.Tensor] = None, z: Optional[torch.Tensor] = None,
                tok_n: Optional[torch.Tensor] = None,
                x_tau: Optional[torch.Tensor] = None,
                cond_extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, H_s, W_s, C_in) normalised window.
        params: (B, 2) physical [re, aoa], or None to read them off `x` with
                `estimate_params` (the model-free estimate the submission uses at
                evaluation time). Passing them explicitly keeps the caller in
                control -- true values on sim, an assimilated pair at eval time.
        keep: (B, H_s, W_s) or (B, T, H_s, W_s), 1 = valid, or None to read off `x`.
        x_tau: (B, T, H_s, W_s, out_dim) pseudo-state, embedded by `state_preprocess`
               and summed with the window's own embedding (needs `si_state=True`).
        cond_extra: (B, cond_hidden) added to the conditioning vector -- how the
               pseudo-time enters in `denoising_si.py`.
        Returns (B, T, H_s, W_s, out_dim)."""
        B, T, Hs, Ws, C = x.shape
        assert (T, Hs, Ws, C) == (self.T, self.Hs, self.Ws, self.in_channels), x.shape
        if params is None:
            params = estimate_params(x, discrete_aoa=self.discrete_aoa)
        assert params.shape == (B, 2), params.shape

        if keep is None:
            keep = self._default_keep(x)
        keep = _expand_keep(keep, T).to(device=x.device, dtype=x.dtype)  # (B, T, H, W)
        keep_grid, keep_flat, ratio = self._mask_forms(keep)

        cond = self.cond_mlp(param_features(params.to(x.device, x.dtype)))  # (B, cond)
        if self.z_dim:
            if z is None:
                z = self.draw_z(B, x.device, x.dtype)
            cond = cond + self.z_proj(z.to(x.device, x.dtype))
        if cond_extra is not None:
            cond = cond + cond_extra.to(x.device, x.dtype)

        fx = torch.cat([x, keep.unsqueeze(-1)], dim=-1).reshape(B, -1, C + 1)
        fx = self.preprocess(fx) + self.placeholder[None, None, :]
        if x_tau is not None:
            if self.state_preprocess is None:
                raise ValueError("x_tau needs a model built with si_state=True")
            assert x_tau.shape == (B, T, Hs, Ws, self.out_dim), x_tau.shape
            fx = fx + self.state_preprocess(x_tau.reshape(B, -1, self.out_dim))
        if self.noise_channels:
            if tok_n is None:
                tok_n = self.draw_tok_noise(B, x.device, x.dtype)
            fx = fx + self._token_noise(tok_n.to(x.device, x.dtype), T)
        for block in self.blocks:
            fx = block(fx, cond, keep_grid, keep_flat, ratio)
        return fx.reshape(B, T, Hs, Ws, self.out_dim)

    def _default_keep(self, x: torch.Tensor) -> torch.Tensor:
        """The keep mask to use when the caller passes none: read off `x`, or all
        ones when `use_mask=False`."""
        if not self.use_mask:
            return x.new_ones(x.shape[0], self.Hs, self.Ws)
        return keep_from_state(x, threshold=self.mask_threshold, robust=self.robust_mask)

    def train_loss(self, input, target, params=None, keep=None):
        """Per-element squared error, masked to valid pixels (mirrors the vendored API)."""
        if keep is None:
            keep = self._default_keep(input)
        pred = self.forward(input, params, keep=keep)
        w = _expand_keep(keep, input.shape[1]).to(pred).unsqueeze(-1)
        return ((pred - target) ** 2) * w
