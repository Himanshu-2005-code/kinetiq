"""Kinetiq motion transformer.

Input is raw landmarks (B, T, 13, 3) straight from the pose estimator. All
feature engineering (centering, scale normalisation, velocities, joint angles)
lives inside the network, so the exported ONNX graph is the single source of
truth and the browser client never has to re-implement preprocessing.

Architecture: per-frame MLP embedding -> [CLS] + pre-norm transformer encoder
with rotary position embeddings, SwiGLU feed-forward and stochastic depth ->
four heads:
  * exercise   (CLS)      softmax over EXERCISES
  * faults     (CLS)      independent sigmoids over FAULTS
  * phase      (per-frame) unit vector (cos, sin) of rep progress
  * active     (per-frame) probability that a rep is in progress
Reps are counted downstream by unwrapping the phase angle (see reps.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton import ANGLE_TRIPLETS, EXERCISES, FAULTS, J, NUM_JOINTS

N_ANGLES = len(ANGLE_TRIPLETS)
FEATURE_DIM = NUM_JOINTS * 2 * 2 + NUM_JOINTS + N_ANGLES * 2 + 2 + 2 + 2


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_mult: float = 2.75
    dropout: float = 0.1
    drop_path: float = 0.1
    max_len: int = 256


class PoseFeaturizer(nn.Module):
    """Raw landmarks -> per-frame feature vectors (B, T, FEATURE_DIM)."""

    def __init__(self):
        super().__init__()
        a, b, c = zip(*ANGLE_TRIPLETS)
        self.register_buffer("ia", torch.tensor(a), persistent=False)
        self.register_buffer("ib", torch.tensor(b), persistent=False)
        self.register_buffer("ic", torch.tensor(c), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xy, vis = x[..., :2], x[..., 2]
        hip = (xy[:, :, J["left_hip"]] + xy[:, :, J["right_hip"]]) * 0.5          # (B,T,2)
        sh = (xy[:, :, J["left_shoulder"]] + xy[:, :, J["right_shoulder"]]) * 0.5
        torso_vec = sh - hip
        torso_len = torch.sqrt((torso_vec ** 2).sum(-1) + 1e-8)                    # (B,T)
        # Visibility-weighted mean torso length over the window is a robust body scale.
        w = torch.minimum(vis[:, :, J["left_hip"]], vis[:, :, J["left_shoulder"]]) + 1e-3
        scale = ((torso_len * w).sum(1) / w.sum(1)).clamp_min(1e-3)[:, None, None]   # (B,1,1)

        rel = (xy - hip[:, :, None]) / scale[..., None]                             # (B,T,J,2)
        vel = torch.cat([torch.zeros_like(rel[:, :1]), rel[:, 1:] - rel[:, :-1]], 1)

        va = xy[:, :, self.ia] - xy[:, :, self.ib]
        vc = xy[:, :, self.ic] - xy[:, :, self.ib]
        na = torch.sqrt((va ** 2).sum(-1) + 1e-8)
        nc = torch.sqrt((vc ** 2).sum(-1) + 1e-8)
        cos = (va * vc).sum(-1) / (na * nc)
        # |cross| keeps angles mirror-invariant (selfie cameras flip x).
        sin = torch.abs(va[..., 0] * vc[..., 1] - va[..., 1] * vc[..., 0]) / (na * nc)

        trunk = torso_vec / torso_len[..., None]
        trunk = torch.stack([trunk[..., 0].abs(), trunk[..., 1]], -1)
        root = (hip - hip.mean(1, keepdim=True)) / scale
        root = torch.stack([root[..., 0].abs(), root[..., 1]], -1)
        root_vel = torch.cat([torch.zeros_like(root[:, :1]), root[:, 1:] - root[:, :-1]], 1)

        B, T = x.shape[:2]
        feats = [rel.reshape(B, T, -1), vel.reshape(B, T, -1) * 10.0, vis, cos, sin, trunk,
                 root, root_vel * 10.0]
        return torch.cat(feats, -1)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(max_len: int, head_dim: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.arange(max_len).float()[:, None] * inv[None]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (T, D/2). Rotate interleaved halves.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        cos, sin = rope_tables(cfg.max_len, self.dh)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        cos, sin = self.cos[:T], self.sin[:T]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # Explicit attention (rather than SDPA) keeps the ONNX graph portable to onnxruntime-web.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dh))
        att = self.drop(att.softmax(-1))
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.d_model * cfg.ffn_mult)
        self.w12 = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.drop(self.w3(F.silu(a) * b))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, drop_path: float):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn, self.ffn = Attention(cfg), SwiGLU(cfg)
        self.drop_path = drop_path

    def _dp(self, x):
        if not self.training or self.drop_path == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path
        return x * keep / (1 - self.drop_path)

    def forward(self, x):
        x = x + self._dp(self.attn(self.n1(x)))
        return x + self._dp(self.ffn(self.n2(x)))


class KinetiqModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.featurizer = PoseFeaturizer()
        self.in_norm = nn.LayerNorm(FEATURE_DIM)
        self.embed = nn.Sequential(nn.Linear(FEATURE_DIM, cfg.d_model), nn.GELU(),
                                   nn.Linear(cfg.d_model, cfg.d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        dps = torch.linspace(0, cfg.drop_path, cfg.n_layers).tolist()
        self.blocks = nn.ModuleList([Block(cfg, dp) for dp in dps])
        self.norm = RMSNorm(cfg.d_model)
        self.head_exercise = nn.Linear(cfg.d_model, len(EXERCISES))
        self.head_faults = nn.Linear(cfg.d_model, len(FAULTS))
        self.head_phase = nn.Linear(cfg.d_model, 2)
        self.head_active = nn.Linear(cfg.d_model, 1)
        self.apply(self._init)
        nn.init.normal_(self.cls, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, landmarks: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.embed(self.in_norm(self.featurizer(landmarks)))
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], 1)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        cls, frames = h[:, 0], h[:, 1:]
        return {
            "exercise": self.head_exercise(cls),
            "faults": self.head_faults(cls),
            "phase": self.head_phase(frames),
            "active": self.head_active(frames).squeeze(-1),
        }


class InferenceWrapper(nn.Module):
    """Probabilities and normalised phase, as consumed by the web client."""

    def __init__(self, model: KinetiqModel):
        super().__init__()
        self.model = model

    def forward(self, landmarks):
        out = self.model(landmarks)
        phase = F.normalize(out["phase"], dim=-1)
        return (out["exercise"].softmax(-1), torch.sigmoid(out["faults"]), phase,
                torch.sigmoid(out["active"]))
