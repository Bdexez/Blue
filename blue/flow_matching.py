"""Tête générative continue par Conditional Flow Matching (Rectified Flow + AdaLN-Zero)."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .modules import RMSNorm, SwiGLU


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        with torch.autocast(device_type=t.device.type, enabled=False):
            freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1))
            angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * (2.0 * math.pi)
            emb = torch.cat((angles.sin(), angles.cos()), dim=-1)
            if self.dim % 2:
                emb = F.pad(emb, (0, 1))
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class AdaLNZeroBlock(nn.Module):
    """Bloc résiduel SwiGLU modulé par AdaLN-Zero (shift, scale, gate)."""

    def __init__(self, dim: int, hidden: int, *, norm_eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(dim, norm_eps)
        self.ffn = SwiGLU(dim, hidden)
        self.modulation = nn.Linear(dim, 3 * dim)
        self.reset_adaln()

    def reset_adaln(self):
        with torch.no_grad():
            self.modulation.weight.zero_()
            self.modulation.bias.zero_()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(F.silu(cond)).chunk(3, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        return x + gate * self.ffn(h)


class ConditionalFlowHead(nn.Module):
    """Champ de vecteurs conditionnel v_theta(z_t, t | c) et intégrateurs ODE Euler/Heun."""

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.out_dim = config.flow_matching_dim
        self.time_embed = SinusoidalTimeEmbedding(d)
        self.cond_proj = nn.Linear(d, d, bias=False)
        self.in_proj = nn.Linear(self.out_dim, d)
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(d, config.hidden_dim, norm_eps=config.norm_eps)
            for _ in range(config.flow_matching_layers)
        ])
        self.final_norm = RMSNorm(d, config.norm_eps)
        self.final_mod = nn.Linear(d, 2 * d)
        self.out_proj = nn.Linear(d, self.out_dim)
        self.reset_adaln_parameters()

    def reset_adaln_parameters(self):
        for block in self.blocks:
            block.reset_adaln()
        with torch.no_grad():
            self.final_mod.weight.zero_()
            self.final_mod.bias.zero_()

    def velocity(self, z_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        cond = self.cond_proj(condition) + self.time_embed(t)
        h = self.in_proj(z_t.to(self.in_proj.weight.dtype)) + cond
        for block in self.blocks:
            h = block(h, cond)
        shift, scale = self.final_mod(F.silu(cond)).chunk(2, dim=-1)
        h = self.final_norm(h) * (1.0 + scale) + shift
        return self.out_proj(h)

    def loss(self, targets: torch.Tensor, condition: torch.Tensor, *,
             noise: torch.Tensor | None = None, t: torch.Tensor | None = None) -> torch.Tensor:
        if targets.ndim != 2 or targets.shape != (condition.size(0), self.out_dim):
            raise ValueError(f"flow_targets doit avoir la forme [B, {self.out_dim}]")
        valid_rows = torch.isfinite(targets).all(dim=-1)
        safe_targets = torch.where(valid_rows.unsqueeze(-1), targets, 0.0)
        b = targets.size(0)
        if noise is None:
            noise = torch.randn_like(safe_targets, dtype=torch.float32)
        if t is None:
            t = torch.rand(b, device=targets.device, dtype=torch.float32)
        with torch.autocast(device_type=targets.device.type, enabled=False):
            z0 = noise.float()
            z1 = safe_targets.float()
            tf = t.float()
            zt = (1.0 - tf[:, None]) * z0 + tf[:, None] * z1
            target_v = z1 - z0
        pred_v = self.velocity(zt.to(condition.dtype), tf, condition).float()
        per_row = (pred_v - target_v).square().mean(dim=-1)
        return (per_row * valid_rows.float()).sum() / valid_rows.sum().clamp_min(1)

    @torch.inference_mode()
    def sample(self, condition: torch.Tensor, *, steps: int = 20, solver: str = "heun",
               generator: torch.Generator | None = None,
               initial_noise: torch.Tensor | None = None) -> torch.Tensor:
        if type(steps) is not int or steps < 1:
            raise ValueError("steps doit être un entier strictement positif")
        if solver not in ("euler", "heun"):
            raise ValueError("solver doit être euler ou heun")
        b = condition.size(0)
        if initial_noise is not None:
            if initial_noise.shape != (b, self.out_dim) or initial_noise.device != condition.device:
                raise ValueError("initial_noise doit avoir la forme [B, flow_matching_dim] sur le même device")
            z = initial_noise.float().clone()
        else:
            z = torch.randn((b, self.out_dim), device=condition.device, dtype=torch.float32, generator=generator)
        dt = 1.0 / steps
        for step in range(steps):
            t0 = torch.full((b,), step * dt, device=condition.device, dtype=torch.float32)
            v0 = self.velocity(z.to(condition.dtype), t0, condition).float()
            if solver == "euler":
                z = z + dt * v0
            else:
                z_pred = z + dt * v0
                t1 = torch.full((b,), (step + 1) * dt, device=condition.device, dtype=torch.float32)
                v1 = self.velocity(z_pred.to(condition.dtype), t1, condition).float()
                z = z + 0.5 * dt * (v0 + v1)
        return z
