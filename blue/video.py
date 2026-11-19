"""Encodeur vidéo spatio-temporel par tubelets 3D et RoPE axial 3D (temps, hauteur, largeur)."""

import torch
from torch import nn
from torch.nn import functional as F

from .modules import RMSNorm, SwiGLU


class RotaryEmbedding3D(nn.Module):
    """Positions rotatives axiales 3D factorisées (temps t, hauteur y, largeur x)."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RotaryEmbedding3D nécessite head_dim pair")
        self.head_dim = head_dim
        self.theta = float(theta)
        if head_dim >= 6:
            self.d_t = 2 * (head_dim // 6)
            self.d_y = 2 * ((head_dim - self.d_t) // 4)
            self.d_x = head_dim - self.d_t - self.d_y
        else:
            self.d_t = head_dim
            self.d_y = 0
            self.d_x = 0

    def _inv_freq(self, dim: int, device: torch.device) -> torch.Tensor:
        exponent = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / max(dim, 2)
        return 1.0 / (self.theta ** exponent)

    @staticmethod
    def _rotate_subspace(part: torch.Tensor, coords: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
        phase = torch.outer(coords, inv_freq)
        cos = phase.cos().to(part.dtype)
        sin = phase.sin().to(part.dtype)
        even, odd = part[..., 0::2], part[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, n_frames: int, grid_size: int) -> torch.Tensor:
        # x: [B, H, F_t * G * G, head_dim]
        if self.d_y == 0:
            with torch.autocast(device_type=x.device.type, enabled=False):
                pos = torch.arange(x.size(-2), device=x.device, dtype=torch.float32)
                inv = self._inv_freq(self.d_t, x.device)
            return self._rotate_subspace(x, pos, inv)
        with torch.autocast(device_type=x.device.type, enabled=False):
            t_axis = torch.arange(n_frames, device=x.device, dtype=torch.float32)
            g_axis = torch.arange(grid_size, device=x.device, dtype=torch.float32)
            tt = t_axis.repeat_interleave(grid_size * grid_size)
            yy = g_axis.repeat_interleave(grid_size).repeat(n_frames)
            xx = g_axis.repeat(n_frames * grid_size)
            inv_t = self._inv_freq(self.d_t, x.device)
            inv_y = self._inv_freq(self.d_y, x.device)
            inv_x = self._inv_freq(self.d_x, x.device)
        p_t, p_y, p_x = x.split((self.d_t, self.d_y, self.d_x), dim=-1)
        return torch.cat((
            self._rotate_subspace(p_t, tt, inv_t),
            self._rotate_subspace(p_y, yy, inv_y),
            self._rotate_subspace(p_x, xx, inv_x),
        ), dim=-1)


class VideoTransformerBlock(nn.Module):
    """Bloc Transformer bidirectionnel spatio-temporel avec RoPE axial 3D et SwiGLU."""

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = d // config.n_heads
        self.norm1 = RMSNorm(d, config.norm_eps)
        self.norm2 = RMSNorm(d, config.norm_eps)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.rope3d = RotaryEmbedding3D(self.head_dim, config.rope_theta)
        self.out = nn.Linear(d, d, bias=False)
        self.ffn = SwiGLU(d, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.attn_dropout = config.dropout

    def forward(self, x: torch.Tensor, valid: torch.Tensor, n_frames: int, grid_size: int) -> torch.Tensor:
        b, n, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).view(b, n, 3, self.n_heads, self.head_dim).unbind(2)
        q = self.rope3d(self.q_norm(q.transpose(1, 2)), n_frames, grid_size)
        k = self.rope3d(self.k_norm(k.transpose(1, 2)), n_frames, grid_size)
        v = v.transpose(1, 2)
        mask = valid[:, None, None, :]
        dp = self.attn_dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)
        x = x + self.dropout(self.out(y.transpose(1, 2).contiguous().view(b, n, d)))
        return (x + self.dropout(self.ffn(self.norm2(x)))) * valid.unsqueeze(-1)


class VideoEncoder(nn.Module):
    """Encodeur vidéo 3D [B, C, F, H, W] -> jetons spatio-temporels [B, F_t * G^2, D].

    Supporte un nombre variable de frames via `video_mask` [B, F] et garantit
    une invariance stricte au padding temporel à droite.
    """

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.tubelet_size = config.video_tubelet_size
        self.patch_size = config.video_patch_size
        self.grid_size = config.video_size // config.video_patch_size
        # Stem spatial par frame (noyau temporel 1) pour préserver l'isolation des tubelets.
        self.stem = nn.Conv3d(config.video_channels, d // 2, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.tubelet = nn.Conv3d(
            d // 2, d,
            kernel_size=(self.tubelet_size, self.patch_size, self.patch_size),
            stride=(self.tubelet_size, self.patch_size, self.patch_size),
        )
        self.spatial_pos = nn.Parameter(torch.empty(1, self.grid_size * self.grid_size, d))
        self.blocks = nn.ModuleList([VideoTransformerBlock(config) for _ in range(config.video_layers)])
        self.norm = RMSNorm(d, config.norm_eps)
        self.reset_video_parameters()

    def reset_video_parameters(self):
        nn.init.normal_(self.spatial_pos, std=0.02)

    def forward(self, video: torch.Tensor, valid_frames: torch.Tensor):
        b, _, f, _, _ = video.shape
        x = video.to(self.stem.weight.dtype) * valid_frames[:, None, :, None, None].to(self.stem.weight.dtype)
        pad_f = (self.tubelet_size - (f % self.tubelet_size)) % self.tubelet_size
        if pad_f:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_f))
            valid_frames = F.pad(valid_frames, (0, pad_f), value=False)
        f_pad = f + pad_f
        n_t = f_pad // self.tubelet_size
        tubelet_valid = valid_frames.view(b, n_t, self.tubelet_size).any(dim=-1)  # [B, F_t]

        h = F.silu(self.stem(x)) * valid_frames[:, None, :, None, None].to(x.dtype)
        # [B, D, F_t, G, G] -> [B, F_t, G*G, D]
        tokens = self.tubelet(h).flatten(3).permute(0, 2, 3, 1)
        tokens = tokens + self.spatial_pos.to(tokens.dtype).unsqueeze(1)
        tokens = tokens.reshape(b, n_t * self.grid_size * self.grid_size, -1)
        token_valid = tubelet_valid.repeat_interleave(self.grid_size * self.grid_size, dim=1)
        tokens = tokens * token_valid.unsqueeze(-1)
        for block in self.blocks:
            tokens = block(tokens, token_valid, n_t, self.grid_size)
        return self.norm(tokens) * token_valid.unsqueeze(-1), token_valid
