"""Briques différentiables du modèle ; aucun poids préentraîné externe."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import BlueConfig
from .fsq import FiniteScalarQuantizer
from .hyper_connections import ManifoldHyperConnection
from .spectral import SpectralTemporalMixer
from .ssm import GatedDeltaNet
from .temporal_precision import temporal_work_dtype
from .ttt import TTTLinear


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, *, stable=False):
        super().__init__()
        if type(stable) is not bool:
            raise ValueError("stable doit être booléen")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.stable = stable

    def forward(self, x):
        if not self.stable:
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        else:
            # Le facteur détaché s'élimine algébriquement ; il borne les carrés
            # sans modifier le gradient, même pour des activations très grandes.
            fp32 = torch.finfo(torch.float32)
            dtype = torch.float64 if x.dtype == torch.float64 or not fp32.tiny <= self.eps <= fp32.max else torch.float32
            value = x.to(dtype=dtype)
            scale = value.detach().abs().amax(-1, keepdim=True).clamp_min(1.)
            scaled = value / scale
            normalized = scaled * torch.rsqrt(scaled.square().mean(-1, keepdim=True) + (self.eps / scale) / scale)
        return normalized.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.up_gate = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gate, up = self.up_gate(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta, *, scaling_type="none", scaling_factor=1.0,
                 original_max_seq_len=None, max_seq_len=2048):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.scaling_type = scaling_type
        self.scaling_factor = float(scaling_factor)
        self.original_max_seq_len = original_max_seq_len
        self.max_seq_len = max_seq_len
        self.mscale = (0.1 * math.log(self.scaling_factor) + 1.0
                       if scaling_type == "yarn" and self.scaling_factor > 1.0 else 1.0)
        self.register_buffer("inv_freq", self.frequencies(), persistent=False)

    def frequencies(self, device=None):
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim
        if self.scaling_type == "ntk" and self.scaling_factor > 1.0:
            power = self.head_dim / max(self.head_dim - 2, 1)
            base_theta = self.theta * (self.scaling_factor ** power)
            return 1.0 / (base_theta ** exponent)
        base = 1.0 / (self.theta ** exponent)
        if self.scaling_type == "linear" and self.scaling_factor > 1.0:
            return base / self.scaling_factor
        if self.scaling_type == "yarn" and self.scaling_factor > 1.0:
            original_len = (self.original_max_seq_len if self.original_max_seq_len is not None
                            else max(1, int(self.max_seq_len / self.scaling_factor)))
            beta_fast, beta_slow = 32.0, 1.0
            rotations = original_len * base / (2.0 * math.pi)
            ramp = ((rotations - beta_slow) / (beta_fast - beta_slow)).clamp(0.0, 1.0)
            return (base / self.scaling_factor) * (1.0 - ramp) + base * ramp
        return base

    def forward(self, x, offset=0):
        # [B, H, T, D] ; calcul des phases en fp32, même en précision mixte.
        with torch.autocast(device_type=x.device.type, enabled=False):
            positions = torch.arange(offset, offset + x.size(-2), device=x.device).float()
            phase = torch.outer(positions, self.inv_freq.float())
            cos = (phase.cos() * self.mscale).to(x.dtype)
            sin = (phase.sin() * self.mscale).to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class RotaryEmbedding2D(RotaryEmbedding):
    """Positions rotatives axiales 2D (hauteur et largeur) pour grille de patches."""

    def __init__(self, head_dim, theta, grid_size):
        if head_dim % 4:
            raise ValueError("RotaryEmbedding2D nécessite head_dim divisible par 4")
        self.grid_size = grid_size
        super().__init__(head_dim, theta)

    def frequencies(self, device=None):
        half = self.head_dim // 2
        exponent = torch.arange(0, half, 2, dtype=torch.float32, device=device) / half
        return 1.0 / (self.theta ** exponent)

    def forward(self, x, offset=0):
        # x: [B, H, G*G, D] ; première moitié en coordonnée verticale, seconde en horizontale.
        if offset != 0 or x.size(-2) != self.grid_size * self.grid_size:
            raise ValueError("RotaryEmbedding2D attend une grille complète de patches sans décalage")
        half = self.head_dim // 2
        with torch.autocast(device_type=x.device.type, enabled=False):
            coords = torch.arange(self.grid_size, device=x.device, dtype=torch.float32)
            rows = coords.repeat_interleave(self.grid_size)
            cols = coords.repeat(self.grid_size)
            freq = self.inv_freq.float()
            phase_y = torch.outer(rows, freq)
            phase_x = torch.outer(cols, freq)
            cos_y, sin_y = phase_y.cos().to(x.dtype), phase_y.sin().to(x.dtype)
            cos_x, sin_x = phase_x.cos().to(x.dtype), phase_x.sin().to(x.dtype)
        y_part, x_part = x[..., :half], x[..., half:]
        y_even, y_odd = y_part[..., 0::2], y_part[..., 1::2]
        x_even, x_odd = x_part[..., 0::2], x_part[..., 1::2]
        rot_y = torch.stack((y_even * cos_y - y_odd * sin_y, y_even * sin_y + y_odd * cos_y), dim=-1).flatten(-2)
        rot_x = torch.stack((x_even * cos_x - x_odd * sin_x, x_even * sin_x + x_odd * cos_x), dim=-1).flatten(-2)
        return torch.cat((rot_y, rot_x), dim=-1)


@dataclass
class LayerCache:
    key: torch.Tensor
    value: torch.Tensor
    valid: torch.Tensor
    conv_state: torch.Tensor
    start_pos: int = 0
    context_key: torch.Tensor | None = None
    context_value: torch.Tensor | None = None
    ssm_state: torch.Tensor | None = None
    ttt_state: torch.Tensor | None = None


def _sdpa(q, k, v, *, attn_mask=None, is_causal=False, dropout_p=0.0, enable_gqa=False, softcap=0.0):
    if softcap <= 0.0:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            dropout_p=dropout_p, enable_gqa=enable_gqa,
        )
    if enable_gqa and k.size(1) != q.size(1):
        repeats = q.size(1) // k.size(1)
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    with torch.autocast(device_type=q.device.type, enabled=False):
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        scores = softcap * torch.tanh(scores / softcap)
        if is_causal:
            t_q, t_k = scores.size(-2), scores.size(-1)
            causal_mask = torch.tril(torch.ones((t_q, t_k), device=q.device, dtype=torch.bool),
                                     diagonal=t_k - t_q)
            scores = scores.masked_fill(~causal_mask, -float("inf"))
        elif attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, -float("inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        if dropout_p > 0.0:
            probs = F.dropout(probs, p=dropout_p, training=True)
        out = torch.matmul(probs, v.float())
    return out.to(q.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: BlueConfig, *, causal=True, window=0, layer_index=0, grid_size=None):
        super().__init__()
        self.n_heads, self.n_kv_heads = config.n_heads, config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.causal, self.window = causal, window
        self.dropout = config.dropout
        self.differential = config.differential_attention
        self.kv_lora_rank = config.kv_lora_rank
        self.attention_sinks = config.attention_sinks if causal else 0
        self.value_residual = config.value_residual and causal
        self.attn_logit_softcap = float(config.attn_logit_softcap)
        self.layer_index = layer_index
        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        if self.kv_lora_rank > 0:
            self.kv_down = nn.Linear(config.d_model, self.kv_lora_rank, bias=False)
            self.kv_norm = RMSNorm(self.kv_lora_rank, config.norm_eps)
            if self.differential:
                self.nope_dim = self.head_dim // 4
                self.rope_dim = self.head_dim // 4
                self.k_up = nn.Linear(self.kv_lora_rank, self.n_heads * 2 * self.nope_dim, bias=False)
                self.k_rope = nn.Linear(config.d_model, 2 * self.rope_dim, bias=False)
                self.q_norm = RMSNorm(self.head_dim // 2, config.norm_eps)
            else:
                self.nope_dim = self.head_dim // 2
                self.rope_dim = self.head_dim // 2
                self.k_up = nn.Linear(self.kv_lora_rank, self.n_heads * self.nope_dim, bias=False)
                self.k_rope = nn.Linear(config.d_model, self.rope_dim, bias=False)
                self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
            self.v_up = nn.Linear(self.kv_lora_rank, self.n_heads * self.head_dim, bias=False)
            self.k_norm = RMSNorm(self.nope_dim, config.norm_eps)
            self.k_rope_norm = RMSNorm(self.rope_dim, config.norm_eps)
            rope_dim = self.rope_dim
            diff_dim = self.head_dim // 2
        else:
            self.kv = nn.Linear(config.d_model, 2 * self.n_kv_heads * self.head_dim, bias=False)
            rope_dim = self.head_dim // 2 if self.differential else self.head_dim
            diff_dim = rope_dim
            self.q_norm = RMSNorm(rope_dim, config.norm_eps)
            self.k_norm = RMSNorm(rope_dim, config.norm_eps)
        if grid_size is not None:
            self.rope = RotaryEmbedding2D(rope_dim, config.rope_theta, grid_size)
        else:
            self.rope = RotaryEmbedding(
                rope_dim, config.rope_theta,
                scaling_type=config.rope_scaling_type,
                scaling_factor=config.rope_scaling_factor,
                original_max_seq_len=config.rope_original_max_seq_len,
                max_seq_len=config.max_seq_len,
            )
        if self.differential:
            self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_index)
            self.lambda_q1 = nn.Parameter(torch.empty(diff_dim))
            self.lambda_k1 = nn.Parameter(torch.empty(diff_dim))
            self.lambda_q2 = nn.Parameter(torch.empty(diff_dim))
            self.lambda_k2 = nn.Parameter(torch.empty(diff_dim))
            self.sub_norm = RMSNorm(self.head_dim, config.norm_eps)
            nn.init.normal_(self.lambda_q1, std=0.1)
            nn.init.normal_(self.lambda_k1, std=0.1)
            nn.init.normal_(self.lambda_q2, std=0.1)
            nn.init.normal_(self.lambda_k2, std=0.1)
        if self.attention_sinks > 0:
            sink_heads = self.n_heads if self.kv_lora_rank > 0 else self.n_kv_heads
            self.sink_k = nn.Parameter(torch.empty(1, sink_heads, self.attention_sinks, self.head_dim))
            self.sink_v = nn.Parameter(torch.empty(1, sink_heads, self.attention_sinks, self.head_dim))
        else:
            self.sink_k = self.sink_v = None
        if self.value_residual and layer_index > 0:
            if self.kv_lora_rank > 0:
                self.v_res_gate = nn.Parameter(torch.full((1, 1, 1, self.kv_lora_rank), -2.0))
            else:
                self.v_res_gate = nn.Parameter(torch.full((1, self.n_kv_heads, 1, 1), -2.0))
        else:
            self.v_res_gate = None
        self.reset_sink_parameters()

    def reset_sink_parameters(self):
        if self.sink_k is not None:
            with torch.no_grad():
                nn.init.normal_(self.sink_k, std=0.02)
                nn.init.zeros_(self.sink_v)
        if self.v_res_gate is not None:
            with torch.no_grad():
                self.v_res_gate.fill_(-2.0)

    def _compute_differential_lambda(self, device, dtype):
        with torch.autocast(device_type=device.type, enabled=False):
            dot1 = (self.lambda_q1.float() * self.lambda_k1.float()).sum().clamp(-10.0, 10.0)
            dot2 = (self.lambda_q2.float() * self.lambda_k2.float()).sum().clamp(-10.0, 10.0)
            return (dot1.exp() - dot2.exp() + self.lambda_init).to(dtype)

    def forward(self, x, valid, *, offset=0, cache=None, use_cache=False, v0=None):
        b, t, _ = x.shape
        v_first = None
        if self.kv_lora_rank > 0:
            c_kv = self.kv_norm(self.kv_down(x)).unsqueeze(1)
            if self.value_residual:
                if self.layer_index == 0:
                    v_first = c_kv
                elif v0 is not None and self.v_res_gate is not None:
                    lam_v = torch.sigmoid(self.v_res_gate).to(c_kv.dtype)
                    c_kv = (1.0 - lam_v) * c_kv + lam_v * v0.to(c_kv.dtype)
            if self.differential:
                sub = self.head_dim // 2
                q1, q2 = self.q(x).view(b, t, self.n_heads, 2, sub).unbind(3)
                q1 = self.q_norm(q1.transpose(1, 2))
                q2 = self.q_norm(q2.transpose(1, 2))
                q1 = torch.cat((q1[..., :self.nope_dim], self.rope(q1[..., self.nope_dim:], offset)), dim=-1)
                q2 = torch.cat((q2[..., :self.nope_dim], self.rope(q2[..., self.nope_dim:], offset)), dim=-1)
                kr1, kr2 = self.k_rope(x).view(b, t, 2, self.rope_dim).unbind(2)
                kr1 = self.rope(self.k_rope_norm(kr1.unsqueeze(1)), offset)
                kr2 = self.rope(self.k_rope_norm(kr2.unsqueeze(1)), offset)
                k_rope = torch.cat((kr1, kr2), dim=-1)
            else:
                q = self.q_norm(self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2))
                q = torch.cat((q[..., :self.nope_dim], self.rope(q[..., self.nope_dim:], offset)), dim=-1)
                k_rope = self.rope(self.k_rope_norm(self.k_rope(x).unsqueeze(1)), offset)
            cache_k, cache_v = c_kv, k_rope
        elif self.differential:
            sub = self.head_dim // 2
            q1, q2 = self.q(x).view(b, t, self.n_heads, 2, sub).unbind(3)
            q1 = self.rope(self.q_norm(q1.transpose(1, 2)), offset)
            q2 = self.rope(self.q_norm(q2.transpose(1, 2)), offset)
            k, v = self.kv(x).view(b, t, 2, self.n_kv_heads, self.head_dim).unbind(2)
            k1, k2 = k.view(b, t, self.n_kv_heads, 2, sub).unbind(3)
            k1 = self.rope(self.k_norm(k1.transpose(1, 2)), offset)
            k2 = self.rope(self.k_norm(k2.transpose(1, 2)), offset)
            v_cur = v.transpose(1, 2)
            if self.value_residual:
                if self.layer_index == 0:
                    v_first = v_cur
                elif v0 is not None and self.v_res_gate is not None:
                    lam_v = torch.sigmoid(self.v_res_gate).to(v_cur.dtype)
                    v_cur = (1.0 - lam_v) * v_cur + lam_v * v0.to(v_cur.dtype)
            cache_k, cache_v = torch.cat((k1, k2), dim=-1), v_cur
        else:
            q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            k, v = self.kv(x).view(b, t, 2, self.n_kv_heads, self.head_dim).unbind(2)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            if self.value_residual:
                if self.layer_index == 0:
                    v_first = v
                elif v0 is not None and self.v_res_gate is not None:
                    lam_v = torch.sigmoid(self.v_res_gate).to(v.dtype)
                    v = (1.0 - lam_v) * v + lam_v * v0.to(v.dtype)
            q = self.rope(self.q_norm(q), offset)
            cache_k, cache_v = self.rope(self.k_norm(k), offset), v

        if cache is not None:
            cache_k = torch.cat((cache.key, cache_k), -2)
            cache_v = torch.cat((cache.value, cache_v), -2)
            key_valid = torch.cat((cache.valid, valid), -1)
        else:
            key_valid = valid

        seq_k = cache_k.size(-2)
        start_pos = 0 if cache is None else cache.start_pos
        # Les positions RoPE restent absolues après l'éviction des anciennes clés.
        inside_window = not self.window or offset + t - 1 - start_pos < self.window
        mask, is_causal = None, False
        if (self.attention_sinks == 0 and bool(key_valid.all()) and inside_window
                and (not self.causal or t == 1 or offset == 0)):
            is_causal = self.causal and t > 1
        else:
            key_pos = torch.arange(start_pos, start_pos + seq_k, device=x.device)
            query_pos = torch.arange(offset, offset + t, device=x.device)
            allowed = torch.ones(t, seq_k, dtype=torch.bool, device=x.device)
            if self.causal:
                allowed = key_pos[None, :] <= query_pos[:, None]
            if self.window:
                allowed = allowed & (key_pos[None, :] > query_pos[:, None] - self.window)
            mask = allowed[None, None, :, :] & key_valid[:, None, None, :]
            if self.attention_sinks > 0:
                sink_mask = torch.ones((b, 1, t, self.attention_sinks), dtype=torch.bool, device=x.device)
                mask = torch.cat((sink_mask, mask), dim=-1)

        dropout_p = self.dropout if self.training else 0.0
        sc = self.attn_logit_softcap
        if self.kv_lora_rank > 0:
            latent = cache_k.squeeze(1)
            v_attn = self.v_up(latent).view(b, seq_k, self.n_heads, self.head_dim).transpose(1, 2)
            if self.sink_v is not None:
                v_attn = torch.cat((self.sink_v.to(v_attn.dtype).expand(b, -1, -1, -1), v_attn), dim=-2)
            if self.differential:
                kn1, kn2 = self.k_up(latent).view(b, seq_k, self.n_heads, 2, self.nope_dim).unbind(3)
                kn1 = self.k_norm(kn1.transpose(1, 2))
                kn2 = self.k_norm(kn2.transpose(1, 2))
                kr1_all, kr2_all = cache_v.expand(b, self.n_heads, seq_k, 2 * self.rope_dim).chunk(2, dim=-1)
                k1_all = torch.cat((kn1, kr1_all), dim=-1)
                k2_all = torch.cat((kn2, kr2_all), dim=-1)
                if self.sink_k is not None:
                    sk1, sk2 = self.sink_k.to(k1_all.dtype).expand(b, -1, -1, -1).chunk(2, dim=-1)
                    k1_all = torch.cat((sk1, k1_all), dim=-2)
                    k2_all = torch.cat((sk2, k2_all), dim=-2)
                a1 = _sdpa(q1, k1_all, v_attn, attn_mask=mask, is_causal=is_causal, dropout_p=dropout_p, softcap=sc)
                a2 = _sdpa(q2, k2_all, v_attn, attn_mask=mask, is_causal=is_causal, dropout_p=dropout_p, softcap=sc)
                lam = self._compute_differential_lambda(x.device, a1.dtype)
                y = self.sub_norm(a1 - lam * a2) * (1.0 - self.lambda_init)
            else:
                k_nope = self.k_norm(self.k_up(latent).view(b, seq_k, self.n_heads, self.nope_dim).transpose(1, 2))
                k_attn = torch.cat((k_nope, cache_v.expand(b, self.n_heads, seq_k, self.rope_dim)), dim=-1)
                if self.sink_k is not None:
                    k_attn = torch.cat((self.sink_k.to(k_attn.dtype).expand(b, -1, -1, -1), k_attn), dim=-2)
                y = _sdpa(q, k_attn, v_attn, attn_mask=mask, is_causal=is_causal, dropout_p=dropout_p, softcap=sc)
        else:
            gqa = self.n_heads != self.n_kv_heads
            v_attn = cache_v
            if self.sink_v is not None:
                v_attn = torch.cat((self.sink_v.to(v_attn.dtype).expand(b, -1, -1, -1), v_attn), dim=-2)
            if self.differential:
                k1_all, k2_all = cache_k.chunk(2, dim=-1)
                if self.sink_k is not None:
                    sk1, sk2 = self.sink_k.to(k1_all.dtype).expand(b, -1, -1, -1).chunk(2, dim=-1)
                    k1_all = torch.cat((self.k_norm(sk1), k1_all), dim=-2)
                    k2_all = torch.cat((self.k_norm(sk2), k2_all), dim=-2)
                a1 = _sdpa(
                    q1, k1_all, v_attn, attn_mask=mask, is_causal=is_causal,
                    dropout_p=dropout_p, enable_gqa=gqa, softcap=sc,
                )
                a2 = _sdpa(
                    q2, k2_all, v_attn, attn_mask=mask, is_causal=is_causal,
                    dropout_p=dropout_p, enable_gqa=gqa, softcap=sc,
                )
                lam = self._compute_differential_lambda(x.device, a1.dtype)
                y = self.sub_norm(a1 - lam * a2) * (1.0 - self.lambda_init)
            else:
                k_attn = cache_k
                if self.sink_k is not None:
                    k_attn = torch.cat((self.k_norm(self.sink_k.to(k_attn.dtype).expand(b, -1, -1, -1)), k_attn), dim=-2)
                y = _sdpa(
                    q, k_attn, v_attn, attn_mask=mask, is_causal=is_causal,
                    dropout_p=dropout_p, enable_gqa=gqa, softcap=sc,
                )
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        if use_cache and self.window and seq_k > self.window - 1:
            begin = seq_k - (self.window - 1)
            # clone libère réellement le stockage du préremplissage, même avec B=H=1.
            cache_k, cache_v = cache_k[:, :, begin:].clone(), cache_v[:, :, begin:].clone()
            key_valid = key_valid[:, begin:].clone()
        return self.out(y) * valid.unsqueeze(-1), (cache_k, cache_v, key_valid, v_first)


class CrossAttention(nn.Module):
    def __init__(self, config, *, layer_index=0):
        super().__init__()
        self.heads, self.head_dim = config.n_heads, config.d_model // config.n_heads
        self.differential = config.cross_differential_attention
        self.kv_lora_rank = config.cross_kv_lora_rank
        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        if self.kv_lora_rank > 0:
            self.kv_down = nn.Linear(config.d_model, self.kv_lora_rank, bias=False)
            self.kv_norm = RMSNorm(self.kv_lora_rank, config.norm_eps)
            self.k_up = nn.Linear(self.kv_lora_rank, config.d_model, bias=False)
            self.v_up = nn.Linear(self.kv_lora_rank, config.d_model, bias=False)
        else:
            self.kv = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        norm_dim = self.head_dim // 2 if self.differential else self.head_dim
        self.q_norm = RMSNorm(norm_dim, config.norm_eps)
        self.k_norm = RMSNorm(norm_dim, config.norm_eps)
        self.dropout = config.dropout
        if self.differential:
            self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_index)
            self.lambda_q1 = nn.Parameter(torch.empty(norm_dim))
            self.lambda_k1 = nn.Parameter(torch.empty(norm_dim))
            self.lambda_q2 = nn.Parameter(torch.empty(norm_dim))
            self.lambda_k2 = nn.Parameter(torch.empty(norm_dim))
            self.sub_norm = RMSNorm(self.head_dim, config.norm_eps)
            nn.init.normal_(self.lambda_q1, std=0.1)
            nn.init.normal_(self.lambda_k1, std=0.1)
            nn.init.normal_(self.lambda_q2, std=0.1)
            nn.init.normal_(self.lambda_k2, std=0.1)

    def _format_k(self, k: torch.Tensor) -> torch.Tensor:
        if self.differential:
            b = k.size(0)
            sub = self.head_dim // 2
            k1, k2 = k.view(b, -1, self.heads, 2, sub).unbind(3)
            k1 = self.k_norm(k1.transpose(1, 2))
            k2 = self.k_norm(k2.transpose(1, 2))
            return torch.cat((k1, k2), dim=-1)
        return self.k_norm(k.transpose(1, 2))

    def project_context(self, context):
        if self.kv_lora_rank > 0:
            return self.kv_norm(self.kv_down(context)), None
        b = context.size(0)
        k, v = self.kv(context).view(b, -1, 2, self.heads, self.head_dim).unbind(2)
        return self._format_k(k), v.transpose(1, 2)

    def forward(self, queries, context=None, valid=None, *, projected=None):
        b, t, d = queries.shape
        proj = self.project_context(context) if projected is None else projected
        if self.kv_lora_rank > 0:
            c_kv = proj[0]
            k = self._format_k(self.k_up(c_kv).view(b, -1, self.heads, self.head_dim))
            v = self.v_up(c_kv).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        else:
            k, v = proj
        mask = None if valid is None else valid[:, None, None, :]
        dropout_p = self.dropout if self.training else 0.0
        if self.differential:
            sub = self.head_dim // 2
            q1, q2 = self.q(queries).view(b, t, self.heads, 2, sub).unbind(3)
            q1 = self.q_norm(q1.transpose(1, 2))
            q2 = self.q_norm(q2.transpose(1, 2))
            k1, k2 = k.chunk(2, dim=-1)
            a1 = F.scaled_dot_product_attention(q1, k1, v, attn_mask=mask, dropout_p=dropout_p)
            a2 = F.scaled_dot_product_attention(q2, k2, v, attn_mask=mask, dropout_p=dropout_p)
            with torch.autocast(device_type=queries.device.type, enabled=False):
                dot1 = (self.lambda_q1.float() * self.lambda_k1.float()).sum().clamp(-10.0, 10.0)
                dot2 = (self.lambda_q2.float() * self.lambda_k2.float()).sum().clamp(-10.0, 10.0)
                lam = (dot1.exp() - dot2.exp() + self.lambda_init).to(a1.dtype)
            y = self.sub_norm(a1 - lam * a2) * (1.0 - self.lambda_init)
        else:
            q = self.q(queries).view(b, t, self.heads, self.head_dim).transpose(1, 2)
            y = F.scaled_dot_product_attention(
                self.q_norm(q), k, v, attn_mask=mask, dropout_p=dropout_p,
            )
        return self.out(y.transpose(1, 2).contiguous().view(b, t, d))


class GatedCausalConv(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.kernel = kernel
        self.proj = nn.Linear(dim, 2 * dim, bias=False)
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, valid, state=None):
        u, gate = self.proj(x).chunk(2, dim=-1)
        u = (u * valid.unsqueeze(-1)).transpose(1, 2)
        if state is None:
            state = u.new_zeros(u.size(0), u.size(1), self.kernel - 1)
        joined = torch.cat((state, u), -1)
        y = self.conv(joined).transpose(1, 2)
        next_state = joined[..., -(self.kernel - 1):] if self.kernel > 1 else joined[..., :0]
        return self.out(F.silu(y) * torch.sigmoid(gate)), next_state


class SparseMoE(nn.Module):
    """Top-k sans abandon de tokens, experts partagés, experts nuls zéro-calcul, équilibrage et z-loss."""

    def __init__(self, config):
        super().__init__()
        self.n_experts, self.top_k = config.n_experts, config.top_k
        self.n_null = config.moe_null_experts
        self.total_slots = self.n_experts + self.n_null
        self.top1_gate = config.top1_gate
        self.router_bias_update_rate = config.router_bias_update_rate
        self.router = nn.Linear(config.d_model, self.total_slots, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(config.d_model, config.hidden_dim) for _ in range(config.n_experts)
        ])
        self.shared = SwiGLU(config.d_model, config.hidden_dim * config.shared_experts)
        self.shared_gate = nn.Linear(config.d_model, 1, bias=False) if config.shared_expert_gate else None
        if config.router_bias_update_rate > 0:
            self.register_buffer("expert_bias", torch.zeros(config.n_experts, dtype=torch.float32))
        else:
            self.expert_bias = None
        self.last_null_fraction: float = 0.0

    @torch.no_grad()
    def update_expert_bias(self, usage: torch.Tensor):
        """Mise à jour sans gradient du biais d'équilibrage (DeepSeek-V3)."""
        if self.expert_bias is None or self.router_bias_update_rate <= 0:
            return
        target = 1.0 / self.n_experts
        error = target - usage.to(device=self.expert_bias.device, dtype=self.expert_bias.dtype)
        self.expert_bias.add_(self.router_bias_update_rate * torch.sign(error))
        self.expert_bias.sub_(self.expert_bias.mean())

    def forward(self, x, valid):
        flat = x.reshape(-1, x.size(-1))
        indices = valid.reshape(-1).nonzero(as_tuple=True)[0]
        tokens = flat[indices]
        if tokens.size(0) == 0:
            zero = x.sum() * 0
            self.last_null_fraction = 0.0
            return torch.zeros_like(x), zero, zero, x.new_zeros(self.n_experts)
        # Router en fp32 pour éviter la saturation en bf16/fp16.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            scores = F.linear(tokens.float(), self.router.weight.float())
            probs = scores.softmax(-1)
            if self.expert_bias is not None:
                bias = (F.pad(self.expert_bias.float(), (0, self.n_null))
                        if self.n_null > 0 else self.expert_bias.float())
                _, selected = (probs + bias).topk(self.top_k, dim=-1)
                weights = probs.gather(-1, selected)
            else:
                weights, selected = probs.topk(self.top_k, dim=-1)
        if self.top_k != 1 or self.top1_gate == "normalized":
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        mixed = self.shared(tokens)
        if self.shared_gate is not None:
            mixed = mixed * torch.sigmoid(self.shared_gate(tokens)).to(mixed.dtype)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(selected == expert_id)
            # Un expert vide reste un graphe valide, sans traitement de tokens inutiles.
            if rows.numel():
                expert_output = expert(tokens[rows])
                contribution = expert_output * weights[rows, slots, None].to(expert_output.dtype)
                mixed = mixed.index_add(0, rows, contribution)
        result = torch.zeros_like(flat).index_copy(0, indices, mixed.to(flat.dtype)).view_as(x)
        full_fraction = F.one_hot(selected, self.total_slots).float().mean((0, 1))
        self.last_null_fraction = float(full_fraction[self.n_experts:].sum().item()) if self.n_null > 0 else 0.0
        balance = self.total_slots * (probs.mean(0) * full_fraction).sum()
        z_loss = scores.logsumexp(-1).square().mean()
        return result, balance, z_loss, full_fraction[:self.n_experts].detach()


class EncoderBlock(nn.Module):
    def __init__(self, config, layer_index=0, *, grid_size=None):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(config.d_model), RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config, causal=False, layer_index=layer_index, grid_size=grid_size)
        self.ffn = SwiGLU(config.d_model, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, valid):
        x = x + self.dropout(self.attn(self.norm1(x), valid)[0])
        return (x + self.dropout(self.ffn(self.norm2(x)))) * valid.unsqueeze(-1)


class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.stem = nn.Conv2d(config.image_channels, d // 2, 3, padding=1)
        self.local3 = nn.Conv2d(d // 2, d // 2, 3, padding=1, groups=d // 2)
        self.local5 = nn.Conv2d(d // 2, d // 2, 5, padding=2, groups=d // 2)
        self.patch = nn.Conv2d(d // 2, d, config.patch_size, stride=config.patch_size)
        grid = config.image_size // config.patch_size
        self.position = nn.Parameter(torch.empty(1, grid * grid, d))
        rope_grid = grid if config.vision_rope_2d else None
        self.blocks = nn.ModuleList([EncoderBlock(config, i, grid_size=rope_grid) for i in range(config.vision_layers)])
        self.norm = RMSNorm(d, config.norm_eps)

    def forward(self, images):
        x = F.silu(self.stem(images.to(self.stem.weight.dtype)))
        x = x + F.silu(self.local3(x)) + F.silu(self.local5(x))
        x = self.patch(x).flatten(2).transpose(1, 2) + self.position
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        for block in self.blocks:
            x = block(x, valid)
        return self.norm(x), valid


class TemporalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(2 * config.series_features, config.d_model)
        self.conv = GatedCausalConv(config.d_model, config.conv_kernel)
        self.spectral = (SpectralTemporalMixer(config.d_model, norm_cls=RMSNorm, norm_eps=config.norm_eps)
                         if config.temporal_spectral_mixing else None)
        self.blocks = nn.ModuleList([EncoderBlock(config, i) for i in range(config.temporal_layers)])
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.forecast_context = config.forecast_context
        self.statistics_relative = config.temporal_statistics_relative
        self.forecast_skip = nn.Linear(config.forecast_context, config.forecast_horizon, bias=False)
        self.statistics_proj = (nn.Linear(6 * config.series_features + 1, config.d_model, bias=False)
                                if config.temporal_statistics else None)

    def forward(self, series, valid):
        series = series.to(dtype=temporal_work_dtype(series, valid))
        observed = torch.isfinite(series) & valid.unsqueeze(-1)
        clean = torch.where(observed, series, 0.0)
        observed_count = observed.sum(1, keepdim=True)
        count = observed_count.clamp_min(1)
        mean = clean.sum(1, keepdim=True) / count
        centered = torch.where(observed, clean - mean, 0.0)
        scale = (centered.square().sum(1, keepdim=True) / count + 1e-5).sqrt()
        scale = torch.where(observed_count > 0, scale, torch.ones_like(scale))
        positions = torch.arange(series.size(1), device=series.device)[None, :, None]
        last_index = positions.expand_as(series).masked_fill(~observed, -1).max(1).values
        anchor = clean.gather(1, last_index.clamp_min(0).unsqueeze(1)).squeeze(1)
        anchor = torch.where(last_index >= 0, anchor, mean.squeeze(1))
        # Les valeurs manquantes sont représentées explicitement et ne polluent pas RevIN.
        features = torch.cat((centered / scale, observed.to(series.dtype)), dim=-1)
        x = self.proj(features.to(self.proj.weight.dtype))
        if self.statistics_proj is not None:
            length = valid.sum(1, keepdim=True).float().clamp_min(1)
            # Les âges comptent les positions valides après la dernière observation,
            # afin de rester invariants au padding ou aux positions exclues.
            age = (valid.unsqueeze(-1) & (positions > last_index[:, None])).sum(1) / length
            if self.statistics_relative:
                first_index = positions.expand_as(series).masked_fill(~observed, series.size(1)).min(1).values
                first = clean.gather(1, first_index.clamp_max(series.size(1) - 1).unsqueeze(1)).squeeze(1)
                first = torch.where(observed_count.squeeze(1) > 0, first, mean.squeeze(1))
                low = clean.masked_fill(~observed, float("inf")).min(1).values
                high = clean.masked_fill(~observed, float("-inf")).max(1).values
                low = torch.where(observed_count.squeeze(1) > 0, low, mean.squeeze(1))
                high = torch.where(observed_count.squeeze(1) > 0, high, mean.squeeze(1))
                normalized = centered / scale
                previous = positions[..., 0].expand_as(valid).masked_fill(~valid, -1).cummax(1).values
                previous = F.pad(previous[:, :-1], (1, 0), value=-1)
                indices = previous.clamp_min(0).unsqueeze(-1).expand_as(series)
                adjacent = observed & observed.gather(1, indices) & (previous >= 0).unsqueeze(-1)
                energy = ((normalized - normalized.gather(1, indices)).square() * adjacent).sum(1)
                energy = energy / adjacent.sum(1).clamp_min(1)
                summary = ((first - mean.squeeze(1)) / scale.squeeze(1),
                           (high - low) / scale.squeeze(1), energy)
            else:
                summary = (mean.squeeze(1).asinh(), scale.squeeze(1).log(), anchor.asinh())
            statistics = torch.cat((*summary, (anchor - mean.squeeze(1)) / scale.squeeze(1),
                                    observed_count.squeeze(1) / length, age,
                                    (length / self.forecast_context).log()), dim=-1)
            x = x + self.statistics_proj(statistics.to(self.statistics_proj.weight.dtype))[:, None]
        x = x * valid.unsqueeze(-1)
        x = x + self.conv(x, valid)[0]
        if self.spectral is not None:
            x = x + self.spectral(x, valid)
        for block in self.blocks:
            x = block(x, valid)
        # Connexion temporelle directe : partir de la persistance puis apprendre
        # une correction multihorizon sur l'historique centré à la dernière observation.
        centered_last = torch.where(observed, (clean - anchor[:, None]) / scale, 0.0)
        histories = []
        for row in range(series.size(0)):
            history = centered_last[row, valid[row]].transpose(0, 1).unsqueeze(0)
            histories.append(F.adaptive_avg_pool1d(history, self.forecast_context).squeeze(0))
        history = torch.stack(histories)
        correction = self.forecast_skip(history.to(self.forecast_skip.weight.dtype)).transpose(1, 2)
        baseline = anchor[:, None] + correction * scale
        return self.norm(x), valid, mean.squeeze(1), scale.squeeze(1), baseline


class LatentResampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hierarchical = config.hierarchical_resampler
        self.latents = nn.Parameter(torch.empty(1, config.n_latents, config.d_model))
        self.null_context = nn.Parameter(torch.empty(1, 1, config.d_model))
        if self.hierarchical:
            local_count = max(4, config.n_latents // 2)
            self.local_latents = nn.Parameter(torch.empty(1, local_count, config.d_model))
            self.local_norm = RMSNorm(config.d_model, config.norm_eps)
            self.local_cross = CrossAttention(config)
            self.local_block = EncoderBlock(config, 0)
        else:
            self.local_latents = self.local_norm = self.local_cross = self.local_block = None
        self.norms = nn.ModuleList([RMSNorm(config.d_model) for _ in range(config.resampler_layers)])
        self.cross = nn.ModuleList([CrossAttention(config, layer_index=i) for i in range(config.resampler_layers)])
        self.blocks = nn.ModuleList([EncoderBlock(config, i) for i in range(config.resampler_layers)])
        self.fsq = (FiniteScalarQuantizer(config.d_model, config.latent_fsq_levels,
                                          norm_cls=RMSNorm, norm_eps=config.norm_eps)
                    if config.latent_fsq_levels else None)
        self.last_codes: torch.Tensor | None = None

    def forward(self, contexts, masks, batch_size):
        null = self.null_context.expand(batch_size, -1, -1)
        if self.hierarchical and contexts:
            stage1_contexts, stage1_masks = [], []
            local_queries = self.local_latents.expand(batch_size, -1, -1)
            local_valid = torch.ones(local_queries.shape[:2], dtype=torch.bool, device=null.device)
            for ctx, mask in zip(contexts, masks):
                # Niveau 1 : chaque modalité est distillée vers un budget équilibré de latents locaux.
                q = local_queries + self.local_cross(self.local_norm(local_queries), ctx, mask)
                q = self.local_block(q, local_valid)
                stage1_contexts.append(q)
                stage1_masks.append(local_valid)
            contexts, masks = stage1_contexts, stage1_masks
        context = torch.cat([null, *contexts], 1)
        valid = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool, device=null.device), *masks
        ], 1)
        x = self.latents.expand(batch_size, -1, -1)
        latent_valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        for norm, cross, block in zip(self.norms, self.cross, self.blocks):
            x = x + cross(norm(x), context, valid)
            x = block(x, latent_valid)
        if self.fsq is not None:
            x, self.last_codes = self.fsq(x)
        else:
            self.last_codes = None
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        self.norm3 = RMSNorm(config.d_model, config.norm_eps)
        window = 0 if (index + 1) % config.global_every == 0 else config.sliding_window
        self.attn = GroupedQueryAttention(config, window=window, layer_index=index)
        self.conv = GatedCausalConv(config.d_model, config.conv_kernel)
        self.mix_gate = nn.Parameter(torch.zeros(config.d_model))
        if config.ssm_state_dim > 0:
            ssm_heads = config.n_heads if config.ssm_heads is None else config.ssm_heads
            self.ssm = GatedDeltaNet(config.d_model, config.ssm_state_dim, ssm_heads,
                                     norm_cls=RMSNorm, norm_eps=config.norm_eps,
                                     chunk_size=config.ssm_chunk_size)
            self.ssm_gate = nn.Parameter(torch.full((config.d_model,), -1.0))
        else:
            self.ssm = None
            self.ssm_gate = None
        if config.ttt_state_dim > 0:
            ttt_heads = config.n_heads if config.ttt_heads is None else config.ttt_heads
            self.ttt = TTTLinear(config.d_model, config.ttt_state_dim, ttt_heads,
                                 norm_cls=RMSNorm, norm_eps=config.norm_eps,
                                 base_lr=config.ttt_base_lr)
            self.ttt_gate = nn.Parameter(torch.full((config.d_model,), -1.0))
        else:
            self.ttt = None
            self.ttt_gate = None
        self.cross = CrossAttention(config, layer_index=index)
        self.cross_gate = nn.Parameter(torch.full((config.d_model,), -1.0))
        self.moe = SparseMoE(config)
        self.depth_capacity = config.depth_routing_capacity
        self.depth_router = nn.Linear(config.d_model, 1, bias=True) if config.depth_routing_capacity < 1.0 else None
        self.reset_depth_router()
        self.dropout = nn.Dropout(config.dropout)
        self.residual_streams = config.residual_streams
        if config.residual_streams > 1:
            self.hc_attn = ManifoldHyperConnection(config.d_model, config.residual_streams,
                                                   norm_cls=RMSNorm, norm_eps=config.norm_eps)
            self.hc_cross = ManifoldHyperConnection(config.d_model, config.residual_streams,
                                                    norm_cls=RMSNorm, norm_eps=config.norm_eps)
            self.hc_moe = ManifoldHyperConnection(config.d_model, config.residual_streams,
                                                  norm_cls=RMSNorm, norm_eps=config.norm_eps)
        else:
            self.hc_attn = self.hc_cross = self.hc_moe = None

    def reset_depth_router(self):
        if self.depth_router is not None:
            with torch.no_grad():
                p = min(max(self.depth_capacity, 1e-4), 1.0 - 1e-4)
                self.depth_router.bias.fill_(math.log(p / (1.0 - p)))

    def forward(self, x, valid, context, *, offset=0, cache=None, use_cache=False, v0=None):
        if self.hc_attn is not None:
            h_pre, h_post, h_res = self.hc_attn.routing_weights(x)
            h = self.norm1(self.hc_attn.combine_input(x, h_pre))
        else:
            h = self.norm1(x)
        attention, (k, v, key_valid, v_first) = self.attn(
            h, valid, offset=offset, cache=cache, use_cache=use_cache, v0=v0,
        )
        local, conv_state = self.conv(h, valid, None if cache is None else cache.conv_state)
        mixed = attention + torch.sigmoid(self.mix_gate) * local
        ssm_state = None
        if self.ssm is not None:
            recurrent, ssm_state = self.ssm(h, valid, None if cache is None else cache.ssm_state)
            mixed = mixed + torch.sigmoid(self.ssm_gate) * recurrent
        ttt_state = None
        if self.ttt is not None:
            fast_weight_out, ttt_state = self.ttt(h, valid, None if cache is None else cache.ttt_state)
            mixed = mixed + torch.sigmoid(self.ttt_gate) * fast_weight_out
        if self.hc_attn is not None:
            x = self.hc_attn.update_streams(x, self.dropout(mixed), h_post, h_res)
        else:
            x = x + self.dropout(mixed)
        if cache is not None and cache.context_key is not None:
            context_kv = cache.context_key, cache.context_value
        else:
            context_kv = self.cross.project_context(context)
        if self.hc_cross is not None:
            c_pre, c_post, c_res = self.hc_cross.routing_weights(x)
            cross_in = self.norm2(self.hc_cross.combine_input(x, c_pre))
            cross_out = self.dropout(torch.sigmoid(self.cross_gate) * self.cross(cross_in, projected=context_kv))
            x = self.hc_cross.update_streams(x, cross_out, c_post, c_res)
        else:
            x = x + self.dropout(torch.sigmoid(self.cross_gate) * self.cross(self.norm2(x), projected=context_kv))
        if self.hc_moe is not None:
            m_pre, m_post, m_res = self.hc_moe.routing_weights(x)
            moe_in = self.norm3(self.hc_moe.combine_input(x, m_pre))
        else:
            moe_in = self.norm3(x)
        depth_loss = None
        if self.depth_router is not None:
            with torch.autocast(device_type=moe_in.device.type, enabled=False):
                logits = F.linear(moe_in.float(), self.depth_router.weight.float(),
                                  self.depth_router.bias.float()).squeeze(-1)
                depth_prob = torch.sigmoid(logits)
                mean_prob = (depth_prob * valid.float()).sum() / valid.sum().clamp_min(1)
                depth_loss = (mean_prob - self.depth_capacity).square()
            selected = valid & (depth_prob >= 0.5)
            update, balance, z_loss, usage = self.moe(moe_in, selected)
            update = update * depth_prob.to(update.dtype).unsqueeze(-1)
        else:
            update, balance, z_loss, usage = self.moe(moe_in, valid)
        if self.hc_moe is not None:
            x = self.hc_moe.update_streams(x, self.dropout(update), m_post, m_res) * valid[:, :, None, None]
        else:
            x = (x + self.dropout(update)) * valid.unsqueeze(-1)
        next_cache = (LayerCache(k, v, key_valid, conv_state, offset + x.size(1) - k.size(-2),
                                 *context_kv, ssm_state=ssm_state, ttt_state=ttt_state)
                      if use_cache else None)
        if self.attn.value_residual and self.attn.layer_index == 0:
            return x, balance, z_loss, usage, next_cache, depth_loss, v_first
        return x, balance, z_loss, usage, next_cache, depth_loss


class QuantileForecastHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.horizon, self.features = config.forecast_horizon, config.series_features
        self.quantiles = config.quantiles
        self.proj = nn.Sequential(
            RMSNorm(config.d_model), nn.Linear(config.d_model, config.hidden_dim),
            nn.SiLU(), nn.Linear(config.hidden_dim, self.horizon * self.features * len(self.quantiles)),
        )

    def forward(self, pooled, mean=None, scale=None, baseline=None):
        raw = self.proj(pooled).view(-1, self.horizon, self.features, len(self.quantiles))
        # Centre médian libre et incréments positifs : aucun croisement de quantiles.
        center_index = self.quantiles.index(0.5)
        increments = F.softplus(raw[..., 1:])
        offsets = torch.cat((torch.zeros_like(raw[..., :1]), increments.cumsum(-1)), -1)
        result = raw[..., :1] + offsets - offsets[..., center_index:center_index + 1]
        if mean is not None:
            result = result * scale[:, None, :, None]
            result = result + (baseline.unsqueeze(-1) if baseline is not None else mean[:, None, :, None])
        return result
