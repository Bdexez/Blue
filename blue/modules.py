"""Briques différentiables du modèle ; aucun poids préentraîné externe."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import BlueConfig
from .temporal_precision import temporal_work_dtype


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


@dataclass
class LayerCache:
    key: torch.Tensor
    value: torch.Tensor
    valid: torch.Tensor
    conv_state: torch.Tensor
    start_pos: int = 0
    context_key: torch.Tensor | None = None
    context_value: torch.Tensor | None = None


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: BlueConfig, *, causal=True, window=0, layer_index=0):
        super().__init__()
        self.n_heads, self.n_kv_heads = config.n_heads, config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.causal, self.window = causal, window
        self.dropout = config.dropout
        self.differential = config.differential_attention
        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.kv = nn.Linear(config.d_model, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        rope_dim = self.head_dim // 2 if self.differential else self.head_dim
        self.q_norm = RMSNorm(rope_dim, config.norm_eps)
        self.k_norm = RMSNorm(rope_dim, config.norm_eps)
        self.rope = RotaryEmbedding(
            rope_dim, config.rope_theta,
            scaling_type=config.rope_scaling_type,
            scaling_factor=config.rope_scaling_factor,
            original_max_seq_len=config.rope_original_max_seq_len,
            max_seq_len=config.max_seq_len,
        )
        if self.differential:
            self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_index)
            self.lambda_q1 = nn.Parameter(torch.empty(rope_dim))
            self.lambda_k1 = nn.Parameter(torch.empty(rope_dim))
            self.lambda_q2 = nn.Parameter(torch.empty(rope_dim))
            self.lambda_k2 = nn.Parameter(torch.empty(rope_dim))
            self.sub_norm = RMSNorm(self.head_dim, config.norm_eps)
            nn.init.normal_(self.lambda_q1, std=0.1)
            nn.init.normal_(self.lambda_k1, std=0.1)
            nn.init.normal_(self.lambda_q2, std=0.1)
            nn.init.normal_(self.lambda_k2, std=0.1)

    def forward(self, x, valid, *, offset=0, cache=None, use_cache=False):
        b, t, _ = x.shape
        if self.differential:
            sub = self.head_dim // 2
            q1, q2 = self.q(x).view(b, t, self.n_heads, 2, sub).unbind(3)
            q1 = self.rope(self.q_norm(q1.transpose(1, 2)), offset)
            q2 = self.rope(self.q_norm(q2.transpose(1, 2)), offset)
            k, v = self.kv(x).view(b, t, 2, self.n_kv_heads, self.head_dim).unbind(2)
            k1, k2 = k.view(b, t, self.n_kv_heads, 2, sub).unbind(3)
            k1 = self.rope(self.k_norm(k1.transpose(1, 2)), offset)
            k2 = self.rope(self.k_norm(k2.transpose(1, 2)), offset)
            k = torch.cat((k1, k2), dim=-1)
            v = v.transpose(1, 2)
        else:
            q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            k, v = self.kv(x).view(b, t, 2, self.n_kv_heads, self.head_dim).unbind(2)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            q = self.rope(self.q_norm(q), offset)
            k = self.rope(self.k_norm(k), offset)
        if cache is not None:
            k, v = torch.cat((cache.key, k), -2), torch.cat((cache.value, v), -2)
            key_valid = torch.cat((cache.valid, valid), -1)
        else:
            key_valid = valid
        start_pos = 0 if cache is None else cache.start_pos
        # Les positions RoPE restent absolues après l'éviction des anciennes clés.
        inside_window = not self.window or offset + t - 1 - start_pos < self.window
        mask, is_causal = None, False
        if bool(key_valid.all()) and inside_window and (not self.causal or t == 1 or offset == 0):
            is_causal = self.causal and t > 1
        else:
            key_pos = torch.arange(start_pos, start_pos + k.size(-2), device=x.device)
            query_pos = torch.arange(offset, offset + t, device=x.device)
            allowed = torch.ones(t, k.size(-2), dtype=torch.bool, device=x.device)
            if self.causal:
                allowed = key_pos[None, :] <= query_pos[:, None]
            if self.window:
                allowed = allowed & (key_pos[None, :] > query_pos[:, None] - self.window)
            mask = allowed[None, None, :, :] & key_valid[:, None, None, :]
        dropout_p = self.dropout if self.training else 0.0
        gqa = self.n_heads != self.n_kv_heads
        if self.differential:
            k1_all, k2_all = k.chunk(2, dim=-1)
            a1 = F.scaled_dot_product_attention(
                q1, k1_all, v, attn_mask=mask, is_causal=is_causal, dropout_p=dropout_p, enable_gqa=gqa,
            )
            a2 = F.scaled_dot_product_attention(
                q2, k2_all, v, attn_mask=mask, is_causal=is_causal, dropout_p=dropout_p, enable_gqa=gqa,
            )
            with torch.autocast(device_type=x.device.type, enabled=False):
                dot1 = (self.lambda_q1.float() * self.lambda_k1.float()).sum().clamp(-10.0, 10.0)
                dot2 = (self.lambda_q2.float() * self.lambda_k2.float()).sum().clamp(-10.0, 10.0)
                lam = (dot1.exp() - dot2.exp() + self.lambda_init).to(a1.dtype)
            y = self.sub_norm(a1 - lam * a2) * (1.0 - self.lambda_init)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=is_causal,
                dropout_p=dropout_p, enable_gqa=gqa,
            )
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        if use_cache and self.window and k.size(-2) > self.window - 1:
            begin = k.size(-2) - (self.window - 1)
            # clone libère réellement le stockage du préremplissage, même avec B=H=1.
            k, v = k[:, :, begin:].clone(), v[:, :, begin:].clone()
            key_valid = key_valid[:, begin:].clone()
        return self.out(y) * valid.unsqueeze(-1), (k, v, key_valid)


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.head_dim = config.n_heads, config.d_model // config.n_heads
        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.kv = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.dropout = config.dropout

    def project_context(self, context):
        b = context.size(0)
        k, v = self.kv(context).view(b, -1, 2, self.heads, self.head_dim).unbind(2)
        return self.k_norm(k.transpose(1, 2)), v.transpose(1, 2)

    def forward(self, queries, context=None, valid=None, *, projected=None):
        b, t, d = queries.shape
        q = self.q(queries).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.project_context(context) if projected is None else projected
        mask = None if valid is None else valid[:, None, None, :]
        y = F.scaled_dot_product_attention(
            self.q_norm(q), k, v,
            attn_mask=mask, dropout_p=self.dropout if self.training else 0.0,
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
    """Top-k sans abandon de tokens, experts partagés, équilibrage et router z-loss."""

    def __init__(self, config):
        super().__init__()
        self.n_experts, self.top_k = config.n_experts, config.top_k
        self.top1_gate = config.top1_gate
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(config.d_model, config.hidden_dim) for _ in range(config.n_experts)
        ])
        self.shared = SwiGLU(config.d_model, config.hidden_dim * config.shared_experts)

    def forward(self, x, valid):
        flat = x.reshape(-1, x.size(-1))
        indices = valid.reshape(-1).nonzero(as_tuple=True)[0]
        tokens = flat[indices]
        if tokens.size(0) == 0:
            zero = x.sum() * 0
            return torch.zeros_like(x), zero, zero, x.new_zeros(self.n_experts)
        # Router en fp32 pour éviter la saturation en bf16/fp16.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            scores = F.linear(tokens.float(), self.router.weight.float())
            probs = scores.softmax(-1)
        weights, selected = probs.topk(self.top_k, dim=-1)
        if self.top_k != 1 or self.top1_gate == "normalized":
            weights = weights / weights.sum(-1, keepdim=True)
        mixed = self.shared(tokens)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(selected == expert_id)
            # Un expert vide reste un graphe valide, sans traitement de tokens inutiles.
            if rows.numel():
                expert_output = expert(tokens[rows])
                contribution = expert_output * weights[rows, slots, None].to(expert_output.dtype)
                mixed = mixed.index_add(0, rows, contribution)
        result = torch.zeros_like(flat).index_copy(0, indices, mixed.to(flat.dtype)).view_as(x)
        fraction = F.one_hot(selected, self.n_experts).float().mean((0, 1))
        balance = self.n_experts * (probs.mean(0) * fraction).sum()
        z_loss = scores.logsumexp(-1).square().mean()
        return result, balance, z_loss, fraction.detach()


class EncoderBlock(nn.Module):
    def __init__(self, config, layer_index=0):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(config.d_model), RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config, causal=False, layer_index=layer_index)
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
        self.blocks = nn.ModuleList([EncoderBlock(config, i) for i in range(config.vision_layers)])
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
        self.latents = nn.Parameter(torch.empty(1, config.n_latents, config.d_model))
        self.null_context = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.norms = nn.ModuleList([RMSNorm(config.d_model) for _ in range(config.resampler_layers)])
        self.cross = nn.ModuleList([CrossAttention(config) for _ in range(config.resampler_layers)])
        self.blocks = nn.ModuleList([EncoderBlock(config, i) for i in range(config.resampler_layers)])

    def forward(self, contexts, masks, batch_size):
        null = self.null_context.expand(batch_size, -1, -1)
        context = torch.cat([null, *contexts], 1)
        valid = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool, device=null.device), *masks
        ], 1)
        x = self.latents.expand(batch_size, -1, -1)
        latent_valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        for norm, cross, block in zip(self.norms, self.cross, self.blocks):
            x = x + cross(norm(x), context, valid)
            x = block(x, latent_valid)
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
        self.cross = CrossAttention(config)
        self.cross_gate = nn.Parameter(torch.full((config.d_model,), -1.0))
        self.moe = SparseMoE(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, valid, context, *, offset=0, cache=None, use_cache=False):
        h = self.norm1(x)
        attention, (k, v, key_valid) = self.attn(h, valid, offset=offset, cache=cache, use_cache=use_cache)
        local, conv_state = self.conv(h, valid, None if cache is None else cache.conv_state)
        x = x + self.dropout(attention + torch.sigmoid(self.mix_gate) * local)
        if cache is not None and cache.context_key is not None:
            context_kv = cache.context_key, cache.context_value
        else:
            context_kv = self.cross.project_context(context)
        x = x + self.dropout(torch.sigmoid(self.cross_gate) * self.cross(self.norm2(x), projected=context_kv))
        update, balance, z_loss, usage = self.moe(self.norm3(x), valid)
        x = (x + self.dropout(update)) * valid.unsqueeze(-1)
        next_cache = (LayerCache(k, v, key_valid, conv_state, offset + x.size(1) - k.size(-2), *context_kv)
                      if use_cache else None)
        return x, balance, z_loss, usage, next_cache


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
