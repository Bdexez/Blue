"""Branche récurrente sélective Gated DeltaNet à état matriciel borné O(1)."""

import torch
from torch import nn
from torch.nn import functional as F


class GatedDeltaNet(nn.Module):
    """Mémoire associative récurrente par règle Delta à décroissance sélective.

    Pour chaque tête h, l'état matriciel S_t de taille [d_k, d_v] évolue selon :
        S_decayed = alpha_t * S_{t-1}
        v_pred    = S_decayed^T k_t
        S_t       = S_decayed + beta_t * k_t (v_t - v_pred)^T
        o_t       = S_t^T q_t

    Les clés k_t et requêtes q_t sont normalisées L2 en FP32, ce qui borne le
    rayon spectral de (I - beta_t k_t k_t^T) <= 1 pour beta_t in (0, 1).
    Les positions masquées (~valid) figent exactement l'état.
    """

    def __init__(self, d_model: int, state_dim: int, n_heads: int, *, norm_cls, norm_eps: float = 1e-6):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model doit être divisible par ssm_heads")
        self.d_model = d_model
        self.state_dim = state_dim
        self.n_heads = n_heads
        self.value_dim = d_model // n_heads
        self.qk_proj = nn.Linear(d_model, 2 * n_heads * state_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.decay_beta_proj = nn.Linear(d_model, 2 * n_heads, bias=True)
        self.out_norm = norm_cls(self.value_dim, norm_eps)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.reset_gate_biases()

    def reset_gate_biases(self):
        # Biais initial : alpha_t démarre proche de 0.88 (rétention), beta_t proche de 0.27.
        with torch.no_grad():
            self.decay_beta_proj.bias[:self.n_heads].fill_(2.0)
            self.decay_beta_proj.bias[self.n_heads:].fill_(-1.0)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, state: torch.Tensor | None = None):
        b, t, _ = x.shape
        q, k = self.qk_proj(x).view(b, t, 2, self.n_heads, self.state_dim).unbind(2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.value_dim)
        gate = F.silu(self.gate_proj(x)).view(b, t, self.n_heads, self.value_dim)
        decay_logits, beta_logits = self.decay_beta_proj(x).chunk(2, dim=-1)

        with torch.autocast(device_type=x.device.type, enabled=False):
            q_f = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
            k_f = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
            v_f = v.float()
            alpha = torch.sigmoid(decay_logits.float())
            beta = torch.sigmoid(beta_logits.float())
            if state is None:
                S = x.new_zeros((b, self.n_heads, self.state_dim, self.value_dim), dtype=torch.float32)
            else:
                S = state.float()
            outputs = []
            for step in range(t):
                mask = valid[:, step].view(b, 1, 1, 1)
                a_t = alpha[:, step].view(b, self.n_heads, 1, 1)
                b_t = beta[:, step].view(b, self.n_heads, 1, 1)
                k_t = k_f[:, step]  # [B, H, K]
                v_t = v_f[:, step]  # [B, H, V]
                q_t = q_f[:, step]  # [B, H, K]
                decayed = a_t * S
                # v_pred: [B, H, V] = sum_k decayed[b, h, k, v] * k_t[b, h, k]
                v_pred = torch.einsum("bhkv,bhk->bhv", decayed, k_t)
                delta = (v_t - v_pred) * b_t.squeeze(-1)
                S_next = decayed + torch.einsum("bhk,bhv->bhkv", k_t, delta)
                S = torch.where(mask, S_next, S)
                o_t = torch.einsum("bhkv,bhk->bhv", S, q_t)
                o_t = torch.where(valid[:, step].view(b, 1, 1), o_t, torch.zeros_like(o_t))
                outputs.append(o_t)
            out = torch.stack(outputs, dim=1)  # [B, T, H, V]
        out = self.out_norm(out.to(x.dtype)) * gate
        out = self.out_proj(out.reshape(b, t, self.d_model)) * valid.unsqueeze(-1)
        return out, S
