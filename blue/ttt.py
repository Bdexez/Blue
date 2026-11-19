"""Couche Test-Time Training (TTT-Linear) à poids rapides neuronaux mis à jour en ligne."""

import torch
from torch import nn
from torch.nn import functional as F


def _inner_layernorm_and_grad(z: torch.Tensor, target: torch.Tensor,
                              gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    """Calculer LN(z) et le gradient analytique exact de 0.5 * ||LN(z) - target||^2 wrt z."""
    mean = z.mean(dim=-1, keepdim=True)
    centered = z - mean
    inv_std = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + eps)
    z_hat = centered * inv_std
    y = z_hat * gamma + beta
    grad_y = y - target
    grad_z_hat = grad_y * gamma
    grad_z = inv_std * (
        grad_z_hat
        - grad_z_hat.mean(dim=-1, keepdim=True)
        - z_hat * (grad_z_hat * z_hat).mean(dim=-1, keepdim=True)
    )
    return y, grad_z


def _inner_layernorm(z: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    mean = z.mean(dim=-1, keepdim=True)
    centered = z - mean
    inv_std = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + eps)
    return (centered * inv_std) * gamma + beta


class TTTLinear(nn.Module):
    """Couche Test-Time Training (Sun et al., 2024) à état neuronal O(1).

    L'état caché de chaque tête est un mini-réseau neuronal f(z; W_t, b_t) = z + LN(z W_t + b_t)
    initialisé aux méta-paramètres appris (W_0, b_0) et adapté à chaque token t
    par descente de gradient analytique sur l'objectif auto-supervisé :
        L_t(W, b) = 0.5 * || LN(k_t W + b) - (v_t - k_t) ||_2^2
    avec pas d'apprentissage appris par token eta_t. L'état empilé [B, H, K+1, K]
    est conservé en mémoire O(1) dans LayerCache.ttt_state ; les positions
    masquées (~valid) figent strictement (W_t, b_t).
    """

    def __init__(self, d_model: int, state_dim: int, n_heads: int, *,
                 norm_cls, norm_eps: float = 1e-6, base_lr: float = 0.5):
        super().__init__()
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("ttt_state_dim doit être un entier strictement positif")
        if type(n_heads) is not int or n_heads < 1:
            raise ValueError("ttt_heads doit être un entier strictement positif")
        self.d_model = d_model
        self.state_dim = state_dim
        self.n_heads = n_heads
        self.base_lr = float(base_lr)
        inner_total = n_heads * state_dim

        self.qkv_proj = nn.Linear(d_model, 3 * inner_total, bias=False)
        self.lr_proj = nn.Linear(d_model, n_heads, bias=True)
        self.gate_proj = nn.Linear(d_model, inner_total, bias=False)
        self.out_norm = norm_cls(state_dim, norm_eps)
        self.out_proj = nn.Linear(inner_total, d_model, bias=False)

        # Méta-paramètres initiaux du modèle interne (appris par la boucle externe).
        self.W0 = nn.Parameter(torch.empty(n_heads, state_dim, state_dim))
        self.b0 = nn.Parameter(torch.zeros(n_heads, state_dim))
        self.ln_weight = nn.Parameter(torch.ones(n_heads, state_dim))
        self.ln_bias = nn.Parameter(torch.zeros(n_heads, state_dim))
        self.reset_ttt_parameters()

    def reset_ttt_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.W0, std=0.02)
            nn.init.zeros_(self.b0)
            nn.init.ones_(self.ln_weight)
            nn.init.zeros_(self.ln_bias)
            self.lr_proj.bias.fill_(-2.0)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        packed = torch.cat((self.W0.float(), self.b0.float().unsqueeze(-2)), dim=-2)  # [H, K+1, K]
        return packed.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous().to(device)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, state: torch.Tensor | None = None):
        b, t, _ = x.shape
        q, k, v = self.qkv_proj(x).view(b, t, 3, self.n_heads, self.state_dim).unbind(2)
        gate = F.silu(self.gate_proj(x)).view(b, t, self.n_heads, self.state_dim)
        lr_logits = self.lr_proj(x)

        with torch.autocast(device_type=x.device.type, enabled=False):
            q_f = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
            k_f = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
            v_f = v.float()
            eta = self.base_lr * torch.sigmoid(lr_logits.float())  # [B, T, H]
            gamma = self.ln_weight.float().unsqueeze(0)  # [1, H, K]
            beta = self.ln_bias.float().unsqueeze(0)     # [1, H, K]

            if state is None:
                W = self.W0.float().unsqueeze(0).expand(b, -1, -1, -1)
                bias = self.b0.float().unsqueeze(0).expand(b, -1, -1)
            else:
                s_f = state.float()
                W = s_f[:, :, :-1, :]
                bias = s_f[:, :, -1, :]

            outputs = []
            for step in range(t):
                mask_w = valid[:, step].view(b, 1, 1, 1)
                mask_b = valid[:, step].view(b, 1, 1)
                k_t = k_f[:, step]          # [B, H, K]
                v_t = v_f[:, step]          # [B, H, K]
                q_t = q_f[:, step]          # [B, H, K]
                eta_t = eta[:, step].unsqueeze(-1)  # [B, H, 1]

                # 1. Passe avant interne et gradient analytique auto-supervisé.
                z_k = torch.einsum("bhk,bhkd->bhd", k_t, W) + bias
                _, grad_z = _inner_layernorm_and_grad(z_k, v_t - k_t, gamma, beta)
                # 2. Mise à jour des poids rapides neuronaux à temps de test.
                dW = torch.einsum("bhk,bhd->bhkd", k_t, eta_t * grad_z)
                W_next = W - dW
                b_next = bias - eta_t * grad_z
                W = torch.where(mask_w, W_next, W)
                bias = torch.where(mask_b, b_next, bias)
                # 3. Inférence du mini-réseau adapté sur la requête courante q_t.
                z_q = torch.einsum("bhk,bhkd->bhd", q_t, W) + bias
                o_t = q_t + _inner_layernorm(z_q, gamma, beta)
                o_t = torch.where(mask_b, o_t, torch.zeros_like(o_t))
                outputs.append(o_t)

            out = torch.stack(outputs, dim=1)  # [B, T, H, K]
            next_state = torch.cat((W, bias.unsqueeze(-2)), dim=-2)  # [B, H, K+1, K]

        out = self.out_norm(out.to(x.dtype)) * gate
        out = self.out_proj(out.reshape(b, t, self.n_heads * self.state_dim)) * valid.unsqueeze(-1)
        return out, next_state
