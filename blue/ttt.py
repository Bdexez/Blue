"""Couche Test-Time Training (TTT-Linear) à poids rapides neuronaux mis à jour en ligne."""

import torch
from torch import nn
from torch.nn import functional as F


def _inner_layernorm(z: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    return F.layer_norm(z, z.shape[-1:], eps=eps) * gamma + beta


def _inner_layernorm_and_grad(z: torch.Tensor, target: torch.Tensor,
                              gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    """Calculer LN(z) et le gradient analytique exact de 0.5 * ||LN(z) - target||^2 wrt z."""
    return (_inner_layernorm(z, gamma, beta, eps),
            _inner_grad(z, gamma.square(), gamma * (beta - target), eps))


def _inner_grad(z: torch.Tensor, gamma_sq: torch.Tensor, offset: torch.Tensor, eps: float = 1e-5):
    """Gradient analytique exact de 0.5 * ||LN(z) - cible||^2 par rapport à z.

    Avec ``offset = gamma * (beta - cible)`` précalculé, le gradient par rapport à
    l'activation normalisée est ``gamma^2 * z_hat + offset`` ; il est ensuite projeté
    par la jacobienne de la normalisation.
    """
    z_hat = F.layer_norm(z, z.shape[-1:], eps=eps)
    inv_std = torch.rsqrt(z.var(dim=-1, keepdim=True, correction=0) + eps)
    grad = torch.addcmul(offset, gamma_sq, z_hat)
    projected = torch.addcmul(grad - grad.mean(dim=-1, keepdim=True), z_hat,
                              (grad * z_hat).mean(dim=-1, keepdim=True), value=-1.0)
    return inv_std * projected


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
                 norm_cls, norm_eps: float = 1e-6, base_lr: float = 0.5,
                 weight_decay: bool = False):
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
        self.decay_proj = nn.Linear(d_model, n_heads, bias=True) if weight_decay else None
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
            if self.decay_proj is not None:
                self.decay_proj.weight.zero_()
                self.decay_proj.bias.fill_(4.0)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        packed = torch.cat((self.W0.float(), self.b0.float().unsqueeze(-2)), dim=-2)  # [H, K+1, K]
        return packed.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, state: torch.Tensor | None = None):
        b, t, _ = x.shape
        q, k, v = self.qkv_proj(x).view(b, t, 3, self.n_heads, self.state_dim).unbind(2)
        gate = F.silu(self.gate_proj(x)).view(b, t, self.n_heads, self.state_dim)
        lr_logits = self.lr_proj(x)
        decay_logits = self.decay_proj(x) if self.decay_proj is not None else None

        with torch.autocast(device_type=x.device.type, enabled=False):
            q_f = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
            k_f = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
            v_f = v.float()
            eta = self.base_lr * torch.sigmoid(lr_logits.float())  # [B, T, H]
            alpha = torch.sigmoid(decay_logits.float()) if decay_logits is not None else None
            gamma = self.ln_weight.float()  # [H, K]
            beta = self.ln_bias.float()
            # Le biais est la dernière ligne de l'état empilé : étendre clé et requête
            # par 1 adapte et applique (W, b) en une seule opération matricielle.
            initial = self.initial_state(b, x.device)
            packed = (initial if state is None else state.float()).flatten(0, 1)  # [BH, K+1, K]
            anchor = initial.flatten(0, 1) if alpha is not None else None

            def steps(values):
                # unbind : un seul nœud arrière, au lieu d'un tenseur complet par pas indexé.
                return values.transpose(1, 2).flatten(0, 1).unbind(1)

            k_ext = F.pad(k_f, (0, 1), value=1.0)
            q_ext = F.pad(q_f, (0, 1), value=1.0)
            k_steps, q_steps = steps(k_ext.unsqueeze(-2)), steps(q_ext.unsqueeze(-2))  # [BH, 1, K+1]
            k_columns = steps(k_ext.unsqueeze(-1))  # [BH, K+1, 1]
            gamma_sq = gamma.square().expand(b, -1, -1).flatten(0, 1)  # [BH, K]
            offset_steps = steps(gamma * (beta - (v_f - k_f)))
            eta_steps = steps(eta.unsqueeze(-1))  # [BH, 1]
            alpha_steps = steps(alpha[..., None, None]) if alpha is not None else None
            mask_steps = None
            if not bool(valid.all()):
                mask_steps = steps(valid[:, :, None, None, None].expand(b, t, self.n_heads, 1, 1))

            outputs = []
            for step in range(t):
                # 1. Passe avant interne et gradient analytique auto-supervisé.
                z_k = torch.bmm(k_steps[step], packed).squeeze(1)
                update = eta_steps[step] * _inner_grad(z_k, gamma_sq, offset_steps[step])
                # 2. Mise à jour des poids rapides (avec relaxation vers W0/b0 si activée).
                base = packed
                if alpha_steps is not None:
                    base = alpha_steps[step] * packed + (1.0 - alpha_steps[step]) * anchor
                updated = torch.baddbmm(base, k_columns[step], update.unsqueeze(1), alpha=-1.0)
                packed = updated if mask_steps is None else torch.where(mask_steps[step], updated, packed)
                # 3. Inférence du mini-réseau adapté sur la requête courante q_t.
                outputs.append(torch.bmm(q_steps[step], packed))

            # La normalisation interne des sorties n'alimente pas la récurrence : une seule passe.
            z_q = torch.cat(outputs, dim=1).view(b, self.n_heads, t, self.state_dim).transpose(1, 2)
            out = q_f + _inner_layernorm(z_q, gamma, beta)
            if mask_steps is not None:
                out = torch.where(valid[:, :, None, None], out, torch.zeros_like(out))
            next_state = packed.view(b, self.n_heads, self.state_dim + 1, self.state_dim)

        out = self.out_norm(out.to(x.dtype)) * gate
        out = self.out_proj(out.reshape(b, t, self.n_heads * self.state_dim)) * valid.unsqueeze(-1)
        return out, next_state
