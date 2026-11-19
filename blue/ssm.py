"""Branche récurrente sélective Gated DeltaNet à état matriciel borné O(1) avec scan parallèle WY."""

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
    Pour t > 1, l'algorithme par blocs WY (résolution triangulaire inférieure
    parallèle intra-bloc + scan matriciel inter-blocs) calcule exactement la
    même récurrence tout en réduisant le nombre d'itérations séquentielles par
    chunk_size. Les positions masquées (~valid) figent exactement l'état.
    """

    def __init__(self, d_model: int, state_dim: int, n_heads: int, *, norm_cls,
                 norm_eps: float = 1e-6, chunk_size: int = 32):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model doit être divisible par ssm_heads")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size doit être un entier strictement positif")
        self.d_model = d_model
        self.state_dim = state_dim
        self.n_heads = n_heads
        self.value_dim = d_model // n_heads
        self.chunk_size = chunk_size
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

    def _recurrence_sequential(self, q_f, k_f, v_f, decay_logits, beta_logits, valid, S):
        b, t, _, _ = q_f.shape
        alpha = torch.sigmoid(decay_logits.float())
        beta = torch.sigmoid(beta_logits.float())
        outputs = []
        for step in range(t):
            mask = valid[:, step].view(b, 1, 1, 1)
            a_t = alpha[:, step].view(b, self.n_heads, 1, 1)
            b_t = beta[:, step].view(b, self.n_heads, 1, 1)
            k_t = k_f[:, step]
            v_t = v_f[:, step]
            q_t = q_f[:, step]
            decayed = a_t * S
            v_pred = torch.einsum("bhkv,bhk->bhv", decayed, k_t)
            delta = (v_t - v_pred) * b_t.squeeze(-1)
            S_next = decayed + torch.einsum("bhk,bhv->bhkv", k_t, delta)
            S = torch.where(mask, S_next, S)
            o_t = torch.einsum("bhkv,bhk->bhv", S, q_t)
            o_t = torch.where(valid[:, step].view(b, 1, 1), o_t, torch.zeros_like(o_t))
            outputs.append(o_t)
        return torch.stack(outputs, dim=1), S

    def _recurrence_chunkwise_wy(self, q_f, k_f, v_f, decay_logits, beta_logits, valid, S):
        b, t, h, k_dim = q_f.shape
        v_dim = v_f.size(-1)
        c = min(self.chunk_size, t)
        pad = (c - (t % c)) % c
        if pad:
            q_f = F.pad(q_f, (0, 0, 0, 0, 0, pad))
            k_f = F.pad(k_f, (0, 0, 0, 0, 0, pad))
            v_f = F.pad(v_f, (0, 0, 0, 0, 0, pad))
            decay_logits = F.pad(decay_logits, (0, 0, 0, pad))
            beta_logits = F.pad(beta_logits, (0, 0, 0, pad))
            valid_pad = F.pad(valid, (0, pad), value=False)
        else:
            valid_pad = valid
        t_pad = t + pad
        n_chunks = t_pad // c

        # Réorganiser en [B, H, N, C, *] pour paralléliser les résolutions intra-bloc.
        q_c = q_f.transpose(1, 2).reshape(b, h, n_chunks, c, k_dim)
        k_c = k_f.transpose(1, 2).reshape(b, h, n_chunks, c, k_dim)
        v_c = v_f.transpose(1, 2).reshape(b, h, n_chunks, c, v_dim)
        val_c = valid_pad.unsqueeze(1).expand(b, h, t_pad).reshape(b, h, n_chunks, c)

        log_a = torch.where(val_c, F.logsigmoid(decay_logits.float().transpose(1, 2).reshape(b, h, n_chunks, c)),
                            torch.zeros_like(val_c, dtype=torch.float32))
        beta_c = torch.where(val_c, torch.sigmoid(beta_logits.float().transpose(1, 2).reshape(b, h, n_chunks, c)),
                             torch.zeros_like(val_c, dtype=torch.float32))
        g = log_a.cumsum(dim=-1)  # [B, H, N, C] <= 0
        exp_g = torch.exp(g)

        tril = torch.tril(torch.ones((c, c), device=q_f.device, dtype=torch.bool))
        strict_tril = torch.tril(torch.ones((c, c), device=q_f.device, dtype=torch.bool), diagonal=-1)
        eye = torch.eye(c, device=q_f.device, dtype=torch.float32)

        diff_g = (g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(~tril, -float("inf"))
        decay_mat = torch.exp(diff_g)  # [B, H, N, C, C] dans [0, 1] sur le triangle inférieur
        kk = torch.matmul(k_c, k_c.transpose(-1, -2))
        M = (beta_c.unsqueeze(-1) * decay_mat * kk).masked_fill(~strict_tril, 0.0)
        L = eye + M

        rhs_v = beta_c.unsqueeze(-1) * v_c
        rhs_k = (beta_c * exp_g).unsqueeze(-1) * k_c
        rhs = torch.cat((rhs_v, rhs_k), dim=-1)
        solved = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)
        u_val, w_mat = solved.split((v_dim, k_dim), dim=-1)  # [B, H, N, C, V], [B, H, N, C, K]

        # Précalcul des facteurs de sortie d'état en fin de bloc.
        g_end = g[..., -1:]  # [B, H, N, 1]
        exp_g_end = torch.exp(g_end).unsqueeze(-1)  # [B, H, N, 1, 1]
        k_exit = torch.exp(g_end - g).unsqueeze(-1) * k_c  # [B, H, N, C, K]

        states_in = []
        u_chunks = []
        for idx in range(n_chunks):
            states_in.append(S)
            u_chunk = u_val[:, :, idx] - torch.matmul(w_mat[:, :, idx], S)
            u_chunks.append(u_chunk)
            S = exp_g_end[:, :, idx] * S + torch.matmul(k_exit[:, :, idx].transpose(-1, -2), u_chunk)

        states_all = torch.stack(states_in, dim=2)  # [B, H, N, K, V]
        u_all = torch.stack(u_chunks, dim=2)  # [B, H, N, C, V]
        qk_decay = decay_mat * torch.matmul(q_c, k_c.transpose(-1, -2))
        o_inter = torch.matmul(exp_g.unsqueeze(-1) * q_c, states_all)
        o_intra = torch.matmul(qk_decay, u_all)
        out = (o_inter + o_intra) * val_c.unsqueeze(-1).float()
        out = out.reshape(b, h, t_pad, v_dim)[:, :, :t].transpose(1, 2)
        return out, S

    def forward(self, x: torch.Tensor, valid: torch.Tensor, state: torch.Tensor | None = None,
                *, use_chunkwise: bool | None = None):
        b, t, _ = x.shape
        q, k = self.qk_proj(x).view(b, t, 2, self.n_heads, self.state_dim).unbind(2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.value_dim)
        gate = F.silu(self.gate_proj(x)).view(b, t, self.n_heads, self.value_dim)
        decay_logits, beta_logits = self.decay_beta_proj(x).chunk(2, dim=-1)

        with torch.autocast(device_type=x.device.type, enabled=False):
            q_f = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
            k_f = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
            v_f = v.float()
            if state is None:
                S = x.new_zeros((b, self.n_heads, self.state_dim, self.value_dim), dtype=torch.float32)
            else:
                S = state.float()
            chunkwise = (t > 1 and self.chunk_size > 1) if use_chunkwise is None else use_chunkwise
            if chunkwise:
                out, S = self._recurrence_chunkwise_wy(q_f, k_f, v_f, decay_logits, beta_logits, valid, S)
            else:
                out, S = self._recurrence_sequential(q_f, k_f, v_f, decay_logits, beta_logits, valid, S)
        out = self.out_norm(out.to(x.dtype)) * gate
        out = self.out_proj(out.reshape(b, t, self.d_model)) * valid.unsqueeze(-1)
        return out, S
