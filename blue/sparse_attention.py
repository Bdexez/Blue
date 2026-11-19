"""Branche Native Sparse Attention (NSA) par compression de blocs causaux et sélection Top-K."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CompressedBlockSparseAttention(nn.Module):
    """Attention clairsemée hiérarchique sur blocs KV compressés (style DeepSeek NSA).

    Partitionne les clés et valeurs passées en blocs contigus de taille B, compresse
    chaque bloc par pondération positionnelle apprise + projection, puis sélectionne
    les top-k blocs causaux les plus pertinents pour chaque tête et position requête.
    Seuls les blocs entièrement achevés à la position requête (fin_bloc <= pos_requête)
    sont visibles : causalité stricte et équivalence exacte du cache garanties.
    """

    def __init__(self, n_heads: int, head_dim: int, block_size: int, top_k: int, *, key_dim: int | None = None):
        super().__init__()
        if type(block_size) is not int or block_size < 2:
            raise ValueError("nsa_block_size doit être un entier >= 2")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("nsa_top_k doit être un entier >= 1")
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.key_dim = head_dim if key_dim is None else key_dim
        self.block_size = block_size
        self.top_k = top_k
        self.pos_logits = nn.Parameter(torch.zeros(block_size))
        self.k_compress = nn.Linear(self.key_dim, self.key_dim, bias=False)
        self.v_compress = nn.Linear(head_dim, head_dim, bias=False)
        self.gate = nn.Parameter(torch.full((1, n_heads, 1, 1), -2.0))
        self.reset_nsa_parameters()

    def reset_nsa_parameters(self):
        with torch.no_grad():
            self.pos_logits.zero_()
            self.gate.fill_(-2.0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_valid: torch.Tensor, *, offset: int = 0, start_pos: int = 0) -> torch.Tensor:
        # q: [B, H, T_q, D_q], k: [B, H_kv, S, D_q], v: [B, H_kv, S, D_v], key_valid: [B, S]
        b, h, t_q, d_q = q.shape
        seq_k = k.size(-2)
        bs = self.block_size
        n_blocks = seq_k // bs
        if n_blocks == 0 or start_pos != 0:
            return q.new_zeros((b, h, t_q, v.size(-1)))

        if k.size(1) != h:
            repeats = h // k.size(1)
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        usable = n_blocks * bs
        k_blk = k[:, :, :usable].view(b, h, n_blocks, bs, d_q)
        v_blk = v[:, :, :usable].view(b, h, n_blocks, bs, v.size(-1))
        val_blk = key_valid[:, :usable].view(b, 1, n_blocks, bs)  # [B, 1, N, bs]
        block_valid = val_blk.any(dim=-1)  # [B, 1, N]

        with torch.autocast(device_type=q.device.type, enabled=False):
            logits_p = self.pos_logits.float().view(1, 1, 1, bs).expand(b, 1, n_blocks, bs)
            logits_p = logits_p.masked_fill(~val_blk, -float("inf"))
            w_p = torch.softmax(logits_p, dim=-1)
            w_p = torch.nan_to_num(w_p, nan=0.0).unsqueeze(-1)  # [B, 1, N, bs, 1]
            k_sum = (k_blk.float() * w_p).sum(dim=-2)  # [B, H, N, d_q]
            v_sum = (v_blk.float() * w_p).sum(dim=-2)  # [B, H, N, d_v]

        k_comp = self.k_compress(k_sum.to(q.dtype))
        v_comp = self.v_compress(v_sum.to(v.dtype))

        with torch.autocast(device_type=q.device.type, enabled=False):
            q_f = q.float()
            k_f = k_comp.float()
            v_f = v_comp.float()
            scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * (1.0 / math.sqrt(d_q))  # [B, H, T_q, N]

            # Causalité stricte : le bloc m couvre [m*bs, (m+1)*bs - 1] ; il est visible ssi (m+1)*bs - 1 <= pos_q.
            block_ends = (torch.arange(1, n_blocks + 1, device=q.device) * bs) - 1  # [N]
            query_pos = torch.arange(offset, offset + t_q, device=q.device)         # [T_q]
            allowed = (block_ends[None, :] <= query_pos[:, None])[None, None, :, :] & block_valid.unsqueeze(2)

            scores = scores.masked_fill(~allowed, -float("inf"))
            k_sel = min(self.top_k, n_blocks)
            if k_sel < n_blocks:
                top_vals, top_idx = scores.topk(k_sel, dim=-1)
                sparse_scores = torch.full_like(scores, -float("inf"))
                sparse_scores.scatter_(-1, top_idx, top_vals)
                scores = sparse_scores

            probs = torch.softmax(scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            # Les positions sans aucun bloc complet passé reçoivent 0.
            has_any = allowed.any(dim=-1, keepdim=True)
            probs = torch.where(has_any, probs, torch.zeros_like(probs))
            out = torch.matmul(probs, v_f)

        gate = torch.sigmoid(self.gate).to(v.dtype)
        return out.to(v.dtype) * gate
