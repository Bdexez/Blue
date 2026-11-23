"""Embeddings causaux multi-échelle de N-grammes d'octets par hachage polynomial (style BLT)."""

import torch
from torch import nn

from .modules import RMSNorm


class ByteNGramEmbedding(nn.Module):
    """Enrichir les embeddings d'octets UTF-8 par des 2/3/4-grammes causaux hachés.

    Strictement causal, invariant au padding gauche/droite (un N-gramme n'est actif
    que si ses N octets sont tous valides) et exactement équivalent avec cache
    incrémental via les 3 derniers octets et masques de validité.
    """

    def __init__(self, vocab_size: int, d_model: int, *, norm_eps: float = 1e-6):
        super().__init__()
        if type(vocab_size) is not int or vocab_size < 16:
            raise ValueError("ngram_vocab_size doit être un entier >= 16")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.table = nn.Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model, norm_eps)
        self.order_gates = nn.Parameter(torch.full((3, d_model), -2.0))
        self.reset_ngram_parameters()

    def reset_ngram_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.table.weight, std=0.02)
            self.order_gates.fill_(-2.0)

    def forward(self, input_ids: torch.Tensor, valid: torch.Tensor,
                prev_bytes: torch.Tensor | None = None, prev_valid: torch.Tensor | None = None):
        b = input_ids.size(0)
        device = input_ids.device
        if prev_bytes is None or prev_valid is None:
            p_bytes = torch.zeros((b, 3), device=device, dtype=torch.long)
            p_valid = torch.zeros((b, 3), device=device, dtype=torch.bool)
        else:
            p_bytes = prev_bytes
            p_valid = prev_valid

        ext_ids = torch.cat((p_bytes, input_ids), dim=1)       # [B, T + 3]
        ext_valid = torch.cat((p_valid, valid), dim=1)          # [B, T + 3]
        clean_ids = torch.where(ext_valid, ext_ids, torch.zeros_like(ext_ids))

        b0 = clean_ids[:, 3:]       # t
        b1 = clean_ids[:, 2:-1]     # t - 1
        b2 = clean_ids[:, 1:-2]     # t - 2
        b3 = clean_ids[:, :-3]      # t - 3

        v0 = ext_valid[:, 3:]
        v1 = v0 & ext_valid[:, 2:-1]
        v2 = v1 & ext_valid[:, 1:-2]
        v3 = v2 & ext_valid[:, :-3]

        mod = self.vocab_size
        h2 = (b1 * 1000003 + b0 * 1000033 + 1) % mod
        h3 = (b2 * 1000037 + b1 * 1000039 + b0 * 1000081 + 2) % mod
        h4 = (b3 * 1000099 + b2 * 1000117 + b1 * 1000121 + b0 * 1000133 + 3) % mod

        gates = torch.sigmoid(self.order_gates).to(self.table.weight.dtype)  # [3, D]
        out = (
            self.table(h2) * (v1.unsqueeze(-1).to(gates.dtype) * gates[0])
            + self.table(h3) * (v2.unsqueeze(-1).to(gates.dtype) * gates[1])
            + self.table(h4) * (v3.unsqueeze(-1).to(gates.dtype) * gates[2])
        )
        out = self.norm(out) * v0.unsqueeze(-1)
        next_bytes = ext_ids[:, -3:].clone()
        next_valid = ext_valid[:, -3:].clone()
        return out, next_bytes, next_valid
