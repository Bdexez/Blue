"""Quantification scalaire finie (FSQ, Mentzer et al., ICLR 2024) sans codebook."""

import math
import torch
from torch import nn


class FiniteScalarQuantizer(nn.Module):
    """Goulot d'étranglement discret FSQ par arrondi borné avec estimateur STE.

    Chaque dimension k parmi K reçoit L_k >= 2 niveaux discrets régulièrement espacés,
    formant un vocabulaire implicite de taille prod(L_k) sans effondrement de codebook.
    """

    def __init__(self, d_model: int, levels: tuple[int, ...], *, norm_cls, norm_eps: float = 1e-6):
        super().__init__()
        if not levels or any(type(level) is not int or level < 2 for level in levels):
            raise ValueError("latent_fsq_levels doit contenir des entiers >= 2")
        self.levels = tuple(levels)
        self.codebook_size = math.prod(self.levels)
        self.bottleneck_dim = len(self.levels)
        self.norm = norm_cls(d_model, norm_eps)
        self.down = nn.Linear(d_model, self.bottleneck_dim, bias=True)
        self.up = nn.Linear(self.bottleneck_dim, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(d_model))

    def _quantize(self, z: torch.Tensor):
        with torch.autocast(device_type=z.device.type, enabled=False):
            zf = z.float()
            levels = torch.tensor(self.levels, device=z.device, dtype=torch.float32)
            half_width = (levels - 1.0) * 0.5
            # Pour tout L >= 2 (pair ou impair), round(tanh(z)*half_width + half_width)
            # produit exactement L niveaux uniformes dans {0, ..., L - 1}.
            shifted = torch.tanh(zf) * half_width
            integer = torch.minimum(torch.round(shifted + half_width).clamp_min(0.0), levels - 1.0)
            quantized = integer - half_width
            ste = shifted + (quantized - shifted).detach()
            normalized = ste / half_width.clamp_min(1.0)
            codes = integer.long()
            # Index scalaire unique dans [0, codebook_size - 1].
            basis = []
            stride = 1
            for lvl in self.levels:
                basis.append(stride)
                stride *= lvl
            basis_t = torch.tensor(basis, device=z.device, dtype=torch.long)
            indices = (codes * basis_t).sum(dim=-1)
            return normalized.to(z.dtype), indices

    def forward(self, x: torch.Tensor):
        z = self.down(self.norm(x))
        q, indices = self._quantize(z)
        out = x + torch.sigmoid(self.gate) * self.up(q)
        return out, indices
