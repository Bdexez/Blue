"""Mélange multi-résolution par ondelettes orthogonales de Haar 1D invariant au padding."""

import math

import torch
from torch import nn


class HaarWaveletMixer(nn.Module):
    """Décomposition et reconstruction exacte en ondelettes de Haar multi-niveaux.

    Opère uniquement sur les positions valides de chaque série en FP32 (invariance
    stricte au padding gauche, droite ou interne). À chaque niveau dyadique l :
        a_{l+1}[k] = (a_l[2k] + a_l[2k+1]) / sqrt(2)
        d_{l+1}[k] = (a_l[2k] - a_l[2k+1]) / sqrt(2)
    Les coefficients de détail subissent un gain appris par canal et par échelle,
    puis la transformation inverse IDWT reconstruit exactement la longueur L.
    """

    def __init__(self, d_model: int, levels: int = 3, *, norm_cls, norm_eps: float = 1e-6):
        super().__init__()
        if type(levels) is not int or levels < 1:
            raise ValueError("levels doit être un entier strictement positif")
        self.d_model = d_model
        self.levels = levels
        self.norm = norm_cls(d_model, norm_eps)
        self.approx_gain = nn.Parameter(torch.ones(d_model))
        self.detail_gain = nn.Parameter(torch.ones(levels, d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Parameter(torch.full((d_model,), -1.0))
        self.reset_wavelet_parameters()

    def reset_wavelet_parameters(self):
        with torch.no_grad():
            self.approx_gain.fill_(1.0)
            self.detail_gain.fill_(1.0)
            self.gate.fill_(-1.0)

    def _dwt_idwt_1d(self, u: torch.Tensor) -> torch.Tensor:
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        a_Gain = self.approx_gain.float()
        d_Gain = self.detail_gain.float()
        lengths = []
        details = []
        a = u
        for lvl in range(self.levels):
            n = a.size(0)
            lengths.append(n)
            if n < 2:
                details.append(None)
                continue
            if n % 2 == 1:
                a = torch.cat((a, a[-1:]), dim=0)
            even = a[0::2]
            odd = a[1::2]
            a = (even + odd) * inv_sqrt2
            d = (even - odd) * inv_sqrt2 * d_Gain[lvl]
            details.append(d)

        a = a * a_Gain
        for lvl in range(self.levels - 1, -1, -1):
            d = details[lvl]
            orig_n = lengths[lvl]
            if d is None:
                continue
            even = (a + d) * inv_sqrt2
            odd = (a - d) * inv_sqrt2
            a = torch.stack((even, odd), dim=1).reshape(-1, self.d_model)[:orig_n]
        return a

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.norm(x)
        rows = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for row in range(b):
                row_valid = valid[row]
                u = h[row, row_valid].float()
                rec = self._dwt_idwt_1d(u).to(x.dtype)
                row_out = x.new_zeros(t, d).masked_scatter(row_valid.unsqueeze(-1).expand(t, d), rec)
                rows.append(row_out)
        stacked = torch.stack(rows, dim=0)
        return torch.sigmoid(self.gate) * self.out_proj(stacked) * valid.unsqueeze(-1)
