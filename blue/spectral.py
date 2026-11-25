"""Mélange spectral continu dans le domaine de Fourier (rFFT) pour séries temporelles."""

import torch
from torch import nn

from .ragged import map_valid_positions


class SpectralTemporalMixer(nn.Module):
    """Filtrage fréquentiel multi-bandes invariant au padding et compatible longueur variable.

    Le spectre est calculé uniquement sur les positions valides de chaque exemple
    en FP32 (y compris sous autocast BF16/FP16), filtré par une combinaison continue
    de bandes de fréquences normalisées dans [0, 1], puis ramené dans le domaine temporel.
    """

    def __init__(self, d_model: int, n_bands: int = 4, *, norm_cls, norm_eps: float = 1e-6):
        super().__init__()
        if n_bands < 1:
            raise ValueError("n_bands doit être un entier strictement positif")
        self.d_model = d_model
        self.n_bands = n_bands
        self.norm = norm_cls(d_model, norm_eps)
        self.band_real = nn.Parameter(torch.ones(n_bands, d_model))
        self.band_imag = nn.Parameter(torch.zeros(n_bands, d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Parameter(torch.full((d_model,), -1.0))
        self.reset_spectral_parameters()

    def reset_spectral_parameters(self):
        with torch.no_grad():
            self.band_real.fill_(1.0)
            self.band_imag.zero_()
            self.gate.fill_(-1.0)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        width = max(1.0 / max(self.n_bands - 1, 1), 0.25)
        with torch.autocast(device_type=x.device.type, enabled=False):
            centers = torch.linspace(0.0, 1.0, self.n_bands, device=x.device, dtype=torch.float32)
            band_r = self.band_real.float()
            band_i = self.band_imag.float()

            def mix(packed):
                # [G, L, D] : spectre de chaque exemple sur ses seules positions valides.
                u = packed.float()
                length = u.size(1)
                spec = torch.fft.rfft(u, dim=1, norm="ortho")
                freqs = torch.linspace(0.0, 1.0, spec.size(1), device=x.device, dtype=torch.float32)
                weights = torch.softmax(-((freqs[:, None] - centers[None, :]) / width).square(), dim=-1)
                filt_r = weights @ band_r
                filt_i = weights @ band_i
                out_r = spec.real * filt_r - spec.imag * filt_i
                out_i = spec.real * filt_i + spec.imag * filt_r
                return torch.fft.irfft(torch.complex(out_r, out_i), n=length, dim=1, norm="ortho").to(x.dtype)

            stacked = map_valid_positions(h, valid, mix)
        return torch.sigmoid(self.gate) * self.out_proj(stacked) * valid.unsqueeze(-1)
