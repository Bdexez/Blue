"""Encodeur audio multi-échelles 1D avec masquage exact du padding."""

import torch
from torch import nn
from torch.nn import functional as F

from .modules import EncoderBlock, RMSNorm
from .spectral import SpectralTemporalMixer


class AudioEncoder(nn.Module):
    """Encodeur de formes d'onde / signaux acoustiques [B, C, L] -> latents [B, T, D].

    Combine un tronc convolutif multi-échelles (noyaux 3, 7, 15) avec découpage en
    trames de pas `audio_patch_size`, mélange spectral optionnel et blocs Transformer
    bidirectionnels. Les échantillons masqués (audio_mask=False) sont remis à zéro
    avant chaque convolution et les trames entièrement masquées sont ignorées.
    """

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.patch_size = config.audio_patch_size
        self.stem = nn.Conv1d(config.audio_channels, d // 2, kernel_size=3, padding=1, bias=False)
        self.scale3 = nn.Conv1d(d // 2, d // 2, kernel_size=3, padding=1, groups=d // 2, bias=False)
        self.scale7 = nn.Conv1d(d // 2, d // 2, kernel_size=7, padding=3, groups=d // 2, bias=False)
        self.scale15 = nn.Conv1d(d // 2, d // 2, kernel_size=15, padding=7, groups=d // 2, bias=False)
        self.patch = nn.Conv1d(d // 2, d, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.spectral = (SpectralTemporalMixer(d, norm_cls=RMSNorm, norm_eps=config.norm_eps)
                         if config.temporal_spectral_mixing else None)
        self.blocks = nn.ModuleList([EncoderBlock(config, i) for i in range(config.audio_layers)])
        self.norm = RMSNorm(d, config.norm_eps)

    def forward(self, audio: torch.Tensor, valid: torch.Tensor):
        # audio: [B, C, L], valid: [B, L] bool
        b, _, length = audio.shape
        rem = length % self.patch_size
        if rem:
            pad = self.patch_size - rem
            audio = F.pad(audio, (0, pad))
            valid = F.pad(valid, (0, pad), value=False)
        sample_mask = valid.unsqueeze(1)  # [B, 1, L_pad]
        x = audio.to(self.stem.weight.dtype) * sample_mask
        # Normalisation par exemple sur les échantillons valides (invariance d'amplitude).
        with torch.autocast(device_type=x.device.type, enabled=False):
            vf = valid.unsqueeze(1).float()
            xf = x.float() * vf
            denom = vf.sum(-1, keepdim=True).clamp_min(1.0)
            mean = xf.sum(-1, keepdim=True) / denom
            centered = (xf - mean) * vf
            rms = (centered.square().sum(-1, keepdim=True) / denom + 1e-5).sqrt()
            x = (centered / rms).to(self.stem.weight.dtype)
        # Application du tronc multi-échelles en remasquant entre étapes pour isoler le padding.
        h = F.silu(self.stem(x)) * sample_mask
        h = (h + F.silu(self.scale3(h)) + F.silu(self.scale7(h)) + F.silu(self.scale15(h))) * sample_mask
        tokens = self.patch(h).transpose(1, 2)  # [B, T_frames, D]
        frame_valid = valid.view(b, -1, self.patch_size).any(dim=-1)  # [B, T_frames]
        tokens = tokens * frame_valid.unsqueeze(-1)
        if self.spectral is not None:
            tokens = tokens + self.spectral(tokens, frame_valid)
        dense = bool(frame_valid.all()) if self.blocks else None
        for block in self.blocks:
            tokens = block(tokens, frame_valid, dense=dense)
        return self.norm(tokens) * frame_valid.unsqueeze(-1), frame_valid
