"""Encodeur tabulaire hétérogène (PLR numérique + embeddings catégoriels + Transformer)."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .modules import EncoderBlock, RMSNorm


class TabularEncoder(nn.Module):
    """Encoder des données tabulaires mixtes (numériques avec NaN + catégorielles).

    Chaque colonne numérique est projetée par une base périodique apprise
    (PLR : sin/cos(2 pi f x) + terme linéaire) avec jeton dédié aux valeurs
    manquantes. Chaque colonne catégorielle possède son propre vocabulaire et
    un jeton de valeur manquante (-1). Les jetons de colonnes interagissent via
    des blocs Transformer bidirectionnels.
    """

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.num_features = config.tabular_num_features
        self.cat_features = config.tabular_cat_features
        self.cat_cardinality = config.tabular_cat_cardinality
        self.plr_frequencies = config.tabular_plr_frequencies
        plr_in = 2 * self.plr_frequencies + 1

        self.plr_freq = nn.Parameter(torch.empty(self.num_features, self.plr_frequencies))
        self.plr_weight = nn.Parameter(torch.empty(self.num_features, plr_in, d))
        self.plr_bias = nn.Parameter(torch.zeros(self.num_features, d))
        self.num_out = nn.Linear(d, d, bias=False)
        self.num_missing = nn.Parameter(torch.empty(self.num_features, d))

        if self.cat_features > 0:
            self.cat_embed = nn.Embedding(self.cat_features * self.cat_cardinality, d)
            self.cat_missing = nn.Parameter(torch.empty(self.cat_features, d))
        else:
            self.cat_embed = None
            self.cat_missing = None

        self.col_embed = nn.Parameter(torch.empty(1, self.num_features + self.cat_features, d))
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.tabular_layers)])
        self.norm = RMSNorm(d, config.norm_eps)
        self.reset_tabular_parameters()

    def reset_tabular_parameters(self):
        nn.init.normal_(self.plr_freq, std=0.2)
        nn.init.normal_(self.plr_weight, std=0.02)
        nn.init.zeros_(self.plr_bias)
        nn.init.normal_(self.num_missing, std=0.02)
        nn.init.normal_(self.col_embed, std=0.02)
        if self.cat_missing is not None:
            nn.init.normal_(self.cat_missing, std=0.02)

    def forward(self, tabular_num: torch.Tensor, tabular_cat: torch.Tensor | None = None):
        b = tabular_num.size(0)
        target_dtype = self.col_embed.dtype

        with torch.autocast(device_type=tabular_num.device.type, enabled=False):
            x_f = tabular_num.float()
            observed = torch.isfinite(x_f)
            clean = torch.where(observed, x_f, torch.zeros_like(x_f))
            angles = (2.0 * math.pi) * clean.unsqueeze(-1) * self.plr_freq.float().unsqueeze(0)
            periodic = torch.cat((torch.sin(angles), torch.cos(angles), clean.unsqueeze(-1)), dim=-1)
            projected = torch.einsum("bfk,fkd->bfd", periodic, self.plr_weight.float()) + self.plr_bias.float()
            num_tokens = F.silu(projected)

        num_tokens = self.num_out(num_tokens.to(target_dtype))
        num_tokens = torch.where(
            observed.unsqueeze(-1),
            num_tokens,
            self.num_missing.to(num_tokens.dtype).unsqueeze(0).expand(b, -1, -1),
        )

        if self.cat_features > 0:
            if tabular_cat is None:
                cat_tokens = self.cat_missing.to(num_tokens.dtype).unsqueeze(0).expand(b, -1, -1)
            else:
                cat_obs = tabular_cat >= 0
                offsets = (torch.arange(self.cat_features, device=tabular_cat.device, dtype=torch.long)
                           * self.cat_cardinality)
                safe_ids = tabular_cat.clamp(0, self.cat_cardinality - 1) + offsets.unsqueeze(0)
                embedded = self.cat_embed(safe_ids).to(num_tokens.dtype)
                cat_tokens = torch.where(
                    cat_obs.unsqueeze(-1),
                    embedded,
                    self.cat_missing.to(num_tokens.dtype).unsqueeze(0).expand(b, -1, -1),
                )
            tokens = torch.cat((num_tokens, cat_tokens), dim=1)
        else:
            tokens = num_tokens

        tokens = tokens + self.col_embed.to(tokens.dtype)
        valid = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        for block in self.blocks:
            tokens = block(tokens, valid, dense=True)
        return self.norm(tokens), valid
