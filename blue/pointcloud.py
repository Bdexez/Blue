"""Encodeur géométrique de nuages de points 3D strictement invariant SE(3) et par permutation."""

import torch
from torch import nn

from .graph import GraphTransformerBlock
from .modules import RMSNorm


def se3_invariant_geometry(points: torch.Tensor, valid: torch.Tensor, *, n_rbf: int = 8):
    """Extraire des descripteurs géométriques nodaux et pair-à-pair invariants SE(3).

    Toute transformation rigide x -> R x + t (R orthogonal, t in R^3) laisse
    strictement invariants :
      - les distances au centroïde ||x_i - c||_2,
      - les distances pair-à-pair ||x_i - x_j||_2 et leur base radiale gaussienne,
      - les produits scalaires normalisés <x_i - c, x_j - c> / (||x_i - c|| ||x_j - c||).
    Les points masqués (~valid) n'affectent jamais le centroïde ni les statistiques.
    """
    b, p, _ = points.shape
    with torch.autocast(device_type=points.device.type, enabled=False):
        pts = points.float()
        vf = valid.float()
        count = vf.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
        centroid = (pts * vf.unsqueeze(-1)).sum(dim=1, keepdim=True) / count.unsqueeze(-1)
        centered = (pts - centroid) * vf.unsqueeze(-1)  # [B, P, 3]
        radius = centered.square().sum(dim=-1).clamp_min(1e-12).sqrt() * vf  # [B, P]

        pair_mask = valid[:, :, None] & valid[:, None, :]
        eye = torch.eye(p, device=points.device, dtype=torch.bool).unsqueeze(0)
        diff = centered[:, :, None, :] - centered[:, None, :, :]  # [B, P, P, 3]
        dist_sq = diff.square().sum(dim=-1).clamp_min(1e-12)
        dist = dist_sq.sqrt().masked_fill(~pair_mask | eye, 0.0)  # [B, P, P]

        # Statistiques locales invariantes par point sur les voisins valides.
        neighbor_mask = pair_mask & ~eye
        neighbor_count = neighbor_mask.float().sum(dim=-1).clamp_min(1.0)  # [B, P]
        mean_dist = dist.sum(dim=-1) / neighbor_count
        max_dist = dist.amax(dim=-1)
        min_dist = dist.masked_fill(~neighbor_mask, float("inf")).amin(dim=-1)
        min_dist = torch.where(neighbor_mask.any(dim=-1), min_dist, torch.zeros_like(min_dist))
        var_dist = (((dist - mean_dist.unsqueeze(-1)).square() * neighbor_mask.float()).sum(dim=-1)
                    / neighbor_count)
        std_dist = var_dist.clamp_min(1e-12).sqrt() * neighbor_mask.any(dim=-1).float()

        # Expansion en base radiale gaussienne sur log(1 + d_ij) et angle relatif au centroïde.
        log_dist = torch.log1p(dist)
        centers = torch.linspace(0.0, 3.0, n_rbf, device=points.device, dtype=torch.float32)
        width = 3.0 / max(n_rbf - 1, 1)
        rbf = torch.exp(-((log_dist.unsqueeze(-1) - centers) / width).square())
        unit = centered / radius.unsqueeze(-1).clamp_min(1e-6)
        cos_angle = torch.einsum("bid,bjd->bij", unit, unit).clamp(-1.0, 1.0)
        pair_features = torch.cat((rbf, cos_angle.unsqueeze(-1)), dim=-1) * pair_mask.unsqueeze(-1).float()

        log_radius = torch.log1p(radius)
        node_rbf = torch.exp(-((log_radius.unsqueeze(-1) - centers) / width).square())
        node_stats = torch.stack((
            log_radius,
            torch.log1p(mean_dist),
            torch.log1p(min_dist),
            torch.log1p(max_dist),
            torch.log1p(std_dist),
        ), dim=-1)
        node_geo = torch.cat((node_stats, node_rbf), dim=-1) * vf.unsqueeze(-1)
        return node_geo, pair_features


class PointCloudEncoder(nn.Module):
    """Encodeur de nuages de points 3D [B, P, 3] (+ attributs optionnels) -> [B, P, D]."""

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.n_rbf = 8
        self.feature_dim = config.pointcloud_features
        self.edge_dim = max(config.n_heads * 2, d // 4)
        node_in_dim = 5 + self.n_rbf + self.feature_dim
        self.node_proj = nn.Linear(node_in_dim, d)
        self.pair_proj = nn.Linear(self.n_rbf + 1, self.edge_dim)
        self.blocks = nn.ModuleList([GraphTransformerBlock(config) for _ in range(config.pointcloud_layers)])
        self.norm = RMSNorm(d, config.norm_eps)

    def forward(self, points: torch.Tensor, valid: torch.Tensor, point_features: torch.Tensor | None = None):
        pair_mask = valid[:, :, None] & valid[:, None, :]
        node_geo, pair_geo = se3_invariant_geometry(points, valid, n_rbf=self.n_rbf)
        if self.feature_dim > 0:
            if point_features is None:
                extra = points.new_zeros((points.size(0), points.size(1), self.feature_dim))
            else:
                extra = point_features
            node_in = torch.cat((node_geo.to(points.dtype), extra.to(points.dtype)), dim=-1)
        else:
            node_in = node_geo.to(points.dtype)
        h = self.node_proj(node_in.to(self.node_proj.weight.dtype)) * valid.unsqueeze(-1)
        e = self.pair_proj(pair_geo.to(self.pair_proj.weight.dtype)) * pair_mask.unsqueeze(-1)
        for block in self.blocks:
            h, e = block(h, e, valid)
        return self.norm(h) * valid.unsqueeze(-1), valid
