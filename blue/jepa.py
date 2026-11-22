"""Prédicteur de modèle du monde latent Multimodal-JEPA (Joint-Embedding Predictive Architecture)."""

import torch
from torch import nn
from torch.nn import functional as F

from .modules import RMSNorm


class MultimodalJEPAPredictor(nn.Module):
    """Prédire chaque modalité cible à partir du contexte des autres modalités en espace latent.

    Pour chaque modalité présente j parmi M >= 2 modalités :
      - Contexte leave-one-out c_{-j} = moyenne des représentations des M - 1 autres modalités.
      - Prédiction y_hat_j = Predictor(RMSNorm(c_{-j}), target_embed[j]).
      - Perte : MSE cosinus vers sg(normalize(y_j)) + régularisation anti-effondrement
        de variance (VICReg) sur le lot lorsque B > 1.
    """

    def __init__(self, d_model: int, *, num_modalities: int = 8, norm_eps: float = 1e-6,
                 variance_weight: float = 0.1, covariance_weight: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        hidden = max(d_model, min(2 * d_model, 2048))
        self.target_embed = nn.Embedding(num_modalities, d_model)
        self.context_norm = RMSNorm(d_model, norm_eps)
        self.in_proj = nn.Linear(2 * d_model, hidden, bias=False)
        self.mid_norm = RMSNorm(hidden, norm_eps)
        self.out_proj = nn.Linear(hidden, d_model, bias=False)
        self.reset_jepa_parameters()

    def reset_jepa_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.target_embed.weight, std=0.02)

    def predict(self, context: torch.Tensor, target_modality_id: int) -> torch.Tensor:
        query = self.target_embed.weight[target_modality_id].view(1, -1).expand_as(context)
        fused = torch.cat((self.context_norm(context), query.to(context.dtype)), dim=-1)
        h = self.mid_norm(F.silu(self.in_proj(fused)))
        return self.out_proj(h)

    def forward(self, modality_vectors: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        if len(modality_vectors) < 2:
            raise ValueError("MultimodalJEPAPredictor nécessite au moins 2 modalités présentes")
        stacked = torch.stack([v for _, v in modality_vectors], dim=0)  # [M, B, D]
        total_sum = stacked.sum(dim=0)  # [B, D]
        denom = float(len(modality_vectors) - 1)
        losses = []
        for idx, (mod_id, target_vec) in enumerate(modality_vectors):
            context = (total_sum - stacked[idx]) / denom
            pred = self.predict(context, mod_id)
            with torch.autocast(device_type=pred.device.type, enabled=False):
                pred_f = pred.float()
                pred_norm = F.normalize(pred_f, p=2, dim=-1, eps=1e-6)
                targ_norm = F.normalize(target_vec.detach().float(), p=2, dim=-1, eps=1e-6)
                mse = (pred_norm - targ_norm).square().sum(dim=-1).mean()
                loss_j = mse
                if pred_f.size(0) > 1 and self.variance_weight > 0.0:
                    std = torch.sqrt(pred_f.var(dim=0, unbiased=False) + 1e-4)
                    var_reg = F.relu(1.0 - std).mean()
                    loss_j = loss_j + self.variance_weight * var_reg
                if pred_f.size(0) > 1 and self.covariance_weight > 0.0:
                    # Identité exacte de trace en O(B^2 D) évitant la matrice [D, D] :
                    # sum_{i!=j} C_{ij}^2 = ||Z_c Z_c^T||_F^2 / (B-1)^2 - sum_i C_{ii}^2.
                    z_c = pred_f - pred_f.mean(dim=0, keepdim=True)
                    denom_b = float(pred_f.size(0) - 1)
                    gram_b = torch.matmul(z_c, z_c.t()) / denom_b
                    diag_c = z_c.square().sum(dim=0) / denom_b
                    off_diag_sq = (gram_b.square().sum() - diag_c.square().sum()).clamp_min(0.0)
                    cov_reg = off_diag_sq / float(pred_f.size(-1))
                    loss_j = loss_j + self.covariance_weight * cov_reg
            losses.append(loss_j)
        return torch.stack(losses).mean()
