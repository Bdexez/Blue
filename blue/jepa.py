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
                 variance_weight: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.variance_weight = float(variance_weight)
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
                if pred_f.size(0) > 1 and self.variance_weight > 0.0:
                    std = torch.sqrt(pred_f.var(dim=0, unbiased=False) + 1e-4)
                    var_reg = F.relu(1.0 - std).mean()
                    loss_j = mse + self.variance_weight * var_reg
                else:
                    loss_j = mse
            losses.append(loss_j)
        return torch.stack(losses).mean()
