"""Géométrie hyperbolique sur la boule de Poincaré pour l'alignement multimodal."""

import math
import torch


def poincare_exp0(tangent: torch.Tensor, curvature: float = 1.0, *, eps: float = 1e-4) -> torch.Tensor:
    """Carte exponentielle à l'origine exp_0^c : T_0 B_c^d -> B_c^d en FP32.

    La norme euclidienne est bornée strictement sous (1 - eps) / sqrt(c) pour
    éviter toute singularité de distance ou de gradient au bord de la boule.
    """
    if not math.isfinite(curvature) or curvature <= 0:
        raise ValueError("curvature doit être fini et strictement positif")
    with torch.autocast(device_type=tangent.device.type, enabled=False):
        v = tangent.float()
        sqrt_c = math.sqrt(curvature)
        # Normaliser l'échelle tangente pour éviter la saturation immédiate de tanh en grande dimension.
        v_norm = v.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        scaled_norm = (sqrt_c * v_norm / math.sqrt(max(v.size(-1), 1))).clamp_max(15.0)
        radius = torch.tanh(scaled_norm) / sqrt_c
        max_radius = (1.0 - eps) / sqrt_c
        radius = radius.clamp_max(max_radius)
        return v * (radius / v_norm)


def poincare_pairwise_distance(left: torch.Tensor, right: torch.Tensor, curvature: float = 1.0,
                               *, eps: float = 1e-6) -> torch.Tensor:
    """Distance géodésique pair-à-pair [N, M] sur la boule de Poincaré B_c^d en FP32.

    Utilise l'identité isométrique exacte en mémoire O(N M) :
        d_c(u, v) = arcosh(1 + 2 c ||u - v||^2 / ((1 - c||u||^2)(1 - c||v||^2))) / sqrt(c).
    """
    if not math.isfinite(curvature) or curvature <= 0:
        raise ValueError("curvature doit être fini et strictement positif")
    with torch.autocast(device_type=left.device.type, enabled=False):
        u = left.float()
        v = right.float()
        sqrt_c = math.sqrt(curvature)
        u_sq = u.square().sum(dim=-1, keepdim=True)  # [N, 1]
        v_sq = v.square().sum(dim=-1, keepdim=True)  # [M, 1]
        # Éviter l'annulation catastrophique de ||u||^2 + ||v||^2 - 2<u, v> sur la diagonale.
        diff_sq = torch.cdist(u, v, p=2.0, compute_mode="donot_use_mm_for_euclid_dist").square()
        denom = (1.0 - curvature * u_sq).clamp_min(eps) * (1.0 - curvature * v_sq.T).clamp_min(eps)
        delta = (2.0 * curvature * diff_sq / denom).clamp_min(0.0)
        # arcosh(1 + delta) = log(1 + delta + sqrt(delta * (delta + 2))) avec dérivée finie en delta=0.
        arcosh = torch.log1p(delta + torch.sqrt(delta * (delta + 2.0) + 1e-12))
        return arcosh / sqrt_c
