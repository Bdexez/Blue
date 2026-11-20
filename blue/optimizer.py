"""Optimiseur hybride Muon (Newton-Schulz) + AdamW compatible reprise exacte."""

import math
import torch
from torch.optim import Optimizer


def newton_schulz_orthogonalize(matrix: torch.Tensor, *, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approcher le facteur polaire U V^T d'une matrice 2D par itération quintique.

    Coefficients quintiques (Jordan et al., 2024) garantissant la convergence rapide
    des valeurs singulières vers 1 en FP32, quel que soit le ratio m/n.
    """
    if matrix.ndim != 2:
        raise ValueError("newton_schulz_orthogonalize attend une matrice 2D")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    norm = x.norm()
    if norm <= eps:
        return torch.zeros_like(matrix)
    x = x / (norm + eps)
    for _ in range(steps):
        gram = x @ x.T
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


class MuonAdamW(Optimizer):
    """Muon pour les matrices cachées 2D et AdamW découplé pour les autres paramètres.

    Chaque groupe indique ``use_muon=True`` ou ``False`` ainsi que ``lr_scale``.
    L'interface standard ``Optimizer`` garantit la compatibilité avec ``GradScaler``
    et la sérialisation exacte des checkpoints.
    """

    def __init__(self, param_groups, *, lr: float = 3e-4, weight_decay: float = 0.01,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr doit être fini et > 0")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay doit être fini et >= 0")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum doit appartenir à [0, 1[")
        if type(ns_steps) is not int or ns_steps < 1:
            raise ValueError("ns_steps doit être un entier >= 1")
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, betas=betas, eps=eps, use_muon=False, lr_scale=1.0,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            use_muon = group["use_muon"]
            if use_muon:
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if p.ndim != 2:
                        raise ValueError("Les groupes Muon ne doivent contenir que des matrices 2D")
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["step"] += 1
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    update = grad.add(buf, alpha=momentum) if nesterov else buf
                    ortho = newton_schulz_orthogonalize(update, steps=ns_steps)
                    # Mise à l'échelle RMS spectrale (Moonlight / Kimi-K2) pour aligner la norme avec AdamW.
                    rms_scale = 0.2 * math.sqrt(max(p.size(0), p.size(1)))
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.add_(ortho, alpha=-lr * rms_scale)
            else:
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    step = state["step"]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    bias_c1 = 1.0 - beta1 ** step
                    bias_c2 = 1.0 - beta2 ** step
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.addcdiv_(exp_avg, denom, value=-lr / bias_c1)
        return loss


def build_optimizer(model, *, name: str = "adamw", lr: float, weight_decay: float):
    """Construire AdamW ou MuonAdamW en préservant lr_scale=10 pour les projections temporelles."""
    if name not in ("adamw", "muon"):
        raise ValueError("optimizer doit être adamw ou muon")
    temporal_parameters = [p for p in model.temporal.forecast_skip.parameters() if p.requires_grad]
    if model.temporal.statistics_proj is not None:
        temporal_parameters.extend(p for p in model.temporal.statistics_proj.parameters() if p.requires_grad)
    temporal_ids = {id(p) for p in temporal_parameters}
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("Aucun paramètre entraînable dans le modèle")
    if name == "adamw":
        groups = [
            {"params": [p for p in trainable if id(p) not in temporal_ids], "lr_scale": 1.0},
            {"params": temporal_parameters, "lr_scale": 10.0},
        ]
        return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, weight_decay=weight_decay)
    excluded_muon_ids = set(temporal_ids)
    excluded_muon_ids.add(id(model.embedding.weight))
    excluded_muon_ids.add(id(model.lm_head.weight))
    if model.reasoning_step_embed is not None:
        excluded_muon_ids.add(id(model.reasoning_step_embed.weight))
    if getattr(model, "ngram", None) is not None:
        excluded_muon_ids.add(id(model.ngram.table.weight))
    if getattr(model, "tabular", None) is not None and model.tabular.cat_embed is not None:
        excluded_muon_ids.add(id(model.tabular.cat_embed.weight))
    if getattr(model, "jepa", None) is not None:
        excluded_muon_ids.add(id(model.jepa.target_embed.weight))
    if getattr(model, "flow_head", None) is not None and model.flow_head.null_cond is not None:
        excluded_muon_ids.add(id(model.flow_head.null_cond))
    muon_params, adamw_params = [], []
    for p in trainable:
        if id(p) in temporal_ids:
            continue
        if p.ndim == 2 and id(p) not in excluded_muon_ids:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    groups = [
        {"params": muon_params, "lr_scale": 1.0, "use_muon": True},
        {"params": adamw_params, "lr_scale": 1.0, "use_muon": False},
        {"params": temporal_parameters, "lr_scale": 10.0, "use_muon": False},
    ]
    return MuonAdamW([g for g in groups if g["params"]], lr=lr, weight_decay=weight_decay)
