"""Adaptation de rang bas DoRA / LoRA avec fusion algébrique exacte sans surcoût."""

import math
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_TARGET_SUFFIXES = (
    "q", "kv", "kv_down", "k_up", "v_up", "k_rope", "out",
    "up_gate", "down", "qk_proj", "v_proj", "gate_proj", "out_proj",
)


class DoRALinear(nn.Module):
    """Couche linéaire DoRA (Weight-Decomposed Low-Rank Adaptation) ou LoRA classique.

    Avec ``use_dora=True`` :
        W' = W_0 + (alpha / r) * B @ A
        W_eff = magnitude * W' / (||W'||_2 + eps)
    Comme B est initialisée à zéro et magnitude = ||W_0||_2, la sortie initiale
    coïncide exactement avec la couche linéaire d'origine.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 16.0, *,
                 dropout: float = 0.0, use_dora: bool = True, eps: float = 1e-6):
        super().__init__()
        if type(rank) is not int or rank < 1:
            raise ValueError("rank doit être un entier strictement positif")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha doit être fini et strictement positif")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout doit appartenir à [0, 1[")
        if type(use_dora) is not bool:
            raise ValueError("use_dora doit être booléen")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.use_dora = use_dora
        self.eps = eps

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        device, dtype = self.weight.device, self.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.use_dora:
            with torch.no_grad():
                row_norm = self.weight.float().norm(p=2, dim=1, keepdim=True).to(dtype)
            self.magnitude = nn.Parameter(row_norm)
        else:
            self.register_parameter("magnitude", None)

    def effective_weight(self) -> torch.Tensor:
        delta = (self.lora_b @ self.lora_a) * self.scale
        combined = self.weight + delta
        if not self.use_dora:
            return combined
        with torch.autocast(device_type=combined.device.type, enabled=False):
            c_f = combined.float()
            row_norm = c_f.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)
            scaled = (self.magnitude.float() / row_norm) * c_f
        return scaled.to(combined.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_dora:
            base_out = F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
            lora_out = F.linear(F.linear(self.drop(x), self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
            return base_out + lora_out * self.scale
        w_eff = self.effective_weight().to(x.dtype)
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(self.drop(x), w_eff, bias)

    def merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            self.in_features, self.out_features, bias=self.bias is not None,
            device=self.weight.device, dtype=self.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self.effective_weight())
            if self.bias is not None:
                merged.bias.copy_(self.bias)
        return merged


def inject_dora(model: nn.Module, *, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0,
                use_dora: bool = True, target_suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES,
                freeze_base: bool = True) -> dict:
    """Injecter DoRA/LoRA sur les projections ciblées en préservant le poids lié lm_head."""
    if freeze_base:
        for param in model.parameters():
            param.requires_grad_(False)
    replaced = []
    # Identifier les modules à remplacer avant de muter l'arbre.
    candidates = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or full_name == "lm_head":
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if leaf in target_suffixes:
            candidates.append((full_name, module))
    for full_name, module in candidates:
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, DoRALinear(module, rank, alpha, dropout=dropout, use_dora=use_dora))
        replaced.append(full_name)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "replaced_modules": len(replaced),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": round(trainable / max(total, 1), 6),
    }


def merge_dora(model: nn.Module) -> int:
    """Fusionner tous les modules DoRALinear en nn.Linear standards sans surcoût d'inférence."""
    to_merge = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, DoRALinear)]
    for full_name, module in to_merge:
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, module.merged_linear())
    if hasattr(model, "lm_head") and hasattr(model, "embedding"):
        model.lm_head.weight = model.embedding.weight
    return len(to_merge)
