"""Quantification INT8 symétrique par canal pour réduire la mémoire en inférence."""

import torch
from torch import nn
from torch.nn import functional as F

# Sous-modules accédant directement à .weight ou liés à l'embedding.
DIRECT_WEIGHT_LEAF_NAMES = frozenset({
    "lm_head", "router", "depth_router", "ponder_halt", "pre_proj", "post_proj", "res_proj",
    "read_k", "read_v", "forecast_skip", "statistics_proj", "proj", "in_proj",
    "node_proj", "edge_proj", "pair_proj",
})


class Int8Linear(nn.Module):
    """Projection linéaire à poids INT8 symétrique par canal de sortie."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones((out_features, 1), dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        module = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        if linear.weight.is_meta:
            return module.to("meta")
        with torch.no_grad():
            weight = linear.weight.detach().float()
            scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
            quantized = (weight / scale).round().clamp(-127, 127).to(torch.int8)
            module.weight_int8.copy_(quantized.to(linear.weight.device))
            module.weight_scale.copy_(scale.to(linear.weight.device))
            if linear.bias is not None:
                module.bias.copy_(linear.bias.detach().float().to(linear.weight.device))
        return module

    def dequantized_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.weight_int8.to(dtype) * self.weight_scale.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantized_weight(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, w, b)


def quantize_model_int8(model: nn.Module, *, min_numel: int = 1024) -> dict:
    """Remplacer les grandes couches nn.Linear compatibles par Int8Linear en place."""
    if type(min_numel) is not int or min_numel < 1:
        raise ValueError("min_numel doit être un entier strictement positif")
    candidates = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if leaf in DIRECT_WEIGHT_LEAF_NAMES or module.weight.numel() < min_numel:
            continue
        candidates.append((full_name, module))

    bytes_before = bytes_after = 0
    max_error = 0.0
    for full_name, module in candidates:
        q_mod = Int8Linear.from_linear(module)
        if not module.weight.is_meta:
            err = (module.weight.detach().float() - q_mod.dequantized_weight(torch.float32)).abs().max().item()
            max_error = max(max_error, err)
        bytes_before += module.weight.numel() * module.weight.element_size()
        bytes_after += (q_mod.weight_int8.numel() * q_mod.weight_int8.element_size()
                        + q_mod.weight_scale.numel() * q_mod.weight_scale.element_size())
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, q_mod)

    return {
        "quantized_modules": len(candidates),
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "compression_ratio": round(bytes_before / max(bytes_after, 1), 3),
        "max_abs_weight_error": round(max_error, 6),
    }


class Int4Linear(nn.Module):
    """Projection linéaire à poids 4-bit symétriques packés (2 poids/octet) par groupes."""

    def __init__(self, in_features: int, out_features: int, *, group_size: int = 32, bias: bool = False):
        super().__init__()
        if type(group_size) is not int or group_size < 2 or group_size % 2 != 0:
            raise ValueError("group_size doit être un entier pair >= 2")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.padded_in = ((in_features + group_size - 1) // group_size) * group_size
        self.n_groups = self.padded_in // group_size
        self.register_buffer("weight_packed", torch.zeros((out_features, self.padded_in // 2), dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.ones((out_features, self.n_groups, 1), dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, group_size: int = 32) -> "Int4Linear":
        module = cls(linear.in_features, linear.out_features, group_size=group_size, bias=linear.bias is not None)
        if linear.weight.is_meta:
            return module.to("meta")
        with torch.no_grad():
            weight = linear.weight.detach().float()
            pad = module.padded_in - module.in_features
            if pad:
                weight = F.pad(weight, (0, pad))
            grouped = weight.view(module.out_features, module.n_groups, group_size)
            scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
            scale_fp16 = scale.to(torch.float16)
            q_signed = (grouped / scale_fp16.float()).round().clamp(-7, 7).to(torch.int16)
            q_u4 = (q_signed + 8).to(torch.uint8).reshape(module.out_features, module.padded_in)
            packed = (q_u4[:, 0::2] & 0x0F) | ((q_u4[:, 1::2] & 0x0F) << 4)
            module.weight_packed.copy_(packed.to(linear.weight.device))
            module.weight_scale.copy_(scale_fp16.to(linear.weight.device))
            if linear.bias is not None:
                module.bias.copy_(linear.bias.detach().float().to(linear.weight.device))
        return module

    def dequantized_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        lo = (self.weight_packed & 0x0F).to(dtype) - 8.0
        hi = ((self.weight_packed >> 4) & 0x0F).to(dtype) - 8.0
        unpacked = torch.stack((lo, hi), dim=-1).reshape(self.out_features, self.n_groups, self.group_size)
        scaled = (unpacked * self.weight_scale.to(dtype)).reshape(self.out_features, self.padded_in)
        return scaled[:, :self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantized_weight(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, w, b)


def quantize_model_int4(model: nn.Module, *, min_numel: int = 1024, group_size: int = 32) -> dict:
    """Remplacer les grandes couches nn.Linear compatibles par Int4Linear packées 4-bit en place."""
    if type(min_numel) is not int or min_numel < 1:
        raise ValueError("min_numel doit être un entier strictement positif")
    if type(group_size) is not int or group_size < 2 or group_size % 2 != 0:
        raise ValueError("group_size doit être un entier pair >= 2")
    candidates = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if leaf in DIRECT_WEIGHT_LEAF_NAMES or module.weight.numel() < min_numel:
            continue
        candidates.append((full_name, module))

    bytes_before = bytes_after = 0
    max_error = 0.0
    for full_name, module in candidates:
        q_mod = Int4Linear.from_linear(module, group_size=group_size)
        if not module.weight.is_meta:
            err = (module.weight.detach().float() - q_mod.dequantized_weight(torch.float32)).abs().max().item()
            max_error = max(max_error, err)
        bytes_before += module.weight.numel() * module.weight.element_size()
        bytes_after += (q_mod.weight_packed.numel() * q_mod.weight_packed.element_size()
                        + q_mod.weight_scale.numel() * q_mod.weight_scale.element_size())
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, q_mod)

    return {
        "quantized_modules": len(candidates),
        "group_size": group_size,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "compression_ratio": round(bytes_before / max(bytes_after, 1), 3),
        "max_abs_weight_error": round(max_error, 6),
    }
