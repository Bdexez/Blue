"""Banc de mémoire épisodique neuronale à écriture modulée par la surprise."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .modules import RMSNorm


class EpisodicMemoryBank(nn.Module):
    """Mémoire de travail persistante [B, M, D] lue puis mise à jour causalement.

    À chaque pas valide t :
      1. Lecture par attention multi-têtes sur les M emplacements M_{t-1} -> r_t.
      2. Mesure de surprise : s_t = log(1 + ||val_t - r_t||_2 / sqrt(D)).
      3. Mise à jour convexe adressée par contenu :
         blend_{t,m} = softmax_m(addr(h_t)) * sigmoid(gate(h_t) + alpha * s_t)
         M_t[m]      = (1 - blend_{t,m}) * M_{t-1}[m] + blend_{t,m} * val_t.

    La mise à jour convexe garantit des normes bornées sur les séquences longues,
    la causalité stricte, l'invariance exacte au padding et l'équivalence exacte
    du cache en génération incrémentale.
    """

    def __init__(self, d_model: int, memory_slots: int, n_heads: int, *, norm_eps: float = 1e-6):
        super().__init__()
        if memory_slots < 1:
            raise ValueError("memory_slots doit être un entier strictement positif")
        if d_model % n_heads:
            raise ValueError("d_model doit être divisible par n_heads")
        self.d_model = d_model
        self.memory_slots = memory_slots
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.init_slots = nn.Parameter(torch.empty(1, memory_slots, d_model))
        self.in_norm = RMSNorm(d_model, norm_eps)
        self.mem_norm = RMSNorm(d_model, norm_eps)
        self.read_q = nn.Linear(d_model, d_model, bias=False)
        self.read_k = nn.Linear(d_model, d_model, bias=False)
        self.read_v = nn.Linear(d_model, d_model, bias=False)
        self.write_val = nn.Linear(d_model, d_model, bias=False)
        self.write_addr = nn.Linear(d_model, memory_slots, bias=False)
        self.write_gate = nn.Linear(d_model, memory_slots, bias=True)
        self.surprise_scale = nn.Parameter(torch.tensor(0.5))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.read_gate = nn.Parameter(torch.full((d_model,), -1.0))
        self.reset_memory_parameters()

    def reset_memory_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.init_slots, std=0.02)
            self.write_gate.bias.fill_(-1.0)
            self.surprise_scale.fill_(0.5)
            self.read_gate.fill_(-1.0)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, memory: torch.Tensor | None = None):
        b, t, d = x.shape
        h = self.in_norm(x)
        q_all = self.read_q(h).view(b, t, self.n_heads, self.head_dim)
        val_all = self.write_val(h)
        addr_all = self.write_addr(h)
        gate_all = self.write_gate(h)
        scale_inv = 1.0 / math.sqrt(self.head_dim)
        dim_inv = 1.0 / math.sqrt(d)

        with torch.autocast(device_type=x.device.type, enabled=False):
            q_f = F.normalize(q_all.float(), p=2, dim=-1, eps=1e-6)
            val_f = val_all.float()
            addr_f = addr_all.float()
            gate_f = gate_all.float()
            rk_w = self.read_k.weight.float()
            rv_w = self.read_v.weight.float()
            mn_w = self.mem_norm.weight.float()
            mn_eps = self.mem_norm.eps
            s_scale = self.surprise_scale.float()

            if memory is None:
                mem = self.init_slots.float().expand(b, -1, -1).contiguous()
            else:
                mem = memory.float()

            reads = []
            for step in range(t):
                # Normalisation RMS FP32 des emplacements mémoire.
                mem_n = mem * torch.rsqrt(mem.square().mean(-1, keepdim=True) + mn_eps) * mn_w
                k_m = F.linear(mem_n, rk_w).view(b, self.memory_slots, self.n_heads, self.head_dim)
                k_m = F.normalize(k_m, p=2, dim=-1, eps=1e-6)
                v_m = F.linear(mem_n, rv_w).view(b, self.memory_slots, self.n_heads, self.head_dim)
                # Lecture multi-têtes : [B, H, M]
                q_t = q_f[:, step]  # [B, H, Dh]
                logits = torch.einsum("bhd,bmhd->bhm", q_t, k_m) * scale_inv
                attn = torch.softmax(logits, dim=-1)
                r_t = torch.einsum("bhm,bmhd->bhd", attn, v_m).reshape(b, d)

                # Surprise et écriture convexe modulée.
                v_t = val_f[:, step]
                surprise = torch.log1p((v_t - r_t).square().sum(dim=-1).clamp_min(1e-12).sqrt() * dim_inv)
                slot_addr = torch.softmax(addr_f[:, step], dim=-1)  # [B, M]
                eta = torch.sigmoid(gate_f[:, step] + s_scale * surprise.unsqueeze(-1))  # [B, M]
                blend = (slot_addr * eta).unsqueeze(-1)  # [B, M, 1]
                mem_next = (1.0 - blend) * mem + blend * v_t.unsqueeze(1)
                mask = valid[:, step]
                mem = torch.where(mask.view(b, 1, 1), mem_next, mem)
                reads.append(torch.where(mask.unsqueeze(-1), r_t, torch.zeros_like(r_t)))
            read_out = torch.stack(reads, dim=1)  # [B, T, D]
        out = x + torch.sigmoid(self.read_gate) * self.out_proj(read_out.to(x.dtype))
        return out * valid.unsqueeze(-1), mem
