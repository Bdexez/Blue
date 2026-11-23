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

    def __init__(self, d_model: int, memory_slots: int, n_heads: int, *, norm_eps: float = 1e-6,
                 memory_decay: bool = False):
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
        self.decay_gate = nn.Linear(d_model, memory_slots, bias=True) if memory_decay else None
        self.surprise_scale = nn.Parameter(torch.tensor(0.5))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.read_gate = nn.Parameter(torch.full((d_model,), -1.0))
        self.reset_memory_parameters()

    def reset_memory_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.init_slots, std=0.02)
            self.write_gate.bias.fill_(-1.0)
            if self.decay_gate is not None:
                self.decay_gate.weight.zero_()
                self.decay_gate.bias.fill_(4.0)
            self.surprise_scale.fill_(0.5)
            self.read_gate.fill_(-1.0)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, memory: torch.Tensor | None = None):
        b, t, d = x.shape
        h = self.in_norm(x)
        q_all = self.read_q(h).view(b, t, self.n_heads, self.head_dim)
        val_all = self.write_val(h)
        addr_all = self.write_addr(h)
        gate_all = self.write_gate(h)
        decay_all = self.decay_gate(h) if self.decay_gate is not None else None
        scale_inv = 1.0 / math.sqrt(self.head_dim)
        dim_inv = 1.0 / math.sqrt(d)

        with torch.autocast(device_type=x.device.type, enabled=False):
            q_f = F.normalize(q_all.float(), p=2, dim=-1, eps=1e-6)
            val_f = val_all.float()
            addr_f = addr_all.float()
            gate_f = gate_all.float()
            decay_f = torch.sigmoid(decay_all.float()) if decay_all is not None else None
            read_w = torch.cat((self.read_k.weight, self.read_v.weight)).float()  # [2D, D]
            mn_w = self.mem_norm.weight.float()
            mn_eps = self.mem_norm.eps
            s_scale = self.surprise_scale.float()
            init_f = self.init_slots.float()

            if memory is None:
                mem = init_f.expand(b, -1, -1).contiguous()
            else:
                mem = memory.float()

            # unbind : un seul nœud arrière, au lieu d'un tenseur complet par pas indexé.
            q_steps = (q_f * scale_inv).unsqueeze(-1).unbind(1)  # [B, H, Dh, 1]
            val_steps = val_f.unsqueeze(-2).unbind(1)  # [B, 1, D]
            addr_steps = torch.softmax(addr_f, dim=-1).unbind(1)  # [B, M]
            gate_steps = gate_f.unbind(1)
            decay_steps = decay_f.unsqueeze(-1).unbind(1) if decay_f is not None else None
            dense = bool(valid.all())
            mask_steps = None if dense else valid[:, :, None, None].unbind(1)

            reads = []
            for step in range(t):
                # Normalisation RMS FP32 des emplacements mémoire.
                mem_n = mem * torch.rsqrt(mem.square().mean(-1, keepdim=True) + mn_eps) * mn_w
                projected = F.linear(mem_n, read_w).view(b, self.memory_slots, 2, self.n_heads, self.head_dim)
                k_m, v_m = projected.permute(2, 0, 3, 1, 4).unbind(0)  # [B, H, M, Dh]
                k_m = F.normalize(k_m, p=2, dim=-1, eps=1e-6)
                # Lecture multi-têtes : [B, H, M]
                attn = torch.softmax(torch.matmul(k_m, q_steps[step]).squeeze(-1), dim=-1)
                r_t = torch.matmul(attn.unsqueeze(-2), v_m).reshape(b, 1, d)

                # Surprise et écriture convexe modulée (avec relaxation vers init_slots si activée).
                v_t = val_steps[step]
                surprise = torch.log1p((v_t - r_t).square().sum(dim=-1).clamp_min(1e-12).sqrt() * dim_inv)
                eta = torch.sigmoid(gate_steps[step] + s_scale * surprise)  # [B, M]
                blend = (addr_steps[step] * eta).unsqueeze(-1)  # [B, M, 1]
                if decay_steps is not None:
                    mem_base = torch.lerp(init_f, mem, decay_steps[step])
                else:
                    mem_base = mem
                mem_next = torch.lerp(mem_base, v_t, blend)
                mem = mem_next if dense else torch.where(mask_steps[step], mem_next, mem)
                reads.append(r_t)
            read_out = torch.cat(reads, dim=1)  # [B, T, D]
            if not dense:
                read_out = torch.where(valid.unsqueeze(-1), read_out, torch.zeros_like(read_out))
        out = x + torch.sigmoid(self.read_gate) * self.out_proj(read_out.to(x.dtype))
        return out * valid.unsqueeze(-1), mem
