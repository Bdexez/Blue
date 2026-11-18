"""Encodeur de graphes attribués avec RWSE et Graph Transformer à mise à jour d'arêtes."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .modules import RMSNorm, SwiGLU


def random_walk_structural_encoding(edges: torch.Tensor, valid: torch.Tensor, steps: int) -> torch.Tensor:
    """Calculer les probabilités de retour en 1..K pas (RWSE) et le log-degré par nœud.

    L'encodage est strictement équivariant par permutation des nœuds et strictement
    invariant au padding (les nœuds masqués ont un degré nul et n'affectent pas les marches).
    """
    b, n, _, _ = edges.shape
    with torch.autocast(device_type=edges.device.type, enabled=False):
        pair_mask = valid[:, :, None] & valid[:, None, :]
        eye = torch.eye(n, device=edges.device, dtype=torch.bool).unsqueeze(0)
        weights = edges.float().abs().sum(dim=-1).masked_fill(~pair_mask | eye, 0.0)
        sym = 0.5 * (weights + weights.transpose(1, 2))
        degree = sym.sum(dim=-1)  # [B, N]
        transition = sym / degree.unsqueeze(-1).clamp_min(1e-8)
        transition = transition.masked_fill(~pair_mask, 0.0)
        diagonals = []
        power = transition
        for _ in range(steps):
            diagonals.append(power.diagonal(dim1=1, dim2=2))
            power = torch.bmm(power, transition)
        log_degree = torch.log1p(degree) * valid.float()
        return torch.cat((*([d.unsqueeze(-1) for d in diagonals]), log_degree.unsqueeze(-1)), dim=-1)


class GraphTransformerBlock(nn.Module):
    """Bloc Graph Transformer à double piste (nœuds + arêtes), équivariant par permutation."""

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = d // config.n_heads
        self.edge_dim = max(config.n_heads * 2, d // 4)
        self.dropout = config.dropout

        self.norm1 = RMSNorm(d, config.norm_eps)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.out = nn.Linear(d, d, bias=False)

        self.edge_norm1 = RMSNorm(self.edge_dim, config.norm_eps)
        self.edge_to_bias = nn.Linear(self.edge_dim, self.n_heads, bias=False)
        self.edge_to_gate = nn.Linear(self.edge_dim, self.n_heads, bias=False)
        self.attn_to_edge = nn.Linear(self.n_heads, self.edge_dim, bias=False)
        self.edge_norm2 = RMSNorm(self.edge_dim, config.norm_eps)
        self.edge_ffn = SwiGLU(self.edge_dim, self.edge_dim * 2)

        self.norm2 = RMSNorm(d, config.norm_eps)
        self.ffn = SwiGLU(d, config.hidden_dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, valid: torch.Tensor):
        b, n, d = nodes.shape
        pair_mask = valid[:, :, None] & valid[:, None, :]  # [B, N, N]
        h = self.norm1(nodes)
        q, k, v = self.qkv(h).view(b, n, 3, self.n_heads, self.head_dim).unbind(2)
        q = self.q_norm(q.transpose(1, 2))  # [B, H, N, Dh]
        k = self.k_norm(k.transpose(1, 2))
        v = v.transpose(1, 2)

        e_norm = self.edge_norm1(edges)
        edge_bias = self.edge_to_bias(e_norm).permute(0, 3, 1, 2)  # [B, H, N, N]
        edge_gate = torch.sigmoid(self.edge_to_gate(e_norm)).permute(0, 3, 1, 2)  # [B, H, N, N]

        with torch.autocast(device_type=nodes.device.type, enabled=False):
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
            logits = logits + edge_bias.float()
            key_mask = valid[:, None, None, :]  # [B, 1, 1, N]
            masked_logits = logits.masked_fill(~key_mask, -torch.inf)
            # Les nœuds de padding restent finis pour éviter NaN dans softmax.
            safe_logits = torch.where(valid[:, None, :, None], masked_logits, torch.zeros_like(masked_logits))
            attn = torch.softmax(safe_logits, dim=-1) * edge_gate.float()
            attn = attn * pair_mask[:, None, :, :].float()
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=True)
        out = torch.matmul(attn.to(v.dtype), v).transpose(1, 2).contiguous().view(b, n, d)
        nodes = (nodes + self.drop(self.out(out))) * valid.unsqueeze(-1)
        nodes = (nodes + self.drop(self.ffn(self.norm2(nodes)))) * valid.unsqueeze(-1)

        # Mise à jour de la piste d'arêtes à partir des logits d'attention (style Evoformer).
        edge_update = self.attn_to_edge(logits.to(edges.dtype).permute(0, 2, 3, 1))
        edges = (edges + self.drop(edge_update)) * pair_mask.unsqueeze(-1)
        edges = (edges + self.drop(self.edge_ffn(self.edge_norm2(edges)))) * pair_mask.unsqueeze(-1)
        return nodes, edges


class GraphEncoder(nn.Module):
    """Encodeur de graphes attribués [B, N, Fn] + [B, N, N, Fe] -> latents [B, N, D]."""

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.rwse_steps = config.graph_rwse_steps
        self.edge_dim = max(config.n_heads * 2, d // 4)
        self.node_proj = nn.Linear(config.graph_node_features + self.rwse_steps + 1, d)
        self.edge_proj = nn.Linear(config.graph_edge_features, self.edge_dim)
        self.blocks = nn.ModuleList([GraphTransformerBlock(config) for _ in range(config.graph_layers)])
        self.norm = RMSNorm(d, config.norm_eps)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, valid: torch.Tensor):
        pair_mask = valid[:, :, None] & valid[:, None, :]
        rwse = random_walk_structural_encoding(edges, valid, self.rwse_steps).to(nodes.dtype)
        node_in = torch.cat((nodes, rwse), dim=-1)
        h = self.node_proj(node_in.to(self.node_proj.weight.dtype)) * valid.unsqueeze(-1)
        e = self.edge_proj(edges.to(self.edge_proj.weight.dtype)) * pair_mask.unsqueeze(-1)
        for block in self.blocks:
            h, e = block(h, e, valid)
        return self.norm(h) * valid.unsqueeze(-1), valid
