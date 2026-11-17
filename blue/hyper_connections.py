"""Manifold-Constrained Hyper-Connections (mHC) sur le polytope de Birkhoff."""

import torch
from torch import nn
from torch.nn import functional as F


def sinkhorn_knopp(logits: torch.Tensor, *, iterations: int = 20) -> torch.Tensor:
    """Projeter une matrice carrée [..., N, N] sur les matrices doublement stochastiques.

    Le calcul s'effectue en domaine logarithmique FP32 (stabilité en BF16/FP16 sans
    débordement ni sous-flux). Par le théorème de Birkhoff-von Neumann, chaque matrice
    produite est non-expansive (norme spectrale <= 1), ce qui stabilise la propagation
    résiduelle sur les réseaux très profonds.
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        log_matrix = logits.float()
        for _ in range(iterations):
            log_matrix = log_matrix - torch.logsumexp(log_matrix, dim=-1, keepdim=True)
            log_matrix = log_matrix - torch.logsumexp(log_matrix, dim=-2, keepdim=True)
        return log_matrix.exp()


class ManifoldHyperConnection(nn.Module):
    """Couplage multi-flux résiduel contraint sur le polytope de Birkhoff (mHC)."""

    def __init__(self, d_model: int, n_streams: int, *, norm_cls, norm_eps: float = 1e-6,
                 sinkhorn_iters: int = 20):
        super().__init__()
        if n_streams < 2:
            raise ValueError("ManifoldHyperConnection nécessite n_streams >= 2")
        self.d_model = d_model
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.norm = norm_cls(d_model, norm_eps)
        flat_dim = n_streams * d_model
        self.pre_proj = nn.Linear(flat_dim, n_streams, bias=False)
        self.post_proj = nn.Linear(flat_dim, n_streams, bias=False)
        self.res_proj = nn.Linear(flat_dim, n_streams * n_streams, bias=False)
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        self.pre_bias = nn.Parameter(torch.zeros(n_streams))
        self.post_bias = nn.Parameter(torch.zeros(n_streams))
        self.res_bias = nn.Parameter(torch.zeros(n_streams, n_streams))
        self.reset_manifold_parameters()

    def reset_manifold_parameters(self):
        """Initialiser proche de l'identité résiduelle (moyenne uniforme, H_post=1, H_res≈I)."""
        with torch.no_grad():
            self.alpha_pre.fill_(0.01)
            self.alpha_post.fill_(0.01)
            self.alpha_res.fill_(0.01)
            self.pre_bias.zero_()
            self.post_bias.zero_()
            self.res_bias.zero_()
            self.res_bias.fill_diagonal_(4.0)

    def routing_weights(self, streams: torch.Tensor):
        """Calculer (H_pre, H_post, H_res) pour un état multi-flux [B, T, N, D]."""
        b, t, n, _ = streams.shape
        flat = self.norm(streams).reshape(b, t, n * self.d_model)
        with torch.autocast(device_type=streams.device.type, enabled=False):
            flat_f = flat.float()
            h_pre = torch.softmax(
                self.alpha_pre.float() * F.linear(flat_f, self.pre_proj.weight.float()) + self.pre_bias.float(), dim=-1
            )
            h_post = 2.0 * torch.sigmoid(
                self.alpha_post.float() * F.linear(flat_f, self.post_proj.weight.float()) + self.post_bias.float()
            )
            res_logits = (
                self.alpha_res.float() * F.linear(flat_f, self.res_proj.weight.float()).view(b, t, n, n)
                + self.res_bias.float()
            )
            h_res = sinkhorn_knopp(res_logits, iterations=self.sinkhorn_iters)
        return h_pre.to(streams.dtype), h_post.to(streams.dtype), h_res.to(streams.dtype)

    def combine_input(self, streams: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btnd,btn->btd", streams, h_pre)

    def update_streams(self, streams: torch.Tensor, sublayer_out: torch.Tensor,
                       h_post: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
        mixed = torch.einsum("btij,btjd->btid", h_res, streams)
        return mixed + sublayer_out.unsqueeze(-2) * h_post.unsqueeze(-1)
