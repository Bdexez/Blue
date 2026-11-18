"""Alignement contrastif entre les paires présentes sur plusieurs rangs."""

import math
import struct

import torch
from torch import distributed as dist
from torch.nn import functional as F

from .hyperbolic import poincare_pairwise_distance
from .losses import symmetric_contrastive_loss


class _GatherWithGrad(torch.autograd.Function):
    """All-gather différentiable ; le backward somme les contributions des rangs."""

    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(output, tensor.contiguous(), group=group)
        return torch.stack(output, dim=0)

    @staticmethod
    def backward(ctx, gradient):
        # Un all-reduce fonctionne également sur Gloo, qui n'offre pas toujours
        # reduce_scatter. Le groupe est séparé de celui du réducteur DDP.
        gradient = gradient.contiguous()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=ctx.group)
        return gradient[ctx.rank], None


def _rectangular_alignment(gathered, metadata, first, second, temperature, mode, *,
                           cross_ids, cross_classes, has_ids, num_modalities=3,
                           geometry="euclidean", curvature=1.0):
    """Aligner les vues ayant un partenaire selon les règles activées."""
    left_rows = (metadata[:, first] != 0).nonzero(as_tuple=True)[0]
    right_rows = (metadata[:, second] != 0).nonzero(as_tuple=True)[0]
    if left_rows.numel() == 0 or right_rows.numel() == 0:
        return None
    label_col, id_col = num_modalities, num_modalities + 1
    left_ids, right_ids = metadata[left_rows, id_col], metadata[right_rows, id_col]
    positive = (left_rows[:, None] == right_rows[None, :])
    id_matches = ((left_ids[:, None] == right_ids[None, :])
                  & (left_ids[:, None] >= 0) & (right_ids[None, :] >= 0))
    if cross_ids:
        positive |= id_matches
    if cross_classes:
        left_labels, right_labels = metadata[left_rows, label_col], metadata[right_rows, label_col]
        positive |= ((left_labels[:, None] == right_labels[None, :])
                     & (left_labels[:, None] != -100) & (right_labels[None, :] != -100))
    left_valid, right_valid = positive.any(-1), positive.any(0)
    left_rows, right_rows = left_rows[left_valid], right_rows[right_valid]
    if max(left_rows.numel(), right_rows.numel()) < 2 or not left_rows.numel() or not right_rows.numel():
        return None
    positive = positive[left_valid][:, right_valid]
    if has_ids:
        positive |= id_matches[left_valid][:, right_valid]
    if mode == "class":
        left_labels, right_labels = metadata[left_rows, label_col], metadata[right_rows, label_col]
        positive |= ((left_labels[:, None] == right_labels[None, :])
                     & (left_labels[:, None] != -100) & (right_labels[None, :] != -100))
    left, right = gathered[left_rows, first], gathered[right_rows, second]
    with torch.autocast(device_type=left.device.type, enabled=False):
        if geometry == "hyperbolic":
            scores = (-poincare_pairwise_distance(left, right, curvature) / temperature).float()
        else:
            scores = (left @ right.T / temperature).float()
        forward_weights = positive.float() / positive.sum(-1, keepdim=True)
        backward_weights = positive.T.float() / positive.T.sum(-1, keepdim=True)
        forward = -(forward_weights * F.log_softmax(scores, dim=-1)).sum(-1).mean()
        backward = -(backward_weights * F.log_softmax(scores.T, dim=-1)).sum(-1).mean()
    return (forward + backward) / 2


def distributed_alignment_loss(vectors, class_labels, temperature, mode, *, group, text_view="joint", sample_ids=None,
                               cross_rank_views=False, cross_class_views=False, num_modalities=3,
                               geometry="euclidean", curvature=1.0):
    """Calculer les paires globales tout en conservant les gradients locaux.

    Les vecteurs, déjà normalisés, sont indexés par modalité (0..num_modalities-1).
    Tous les rangs du groupe doivent appeler cette fonction dans le même ordre.
    """
    if not vectors:
        raise ValueError("Au moins une modalité connue est requise")
    sample = next(iter(vectors.values()))
    batch = sample.size(0) if sample.ndim >= 1 else 0
    dim = sample.size(1) if sample.ndim >= 2 else 0
    input_error = 0
    valid_keys = tuple(range(num_modalities))
    if any(key not in valid_keys for key in vectors):
        input_error = 5
    elif sample.ndim != 2 or not sample.is_floating_point() or batch < 1:
        input_error = 1
    elif any(value.shape != sample.shape or value.device != sample.device or value.dtype != sample.dtype
             for value in vectors.values()):
        input_error = 2
    elif class_labels is not None and (class_labels.shape != (batch,) or class_labels.dtype != torch.long
                                       or class_labels.device != sample.device):
        input_error = 3
    elif sample_ids is not None and (sample_ids.shape != (batch,) or sample_ids.dtype != torch.long
                                     or sample_ids.device != sample.device or (sample_ids < -1).any()):
        input_error = 4
    dtype_codes = {torch.float16: 1, torch.bfloat16: 2, torch.float32: 3, torch.float64: 4}
    dtype_code = dtype_codes.get(sample.dtype, 0)
    mode_code = {"paired": 0, "class": 1}.get(mode, -1) if isinstance(mode, str) else -1
    view_code = {"joint": 0, "isolated": 1}.get(text_view, -1) if isinstance(text_view, str) else -1
    try:
        valid_temperature = bool(math.isfinite(temperature) and temperature > 0)
        temperature_bits = struct.unpack("!q", struct.pack("!d", float(temperature)))[0] if valid_temperature else 0
    except (TypeError, ValueError, OverflowError, RuntimeError):
        valid_temperature, temperature_bits = False, 0
    world = dist.get_world_size(group)
    cross_rank_code = int(cross_rank_views) if type(cross_rank_views) is bool else -1
    cross_class_code = int(cross_class_views) if type(cross_class_views) is bool else -1
    geom_code = {"euclidean": 0, "hyperbolic": 1}.get(geometry, -1) if isinstance(geometry, str) else -1
    try:
        valid_curvature = bool(math.isfinite(curvature) and curvature > 0)
        curvature_bits = struct.unpack("!q", struct.pack("!d", float(curvature)))[0] if valid_curvature else 0
    except (TypeError, ValueError, OverflowError, RuntimeError):
        valid_curvature, curvature_bits = False, 0
    specification = torch.tensor([batch, dim, dtype_code, mode_code,
                                  view_code, temperature_bits, cross_rank_code,
                                  cross_class_code, geom_code, curvature_bits, input_error],
                                 dtype=torch.long, device=sample.device)
    specifications = [torch.empty_like(specification) for _ in range(world)]
    dist.all_gather(specifications, specification, group=group)
    # Détecter les désaccords avant les collectives dont les tailles doivent coïncider.
    rows = [value.tolist() for value in specifications]
    errors = ("", "Les représentations doivent avoir la forme [B, D]",
              "Les représentations doivent partager forme, dtype et device",
              "class_labels doit avoir la forme [B] et le même device",
              "sample_ids doit être long [B], avec -1 pour les identités inconnues",
              "Au moins une modalité connue est requise")
    for rank, row in enumerate(rows):
        if row[-1]:
            raise ValueError(f"Rang {rank} : {errors[row[-1]]}")
    if any(row[1:-1] != rows[0][1:-1] for row in rows[1:]):
        raise ValueError("Dimensions, dtype ou paramètres d'alignement différents selon les rangs")
    if not dtype_code:
        raise ValueError("Dtype de représentation non pris en charge")
    if mode_code < 0 or view_code < 0 or not valid_temperature or geom_code < 0 or not valid_curvature:
        raise ValueError("Mode, vue textuelle, géométrie ou température d'alignement invalide")
    if cross_rank_code < 0:
        raise ValueError("cross_rank_views doit être booléen")
    if cross_class_code < 0:
        raise ValueError("cross_class_views doit être booléen")
    if cross_class_views and mode != "class":
        raise ValueError("cross_class_views nécessite le mode class")
    max_batch = max(row[0] for row in rows)
    zeros = torch.zeros_like(sample)
    packed = torch.stack([vectors.get(index, zeros) for index in range(num_modalities)], dim=1)
    if max_batch > batch:
        packed = F.pad(packed, (0, 0, 0, 0, 0, max_batch - batch))
    # Même un rang dont les paramètres sont gelés doit participer au backward.
    if torch.is_grad_enabled() and not packed.requires_grad:
        packed = packed.detach().requires_grad_(True)
    gathered = _GatherWithGrad.apply(packed, group).reshape(world * max_batch, num_modalities, dim)
    metadata = torch.zeros(batch, num_modalities + 2, dtype=torch.long, device=sample.device)
    for index in vectors:
        metadata[:, index] = 1
    label_col, id_col = num_modalities, num_modalities + 1
    metadata[:, label_col] = class_labels if class_labels is not None else -100
    metadata[:, id_col] = sample_ids if sample_ids is not None else -1
    if max_batch > batch:
        metadata = F.pad(metadata, (0, 0, 0, max_batch - batch))
    all_metadata = [torch.empty_like(metadata) for _ in range(world)]
    dist.all_gather(all_metadata, metadata, group=group)
    all_metadata = torch.cat(all_metadata, dim=0)
    has_ids = ((all_metadata[:, id_col] >= 0) & (all_metadata[:, :num_modalities] != 0).any(-1)).any().item()
    losses = []
    for first in range(num_modalities):
        for second in range(first + 1, num_modalities):
            if (has_ids and cross_rank_views) or cross_class_views:
                loss = _rectangular_alignment(gathered, all_metadata, first, second, temperature, mode,
                                              cross_ids=has_ids and cross_rank_views,
                                              cross_classes=cross_class_views, has_ids=has_ids,
                                              num_modalities=num_modalities,
                                              geometry=geometry, curvature=curvature)
                if loss is not None:
                    losses.append(loss)
                continue
            valid = (all_metadata[:, first] != 0) & (all_metadata[:, second] != 0)
            if valid.sum().item() < 2:
                continue
            left, right = gathered[valid, first], gathered[valid, second]
            with torch.autocast(device_type=sample.device.type, enabled=False):
                if geometry == "hyperbolic":
                    similarity = -poincare_pairwise_distance(left, right, curvature) / temperature
                else:
                    similarity = left @ right.T / temperature
            labels = all_metadata[valid, label_col] if mode == "class" else None
            identities = all_metadata[valid, id_col] if has_ids else None
            losses.append(symmetric_contrastive_loss(similarity, labels, sample_ids=identities))
    # Lien explicite au gather même sans paire : tous les rangs participent
    # alors au backward de la collective avec un gradient nul.
    return torch.stack(losses).mean() + gathered.sum() * 0 if losses else gathered.sum() * 0
