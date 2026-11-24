"""Pondération des pertes supervisées par cibles observées sur une étape."""

import torch

# Poids de configuration de chaque perte émise par Blue.forward : source unique
# pour l'objectif observé et la perte de validation (test de couverture associé).
REGULARIZER_WEIGHTS = {
    "balance": "balance_weight", "router_z": "router_z_weight", "depth_budget": "depth_budget_weight",
    "ponder_kl": "ponder_kl_weight", "latent_diversity": "latent_diversity_weight",
    "router_entropy": "moe_entropy_weight", "router_orthogonality": "moe_router_orthogonality_weight",
    "mtp_kl": "mtp_kl_weight", "lm_z": "lm_z_loss_weight", "forecast_smoothness": "forecast_smoothness_weight",
}
# Pertes de tâche pondérables par incertitude (adaptive_loss_weighting).
TASK_WEIGHTS = {
    "text": "text_weight", "classification": "classification_weight", "forecast": "forecast_weight",
    "multi_token": "mtp_weight", "flow": "flow_matching_weight", "jepa": "jepa_weight",
    "alignment": "alignment_weight",
}
# Ordre de sommation historique de la perte de validation observée.
LOSS_WEIGHTS = {name: TASK_WEIGHTS[name] for name in ("text", "classification", "forecast", "multi_token")}
LOSS_WEIGHTS.update(REGULARIZER_WEIGHTS)
LOSS_WEIGHTS.update((name, TASK_WEIGHTS[name]) for name in ("flow", "jepa", "alignment"))
# Tâches normalisées par leurs cibles observées, dans l'ordre de target_counts.
COUNTED_TASKS = ("text", "classification", "forecast")


def target_counts(batch, prediction_steps):
    """Compter texte, classes, prévisions, puis chaque horizon de texte auxiliaire."""
    device = next(iter(batch.values())).device
    counts = torch.zeros(3 + max(prediction_steps - 1, 0), dtype=torch.long, device=device)
    labels = batch.get("labels")
    if labels is not None:
        valid = batch.get("attention_mask", torch.ones_like(labels, dtype=torch.bool)).bool()
        for distance in range(1, min(prediction_steps + 1, labels.size(1))):
            index = 0 if distance == 1 else distance + 1
            counts[index] = ((labels[:, distance:] != -100) & valid[:, distance:]
                             & valid[:, :-distance]).sum()
    if "class_labels" in batch:
        counts[1] = (batch["class_labels"] != -100).sum()
    if "forecast_targets" in batch:
        counts[2] = torch.isfinite(batch["forecast_targets"]).sum()
    return counts


def observed_objective(output, local_counts, total_counts, config, *, world, accumulation, weight_task=None):
    """Reconstituer la perte ; DDP moyenne les gradients entre les rangs.

    Avec un seul micro-lot sur un seul rang, le résultat vaut exactement
    ``output.loss``. ``weight_task`` est ``Blue._weight_task_loss`` : il applique la
    pondération adaptative par incertitude (adaptive_loss_weighting) aux mêmes
    pertes que la passe avant ; son terme de régularisation est compté une
    seule fois par étape.
    """
    losses = output.losses

    def weighted(name, base, value, factor=1.0):
        if weight_task is None:
            return base * value * factor
        # base * (exp(-s) * value * factor + softplus(s) / accumulation)
        return weight_task(name, base, value * (factor * accumulation)) / accumulation

    loss = (config.balance_weight * losses["balance"] + config.router_z_weight * losses["router_z"]) / accumulation
    for name, attribute in REGULARIZER_WEIGHTS.items():
        if name in losses and name not in ("balance", "router_z"):
            loss = loss + getattr(config, attribute) * losses[name] / accumulation
    for name in ("flow", "jepa", "alignment"):
        if name in losses:
            loss = loss + weighted(name, getattr(config, TASK_WEIGHTS[name]), losses[name], 1.0 / accumulation)
    for index, name in enumerate(COUNTED_TASKS):
        if name in losses:
            factor = world * local_counts[index] / total_counts[index].clamp_min(1)
            loss = loss + weighted(name, getattr(config, TASK_WEIGHTS[name]), losses[name], factor)
    if output.auxiliary_text_losses:
        # Même pondération que la passe avant : décroissance géométrique mtp_decay
        # normalisée sur les horizons ayant au moins une cible dans l'étape.
        active = total_counts[3:] > 0
        decay = config.mtp_decay ** torch.arange(active.numel(), device=active.device, dtype=torch.float64)
        normalizer = (decay * active).sum().clamp_min(1e-8)
        combined = 0.0
        for distance, (head_loss, count) in output.auxiliary_text_losses.items():
            horizon_weight = (decay[distance - 2] / normalizer).to(head_loss.dtype)
            combined = combined + head_loss * (world * horizon_weight * count / total_counts[distance + 1].clamp_min(1))
        if "multi_token" in losses:
            loss = loss + weighted("multi_token", config.mtp_weight, combined)
        else:
            loss = loss + config.mtp_weight * combined
    return loss
