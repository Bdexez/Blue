"""Traitement par lots de séquences de longueurs valides différentes.

Les mélangeurs spectraux et en ondelettes opèrent sur les seules positions
valides de chaque exemple. Plutôt qu'une boucle Python par exemple, les
exemples de même longueur valide sont traités ensemble : le calcul reste
exactement celui de chaque ligne prise isolément, avec une seule
synchronisation pour lire les longueurs.
"""

import torch


def _grouped(values, valid, function, positional):
    batch, length = valid.shape
    groups: dict[int, list[int]] = {}
    for row, count in enumerate(valid.sum(1).tolist()):
        groups.setdefault(count, []).append(row)
    outputs, order = [], []
    for count, rows in groups.items():
        if len(groups) == 1:
            # Toutes les lignes ont la même longueur : aucune permutation.
            subset, subset_valid = values, valid
        else:
            index = torch.tensor(rows, device=values.device)
            subset, subset_valid = values.index_select(0, index), valid.index_select(0, index)
            order.append(index)
        if count == length:
            packed = subset
        else:
            packed = subset[subset_valid].view(len(rows), count, *values.shape[2:])
        result = function(packed)
        if positional and count != length:
            mask = subset_valid.view(len(rows), length, *([1] * (result.ndim - 2))).expand(
                len(rows), length, *result.shape[2:])
            result = result.new_zeros(mask.shape).masked_scatter(mask, result)
        outputs.append(result)
    if len(outputs) == 1:
        return outputs[0]
    inverse = torch.empty(batch, dtype=torch.long, device=values.device)
    inverse[torch.cat(order)] = torch.arange(batch, device=values.device)
    return torch.cat(outputs).index_select(0, inverse)


def map_valid_positions(values: torch.Tensor, valid: torch.Tensor, function) -> torch.Tensor:
    """Appliquer ``function`` aux positions valides compactées de chaque ligne.

    ``values`` [B, T, *], ``valid`` [B, T] booléen. ``function`` reçoit un tenseur
    [G, L, *] (G lignes ayant L positions valides, dans leur ordre d'origine) et
    renvoie [G, L, *sortie]. Le résultat [B, T, *sortie] vaut zéro hors des
    positions valides.
    """
    return _grouped(values, valid, function, positional=True)


def map_valid_rows(values: torch.Tensor, valid: torch.Tensor, function) -> torch.Tensor:
    """Comme map_valid_positions, mais ``function`` renvoie un résultat par ligne [G, *sortie]."""
    return _grouped(values, valid, function, positional=False)
