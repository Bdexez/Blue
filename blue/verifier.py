"""Vérificateur automatique d'invariances mathématiques et d'intégrité des gradients (blue verify)."""

import torch

from .config import BlueConfig
from .data import synthetic_batch
from .model import Blue


def verify_architecture(config: BlueConfig, *, seed: int = 0, device: str = "cpu") -> dict:
    """Auditer causalité, équivalence du cache, invariances de padding/géométrie et finitude des gradients."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = Blue(config).to(dev).eval()
    checks = {}

    # 1. Causalité stricte sur les logits texte.
    ids = torch.randint(4, config.vocab_size, (2, min(16, config.max_seq_len)), device=dev)
    split_pos = ids.size(1) // 2
    changed = ids.clone()
    changed[:, split_pos:] = (changed[:, split_pos:] + 7) % (config.vocab_size - 4) + 4
    with torch.no_grad():
        l1 = model(ids).logits
        l2 = model(changed).logits
    causality_diff = (l1[:, :split_pos] - l2[:, :split_pos]).abs().max().item()
    checks["causality_max_abs_diff"] = round(causality_diff, 8)
    checks["causality_passed"] = causality_diff <= 1e-4

    # 2. Équivalence exacte du cache incrémental (KV + Conv + SSM + TTT + Mémoire + Byte-NGrams).
    with torch.no_grad():
        p1 = model(ids[:, :split_pos], use_cache=True)
        p2 = model(ids[:, split_pos:], cache=p1.cache, use_cache=True)
    cache_diff = (torch.cat((p1.logits, p2.logits), dim=1) - l1).abs().max().item()
    checks["cache_max_abs_diff"] = round(cache_diff, 8)
    checks["cache_passed"] = cache_diff <= 2e-4

    # 3. Invariance au padding temporel et monotonie stricte des quantiles.
    s_len = min(12, config.max_series_len // 2)
    series = torch.randn(2, s_len, config.series_features, device=dev)
    s_mask = torch.ones(2, s_len, dtype=torch.bool, device=dev)
    s_pad = torch.cat((series, torch.randn(2, 3, config.series_features, device=dev) * 25.0), dim=1)
    m_pad = torch.cat((s_mask, torch.zeros(2, 3, dtype=torch.bool, device=dev)), dim=1)
    with torch.no_grad():
        o_s = model(series=series, series_mask=s_mask)
        o_p = model(series=s_pad, series_mask=m_pad)
    series_pad_diff = (o_s.forecast - o_p.forecast).abs().max().item()
    checks["series_padding_max_abs_diff"] = round(series_pad_diff, 8)
    checks["series_padding_passed"] = series_pad_diff <= 1e-4

    min_inc = (o_s.forecast[..., 1:] - o_s.forecast[..., :-1]).min().item()
    checks["quantile_min_increment"] = round(min_inc, 8)
    checks["quantile_monotonic_passed"] = min_inc >= -1e-7

    # 4. Invariance par permutation de graphe (si activé).
    if config.graph_enabled:
        n_g = min(6, config.max_graph_nodes)
        nodes = torch.randn(2, n_g, config.graph_node_features, device=dev)
        edges = torch.randn(2, n_g, n_g, config.graph_edge_features, device=dev)
        perm = torch.randperm(n_g, device=dev)
        with torch.no_grad():
            g1 = model(graph_nodes=nodes, graph_edges=edges).classification_logits
            g2 = model(graph_nodes=nodes[:, perm], graph_edges=edges[:, perm][:, :, perm]).classification_logits
        g_diff = (g1 - g2).abs().max().item()
        checks["graph_permutation_max_abs_diff"] = round(g_diff, 8)
        checks["graph_permutation_passed"] = g_diff <= 1e-4

    # 5. Invariance géométrique rigide SE(3) sur nuages de points 3D (si activé).
    if config.pointcloud_enabled:
        n_p = min(8, config.max_pointcloud_points)
        pts = torch.randn(2, n_p, 3, device=dev)
        pf = (torch.randn(2, n_p, config.pointcloud_features, device=dev)
              if config.pointcloud_features > 0 else None)
        q_mat, _ = torch.linalg.qr(torch.randn(3, 3, device=dev))
        if torch.linalg.det(q_mat) < 0:
            q_mat[:, 0] = -q_mat[:, 0]
        trans = torch.tensor([3.5, -2.0, 1.25], device=dev)
        pts_se3 = pts @ q_mat.T + trans
        with torch.no_grad():
            pc1 = model(points=pts, point_features=pf).classification_logits
            pc2 = model(points=pts_se3, point_features=pf).classification_logits
        se3_diff = (pc1 - pc2).abs().max().item()
        checks["pointcloud_se3_max_abs_diff"] = round(se3_diff, 8)
        checks["pointcloud_se3_passed"] = se3_diff <= 1e-4

    # 6. Finitude de la perte et des gradients sur toutes les modalités actives.
    model.train()
    batch = synthetic_batch(config, batch_size=2, seed=seed + 1, device=device)
    out = model(**batch)
    loss_finite = bool(torch.isfinite(out.loss).item())
    out.loss.backward()
    grads_with_val = [p.grad for p in model.parameters() if p.grad is not None]
    grads_finite = bool(grads_with_val) and all(bool(torch.isfinite(g).all().item()) for g in grads_with_val)
    checks["loss_finite"] = loss_finite
    checks["gradients_finite"] = grads_finite
    checks["parameters_with_gradient"] = len(grads_with_val)

    passed = all(v for k, v in checks.items() if k.endswith("_passed") or k.endswith("_finite"))
    return {"passed": passed, "checks": checks}
