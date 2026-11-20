"""Diagnostic analytique et mesure de latence pour les architectures Blue."""

import time
import torch

from .config import BlueConfig
from .data import synthetic_batch
from .model import Blue


def profile_architecture(config: BlueConfig, *, seq_len: int = 1024, batch_size: int = 1,
                         measure: bool = False, device: str = "cpu") -> dict:
    """Calculer paramètres, empreinte cache KV/SSM/mémoire et latence optionnelle."""
    if type(seq_len) is not int or seq_len < 1:
        raise ValueError("seq_len doit être un entier strictement positif")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size doit être un entier strictement positif")

    with torch.device("meta"):
        meta_model = Blue(config)
    total = sum(p.numel() for p in meta_model.parameters())
    expert_size = 3 * config.d_model * config.hidden_dim
    unselected = config.n_layers * (config.n_experts - config.top_k) * expert_size
    head_dim = config.d_model // config.n_heads

    # Empreinte exacte du cache KV en BF16 (2 octets/élément), fenêtre glissante incluse.
    eff_seq = min(seq_len, config.max_seq_len)
    if config.kv_lora_rank > 0:
        rope_elems = 2 * (head_dim // 4) if config.differential_attention else (head_dim // 2)
        elems_per_pos = config.kv_lora_rank + rope_elems
    else:
        elems_per_pos = 2 * config.n_kv_heads * head_dim
    std_gqa_elems_per_pos = 2 * config.n_kv_heads * head_dim

    retained_positions = 0
    for layer_idx in range(config.n_layers):
        is_global = (layer_idx + 1) % config.global_every == 0
        retained_positions += eff_seq if is_global else min(eff_seq, max(config.sliding_window - 1, 0))
    retained_positions *= config.latent_reasoning_steps

    kv_bytes_bf16 = batch_size * retained_positions * elems_per_pos * 2
    std_kv_bytes_bf16 = batch_size * retained_positions * std_gqa_elems_per_pos * 2

    # États O(1) indépendants de la longueur de séquence.
    conv_bytes_fp32 = batch_size * config.n_layers * config.latent_reasoning_steps * config.d_model * max(config.conv_kernel - 1, 0) * 4
    ssm_heads = config.n_heads if config.ssm_heads is None else config.ssm_heads
    ssm_bytes_fp32 = (batch_size * config.n_layers * config.latent_reasoning_steps
                      * ssm_heads * config.ssm_state_dim * (config.d_model // ssm_heads) * 4
                      if config.ssm_state_dim > 0 else 0)
    ttt_heads = config.n_heads if config.ttt_heads is None else config.ttt_heads
    ttt_bytes_fp32 = (batch_size * config.n_layers * config.latent_reasoning_steps
                      * ttt_heads * (config.ttt_state_dim + 1) * config.ttt_state_dim * 4
                      if config.ttt_state_dim > 0 else 0)
    memory_bytes_fp32 = batch_size * config.memory_slots * config.d_model * 4 if config.memory_slots > 0 else 0

    modalities = ["text", "image", "series"]
    if config.audio_enabled:
        modalities.append("audio")
    if config.graph_enabled:
        modalities.append("graph")
    if config.pointcloud_enabled:
        modalities.append("pointcloud")
    if config.tabular_enabled:
        modalities.append("tabular")
    if config.video_enabled:
        modalities.append("video")
    n_mod = len(modalities)

    report = {
        "parameters": total,
        "active_parameters_upper_bound": total - unselected,
        "weights_gib_bf16": round(total * 2 / 2**30, 3),
        "weights_gib_int8_estimate": round(total / 2**30, 3),
        "decoder_layers": config.n_layers,
        "effective_decoder_layers": config.n_layers * config.latent_reasoning_steps,
        "residual_streams": config.residual_streams,
        "modalities": modalities,
        "modality_combinations": (1 << n_mod) - 1,
        "contrastive_pairs": n_mod * (n_mod - 1) // 2,
        "cache_profile": {
            "seq_len": eff_seq,
            "batch_size": batch_size,
            "kv_cache_mib_bf16": round(kv_bytes_bf16 / 2**20, 4),
            "standard_gqa_kv_mib_bf16": round(std_kv_bytes_bf16 / 2**20, 4),
            "kv_compression_ratio": round(std_kv_bytes_bf16 / max(kv_bytes_bf16, 1), 3),
            "conv_state_mib_fp32": round(conv_bytes_fp32 / 2**20, 4),
            "ssm_state_mib_fp32": round(ssm_bytes_fp32 / 2**20, 4),
            "ttt_state_mib_fp32": round(ttt_bytes_fp32 / 2**20, 4),
            "episodic_memory_mib_fp32": round(memory_bytes_fp32 / 2**20, 4),
        },
        "modules": {name: sum(p.numel() for p in module.parameters())
                    for name, module in meta_model.named_children() if name != "lm_head"},
    }

    if measure:
        model = Blue(config).to(device)
        batch = synthetic_batch(config, batch_size=batch_size, seed=0, device=device)
        t0 = time.perf_counter()
        out = model(**batch)
        t1 = time.perf_counter()
        out.loss.backward()
        t2 = time.perf_counter()
        report["runtime"] = {
            "device": str(device),
            "forward_ms": round((t1 - t0) * 1000.0, 2),
            "backward_ms": round((t2 - t1) * 1000.0, 2),
            "loss": round(out.loss.item(), 5),
        }
    return report
