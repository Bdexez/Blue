"""Générateur automatique de fiche modèle (Model Card) Markdown pour les architectures Blue."""

import math

from .config import BlueConfig
from .profiler import profile_architecture


def _fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _build_mermaid(config: BlueConfig, modalities: list[str]) -> list[str]:
    mod_nodes = [
        ("IMG", "Images (2D-RoPE)" if config.vision_rope_2d else "Images"),
        ("SER", "Séries (RevIN + rFFT)" if config.temporal_spectral_mixing else "Séries (RevIN)"),
    ]
    if config.audio_enabled:
        mod_nodes.append(("AUD", "Audio 1D"))
    if config.graph_enabled:
        mod_nodes.append(("GRP", "Graphes (RWSE + Evoformer)"))
    if config.pointcloud_enabled:
        mod_nodes.append(("PTS", "Nuages 3D SE(3)"))
    if config.tabular_enabled:
        mod_nodes.append(("TAB", "Tabulaire (PLR + Cat)"))
    if config.video_enabled:
        mod_nodes.append(("VID", "Vidéo (Tubelets + 3D-RoPE)"))

    res_label = f"LatentResampler ({config.n_latents} latents"
    if config.hierarchical_resampler:
        res_label += ", hiérarchique 2 niveaux"
    if config.latent_fsq_levels:
        res_label += f", FSQ {_fmt_int(math.prod(config.latent_fsq_levels))}"
    res_label += ")"

    attn_parts = ["MLA" if config.kv_lora_rank else "GQA"]
    if config.differential_attention:
        attn_parts.append("DiffAttn")
    if config.attention_sinks:
        attn_parts.append(f"{config.attention_sinks} Sinks")
    if config.value_residual:
        attn_parts.append("ResFormer V-Res")
    mixer_parts = [" + ".join(attn_parts), "CausalConv"]
    if config.ssm_state_dim:
        mixer_parts.append("GatedDeltaNet WY")
    if config.ttt_state_dim:
        mixer_parts.append("TTT-Linear")
    dec_label = f"Décodeur {config.n_layers}x ({' + '.join(mixer_parts)})"
    if config.residual_streams > 1:
        dec_label += f" [mHC {config.residual_streams} flux]"
    if config.latent_reasoning_steps > 1:
        dec_label += f" [Raisonnement {config.latent_reasoning_steps}x{' PonderNet' if config.latent_reasoning_adaptive else ''}]"

    lines = ["```mermaid", "flowchart TD"]
    for node_id, label in mod_nodes:
        lines.append(f'    {node_id}["{label}"] --> RES["{res_label}"]')
    lines.append(
        f'    TXT["Texte UTF-8{f" + Byte-NGrams ({_fmt_int(config.ngram_vocab_size)})" if config.ngram_vocab_size else ""}"]'
        f' --> DEC["{dec_label}"]'
    )
    lines.append(f'    RES --> DEC')
    if config.memory_slots > 0:
        lines.append(f'    DEC --> MEM["Mémoire épisodique ({config.memory_slots} emplacements)"]')
        head_src = "MEM"
    else:
        head_src = "DEC"
    lines.append(f'    {head_src} --> LM["Texte + Multi-Token ({config.prediction_steps} pas)"]')
    lines.append(f'    {head_src} --> CLS["Classification ({config.num_classes} classes)"]')
    lines.append(f'    {head_src} --> FC["Prévision quantiles ({config.forecast_horizon} pas)"]')
    if config.flow_matching_dim > 0:
        lines.append(f'    {head_src} --> FM["Conditional Flow Matching (dim={config.flow_matching_dim})"]')
    if config.jepa_weight > 0:
        lines.append(f'    RES --> JEPA["Prédicteur Latent JEPA (w={config.jepa_weight})"]')
    lines.append(f'    RES --> ALN["Alignement {config.alignment_geometry} ({len(modalities) * (len(modalities) - 1) // 2} paires)"]')
    lines.append("```")
    return lines


def generate_model_card(config: BlueConfig, *, name: str | None = None,
                        seq_len: int = 2048, batch_size: int = 1) -> str:
    """Générer une fiche technique Markdown complète sans allouer les poids."""
    report = profile_architecture(config, seq_len=seq_len, batch_size=batch_size, measure=False)
    cache = report["cache_profile"]
    title = f"# BLUE — Fiche d'architecture ({name})" if name else "# BLUE — Fiche d'architecture"

    fsq_size = math.prod(config.latent_fsq_levels) if config.latent_fsq_levels else 0
    lines = [
        title,
        "",
        "## Schéma architectural dynamique",
        "",
        *_build_mermaid(config, report["modalities"]),
        "",
        "## Synthèse",
        "",
        "| Métrique | Valeur |",
        "| --- | ---: |",
        f"| Paramètres totaux | {_fmt_int(report['parameters'])} |",
        f"| Borne supérieure actifs / token | {_fmt_int(report['active_parameters_upper_bound'])} |",
        f"| Poids BF16 estimés | {report['weights_gib_bf16']:.3f} Gio |",
        f"| Poids INT8 estimés | {report['weights_gib_int8_estimate']:.3f} Gio |",
        f"| Dimension cachée (`d_model`) | {config.d_model} |",
        f"| Couches décodeur (effectives) | {report['decoder_layers']} ({report['effective_decoder_layers']}) |",
        f"| Flux résiduels mHC | {report['residual_streams']} |",
        f"| Experts routés / top-k / partagés | {config.n_experts} / {config.top_k} / {config.shared_experts} |",
        f"| Modalités natives | {len(report['modalities'])} ({', '.join(report['modalities'])}) |",
        f"| Combinaisons multimodales | {_fmt_int(report['modality_combinations'])} |",
        f"| Paires contrastives | {_fmt_int(report['contrastive_pairs'])} |",
        "",
        "## Composants architecturaux actifs",
        "",
        f"- **Attention décodeur** : "
        f"{'MLA (rang ' + str(config.kv_lora_rank) + ')' if config.kv_lora_rank else 'GQA'} "
        f"({config.n_heads} têtes Q / {config.n_kv_heads} KV)"
        f"{' + Differential Attention' if config.differential_attention else ''}"
        f"{f' + {config.attention_sinks} Attention Sinks' if config.attention_sinks else ''}"
        f"{' + Value Residual (ResFormer)' if config.value_residual else ''}"
        f"{f' + Soft-Cap (attn={config.attn_logit_softcap}, lm={config.logit_softcap})' if (config.attn_logit_softcap > 0 or config.logit_softcap > 0) else ''}"
        f", RoPE `{config.rope_scaling_type}` (×{config.rope_scaling_factor}, contexte max {_fmt_int(config.max_seq_len)}).",
        f"- **Attention croisée multimodale** : "
        f"{'Differential Cross-Attention' if config.cross_differential_attention else 'Cross-Attention standard'}"
        f"{f' + Latent KV (rang {config.cross_kv_lora_rank})' if config.cross_kv_lora_rank else ''} "
        f"vers {config.n_latents} latents "
        f"({'hiérarchique 2 niveaux' if config.hierarchical_resampler else '1 niveau'}"
        f"{f', goulot discret FSQ ({_fmt_int(fsq_size)} codes)' if fsq_size else ''}).",
        f"- **Branche récurrente / SSM / TTT** : "
        f"Convolution causale (k={config.conv_kernel})"
        f"{f' + Gated DeltaNet (état {config.ssm_state_dim}, {config.ssm_heads or config.n_heads} têtes, blocs WY {config.ssm_chunk_size})' if config.ssm_state_dim else ''}"
        f"{f' + Test-Time Training TTT-Linear (état {config.ttt_state_dim}, {config.ttt_heads or config.n_heads} têtes)' if config.ttt_state_dim else ''}.",
        f"- **MoE & Profondeur dynamique** : "
        f"SwiGLU MoE top-{config.top_k}/{config.n_experts}"
        f"{f' (+ {config.moe_null_experts} experts nuls zéro-FLOP)' if config.moe_null_experts else ''}"
        f"{' + porte partagée' if config.shared_expert_gate else ''}"
        f"{f' + équilibrage sans perte auxiliaire (rate={config.router_bias_update_rate})' if config.router_bias_update_rate > 0 else ''}"
        f"{f' + Causal Mixture-of-Depths (capacité={config.depth_routing_capacity})' if config.depth_routing_capacity < 1.0 else ''}.",
        f"- **Raisonnement & Mémoire** : "
        f"{config.latent_reasoning_steps} itération(s) latente(s)"
        f"{f' avec arrêt adaptatif PonderNet (prior={config.ponder_prior})' if config.latent_reasoning_adaptive else ''}"
        f"{f', mémoire épisodique ({config.memory_slots} emplacements)' if config.memory_slots else ''}.",
        f"- **Alignement & Génération continue** : "
        f"géométrie `{config.alignment_geometry}`"
        f"{f' (courbure c={config.alignment_curvature})' if config.alignment_geometry == 'hyperbolic' else ''}"
        f"{f', Conditional Flow Matching (dim={config.flow_matching_dim}, {config.flow_matching_layers} blocs AdaLN-Zero)' if config.flow_matching_dim else ''}"
        f"{f', Multimodal-JEPA (poids={config.jepa_weight})' if config.jepa_weight > 0 else ''}"
        f"{', pondération adaptative par incertitude' if config.adaptive_loss_weighting else ''}.",
        "",
        f"## Empreinte mémoire de génération (seq_len={cache['seq_len']}, batch={cache['batch_size']})",
        "",
        "| État | Empreinte |",
        "| --- | ---: |",
        f"| Cache KV (BF16) | {cache['kv_cache_mib_bf16']:.4f} Mio |",
        f"| Référence GQA standard (BF16) | {cache['standard_gqa_kv_mib_bf16']:.4f} Mio |",
        f"| Ratio de compression KV | {cache['kv_compression_ratio']:.3f}× |",
        f"| État convolution causale O(1) (FP32) | {cache['conv_state_mib_fp32']:.4f} Mio |",
        f"| État Gated DeltaNet O(1) (FP32) | {cache['ssm_state_mib_fp32']:.4f} Mio |",
        f"| État Test-Time Training O(1) (FP32) | {cache['ttt_state_mib_fp32']:.4f} Mio |",
        f"| Mémoire épisodique O(1) (FP32) | {cache['episodic_memory_mib_fp32']:.4f} Mio |",
        "",
        "## Répartition des paramètres par module",
        "",
        "| Module | Paramètres | Part |",
        "| --- | ---: | ---: |",
    ]
    total = max(report["parameters"], 1)
    for mod_name, count in sorted(report["modules"].items(), key=lambda kv: kv[1], reverse=True):
        pct = 100.0 * count / total
        lines.append(f"| `{mod_name}` | {_fmt_int(count)} | {pct:.2f}% |")
    lines.append("")
    return "\n".join(lines)
