"""Opérations modulaires de la ligne de commande Blue (inspection, démo, export, flux atomiques)."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

from .calibration import file_sha256
from .config import BlueConfig
from .data import synthetic_batch
from .lora import merge_dora
from .model import Blue
from .quantize import quantize_model_int4, quantize_model_int8
from .tokenizer import ByteTokenizer
from .training import load_checkpoint, save_checkpoint


def same_file(first, second):
    """Comparer aussi les chemins existants qui partagent un lien physique."""
    if Path(first).resolve() == Path(second).resolve():
        return True
    try:
        return os.path.samefile(first, second)
    except FileNotFoundError:
        return False


@contextmanager
def prediction_stream(path):
    if path is None:
        yield sys.stdout
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_config(args):
    return BlueConfig.load(args.config) if args.config else BlueConfig.preset(args.preset)


def inspect_config(config):
    # Pas d'allocation des poids, même pour une architecture de centaines de milliards.
    with torch.device("meta"):
        model = Blue(config)
    total = sum(p.numel() for p in model.parameters())
    expert_size = 3 * config.d_model * config.hidden_dim
    unselected = config.n_layers * (config.n_experts - config.top_k) * expert_size
    return {
        "parameters": total,
        "parameters_with_selected_decoder_experts_upper_bound": total - unselected,
        "decoder_layers": config.n_layers,
        "effective_decoder_layers": config.n_layers * config.latent_reasoning_steps,
        "residual_streams": config.residual_streams,
        "experts_per_layer": config.n_experts,
        "selected_experts_per_token": config.top_k,
        "shared_experts_per_layer": config.shared_experts,
        "architecture_features": {
            "rope_scaling": config.rope_scaling_type,
            "ngram_vocab_size": config.ngram_vocab_size,
            "differential_attention": config.differential_attention,
            "cross_differential_attention": config.cross_differential_attention,
            "multimodal_adaln": config.multimodal_adaln,
            "kv_lora_rank": config.kv_lora_rank,
            "cross_kv_lora_rank": config.cross_kv_lora_rank,
            "attention_sinks": config.attention_sinks,
            "value_residual": config.value_residual,
            "logit_softcap": config.logit_softcap,
            "attn_logit_softcap": config.attn_logit_softcap,
            "dynamic_attn_scale": config.dynamic_attn_scale,
            "nsa_block_size": config.nsa_block_size,
            "nsa_top_k": config.nsa_top_k,
            "ssm_state_dim": config.ssm_state_dim,
            "ssm_every": config.ssm_every,
            "ttt_state_dim": config.ttt_state_dim,
            "ttt_every": config.ttt_every,
            "shared_expert_gate": config.shared_expert_gate,
            "moe_null_experts": config.moe_null_experts,
            "moe_shared_latent_dim": config.moe_shared_latent_dim,
            "moe_router_geometry": config.moe_router_geometry,
            "router_bias_update_rate": config.router_bias_update_rate,
            "depth_routing_capacity": config.depth_routing_capacity,
            "vision_rope_2d": config.vision_rope_2d,
            "temporal_spectral_mixing": config.temporal_spectral_mixing,
            "temporal_wavelet_mixing": config.temporal_wavelet_mixing,
            "forecast_tail_power": config.forecast_tail_power,
            "audio_enabled": config.audio_enabled,
            "graph_enabled": config.graph_enabled,
            "pointcloud_enabled": config.pointcloud_enabled,
            "tabular_enabled": config.tabular_enabled,
            "video_enabled": config.video_enabled,
            "hierarchical_resampler": config.hierarchical_resampler,
            "cross_modal_fusion_layers": config.cross_modal_fusion_layers,
            "register_latents": config.register_latents,
            "latent_fsq_levels": list(config.latent_fsq_levels),
            "latent_diversity_weight": config.latent_diversity_weight,
            "flow_matching_dim": config.flow_matching_dim,
            "jepa_weight": config.jepa_weight,
            "adaptive_loss_weighting": config.adaptive_loss_weighting,
            "memory_slots": config.memory_slots,
            "alignment_geometry": config.alignment_geometry,
            "latent_reasoning_steps": config.latent_reasoning_steps,
            "latent_reasoning_adaptive": config.latent_reasoning_adaptive,
            "dynamic_tanh_norm": config.dynamic_tanh_norm,
            "kan_degree": config.kan_degree,
            "checkpoint_every": config.checkpoint_every,
            "evidential_classification": config.evidential_classification,
            "mtp_kl_weight": config.mtp_kl_weight,
        },
        "weights_gib_fp32": round(total * 4 / 2**30, 3),
        "weights_gib_bf16": round(total * 2 / 2**30, 3),
        "adamw_gib_fp32_excluding_activations": round(total * 16 / 2**30, 3),
        "modules": {name: sum(p.numel() for p in module.parameters())
                    for name, module in model.named_children() if name != "lm_head"},
    }


def run_demo(args, *, load_checkpoint_fn=load_checkpoint):
    torch.manual_seed(42)
    model = (load_checkpoint_fn(args.checkpoint, args.device)[0] if args.checkpoint
             else Blue(get_config(args)).to(args.device))
    model.eval()
    batch = synthetic_batch(model.config, batch_size=2, seed=7, device=args.device)
    with torch.inference_mode():
        output = model(**batch)
        prompt = torch.tensor([ByteTokenizer().encode("Signal", eos=False)], device=args.device)
        generated = model.generate(prompt, images=batch["images"][:1], series=batch["series"][:1],
                                   max_new_tokens=16, temperature=0)
    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "text_logits_shape": list(output.logits.shape),
        "classification_shape": list(output.classification_logits.shape),
        "forecast_shape": list(output.forecast.shape),
        "predicted_classes": output.classification_logits.argmax(-1).tolist(),
        "target_classes": batch["class_labels"].tolist(),
        "quantiles": list(model.config.quantiles),
        "forecast_first_example": output.forecast[0].tolist(),
        "loss": output.loss.item(),
        "losses": {key: value.item() for key, value in output.losses.items()},
        "expert_usage": output.expert_usage.tolist(),
        "generated_text": ByteTokenizer().decode(generated[0]),
        "checkpoint": args.checkpoint,
        "note": "Données synthétiques ; aucune capacité générale n'est démontrée par cette démo.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_export(args, parser, *, load_checkpoint_fn=load_checkpoint,
               save_checkpoint_fn=save_checkpoint, file_sha256_fn=file_sha256):
    if args.quantize_int8 and args.quantize_int4:
        parser.error("Choisir soit --quantize-int8 soit --quantize-int4, pas les deux")
    transforms = args.stable_rms_norm or args.merge_dora or args.quantize_int8 or args.quantize_int4
    if transforms and same_file(args.output, args.checkpoint):
        parser.error("Le checkpoint exporté doit être distinct du checkpoint source")
    if args.int8_min_numel < 1:
        parser.error("int8-min-numel doit être >= 1")
    parent_sha = file_sha256_fn(args.checkpoint) if args.stable_rms_norm else None
    model, payload = load_checkpoint_fn(args.checkpoint, mmap=not transforms,
                                        overrides={"stable_rms_norm": True} if args.stable_rms_norm else None)
    step = payload["step"]
    trainer = dict(payload.get("trainer", {}), inference_only=True)
    if args.use_ema:
        ema_state = payload.get("ema")
        if ema_state is None:
            raise ValueError("Le checkpoint source ne contient pas d'état EMA")
        from .ema import ModelEMA
        ema = ModelEMA(model, ema_state["decay"])
        ema.load_state_dict(ema_state)
        with torch.no_grad():
            params = dict(model.named_parameters())
            for name, shadow in ema.shadow.items():
                params[name].copy_(shadow.to(device=params[name].device, dtype=params[name].dtype))
            model.lm_head.weight = model.embedding.weight
        trainer["exported_ema"] = True
    if args.merge_dora:
        merged_count = merge_dora(model)
        if merged_count == 0:
            raise ValueError("Le checkpoint source ne contient aucun adaptateur DoRA/LoRA à fusionner")
        trainer.pop("dora", None)
        trainer["merged_dora"] = merged_count
    if args.quantize_int8 or args.quantize_int4:
        if trainer.get("dora") is not None:
            raise ValueError("Fusionner d'abord DoRA avec --merge-dora avant la quantification")
        if trainer.get("quantized_int8") is not None or trainer.get("quantized_int4") is not None:
            raise ValueError("Le checkpoint source est déjà quantifié")
        if args.quantize_int8:
            q_stats = quantize_model_int8(model, min_numel=args.int8_min_numel)
            trainer["quantized_int8"] = {"min_numel": args.int8_min_numel, **q_stats}
        else:
            q_stats = quantize_model_int4(model, min_numel=args.int8_min_numel,
                                          group_size=args.int4_group_size)
            trainer["quantized_int4"] = {"min_numel": args.int8_min_numel, **q_stats}
    if args.stable_rms_norm:
        if file_sha256_fn(args.checkpoint) != parent_sha:
            raise ValueError("Le checkpoint source a changé pendant l'export")
        trainer.update(stable_rms_norm_parent_sha256=parent_sha)
    del payload
    save_checkpoint_fn(args.output, model, step=step, trainer=trainer)
    print(f"Poids d'inférence écrits dans {args.output}")
