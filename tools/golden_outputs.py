"""Enregistrer puis comparer des sorties de référence pour valider une refactorisation.

Pour quatorze combinaisons de fonctionnalités du preset ``tiny``, l'enregistrement
contient les logits, prévisions, pertes, gradients de tous les paramètres, et
des générations gloutonnes, stochastiques, spéculatives, Jacobi et en faisceau
(avec padding à gauche). Comparer deux enregistrements, faits sur la même
machine avec deux versions du code, montre toute dérive numérique.

    python tools/golden_outputs.py record --output runs/golden-before.pt
    # ... modifier le code ...
    python tools/golden_outputs.py record --output runs/golden-after.pt
    python tools/golden_outputs.py compare runs/golden-before.pt runs/golden-after.pt --tolerance 1e-5
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blue.config import BlueConfig  # noqa: E402
from blue.data import synthetic_batch  # noqa: E402
from blue.model import Blue  # noqa: E402

ALL_MODALITIES = dict(audio_enabled=True, graph_enabled=True, pointcloud_enabled=True,
                      tabular_enabled=True, video_enabled=True)
CONFIGURATIONS = {
    "tiny": {},
    "mla": dict(kv_lora_rank=16),
    "differential": dict(differential_attention=True, cross_differential_attention=True),
    "sinks": dict(attention_sinks=2, forgetting_attention=True),
    "nsa": dict(nsa_block_size=2),
    "recurrent": dict(ssm_state_dim=8, ttt_state_dim=8, memory_slots=4, ngram_vocab_size=64, memory_decay=True,
                      ttt_weight_decay=True, ssm_d_skip=True),
    "residual": dict(value_residual=True, qk_residual=True, residual_streams=2),
    "checkpointing": dict(value_residual=True, qk_residual=True, gradient_checkpointing=True, moe_entropy_weight=0.1),
    "moe": dict(moe_null_experts=2, moe_shared_latent_dim=8, moe_entropy_weight=0.1, router_bias_update_rate=0.01,
                moe_router_temperature=True, depth_routing_capacity=0.7),
    "hyperbolic": dict(moe_router_geometry="hyperbolic", alignment_geometry="hyperbolic", kan_degree=3,
                       conv_diff_branch=True, stable_rms_norm=True),
    "ponder": dict(latent_reasoning_steps=3, latent_reasoning_adaptive=True),
    "temporal": dict(temporal_spectral_mixing=True, temporal_wavelet_mixing=True, temporal_statistics=True),
    "modalities": dict(hierarchical_resampler=True, cross_modal_fusion_layers=1, jepa_weight=0.1, prediction_steps=4,
                       chained_future_heads=True, mtp_kl_weight=0.1, **ALL_MODALITIES),
    "attention": dict(talking_heads=True, head_confidence_gate=True, gated_value_attention=True,
                      dynamic_attn_scale=True, qk_norm_temperature=True, dynamic_mixer_router=True,
                      multimodal_adaln=True, head_cross_pool=True, attn_logit_softcap=8.0, logit_softcap=20.0,
                      dynamic_tanh_norm=True, moe_input_residual_gate=True, layer_scale_init=0.1),
}
SAMPLERS = {
    "top_k": {},
    "filters": dict(min_p=0.05, typical_p=0.9, tfs_z=0.9, top_a=0.1, eta_cutoff=0.01, edt_theta=0.2),
    "mirostat": dict(mirostat_tau=3.0),
    "penalties": dict(repetition_penalty=1.2, no_repeat_ngram_size=2, top_k=0, top_p=1.0),
}


def record(device):
    results = {}
    for name, changes in CONFIGURATIONS.items():
        torch.manual_seed(1)
        model = Blue(replace(BlueConfig.preset("tiny"), **changes)).to(device)
        batch = synthetic_batch(model.config, batch_size=3, device=device)
        mask = batch["series_mask"].clone()
        mask[0, 20:] = False
        mask[1, :4] = False
        batch["series_mask"] = mask
        output = model(**batch)
        output.loss.backward()
        entry = dict(loss=output.loss.detach(), logits=output.logits.detach(), forecast=output.forecast.detach(),
                     usage=output.expert_usage)
        entry.update({f"loss/{key}": value.detach() for key, value in output.losses.items()})
        entry.update({f"grad/{key}": p.grad for key, p in model.named_parameters() if p.grad is not None})
        model.eval()
        ids = batch["input_ids"][:2, :8]
        context = dict(images=batch["images"][:2], series=batch["series"][:2])
        left_padding = torch.ones_like(ids, dtype=torch.bool)
        left_padding[0, :2] = False
        single = dict(images=batch["images"][:1], series=batch["series"][:1])
        entry["generate/greedy"] = model.generate(ids, max_new_tokens=16, temperature=0, eos_id=None,
                                                  attention_mask=left_padding, **context)
        entry["generate/speculative"] = model.generate(ids[:1], max_new_tokens=12, temperature=0, eos_id=None,
                                                       speculative=True, **single)
        entry["generate/jacobi"] = model.generate(ids[:1], max_new_tokens=12, temperature=0, eos_id=None,
                                                  jacobi=True, **single)
        entry["generate/beam"] = model.generate(ids[:1], max_new_tokens=8, beam_width=3, **single)
        for sampler, options in SAMPLERS.items():
            torch.manual_seed(3)
            entry[f"sample/{sampler}"] = model.generate(ids, max_new_tokens=12, eos_id=None, **context, **options)
        torch.manual_seed(3)
        entry["sample/speculative"] = model.generate(ids[:1], max_new_tokens=12, eos_id=None, speculative=True, **single)
        if changes.get("audio_enabled"):
            inputs = {key: value for key, value in batch.items() if key in (
                "input_ids", "images", "series", "audio", "graph_nodes", "graph_edges", "points",
                "tabular_num", "tabular_cat", "video")}
            entry.update({f"embedding/{key}": value.detach() for key, value in model.encode_modalities(**inputs).items()})
        results[name] = {key: value.detach().cpu() for key, value in entry.items()}
    return results


def compare(reference, candidate, tolerance):
    """Écart relatif maximal par configuration ; les sorties entières doivent être égales."""
    failures = 0
    for name, expected in reference.items():
        actual = candidate.get(name, {})
        worst, worst_key = 0.0, ""
        for key, value in expected.items():
            other = actual.get(key)
            if other is None or other.shape != value.shape:
                print(f"{name}: {key} absent ou de forme différente")
                failures += 1
                continue
            if not value.is_floating_point():
                if not torch.equal(value, other):
                    print(f"{name}: {key} diffère")
                    failures += 1
                continue
            if value.numel():
                difference = ((value - other).abs().max() / (value.abs().max() + 1e-6)).item()
                if difference > worst:
                    worst, worst_key = difference, key
        status = "ok" if worst <= tolerance else "ÉCART"
        failures += worst > tolerance
        print(f"{name}: écart relatif maximal {worst:.2e} ({worst_key or '-'}) {status}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    recorder = commands.add_parser("record", help="Enregistrer les sorties de la version courante")
    recorder.add_argument("--output", required=True)
    recorder.add_argument("--device", default="cpu")
    recorder.add_argument("--threads", type=int, default=4)
    comparer = commands.add_parser("compare", help="Comparer deux enregistrements")
    comparer.add_argument("reference")
    comparer.add_argument("candidate")
    comparer.add_argument("--tolerance", type=float, default=0.0, help="Écart relatif admis (défaut : 0)")
    args = parser.parse_args()
    if args.command == "record":
        torch.set_num_threads(args.threads)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(record(torch.device(args.device)), args.output)
        print(f"Sorties de référence écrites dans {args.output}")
        return
    failures = compare(torch.load(args.reference, weights_only=True), torch.load(args.candidate, weights_only=True),
                       args.tolerance)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
