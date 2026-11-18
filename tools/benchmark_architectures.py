"""Comparer Blue classique et Blue hybride complet (5 modalités, mHC, DeltaNet, MLA, Muon)."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.optimizer import build_optimizer


def build_variant(name: str) -> BlueConfig:
    base = BlueConfig.preset("tiny")
    if name == "baseline":
        return base
    if name == "hybrid":
        return replace(
            base,
            rope_scaling_type="yarn",
            rope_scaling_factor=2.0,
            differential_attention=True,
            kv_lora_rank=16,
            attention_sinks=2,
            ssm_state_dim=8,
            ssm_heads=4,
            shared_expert_gate=True,
            router_bias_update_rate=0.005,
            depth_routing_capacity=0.75,
            residual_streams=2,
            latent_reasoning_steps=2,
            memory_slots=8,
            vision_rope_2d=True,
            temporal_statistics=True,
            temporal_statistics_relative=True,
            temporal_spectral_mixing=True,
            audio_enabled=True,
            audio_patch_size=8,
            audio_layers=1,
            graph_enabled=True,
            graph_node_features=6,
            graph_edge_features=3,
            graph_layers=1,
            graph_rwse_steps=6,
            pointcloud_enabled=True,
            pointcloud_features=2,
            pointcloud_layers=1,
            hierarchical_resampler=True,
            flow_matching_dim=12,
            flow_matching_layers=1,
            prediction_steps=3,
            stable_rms_norm=True,
        )
    raise ValueError(f"Variante inconnue : {name}")


def measure_variant(variant_name: str, optimizer_name: str, *, steps: int, batch_size: int, seed: int) -> dict:
    torch.manual_seed(seed)
    config = build_variant(variant_name)
    model = Blue(config)
    optimizer = build_optimizer(model, name=optimizer_name, lr=2e-3, weight_decay=0.01)
    train_batch = synthetic_batch(config, batch_size=batch_size, seed=seed + 100)
    val_batch = synthetic_batch(config, batch_size=batch_size, seed=seed + 200)
    flow_train = (train_batch["forecast_targets"].reshape(batch_size, -1)[:, :config.flow_matching_dim]
                  if config.flow_matching_dim else None)
    flow_val = (val_batch["forecast_targets"].reshape(batch_size, -1)[:, :config.flow_matching_dim]
                if config.flow_matching_dim else None)

    def forward_batch(batch, flow_targets):
        return model(**batch, flow_targets=flow_targets) if flow_targets is not None else model(**batch)

    model.eval()
    with torch.no_grad():
        initial = forward_batch(val_batch, flow_val)
        cache_out = model(val_batch["input_ids"][:1, :16], use_cache=True)
    kv_elements_per_token = sum(
        (layer.key.numel() + layer.value.numel()) / layer.key.size(-2)
        for layer in cache_out.cache.layers
    )

    model.train()
    t0 = time.monotonic()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = forward_batch(train_batch, flow_train)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if config.router_bias_update_rate > 0:
            model.update_router_biases(out.expert_usage)
    elapsed = round(time.monotonic() - t0, 4)

    model.eval()
    with torch.no_grad():
        final = forward_batch(val_batch, flow_val)
    return {
        "variant": variant_name,
        "optimizer": optimizer_name,
        "parameters": sum(p.numel() for p in model.parameters()),
        "kv_cache_elements_per_token": round(kv_elements_per_token, 2),
        "elapsed_seconds": elapsed,
        "initial_val_loss": round(initial.loss.item(), 5),
        "final_val_loss": round(final.loss.item(), 5),
        "initial_losses": {k: round(v.item(), 5) for k, v in initial.losses.items()},
        "final_losses": {k: round(v.item(), 5) for k, v in final.losses.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    results = [
        measure_variant("baseline", "adamw", steps=args.steps, batch_size=args.batch_size, seed=args.seed),
        measure_variant("hybrid", "adamw", steps=args.steps, batch_size=args.batch_size, seed=args.seed),
        measure_variant("hybrid", "muon", steps=args.steps, batch_size=args.batch_size, seed=args.seed),
    ]
    payload = {"steps": args.steps, "batch_size": args.batch_size, "seed": args.seed, "results": results}
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
