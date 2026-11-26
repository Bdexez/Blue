"""Mesurer les chemins critiques : pas d'entraînement, décodage et branches récurrentes.

Chaque mesure est la médiane de plusieurs répétitions après échauffement. Les
durées dépendent de la machine, du nombre de threads et de la charge du GPU :
comparer deux versions sur la même machine, dans la même session.

    python tools/benchmark_hotpaths.py --device cpu --device cuda --output runs/hotpaths.json
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blue.config import BlueConfig  # noqa: E402
from blue.data import synthetic_batch  # noqa: E402
from blue.memory import EpisodicMemoryBank  # noqa: E402
from blue.model import Blue  # noqa: E402
from blue.modules import RMSNorm  # noqa: E402
from blue.ssm import GatedDeltaNet  # noqa: E402
from blue.ttt import TTTLinear  # noqa: E402

VARIANTS = {
    "": {},
    "+recurrent": dict(ssm_state_dim=16, ttt_state_dim=16, memory_slots=8),
    "+temporal": dict(temporal_spectral_mixing=True, temporal_wavelet_mixing=True),
}


def median_ms(function, device, repeats):
    durations = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
    durations.sort()
    return 1000 * durations[len(durations) // 2]


def model_measures(name, changes, device, args):
    torch.manual_seed(0)
    model = Blue(replace(BlueConfig.preset(args.preset), **changes)).to(device)
    batch = synthetic_batch(model.config, batch_size=args.batch_size, device=device)
    ids = torch.randint(4, 200, (args.batch_size, min(args.text_length, model.config.max_seq_len)), device=device)
    batch.update(input_ids=ids, labels=ids.clone(), attention_mask=torch.ones_like(ids, dtype=torch.bool))

    def train_step():
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()

    model.train()
    train_step()
    result = dict(variant=args.preset + name, train_step_ms=median_ms(train_step, device, args.repeats))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        train_step()
        result["train_peak_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    model.eval()
    prompt = ids[:1, :min(32, model.config.max_seq_len - args.new_tokens)]
    context = dict(images=batch["images"][:1], series=batch["series"][:1])

    def decode():
        model.generate(prompt, max_new_tokens=args.new_tokens, temperature=0.8, eos_id=None, **context)

    decode()
    result["decode_ms_per_token"] = median_ms(decode, device, max(args.repeats // 2, 1)) / args.new_tokens
    return result


def branch_measures(device, args):
    layers = dict(
        gated_deltanet=GatedDeltaNet(256, 16, 8, norm_cls=RMSNorm),
        ttt_linear=TTTLinear(256, 16, 8, norm_cls=RMSNorm),
        episodic_memory=EpisodicMemoryBank(256, 8, 8),
    )
    results = {}
    for name, layer in layers.items():
        layer = layer.to(device)
        x = torch.randn(args.batch_size, 96, 256, device=device, requires_grad=True)
        valid = torch.ones(args.batch_size, 96, dtype=torch.bool, device=device)

        def step():
            layer(x, valid)[0].sum().backward()

        step()
        results[name + "_forward_backward_ms"] = median_ms(step, device, args.repeats)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", action="append", help="cpu, cuda... (plusieurs possibles)")
    parser.add_argument("--preset", default="small", help="Preset mesuré (défaut : small)")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-length", type=int, default=96)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", help="Rapport JSON ; sinon sortie standard")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = dict(torch=torch.__version__, python=platform.python_version(), threads=args.threads, preset=args.preset,
                  batch_size=args.batch_size, text_length=args.text_length, new_tokens=args.new_tokens,
                  devices={})
    for name in args.device or ["cpu"]:
        device = torch.device(name)
        entry = dict(models=[model_measures(variant, changes, device, args) for variant, changes in VARIANTS.items()],
                     branches=branch_measures(device, args))
        if device.type == "cuda":
            entry["gpu"] = torch.cuda.get_device_name(device)
        report["devices"][name] = entry
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
