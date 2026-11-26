"""Chemins CUDA : ignorés sans GPU, exécutés localement quand CUDA est disponible."""

import unittest
from dataclasses import replace

import torch

from blue.config import BlueConfig
from blue.data import synthetic_batch
from blue.model import Blue

CONFIGURATIONS = {
    "tiny": {},
    "recurrent": dict(ssm_state_dim=8, ttt_state_dim=8, memory_slots=4, ngram_vocab_size=64),
    "attention": dict(kv_lora_rank=16, attention_sinks=2, nsa_block_size=2, forgetting_attention=True),
    "temporal": dict(temporal_spectral_mixing=True, temporal_wavelet_mixing=True, temporal_statistics=True,
                     forecast_adapter_context=8),
    "routing": dict(moe_null_experts=1, depth_routing_capacity=0.6, residual_streams=2),
}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA indisponible")
class CudaSmokeTests(unittest.TestCase):
    def test_training_matches_cpu_and_runs_under_bf16(self):
        for name, changes in CONFIGURATIONS.items():
            with self.subTest(name):
                torch.manual_seed(0)
                cpu = Blue(replace(BlueConfig.preset("tiny"), **changes))
                gpu = Blue(replace(BlueConfig.preset("tiny"), **changes))
                gpu.load_state_dict(cpu.state_dict())
                gpu = gpu.cuda()
                batch = synthetic_batch(cpu.config, batch_size=3)
                expected = cpu(**batch).loss
                output = gpu(**{key: value.cuda() for key, value in batch.items()})
                torch.testing.assert_close(output.loss.cpu(), expected, rtol=1e-3, atol=1e-3)
                output.loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in gpu.parameters() if p.grad is not None))
                gpu.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    mixed = gpu(**{key: value.cuda() for key, value in batch.items()}).loss
                self.assertTrue(torch.isfinite(mixed))
                mixed.backward()

    def test_deterministic_mode_reproduces_gpu_training_bit_for_bit(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            for run in ("first", "second"):
                subprocess.run([sys.executable, "-m", "blue", "--threads", "1", "--deterministic", "train",
                                "--preset", "tiny", "--device", "cuda", "--steps", "4", "--batch-size", "4",
                                "--eval-every", "4", "--save-every", "4", "--output", str(Path(directory) / run)],
                               check=True, capture_output=True)
            first, second = (torch.load(Path(directory) / run / "latest.pt", weights_only=True)["model"]
                             for run in ("first", "second"))
        for name, value in first.items():
            self.assertTrue(torch.equal(value, second[name]), name)

    def test_verifier_passes_on_gpu(self):
        from blue.verifier import verify_architecture
        report = verify_architecture(replace(BlueConfig.preset("tiny"), **CONFIGURATIONS["recurrent"]), device="cuda")
        self.assertTrue(report["passed"], report)

    def test_cached_generation_matches_cpu(self):
        torch.manual_seed(0)
        model = Blue(replace(BlueConfig.preset("tiny"), **CONFIGURATIONS["recurrent"])).eval()
        prompt = torch.randint(4, 200, (2, 6), generator=torch.Generator().manual_seed(1))
        series = torch.randn(2, 12, model.config.series_features, generator=torch.Generator().manual_seed(2))
        expected = model.generate(prompt, max_new_tokens=8, temperature=0, eos_id=None, series=series)
        model = model.cuda()
        result = model.generate(prompt.cuda(), max_new_tokens=8, temperature=0, eos_id=None, series=series.cuda())
        self.assertTrue(torch.equal(result.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
