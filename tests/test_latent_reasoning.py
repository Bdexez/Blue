from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class LatentReasoningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        for bad in (0, -1, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, latent_reasoning_steps=bad)
        model = Blue(base).eval()
        ids = torch.randint(4, 260, (1, 6))
        with self.assertRaises(ValueError):
            model(ids, reasoning_steps=2)
        with self.assertRaises(ValueError):
            model(ids, reasoning_steps=0)

    def test_recurrent_depth_causality_cache_test_time_scaling_and_gradients(self):
        torch.manual_seed(9001)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         latent_reasoning_steps=3,
                         residual_streams=2,
                         ssm_state_dim=8,
                         differential_attention=True,
                         kv_lora_rank=16,
                         vision_rope_2d=True,
                         temporal_spectral_mixing=True,
                         shared_expert_gate=True,
                         router_bias_update_rate=0.01,
                         stable_rms_norm=True,
                         gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertGreater(info["parameters"], inspect_config(BlueConfig.preset("tiny"))["parameters"])
        model = Blue(config).eval()
        batch = synthetic_batch(config, batch_size=2, seed=9002)
        ids = torch.randint(4, 260, (2, 40))

        # 1. Causality across 3 recurrent depth iterations.
        changed = ids.clone()
        changed[:, 10:] = torch.randint(4, 260, changed[:, 10:].shape)
        with torch.no_grad():
            first = model(ids, images=batch["images"], series=batch["series"]).logits
            second = model(changed, images=batch["images"], series=batch["series"]).logits
        torch.testing.assert_close(first[:, :10], second[:, :10], rtol=1e-5, atol=1e-6)

        # 2. Exact sliding-window + recurrent-slot cache equivalence.
        with torch.no_grad():
            full = model(ids, images=batch["images"], series=batch["series"]).logits
            p1 = model(ids[:, :34], images=batch["images"], series=batch["series"], use_cache=True)
            p2 = model(ids[:, 34:38], cache=p1.cache, use_cache=True)
            p3 = model(ids[:, 38:], cache=p2.cache, use_cache=True)
        self.assertEqual(len(p3.cache.layers), config.n_layers * config.latent_reasoning_steps)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits), 1), full, rtol=3e-5, atol=3e-6)

        # 3. Dynamic test-time compute scaling (reasoning_steps=1 vs 3 vs 4).
        with torch.no_grad():
            s1 = model(ids[:, :12], reasoning_steps=1)
            s3 = model(ids[:, :12], reasoning_steps=3)
            s4_full = model(ids[:, :12], reasoning_steps=4)
            s4_p1 = model(ids[:, :8], reasoning_steps=4, use_cache=True)
            s4_p2 = model(ids[:, 8:12], cache=s4_p1.cache, use_cache=True, reasoning_steps=4)
        self.assertFalse(torch.allclose(s1.logits, s3.logits))
        self.assertFalse(torch.allclose(s3.logits, s4_full.logits))
        torch.testing.assert_close(torch.cat((s4_p1.logits, s4_p2.logits), 1), s4_full.logits, rtol=3e-5, atol=3e-6)
        gen = model.generate(ids[:1, :6], max_new_tokens=4, temperature=0, reasoning_steps=4)
        self.assertEqual(gen.shape, (1, 10))

        # 4. Gradients reach all recurrent reasoning parameters under bf16 + checkpointing.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        self.assertEqual(out.expert_usage.shape, (config.n_layers, config.n_experts))
        out.loss.backward()
        for param in (model.reasoning_step_embed.weight, model.reasoning_norm.weight,
                      model.reasoning_adapter.weight, model.reasoning_gate):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(), 0.0)

        # 5. Checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reasoning.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertEqual(restored.config.latent_reasoning_steps, 3)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
