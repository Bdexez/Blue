from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class MultiHeadLatentAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        with self.assertRaises(ValueError):
            replace(base, kv_lora_rank=-1)
        with self.assertRaises(ValueError):
            replace(base, kv_lora_rank=True)
        with self.assertRaises(ValueError):
            replace(base, d_model=12, n_heads=2, n_kv_heads=2, kv_lora_rank=8)
        with self.assertRaises(ValueError):
            replace(base, d_model=24, n_heads=2, n_kv_heads=2, kv_lora_rank=8, differential_attention=True)

    def test_mla_and_mla_differential_cache_compression_causality_and_gradients(self):
        for diff in (False, True):
            with self.subTest(differential=diff):
                torch.manual_seed(3010 + int(diff))
                config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                                 kv_lora_rank=16, differential_attention=diff,
                                 rope_scaling_type="yarn", rope_scaling_factor=2.0,
                                 gradient_checkpointing=True)
                info = inspect_config(config)
                self.assertGreater(info["parameters"], 0)
                model = Blue(config).eval()
                batch = synthetic_batch(config, batch_size=2, seed=3011)
                ids = torch.randint(4, 260, (2, 44))
                # Verify KV cache is actually compressed to 1 latent head of size kv_lora_rank + rope_dim.
                with torch.no_grad():
                    full = model(ids, images=batch["images"], series=batch["series"]).logits
                    p1 = model(ids[:, :34], images=batch["images"], series=batch["series"], use_cache=True)
                    p2 = model(ids[:, 34:39], cache=p1.cache, use_cache=True)
                    p3 = model(ids[:, 39:], cache=p2.cache, use_cache=True)
                joined = torch.cat((p1.logits, p2.logits, p3.logits), 1)
                torch.testing.assert_close(joined, full, rtol=2e-5, atol=2e-6)
                # Layer 0 has sliding window 32 -> retains 31 entries; layer 1 is global -> retains 44.
                self.assertEqual(p3.cache.layers[0].key.shape, (2, 1, 31, 16))
                self.assertEqual(p3.cache.layers[1].key.shape, (2, 1, 44, 16))
                expected_rope = 8 if not diff else 8  # 2 * (16 // 4) == 8 for diff, 16 // 2 == 8 for standard
                self.assertEqual(p3.cache.layers[1].value.shape, (2, 1, 44, expected_rope))
                # Verify cache is strictly smaller than standard GQA KV cache (2 * 2 * 16 = 64 vs 16 + 8 = 24).
                mla_elems = p3.cache.layers[1].key.numel() + p3.cache.layers[1].value.numel()
                std_elems = 2 * 2 * config.n_kv_heads * 44 * (config.d_model // config.n_heads)
                self.assertLess(mla_elems, std_elems)
                # Causality.
                changed = ids.clone()
                changed[:, 10:] = torch.randint(4, 260, changed[:, 10:].shape)
                with torch.no_grad():
                    alt = model(changed, images=batch["images"], series=batch["series"]).logits
                torch.testing.assert_close(full[:, :10], alt[:, :10], rtol=1e-5, atol=1e-6)
                # Gradients + bf16 + checkpointing.
                model.train()
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out = model(**batch)
                self.assertTrue(torch.isfinite(out.loss))
                out.loss.backward()
                for block in model.blocks:
                    for name in ("kv_down", "k_up", "v_up", "k_rope"):
                        grad = getattr(block.attn, name).weight.grad
                        self.assertIsNotNone(grad)
                        self.assertTrue(torch.isfinite(grad).all())
                        self.assertGreater(grad.abs().sum().item(), 0.0)
                # Checkpoint roundtrip.
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "mla.pt"
                    save_checkpoint(path, model.eval())
                    restored, _ = load_checkpoint(path)
                    self.assertEqual(restored.config.kv_lora_rank, 16)
                    with torch.no_grad():
                        torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
