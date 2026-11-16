from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class DifferentialAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        with self.assertRaises(ValueError):
            replace(base, differential_attention=1)
        with self.assertRaises(ValueError):
            replace(base, d_model=12, n_heads=2, n_kv_heads=2, differential_attention=True)

    def test_causality_cache_padding_gradients_and_checkpoint(self):
        torch.manual_seed(2025)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         differential_attention=True, stable_rms_norm=True,
                         gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertGreater(info["parameters"], inspect_config(BlueConfig.preset("tiny"))["parameters"])
        model = Blue(config).eval()
        batch = synthetic_batch(config, batch_size=2, seed=2026)

        # Causality check.
        ids = torch.randint(4, 260, (2, 42))
        changed = ids.clone()
        changed[:, 12:] = torch.randint(4, 260, changed[:, 12:].shape)
        with torch.no_grad():
            first = model(ids, images=batch["images"], series=batch["series"]).logits
            second = model(changed, images=batch["images"], series=batch["series"]).logits
        torch.testing.assert_close(first[:, :12], second[:, :12], rtol=1e-5, atol=1e-6)

        # Sliding-window KV cache equivalence.
        with torch.no_grad():
            full = model(ids, images=batch["images"], series=batch["series"]).logits
            p1 = model(ids[:, :34], images=batch["images"], series=batch["series"], use_cache=True)
            p2 = model(ids[:, 34:39], cache=p1.cache, use_cache=True)
            p3 = model(ids[:, 39:], cache=p2.cache, use_cache=True)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits), 1), full, rtol=2e-5, atol=2e-6)

        # Gradients reach differential lambda parameters and sub_norm under bf16 + checkpointing.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for block in model.blocks:
            for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
                grad = getattr(block.attn, name).grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0.0)
            self.assertGreater(block.attn.sub_norm.weight.grad.abs().sum().item(), 0.0)
            self.assertTrue(block.attn.sub_norm.stable)

        # Checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diff.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertTrue(restored.config.differential_attention)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
