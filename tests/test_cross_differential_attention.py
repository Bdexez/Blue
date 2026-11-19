from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class CrossDifferentialAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        with self.assertRaises(ValueError):
            replace(base, cross_differential_attention=1)
        with self.assertRaises(ValueError):
            replace(base, d_model=12, n_heads=2, n_kv_heads=2, cross_differential_attention=True)

    def test_cross_differential_cache_context_projection_once_and_gradients(self):
        torch.manual_seed(24001)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         cross_differential_attention=True, hierarchical_resampler=True,
                         stable_rms_norm=True, gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertTrue(info["architecture_features"]["cross_differential_attention"])
        model = Blue(config).eval()
        self.assertTrue(model.blocks[0].cross.sub_norm.stable)
        self.assertTrue(model.resampler.local_cross.sub_norm.stable)
        batch = synthetic_batch(config, batch_size=2, seed=24002)
        ids = torch.randint(4, 260, (2, 40))

        # Context projection happens once on prefill and is reused from LayerCache on decode steps.
        proj_calls = 0
        original_proj = model.blocks[0].cross.project_context

        def counting_proj(ctx):
            nonlocal proj_calls
            proj_calls += 1
            return original_proj(ctx)

        model.blocks[0].cross.project_context = counting_proj
        try:
            with torch.no_grad():
                p1 = model(ids[:, :34], images=batch["images"], series=batch["series"], use_cache=True)
                p2 = model(ids[:, 34:38], cache=p1.cache, use_cache=True)
                p3 = model(ids[:, 38:], cache=p2.cache, use_cache=True)
        finally:
            model.blocks[0].cross.project_context = original_proj
        self.assertEqual(proj_calls, 1)

        with torch.no_grad():
            full = model(ids, images=batch["images"], series=batch["series"]).logits
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits), 1), full, rtol=3e-5, atol=3e-6)

        # Gradients reach cross-attention lambda parameters and sub_norm under bf16.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for cross in (model.blocks[0].cross, model.resampler.cross[0], model.resampler.local_cross):
            for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
                self.assertGreater(getattr(cross, name).grad.abs().sum().item(), 0.0, name)
            self.assertGreater(cross.sub_norm.weight.grad.abs().sum().item(), 0.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cross_diff.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertTrue(restored.config.cross_differential_attention)
            with torch.no_grad():
                torch.testing.assert_close(model(ids, images=batch["images"]).logits,
                                           restored(ids, images=batch["images"]).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
