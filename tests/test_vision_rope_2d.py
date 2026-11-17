from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.modules import RotaryEmbedding2D
from blue.training import load_checkpoint, save_checkpoint


class VisionRope2DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        with self.assertRaises(ValueError):
            replace(base, vision_rope_2d=1)
        # tiny has head_dim=16; MLA + diff -> rope_dim=4 (valid), head_dim=8 + diff -> rope_dim=4 (valid),
        # head_dim=8 + MLA + diff -> rope_dim=2 (invalid for 2D RoPE).
        with self.assertRaises(ValueError):
            replace(base, d_model=32, n_heads=4, n_kv_heads=2, kv_lora_rank=8,
                    differential_attention=True, vision_rope_2d=True)

    def test_2d_relative_translation_invariance_of_inner_products(self):
        rope = RotaryEmbedding2D(head_dim=16, theta=10000.0, grid_size=4)
        torch.manual_seed(7001)
        q_vec = torch.randn(16)
        k_vec = torch.randn(16)
        q_grid = q_vec.view(1, 1, 1, 16).expand(1, 1, 16, 16).clone()
        k_grid = k_vec.view(1, 1, 1, 16).expand(1, 1, 16, 16).clone()
        rq = rope(q_grid)[0, 0]  # [16, 16]
        rk = rope(k_grid)[0, 0]  # [16, 16]
        # Pair ((r=0, c=1), (r=1, c=2)) and ((r=2, c=0), (r=3, c=1)) share delta = (+1, +1).
        dot_a = (rq[0 * 4 + 1] * rk[1 * 4 + 2]).sum()
        dot_b = (rq[2 * 4 + 0] * rk[3 * 4 + 1]).sum()
        torch.testing.assert_close(dot_a, dot_b, rtol=1e-5, atol=1e-6)
        # Vertical vs horizontal delta (+1, 0) vs (0, +1) are distinct.
        dot_v = (rq[0 * 4 + 0] * rk[1 * 4 + 0]).sum()
        dot_h = (rq[0 * 4 + 0] * rk[0 * 4 + 1]).sum()
        self.assertGreater((dot_v - dot_h).abs().item(), 1e-4)

    def test_vision_rope_2d_with_mla_diff_bf16_and_checkpoint_roundtrip(self):
        torch.manual_seed(7002)
        config = replace(BlueConfig.preset("tiny"), vision_rope_2d=True,
                         differential_attention=True, kv_lora_rank=16,
                         gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertGreater(info["parameters"], 0)
        model = Blue(config)
        self.assertIsInstance(model.vision.blocks[0].attn.rope, RotaryEmbedding2D)
        batch = synthetic_batch(config, batch_size=2, seed=7003)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertGreater(model.vision.blocks[0].attn.q.weight.grad.abs().sum().item(), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v2d.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertIsInstance(restored.vision.blocks[0].attn.rope, RotaryEmbedding2D)
            with torch.no_grad():
                torch.testing.assert_close(model(images=batch["images"]).classification_logits,
                                           restored(images=batch["images"]).classification_logits,
                                           rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
