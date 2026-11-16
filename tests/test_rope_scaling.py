from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.modules import RotaryEmbedding
from blue.training import load_checkpoint, save_checkpoint


class RopeScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        for bad in (
            dict(rope_scaling_type="invalid"),
            dict(rope_scaling_factor=0.5),
            dict(rope_scaling_factor=float("nan")),
            dict(rope_scaling_type="none", rope_scaling_factor=2.0),
            dict(rope_scaling_type="linear", rope_scaling_factor=1.0),
            dict(rope_scaling_type="yarn", rope_scaling_factor=2.0, rope_original_max_seq_len=0),
            dict(rope_scaling_type="yarn", rope_scaling_factor=2.0, rope_original_max_seq_len=9999),
            dict(rope_scaling_type="ntk", rope_scaling_factor=2.0, rope_original_max_seq_len=128),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, **bad)

    def test_frequency_properties_linear_ntk_and_yarn(self):
        base = RotaryEmbedding(16, 10000.0)
        linear = RotaryEmbedding(16, 10000.0, scaling_type="linear", scaling_factor=4.0)
        ntk = RotaryEmbedding(16, 10000.0, scaling_type="ntk", scaling_factor=4.0)
        yarn = RotaryEmbedding(16, 10000.0, scaling_type="yarn", scaling_factor=4.0,
                               original_max_seq_len=256, max_seq_len=1024)
        torch.testing.assert_close(linear.inv_freq, base.inv_freq / 4.0)
        self.assertTrue((ntk.inv_freq <= base.inv_freq).all())
        self.assertTrue((yarn.inv_freq >= base.inv_freq / 4.0 - 1e-7).all())
        self.assertTrue((yarn.inv_freq <= base.inv_freq + 1e-7).all())
        self.assertAlmostEqual(yarn.mscale, 0.1 * math.log(4.0) + 1.0, places=6)
        # Verify mscale scales vector norms identically across positions.
        x = torch.randn(2, 4, 10, 16)
        y = yarn(x, offset=5)
        torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1) * yarn.mscale, rtol=1e-5, atol=1e-6)

    def test_cache_equivalence_bf16_and_checkpoint_override(self):
        for scaling in ("linear", "ntk", "yarn"):
            with self.subTest(scaling=scaling):
                torch.manual_seed(901)
                extra = dict(rope_original_max_seq_len=128) if scaling == "yarn" else {}
                config = replace(BlueConfig.preset("tiny"), max_seq_len=256,
                                 rope_scaling_type=scaling, rope_scaling_factor=2.0, **extra)
                model = Blue(config).eval()
                ids = torch.randint(4, 260, (2, 45))
                batch = synthetic_batch(config, batch_size=2, seed=902)
                with torch.no_grad():
                    full = model(ids, images=batch["images"], series=batch["series"]).logits
                    p1 = model(ids[:, :30], images=batch["images"], series=batch["series"], use_cache=True)
                    p2 = model(ids[:, 30:38], cache=p1.cache, use_cache=True)
                    p3 = model(ids[:, 38:], cache=p2.cache, use_cache=True)
                joined = torch.cat((p1.logits, p2.logits, p3.logits), dim=1)
                torch.testing.assert_close(joined, full, rtol=2e-5, atol=2e-6)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = model.train()(**batch).loss
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
        # Override an existing standard checkpoint to YaRN without altering weights.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.pt"
            base_model = Blue(BlueConfig.preset("tiny"))
            save_checkpoint(path, base_model)
            extended, _ = load_checkpoint(path, overrides={"rope_scaling_type": "yarn", "rope_scaling_factor": 2.0})
            self.assertEqual(extended.config.rope_scaling_type, "yarn")
            self.assertGreater(extended.blocks[0].attn.rope.mscale, 1.0)
            self.assertFalse(torch.allclose(extended.blocks[0].attn.rope.inv_freq,
                                            base_model.blocks[0].attn.rope.inv_freq))


if __name__ == "__main__":
    unittest.main()
