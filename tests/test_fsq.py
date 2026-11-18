from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class FiniteScalarQuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        for bad in ((1, 4), (0, 3), (-2,), (True,)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, latent_fsq_levels=bad)

    def test_fsq_discrete_codes_ste_gradients_bf16_and_checkpoint(self):
        torch.manual_seed(22001)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         latent_fsq_levels=(8, 5, 5), hierarchical_resampler=True,
                         stable_rms_norm=True)
        info = inspect_config(config)
        self.assertEqual(info["architecture_features"]["latent_fsq_levels"], [8, 5, 5])
        model = Blue(config)
        self.assertEqual(model.resampler.fsq.codebook_size, 200)
        self.assertTrue(model.resampler.fsq.norm.stable)
        batch = synthetic_batch(config, batch_size=3, seed=22002)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        codes = model.resampler.last_codes
        self.assertIsNotNone(codes)
        self.assertEqual(codes.shape, (3, config.n_latents))
        self.assertEqual(codes.dtype, torch.long)
        self.assertTrue(((codes >= 0) & (codes < 200)).all())
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        # STE transmits non-zero gradients through rounding into down, up, gate, and vision encoder.
        self.assertGreater(model.resampler.fsq.down.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.resampler.fsq.up.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.resampler.fsq.gate.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.vision.stem.weight.grad.abs().sum().item(), 0.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fsq.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertEqual(restored.config.latent_fsq_levels, (8, 5, 5))
            with torch.no_grad():
                torch.testing.assert_close(model(images=batch["images"]).classification_logits,
                                           restored(images=batch["images"]).classification_logits,
                                           rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
