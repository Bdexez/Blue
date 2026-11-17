from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class SpectralTemporalMixingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            replace(BlueConfig.preset("tiny"), temporal_spectral_mixing=1)

    def test_padding_translation_invariance_length1_bf16_gradients_and_checkpoint(self):
        torch.manual_seed(8001)
        config = replace(BlueConfig.preset("tiny"),
                         temporal_spectral_mixing=True,
                         temporal_statistics=True,
                         temporal_statistics_relative=True,
                         stable_rms_norm=True,
                         gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertGreater(info["parameters"], inspect_config(BlueConfig.preset("tiny"))["parameters"])
        model = Blue(config).eval()
        self.assertTrue(model.temporal.spectral.norm.stable)
        batch = synthetic_batch(config, batch_size=2, seed=8002)
        series = batch["series"][:, -16:].clone()
        series[0, 3, 1] = float("nan")
        valid = torch.ones(series.shape[:2], dtype=torch.bool)
        valid[1, 5] = False

        # Exact padding invariance (both left & right padding) and translation invariance.
        with torch.no_grad():
            base = model(batch["input_ids"], series=series, series_mask=valid)
            padded = model(
                batch["input_ids"],
                series=torch.nn.functional.pad(series, (0, 0, 3, 5), value=9999.0),
                series_mask=torch.nn.functional.pad(valid, (3, 5), value=False),
            )
            shifted = model(batch["input_ids"], series=series + 25.0, series_mask=valid)
            # Single-step history (L=1) edge case.
            single = model(series=series[:, :1], series_mask=torch.ones(2, 1, dtype=torch.bool))
        torch.testing.assert_close(base.forecast, padded.forecast, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(base.classification_logits, padded.classification_logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(base.classification_logits, shifted.classification_logits, rtol=1e-4, atol=1e-5)
        self.assertTrue(torch.isfinite(single.forecast).all())

        # Gradients reach spectral parameters under bf16 autocast.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for name, param in model.temporal.spectral.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(param.grad.abs().sum().item(), 0.0, name)

        # Checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectral.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertTrue(restored.config.temporal_spectral_mixing)
            with torch.no_grad():
                torch.testing.assert_close(model(series=series, series_mask=valid).forecast,
                                           restored(series=series, series_mask=valid).forecast,
                                           rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
