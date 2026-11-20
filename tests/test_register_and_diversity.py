from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class RegisterLatentsAndDiversityTests(unittest.TestCase):
    def test_register_latents_and_orthogonal_diversity_regularization(self):
        base = BlueConfig.preset("tiny")
        self.assertEqual(base.register_latents, 0)
        self.assertEqual(base.latent_diversity_weight, 0.0)
        for bad in (
            dict(register_latents=-1),
            dict(register_latents=True),
            dict(latent_diversity_weight=-0.1),
            dict(latent_diversity_weight=float("nan")),
        ):
            with self.assertRaises(ValueError):
                replace(base, **bad)

        torch.manual_seed(500)
        cfg = replace(
            base,
            n_latents=8, register_latents=4,
            hierarchical_resampler=True,
            latent_diversity_weight=0.05,
        )
        model = Blue(cfg)
        self.assertEqual(model.resampler.registers.shape, (1, 4, cfg.d_model))

        batch = synthetic_batch(cfg, batch_size=2, seed=501)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertIn("latent_diversity", out.losses)
        self.assertGreaterEqual(out.losses["latent_diversity"].item(), 0.0)
        self.assertLessEqual(out.losses["latent_diversity"].item(), 1.0)
        out.loss.backward()
        self.assertIsNotNone(model.resampler.registers.grad)
        self.assertGreater(model.resampler.registers.grad.abs().sum().item(), 0.0)

        # Register latents are discarded before decoder: cached latents still have shape [B, n_latents, D]!
        model.eval()
        inputs = {k: v for k, v in batch.items() if k not in ("labels", "class_labels", "forecast_targets")}
        with torch.no_grad():
            full = model(**inputs).logits
            p1 = model(input_ids=batch["input_ids"][:, :-3], images=batch["images"],
                       series=batch["series"], series_mask=batch["series_mask"], use_cache=True)
            p2 = model(input_ids=batch["input_ids"][:, -3:], cache=p1.cache, use_cache=True)
        self.assertEqual(p1.cache.latents.shape, (2, 8, cfg.d_model))
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits), 1), full, rtol=1e-5, atol=1e-5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reg_div.pt"
            save_checkpoint(path, model)
            restored, _ = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(model(**inputs).logits, restored(**inputs).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
