from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class MultimodalJEPATests(unittest.TestCase):
    def test_config_validation_and_backward_compat(self):
        default = BlueConfig.preset("tiny")
        self.assertEqual(default.jepa_weight, 0.0)
        self.assertIsNone(Blue(default).jepa)
        for bad in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                replace(default, jepa_weight=bad)

    def test_jepa_self_supervised_multimodal_prediction_and_training(self):
        torch.manual_seed(360)
        cfg = replace(
            BlueConfig.preset("tiny"),
            audio_enabled=True, audio_layers=1, max_audio_len=64,
            tabular_enabled=True, tabular_num_features=6, tabular_cat_features=2,
            tabular_cat_cardinality=8, tabular_layers=1, tabular_plr_frequencies=4,
            jepa_weight=0.2,
        )
        model = Blue(cfg)
        batch = synthetic_batch(cfg, batch_size=3, seed=361)

        # 1. Self-supervised mode (no external labels needed to compute JEPA loss!).
        unlabeled = {k: v for k, v in batch.items() if k not in ("labels", "class_labels", "forecast_targets")}
        out_unlab = model(**unlabeled)
        self.assertIsNone(out_unlab.loss)
        self.assertIn("jepa", out_unlab.losses)
        self.assertTrue(torch.isfinite(out_unlab.losses["jepa"]))

        # 2. Single-modality input gracefully skips JEPA.
        out_single = model(images=batch["images"])
        self.assertNotIn("jepa", out_single.losses)

        # 3. Full supervised + JEPA forward & backward under BF16; JEPA loss decreases over steps.
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        losses = []
        for _ in range(10):
            opt.zero_grad()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = model(**batch)
            self.assertTrue(torch.isfinite(out.loss))
            out.loss.backward()
            self.assertGreater(model.jepa.in_proj.weight.grad.abs().sum().item(), 0.0)
            self.assertGreater(model.jepa.target_embed.weight.grad.abs().sum().item(), 0.0)
            opt.step()
            losses.append(out.losses["jepa"].item())
        self.assertLess(losses[-1], losses[0])

        # 4. Checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jepa.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            with torch.no_grad():
                o1 = model(**batch).losses["jepa"]
                o2 = restored(**batch).losses["jepa"]
            torch.testing.assert_close(o1, o2, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
