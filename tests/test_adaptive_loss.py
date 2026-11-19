from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.optimizer import build_optimizer
from blue.training import load_checkpoint, save_checkpoint


class AdaptiveLossWeightingTests(unittest.TestCase):
    def test_config_validation_and_strict_positivity_and_adaptation(self):
        default = BlueConfig.preset("tiny")
        self.assertFalse(default.adaptive_loss_weighting)
        self.assertIsNone(Blue(default).task_log_vars)
        with self.assertRaises(ValueError):
            replace(default, adaptive_loss_weighting=1)

        torch.manual_seed(370)
        cfg = replace(
            default,
            flow_matching_dim=8, jepa_weight=0.1,
            adaptive_loss_weighting=True,
        )
        model = Blue(cfg)
        torch.testing.assert_close(model.task_log_vars, torch.zeros(7))

        batch = synthetic_batch(cfg, batch_size=3, seed=371)
        batch["flow_targets"] = torch.randn(3, 8)
        opt = build_optimizer(model, name="muon", lr=5e-3, weight_decay=0.01)

        initial_lv = model.task_log_vars.detach().clone()
        for _ in range(5):
            opt.zero_grad()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = model(**batch)
            self.assertTrue(torch.isfinite(out.loss))
            # Strictly positive even under extreme log-variance values!
            self.assertGreater(out.loss.item(), 0.0)
            out.loss.backward()
            self.assertIsNotNone(model.task_log_vars.grad)
            self.assertTrue(torch.isfinite(model.task_log_vars.grad).all())
            opt.step()

        # All 7 task log-variances updated!
        self.assertTrue((model.task_log_vars.detach() != initial_lv).all())

        # Strict positivity holds even at extreme clamped bounds [-4, +4].
        with torch.no_grad():
            model.task_log_vars.fill_(-10.0)
            out_lo = model(**batch)
            model.task_log_vars.fill_(10.0)
            out_hi = model(**batch)
        self.assertGreater(out_lo.loss.item(), 0.0)
        self.assertGreater(out_hi.loss.item(), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adaptive.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            torch.testing.assert_close(model.task_log_vars, restored.task_log_vars, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
