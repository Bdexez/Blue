from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.modules import RMSNorm
from blue.training import load_checkpoint, save_checkpoint


class DynamicTanhNormTests(unittest.TestCase):
    def test_dyt_without_reduction_gradients_cache_and_checkpoint(self):
        base = BlueConfig.preset("tiny")
        self.assertFalse(base.dynamic_tanh_norm)
        m_default = Blue(base)
        self.assertIsNone(m_default.norm.dyt_alpha)
        with self.assertRaises(ValueError):
            replace(base, dynamic_tanh_norm=1)

        torch.manual_seed(490)
        cfg = replace(base, dynamic_tanh_norm=True)
        model = Blue(cfg)
        norms = [m for m in model.modules() if isinstance(m, RMSNorm)]
        self.assertGreater(len(norms), 5)
        for n in norms:
            self.assertIsNotNone(n.dyt_alpha)
            self.assertAlmostEqual(n.dyt_alpha.item(), 0.5, places=6)

        # Extreme finite inputs remain strictly bounded in [-1, 1] * weight without overflow.
        extreme = torch.full((2, 4, cfg.d_model), 1e12)
        out_ext = model.norm(extreme)
        self.assertTrue(torch.isfinite(out_ext).all())
        self.assertLessEqual(out_ext.abs().max().item(), 1.0 + 1e-6)

        batch = synthetic_batch(cfg, batch_size=2, seed=491)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertIsNotNone(model.norm.dyt_alpha.grad)
        self.assertGreater(model.norm.dyt_alpha.grad.abs().item(), 0.0)

        model.eval()
        ids = torch.randint(4, 260, (2, 16))
        with torch.no_grad():
            full = model(ids).logits
            p1 = model(ids[:, :9], use_cache=True)
            p2 = model(ids[:, 9:], cache=p1.cache, use_cache=True)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits), 1), full, rtol=1e-5, atol=1e-5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dyt.pt"
            save_checkpoint(path, model)
            restored, _ = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
