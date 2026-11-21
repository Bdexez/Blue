from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.modules import SwiGLU
from blue.training import load_checkpoint, save_checkpoint


class KANSwiGLUTests(unittest.TestCase):
    def test_chebyshev_kan_swiglu_zero_init_bounds_gradients_and_checkpoint(self):
        base = BlueConfig.preset("tiny")
        self.assertEqual(base.kan_degree, 0)
        for bad in (-1, 1, 9, True):
            with self.assertRaises(ValueError):
                replace(base, kan_degree=bad)

        torch.manual_seed(540)
        cfg_std = base
        torch.manual_seed(540)
        cfg_kan = replace(base, kan_degree=4)
        m_std = Blue(cfg_std).eval()
        torch.manual_seed(540)
        m_kan = Blue(cfg_kan).eval()

        # At initialization, kan_coeffs == 0 so KAN-SwiGLU matches standard SwiGLU identically!
        ids = torch.randint(4, 260, (2, 14))
        with torch.no_grad():
            torch.testing.assert_close(m_std(ids).logits, m_kan(ids).logits, rtol=1e-6, atol=1e-6)

        # Gradients flow immediately into all Chebyshev orders (T_2, T_3, T_4) on step 0!
        m_kan.train()
        batch = synthetic_batch(cfg_kan, batch_size=2, seed=541)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = m_kan(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        kan_modules = [m for m in m_kan.modules() if isinstance(m, SwiGLU)]
        self.assertGreater(len(kan_modules), 5)
        self.assertTrue(any(m.kan_coeffs.grad is not None and m.kan_coeffs.grad.abs().sum().item() > 0 for m in kan_modules))

        # Perturb kan_coeffs and verify exact cache equivalence + checkpoint roundtrip.
        with torch.no_grad():
            for m in kan_modules:
                m.kan_coeffs.add_(torch.randn_like(m.kan_coeffs) * 0.05)
        m_kan.eval()
        with torch.no_grad():
            full = m_kan(ids).logits
            p1 = m_kan(ids[:, :8], use_cache=True)
            p2 = m_kan(ids[:, 8:], cache=p1.cache, use_cache=True)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits), 1), full, rtol=1e-5, atol=1e-5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kan.pt"
            save_checkpoint(path, m_kan)
            restored, _ = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(m_kan(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
