from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class HyperbolicMoERouterTests(unittest.TestCase):
    def test_hyperbolic_poincare_router_validation_causality_cache_and_gradients(self):
        base = BlueConfig.preset("tiny")
        self.assertEqual(base.moe_router_geometry, "euclidean")
        with self.assertRaises(ValueError):
            replace(base, moe_router_geometry="spherical")

        torch.manual_seed(450)
        cfg = replace(
            base,
            n_experts=4, top_k=2, moe_null_experts=2,
            moe_router_geometry="hyperbolic", alignment_curvature=1.5,
            router_bias_update_rate=0.01, shared_expert_gate=True,
        )
        model = Blue(cfg)
        batch = synthetic_batch(cfg, batch_size=2, seed=451)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for block in model.blocks:
            grad = block.moe.router.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0.0)

        model.eval()
        ids = torch.randint(4, 260, (2, 16))
        with torch.no_grad():
            full = model(ids).logits
            p1 = model(ids[:, :9], use_cache=True)
            p2 = model(ids[:, 9:], cache=p1.cache, use_cache=True)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits), 1), full, rtol=1e-5, atol=1e-5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hmoe.pt"
            save_checkpoint(path, model)
            restored, _ = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
