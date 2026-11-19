from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class MoENullExpertsTests(unittest.TestCase):
    def test_config_validation_and_zero_compute_routing(self):
        base = BlueConfig.preset("tiny")
        self.assertEqual(base.moe_null_experts, 0)
        for bad in (-1, True):
            with self.assertRaises(ValueError):
                replace(base, moe_null_experts=bad)

        torch.manual_seed(400)
        cfg = replace(
            base,
            n_experts=4, top_k=2, moe_null_experts=2,
            shared_expert_gate=True, router_bias_update_rate=0.01,
            depth_routing_capacity=0.75,
        )
        model = Blue(cfg)
        self.assertEqual(model.blocks[0].moe.router.out_features, 6)
        self.assertEqual(len(model.blocks[0].moe.experts), 4)

        batch = synthetic_batch(cfg, batch_size=2, seed=401)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        self.assertEqual(out.expert_usage.shape, (cfg.n_layers, cfg.n_experts))
        # At init, some tokens naturally route to null slots.
        null_rates = [b.moe.last_null_fraction for b in model.blocks]
        self.assertTrue(any(r > 0.0 for r in null_rates))
        out.loss.backward()
        self.assertGreater(model.blocks[0].moe.router.weight.grad.abs().sum().item(), 0.0)
        model.update_router_biases(out.expert_usage)

        # Strict causality & cache equivalence.
        model.eval()
        ids = torch.randint(4, 260, (2, 16))
        changed = ids.clone()
        changed[:, 8:] = torch.randint(4, 260, changed[:, 8:].shape)
        with torch.no_grad():
            o1 = model(ids).logits
            o2 = model(changed).logits
            p1 = model(ids[:, :10], use_cache=True)
            p2 = model(ids[:, 10:], cache=p1.cache, use_cache=True)
        torch.testing.assert_close(o1[:, :8], o2[:, :8], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits), 1), o1, rtol=1e-5, atol=1e-5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "null_moe.pt"
            save_checkpoint(path, model)
            restored, _ = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
