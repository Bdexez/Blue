from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class MultimodalAdaLNTests(unittest.TestCase):
    def test_zero_init_identity_gradients_cache_and_checkpoint(self):
        base = BlueConfig.preset("tiny")
        self.assertFalse(base.multimodal_adaln)
        with self.assertRaises(ValueError):
            replace(base, multimodal_adaln=1)

        torch.manual_seed(460)
        cfg = replace(
            base,
            multimodal_adaln=True,
            residual_streams=2,
            depth_routing_capacity=0.75,
            gradient_checkpointing=True,
        )
        model = Blue(cfg)
        for block in model.blocks:
            self.assertEqual(block.adaln_mod.weight.abs().sum().item(), 0.0)
            self.assertEqual(block.adaln_mod.bias.abs().sum().item(), 0.0)

        batch = synthetic_batch(cfg, batch_size=2, seed=461)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for block in model.blocks:
            self.assertGreater(block.adaln_mod.weight.grad.abs().sum().item(), 0.0)
            self.assertGreater(block.adaln_mod.bias.grad.abs().sum().item(), 0.0)

        # After a gradient step, modulation is active and cache equivalence remains exact!
        with torch.no_grad():
            for block in model.blocks:
                block.adaln_mod.weight.add_(torch.randn_like(block.adaln_mod.weight) * 0.02)
        model.eval()
        inputs = {k: v for k, v in batch.items() if k not in ("labels", "class_labels", "forecast_targets")}
        with torch.no_grad():
            full = model(**inputs).logits
            p1 = model(input_ids=batch["input_ids"][:, :-3], images=batch["images"],
                       series=batch["series"], series_mask=batch["series_mask"], use_cache=True)
            p2 = model(input_ids=batch["input_ids"][:, -3:], cache=p1.cache, use_cache=True)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits), 1), full, rtol=1e-5, atol=1e-5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adaln.pt"
            save_checkpoint(path, model)
            restored, _ = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(model(**inputs).logits, restored(**inputs).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
