from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class CrossMLATests(unittest.TestCase):
    def test_config_validation_compression_cache_and_differential(self):
        base = BlueConfig.preset("tiny")
        self.assertEqual(base.cross_kv_lora_rank, 0)
        for bad in (-1, True):
            with self.assertRaises(ValueError):
                replace(base, cross_kv_lora_rank=bad)

        for diff in (False, True):
            with self.subTest(cross_differential_attention=diff):
                torch.manual_seed(410 + int(diff))
                cfg = replace(
                    base,
                    cross_kv_lora_rank=16,
                    cross_differential_attention=diff,
                    hierarchical_resampler=True,
                )
                model = Blue(cfg).eval()
                batch = synthetic_batch(cfg, batch_size=2, seed=411)
                inputs = {k: v for k, v in batch.items() if k not in ("labels", "class_labels", "forecast_targets")}

                # Exact cache equivalence + compressed context_key [B, n_latents, 16] and context_value=None.
                with torch.no_grad():
                    full = model(**inputs).logits
                    p1 = model(input_ids=batch["input_ids"][:, :-3], images=batch["images"],
                               series=batch["series"], series_mask=batch["series_mask"], use_cache=True)
                    p2 = model(input_ids=batch["input_ids"][:, -3:], cache=p1.cache, use_cache=True)
                self.assertEqual(p1.cache.layers[0].context_key.shape, (2, cfg.n_latents, 16))
                self.assertIsNone(p1.cache.layers[0].context_value)
                torch.testing.assert_close(torch.cat((p1.logits, p2.logits), 1), full, rtol=1e-5, atol=1e-5)

                # BF16 backward reaches kv_down, k_up, v_up.
                model.train()
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out = model(**batch)
                self.assertTrue(torch.isfinite(out.loss))
                out.loss.backward()
                self.assertGreater(model.blocks[0].cross.kv_down.weight.grad.abs().sum().item(), 0.0)
                self.assertGreater(model.blocks[0].cross.k_up.weight.grad.abs().sum().item(), 0.0)
                self.assertGreater(model.blocks[0].cross.v_up.weight.grad.abs().sum().item(), 0.0)

                if diff:
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "cross_mla.pt"
                        save_checkpoint(path, model.eval())
                        restored, _ = load_checkpoint(path)
                        with torch.no_grad():
                            torch.testing.assert_close(model(**inputs).logits, restored(**inputs).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
