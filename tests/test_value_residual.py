from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class ValueResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation_and_backward_compat(self):
        default = BlueConfig.preset("tiny")
        self.assertFalse(default.value_residual)
        m = Blue(default)
        self.assertIsNone(m.blocks[1].attn.v_res_gate)
        with self.assertRaises(ValueError):
            replace(default, value_residual=1)

    def test_all_four_attention_modes_causality_cache_and_gradients(self):
        base = BlueConfig.preset("tiny")
        modes = (
            dict(differential_attention=False, kv_lora_rank=0),
            dict(differential_attention=True, kv_lora_rank=0),
            dict(differential_attention=False, kv_lora_rank=16),
            dict(differential_attention=True, kv_lora_rank=16),
        )
        for idx, mode in enumerate(modes):
            with self.subTest(mode=mode):
                torch.manual_seed(700 + idx)
                cfg = replace(
                    base,
                    d_model=64, n_layers=3, n_heads=4, n_kv_heads=2,
                    sliding_window=8, global_every=2,
                    attention_sinks=2, value_residual=True,
                    latent_reasoning_steps=2,
                    gradient_checkpointing=True,
                    **mode,
                )
                model = Blue(cfg).eval()
                self.assertIsNone(model.blocks[0].attn.v_res_gate)
                self.assertIsNotNone(model.blocks[1].attn.v_res_gate)
                torch.testing.assert_close(
                    model.blocks[1].attn.v_res_gate,
                    torch.full_like(model.blocks[1].attn.v_res_gate, -2.0),
                )

                ids = torch.randint(4, 260, (2, 18))
                changed = ids.clone()
                changed[:, 9:] = torch.randint(4, 260, changed[:, 9:].shape)
                with torch.no_grad():
                    first = model(ids).logits
                    second = model(changed).logits
                torch.testing.assert_close(first[:, :9], second[:, :9], rtol=1e-5, atol=1e-6)

                # Exact cache equivalence across sliding window + global layers + multi-step reasoning.
                with torch.no_grad():
                    p1 = model(ids[:, :11], use_cache=True)
                    p2 = model(ids[:, 11:15], cache=p1.cache, use_cache=True)
                    p3 = model(ids[:, 15:], cache=p2.cache, use_cache=True)
                torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits), 1), first, rtol=2e-5, atol=2e-5)

                # BF16 + gradient checkpointing flows gradients into v_res_gate.
                model.train()
                batch = synthetic_batch(cfg, batch_size=2, seed=710 + idx)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out = model(**batch)
                self.assertTrue(torch.isfinite(out.loss))
                out.loss.backward()
                for layer_idx in range(1, cfg.n_layers):
                    grad = model.blocks[layer_idx].attn.v_res_gate.grad
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0.0)

                if idx == 3:
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "vres.pt"
                        save_checkpoint(path, model.eval())
                        restored, _ = load_checkpoint(path)
                        with torch.no_grad():
                            torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
