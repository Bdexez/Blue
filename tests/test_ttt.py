from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint
from blue.ttt import _inner_layernorm_and_grad


class TTTLinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation_and_analytical_layernorm_grad(self):
        base = BlueConfig.preset("tiny")
        for bad in (
            dict(ttt_state_dim=-1),
            dict(ttt_state_dim=True),
            dict(ttt_heads=2),
            dict(ttt_state_dim=8, ttt_heads=0),
            dict(ttt_base_lr=0.0),
            dict(ttt_base_lr=-0.1),
            dict(ttt_base_lr=float("nan")),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, **bad)

        # Verify analytical inner LayerNorm gradient matches torch.autograd.grad to 1e-6!
        torch.manual_seed(0)
        z = torch.randn(3, 4, 12, requires_grad=True)
        target = torch.randn(3, 4, 12)
        gamma = torch.randn(1, 4, 12) + 1.0
        beta = torch.randn(1, 4, 12) * 0.1
        y_ana, grad_ana = _inner_layernorm_and_grad(z.detach(), target, gamma, beta)
        mean = z.mean(-1, keepdim=True)
        var = (z - mean).square().mean(-1, keepdim=True)
        y_auto = ((z - mean) * torch.rsqrt(var + 1e-5)) * gamma + beta
        loss = 0.5 * (y_auto - target).square().sum()
        (grad_auto,) = torch.autograd.grad(loss, z)
        torch.testing.assert_close(y_ana, y_auto.detach(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(grad_ana, grad_auto, rtol=1e-5, atol=1e-6)

    def test_quad_paradigm_causality_cache_padding_gradients_and_checkpoint(self):
        torch.manual_seed(555)
        config = replace(
            BlueConfig.preset("tiny"),
            max_seq_len=96,
            ssm_state_dim=8, ssm_heads=4,
            ttt_state_dim=8, ttt_heads=4, ttt_base_lr=0.4,
            differential_attention=True, kv_lora_rank=16,
            residual_streams=2,
            stable_rms_norm=True, gradient_checkpointing=True,
        )
        model = Blue(config).eval()
        # Verify meta-parameter initialization survives Blue.__init__.
        torch.testing.assert_close(model.blocks[0].ttt.lr_proj.bias, torch.full((4,), -2.0))
        torch.testing.assert_close(model.blocks[0].ttt.ln_weight, torch.ones((4, 8)))

        batch = synthetic_batch(config, batch_size=2, seed=556)
        ids = torch.randint(4, 260, (2, 24))

        # 1. Strict causality.
        changed = ids.clone()
        changed[:, 10:] = torch.randint(4, 260, changed[:, 10:].shape)
        with torch.no_grad():
            first = model(ids, images=batch["images"], series=batch["series"]).logits
            second = model(changed, images=batch["images"], series=batch["series"]).logits
        torch.testing.assert_close(first[:, :10], second[:, :10], rtol=1e-5, atol=1e-6)

        # 2. Exact O(1) TTT fast-weight cache equivalence across prefill + multi-step decode.
        with torch.no_grad():
            full = model(ids, images=batch["images"], series=batch["series"]).logits
            p1 = model(ids[:, :14], images=batch["images"], series=batch["series"], use_cache=True)
            p2 = model(ids[:, 14:20], cache=p1.cache, use_cache=True)
            p3 = model(ids[:, 20:], cache=p2.cache, use_cache=True)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits), 1), full, rtol=2e-5, atol=2e-5)
        self.assertEqual(p3.cache.layers[0].ttt_state.shape, (2, 4, 9, 8))

        # 3. Exact padding invariance (masked positions freeze W_t, b_t).
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[0, :3] = False
        mask[1, -5:] = False
        with torch.no_grad():
            full_m = model(ids, attention_mask=mask).logits
            m1 = model(ids[:, :12], attention_mask=mask[:, :12], use_cache=True)
            m2 = model(ids[:, 12:], attention_mask=mask[:, 12:], cache=m1.cache, use_cache=True)
        torch.testing.assert_close(m2.logits, full_m[:, 12:], rtol=2e-5, atol=2e-5)

        # 4. Outer gradients flow into meta-parameters (W0, b0, ln_weight, ln_bias, lr_proj) under bf16 + checkpointing.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for block in model.blocks:
            for name, param in block.ttt.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
                self.assertGreater(param.grad.abs().sum().item(), 0.0, name)
            self.assertGreater(block.ttt_gate.grad.abs().sum().item(), 0.0)

        # 5. Checkpoint roundtrip + greedy generation.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ttt.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)
            gen = restored.generate(ids[:1, :6], max_new_tokens=4, temperature=0)
            self.assertEqual(gen.shape, (1, 10))


if __name__ == "__main__":
    unittest.main()
