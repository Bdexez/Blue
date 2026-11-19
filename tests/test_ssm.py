from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class GatedDeltaNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        for bad in (
            dict(ssm_state_dim=-1),
            dict(ssm_state_dim=True),
            dict(ssm_heads=2),
            dict(ssm_state_dim=8, ssm_heads=0),
            dict(ssm_state_dim=8, ssm_heads=3),
            dict(ssm_chunk_size=0),
            dict(ssm_chunk_size=True),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, **bad)

    def test_chunkwise_wy_matches_sequential_forward_state_and_gradients(self):
        from blue.modules import RMSNorm
        from blue.ssm import GatedDeltaNet

        torch.manual_seed(1234)
        for seq_len, chunk_size in ((17, 8), (64, 16), (37, 32)):
            ssm = GatedDeltaNet(64, 8, 4, norm_cls=RMSNorm, chunk_size=chunk_size)
            x1 = torch.randn(2, seq_len, 64, requires_grad=True)
            x2 = x1.detach().clone().requires_grad_(True)
            valid = torch.ones(2, seq_len, dtype=torch.bool)
            valid[0, :3] = False
            valid[0, 10] = False
            valid[1, -4:] = False
            init_state = torch.randn(2, 4, 8, 16) * 0.2

            out_seq, s_seq = ssm(x1, valid, init_state, use_chunkwise=False)
            out_wy, s_wy = ssm(x2, valid, init_state, use_chunkwise=True)
            torch.testing.assert_close(out_wy, out_seq, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(s_wy, s_seq, rtol=1e-4, atol=1e-5)

            (out_seq.square().mean() + s_seq.square().mean()).backward()
            grads_seq = {n: p.grad.clone() for n, p in ssm.named_parameters()}
            gx_seq = x1.grad.clone()
            ssm.zero_grad()

            (out_wy.square().mean() + s_wy.square().mean()).backward()
            torch.testing.assert_close(x2.grad, gx_seq, rtol=1e-4, atol=1e-5)
            for n, p in ssm.named_parameters():
                torch.testing.assert_close(p.grad, grads_seq[n], rtol=1e-4, atol=1e-5)

    def test_causality_padding_invariance_cache_equivalence_and_gradients(self):
        torch.manual_seed(4041)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         ssm_state_dim=8, ssm_heads=4,
                         differential_attention=True, kv_lora_rank=16,
                         stable_rms_norm=True, gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertGreater(info["parameters"], inspect_config(BlueConfig.preset("tiny"))["parameters"])
        model = Blue(config).eval()
        # Verify gate biases survive Blue.__init__.
        torch.testing.assert_close(model.blocks[0].ssm.decay_beta_proj.bias[:4], torch.full((4,), 2.0))
        torch.testing.assert_close(model.blocks[0].ssm.decay_beta_proj.bias[4:], torch.full((4,), -1.0))
        self.assertTrue(model.blocks[0].ssm.out_norm.stable)

        batch = synthetic_batch(config, batch_size=2, seed=4042)
        ids = torch.randint(4, 260, (2, 42))

        # 1. Causality.
        changed = ids.clone()
        changed[:, 12:] = torch.randint(4, 260, changed[:, 12:].shape)
        with torch.no_grad():
            first = model(ids, images=batch["images"], series=batch["series"]).logits
            second = model(changed, images=batch["images"], series=batch["series"]).logits
        torch.testing.assert_close(first[:, :12], second[:, :12], rtol=1e-5, atol=1e-6)

        # 2. Exact sliding-window + O(1) SSM state cache equivalence.
        with torch.no_grad():
            full = model(ids, images=batch["images"], series=batch["series"]).logits
            p1 = model(ids[:, :34], images=batch["images"], series=batch["series"], use_cache=True)
            p2 = model(ids[:, 34:39], cache=p1.cache, use_cache=True)
            p3 = model(ids[:, 39:], cache=p2.cache, use_cache=True)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits), 1), full, rtol=2e-5, atol=2e-6)
        self.assertEqual(p3.cache.layers[0].ssm_state.shape, (2, 4, 8, 16))

        # 3. Exact padding invariance (both left and right padding freeze SSM state).
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[0, :4] = False
        with torch.no_grad():
            full_lp = model(ids, attention_mask=mask).logits
            lp1 = model(ids[:, :20], attention_mask=mask[:, :20], use_cache=True)
            lp2 = model(ids[:, 20:], cache=lp1.cache, use_cache=True)
            unpadded = model(ids[0:1, 4:]).logits
        torch.testing.assert_close(lp2.logits, full_lp[:, 20:], rtol=2e-5, atol=2e-6)
        # Right padding does not alter losses.
        padded_batch = batch.copy()
        with torch.no_grad():
            base_out = model(**padded_batch)
            padded_batch["input_ids"] = torch.nn.functional.pad(padded_batch["input_ids"], (0, 6), value=0)
            padded_batch["labels"] = padded_batch["input_ids"].clone()
            padded_batch["attention_mask"] = torch.nn.functional.pad(padded_batch["attention_mask"], (0, 6), value=False)
            pad_out = model(**padded_batch)
        torch.testing.assert_close(base_out.loss, pad_out.loss, rtol=1e-5, atol=1e-6)

        # 4. Gradients reach every SSM parameter under bf16 + gradient checkpointing.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for block in model.blocks:
            for name, param in block.ssm.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
                self.assertGreater(param.grad.abs().sum().item(), 0.0, name)
            self.assertGreater(block.ssm_gate.grad.abs().sum().item(), 0.0)

        # 5. Checkpoint roundtrip and generation smoke test.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ssm.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertEqual(restored.config.ssm_state_dim, 8)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)
            gen = restored.generate(ids[:1, :6], max_new_tokens=5, temperature=0)
            self.assertEqual(gen.shape, (1, 11))


if __name__ == "__main__":
    unittest.main()
