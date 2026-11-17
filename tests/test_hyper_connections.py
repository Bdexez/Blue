from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.hyper_connections import sinkhorn_knopp
from blue.training import load_checkpoint, save_checkpoint


class ManifoldHyperConnectionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        for bad in (0, -2, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, residual_streams=bad)

    def test_sinkhorn_knopp_produces_doubly_stochastic_matrices(self):
        torch.manual_seed(6001)
        logits = torch.randn(3, 7, 4, 4) * 2.0
        matrix = sinkhorn_knopp(logits, iterations=80)
        self.assertTrue((matrix > 0).all())
        torch.testing.assert_close(matrix.sum(dim=-1), torch.ones(3, 7, 4), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(matrix.sum(dim=-2), torch.ones(3, 7, 4), rtol=1e-5, atol=1e-5)
        # Spectral norm of every doubly stochastic matrix is <= 1.
        singular_values = torch.linalg.svdvals(matrix)
        self.assertLessEqual(singular_values.max().item(), 1.0 + 1e-5)
        # Default 20 iterations on realistic near-identity logits converges tightly.
        near_id = torch.eye(4).expand(3, 7, 4, 4) * 4.0 + torch.randn(3, 7, 4, 4) * 0.2
        fast = sinkhorn_knopp(near_id, iterations=20)
        torch.testing.assert_close(fast.sum(dim=-1), torch.ones(3, 7, 4), rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(fast.sum(dim=-2), torch.ones(3, 7, 4), rtol=1e-5, atol=1e-5)
        # Extreme logits remain finite in log-domain Sinkhorn.
        extreme = sinkhorn_knopp(torch.randn(2, 4, 4) * 50.0, iterations=20)
        self.assertTrue(torch.isfinite(extreme).all())

    def test_mhc_full_stack_causality_cache_padding_bf16_and_isolated_text(self):
        torch.manual_seed(6002)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         residual_streams=4, ssm_state_dim=8,
                         differential_attention=True, kv_lora_rank=16,
                         shared_expert_gate=True, router_bias_update_rate=0.01,
                         alignment_text_view="isolated", stable_rms_norm=True,
                         gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertGreater(info["parameters"], inspect_config(BlueConfig.preset("tiny"))["parameters"])
        model = Blue(config).eval()
        # Verify res_bias diagonal initialization survived Blue.__init__.
        torch.testing.assert_close(model.blocks[0].hc_attn.res_bias.diag(), torch.full((4,), 4.0))
        self.assertTrue(model.blocks[0].hc_attn.norm.stable)

        batch = synthetic_batch(config, batch_size=2, seed=6003)
        ids = torch.randint(4, 260, (2, 40))

        # 1. Causality.
        changed = ids.clone()
        changed[:, 10:] = torch.randint(4, 260, changed[:, 10:].shape)
        with torch.no_grad():
            first = model(ids, images=batch["images"], series=batch["series"]).logits
            second = model(changed, images=batch["images"], series=batch["series"]).logits
        torch.testing.assert_close(first[:, :10], second[:, :10], rtol=1e-5, atol=1e-6)

        # 2. Sliding-window + MLA + SSM + mHC exact cache equivalence.
        with torch.no_grad():
            full = model(ids, images=batch["images"], series=batch["series"]).logits
            p1 = model(ids[:, :33], images=batch["images"], series=batch["series"], use_cache=True)
            p2 = model(ids[:, 33:37], cache=p1.cache, use_cache=True)
            p3 = model(ids[:, 37:], cache=p2.cache, use_cache=True)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits), 1), full, rtol=3e-5, atol=3e-6)

        # 3. Padding invariance & isolated text encoding.
        with torch.no_grad():
            base_out = model(**batch)
            padded = batch.copy()
            padded["input_ids"] = torch.nn.functional.pad(padded["input_ids"], (0, 5), value=0)
            padded["labels"] = padded["input_ids"].clone()
            padded["attention_mask"] = torch.nn.functional.pad(padded["attention_mask"], (0, 5), value=False)
            pad_out = model(**padded)
            embeds = model.encode_modalities(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        torch.testing.assert_close(base_out.loss, pad_out.loss, rtol=1e-5, atol=1e-6)
        self.assertEqual(embeds["text"].shape, (2, config.d_model))

        # 4. Gradients reach every mHC parameter under bf16 + gradient checkpointing.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for block in model.blocks:
            for hc in (block.hc_attn, block.hc_cross, block.hc_moe):
                for name, param in hc.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all(), name)
                    self.assertGreater(param.grad.abs().sum().item(), 0.0, name)

        # 5. Checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mhc.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertEqual(restored.config.residual_streams, 4)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
