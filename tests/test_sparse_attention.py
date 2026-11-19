from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class NativeSparseAttentionTests(unittest.TestCase):
    def test_config_validation_causality_cache_and_all_four_attention_modes(self):
        base = BlueConfig.preset("tiny")
        self.assertEqual(base.nsa_block_size, 0)
        for bad in (dict(nsa_block_size=-1), dict(nsa_block_size=1), dict(nsa_block_size=True), dict(nsa_top_k=0)):
            with self.assertRaises(ValueError):
                replace(base, **bad)

        modes = (
            dict(differential_attention=False, kv_lora_rank=0),
            dict(differential_attention=True, kv_lora_rank=0),
            dict(differential_attention=False, kv_lora_rank=16),
            dict(differential_attention=True, kv_lora_rank=16),
        )
        for idx, mode in enumerate(modes):
            with self.subTest(mode=mode):
                torch.manual_seed(440 + idx)
                cfg = replace(
                    base,
                    n_layers=3, global_every=2, sliding_window=8,
                    attention_sinks=2, value_residual=True,
                    nsa_block_size=4, nsa_top_k=2,
                    **mode,
                )
                model = Blue(cfg).eval()
                # Global layer index 1 ((1+1) % 2 == 0) has NSA; local layers 0 & 2 do not.
                self.assertIsNone(model.blocks[0].attn.nsa)
                self.assertIsNotNone(model.blocks[1].attn.nsa)
                self.assertIsNone(model.blocks[2].attn.nsa)

                ids = torch.randint(4, 260, (2, 22))
                changed = ids.clone()
                changed[:, 11:] = torch.randint(4, 260, changed[:, 11:].shape)
                with torch.no_grad():
                    o1 = model(ids).logits
                    o2 = model(changed).logits
                torch.testing.assert_close(o1[:, :11], o2[:, :11], rtol=1e-5, atol=1e-6)

                # Exact incremental cache equivalence across block boundaries (4, 8, 12, 16, 20)!
                with torch.no_grad():
                    p1 = model(ids[:, :7], use_cache=True)
                    p2 = model(ids[:, 7:15], cache=p1.cache, use_cache=True)
                    p3 = model(ids[:, 15:], cache=p2.cache, use_cache=True)
                torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits), 1), o1, rtol=1e-4, atol=1e-4)

                # Gradients reach NSA parameters under BF16.
                model.train()
                batch = synthetic_batch(cfg, batch_size=2, seed=445 + idx)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out = model(**batch)
                self.assertTrue(torch.isfinite(out.loss))
                out.loss.backward()
                nsa = model.blocks[1].attn.nsa
                self.assertGreater(nsa.k_compress.weight.grad.abs().sum().item(), 0.0)
                self.assertGreater(nsa.v_compress.weight.grad.abs().sum().item(), 0.0)
                self.assertGreater(nsa.gate.grad.abs().sum().item(), 0.0)

                if idx == 3:
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "nsa.pt"
                        save_checkpoint(path, model.eval())
                        restored, _ = load_checkpoint(path)
                        with torch.no_grad():
                            torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
