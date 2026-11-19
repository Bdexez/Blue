from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class ByteNGramTests(unittest.TestCase):
    def test_config_validation_and_backward_compat(self):
        default = BlueConfig.preset("tiny")
        self.assertEqual(default.ngram_vocab_size, 0)
        self.assertIsNone(Blue(default).ngram)
        for bad in (-1, 1, 15, True):
            with self.assertRaises(ValueError):
                replace(default, ngram_vocab_size=bad)

    def test_causality_padding_invariance_cache_equivalence_and_gradients(self):
        torch.manual_seed(340)
        cfg = replace(BlueConfig.preset("tiny"), ngram_vocab_size=512, stable_rms_norm=True)
        model = Blue(cfg).eval()
        torch.testing.assert_close(model.ngram.order_gates, torch.full((3, cfg.d_model), -2.0))

        ids = torch.randint(4, 260, (2, 20))
        # 1. Strict causality.
        changed = ids.clone()
        changed[:, 9:] = torch.randint(4, 260, changed[:, 9:].shape)
        with torch.no_grad():
            o1 = model(ids).logits
            o2 = model(changed).logits
        torch.testing.assert_close(o1[:, :9], o2[:, :9], rtol=1e-5, atol=1e-6)

        # 2. Exact incremental cache equivalence (including short 1-token and 2-token chunks < 3!).
        with torch.no_grad():
            p1 = model(ids[:, :1], use_cache=True)
            p2 = model(ids[:, 1:3], cache=p1.cache, use_cache=True)
            p3 = model(ids[:, 3:11], cache=p2.cache, use_cache=True)
            p4 = model(ids[:, 11:], cache=p3.cache, use_cache=True)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits, p4.logits), 1), o1, rtol=1e-5, atol=1e-5)

        # 3. Exact left-padding invariance (masked bytes never contaminate valid n-grams).
        pad_ids = torch.cat((torch.randint(4, 260, (1, 4)), ids[:1]), dim=1)
        pad_mask = torch.cat((torch.zeros(1, 4, dtype=torch.bool), torch.ones(1, 20, dtype=torch.bool)), dim=1)
        with torch.no_grad():
            unpadded_ng, _, _ = model.ngram(ids[:1], torch.ones(1, 20, dtype=torch.bool))
            padded_ng, _, _ = model.ngram(pad_ids, pad_mask)
        torch.testing.assert_close(padded_ng[:, 4:], unpadded_ng, rtol=0, atol=0)
        torch.testing.assert_close(padded_ng[:, :4], torch.zeros_like(padded_ng[:, :4]), rtol=0, atol=0)

        # 4. Gradients under BF16 + checkpoint roundtrip + generation.
        model.train()
        batch = synthetic_batch(cfg, batch_size=2, seed=341)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertGreater(model.ngram.table.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.ngram.order_gates.grad.abs().sum().item(), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ngram.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)
            gen = restored.generate(ids[:1, :5], max_new_tokens=6, temperature=0)
            self.assertEqual(gen.shape, (1, 11))


if __name__ == "__main__":
    unittest.main()
