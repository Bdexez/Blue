from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class PonderHaltingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        for bad in (
            dict(latent_reasoning_adaptive=1),
            dict(latent_reasoning_adaptive=True, latent_reasoning_steps=1),
            dict(ponder_prior=0.0),
            dict(ponder_prior=1.0),
            dict(ponder_kl_weight=-0.1),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, **bad)

    def test_pondernet_halting_causality_cache_padding_gradients_and_checkpoint(self):
        torch.manual_seed(25001)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         latent_reasoning_steps=3, latent_reasoning_adaptive=True,
                         ponder_prior=0.4, ponder_kl_weight=0.05,
                         residual_streams=2, gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertTrue(info["architecture_features"]["latent_reasoning_adaptive"])
        model = Blue(config).eval()
        batch = synthetic_batch(config, batch_size=2, seed=25002)
        ids = torch.randint(4, 260, (2, 40))

        # 1. Causality.
        changed = ids.clone()
        changed[:, 12:] = torch.randint(4, 260, changed[:, 12:].shape)
        with torch.no_grad():
            first = model(ids, images=batch["images"], series=batch["series"]).logits
            second = model(changed, images=batch["images"], series=batch["series"]).logits
        torch.testing.assert_close(first[:, :12], second[:, :12], rtol=1e-5, atol=1e-6)

        # 2. Exact sliding-window + recurrent cache equivalence.
        with torch.no_grad():
            full = model(ids, images=batch["images"], series=batch["series"]).logits
            p1 = model(ids[:, :34], images=batch["images"], series=batch["series"], use_cache=True)
            p2 = model(ids[:, 34:38], cache=p1.cache, use_cache=True)
            p3 = model(ids[:, 38:], cache=p2.cache, use_cache=True)
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits), 1), full, rtol=3e-5, atol=3e-6)

        # 3. Exact padding invariance of ponder_kl and total loss.
        with torch.no_grad():
            base_out = model(**batch)
            padded = batch.copy()
            padded["input_ids"] = torch.nn.functional.pad(padded["input_ids"], (0, 6), value=0)
            padded["labels"] = padded["input_ids"].clone()
            padded["attention_mask"] = torch.nn.functional.pad(padded["attention_mask"], (0, 6), value=False)
            pad_out = model(**padded)
        self.assertIn("ponder_kl", base_out.losses)
        self.assertGreaterEqual(base_out.losses["ponder_kl"].item(), -1e-6)
        torch.testing.assert_close(base_out.losses["ponder_kl"], pad_out.losses["ponder_kl"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(base_out.loss, pad_out.loss, rtol=1e-5, atol=1e-6)

        # 4. Gradients reach ponder_halt weight and bias under bf16 + checkpointing.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertGreater(model.ponder_halt.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.ponder_halt.bias.grad.abs().sum().item(), 0.0)

        # 5. Checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ponder.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertTrue(restored.config.latent_reasoning_adaptive)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
