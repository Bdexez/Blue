from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class EpisodicMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        for bad in (-1, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, memory_slots=bad)

    def test_causality_cache_padding_speculative_gradients_and_checkpoint(self):
        torch.manual_seed(9971)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         memory_slots=8, ssm_state_dim=8, residual_streams=2,
                         differential_attention=True, kv_lora_rank=16,
                         latent_reasoning_steps=2, prediction_steps=3,
                         stable_rms_norm=True, gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertIn("episodic_memory", info["modules"])
        self.assertEqual(info["architecture_features"]["memory_slots"], 8)
        model = Blue(config).eval()
        batch = synthetic_batch(config, batch_size=2, seed=9972)
        ids = torch.randint(4, 260, (2, 40))

        # 1. Strict causality.
        changed = ids.clone()
        changed[:, 12:] = torch.randint(4, 260, changed[:, 12:].shape)
        with torch.no_grad():
            first = model(ids, images=batch["images"], series=batch["series"]).logits
            second = model(changed, images=batch["images"], series=batch["series"]).logits
        torch.testing.assert_close(first[:, :12], second[:, :12], rtol=1e-5, atol=1e-6)

        # 2. Exact sliding-window + episodic memory cache equivalence.
        with torch.no_grad():
            full = model(ids, images=batch["images"], series=batch["series"]).logits
            p1 = model(ids[:, :33], images=batch["images"], series=batch["series"], use_cache=True)
            p2 = model(ids[:, 33:37], cache=p1.cache, use_cache=True)
            p3 = model(ids[:, 37:], cache=p2.cache, use_cache=True)
        self.assertEqual(p3.cache.episodic_memory.shape, (2, 8, config.d_model))
        torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits), 1), full, rtol=3e-5, atol=3e-6)

        # 3. Exact left & right padding invariance (memory freezes on invalid tokens).
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[0, :4] = False
        with torch.no_grad():
            full_lp = model(ids, attention_mask=mask).logits
            lp1 = model(ids[:, :20], attention_mask=mask[:, :20], use_cache=True)
            lp2 = model(ids[:, 20:], cache=lp1.cache, use_cache=True)
        torch.testing.assert_close(lp2.logits, full_lp[:, 20:], rtol=3e-5, atol=3e-6)
        with torch.no_grad():
            base_out = model(**batch)
            padded = batch.copy()
            padded["input_ids"] = torch.nn.functional.pad(padded["input_ids"], (0, 5), value=0)
            padded["labels"] = padded["input_ids"].clone()
            padded["attention_mask"] = torch.nn.functional.pad(padded["attention_mask"], (0, 5), value=False)
            pad_out = model(**padded)
        torch.testing.assert_close(base_out.loss, pad_out.loss, rtol=1e-5, atol=1e-6)

        # 4. Self-speculative greedy decoding matches standard greedy exactly with episodic memory.
        std_gen = model.generate(ids[:1, :6], max_new_tokens=10, temperature=0, eos_id=None)
        spec_gen = model.generate(ids[:1, :6], max_new_tokens=10, temperature=0, eos_id=None, speculative=True)
        torch.testing.assert_close(spec_gen, std_gen, rtol=0, atol=0)

        # 5. Gradients reach every episodic memory parameter under bf16 + checkpointing.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for name, param in model.episodic_memory.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(param.grad.abs().sum().item(), 0.0, name)

        # 6. Checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertEqual(restored.config.memory_slots, 8)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
