from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config, main
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class AttentionSinksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        for bad in (-1, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, attention_sinks=bad)

    def test_sinks_across_all_four_attention_modes_cache_causality_and_gradients(self):
        for kv_lora, diff in ((0, False), (0, True), (16, False), (16, True)):
            with self.subTest(kv_lora=kv_lora, diff=diff):
                torch.manual_seed(14001 + kv_lora + int(diff))
                config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                                 attention_sinks=4, kv_lora_rank=kv_lora,
                                 differential_attention=diff, gradient_checkpointing=True)
                info = inspect_config(config)
                self.assertEqual(info["architecture_features"]["attention_sinks"], 4)
                model = Blue(config).eval()
                # Bidirectional encoders do not attach sinks; causal decoder blocks do.
                self.assertIsNone(model.vision.blocks[0].attn.sink_k)
                self.assertIsNotNone(model.blocks[0].attn.sink_k)
                self.assertEqual(model.blocks[0].attn.sink_v.abs().sum().item(), 0.0)
                # Populate sink_v slightly to verify cache equivalence depends on sinks cleanly.
                with torch.no_grad():
                    for block in model.blocks:
                        block.attn.sink_v.normal_(std=0.05)

                batch = synthetic_batch(config, batch_size=2, seed=14002)
                ids = torch.randint(4, 260, (2, 42))

                # 1. Causality.
                changed = ids.clone()
                changed[:, 12:] = torch.randint(4, 260, changed[:, 12:].shape)
                with torch.no_grad():
                    first = model(ids).logits
                    second = model(changed).logits
                torch.testing.assert_close(first[:, :12], second[:, :12], rtol=1e-5, atol=1e-6)

                # 2. Sliding-window cache equivalence (sinks are never evicted and do not bloat cache).
                with torch.no_grad():
                    full = model(ids).logits
                    p1 = model(ids[:, :34], use_cache=True)
                    p2 = model(ids[:, 34:39], cache=p1.cache, use_cache=True)
                    p3 = model(ids[:, 39:], cache=p2.cache, use_cache=True)
                torch.testing.assert_close(torch.cat((p1.logits, p2.logits, p3.logits), 1), full, rtol=3e-5, atol=3e-6)
                self.assertEqual(p3.cache.layers[0].key.size(-2), config.sliding_window - 1)

                # 3. Gradients reach sink_k and sink_v under bf16 + gradient checkpointing.
                model.train()
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out = model(**batch)
                self.assertTrue(torch.isfinite(out.loss))
                out.loss.backward()
                for block in model.blocks:
                    self.assertGreater(block.attn.sink_k.grad.abs().sum().item(), 0.0)
                    self.assertGreater(block.attn.sink_v.grad.abs().sum().item(), 0.0)

    def test_cli_generate_supports_speculative_and_reasoning_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                             attention_sinks=2, latent_reasoning_steps=2, prediction_steps=3)
            model = Blue(config).eval()
            save_checkpoint(path, model)
            buf1, buf2 = io.StringIO(), io.StringIO()
            with redirect_stdout(buf1):
                main(["--threads", "2", "generate", "--checkpoint", str(path), "--prompt", "Hi",
                      "--max-new-tokens", "6", "--temperature", "0", "--reasoning-steps", "2"])
            with redirect_stdout(buf2):
                main(["--threads", "2", "generate", "--checkpoint", str(path), "--prompt", "Hi",
                      "--max-new-tokens", "6", "--temperature", "0", "--reasoning-steps", "2", "--speculative"])
            self.assertEqual(buf1.getvalue(), buf2.getvalue())
            restored, _ = load_checkpoint(path)
            self.assertEqual(restored.config.attention_sinks, 2)


if __name__ == "__main__":
    unittest.main()
