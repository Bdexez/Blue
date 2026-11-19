from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.lora import DoRALinear, inject_dora, merge_dora
from blue.training import load_checkpoint, save_checkpoint


class DoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dora_and_lora_exact_identity_at_init_training_and_exact_merge(self):
        for use_dora in (True, False):
            with self.subTest(use_dora=use_dora):
                torch.manual_seed(20001 + int(use_dora))
                config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                                 differential_attention=True, kv_lora_rank=16, ssm_state_dim=8)
                model = Blue(config).eval()
                batch = synthetic_batch(config, batch_size=2, seed=20002)
                with torch.no_grad():
                    before = model(**batch)

                summary = inject_dora(model, rank=4, alpha=8.0, use_dora=use_dora)
                self.assertGreater(summary["replaced_modules"], 0)
                self.assertLess(summary["trainable_ratio"], 0.3)
                self.assertFalse(model.embedding.weight.requires_grad)

                # Exact identity before first optimization step.
                with torch.no_grad():
                    at_init = model(**batch)
                torch.testing.assert_close(before.logits, at_init.logits, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(before.forecast, at_init.forecast, rtol=1e-5, atol=1e-6)

                # Train only DoRA/LoRA parameters under bf16 and verify loss decreases.
                model.train()
                base_embed_before = model.embedding.weight.detach().clone()
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
                initial_loss = at_init.loss.item()
                for _ in range(8):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        out = model(**batch)
                    self.assertTrue(torch.isfinite(out.loss))
                    out.loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    trained = model(**batch)
                self.assertLess(trained.loss.item(), initial_loss)
                # Base frozen weights remain strictly untouched.
                torch.testing.assert_close(model.embedding.weight, base_embed_before, rtol=0, atol=0)

                # Exact merge removes all DoRALinear wrappers and preserves outputs.
                merged_count = merge_dora(model)
                self.assertEqual(merged_count, summary["replaced_modules"])
                self.assertFalse(any(isinstance(m, DoRALinear) for m in model.modules()))
                self.assertIs(model.lm_head.weight, model.embedding.weight)
                with torch.no_grad():
                    after_merge = model(**batch)
                torch.testing.assert_close(trained.logits, after_merge.logits, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(trained.classification_logits, after_merge.classification_logits,
                                           rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(trained.forecast, after_merge.forecast, rtol=1e-5, atol=1e-5)

                # Merged model saves & reloads as a standard Blue checkpoint.
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "merged.pt"
                    save_checkpoint(path, model)
                    restored, _ = load_checkpoint(path)
                    with torch.no_grad():
                        torch.testing.assert_close(after_merge.logits, restored(**batch).logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
