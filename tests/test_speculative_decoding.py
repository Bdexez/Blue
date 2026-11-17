from dataclasses import replace
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch


class SpeculativeDecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_validation(self):
        model = Blue(BlueConfig.preset("tiny")).eval()
        prompt = torch.tensor([[1, 20, 21]])
        with self.assertRaises(ValueError):
            model.generate(prompt, max_new_tokens=4, temperature=0, speculative=1)

    def test_speculative_greedy_matches_standard_exactly_across_full_hybrid_stack(self):
        for cfg in (
            replace(BlueConfig.preset("tiny"), max_seq_len=128, prediction_steps=4),
            replace(BlueConfig.preset("tiny"), max_seq_len=128, prediction_steps=4,
                    differential_attention=True, kv_lora_rank=16, ssm_state_dim=8,
                    residual_streams=2, latent_reasoning_steps=2, audio_enabled=True),
        ):
            with self.subTest(ssm=cfg.ssm_state_dim, steps=cfg.latent_reasoning_steps):
                torch.manual_seed(9801)
                model = Blue(cfg).eval()
                batch = synthetic_batch(cfg, batch_size=1, seed=9802)
                prompt = batch["input_ids"][:, :6]
                context = dict(images=batch["images"], series=batch["series"])
                if cfg.audio_enabled:
                    context["audio"] = batch["audio"]
                    context["audio_mask"] = batch["audio_mask"]
                standard = model.generate(prompt, max_new_tokens=18, temperature=0, eos_id=None, **context)
                speculative = model.generate(prompt, max_new_tokens=18, temperature=0, eos_id=None,
                                             speculative=True, **context)
                torch.testing.assert_close(speculative, standard, rtol=0, atol=0)

    def test_speculative_accepts_accurate_future_drafts_in_fewer_forward_calls(self):
        torch.manual_seed(9803)
        cfg = replace(BlueConfig.preset("tiny"), max_seq_len=128, prediction_steps=3)
        model = Blue(cfg)
        batch = synthetic_batch(cfg, batch_size=1, seed=9804)
        # Train briefly on a single sequence so future_heads agree with the main LM head.
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(35):
            opt.zero_grad(set_to_none=True)
            model(batch["input_ids"], labels=batch["labels"]).loss.backward()
            opt.step()
        model.eval()
        prompt = batch["input_ids"][:, :4]
        standard = model.generate(prompt, max_new_tokens=12, temperature=0, eos_id=2)
        calls = 0
        original_forward = model.forward

        def counting_forward(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_forward(*args, **kwargs)

        model.forward = counting_forward
        try:
            speculative = model.generate(prompt, max_new_tokens=12, temperature=0, eos_id=2, speculative=True)
        finally:
            model.forward = original_forward
        torch.testing.assert_close(speculative, standard, rtol=0, atol=0)
        generated_len = standard.size(1) - prompt.size(1)
        self.assertGreater(generated_len, 2)
        self.assertLess(calls, generated_len)


if __name__ == "__main__":
    unittest.main()
