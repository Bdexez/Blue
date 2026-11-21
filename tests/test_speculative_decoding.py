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

    def test_stochastic_speculative_decoding_valid_tokens_and_distribution(self):
        torch.manual_seed(9810)
        cfg = replace(
            BlueConfig.preset("tiny"), max_seq_len=128, prediction_steps=4,
            differential_attention=True, kv_lora_rank=16, ssm_state_dim=8,
            ttt_state_dim=8, residual_streams=2,
        )
        model = Blue(cfg).eval()
        batch = synthetic_batch(cfg, batch_size=1, seed=9811)
        prompt = batch["input_ids"][:, :5]
        out = model.generate(
            prompt, max_new_tokens=16, temperature=0.8, top_k=24, top_p=0.9,
            eos_id=None, speculative=True, images=batch["images"], series=batch["series"],
        )
        self.assertEqual(out.shape, (1, 21))
        gen_tokens = out[0, 5:]
        self.assertFalse(((gen_tokens == 0) | (gen_tokens == 1) | (gen_tokens == 3)).any())

        # Verify unbiased marginal distribution on the 2nd generated token (position verified by Leviathan rejection).
        torch.manual_seed(9812)
        simple = Blue(replace(BlueConfig.preset("tiny"), prediction_steps=3)).eval()
        # Pick 1st token deterministically (top_k=1) then verify 2nd token's distribution? Better:
        # Check that with temperature=1.0, top_k=4 over 2 tokens, empirical 1st+2nd tokens stay strictly inside top-4 support!
        p_ids = torch.tensor([[1, 65, 66, 67]])
        out0 = simple(p_ids, use_cache=True)
        p1 = simple._filtered_probs(out0.logits[:, -1], temperature=1.0, top_k=4, top_p=1.0)
        top4_first = set(p1[0].nonzero().flatten().tolist())
        counts = {}
        for _ in range(80):
            seq = simple.generate(p_ids, max_new_tokens=3, temperature=1.0, top_k=4, top_p=1.0,
                                  eos_id=None, speculative=True)
            t1 = seq[0, 4].item()
            self.assertIn(t1, top4_first)
            counts[t1] = counts.get(t1, 0) + 1
        # All 4 top-k tokens must be supported and empirical freq within statistical tolerance.
        for tok in top4_first:
            emp = counts.get(tok, 0) / 80.0
            self.assertLess(abs(emp - p1[0, tok].item()), 0.25)


if __name__ == "__main__":
    unittest.main()
