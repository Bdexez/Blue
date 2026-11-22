import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.verifier import verify_architecture


class Wave16FeaturesTests(unittest.TestCase):
    def test_gated_value_attention_mtp_decay_and_min_p_sampling(self):
        config = BlueConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            hidden_dim=64,
            n_experts=4,
            top_k=2,
            gated_value_attention=True,
            forgetting_attention=True,
            dynamic_attn_scale=True,
            kv_lora_rank=16,
            differential_attention=True,
            prediction_steps=4,
            mtp_kl_weight=0.05,
            mtp_decay=0.75,
            n_latents=4,
            resampler_layers=1,
            vision_layers=1,
            temporal_layers=1,
            image_size=16,
            patch_size=8,
            max_seq_len=32,
            max_series_len=16,
            forecast_horizon=4,
            forecast_context=4,
            series_features=2,
            conv_kernel=3,
            sliding_window=8,
            global_every=1,
            dropout=0.0,
        )
        rep = verify_architecture(config, seed=23)
        self.assertTrue(rep["passed"], rep)

        model = Blue(config).train()
        tokens = torch.randint(0, 260, (2, 12))
        out = model(tokens, labels=tokens)
        self.assertIn("multi_token", out.losses)
        self.assertIn("mtp_kl", out.losses)
        out.loss.backward()
        for block in model.blocks:
            self.assertIsNotNone(block.attn.v_out_down.weight.grad)
            self.assertIsNotNone(block.attn.v_out_up.weight.grad)
            self.assertGreater(block.attn.v_out_down.weight.grad.abs().sum().item(), 0.0)
            self.assertGreater(block.attn.v_out_up.weight.grad.abs().sum().item(), 0.0)

        # Verify Min-P filtering truncates tail tokens below min_p * max_prob
        scores = torch.full((1, 260), -10.0)
        scores[0, 65] = 4.0   # dominant token
        scores[0, 66] = 2.5   # ~22% of max_prob
        scores[0, 67] = 0.0   # ~1.8% of max_prob
        probs = Blue._filtered_probs(scores, temperature=1.0, top_k=0, top_p=1.0, min_p=0.1)
        self.assertGreater(probs[0, 65].item(), 0.7)
        self.assertGreater(probs[0, 66].item(), 0.15)
        self.assertEqual(probs[0, 67].item(), 0.0)

        model.eval()
        prompt = torch.tensor([[1, 66, 108]])
        gen_std = model.generate(prompt, max_new_tokens=6, temperature=0.8, min_p=0.15)
        gen_spec = model.generate(prompt, max_new_tokens=6, temperature=0.8, min_p=0.15, speculative=True)
        self.assertEqual(gen_std.shape, (1, 9))
        self.assertEqual(gen_spec.shape, (1, 9))


if __name__ == "__main__":
    unittest.main()
