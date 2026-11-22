import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.verifier import verify_architecture


class Wave22FeaturesTests(unittest.TestCase):
    def test_conv_diff_branch_lm_z_loss_and_mirostat_v2(self):
        config = BlueConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            hidden_dim=64,
            n_experts=4,
            top_k=2,
            conv_kernel=5,
            conv_diff_branch=True,
            lm_z_loss_weight=1e-3,
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
            sliding_window=8,
            global_every=1,
            dropout=0.0,
        )
        rep = verify_architecture(config, seed=47)
        self.assertTrue(rep["passed"], rep)

        model = Blue(config).train()
        tokens = torch.randint(0, 260, (2, 10))
        series = torch.randn(2, 12, 2)
        out = model(tokens, series=series, labels=tokens)
        self.assertIn("lm_z", out.losses)
        self.assertGreater(out.losses["lm_z"].item(), 0.0)
        out.loss.backward()
        for block in model.blocks:
            self.assertIsNotNone(block.conv.vel_gain.grad)
            self.assertIsNotNone(block.conv.acc_gain.grad)
            self.assertGreater(block.conv.vel_gain.grad.abs().sum().item(), 0.0)
            self.assertGreater(block.conv.acc_gain.grad.abs().sum().item(), 0.0)

        # Verify Mirostat v2 step updates mu feedback state in closed loop
        scores = torch.full((2, 260), -8.0)
        scores[:, 65] = 2.0
        scores[:, 66] = 1.0
        mu = torch.tensor([4.0, 0.5])
        tok, next_mu = Blue._mirostat_v2_step(scores, mu, temperature=1.0, tau=2.0, eta=0.2)
        self.assertEqual(tok.shape, (2,))
        self.assertEqual(next_mu.shape, (2,))
        self.assertFalse(torch.equal(mu, next_mu))

        model.eval()
        prompt = torch.tensor([[1, 66, 108]])
        gen_miro = model.generate(
            prompt, max_new_tokens=8, temperature=0.9,
            mirostat_tau=3.0, mirostat_eta=0.15,
        )
        self.assertEqual(gen_miro.shape, (1, 11))


if __name__ == "__main__":
    unittest.main()
