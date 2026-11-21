import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.verifier import verify_architecture


class Wave17FeaturesTests(unittest.TestCase):
    def test_qk_residual_dynamic_mixer_router_and_forecast_smoothness(self):
        for kv_rank, diff in ((0, False), (16, True)):
            config = BlueConfig(
                d_model=32,
                n_heads=4,
                n_kv_heads=2,
                n_layers=3,
                hidden_dim=64,
                n_experts=4,
                top_k=2,
                value_residual=True,
                qk_residual=True,
                dynamic_mixer_router=True,
                ssm_state_dim=8,
                ttt_state_dim=8,
                kv_lora_rank=kv_rank,
                differential_attention=diff,
                forecast_smoothness_weight=0.05,
                n_latents=4,
                resampler_layers=1,
                vision_layers=1,
                temporal_layers=1,
                image_size=16,
                patch_size=8,
                max_seq_len=32,
                max_series_len=16,
                forecast_horizon=6,
                forecast_context=4,
                series_features=2,
                conv_kernel=3,
                sliding_window=8,
                global_every=1,
                gradient_checkpointing=True,
                checkpoint_every=2,
                dropout=0.0,
            )
            rep = verify_architecture(config, seed=29)
            self.assertTrue(rep["passed"], f"Failed verifier for kv={kv_rank}, diff={diff}: {rep}")

            model = Blue(config).train()
            tokens = torch.randint(0, 260, (2, 10))
            series = torch.randn(2, 12, 2)
            forecast_targets = torch.randn(2, 6, 2)
            out = model(tokens, series=series, labels=tokens, forecast_targets=forecast_targets)
            self.assertIn("forecast_smoothness", out.losses)
            self.assertGreaterEqual(out.losses["forecast_smoothness"].item(), 0.0)

            out.loss.backward()
            for idx, block in enumerate(model.blocks):
                self.assertIsNotNone(block.mixer_router.weight.grad)
                self.assertGreater(block.mixer_router.weight.grad.abs().sum().item(), 0.0)
                if idx > 0:
                    self.assertIsNotNone(block.attn.q_res_gate.grad)
                    self.assertGreater(block.attn.q_res_gate.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
