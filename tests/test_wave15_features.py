import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.verifier import verify_architecture


class Wave15FeaturesTests(unittest.TestCase):
    def test_forgetting_attention_router_entropy_and_spectral_wavelet_gate(self):
        for softcap in (0.0, 30.0):
            config = BlueConfig(
                d_model=32,
                n_heads=4,
                n_kv_heads=2,
                n_layers=2,
                hidden_dim=64,
                n_experts=4,
                top_k=2,
                moe_null_experts=1,
                moe_router_geometry="hyperbolic",
                moe_router_temperature=True,
                moe_entropy_weight=0.01,
                forgetting_attention=True,
                attention_sinks=2,
                attn_logit_softcap=softcap,
                kv_lora_rank=16,
                differential_attention=True,
                temporal_spectral_mixing=True,
                temporal_wavelet_mixing=True,
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
                gradient_checkpointing=True,
                dropout=0.0,
            )
            rep = verify_architecture(config, seed=19)
            self.assertTrue(rep["passed"], f"Failed verifier for softcap={softcap}: {rep}")

            model = Blue(config).train()
            tokens = torch.randint(0, 260, (2, 10))
            series = torch.randn(2, 12, 2)
            out = model(tokens, series=series, labels=tokens)
            self.assertIn("router_entropy", out.losses)
            self.assertGreater(out.losses["router_entropy"].item(), 0.0)
            self.assertLessEqual(out.losses["router_entropy"].item(), 1.0 + 1e-5)

            out.loss.backward()
            self.assertIsNotNone(model.temporal.freq_wavelet_gate.grad)
            self.assertGreater(model.temporal.freq_wavelet_gate.grad.abs().sum().item(), 0.0)
            for block in model.blocks:
                self.assertIsNotNone(block.attn.forget_proj.weight.grad)
                self.assertIsNotNone(block.attn.forget_slopes.grad)
                self.assertIsNotNone(block.moe.router_log_temp.grad)
                self.assertGreater(block.attn.forget_proj.weight.grad.abs().sum().item(), 0.0)
                self.assertGreater(block.attn.forget_slopes.grad.abs().sum().item(), 0.0)
                self.assertGreater(block.moe.router_log_temp.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
