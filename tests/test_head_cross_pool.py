import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.verifier import verify_architecture


class HeadCrossPoolTests(unittest.TestCase):
    def test_head_cross_pool_with_text_and_without_text(self):
        config = BlueConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            hidden_dim=64,
            n_experts=4,
            top_k=2,
            n_latents=6,
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
            cross_differential_attention=True,
            cross_kv_lora_rank=16,
            head_cross_pool=True,
            dropout=0.0,
        )
        torch.manual_seed(5)
        model = Blue(config)
        images = torch.randn(2, 3, 16, 16)
        series = torch.randn(2, 12, 2)
        class_labels = torch.tensor([1, 2], dtype=torch.long)
        forecast_targets = torch.randn(2, 4, 2)

        # Forward without text (task_queries path) and with text
        out_no_text = model(
            images=images, series=series,
            class_labels=class_labels, forecast_targets=forecast_targets,
        )
        out_no_text.loss.backward()
        self.assertIsNotNone(model.head_pool_gate.grad)
        self.assertGreater(model.head_pool_gate.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.head_pool_attn.q.weight.grad)
        self.assertGreater(model.head_pool_attn.q.weight.grad.abs().sum().item(), 0.0)

        rep = verify_architecture(config)
        self.assertTrue(rep["passed"], rep)


if __name__ == "__main__":
    unittest.main()
