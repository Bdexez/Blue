import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.verifier import verify_architecture


class CrossModalFusionTests(unittest.TestCase):
    def test_cross_modal_fusion_preserves_all_invariances_and_gradients(self):
        config = BlueConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            hidden_dim=64,
            kan_degree=3,
            n_experts=4,
            top_k=2,
            n_latents=4,
            register_latents=2,
            cross_modal_fusion_layers=2,
            resampler_layers=1,
            hierarchical_resampler=True,
            cross_differential_attention=True,
            cross_kv_lora_rank=16,
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
            audio_enabled=True,
            audio_patch_size=4,
            max_audio_len=32,
            audio_layers=1,
            graph_enabled=True,
            graph_node_features=4,
            graph_edge_features=2,
            max_graph_nodes=12,
            graph_layers=1,
            pointcloud_enabled=True,
            pointcloud_features=2,
            max_pointcloud_points=12,
            pointcloud_layers=1,
            tabular_enabled=True,
            tabular_num_features=4,
            tabular_cat_features=2,
            tabular_cat_cardinality=8,
            tabular_layers=1,
            video_enabled=True,
            video_channels=3,
            video_size=16,
            video_patch_size=8,
            video_tubelet_size=2,
            max_video_frames=6,
            video_layers=1,
            dropout=0.0,
        )
        report = verify_architecture(config, seed=7)
        self.assertTrue(report["passed"], report)
        for name, check in report["checks"].items():
            if isinstance(check, dict) and "passed" in check:
                self.assertTrue(check["passed"], f"Failed check {name}: {check}")

        # Verify gradients flow through fusion_blocks
        model = Blue(config)
        self.assertEqual(len(model.resampler.fusion_blocks), 2)
        images = torch.randn(2, 3, 16, 16)
        series = torch.randn(2, 10, 2)
        labels = torch.randint(0, 260, (2, 8))
        out = model(labels, images=images, series=series, labels=labels)
        out.loss.backward()
        for idx, block in enumerate(model.resampler.fusion_blocks):
            self.assertIsNotNone(block.attn.q.weight.grad, f"Missing grad in fusion block {idx}")
            self.assertGreater(block.attn.q.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
