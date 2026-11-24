import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.verifier import verify_architecture


class Wave21FeaturesTests(unittest.TestCase):
    def test_layer_scale_alignment_margin_and_typical_sampling(self):
        for geom in ("euclidean", "hyperbolic"):
            config = BlueConfig(
                d_model=32,
                n_heads=4,
                n_kv_heads=2,
                n_layers=2,
                hidden_dim=64,
                n_experts=4,
                top_k=2,
                layer_scale_init=0.5,
                alignment_geometry=geom,
                alignment_margin=0.05,
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
            rep = verify_architecture(config, seed=43)
            self.assertTrue(rep["passed"], rep)

            model = Blue(config).train()
            tokens = torch.randint(0, 260, (2, 10))
            images = torch.randn(2, 3, 16, 16)
            series = torch.randn(2, 12, 2)
            out = model(tokens, images=images, series=series, labels=tokens)
            self.assertIn("alignment", out.losses)
            self.assertGreater(out.losses["alignment"].item(), 0.0)
            out.loss.backward()
            for block in model.blocks:
                self.assertIsNotNone(block.ls_attn.grad)
                self.assertIsNotNone(block.ls_cross.grad)
                self.assertIsNotNone(block.ls_moe.grad)
                self.assertGreater(block.ls_attn.grad.abs().sum().item(), 0.0)
                self.assertGreater(block.ls_cross.grad.abs().sum().item(), 0.0)
                self.assertGreater(block.ls_moe.grad.abs().sum().item(), 0.0)

        # Verify locally typical sampling filters atypical tokens while preserving valid distribution
        scores = torch.full((1, 260), -12.0)
        scores[0, 65] = 3.0
        scores[0, 66] = 1.5
        scores[0, 67] = 1.4
        scores[0, 68] = -2.0
        probs = Blue._filtered_probs(scores, temperature=1.0, top_k=0, top_p=1.0, typical_p=0.5)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)
        self.assertLess((probs > 0).sum().item(), 4)

        model.eval()
        prompt = torch.tensor([[1, 66, 108]])
        # Sans eos : la forme ne dépend pas d'un éventuel tirage du token de fin.
        g_std = model.generate(prompt, max_new_tokens=6, temperature=0.8, typical_p=0.6, eos_id=None)
        g_spec = model.generate(prompt, max_new_tokens=6, temperature=0.8, typical_p=0.6, speculative=True,
                                eos_id=None)
        self.assertEqual(g_std.shape, (1, 9))
        self.assertEqual(g_spec.shape, (1, 9))


if __name__ == "__main__":
    unittest.main()
