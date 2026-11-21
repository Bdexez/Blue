import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue


class EvidentialAndMTPDistillationTests(unittest.TestCase):
    def test_evidential_dirichlet_classification_and_uncertainty_identities(self):
        config = BlueConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            hidden_dim=64,
            n_experts=4,
            top_k=2,
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
            num_classes=5,
            evidential_classification=True,
            mtp_kl_weight=0.05,
            prediction_steps=3,
            dropout=0.0,
        )
        torch.manual_seed(11)
        model = Blue(config)
        tokens = torch.randint(0, 260, (3, 12))
        images = torch.randn(3, 3, 16, 16)
        class_labels = torch.tensor([0, 3, -100], dtype=torch.long)

        out = model(tokens, images=images, labels=tokens, class_labels=class_labels)
        self.assertIn("classification", out.losses)
        self.assertIn("mtp_kl", out.losses)
        self.assertTrue(torch.isfinite(out.losses["classification"]))
        self.assertTrue(torch.isfinite(out.losses["mtp_kl"]))
        self.assertGreater(out.losses["classification"].item(), 0.0)
        self.assertGreaterEqual(out.losses["mtp_kl"].item(), 0.0)

        out.loss.backward()
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0.0)
        for head in model.future_heads:
            self.assertIsNotNone(head[1].weight.grad)
            self.assertGreater(head[1].weight.grad.abs().sum().item(), 0.0)

        # Verify Subjective Logic identity: sum(belief) + uncertainty == 1 and argmax invariance
        unc = model.dirichlet_uncertainty(out.classification_logits.detach())
        self.assertEqual(unc["alpha"].shape, (3, 5))
        self.assertEqual(unc["uncertainty"].shape, (3,))
        self.assertTrue((unc["alpha"] > 1.0).all())
        self.assertTrue(((unc["uncertainty"] > 0.0) & (unc["uncertainty"] <= 1.0)).all())
        total_mass = unc["belief"].sum(dim=-1) + unc["uncertainty"]
        self.assertTrue(torch.allclose(total_mass, torch.ones_like(total_mass), atol=1e-6))
        self.assertTrue(torch.allclose(unc["probabilities"].sum(dim=-1), torch.ones(3), atol=1e-6))
        self.assertTrue(torch.equal(
            out.classification_logits.argmax(dim=-1),
            unc["probabilities"].argmax(dim=-1),
        ))


if __name__ == "__main__":
    unittest.main()
