import math
import unittest

import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.observed_objective import observed_objective, target_counts


class Wave23FeaturesTest(unittest.TestCase):
    def test_head_confidence_gate_and_cache_equivalence(self):
        cfg = BlueConfig.preset("tiny")
        cfg.head_confidence_gate = True
        torch.manual_seed(2301)
        model = Blue(cfg).eval()
        for block in model.blocks:
            self.assertIsNotNone(block.attn.head_gate_proj)
            self.assertTrue(torch.allclose(block.attn.head_gate_proj.bias, torch.zeros_like(block.attn.head_gate_proj.bias)))
            with torch.no_grad():
                block.attn.head_gate_proj.weight.normal_(std=0.05)

        ids = torch.tensor([[1, 14, 27, 42, 55, 68]])
        with torch.no_grad():
            full_logits = model(ids).logits
            out_prefill = model(ids[:, :3], use_cache=True)
            cache = out_prefill.cache
            step_logits = [out_prefill.logits]
            for t in range(3, ids.size(1)):
                out_step = model(ids[:, t:t + 1], cache=cache, use_cache=True)
                cache = out_step.cache
                step_logits.append(out_step.logits)
            cached_logits = torch.cat(step_logits, dim=1)
        self.assertTrue(torch.allclose(full_logits, cached_logits, atol=1e-5))

        model.train()
        out = model(ids, labels=ids)
        out.loss.backward()
        self.assertIsNotNone(model.blocks[0].attn.head_gate_proj.weight.grad)
        self.assertGreater(model.blocks[0].attn.head_gate_proj.weight.grad.abs().sum().item(), 0.0)

    def test_label_smoothing_and_observed_objective_completeness(self):
        cfg_hard = BlueConfig.preset("tiny")
        cfg_smooth = BlueConfig.preset("tiny")
        cfg_smooth.label_smoothing = 0.1
        cfg_smooth.moe_entropy_weight = 0.01
        cfg_smooth.moe_router_orthogonality_weight = 0.01
        cfg_smooth.mtp_kl_weight = 0.05
        cfg_smooth.lm_z_loss_weight = 1e-4
        cfg_smooth.forecast_smoothness_weight = 0.02

        torch.manual_seed(2302)
        m_hard = Blue(cfg_hard).train()
        torch.manual_seed(2302)
        m_smooth = Blue(cfg_smooth).train()

        ids = torch.tensor([[1, 11, 22, 33, 44, 55], [1, 15, 25, 35, 45, 55]])
        cls = torch.tensor([0, 1])
        fc_targets = torch.randn(2, cfg_smooth.forecast_horizon, cfg_smooth.series_features)

        out_hard = m_hard(ids, labels=ids, class_labels=cls)
        out_smooth = m_smooth(ids, labels=ids, class_labels=cls, forecast_targets=fc_targets)

        self.assertNotAlmostEqual(out_hard.losses["text"].item(), out_smooth.losses["text"].item(), places=4)
        self.assertNotAlmostEqual(out_hard.losses["classification"].item(), out_smooth.losses["classification"].item(), places=4)

        counts = target_counts(
            {"labels": ids, "class_labels": cls, "forecast_targets": fc_targets},
            cfg_smooth.prediction_steps,
        )
        obs = observed_objective(out_smooth, counts, counts, cfg_smooth, world=1, accumulation=1)
        self.assertAlmostEqual(obs.item(), out_smooth.loss.item(), places=5)

    def test_eta_sampling_entropy_adaptive_cutoff(self):
        # Low-entropy peaked distribution: H(p) ~ 0, so threshold = min(eta, sqrt(eta) * exp(-H)) = eta
        scores_peaked = torch.full((1, 260), -20.0)
        scores_peaked[0, 10] = math.log(0.85)
        scores_peaked[0, 11] = math.log(0.10)
        scores_peaked[0, 12] = math.log(0.05)
        probs_peaked = Blue._filtered_probs(
            scores_peaked, temperature=1.0, top_k=0, top_p=1.0, eta_cutoff=0.08,
        )
        # Token 12 has p=0.05 < 0.08 -> pruned; tokens 10 (0.85) and 11 (0.10) kept
        self.assertGreater(probs_peaked[0, 10].item(), 0.0)
        self.assertGreater(probs_peaked[0, 11].item(), 0.0)
        self.assertEqual(probs_peaked[0, 12].item(), 0.0)

        # High-entropy flat distribution across 20 tokens (each p=0.05, H = ln(20) = 2.9957):
        # sqrt(0.08) * exp(-2.9957) = 0.2828 / 20 = 0.01414 < 0.05 -> all 20 tokens are KEPT!
        scores_flat = torch.full((1, 260), -1e9)
        scores_flat[0, 10:30] = 0.0
        probs_flat = Blue._filtered_probs(
            scores_flat, temperature=1.0, top_k=0, top_p=1.0, eta_cutoff=0.08,
        )
        self.assertEqual(int((probs_flat[0, 10:30] > 0).sum().item()), 20)

        cfg = BlueConfig.preset("tiny")
        model = Blue(cfg).eval()
        prompt = torch.tensor([[1, 18, 29]])
        gen = model.generate(prompt, max_new_tokens=5, temperature=0.9, eta_cutoff=0.05, eos_id=None)
        self.assertEqual(gen.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
