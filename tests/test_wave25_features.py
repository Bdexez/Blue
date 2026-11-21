import math
import unittest

import torch

from blue.config import BlueConfig
from blue.model import Blue


class Wave25FeaturesTest(unittest.TestCase):
    def test_cross_logit_softcap_and_moe_input_residual_gate(self):
        cfg = BlueConfig.preset("tiny")
        cfg.cross_logit_softcap = 30.0
        cfg.moe_input_residual_gate = True
        torch.manual_seed(2501)
        model = Blue(cfg).eval()
        for block in model.blocks:
            self.assertEqual(block.cross.logit_softcap, 30.0)
            self.assertIsNotNone(block.moe_pre_gate)
            self.assertTrue(torch.allclose(block.moe_pre_gate, torch.full_like(block.moe_pre_gate, -3.0)))

        ids = torch.tensor([[1, 15, 29, 43, 57, 71]])
        img = torch.randn(1, cfg.image_channels, cfg.image_size, cfg.image_size)
        with torch.no_grad():
            full_logits = model(ids, images=img).logits
            prefill = model(ids[:, :3], images=img, use_cache=True)
            cache = prefill.cache
            pieces = [prefill.logits]
            for t in range(3, ids.size(1)):
                step_out = model(ids[:, t:t + 1], cache=cache, use_cache=True)
                cache = step_out.cache
                pieces.append(step_out.logits)
            cached_logits = torch.cat(pieces, dim=1)
        self.assertTrue(torch.allclose(full_logits, cached_logits, atol=1e-5))

        model.train()
        out = model(ids, images=img, labels=ids)
        out.loss.backward()
        self.assertIsNotNone(model.blocks[0].moe_pre_gate.grad)
        self.assertGreater(model.blocks[0].moe_pre_gate.grad.abs().sum().item(), 0.0)

    def test_top_a_quadratic_max_probability_sampling(self):
        # Distribution with p_max = 0.70, p1 = 0.22, p2 = 0.08:
        # With top_a = 0.25: threshold = 0.25 * (0.70^2) = 0.1225
        # -> p2 (0.08 < 0.1225) is pruned, while p0 (0.70) and p1 (0.22) are kept!
        scores_high = torch.full((1, 260), -1e9)
        scores_high[0, 10] = math.log(0.70)
        scores_high[0, 11] = math.log(0.22)
        scores_high[0, 12] = math.log(0.08)
        probs_high = Blue._filtered_probs(
            scores_high, temperature=1.0, top_k=0, top_p=1.0, top_a=0.25,
        )
        self.assertGreater(probs_high[0, 10].item(), 0.0)
        self.assertGreater(probs_high[0, 11].item(), 0.0)
        self.assertEqual(probs_high[0, 12].item(), 0.0)

        # Uncertain distribution across 10 tokens where p_max = 0.16 and 9 tokens have p = 0.0933:
        # threshold = 0.25 * (0.16^2) = 0.0064 < 0.0933 -> all 10 tokens are KEPT!
        scores_low = torch.full((1, 260), -1e9)
        scores_low[0, 10] = math.log(0.16)
        scores_low[0, 11:20] = math.log(0.84 / 9.0)
        probs_low = Blue._filtered_probs(
            scores_low, temperature=1.0, top_k=0, top_p=1.0, top_a=0.25,
        )
        self.assertEqual(int((probs_low[0, 10:20] > 0).sum().item()), 10)

        cfg = BlueConfig.preset("tiny")
        model = Blue(cfg).eval()
        prompt = torch.tensor([[1, 19, 34]])
        gen = model.generate(prompt, max_new_tokens=5, temperature=0.9, top_a=0.2, eos_id=None)
        self.assertEqual(gen.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
