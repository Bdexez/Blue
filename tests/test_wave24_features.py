import math
import unittest

import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.modules import CrossAttention


class Wave24FeaturesTest(unittest.TestCase):
    def test_cross_head_gating_and_talking_heads(self):
        cfg_base = BlueConfig.preset("tiny")
        cfg_gate = BlueConfig.preset("tiny")
        cfg_gate.cross_head_gating = True

        torch.manual_seed(2401)
        m_base = Blue(cfg_base).eval()
        torch.manual_seed(2401)
        m_gate = Blue(cfg_gate).eval()

        # Verify all CrossAttention modules have zero-init head_gate_proj & head_mix
        cross_modules = [mod for mod in m_gate.modules() if isinstance(mod, CrossAttention)]
        self.assertGreater(len(cross_modules), 0)
        for mod in cross_modules:
            self.assertIsNotNone(mod.head_gate_proj)
            self.assertIsNotNone(mod.head_mix)
            self.assertEqual(mod.head_gate_proj.weight.abs().max().item(), 0.0)
            self.assertEqual(mod.head_gate_proj.bias.abs().max().item(), 0.0)
            self.assertEqual(mod.head_mix.weight.abs().max().item(), 0.0)

        ids = torch.tensor([[1, 16, 28, 39]])
        img = torch.randn(1, cfg_gate.image_channels, cfg_gate.image_size, cfg_gate.image_size)
        m_gate.train()
        out = m_gate(ids, images=img, labels=ids)
        out.loss.backward()
        self.assertIsNotNone(m_gate.blocks[0].cross.head_gate_proj.weight.grad)
        self.assertIsNotNone(m_gate.blocks[0].cross.head_mix.weight.grad)
        self.assertGreater(m_gate.blocks[0].cross.head_mix.weight.grad.abs().sum().item(), 0.0)

    def test_moe_routed_scaling_factor_and_learnable_scaling(self):
        cfg = BlueConfig.preset("tiny")
        cfg.moe_routed_scaling_factor = 1.5
        cfg.moe_learnable_scaling = True
        torch.manual_seed(2402)
        model = Blue(cfg).train()
        for block in model.blocks:
            self.assertIsNotNone(block.moe.routed_log_scale)
            self.assertEqual(block.moe.routed_log_scale.item(), 0.0)

        ids = torch.tensor([[1, 14, 29, 44, 59]])
        out = model(ids, labels=ids)
        out.loss.backward()
        grad_sum = sum(
            block.moe.routed_log_scale.grad.abs().item()
            for block in model.blocks
            if block.moe.routed_log_scale.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)

    def test_tail_free_sampling_second_derivative_cutoff(self):
        # Construct distribution with sharp probability drop then flat tail:
        # p0=0.50, p1=0.30, p2=0.05, p3=0.05, p4=0.05, p5=0.05
        # d1 = [0.20, 0.25, 0.00, 0.00, 0.00]
        # |d2| = [0.05, 0.25, 0.00, 0.00] -> normalized |d2| = [1/6, 5/6, 0, 0]
        # cumulative before index: idx0=0, idx1=1/6, idx2=1.0, idx3=1.0...
        # With tfs_z=0.5, tokens at idx >= 2 have prev_cum = 1.0 >= 0.5 -> masked out!
        scores = torch.full((1, 260), -1e9)
        probs_target = [0.50, 0.30, 0.05, 0.05, 0.05, 0.05]
        for idx, p in enumerate(probs_target, start=10):
            scores[0, idx] = math.log(p)

        filtered = Blue._filtered_probs(
            scores, temperature=1.0, top_k=0, top_p=1.0, tfs_z=0.5,
        )
        self.assertGreater(filtered[0, 10].item(), 0.0)
        self.assertGreater(filtered[0, 11].item(), 0.0)
        for idx in range(12, 16):
            self.assertEqual(filtered[0, idx].item(), 0.0)

        cfg = BlueConfig.preset("tiny")
        model = Blue(cfg).eval()
        prompt = torch.tensor([[1, 20, 31]])
        gen = model.generate(prompt, max_new_tokens=5, temperature=0.9, tfs_z=0.85, eos_id=None)
        self.assertEqual(gen.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
