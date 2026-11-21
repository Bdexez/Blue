import unittest

import torch

from blue.config import BlueConfig
from blue.model import Blue


class Wave27FlowTest(unittest.TestCase):
    def test_logit_normal_time_schedule_midpoint_and_cosine_grid(self):
        cfg = BlueConfig.preset("tiny")
        cfg.flow_matching_dim = 8
        cfg.flow_cfg_dropout = 0.1
        cfg.flow_time_schedule = "logit_normal"
        torch.manual_seed(2701)
        model = Blue(cfg).train()
        ids = torch.tensor([[1, 12, 24, 36], [1, 15, 25, 35]])
        targets = torch.randn(2, 8)
        out = model(ids, flow_targets=targets)
        self.assertIn("flow", out.losses)
        self.assertTrue(torch.isfinite(out.losses["flow"]))
        out.loss.backward()

        model.eval()
        noise = torch.randn(2, 8)
        s_mid = model.sample_flow(input_ids=ids, steps=6, solver="midpoint", time_grid="cosine",
                                  guidance_scale=1.5, initial_noise=noise)
        s_rk4 = model.sample_flow(input_ids=ids, steps=6, solver="rk4", time_grid="cosine",
                                  guidance_scale=1.5, initial_noise=noise)
        self.assertEqual(s_mid.shape, (2, 8))
        self.assertTrue(torch.isfinite(s_mid).all())
        self.assertTrue(torch.isfinite(s_rk4).all())


if __name__ == "__main__":
    unittest.main()
