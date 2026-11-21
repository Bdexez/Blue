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

    def test_jepa_vicreg_exact_covariance_decorrelation(self):
        from blue.jepa import MultimodalJEPAPredictor
        torch.manual_seed(2702)
        pred_no_cov = MultimodalJEPAPredictor(32, variance_weight=0.0, covariance_weight=0.0)
        pred_cov = MultimodalJEPAPredictor(32, variance_weight=0.0, covariance_weight=1.0)
        pred_cov.load_state_dict(pred_no_cov.state_dict())

        v0 = torch.randn(4, 32, requires_grad=True)
        v1 = torch.randn(4, 32, requires_grad=True)
        loss_base = pred_no_cov([(0, v0), (1, v1)])
        loss_cov = pred_cov([(0, v0), (1, v1)])
        cov_diff = (loss_cov - loss_base).item()

        # Verify exact algebraic match with explicit D x D covariance matrix
        manual_covs = []
        for mod_id, ctx in ((0, v1), (1, v0)):
            z = pred_cov.predict(ctx, mod_id).float()
            zc = z - z.mean(dim=0, keepdim=True)
            cov = (zc.t() @ zc) / (z.size(0) - 1)
            off = cov - torch.diag(torch.diag(cov))
            manual_covs.append((off.square().sum() / z.size(-1)).item())
        self.assertAlmostEqual(cov_diff, sum(manual_covs) / 2.0, places=5)

        cfg = BlueConfig.preset("tiny")
        cfg.jepa_weight = 0.1
        cfg.jepa_cov_weight = 0.04
        model = Blue(cfg).train()
        ids = torch.tensor([[1, 10, 20, 30], [1, 12, 22, 32], [1, 14, 24, 34]])
        img = torch.randn(3, cfg.image_channels, cfg.image_size, cfg.image_size)
        out = model(ids, images=img, labels=ids)
        self.assertIn("jepa", out.losses)
        out.loss.backward()
        self.assertIsNotNone(model.jepa.out_proj.weight.grad)
        self.assertGreater(model.jepa.out_proj.weight.grad.abs().sum().item(), 0.0)

    def test_episodic_memory_adaptive_decay_cache_and_padding_invariance(self):
        cfg = BlueConfig.preset("tiny")
        cfg.memory_slots = 6
        cfg.memory_decay = True
        torch.manual_seed(2703)
        model = Blue(cfg).eval()
        self.assertIsNotNone(model.episodic_memory.decay_gate)
        self.assertTrue(torch.allclose(
            model.episodic_memory.decay_gate.bias,
            torch.full_like(model.episodic_memory.decay_gate.bias, 4.0),
        ))
        with torch.no_grad():
            model.episodic_memory.decay_gate.weight.normal_(std=0.1)

        ids = torch.tensor([[1, 14, 28, 42, 56, 70]])
        with torch.no_grad():
            full_logits = model(ids).logits
            prefill = model(ids[:, :3], use_cache=True)
            cache = prefill.cache
            pieces = [prefill.logits]
            for t in range(3, ids.size(1)):
                step_out = model(ids[:, t:t + 1], cache=cache, use_cache=True)
                cache = step_out.cache
                pieces.append(step_out.logits)
            cached_logits = torch.cat(pieces, dim=1)
        self.assertTrue(torch.allclose(full_logits, cached_logits, atol=1e-5))

        model.train()
        out = model(ids, labels=ids)
        out.loss.backward()
        self.assertIsNotNone(model.episodic_memory.decay_gate.weight.grad)
        self.assertGreater(model.episodic_memory.decay_gate.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
