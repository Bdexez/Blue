from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class FlowMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_and_input_validation(self):
        base = BlueConfig.preset("tiny")
        for bad in (
            dict(flow_matching_dim=-1),
            dict(flow_matching_dim=True),
            dict(flow_matching_layers=0),
            dict(flow_matching_weight=-0.1),
            dict(flow_cfg_dropout=-0.1),
            dict(flow_cfg_dropout=1.0),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, **bad)
        model = Blue(base).eval()
        with self.assertRaises(ValueError):
            model(input_ids=torch.tensor([[1, 20]]), flow_targets=torch.randn(1, 8))
        with self.assertRaises(ValueError):
            model.sample_flow(input_ids=torch.tensor([[1, 20]]))

    def test_cfg_dropout_guidance_scale_and_rk4_accuracy(self):
        torch.manual_seed(390)
        cfg = replace(BlueConfig.preset("tiny"), flow_matching_dim=8, flow_matching_layers=1,
                      flow_cfg_dropout=0.5)
        model = Blue(cfg)
        batch = synthetic_batch(cfg, batch_size=4, seed=391)
        targets = torch.randn(4, 8)

        # CFG dropout trains null_cond alongside conditional path.
        out = model(**batch, flow_targets=targets)
        out.loss.backward()
        self.assertIsNotNone(model.flow_head.null_cond.grad)
        self.assertGreater(model.flow_head.null_cond.grad.abs().sum().item(), 0.0)

        # Guidance scale != 1.0 shifts samples smoothly vs guidance_scale == 1.0, and RK4 works.
        inputs = {k: v for k, v in batch.items() if k not in ("labels", "class_labels", "forecast_targets")}
        noise = torch.randn(4, 8)
        s_uncond = model.sample_flow(steps=4, solver="rk4", guidance_scale=0.0, initial_noise=noise, **inputs)
        s_cond = model.sample_flow(steps=4, solver="rk4", guidance_scale=1.0, initial_noise=noise, **inputs)
        s_cfg = model.sample_flow(steps=4, solver="rk4", guidance_scale=2.5, initial_noise=noise, **inputs)
        self.assertTrue(torch.isfinite(s_cfg).all())
        self.assertGreater((s_cfg - s_cond).abs().max().item(), 1e-5)
        self.assertGreater((s_cond - s_uncond).abs().max().item(), 1e-5)

        # Analytical ODE check: dz/dt = z, z(0) = 1 -> exact z(1) = e.
        head = model.flow_head.eval()
        orig_velocity = head.velocity
        try:
            head.velocity = lambda z_t, t, cond: z_t
            z0 = torch.ones(2, 8)
            cond = torch.zeros(2, cfg.d_model)
            exact = torch.full((2, 8), float(torch.e))
            err_euler = (head.sample(cond, steps=4, solver="euler", initial_noise=z0) - exact).abs().max().item()
            err_heun = (head.sample(cond, steps=4, solver="heun", initial_noise=z0) - exact).abs().max().item()
            err_rk4 = (head.sample(cond, steps=4, solver="rk4", initial_noise=z0) - exact).abs().max().item()
            self.assertLess(err_heun, err_euler * 0.15)
            self.assertLess(err_rk4, err_heun * 0.02)
            self.assertLess(err_rk4, 1e-4)
        finally:
            head.velocity = orig_velocity

    def test_adaln_zero_learning_ode_solvers_5modalities_and_checkpoint(self):
        torch.manual_seed(9951)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         audio_enabled=True, audio_patch_size=8, audio_layers=1,
                         graph_enabled=True, graph_node_features=6, graph_edge_features=3,
                         flow_matching_dim=12, flow_matching_layers=2,
                         stable_rms_norm=True)
        info = inspect_config(config)
        self.assertIn("flow_head", info["modules"])
        self.assertEqual(info["architecture_features"]["flow_matching_dim"], 12)
        model = Blue(config)
        # Verify AdaLN-Zero modulation starts at zero.
        for block in model.flow_head.blocks:
            self.assertEqual(block.modulation.weight.abs().sum().item(), 0.0)
            self.assertEqual(block.modulation.bias.abs().sum().item(), 0.0)
        self.assertTrue(model.flow_head.final_norm.stable)

        # Train conditional flow matching on class-conditioned target modes.
        batch = synthetic_batch(config, batch_size=4, seed=9952)
        labels = batch["class_labels"].tolist()
        target_vectors = torch.stack([
            torch.full((12,), 2.0 if label % 2 == 0 else -2.0) for label in labels
        ])
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        initial_loss = model(**batch, flow_targets=target_vectors).losses["flow"].item()
        for _ in range(50):
            opt.zero_grad(set_to_none=True)
            out = model(**batch, flow_targets=target_vectors)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        final_loss = model(**batch, flow_targets=target_vectors).losses["flow"].item()
        self.assertLess(final_loss, initial_loss * 0.3)

        # Sampling with Euler and Heun (RK2) solvers conditioned on all 5 modalities.
        inputs = {k: v for k, v in batch.items()
                  if k not in ("labels", "class_labels", "forecast_targets")}
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)
        samples_heun = model.sample_flow(steps=16, solver="heun", generator=gen1, **inputs)
        samples_euler = model.sample_flow(steps=16, solver="euler", generator=gen2, **inputs)
        self.assertEqual(samples_heun.shape, (4, 12))
        self.assertEqual(samples_euler.shape, (4, 12))
        self.assertTrue(torch.isfinite(samples_heun).all())
        self.assertTrue(torch.isfinite(samples_euler).all())
        for idx, label in enumerate(labels):
            if label % 2 == 0:
                self.assertGreater(samples_heun[idx].mean().item(), 0.5)
            else:
                self.assertLess(samples_heun[idx].mean().item(), -0.5)

        # Non-finite target rows are safely masked without NaN, and bf16 backward works.
        partial_targets = target_vectors.clone()
        partial_targets[0, 3] = float("nan")
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch, flow_targets=partial_targets)
        self.assertTrue(torch.isfinite(out.losses["flow"]))
        out.loss.backward()
        self.assertGreater(model.flow_head.out_proj.weight.grad.abs().sum().item(), 0.0)

        # Checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flow.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertEqual(restored.config.flow_matching_dim, 12)
            noise = torch.randn(4, 12)
            torch.testing.assert_close(
                model.sample_flow(steps=8, solver="heun", initial_noise=noise, **inputs),
                restored.sample_flow(steps=8, solver="heun", initial_noise=noise, **inputs),
                rtol=0, atol=0,
            )


if __name__ == "__main__":
    unittest.main()
