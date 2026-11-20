import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from blue.cli import main
from blue.config import BlueConfig
from blue.model import Blue
from blue.training import learning_rate, load_checkpoint


class WSDAndCheckpointPolicyTests(unittest.TestCase):
    def test_wsd_schedule_curve_and_exact_cli_resume(self):
        # Warmup 2 steps (0, 1), stable plateau 6 steps (2..7), decay 2 steps (8, 9)
        lrs = [
            learning_rate(s, 10, peak=1e-3, warmup=2, schedule="wsd", wsd_decay_fraction=0.25)
            for s in range(10)
        ]
        self.assertAlmostEqual(lrs[0], 0.5e-3)
        self.assertAlmostEqual(lrs[1], 1.0e-3)
        for s in range(2, 8):
            self.assertAlmostEqual(lrs[s], 1.0e-3)
        self.assertLess(lrs[8], 1.0e-3)
        self.assertGreater(lrs[8], 0.1e-3)
        self.assertAlmostEqual(lrs[9], 0.1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            full_dir = Path(tmpdir) / "full"
            part_dir = Path(tmpdir) / "part"
            with redirect_stdout(io.StringIO()):
                main([
                    "train", "--preset", "tiny", "--steps", "4", "--stop-after", "2",
                    "--batch-size", "2", "--warmup", "1", "--lr-schedule", "wsd",
                    "--wsd-decay-fraction", "0.5", "--eval-every", "4", "--save-every", "2",
                    "--output", str(part_dir),
                ])
                # Reject resuming with a different schedule
                with self.assertRaisesRegex(ValueError, "lr_schedule"):
                    main([
                        "train", "--preset", "tiny", "--steps", "4",
                        "--batch-size", "2", "--warmup", "1", "--lr-schedule", "cosine",
                        "--eval-every", "4", "--save-every", "2",
                        "--resume", str(part_dir / "latest.pt"), "--output", str(part_dir),
                    ])
                main([
                    "train", "--preset", "tiny", "--steps", "4",
                    "--batch-size", "2", "--warmup", "1",
                    "--eval-every", "4", "--save-every", "2",
                    "--resume", str(part_dir / "latest.pt"), "--output", str(part_dir),
                ])
                main([
                    "train", "--preset", "tiny", "--steps", "4",
                    "--batch-size", "2", "--warmup", "1", "--lr-schedule", "wsd",
                    "--wsd-decay-fraction", "0.5", "--eval-every", "4", "--save-every", "2",
                    "--output", str(full_dir),
                ])
            m_full, _ = load_checkpoint(full_dir / "latest.pt")
            m_part, payload = load_checkpoint(part_dir / "latest.pt")
            self.assertEqual(payload["trainer"]["lr_schedule"], "wsd")
            self.assertEqual(payload["trainer"]["wsd_decay_fraction"], 0.5)
            for (k1, v1), (k2, v2) in zip(m_full.state_dict().items(), m_part.state_dict().items()):
                self.assertEqual(k1, k2)
                self.assertTrue(torch.equal(v1, v2), f"Mismatch after WSD resume on {k1}")

    def test_selective_activation_checkpointing_exact_gradients(self):
        base_kwargs = dict(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=4,
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
            global_every=2,
            value_residual=True,
            residual_streams=2,
            ssm_state_dim=8,
            ttt_state_dim=8,
            dropout=0.0,
        )
        cfg_no_ckpt = BlueConfig(**base_kwargs, gradient_checkpointing=False)
        cfg_every_1 = BlueConfig(**base_kwargs, gradient_checkpointing=True, checkpoint_every=1)
        cfg_every_2 = BlueConfig(**base_kwargs, gradient_checkpointing=True, checkpoint_every=2)

        torch.manual_seed(42)
        m0 = Blue(cfg_no_ckpt).train()
        m1 = Blue(cfg_every_1).train()
        m2 = Blue(cfg_every_2).train()
        m1.load_state_dict(m0.state_dict())
        m2.load_state_dict(m0.state_dict())

        tokens = torch.randint(0, 260, (2, 10))
        images = torch.randn(2, 3, 16, 16)
        series = torch.randn(2, 12, 2)

        out0 = m0(tokens, images=images, series=series, labels=tokens)
        out0.loss.backward()
        out1 = m1(tokens, images=images, series=series, labels=tokens)
        out1.loss.backward()
        out2 = m2(tokens, images=images, series=series, labels=tokens)
        out2.loss.backward()

        self.assertTrue(torch.allclose(out0.loss, out1.loss, atol=1e-6))
        self.assertTrue(torch.allclose(out0.loss, out2.loss, atol=1e-6))
        for (name, p0), (_, p1), (_, p2) in zip(
            m0.named_parameters(), m1.named_parameters(), m2.named_parameters()
        ):
            if p0.grad is not None:
                self.assertTrue(torch.allclose(p0.grad, p1.grad, atol=1e-5), f"Grad mismatch every=1 on {name}")
                self.assertTrue(torch.allclose(p0.grad, p2.grad, atol=1e-5), f"Grad mismatch every=2 on {name}")


if __name__ == "__main__":
    unittest.main()
