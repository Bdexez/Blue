import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from blue.cli import main
from blue.config import BlueConfig
from blue.model import Blue
from blue.training import save_checkpoint


class JacobiDecodingTests(unittest.TestCase):
    def test_jacobi_matches_greedy_with_and_without_future_heads_and_full_hybrid_state(self):
        torch.manual_seed(0)
        for prediction_steps in (1, 3):
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
                max_seq_len=64,
                max_series_len=16,
                forecast_horizon=4,
                forecast_context=4,
                series_features=2,
                conv_kernel=3,
                sliding_window=8,
                global_every=1,
                prediction_steps=prediction_steps,
                ngram_vocab_size=64,
                kv_lora_rank=12,
                ssm_state_dim=8,
                ttt_state_dim=8,
                memory_slots=4,
                value_residual=True,
                attention_sinks=2,
                nsa_block_size=4,
                nsa_top_k=2,
                dropout=0.0,
            )
            model = Blue(config).eval()
            prompt = torch.tensor([[1, 66, 108, 117, 101]])
            images = torch.randn(1, 3, 16, 16)
            series = torch.randn(1, 12, 2)

            baseline = model.generate(
                prompt, max_new_tokens=15, temperature=0, eos_id=None,
                images=images, series=series,
            )
            for window in (1, 2, 4, 6):
                jacobi_out = model.generate(
                    prompt, max_new_tokens=15, temperature=0, eos_id=None,
                    jacobi=True, jacobi_window=window,
                    images=images, series=series,
                )
                self.assertTrue(
                    torch.equal(baseline, jacobi_out),
                    f"Jacobi mismatch for prediction_steps={prediction_steps}, window={window}",
                )

    def test_jacobi_validation_and_cli_integration(self):
        config = BlueConfig.preset("tiny")
        model = Blue(config).eval()
        prompt = torch.tensor([[1, 65, 66]])

        with self.assertRaisesRegex(ValueError, "temperature=0"):
            model.generate(prompt, max_new_tokens=4, temperature=0.7, jacobi=True)
        with self.assertRaisesRegex(ValueError, "jacobi_window"):
            model.generate(prompt, max_new_tokens=4, temperature=0, jacobi=True, jacobi_window=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "tiny.pt"
            save_checkpoint(ckpt, model, step=1)
            buf1 = io.StringIO()
            with redirect_stdout(buf1):
                main([
                    "generate", "--checkpoint", str(ckpt), "--prompt", "Blue",
                    "--max-new-tokens", "10", "--temperature", "0",
                ])
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                main([
                    "generate", "--checkpoint", str(ckpt), "--prompt", "Blue",
                    "--max-new-tokens", "10", "--temperature", "0",
                    "--jacobi", "--jacobi-window", "3",
                ])
            self.assertEqual(buf1.getvalue(), buf2.getvalue())


if __name__ == "__main__":
    unittest.main()
