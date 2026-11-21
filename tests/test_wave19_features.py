import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue
from blue.verifier import verify_architecture


class Wave19FeaturesTests(unittest.TestCase):
    def test_cross_qk_temperature_embedding_skip_and_no_repeat_ngram(self):
        for diff in (False, True):
            config = BlueConfig(
                d_model=32,
                n_heads=4,
                n_kv_heads=2,
                n_layers=2,
                hidden_dim=64,
                n_experts=4,
                top_k=2,
                cross_differential_attention=diff,
                cross_qk_temperature=True,
                embedding_output_skip=True,
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
            rep = verify_architecture(config, seed=37)
            self.assertTrue(rep["passed"], rep)

            model = Blue(config).train()
            tokens = torch.randint(0, 260, (2, 10))
            images = torch.randn(2, 3, 16, 16)
            out = model(tokens, images=images, labels=tokens)
            out.loss.backward()
            self.assertIsNotNone(model.embed_skip_gate.grad)
            self.assertGreater(model.embed_skip_gate.grad.abs().sum().item(), 0.0)
            for block in model.blocks:
                self.assertIsNotNone(block.cross.qk_log_scale.grad)
                self.assertGreater(block.cross.qk_log_scale.grad.abs().sum().item(), 0.0)

        # Verify no_repeat_ngram_size blocks repeated n-grams identically across standard, speculative, and Jacobi
        model.eval()
        prompt = torch.tensor([[1, 65, 66, 65, 66]])
        for ngram_n in (1, 2, 3):
            std = model.generate(
                prompt, max_new_tokens=12, temperature=0, eos_id=None,
                no_repeat_ngram_size=ngram_n,
            )
            spec = model.generate(
                prompt, max_new_tokens=12, temperature=0, eos_id=None,
                speculative=True, no_repeat_ngram_size=ngram_n,
            )
            jac = model.generate(
                prompt, max_new_tokens=12, temperature=0, eos_id=None,
                jacobi=True, jacobi_window=3, no_repeat_ngram_size=ngram_n,
            )
            self.assertTrue(torch.equal(std, spec), f"Speculative mismatch for n={ngram_n}")
            self.assertTrue(torch.equal(std, jac), f"Jacobi mismatch for n={ngram_n}")

            # Check no generated n-gram repeats any previous n-gram
            seq = std[0].tolist()
            p_len = prompt.size(1)
            for pos in range(p_len, len(seq)):
                if pos + 1 >= ngram_n:
                    cur_gram = tuple(seq[pos - ngram_n + 1:pos + 1])
                    prior_grams = {
                        tuple(seq[i:i + ngram_n])
                        for i in range(0, pos - ngram_n + 1)
                    }
                    self.assertNotIn(cur_gram, prior_grams)


if __name__ == "__main__":
    unittest.main()
