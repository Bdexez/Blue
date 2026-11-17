from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import main
from blue.data import synthetic_batch
from blue.optimizer import MuonAdamW, build_optimizer, newton_schulz_orthogonalize
from blue.training import load_checkpoint


class MuonOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_newton_schulz_orthogonalizes_tall_wide_square_and_zero_matrices(self):
        torch.manual_seed(9701)
        for shape in ((16, 16), (32, 12), (12, 32)):
            with self.subTest(shape=shape):
                g = torch.randn(*shape)
                ortho = newton_schulz_orthogonalize(g, steps=5)
                self.assertEqual(ortho.shape, shape)
                self.assertTrue(torch.isfinite(ortho).all())
                s = torch.linalg.svdvals(ortho)
                # Quintic Newton-Schulz maps non-zero singular values near 1.
                self.assertGreater(s.min().item(), 0.5)
                self.assertLess(s.max().item(), 1.35)
        zeros = newton_schulz_orthogonalize(torch.zeros(8, 8))
        torch.testing.assert_close(zeros, torch.zeros(8, 8))
        with self.assertRaises(ValueError):
            newton_schulz_orthogonalize(torch.randn(8))

    def test_muon_learns_all_three_losses(self):
        torch.manual_seed(9702)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128)
        model = Blue(config)
        batch = synthetic_batch(config, batch_size=3, seed=9703)
        optimizer = build_optimizer(model, name="muon", lr=3e-3, weight_decay=0.01)
        self.assertIsInstance(optimizer, MuonAdamW)
        # Embedding/lm_head must NOT be in a Muon group.
        muon_ids = {id(p) for g in optimizer.param_groups if g["use_muon"] for p in g["params"]}
        self.assertNotIn(id(model.embedding.weight), muon_ids)
        self.assertGreater(len(muon_ids), 0)
        initial = model(**batch)
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        final = model(**batch)
        for name in ("text", "classification", "forecast"):
            self.assertLess(final.losses[name].item(), initial.losses[name].item(), name)

    def test_muon_exact_resume_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            common = ["--threads", "2", "train", "--preset", "tiny", "--steps", "6",
                      "--batch-size", "2", "--eval-every", "6", "--seed", "55"]
            full, split = root / "full", root / "split"
            main(common + ["--optimizer", "muon", "--output", str(full)])
            main(common + ["--optimizer", "muon", "--output", str(split), "--stop-after", "3"])
            main(common + ["--output", str(split), "--resume", str(split / "latest.pt")])
            a, pa = load_checkpoint(full / "latest.pt")
            b, pb = load_checkpoint(split / "latest.pt")
            self.assertEqual(pb["trainer"]["optimizer"], "muon")
            for name, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0, msg=name)
            with self.assertRaisesRegex(ValueError, "optimizer"):
                main(common + ["--optimizer", "adamw", "--output", str(split),
                               "--resume", str(split / "latest.pt")])


if __name__ == "__main__":
    unittest.main()
