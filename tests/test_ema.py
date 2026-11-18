from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import main
from blue.ema import ModelEMA
from blue.training import load_checkpoint


class ModelEMATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ema_validation_and_apply_to_restores_live_weights(self):
        model = Blue(BlueConfig.preset("tiny"))
        for bad in (0.0, 1.0, -0.5, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ModelEMA(model, bad)
        ema = ModelEMA(model, 0.9)
        self.assertNotIn("lm_head.weight", ema.shadow)
        initial = model.embedding.weight.detach().clone()
        with torch.no_grad():
            model.embedding.weight.add_(1.0)
        live = model.embedding.weight.detach().clone()
        ema.update(model)
        self.assertFalse(torch.allclose(ema.shadow["embedding.weight"], live))
        with ema.apply_to(model):
            torch.testing.assert_close(model.embedding.weight, ema.shadow["embedding.weight"])
            self.assertIs(model.lm_head.weight, model.embedding.weight)
        torch.testing.assert_close(model.embedding.weight, live)
        self.assertIs(model.lm_head.weight, model.embedding.weight)
        self.assertFalse(torch.allclose(ema.shadow["embedding.weight"], initial))

    def test_exact_resume_and_export_use_ema(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            common = ["--threads", "2", "train", "--preset", "tiny", "--steps", "6",
                      "--batch-size", "2", "--eval-every", "3", "--seed", "91"]
            full, split = root / "full", root / "split"
            main(common + ["--ema-decay", "0.95", "--output", str(full)])
            main(common + ["--ema-decay", "0.95", "--output", str(split), "--stop-after", "3"])
            main(common + ["--output", str(split), "--resume", str(split / "latest.pt")])
            a, pa = load_checkpoint(full / "latest.pt")
            b, pb = load_checkpoint(split / "latest.pt")
            self.assertEqual(pb["trainer"]["ema_decay"], 0.95)
            self.assertEqual(pa["ema"]["updates"], 6)
            self.assertEqual(pb["ema"]["updates"], 6)
            for name, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0, msg=name)
            for name, tensor in pa["ema"]["shadow"].items():
                torch.testing.assert_close(tensor, pb["ema"]["shadow"][name], rtol=0, atol=0, msg=name)
            # Mismatch on resume is rejected.
            with self.assertRaisesRegex(ValueError, "ema_decay"):
                main(common + ["--ema-decay", "0.5", "--output", str(split),
                               "--resume", str(split / "latest.pt")])
            # Export --use-ema writes EMA weights and preserves tied lm_head.
            exported = root / "ema_model.pt"
            main(["--threads", "2", "export", "--checkpoint", str(full / "latest.pt"),
                  "--use-ema", "--output", str(exported)])
            ema_model, ema_payload = load_checkpoint(exported)
            self.assertTrue(ema_payload["trainer"]["exported_ema"])
            self.assertIs(ema_model.lm_head.weight, ema_model.embedding.weight)
            torch.testing.assert_close(ema_model.embedding.weight, pa["ema"]["shadow"]["embedding.weight"],
                                       rtol=0, atol=0)
            self.assertFalse(torch.allclose(ema_model.embedding.weight, a.embedding.weight))


if __name__ == "__main__":
    unittest.main()
