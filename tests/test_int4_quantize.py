from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import main
from blue.data import synthetic_batch
from blue.quantize import Int4Linear, quantize_model_int4
from blue.training import load_checkpoint, save_checkpoint


class Int4QuantizationTests(unittest.TestCase):
    def test_packed_int4_compression_and_cli_export_roundtrip(self):
        torch.manual_seed(430)
        cfg = BlueConfig.preset("tiny")
        model = Blue(cfg).eval()
        batch = synthetic_batch(cfg, batch_size=2, seed=431)
        inputs = {k: v for k, v in batch.items() if k not in ("labels", "class_labels", "forecast_targets")}
        with torch.no_grad():
            ref = model(**inputs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp32_path = root / "fp32.pt"
            int4_path = root / "int4.pt"
            save_checkpoint(fp32_path, model)

            main([
                "export", "--checkpoint", str(fp32_path), "--output", str(int4_path),
                "--quantize-int4", "--int8-min-numel", "512", "--int4-group-size", "32",
            ])
            restored, payload = load_checkpoint(int4_path)
            stats = payload["trainer"]["quantized_int4"]
            self.assertGreater(stats["quantized_modules"], 0)
            self.assertGreater(stats["compression_ratio"], 6.0)
            self.assertTrue(any(isinstance(m, Int4Linear) for m in restored.modules()))
            # Packed uint8 buffer truly halves the byte count vs int8 and shrinks file size dramatically!
            self.assertLess(int4_path.stat().st_size, fp32_path.stat().st_size * 0.35)

            restored.eval()
            with torch.no_grad():
                out_q = restored(**inputs)
            self.assertTrue(torch.isfinite(out_q.logits).all())
            self.assertTrue(torch.allclose(ref.logits, out_q.logits, atol=0.35))

        # Direct unit check with non-divisible in_features padding!
        lin = torch.nn.Linear(37, 19, bias=True)
        q_lin = Int4Linear.from_linear(lin, group_size=16)
        self.assertEqual(q_lin.weight_packed.shape, (19, 24))
        x = torch.randn(3, 37)
        torch.testing.assert_close(lin(x), q_lin(x), rtol=0.15, atol=0.15)


if __name__ == "__main__":
    unittest.main()
