from pathlib import Path
import tempfile
import unittest

import torch

from blue.cli import main
from blue.data import synthetic_batch
from blue.lora import DoRALinear
from blue.quantize import Int8Linear
from blue.training import load_checkpoint


class DoRACLIAndExportTests(unittest.TestCase):
    def test_dora_train_resume_merge_and_int8_export_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "base"
            dora_dir = root / "dora"
            dora_split_dir = root / "dora_split"

            # 1. Base pretraining (2 steps).
            main([
                "train", "--preset", "tiny", "--steps", "2", "--batch-size", "2",
                "--save-every", "2", "--eval-every", "2", "--seed", "7", "--output", str(base_dir),
            ])
            base_ckpt = base_dir / "latest.pt"

            # 2. DoRA fine-tuning from base_ckpt (4 steps uninterrupted with EMA).
            main([
                "train", "--init-from", str(base_ckpt), "--steps", "4", "--batch-size", "2",
                "--save-every", "4", "--eval-every", "4", "--seed", "13",
                "--dora-rank", "4", "--dora-alpha", "8", "--ema-decay", "0.9",
                "--optimizer", "muon", "--output", str(dora_dir),
            ])

            # 3. Same DoRA run split at step 2 and resumed to step 4 -> bit-exact match!
            main([
                "train", "--init-from", str(base_ckpt), "--steps", "4", "--stop-after", "2",
                "--batch-size", "2", "--save-every", "2", "--eval-every", "4", "--seed", "13",
                "--dora-rank", "4", "--dora-alpha", "8", "--ema-decay", "0.9",
                "--optimizer", "muon", "--output", str(dora_split_dir),
            ])
            main([
                "train", "--resume", str(dora_split_dir / "latest.pt"), "--steps", "4",
                "--batch-size", "2", "--save-every", "4", "--eval-every", "4", "--seed", "13",
                "--output", str(dora_split_dir),
            ])

            m_full, p_full = load_checkpoint(dora_dir / "latest.pt")
            m_res, p_res = load_checkpoint(dora_split_dir / "latest.pt")
            self.assertTrue(any(isinstance(m, DoRALinear) for m in m_full.modules()))
            for (k1, v1), (k2, v2) in zip(m_full.state_dict().items(), m_res.state_dict().items()):
                self.assertEqual(k1, k2)
                self.assertTrue(torch.equal(v1, v2), f"Mismatch on {k1}")
            self.assertEqual(p_full["trainer"]["dora"]["rank"], 4)

            # 4. Export with --use-ema + --merge-dora -> zero DoRALinear modules, exact output match.
            merged_path = root / "merged_ema.pt"
            main([
                "export", "--checkpoint", str(dora_dir / "latest.pt"),
                "--output", str(merged_path), "--use-ema", "--merge-dora",
            ])
            m_merged, p_merged = load_checkpoint(merged_path)
            self.assertFalse(any(isinstance(m, DoRALinear) for m in m_merged.modules()))
            self.assertTrue(p_merged["trainer"]["exported_ema"])
            self.assertGreater(p_merged["trainer"]["merged_dora"], 0)
            self.assertNotIn("dora", p_merged["trainer"])

            # Verify merged output matches EMA-applied DoRA model identically!
            from blue.ema import ModelEMA
            ema = ModelEMA(m_full, p_full["ema"]["decay"])
            ema.load_state_dict(p_full["ema"])
            batch = synthetic_batch(m_merged.config, batch_size=2, seed=99)
            inputs = {k: v for k, v in batch.items() if k not in ("labels", "class_labels", "forecast_targets")}
            m_full.eval()
            m_merged.eval()
            with torch.no_grad(), ema.apply_to(m_full):
                out_dora_ema = m_full(**inputs)
            with torch.no_grad():
                out_merged = m_merged(**inputs)
            self.assertTrue(torch.allclose(out_dora_ema.logits, out_merged.logits, atol=1e-5))
            self.assertTrue(torch.allclose(out_dora_ema.classification_logits, out_merged.classification_logits, atol=1e-5))

            # 5. Export with --merge-dora + --quantize-int8 -> loads directly via load_checkpoint!
            int8_path = root / "merged_int8.pt"
            main([
                "export", "--checkpoint", str(dora_dir / "latest.pt"),
                "--output", str(int8_path), "--use-ema", "--merge-dora", "--quantize-int8",
                "--int8-min-numel", "512",
            ])
            m_int8, p_int8 = load_checkpoint(int8_path)
            self.assertTrue(any(isinstance(m, Int8Linear) for m in m_int8.modules()))
            self.assertGreater(p_int8["trainer"]["quantized_int8"]["compression_ratio"], 2.0)
            self.assertLess(int8_path.stat().st_size, merged_path.stat().st_size)
            m_int8.eval()
            with torch.no_grad():
                out_int8 = m_int8(**inputs)
            self.assertTrue(torch.isfinite(out_int8.logits).all())
            self.assertTrue(torch.allclose(out_merged.logits, out_int8.logits, atol=0.15))

            # 6. Error path: --quantize-int8 on unmerged DoRA without --merge-dora fails cleanly.
            with self.assertRaises(ValueError):
                main([
                    "export", "--checkpoint", str(dora_dir / "latest.pt"),
                    "--output", str(root / "bad.pt"), "--quantize-int8",
                ])


if __name__ == "__main__":
    unittest.main()
