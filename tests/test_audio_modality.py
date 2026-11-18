from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import JointDataset, collate_records, synthetic_batch
from blue.modalities import balanced_modalities
from blue.training import load_checkpoint, save_checkpoint


class AudioModalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_and_input_validation(self):
        base = BlueConfig.preset("tiny")
        with self.assertRaises(ValueError):
            replace(base, audio_enabled=1)
        with self.assertRaises(ValueError):
            replace(base, audio_patch_size=0)
        # Audio rejected when audio_enabled=False.
        model = Blue(base).eval()
        with self.assertRaises(ValueError):
            model(audio=torch.randn(1, 1, 32))
        # Enabled model validates shape, length, finiteness, and mask.
        cfg = replace(base, audio_enabled=True, audio_channels=2, max_audio_len=64)
        model = Blue(cfg).eval()
        with self.assertRaises(ValueError):
            model(audio=torch.randn(1, 1, 32))
        with self.assertRaises(ValueError):
            model(audio=torch.randn(1, 2, 65))
        bad = torch.randn(1, 2, 32)
        bad[0, 0, 5] = float("nan")
        with self.assertRaises(ValueError):
            model(audio=bad)
        with self.assertRaises(ValueError):
            model(audio_mask=torch.ones(1, 32, dtype=torch.bool))

    def test_all_15_combinations_padding_invariance_alignment_and_gradients(self):
        torch.manual_seed(9501)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         audio_enabled=True, audio_channels=2, audio_patch_size=8,
                         audio_layers=1, temporal_spectral_mixing=True,
                         stable_rms_norm=True, gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertIn("audio", info["modules"])
        self.assertGreater(info["modules"]["audio"], 0)
        model = Blue(config).eval()
        batch = synthetic_batch(config, batch_size=3, seed=9502)
        self.assertIn("audio", batch)
        self.assertIn("audio_mask", batch)

        # All 15 non-empty subsets of the 4 modalities work cleanly.
        modalities = ("input_ids", "images", "series", "audio")
        for count in range(1, 5):
            for subset in itertools.combinations(modalities, count):
                kwargs = {name: batch[name] for name in subset}
                if "audio" in subset:
                    kwargs["audio_mask"] = batch["audio_mask"]
                with self.subTest(subset=subset), torch.no_grad():
                    out = model(**kwargs)
                    self.assertEqual(out.classification_logits.shape, (3, config.num_classes))
                    self.assertTrue(torch.isfinite(out.classification_logits).all())
                    self.assertTrue(torch.isfinite(out.forecast).all())

        # Balanced modality schedule covers all 15 combinations per cycle.
        seen = set()
        for step in range(15):
            picked = balanced_modalities(batch, step, seed=11)
            seen.add(tuple(m for m in modalities if m in picked))
        self.assertEqual(len(seen), 15)

        # Right-padding audio by whole frames leaves outputs strictly invariant.
        audio = batch["audio"][:, :, :32]
        mask = torch.ones(3, 32, dtype=torch.bool)
        padded_audio = torch.nn.functional.pad(audio, (0, 16), value=42.0)
        padded_mask = torch.nn.functional.pad(mask, (0, 16), value=False)
        with torch.no_grad():
            base_out = model(audio=audio, audio_mask=mask)
            pad_out = model(audio=padded_audio, audio_mask=padded_mask)
        torch.testing.assert_close(base_out.classification_logits, pad_out.classification_logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(base_out.forecast, pad_out.forecast, rtol=1e-5, atol=1e-6)

        # encode_modalities returns all 4 normalized embeddings.
        with torch.no_grad():
            embeds = model.encode_modalities(
                input_ids=batch["input_ids"], images=batch["images"],
                series=batch["series"], audio=audio,
                attention_mask=batch["attention_mask"],
                series_mask=batch["series_mask"], audio_mask=mask,
            )
        self.assertEqual(set(embeds), {"text", "image", "series", "audio"})
        for vec in embeds.values():
            self.assertEqual(vec.shape, (3, config.d_model))
            torch.testing.assert_close(vec.norm(dim=-1), torch.ones(3), rtol=1e-5, atol=1e-5)

        # Gradients reach audio encoder, audio_modality, and audio_alignment under bf16.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertGreater(model.audio_modality.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.audio_alignment.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.audio.stem.weight.grad.abs().sum().item(), 0.0)

        # JSONL loading with variable audio lengths + generation conditioned on audio + checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "audio.jsonl"
            rows = [
                {"audio": [[0.1] * 24, [0.2] * 24], "class_label": 0},
                {"audio": [[0.3] * 35, [0.4] * 35], "class_label": 1},
            ]
            jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            ds = JointDataset(jsonl, config, allow_mixed=True)
            collated = collate_records([ds[0], ds[1]])
            self.assertEqual(collated["audio"].shape, (2, 2, 35))
            self.assertEqual(collated["audio_mask"].sum(dim=-1).tolist(), [24, 35])
            ckpt = root / "audio.pt"
            save_checkpoint(ckpt, model.eval())
            restored, _ = load_checkpoint(ckpt)
            self.assertTrue(restored.config.audio_enabled)
            gen = restored.generate(batch["input_ids"][:1, :5], audio=audio[:1], audio_mask=mask[:1],
                                    max_new_tokens=4, temperature=0)
            self.assertEqual(gen.shape, (1, 9))


if __name__ == "__main__":
    unittest.main()
