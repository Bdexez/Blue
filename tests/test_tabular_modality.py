import json
from pathlib import Path
import tempfile
import unittest

import torch

from blue.config import BlueConfig
from blue.data import JointDataset, collate_records, synthetic_batch
from blue.modalities import balanced_modalities
from blue.modality_batch_sampler import ModalityBatchSampler
from blue.model import Blue
from blue.profiler import profile_architecture
from blue.quantize import quantize_model_int8


def _seven_modality_config(**overrides):
    base = dict(
        d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=128,
        n_experts=4, top_k=2, max_seq_len=64, sliding_window=16, global_every=2,
        image_size=16, patch_size=4, vision_layers=1, temporal_layers=1,
        max_series_len=32, n_latents=8, resampler_layers=1,
        audio_enabled=True, audio_layers=1, max_audio_len=64,
        graph_enabled=True, graph_layers=1, max_graph_nodes=12, graph_rwse_steps=4,
        pointcloud_enabled=True, pointcloud_features=2, pointcloud_layers=1, max_pointcloud_points=16,
        tabular_enabled=True, tabular_num_features=6, tabular_cat_features=3,
        tabular_cat_cardinality=12, tabular_layers=1, tabular_plr_frequencies=8,
        alignment_geometry="hyperbolic",
    )
    base.update(overrides)
    return BlueConfig(**base)


class TabularModalityTest(unittest.TestCase):
    def test_config_validation_and_backward_compat(self):
        default = BlueConfig.preset("tiny")
        self.assertFalse(default.tabular_enabled)
        model_default = Blue(default)
        self.assertIsNone(model_default.tabular)
        with self.assertRaises(ValueError):
            model_default(tabular_num=torch.zeros(2, default.tabular_num_features))
        for bad in (
            dict(tabular_enabled=1),
            dict(tabular_num_features=0),
            dict(tabular_cat_features=-1),
            dict(tabular_cat_cardinality=1),
            dict(tabular_layers=0),
            dict(tabular_plr_frequencies=0),
        ):
            with self.assertRaises(ValueError):
                BlueConfig(**bad)

    def test_missing_values_and_validation(self):
        torch.manual_seed(0)
        cfg = _seven_modality_config()
        model = Blue(cfg)
        num = torch.randn(3, cfg.tabular_num_features)
        num[0, 1] = float("nan")
        num[2, :] = float("nan")  # Entire row missing numericals!
        cat = torch.tensor([[0, -1, 5], [11, 3, -1], [-1, -1, -1]], dtype=torch.long)
        labels = torch.tensor([0, 1, 2], dtype=torch.long)

        out = model(tabular_num=num, tabular_cat=cat, class_labels=labels)
        self.assertTrue(torch.isfinite(out.loss))
        self.assertTrue(torch.isfinite(out.classification_logits).all())
        out.loss.backward()
        self.assertIsNotNone(model.tabular.plr_freq.grad)
        self.assertTrue(torch.isfinite(model.tabular.plr_freq.grad).all())
        self.assertTrue(torch.isfinite(model.tabular.num_missing.grad).all())
        self.assertTrue(torch.isfinite(model.tabular.cat_missing.grad).all())

        # Omitting tabular_cat uses cat_missing cleanly.
        model.eval()
        with torch.no_grad():
            out_no_cat = model(tabular_num=num)
            out_all_missing_cat = model(tabular_num=num, tabular_cat=torch.full_like(cat, -1))
        self.assertTrue(torch.allclose(out_no_cat.classification_logits, out_all_missing_cat.classification_logits, atol=1e-6))

        # Validation rejects inf and out-of-range categories.
        bad_inf = num.clone()
        bad_inf[0, 0] = float("inf")
        with self.assertRaises(ValueError):
            model(tabular_num=bad_inf)
        with self.assertRaises(ValueError):
            model(tabular_num=torch.zeros_like(num), tabular_cat=torch.full_like(cat, cfg.tabular_cat_cardinality))
        with self.assertRaises(ValueError):
            model(tabular_num=torch.zeros_like(num), tabular_cat=torch.full_like(cat, -2))

    def test_seven_modalities_all_127_combinations_and_cache(self):
        torch.manual_seed(7)
        cfg = _seven_modality_config(
            differential_attention=True, cross_differential_attention=True,
            kv_lora_rank=16, attention_sinks=2, ssm_state_dim=8,
            residual_streams=2, latent_reasoning_steps=2, latent_reasoning_adaptive=True,
            memory_slots=4, hierarchical_resampler=True, latent_fsq_levels=(5, 5),
        )
        model = Blue(cfg)
        batch = synthetic_batch(cfg, batch_size=3, seed=11)

        # Verify all 127 modality combinations are covered in one cycle.
        seen = set()
        for idx in range(127):
            sub = balanced_modalities(batch, idx, seed=5, prediction_steps=cfg.prediction_steps)
            key = tuple(k for k in ("input_ids", "images", "series", "audio", "graph_nodes", "points", "tabular_num") if k in sub)
            seen.add(key)
        self.assertEqual(len(seen), 127)

        # Forward + backward across all 7 modalities + 21 hyperbolic contrastive pairs.
        out = model(**batch)
        self.assertIn("alignment", out.losses)
        self.assertIn("ponder_kl", out.losses)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()

        # encode_modalities returns all 7 modalities.
        model.eval()
        with torch.no_grad():
            enc = model.encode_modalities(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                images=batch["images"], series=batch["series"], series_mask=batch["series_mask"],
                audio=batch["audio"], audio_mask=batch["audio_mask"],
                graph_nodes=batch["graph_nodes"], graph_edges=batch["graph_edges"], graph_mask=batch["graph_mask"],
                points=batch["points"], point_features=batch["point_features"], point_mask=batch["point_mask"],
                tabular_num=batch["tabular_num"], tabular_cat=batch["tabular_cat"],
            )
        self.assertEqual(set(enc.keys()), {"text", "image", "series", "audio", "graph", "pointcloud", "tabular"})
        for v in enc.values():
            self.assertTrue(torch.isfinite(v).all())

        # Exact cache equivalence with tabular context.
        with torch.no_grad():
            full = model(input_ids=batch["input_ids"], tabular_num=batch["tabular_num"], tabular_cat=batch["tabular_cat"])
            part1 = model(input_ids=batch["input_ids"][:, :-2], tabular_num=batch["tabular_num"],
                          tabular_cat=batch["tabular_cat"], use_cache=True)
            part2 = model(input_ids=batch["input_ids"][:, -2:], cache=part1.cache, use_cache=True)
        self.assertTrue(torch.allclose(full.logits[:, -2:], part2.logits, atol=1e-4))

        # BF16 autocast + INT8 quantization compatibility.
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out_bf16 = model(**{k: v for k, v in batch.items() if k not in ("labels", "class_labels", "forecast_targets")})
        self.assertTrue(torch.isfinite(out_bf16.logits).all())

        stats = quantize_model_int8(model, min_numel=256)
        self.assertGreater(stats["quantized_modules"], 0)
        with torch.no_grad():
            out_q = model(tabular_num=batch["tabular_num"], tabular_cat=batch["tabular_cat"])
        self.assertTrue(torch.isfinite(out_q.classification_logits).all())

    def test_dataset_sampler_and_profiler(self):
        cfg = _seven_modality_config()
        records = [
            {"tabular_num": [1.0, None, -0.5, 2.0, 0.0, 0.3], "tabular_cat": [1, None, 4], "class_label": 0},
            {"tabular_num": [0.2, 0.5, 1.0, -1.0, 0.1, None], "tabular_cat": [2, 3, 0], "class_label": 1},
            {"text": "Hello", "tabular_num": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "tabular_cat": [0, 1, 2], "class_label": 2},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            ds = JointDataset(path, cfg, allow_mixed=True)
            self.assertEqual(list(ds.signatures), [64, 64, 65])
            sampler = ModalityBatchSampler(ds, batch_size=2, shuffle=False)
            batches = [collate_records([ds[i] for i in idxs]) for idxs in sampler]
            self.assertEqual(len(batches), 2)
            model = Blue(cfg)
            for b in batches:
                out = model(**b)
                self.assertTrue(torch.isfinite(out.loss))

        report = profile_architecture(cfg, seq_len=32, batch_size=1)
        self.assertEqual(len(report["modalities"]), 7)
        self.assertEqual(report["modality_combinations"], 127)
        self.assertEqual(report["contrastive_pairs"], 21)


if __name__ == "__main__":
    unittest.main()
