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


def _eight_modality_config(**overrides):
    base = dict(
        d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=128,
        n_experts=4, top_k=2, max_seq_len=64, sliding_window=16, global_every=2,
        image_size=16, patch_size=4, vision_layers=1, temporal_layers=1,
        max_series_len=32, n_latents=8, resampler_layers=1,
        audio_enabled=True, audio_layers=1, max_audio_len=64,
        graph_enabled=True, graph_layers=1, max_graph_nodes=10, graph_rwse_steps=4,
        pointcloud_enabled=True, pointcloud_features=2, pointcloud_layers=1, max_pointcloud_points=12,
        tabular_enabled=True, tabular_num_features=6, tabular_cat_features=3,
        tabular_cat_cardinality=12, tabular_layers=1, tabular_plr_frequencies=8,
        video_enabled=True, video_channels=3, video_size=16, video_patch_size=8,
        video_tubelet_size=2, max_video_frames=8, video_layers=1,
        alignment_geometry="hyperbolic",
    )
    base.update(overrides)
    return BlueConfig(**base)


class VideoModalityTest(unittest.TestCase):
    def test_config_validation_and_backward_compat(self):
        default = BlueConfig.preset("tiny")
        self.assertFalse(default.video_enabled)
        m = Blue(default)
        self.assertIsNone(m.video)
        with self.assertRaises(ValueError):
            m(video=torch.zeros(1, 3, 4, 32, 32))
        for bad in (
            dict(video_enabled=1),
            dict(video_size=14, video_patch_size=4),
            dict(video_tubelet_size=0),
            dict(max_video_frames=0),
            dict(video_layers=0),
        ):
            with self.assertRaises(ValueError):
                BlueConfig(**bad)

    def test_exact_frame_padding_invariance_and_3d_rope_temporal_sensitivity(self):
        torch.manual_seed(42)
        cfg = _eight_modality_config()
        model = Blue(cfg).eval()

        # Non-divisible frame count (3 frames with tubelet=2) vs padded to 6 frames with video_mask!
        vid_short = torch.randn(2, 3, 3, 16, 16)
        mask_short = torch.ones(2, 3, dtype=torch.bool)
        vid_padded = torch.cat((vid_short, torch.randn(2, 3, 3, 16, 16) * 50.0), dim=2)
        mask_padded = torch.cat((mask_short, torch.zeros(2, 3, dtype=torch.bool)), dim=1)

        with torch.no_grad():
            out_short = model(video=vid_short, video_mask=mask_short)
            out_padded = model(video=vid_padded, video_mask=mask_padded)
        torch.testing.assert_close(out_short.classification_logits, out_padded.classification_logits, rtol=1e-5, atol=1e-5)

        # Temporal order matters thanks to 3D Axial RoPE!
        vid_four = torch.randn(1, 3, 4, 16, 16)
        vid_reversed = torch.flip(vid_four, dims=(2,))
        with torch.no_grad():
            o1 = model(video=vid_four).classification_logits
            o2 = model(video=vid_reversed).classification_logits
        self.assertGreater((o1 - o2).abs().max().item(), 1e-4)

    def test_all_8_modalities_255_combinations_quad_paradigm_and_int8(self):
        torch.manual_seed(99)
        cfg = _eight_modality_config(
            differential_attention=True, cross_differential_attention=True,
            kv_lora_rank=16, attention_sinks=2, value_residual=True,
            ssm_state_dim=8, ttt_state_dim=8,
            residual_streams=2, latent_reasoning_steps=2, latent_reasoning_adaptive=True,
            memory_slots=4, hierarchical_resampler=True, latent_fsq_levels=(5, 5),
        )
        model = Blue(cfg)
        batch = synthetic_batch(cfg, batch_size=3, seed=19)

        # All 255 non-empty modality combinations covered in one balanced cycle!
        seen = set()
        keys = ("input_ids", "images", "series", "audio", "graph_nodes", "points", "tabular_num", "video")
        for idx in range(255):
            sub = balanced_modalities(batch, idx, seed=3, prediction_steps=cfg.prediction_steps)
            seen.add(tuple(k for k in keys if k in sub))
        self.assertEqual(len(seen), 255)

        # Full 8-modality + 28 hyperbolic contrastive pairs forward & backward.
        out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        self.assertIn("alignment", out.losses)
        self.assertIn("ponder_kl", out.losses)
        out.loss.backward()
        self.assertIsNotNone(model.video.tubelet.weight.grad)
        self.assertTrue(torch.isfinite(model.video.tubelet.weight.grad).all())

        # encode_modalities returns all 8 keys.
        model.eval()
        with torch.no_grad():
            enc = model.encode_modalities(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                images=batch["images"], series=batch["series"], series_mask=batch["series_mask"],
                audio=batch["audio"], audio_mask=batch["audio_mask"],
                graph_nodes=batch["graph_nodes"], graph_edges=batch["graph_edges"], graph_mask=batch["graph_mask"],
                points=batch["points"], point_features=batch["point_features"], point_mask=batch["point_mask"],
                tabular_num=batch["tabular_num"], tabular_cat=batch["tabular_cat"],
                video=batch["video"], video_mask=batch["video_mask"],
            )
        self.assertEqual(set(enc.keys()), {"text", "image", "series", "audio", "graph", "pointcloud", "tabular", "video"})

        # Exact cache equivalence with video context + quad-paradigm decoder.
        with torch.no_grad():
            full = model(input_ids=batch["input_ids"], video=batch["video"], video_mask=batch["video_mask"])
            p1 = model(input_ids=batch["input_ids"][:, :-2], video=batch["video"], video_mask=batch["video_mask"], use_cache=True)
            p2 = model(input_ids=batch["input_ids"][:, -2:], cache=p1.cache, use_cache=True)
        torch.testing.assert_close(full.logits[:, -2:], p2.logits, rtol=1e-4, atol=1e-4)

        # BF16 autocast + INT8 quantization compatibility.
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out_bf16 = model(**{k: v for k, v in batch.items() if k not in ("labels", "class_labels", "forecast_targets")})
        self.assertTrue(torch.isfinite(out_bf16.logits).all())

        qstats = quantize_model_int8(model, min_numel=256)
        self.assertGreater(qstats["quantized_modules"], 0)
        with torch.no_grad():
            out_q = model(video=batch["video"], video_mask=batch["video_mask"])
        self.assertTrue(torch.isfinite(out_q.classification_logits).all())

    def test_mixed_dataset_signature_255_and_profiler(self):
        cfg = _eight_modality_config()
        v1 = torch.zeros(3, 2, 16, 16).tolist()
        v2 = torch.ones(3, 4, 16, 16).tolist()
        records = [
            {"video": v1, "class_label": 0},
            {"video": v2, "class_label": 1},
            {"text": "Clip", "video": v1, "class_label": 2},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video_mixed.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            ds = JointDataset(path, cfg, allow_mixed=True)
            self.assertEqual(list(ds.signatures), [128, 128, 129])
            sampler = ModalityBatchSampler(ds, batch_size=2, shuffle=False)
            batches = [collate_records([ds[i] for i in idxs]) for idxs in sampler]
            self.assertEqual(len(batches), 2)
            model = Blue(cfg)
            for b in batches:
                out = model(**b)
                self.assertTrue(torch.isfinite(out.loss))

        report = profile_architecture(cfg, seq_len=32, batch_size=1)
        self.assertEqual(len(report["modalities"]), 8)
        self.assertEqual(report["modality_combinations"], 255)
        self.assertEqual(report["contrastive_pairs"], 28)


if __name__ == "__main__":
    unittest.main()
