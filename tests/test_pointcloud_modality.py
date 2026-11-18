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


class PointCloudModalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_and_input_validation(self):
        base = BlueConfig.preset("tiny")
        with self.assertRaises(ValueError):
            replace(base, pointcloud_enabled=1)
        with self.assertRaises(ValueError):
            replace(base, pointcloud_features=-1)
        model = Blue(base).eval()
        with self.assertRaises(ValueError):
            model(points=torch.randn(1, 6, 3))
        cfg = replace(base, pointcloud_enabled=True, pointcloud_features=4, max_pointcloud_points=16)
        model = Blue(cfg).eval()
        with self.assertRaises(ValueError):
            model(points=torch.randn(1, 6, 2))
        with self.assertRaises(ValueError):
            model(points=torch.randn(1, 17, 3))
        with self.assertRaises(ValueError):
            model(points=torch.randn(1, 6, 3), point_features=torch.randn(1, 6, 2))
        with self.assertRaises(ValueError):
            model(point_mask=torch.ones(1, 6, dtype=torch.bool))

    def test_exact_se3_rotation_translation_permutation_and_padding_invariance(self):
        torch.manual_seed(16001)
        config = replace(BlueConfig.preset("tiny"), pointcloud_enabled=True,
                         pointcloud_features=3, pointcloud_layers=2, stable_rms_norm=True)
        model = Blue(config).eval()
        pts = torch.randn(2, 9, 3)
        feats = torch.randn(2, 9, 3)
        mask = torch.ones(2, 9, dtype=torch.bool)
        mask[1, 6:] = False

        # Construct a proper 3D rotation matrix R in SO(3) via QR decomposition + arbitrary translation.
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        translation = torch.tensor([12.5, -34.2, 7.8])
        se3_pts = pts @ q.T + translation

        # Simultaneous permutation of points & features + appended padding points.
        perm = torch.tensor([4, 1, 5, 0, 3, 2, 6, 7, 8])
        perm_pts = se3_pts[:, perm]
        perm_feats = feats[:, perm]
        perm_mask = mask[:, perm]

        pad_pts = torch.nn.functional.pad(perm_pts, (0, 0, 0, 5), value=555.0)
        pad_feats = torch.nn.functional.pad(perm_feats, (0, 0, 0, 5), value=999.0)
        pad_mask = torch.nn.functional.pad(perm_mask, (0, 5), value=False)

        with torch.no_grad():
            base_out = model(points=pts, point_features=feats, point_mask=mask)
            se3_out = model(points=se3_pts, point_features=feats, point_mask=mask)
            full_inv_out = model(points=pad_pts, point_features=pad_feats, point_mask=pad_mask)
            base_emb = model.encode_modalities(points=pts, point_features=feats, point_mask=mask)["pointcloud"]
            full_inv_emb = model.encode_modalities(points=pad_pts, point_features=pad_feats,
                                                   point_mask=pad_mask)["pointcloud"]

        torch.testing.assert_close(base_out.classification_logits, se3_out.classification_logits, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(base_out.forecast, se3_out.forecast, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(base_out.classification_logits, full_inv_out.classification_logits, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(base_out.forecast, full_inv_out.forecast, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(base_emb, full_inv_emb, rtol=1e-4, atol=1e-5)

    def test_all_63_six_modality_combinations_alignment_jsonl_and_gradients(self):
        torch.manual_seed(16002)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         audio_enabled=True, audio_patch_size=8, audio_layers=1,
                         graph_enabled=True, graph_node_features=6, graph_edge_features=3, graph_layers=1,
                         pointcloud_enabled=True, pointcloud_features=2, pointcloud_layers=1,
                         gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertIn("pointcloud", info["modules"])
        self.assertTrue(info["architecture_features"]["pointcloud_enabled"])
        model = Blue(config).eval()
        batch = synthetic_batch(config, batch_size=2, seed=16003)
        self.assertIn("points", batch)
        self.assertIn("point_features", batch)
        self.assertIn("point_mask", batch)

        modalities = ("input_ids", "images", "series", "audio", "graph_nodes", "points")
        # Balanced modality schedule covers all 63 non-empty subsets of 6 modalities per cycle.
        seen = set()
        for step in range(63):
            picked = balanced_modalities(batch, step, seed=31)
            seen.add(tuple(m for m in modalities if m in picked))
        self.assertEqual(len(seen), 63)

        # Spot-check all 6 singletons, all 15 pairs, and the full 6-modality forward.
        subsets = (
            list(itertools.combinations(modalities, 1))
            + list(itertools.combinations(modalities, 2))
            + [modalities]
        )
        for subset in subsets:
            kwargs = {name: batch[name] for name in subset}
            if "audio" in subset:
                kwargs["audio_mask"] = batch["audio_mask"]
            if "graph_nodes" in subset:
                kwargs["graph_edges"] = batch["graph_edges"]
                kwargs["graph_mask"] = batch["graph_mask"]
            if "points" in subset:
                kwargs["point_features"] = batch["point_features"]
                kwargs["point_mask"] = batch["point_mask"]
            with self.subTest(subset=subset), torch.no_grad():
                out = model(**kwargs)
                self.assertEqual(out.classification_logits.shape, (2, config.num_classes))
                self.assertTrue(torch.isfinite(out.classification_logits).all())

        # encode_modalities returns all 6 normalized representations.
        with torch.no_grad():
            embeds = model.encode_modalities(
                input_ids=batch["input_ids"], images=batch["images"],
                series=batch["series"], audio=batch["audio"],
                graph_nodes=batch["graph_nodes"], graph_edges=batch["graph_edges"],
                points=batch["points"], point_features=batch["point_features"],
                attention_mask=batch["attention_mask"], series_mask=batch["series_mask"],
                audio_mask=batch["audio_mask"], graph_mask=batch["graph_mask"],
                point_mask=batch["point_mask"],
            )
        self.assertEqual(set(embeds), {"text", "image", "series", "audio", "graph", "pointcloud"})

        # Gradients reach pointcloud encoder, modality token, and alignment projection under bf16.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for param in (model.pointcloud_modality, model.pointcloud_alignment.weight,
                      model.pointcloud.node_proj.weight, model.pointcloud.pair_proj.weight):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(), 0.0)

        # JSONL collation with variable point counts + checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "pc.jsonl"
            rows = [
                {"points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                 "point_features": [[0.1, 0.2]] * 3, "class_label": 0},
                {"points": [[0.0, 0.0, 1.0]] * 5, "point_features": [[0.3, 0.4]] * 5, "class_label": 1},
            ]
            jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            ds = JointDataset(jsonl, config, allow_mixed=True)
            collated = collate_records([ds[0], ds[1]])
            self.assertEqual(collated["points"].shape, (2, 5, 3))
            self.assertEqual(collated["point_features"].shape, (2, 5, 2))
            self.assertEqual(collated["point_mask"].sum(dim=-1).tolist(), [3, 5])
            ckpt = root / "pc.pt"
            save_checkpoint(ckpt, model.eval())
            restored, _ = load_checkpoint(ckpt)
            self.assertTrue(restored.config.pointcloud_enabled)
            with torch.no_grad():
                torch.testing.assert_close(
                    model(points=batch["points"], point_features=batch["point_features"]).classification_logits,
                    restored(points=batch["points"], point_features=batch["point_features"]).classification_logits,
                    rtol=0, atol=0,
                )


if __name__ == "__main__":
    unittest.main()
