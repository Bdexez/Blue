from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.hyperbolic import poincare_exp0, poincare_pairwise_distance
from blue.training import load_checkpoint, save_checkpoint


class HyperbolicAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        base = BlueConfig.preset("tiny")
        for bad in (
            dict(alignment_geometry="spherical"),
            dict(alignment_curvature=0.0),
            dict(alignment_curvature=-1.0),
            dict(alignment_curvature=float("nan")),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(base, **bad)

    def test_poincare_metric_axioms_boundary_safety_and_gradients(self):
        torch.manual_seed(19001)
        for c in (0.5, 1.0, 2.5):
            with self.subTest(curvature=c):
                v = torch.randn(5, 16, requires_grad=True)
                pts = poincare_exp0(v, c)
                # Strictly inside the Poincare ball of radius 1 / sqrt(c).
                max_norm = pts.norm(dim=-1).max().item()
                self.assertLess(max_norm, 1.0 / math.sqrt(c))
                # Extreme tangent vectors stay safely inside the ball.
                extreme = poincare_exp0(torch.randn(4, 16) * 1e4, c)
                self.assertTrue(torch.isfinite(extreme).all())
                self.assertLess(extreme.norm(dim=-1).max().item(), 1.0 / math.sqrt(c))

                dist = poincare_pairwise_distance(pts, pts, c)
                # 1. Zero self-distance.
                torch.testing.assert_close(dist.diagonal(), torch.zeros(5), rtol=1e-4, atol=1e-4)
                # 2. Symmetry.
                torch.testing.assert_close(dist, dist.T, rtol=1e-5, atol=1e-5)
                # 3. Triangle inequality d(i, k) <= d(i, j) + d(j, k).
                for i in range(5):
                    for j in range(5):
                        for k in range(5):
                            self.assertLessEqual(dist[i, k].item(), dist[i, j].item() + dist[j, k].item() + 1e-4)
                # 4. Gradients remain finite even on the zero-distance diagonal.
                dist.sum().backward()
                self.assertTrue(torch.isfinite(v.grad).all())
                self.assertGreater(v.grad.abs().sum().item(), 0.0)

    def test_hyperbolic_multimodal_alignment_bf16_and_checkpoint(self):
        torch.manual_seed(19002)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         alignment_geometry="hyperbolic", alignment_curvature=1.5,
                         audio_enabled=True, audio_patch_size=8, audio_layers=1,
                         graph_enabled=True, graph_node_features=6, graph_edge_features=3, graph_layers=1,
                         pointcloud_enabled=True, pointcloud_features=2, pointcloud_layers=1)
        info = inspect_config(config)
        self.assertEqual(info["architecture_features"]["alignment_geometry"], "hyperbolic")
        model = Blue(config)
        batch = synthetic_batch(config, batch_size=3, seed=19003)

        with torch.no_grad():
            embeds = model.eval().encode_modalities(
                input_ids=batch["input_ids"], images=batch["images"],
                series=batch["series"], audio=batch["audio"],
                graph_nodes=batch["graph_nodes"], graph_edges=batch["graph_edges"],
                points=batch["points"], point_features=batch["point_features"],
                attention_mask=batch["attention_mask"], series_mask=batch["series_mask"],
                audio_mask=batch["audio_mask"], graph_mask=batch["graph_mask"],
                point_mask=batch["point_mask"],
            )
        for name, vec in embeds.items():
            self.assertLess(vec.norm(dim=-1).max().item(), 1.0 / math.sqrt(1.5), name)

        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertIn("alignment", out.losses)
        self.assertTrue(torch.isfinite(out.losses["alignment"]))
        out.loss.backward()
        for proj in (*model.alignment, model.audio_alignment, model.graph_alignment, model.pointcloud_alignment):
            self.assertGreater(proj.weight.grad.abs().sum().item(), 0.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hyp.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertEqual(restored.config.alignment_geometry, "hyperbolic")
            self.assertEqual(restored.config.alignment_curvature, 1.5)


if __name__ == "__main__":
    unittest.main()
