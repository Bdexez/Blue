from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import inspect_config
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class HierarchicalResamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            replace(BlueConfig.preset("tiny"), hierarchical_resampler=1)

    def test_two_stage_fusion_preserves_invariances_text_only_and_gradients(self):
        torch.manual_seed(17001)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         hierarchical_resampler=True,
                         audio_enabled=True, audio_patch_size=8, audio_layers=1,
                         graph_enabled=True, graph_node_features=6, graph_edge_features=3, graph_layers=1,
                         pointcloud_enabled=True, pointcloud_features=2, pointcloud_layers=1,
                         stable_rms_norm=True)
        info = inspect_config(config)
        self.assertTrue(info["architecture_features"]["hierarchical_resampler"])
        model = Blue(config).eval()
        self.assertTrue(model.resampler.local_norm.stable)
        batch = synthetic_batch(config, batch_size=2, seed=17002)

        # Text-only works cleanly when Stage 1 receives no non-text contexts.
        with torch.no_grad():
            text_only = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        self.assertTrue(torch.isfinite(text_only.logits).all())

        # Permutation invariance on graphs and SE(3) invariance on point clouds are preserved through Stage 1 + Stage 2.
        perm = torch.tensor([3, 0, 5, 1, 7, 2, 6, 4])
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        with torch.no_grad():
            g_base = model(graph_nodes=batch["graph_nodes"], graph_edges=batch["graph_edges"])
            g_perm = model(graph_nodes=batch["graph_nodes"][:, perm],
                           graph_edges=batch["graph_edges"][:, perm][:, :, perm])
            p_base = model(points=batch["points"], point_features=batch["point_features"])
            p_se3 = model(points=batch["points"] @ q.T + 15.0, point_features=batch["point_features"])
        torch.testing.assert_close(g_base.classification_logits, g_perm.classification_logits, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(p_base.classification_logits, p_se3.classification_logits, rtol=1e-4, atol=1e-5)

        # Full 6-modality backward reaches Stage 1 local_latents, local_cross, and local_block under bf16.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertGreater(model.resampler.local_latents.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.resampler.local_cross.q.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.resampler.local_block.attn.q.weight.grad.abs().sum().item(), 0.0)

        # Checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hier.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            self.assertTrue(restored.config.hierarchical_resampler)
            with torch.no_grad():
                torch.testing.assert_close(model(images=batch["images"]).classification_logits,
                                           restored(images=batch["images"]).classification_logits,
                                           rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
