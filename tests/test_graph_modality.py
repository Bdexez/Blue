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


class GraphModalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_and_input_validation(self):
        base = BlueConfig.preset("tiny")
        with self.assertRaises(ValueError):
            replace(base, graph_enabled=1)
        with self.assertRaises(ValueError):
            replace(base, graph_rwse_steps=0)
        model = Blue(base).eval()
        with self.assertRaises(ValueError):
            model(graph_nodes=torch.randn(1, 4, 8))
        cfg = replace(base, graph_enabled=True, graph_node_features=6, graph_edge_features=3, max_graph_nodes=12)
        model = Blue(cfg).eval()
        with self.assertRaises(ValueError):
            model(graph_nodes=torch.randn(1, 4, 8))
        with self.assertRaises(ValueError):
            model(graph_nodes=torch.randn(1, 13, 6))
        with self.assertRaises(ValueError):
            model(graph_nodes=torch.randn(1, 4, 6), graph_edges=torch.randn(1, 4, 5, 3))
        with self.assertRaises(ValueError):
            model(graph_edges=torch.randn(1, 4, 4, 3))

    def test_exact_permutation_and_padding_invariance(self):
        torch.manual_seed(9901)
        config = replace(BlueConfig.preset("tiny"), graph_enabled=True,
                         graph_node_features=6, graph_edge_features=3,
                         graph_layers=2, graph_rwse_steps=6, stable_rms_norm=True)
        model = Blue(config).eval()
        nodes = torch.randn(2, 7, 6)
        edges = torch.randn(2, 7, 7, 3)
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[1, 5:] = False

        # Permuting valid nodes and their incident edges must produce identical outputs.
        perm = torch.tensor([3, 0, 4, 1, 2, 5, 6])
        p_nodes = nodes[:, perm]
        p_edges = edges[:, perm][:, :, perm]
        p_mask = mask[:, perm]

        # Appending padded nodes must also leave outputs identical.
        pad_nodes = torch.nn.functional.pad(nodes, (0, 0, 0, 4), value=77.0)
        pad_edges = torch.nn.functional.pad(edges, (0, 0, 0, 4, 0, 4), value=99.0)
        pad_mask = torch.nn.functional.pad(mask, (0, 4), value=False)

        with torch.no_grad():
            base_out = model(graph_nodes=nodes, graph_edges=edges, graph_mask=mask)
            perm_out = model(graph_nodes=p_nodes, graph_edges=p_edges, graph_mask=p_mask)
            pad_out = model(graph_nodes=pad_nodes, graph_edges=pad_edges, graph_mask=pad_mask)
            base_emb = model.encode_modalities(graph_nodes=nodes, graph_edges=edges, graph_mask=mask)["graph"]
            perm_emb = model.encode_modalities(graph_nodes=p_nodes, graph_edges=p_edges, graph_mask=p_mask)["graph"]
            pad_emb = model.encode_modalities(graph_nodes=pad_nodes, graph_edges=pad_edges, graph_mask=pad_mask)["graph"]

        torch.testing.assert_close(base_out.classification_logits, perm_out.classification_logits, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(base_out.forecast, perm_out.forecast, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(base_emb, perm_emb, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(base_out.classification_logits, pad_out.classification_logits, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(base_out.forecast, pad_out.forecast, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(base_emb, pad_emb, rtol=1e-4, atol=1e-5)

    def test_all_31_five_modality_combinations_alignment_jsonl_and_gradients(self):
        torch.manual_seed(9902)
        config = replace(BlueConfig.preset("tiny"), max_seq_len=128,
                         audio_enabled=True, audio_patch_size=8, audio_layers=1,
                         graph_enabled=True, graph_node_features=6, graph_edge_features=3,
                         graph_layers=2, graph_rwse_steps=6, gradient_checkpointing=True)
        info = inspect_config(config)
        self.assertIn("graph", info["modules"])
        self.assertTrue(info["architecture_features"]["graph_enabled"])
        model = Blue(config).eval()
        batch = synthetic_batch(config, batch_size=3, seed=9903)
        self.assertIn("graph_nodes", batch)
        self.assertIn("graph_edges", batch)
        self.assertIn("graph_mask", batch)

        modalities = ("input_ids", "images", "series", "audio", "graph_nodes")
        for count in range(1, 6):
            for subset in itertools.combinations(modalities, count):
                kwargs = {name: batch[name] for name in subset}
                if "audio" in subset:
                    kwargs["audio_mask"] = batch["audio_mask"]
                if "graph_nodes" in subset:
                    kwargs["graph_edges"] = batch["graph_edges"]
                    kwargs["graph_mask"] = batch["graph_mask"]
                with self.subTest(subset=subset), torch.no_grad():
                    out = model(**kwargs)
                    self.assertEqual(out.classification_logits.shape, (3, config.num_classes))
                    self.assertTrue(torch.isfinite(out.classification_logits).all())

        # Balanced schedule covers all 31 subsets of 5 modalities per cycle.
        seen = set()
        for step in range(31):
            picked = balanced_modalities(batch, step, seed=21)
            seen.add(tuple(m for m in modalities if m in picked))
        self.assertEqual(len(seen), 31)

        # encode_modalities returns all 5 normalized representations.
        with torch.no_grad():
            embeds = model.encode_modalities(
                input_ids=batch["input_ids"], images=batch["images"],
                series=batch["series"], audio=batch["audio"],
                graph_nodes=batch["graph_nodes"], graph_edges=batch["graph_edges"],
                attention_mask=batch["attention_mask"], series_mask=batch["series_mask"],
                audio_mask=batch["audio_mask"], graph_mask=batch["graph_mask"],
            )
        self.assertEqual(set(embeds), {"text", "image", "series", "audio", "graph"})

        # Gradients reach node, edge, modality, and alignment parameters under bf16.
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for param in (model.graph_modality, model.graph_alignment.weight,
                      model.graph.node_proj.weight, model.graph.edge_proj.weight,
                      model.graph.blocks[0].edge_to_bias.weight, model.graph.blocks[0].attn_to_edge.weight):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(), 0.0)

        # JSONL collation with variable node counts + checkpoint roundtrip.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "graphs.jsonl"
            rows = [
                {"graph_nodes": [[0.1] * 6 for _ in range(3)],
                 "graph_edges": [[[0.5] * 3 for _ in range(3)] for _ in range(3)], "class_label": 0},
                {"graph_nodes": [[0.2] * 6 for _ in range(5)],
                 "graph_edges": [[[0.1] * 3 for _ in range(5)] for _ in range(5)], "class_label": 1},
            ]
            jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            ds = JointDataset(jsonl, config, allow_mixed=True)
            collated = collate_records([ds[0], ds[1]])
            self.assertEqual(collated["graph_nodes"].shape, (2, 5, 6))
            self.assertEqual(collated["graph_edges"].shape, (2, 5, 5, 3))
            self.assertEqual(collated["graph_mask"].sum(dim=-1).tolist(), [3, 5])
            ckpt = root / "graph.pt"
            save_checkpoint(ckpt, model.eval())
            restored, _ = load_checkpoint(ckpt)
            self.assertTrue(restored.config.graph_enabled)
            with torch.no_grad():
                torch.testing.assert_close(
                    model(graph_nodes=batch["graph_nodes"], graph_edges=batch["graph_edges"]).classification_logits,
                    restored(graph_nodes=batch["graph_nodes"], graph_edges=batch["graph_edges"]).classification_logits,
                    rtol=0, atol=0,
                )


if __name__ == "__main__":
    unittest.main()
