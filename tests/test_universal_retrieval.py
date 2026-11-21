from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.cli import main
from blue.training import save_checkpoint


class Universal8ModalityRetrievalTests(unittest.TestCase):
    def test_cross_modal_retrieval_across_new_modalities_and_hyperbolic_poincare_geometry(self):
        torch.manual_seed(520)
        cfg = replace(
            BlueConfig.preset("tiny"),
            audio_enabled=True, audio_layers=1, max_audio_len=64,
            graph_enabled=True, graph_layers=1, graph_node_features=4, graph_edge_features=2,
            pointcloud_enabled=True, pointcloud_layers=1, pointcloud_features=0,
            tabular_enabled=True, tabular_num_features=4, tabular_cat_features=2,
            tabular_cat_cardinality=8, tabular_layers=1, tabular_plr_frequencies=4,
            video_enabled=True, video_channels=3, video_size=16, video_patch_size=8,
            video_tubelet_size=2, max_video_frames=4, video_layers=1,
            alignment_geometry="hyperbolic", alignment_curvature=1.2,
        )
        model = Blue(cfg).eval()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "model.pt"
            data_path = root / "data.jsonl"
            emb_path = root / "emb.jsonl"
            rel_path = root / "rel.jsonl"
            search_path = root / "search.jsonl"
            eval_path = root / "eval.json"

            save_checkpoint(ckpt, model)
            rows = []
            for idx in range(4):
                rows.append({
                    "audio": [[0.1 * (idx + 1)] * 32],
                    "tabular_num": [float(idx), 0.5, -0.2, 1.0],
                    "tabular_cat": [idx % 4, (idx + 1) % 4],
                    "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, float(idx + 1)]],
                    "video": torch.full((3, 2, 16, 16), 0.1 * (idx + 1)).tolist(),
                    "class_label": idx % 2,
                })
            data_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            main(["embed", "--checkpoint", str(ckpt), "--data", str(data_path), "--output", str(emb_path)])
            main([
                "prepare-class-relevance",
                "--query-data", str(data_path), "--candidate-data", str(data_path),
                "--queries", str(emb_path), "--candidates", str(emb_path),
                "--query-modality", "tabular", "--candidate-modality", "video",
                "--exclude-self", "--output", str(rel_path),
            ])
            main([
                "search-embeddings",
                "--queries", str(emb_path), "--candidates", str(emb_path),
                "--query-modality", "tabular", "--candidate-modality", "video",
                "--top-k", "2", "--exclude-self",
                "--geometry", "hyperbolic", "--curvature", "1.2",
                "--output", str(search_path),
            ])
            main([
                "evaluate-retrieval",
                "--queries", str(emb_path), "--candidates", str(emb_path),
                "--relevance", str(rel_path),
                "--query-modality", "tabular", "--candidate-modality", "video",
                "--top-k", "2", "--exclude-self",
                "--geometry", "hyperbolic", "--curvature", "1.2",
                "--output", str(eval_path),
            ])
            report = json.loads(eval_path.read_text(encoding="utf-8"))
            self.assertEqual(report["query_modality"], "tabular")
            self.assertEqual(report["candidate_modality"], "video")
            self.assertEqual(report["judged_queries"], 4)
            self.assertIn("ndcg", report["metrics"])


if __name__ == "__main__":
    unittest.main()
