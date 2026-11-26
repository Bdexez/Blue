"""Exécution minimale des outils sans test dédié, pour qu'ils ne se dégradent pas en silence."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def tool(name, *arguments):
    return subprocess.run([sys.executable, str(ROOT / "tools" / f"{name}.py"), *arguments],
                          check=True, capture_output=True, text=True, cwd=ROOT)


class ToolSmokeTests(unittest.TestCase):
    def test_make_dataset_writes_train_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "data"
            tool("make_dataset", "--preset", "tiny", "--output", str(output),
                 "--train-examples", "5", "--validation-examples", "2")
            self.assertEqual(len((output / "train.jsonl").read_text().splitlines()), 5)
            self.assertEqual(len((output / "validation.jsonl").read_text().splitlines()), 2)

    def test_profile_runtime_cache_footprint_matches_the_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile.json"
            tool("profile_runtime", "--cache", "--threads", "1", "--output", str(output))
            cache = json.loads(output.read_text())["cache"]
        # Valeurs publiées dans IMPROVEMENTS.md (fenêtres locales bornées, contexte projeté une fois).
        self.assertEqual(cache["self_attention_kv_bytes"], 56832)
        self.assertEqual(cache["cached_tokens_per_layer"], [31, 160, 31])
        self.assertEqual(cache["context_projections_per_layer"], [1, 1, 1])

    def test_comparison_tools_complete_with_a_minimal_budget(self):
        for name in ("compare_modality_training", "compare_top1_routing"):
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                tool(name, "--seeds", "1", "--steps", "2", "--batch-size", "2", "--test-batches", "1",
                     "--threads", "1", "--output-dir", str(Path(directory) / "run"))
                self.assertTrue((Path(directory) / "run" / "comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
