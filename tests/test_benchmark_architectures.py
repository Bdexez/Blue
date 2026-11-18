from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from blue import BlueConfig
from blue.cli import inspect_config
from tools.benchmark_architectures import main


class BenchmarkArchitecturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_benchmark_compares_baseline_hybrid_adamw_and_muon(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            out_path = Path(directory) / "bench.json"
            main(["--steps", "3", "--batch-size", "2", "--seed", "17", "--threads", "2",
                  "--output", str(out_path)])
            report = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(len(report["results"]), 3)
        baseline, hybrid_adamw, hybrid_muon = report["results"]
        self.assertEqual(baseline["variant"], "baseline")
        self.assertEqual(hybrid_adamw["variant"], "hybrid")
        self.assertEqual(hybrid_muon["optimizer"], "muon")
        self.assertGreater(hybrid_adamw["parameters"], baseline["parameters"])
        self.assertIn("flow", hybrid_adamw["final_losses"])
        self.assertLess(hybrid_muon["final_val_loss"], hybrid_muon["initial_val_loss"])

    def test_hyper_and_titan_presets_inspect_without_allocating_weights(self):
        hyper = inspect_config(BlueConfig.preset("hyper"))
        titan = inspect_config(BlueConfig.preset("titan"))
        self.assertGreater(hyper["parameters"], 1_600_000_000)
        self.assertGreater(titan["parameters"], 1_650_000_000_000)
        for report in (hyper, titan):
            features = report["architecture_features"]
            self.assertTrue(features["audio_enabled"])
            self.assertTrue(features["graph_enabled"])
            self.assertGreater(features["flow_matching_dim"], 0)
            self.assertGreater(features["memory_slots"], 0)


if __name__ == "__main__":
    unittest.main()
