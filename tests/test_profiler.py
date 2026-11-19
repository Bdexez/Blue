from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from blue import BlueConfig
from blue.cli import main
from blue.profiler import profile_architecture


class ProfilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_profile_titan_without_allocating_weights(self):
        report = profile_architecture(BlueConfig.preset("titan"), seq_len=8192, batch_size=1, measure=False)
        self.assertGreater(report["parameters"], 1_668_000_000_000)
        self.assertEqual(report["effective_decoder_layers"], 160)
        self.assertEqual(len(report["modalities"]), 7)
        self.assertEqual(report["modality_combinations"], 127)
        self.assertEqual(report["contrastive_pairs"], 21)
        self.assertGreater(report["cache_profile"]["kv_compression_ratio"], 3.0)
        self.assertGreater(report["cache_profile"]["ssm_state_mib_fp32"], 0.0)
        self.assertGreater(report["cache_profile"]["episodic_memory_mib_fp32"], 0.0)

    def test_cli_profile_with_runtime_measurement_and_output_file(self):
        with tempfile.TemporaryDirectory() as directory:
            out_path = Path(directory) / "profile.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["--threads", "2", "profile", "--preset", "tiny", "--seq-len", "64",
                      "--batch-size", "2", "--measure", "--output", str(out_path)])
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("runtime", report)
            self.assertGreater(report["runtime"]["forward_ms"], 0.0)
            self.assertGreater(report["runtime"]["backward_ms"], 0.0)
            self.assertTrue(report["runtime"]["loss"] > 0.0)


if __name__ == "__main__":
    unittest.main()
