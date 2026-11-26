import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HotpathBenchmarkTests(unittest.TestCase):
    def test_tiny_benchmark_reports_every_measure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            subprocess.run([sys.executable, str(ROOT / "tools" / "benchmark_hotpaths.py"), "--preset", "tiny",
                            "--threads", "1", "--batch-size", "2", "--text-length", "12", "--new-tokens", "4",
                            "--repeats", "1", "--output", str(output)], check=True, capture_output=True)
            report = json.loads(output.read_text())
        entry = report["devices"]["cpu"]
        self.assertEqual([model["variant"] for model in entry["models"]], ["tiny", "tiny+recurrent", "tiny+temporal"])
        for model in entry["models"]:
            self.assertGreater(model["train_step_ms"], 0)
            self.assertGreater(model["decode_ms_per_token"], 0)
        self.assertEqual(set(entry["branches"]), {"gated_deltanet_forward_backward_ms", "ttt_linear_forward_backward_ms",
                                                  "episodic_memory_forward_backward_ms"})


if __name__ == "__main__":
    unittest.main()
