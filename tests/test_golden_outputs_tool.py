from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from golden_outputs import compare  # noqa: E402


class GoldenOutputsToolTests(unittest.TestCase):
    def test_compare_flags_float_drift_and_integer_changes(self):
        reference = {"a": {"x": torch.tensor([1.0, 2.0]), "tokens": torch.tensor([3, 4])}}
        self.assertEqual(compare(reference, {"a": {"x": torch.tensor([1.0, 2.0]), "tokens": torch.tensor([3, 4])}}, 0.0), 0)
        drifted = {"a": {"x": torch.tensor([1.0, 2.001]), "tokens": torch.tensor([3, 4])}}
        self.assertEqual(compare(reference, drifted, 1e-2), 0)
        self.assertEqual(compare(reference, drifted, 1e-5), 1)
        self.assertEqual(compare(reference, {"a": {"x": torch.tensor([1.0, 2.0]), "tokens": torch.tensor([3, 5])}}, 0.0), 1)
        self.assertEqual(compare(reference, {"a": {"tokens": torch.tensor([3, 4])}}, 0.0), 1)


if __name__ == "__main__":
    unittest.main()
