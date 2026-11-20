from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from blue import BlueConfig
from blue.card import generate_model_card
from blue.cli import main


class ModelCardTests(unittest.TestCase):
    def test_generate_model_card_titan_and_cli(self):
        card = generate_model_card(BlueConfig.preset("titan"), name="titan", seq_len=8192)
        self.assertIn("# BLUE — Fiche d'architecture (titan)", card)
        self.assertIn("```mermaid", card)
        self.assertIn("flowchart TD", card)
        self.assertIn("Differential Cross-Attention", card)
        self.assertIn("PonderNet", card)
        self.assertIn("TTT-Linear", card)
        self.assertIn("ResFormer", card)
        self.assertIn("video", card)
        self.assertIn("255", card)
        self.assertIn("28", card)
        self.assertIn("hyperbolic", card)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "CARD.md"
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["--threads", "2", "card", "--preset", "hyper", "--output", str(out)])
            written = out.read_text(encoding="utf-8")
            self.assertIn("# BLUE — Fiche d'architecture (hyper)", written)
            self.assertIn("`blocks`", written)
            self.assertIn("`tabular`", written)


if __name__ == "__main__":
    unittest.main()
