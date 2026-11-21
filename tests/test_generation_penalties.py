import unittest
import torch

from blue.config import BlueConfig
from blue.model import Blue


class GenerationPenaltiesTests(unittest.TestCase):
    def test_penalties_match_across_standard_speculative_and_jacobi_decoding(self):
        torch.manual_seed(3)
        config = BlueConfig.preset("tiny")
        model = Blue(config).eval()
        prompt = torch.tensor([[1, 66, 108, 117, 101]])

        for rep, pres, freq in ((1.2, 0.0, 0.0), (1.0, 0.5, 0.25), (1.3, 0.4, 0.2)):
            std = model.generate(
                prompt, max_new_tokens=14, temperature=0, eos_id=None,
                repetition_penalty=rep, presence_penalty=pres, frequency_penalty=freq,
            )
            spec = model.generate(
                prompt, max_new_tokens=14, temperature=0, eos_id=None,
                speculative=True,
                repetition_penalty=rep, presence_penalty=pres, frequency_penalty=freq,
            )
            jac = model.generate(
                prompt, max_new_tokens=14, temperature=0, eos_id=None,
                jacobi=True, jacobi_window=3,
                repetition_penalty=rep, presence_penalty=pres, frequency_penalty=freq,
            )
            self.assertTrue(torch.equal(std, spec), f"Speculative mismatch for {(rep, pres, freq)}")
            self.assertTrue(torch.equal(std, jac), f"Jacobi mismatch for {(rep, pres, freq)}")

        # Strong repetition penalty increases unique token count vs no penalty
        no_pen = model.generate(prompt, max_new_tokens=16, temperature=0, eos_id=None)
        hi_pen = model.generate(
            prompt, max_new_tokens=16, temperature=0, eos_id=None,
            repetition_penalty=2.5, presence_penalty=2.0, frequency_penalty=1.0,
        )
        self.assertGreaterEqual(hi_pen.unique().numel(), no_pen.unique().numel())


if __name__ == "__main__":
    unittest.main()
