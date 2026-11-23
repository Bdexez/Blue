import unittest

import torch

from blue.modules import RMSNorm
from blue.ragged import map_valid_positions, map_valid_rows
from blue.spectral import SpectralTemporalMixer
from blue.wavelet import HaarWaveletMixer


class RaggedBatchingTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(4)
        self.values = torch.randn(5, 9, 6, generator=generator)
        self.valid = torch.rand(5, 9, generator=generator) > 0.35
        self.valid[:, 0] = True
        self.valid[2] = True  # Une ligne complète, d'autres de même longueur ou non.
        self.valid[3] = self.valid[4]

    def test_positions_match_a_loop_over_rows(self):
        function = lambda rows: rows.cumsum(1) * torch.arange(1, rows.size(1) + 1)[None, :, None]
        expected = torch.zeros_like(self.values)
        for row in range(5):
            expected[row, self.valid[row]] = function(self.values[row:row + 1, self.valid[row]])[0]
        torch.testing.assert_close(map_valid_positions(self.values, self.valid, function), expected,
                                   rtol=0, atol=0)

    def test_rows_match_a_loop_over_rows(self):
        function = lambda rows: torch.stack((rows.mean(1), rows.amax(1)), dim=1)
        expected = torch.stack([function(self.values[row:row + 1, self.valid[row]])[0] for row in range(5)])
        torch.testing.assert_close(map_valid_rows(self.values, self.valid, function), expected, rtol=0, atol=0)

    def test_gradients_flow_to_valid_positions_only(self):
        values = self.values.clone().requires_grad_(True)
        map_valid_positions(values, self.valid, lambda rows: rows.square()).sum().backward()
        torch.testing.assert_close(values.grad, torch.where(self.valid[..., None], 2 * values, 0).detach())

    def test_mixers_are_invariant_to_padding_position(self):
        for mixer in (SpectralTemporalMixer(6, norm_cls=RMSNorm), HaarWaveletMixer(6, norm_cls=RMSNorm)):
            with torch.no_grad():
                full = mixer(self.values, self.valid)
                for row in range(5):
                    alone = mixer(self.values[row:row + 1, self.valid[row]],
                                  torch.ones(1, int(self.valid[row].sum()), dtype=torch.bool))
                    torch.testing.assert_close(full[row, self.valid[row]], alone[0], rtol=1e-5, atol=1e-6)
                    self.assertTrue((full[row, ~self.valid[row]] == 0).all())


if __name__ == "__main__":
    unittest.main()
