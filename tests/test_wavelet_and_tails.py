from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from blue import Blue, BlueConfig
from blue.data import synthetic_batch
from blue.training import load_checkpoint, save_checkpoint


class WaveletAndTailPowerTests(unittest.TestCase):
    def test_haar_wavelet_perfect_reconstruction_padding_invariance_and_tail_monotonicity(self):
        base = BlueConfig.preset("tiny")
        self.assertFalse(base.temporal_wavelet_mixing)
        self.assertFalse(base.forecast_tail_power)
        for bad in (dict(temporal_wavelet_mixing=1), dict(forecast_tail_power="yes")):
            with self.assertRaises(ValueError):
                replace(base, **bad)

        torch.manual_seed(550)
        cfg = replace(
            base,
            temporal_spectral_mixing=True,
            temporal_wavelet_mixing=True,
            forecast_tail_power=True,
            quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        )
        model = Blue(cfg).eval()
        wav = model.temporal.wavelet

        # 1. Exact DWT -> IDWT reconstruction identity when gains == 1 (for odd and even lengths!).
        for length in (1, 7, 16, 23):
            u = torch.randn(length, cfg.d_model)
            rec = wav._dwt_idwt_1d(u)
            torch.testing.assert_close(rec, u, rtol=1e-5, atol=1e-5)

        # 2. Strict left and right padding invariance on series!
        series = torch.randn(2, 19, cfg.series_features)
        mask = torch.ones(2, 19, dtype=torch.bool)
        padded_series = torch.cat((torch.randn(2, 4, cfg.series_features) * 100.0, series,
                                   torch.randn(2, 5, cfg.series_features) * 100.0), dim=1)
        padded_mask = torch.cat((torch.zeros(2, 4, dtype=torch.bool), mask,
                                 torch.zeros(2, 5, dtype=torch.bool)), dim=1)
        with torch.no_grad():
            o_clean = model(series=series, series_mask=mask)
            o_pad = model(series=padded_series, series_mask=padded_mask)
        torch.testing.assert_close(o_clean.forecast, o_pad.forecast, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(o_clean.classification_logits, o_pad.classification_logits, rtol=1e-5, atol=1e-5)

        # 3. Monotonic non-crossing quantiles guaranteed even under extreme tail_log_scale!
        with torch.no_grad():
            model.forecaster.tail_log_scale.copy_(torch.tensor([[[[-2.0, 2.5, -1.0, 3.0]]] * cfg.series_features]).reshape(1, 1, cfg.series_features, 4))
            fc = model(series=series, series_mask=mask).forecast
        diffs = fc[..., 1:] - fc[..., :-1]
        self.assertTrue((diffs >= 0.0).all())

        # 4. Gradients reach wavelet gains and tail_log_scale under BF16 + checkpoint roundtrip.
        model.train()
        batch = synthetic_batch(cfg, batch_size=2, seed=551)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertGreater(wav.detail_gain.grad.abs().sum().item(), 0.0)
        self.assertGreater(wav.approx_gain.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.forecaster.tail_log_scale.grad.abs().sum().item(), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wavelet.pt"
            save_checkpoint(path, model.eval())
            restored, _ = load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(model(series=series).forecast, restored(series=series).forecast, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
